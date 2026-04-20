"""End-to-end evaluation: SSM-3D vs. original Depth-Anything-3 on ETH3D.

Pipeline (see plan at ~/.claude/plans/evalutate-this-algirhtym-...md §§1-6):
  1. Download ETH3D `terrains` RGB + GT depth (~0.5 GB).
  2. Build SSM-3D (ViT-Small, Mamba3 attn), load DINOv2 weights (no warm-start).
  3. Load DA3-SMALL (`depth-anything/DA3-SMALL`).
  4. Per image: run DA3 inference → depth + features; run SSM-3D → features only
     (no trained depth head; its SimpleDepthHead output is just labeled and
     shown as "head activation" in the existing run_demo).
  5. Metrics
       Depth (DA3 vs GT, median-aligned): abs_rel, δ<1.25, δ<1.25^2, rmse, log10.
       Repr (head-to-head):              feat_cos_mean, effective_rank,
                                          cross_view_nn_agreement (GT-warped).
  6. Visualizations:
       depth_grid_{i}.png, error_{i}.png, features_{i}.png,
       metric_bars_{depth,repr}.png, summary.md.

Depth is NOT compared side-by-side because DA3-SMALL uses `cat_token=True`
(768-dim features) and different `out_layers=[5,7,9,11]`, so DA3's DualDPT
cannot run on SSM-3D's 384-dim features without retraining.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_dinov2 import ensure_dinov2_vits14  # noqa: E402

from ssm3d.data.eth3d import download_eth3d_terrains, load_eth3d_scene
from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, da3_depth, da3_features, load_da3
from ssm3d.eval.eth3d_gt import load_eth3d_cams
from ssm3d.eval.metrics import (
    cross_view_nn_agreement,
    depth_metrics,
    effective_rank,
    feat_cos_mean,
)
from ssm3d.eval.visualize import (
    save_depth_grid,
    save_depth_metric_bars,
    save_error_heatmap,
    save_feature_comparison,
    save_repr_metric_bars,
    write_summary_md,
)
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_dinov2_backbone


def _build_ssm3d(img_size: int, patch_size: int, depth: int) -> SSM3DNet:
    net = SSM3DNet(size="small", img_size=img_size, patch_size=patch_size, depth=depth)
    load_dinov2_backbone(net.backbone.vit, ensure_dinov2_vits14())
    net.eval()
    return net


@torch.inference_mode()
def _ssm3d_features(net: SSM3DNet, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """(N, 3, H, W) -> (N, T, C) patch features + grid."""
    out = net.backbone(images.unsqueeze(0))
    feats = out.features[0]  # (N, T, C)
    return feats, out.grid_hw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out-root", type=Path, default=Path("outputs/eval"))
    ap.add_argument("--scene", type=str, default="terrains")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--max-images", type=int, default=12,
                    help="cap for CPU runtime; terrains has ~42 total")
    ap.add_argument("--da3-model", type=str, default=DEFAULT_HF_MODEL)
    ap.add_argument("--da3-process-res", type=int, default=504)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--skip-da3", action="store_true",
                    help="skip DA3 entirely (e.g., for plumbing smoke tests)")
    args = ap.parse_args()

    torch.manual_seed(0)
    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] downloading ETH3D `{args.scene}` (RGB + GT depth) ...")
    scene_dir = download_eth3d_terrains(args.data_root, scene=args.scene, download_depth=True)
    sample = load_eth3d_scene(
        scene_dir, max_images=args.max_images, image_size=args.img_size, load_gt_depth=True
    )
    N = sample.images.shape[0]
    print(f"  -> {N} images, GT depth present: {sample.gt_depth is not None}")
    if sample.gt_depth is None:
        raise RuntimeError("GT depth missing; evaluation cannot proceed.")

    image_names = [p.name for p in sample.image_paths]
    cams = load_eth3d_cams(scene_dir, image_size=args.img_size, image_names=image_names)

    print("[2/5] building SSM-3D (DINOv2 loaded, no warm-start) ...")
    ssm = _build_ssm3d(args.img_size, args.patch_size, args.depth).to(args.device)
    ssm_feats_all, ssm_grid = _ssm3d_features(ssm, sample.images.to(args.device))

    da3 = None
    da3_depth_preds: list[torch.Tensor] = []
    da3_feats_all: list[torch.Tensor] = []
    da3_grid: tuple[int, int] = (0, 0)
    if not args.skip_da3:
        print(f"[3/5] loading DA3 `{args.da3_model}` ...")
        try:
            da3 = load_da3(args.da3_model, device=args.device)
        except Exception:
            print("  !! DA3 load failed:")
            traceback.print_exc()
            print("  -> continuing with --skip-da3 semantics")
            args.skip_da3 = True

    per_image_depth: list[dict[str, float]] = []
    per_image_repr_da3: list[dict[str, float]] = []
    per_image_repr_ssm: list[dict[str, float]] = []

    print(f"[4/5] per-image eval loop ({N} images) ...")
    for i in range(N):
        rgb = sample.images[i]
        gt = sample.gt_depth[i]
        valid = sample.valid_mask[i]

        # --- DA3 depth + features ---
        if da3 is not None:
            try:
                dep = da3_depth(da3, rgb.unsqueeze(0), process_res=args.da3_process_res)[0]
            except Exception:
                print(f"  !! DA3 depth failed for image {i}:")
                traceback.print_exc()
                dep = torch.full_like(gt, float("nan"))
            try:
                feats_da3, da3_grid_i = da3_features(
                    da3, rgb.unsqueeze(0), process_res=args.da3_process_res
                )
                feats_da3 = feats_da3[0]  # (T, C)
                da3_grid = da3_grid_i
            except Exception:
                print(f"  !! DA3 features failed for image {i}:")
                traceback.print_exc()
                feats_da3 = None
            da3_depth_preds.append(dep)
            if feats_da3 is not None:
                da3_feats_all.append(feats_da3)

            dm = depth_metrics(dep, gt, valid, align=True)
            per_image_depth.append(dm.as_dict())
        else:
            da3_depth_preds.append(torch.full_like(gt, float("nan")))
            per_image_depth.append({})

        # --- Representation metrics (per image; cross-view is added after loop) ---
        ssm_feats_i = ssm_feats_all[i]
        ssm_metrics = {
            "feat_cos_mean": feat_cos_mean(ssm_feats_i),
            "effective_rank": effective_rank(ssm_feats_i),
        }
        per_image_repr_ssm.append(ssm_metrics)

        da3_metrics = {}
        if len(da3_feats_all) > i:
            da3_metrics = {
                "feat_cos_mean": feat_cos_mean(da3_feats_all[i]),
                "effective_rank": effective_rank(da3_feats_all[i]),
            }
        per_image_repr_da3.append(da3_metrics)

    # Cross-view agreement: pair each image i with image (i+1) % N.
    print("  computing cross-view NN agreement (GT-warped) ...")
    for i in range(N):
        j = (i + 1) % N
        name_i = sample.image_paths[i].name
        name_j = sample.image_paths[j].name
        if name_i not in cams.intrinsics or name_j not in cams.intrinsics:
            continue
        K_i = torch.from_numpy(cams.intrinsics[name_i])
        K_j = torch.from_numpy(cams.intrinsics[name_j])
        E_i = torch.from_numpy(cams.extrinsics[name_i])
        E_j = torch.from_numpy(cams.extrinsics[name_j])

        agree_ssm = cross_view_nn_agreement(
            ssm_feats_all[i], ssm_feats_all[j],
            grid_hw=ssm_grid,
            depth_a=sample.gt_depth[i],
            intrinsic_a=K_i, intrinsic_b=K_j,
            extrinsic_a_w2c=E_i, extrinsic_b_w2c=E_j,
            image_hw_b=(args.img_size, args.img_size),
        )
        per_image_repr_ssm[i]["cross_view_nn_agreement"] = agree_ssm

        if len(da3_feats_all) > max(i, j):
            agree_da3 = cross_view_nn_agreement(
                da3_feats_all[i], da3_feats_all[j],
                grid_hw=da3_grid,
                depth_a=sample.gt_depth[i],
                intrinsic_a=K_i, intrinsic_b=K_j,
                extrinsic_a_w2c=E_i, extrinsic_b_w2c=E_j,
                image_hw_b=(args.img_size, args.img_size),
            )
            per_image_repr_da3[i]["cross_view_nn_agreement"] = agree_da3

    print("[5/5] writing visualizations ...")
    for i in range(N):
        rgb = sample.images[i]
        gt = sample.gt_depth[i]
        valid = sample.valid_mask[i]
        if da3 is not None and torch.isfinite(da3_depth_preds[i]).any():
            save_depth_grid(
                rgb, gt, da3_depth_preds[i], valid,
                args.out_root / f"depth_grid_{i:02d}.png",
            )
            save_error_heatmap(
                da3_depth_preds[i], gt, valid,
                args.out_root / f"error_{i:02d}.png",
            )
        if len(da3_feats_all) > i:
            save_feature_comparison(
                rgb, da3_feats_all[i], da3_grid,
                ssm_feats_all[i], ssm_grid,
                args.out_root / f"features_{i:02d}.png",
            )

    if da3 is not None:
        save_depth_metric_bars(per_image_depth, args.out_root / "metric_bars_depth.png")
    save_repr_metric_bars(
        per_image_repr_da3, per_image_repr_ssm, args.out_root / "metric_bars_repr.png"
    )
    note = (
        "DA3-SMALL features are 768-dim (`cat_token=True`); SSM-3D features are "
        "384-dim. Representation metrics are dim-invariant and compared as scores. "
        "Depth is DA3 vs GT only — SSM-3D has no trained depth head.\n\n"
        "Interpretation: **lower** `feat_cos_mean` is better (less token collapse); "
        "**higher** `effective_rank` and `cross_view_nn_agreement` are better. "
        "DA3's high `feat_cos_mean` is a known property of `cat_token=True`: each "
        "patch token is concatenated with a global pooled token, which pushes "
        "pairwise cosines up without hurting downstream depth accuracy. "
        "`cross_view_nn_agreement` (GT-warped across image i <-> (i+1) mod N) is "
        "the cleanest head-to-head feature-quality signal here: the attention swap "
        "in SSM-3D was never trained, so it loses DA3's 3D-consistent matching."
    )
    if args.skip_da3:
        note = "Ran with --skip-da3: DA3 rows are empty. " + note
    write_summary_md(
        args.out_root / "summary.md",
        da3_depth=per_image_depth,
        da3_repr=per_image_repr_da3,
        ssm_repr=per_image_repr_ssm,
        note=note,
    )

    print("\ndone. Outputs:")
    for p in sorted(args.out_root.glob("*.png"))[:12]:
        print(f"  {p}")
    print(f"  {args.out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
