"""End-to-end evaluation: SSM-3D vs. original Depth-Anything-3 on ETH3D.

Pipeline (see plan at ~/.claude/plans/evalutate-this-algirhtym-...md §§1-6, §8):
  1. Download ETH3D `terrains` RGB + GT depth (~0.5 GB).
  2. Build SSM-3D (ViT-Small, Mamba3 attn), load DINOv2 weights (no warm-start).
  3. Load DA3-SMALL (`depth-anything/DA3-SMALL`).
  4. Per image:
       - DA3: inference → depth + features.
       - SSM-3D: backbone features (final + layers [5,7,9,11]). For the depth
         panel, feed the 4 intermediate layers through DA3's own DualDPT
         (shared-DPT smoke test; 384-d features duplicated → 768-d to match
         DA3's cat_token=True).
  5. Metrics
       Depth (DA3 vs GT, median-aligned): abs_rel, δ<1.25, δ<1.25^2, rmse, log10.
       Repr (head-to-head):              feat_cos_mean, effective_rank,
                                          cross_view_nn_agreement (GT-warped).
  6. Memory: param counts + peak RSS delta around one warm forward pass each.
  7. Visualizations:
       depth_grid_{i}.png   (2x2: Input | GT / DA3 | SSM-3D),
       error_{i}.png        (1x2: |DA3-GT| | |SSM-3D-GT|, shared scale),
       features_{i}.png,
       arch_da3/ssm3d/diff.png (emitted once at end),
       metric_bars_{depth,repr,memory}.png, summary.md.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_dinov2 import ensure_dinov2_vits14  # noqa: E402

from render_architecture_diagrams import render_all as render_arch_diagrams  # noqa: E402

from ssm3d.data.eth3d import download_eth3d_terrains, load_eth3d_scene
from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, da3_depth, da3_features, load_da3
from ssm3d.eval.dpt_adapter import SHARED_DPT_LAYERS, get_dualdpt, shared_dpt_depth
from ssm3d.eval.eth3d_gt import load_eth3d_cams
from ssm3d.eval.memory import (
    MemoryReport,
    RSSPoller,
    param_count,
    reset_cuda_peak,
    snapshot_cuda_peak,
)
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
    save_memory_bars,
    save_repr_metric_bars,
    write_summary_md,
)
from ssm3d.bridge import DimBridgeStack
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_dinov2_backbone


_AMP_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _maybe_resize_pos_embed(state: dict, model: torch.nn.Module) -> None:
    """Bicubic-resize `backbone.vit.pos_embed` in `state` to match `model`.

    Lets us evaluate a 224-trained checkpoint at img_size=504 (CM2). CLS token
    passes through; patch tokens are resampled on the 2-D patch grid.
    """
    import math
    key = "backbone.vit.pos_embed"
    if key not in state:
        return
    src = state[key]
    dst = model.state_dict().get(key)
    if dst is None or src.shape == dst.shape:
        return
    cls_src, patch_src = src[:, :1], src[:, 1:]
    n_src = patch_src.shape[1]
    n_dst = dst.shape[1] - 1
    g_src = int(math.sqrt(n_src))
    g_dst = int(math.sqrt(n_dst))
    assert g_src * g_src == n_src and g_dst * g_dst == n_dst, "non-square grid"
    dim = patch_src.shape[-1]
    grid = patch_src.reshape(1, g_src, g_src, dim).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid.float(), size=(g_dst, g_dst), mode="bicubic", align_corners=False
    ).to(patch_src.dtype)
    patch_dst = grid.permute(0, 2, 3, 1).reshape(1, g_dst * g_dst, dim)
    state[key] = torch.cat([cls_src, patch_dst], dim=1)
    print(f"  pos_embed resized: {src.shape} → {state[key].shape}")


def _build_ssm3d(
    img_size: int,
    patch_size: int,
    depth: int,
    chunk_size: int | None = None,
    state_dim: int = 64,
) -> SSM3DNet:
    net = SSM3DNet(
        size="small", img_size=img_size, patch_size=patch_size,
        depth=depth, chunk_size=chunk_size,
        mamba_state_dim=state_dim,
    )
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
    ap.add_argument("--dtype", choices=list(_AMP_DTYPES.keys()), default="fp32",
                    help="autocast dtype for SSM-3D forward (bf16/fp16 on CUDA)")
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="chunked SSD query-axis chunk size (None = full T×T mask)")
    ap.add_argument("--state-dim", type=int, default=64,
                    help="CM11: Mamba-3 SSD recurrent state dim (default 64; "
                         "CM11 uses 32). Must match the ckpt being loaded.")
    ap.add_argument("--student-ckpt", type=Path, default=None,
                    help="Phase-C checkpoint; loads student + bridge state")
    ap.add_argument("--head", choices=["shared_dpt", "simple"], default="shared_dpt",
                    help="shared_dpt = DA3 DualDPT via DimBridge; simple = SSM-3D SimpleDepthHead")
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
    ssm = _build_ssm3d(
        args.img_size, args.patch_size, args.depth,
        chunk_size=args.chunk_size, state_dim=args.state_dim,
    ).to(args.device)
    bridge: DimBridgeStack | None = None
    tuned_dpt_state: dict | None = None
    if args.student_ckpt is not None:
        print(f"  loading student ckpt from {args.student_ckpt}")
        state = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
        _maybe_resize_pos_embed(state["student"], ssm)
        ssm.load_state_dict(state["student"])
        if state.get("bridge") is not None:
            bridge = DimBridgeStack(num_layers=len(SHARED_DPT_LAYERS), in_dim=384)
            bridge.load_state_dict(state["bridge"])
            bridge.to(args.device).eval()
        tuned_dpt_state = state.get("dualdpt")
    ssm_feats_all, ssm_grid = _ssm3d_features(ssm, sample.images.to(args.device))

    da3 = None
    da3_depth_preds: list[torch.Tensor] = []
    ssm_depth_preds: list[torch.Tensor | None] = []
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

    shared_dpt = None
    if da3 is not None:
        try:
            shared_dpt = get_dualdpt(da3)
            if tuned_dpt_state is not None:
                import copy
                shared_dpt = copy.deepcopy(shared_dpt)
                shared_dpt.load_state_dict(tuned_dpt_state)
                shared_dpt.to(args.device).eval()
                print(f"  shared-DPT: loaded CM21-tuned DualDPT from ckpt "
                      f"(DA3 baseline still uses its original DualDPT)")
            else:
                print(f"  shared-DPT: using DA3's DualDPT on SSM-3D layers {SHARED_DPT_LAYERS} "
                      "(384→768 via channel duplication; smoke test)")
        except Exception:
            print("  !! could not bind shared DualDPT:")
            traceback.print_exc()

    per_image_depth: list[dict[str, float]] = []
    per_image_depth_ssm: list[dict[str, float]] = []
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

        if args.head == "simple":
            try:
                ssm_dep = ssm(rgb.unsqueeze(0).unsqueeze(0).to(args.device))["depth"][0, 0, 0]
            except Exception:
                print(f"  !! simple-head failed for image {i}:")
                traceback.print_exc()
                ssm_dep = None
            ssm_depth_preds.append(ssm_dep)
        elif shared_dpt is not None:
            try:
                ssm_dep = shared_dpt_depth(
                    ssm, shared_dpt, rgb.unsqueeze(0).to(args.device),
                    bridge=bridge,
                )[0]
            except Exception:
                print(f"  !! shared-DPT failed for image {i}:")
                traceback.print_exc()
                ssm_dep = None
            ssm_depth_preds.append(ssm_dep)
        else:
            ssm_depth_preds.append(None)

        ssm_dep_for_metric = ssm_depth_preds[-1]
        if ssm_dep_for_metric is not None and torch.isfinite(ssm_dep_for_metric).any():
            dm_ssm = depth_metrics(
                ssm_dep_for_metric.detach().cpu(), gt, valid, align=True
            )
            per_image_depth_ssm.append(dm_ssm.as_dict())
        else:
            per_image_depth_ssm.append({})

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

    mem_da3 = MemoryReport(param_count=0, peak_rss_delta_mb=0.0)
    mem_ssm = MemoryReport(param_count=0, peak_rss_delta_mb=0.0)
    if da3 is not None:
        print("[4.5/5] measuring memory (one forward each, warm) ...")
        rgb0 = sample.images[0]
        reset_cuda_peak(args.device)
        with RSSPoller() as poller:
            _ = da3_depth(da3, rgb0.unsqueeze(0), process_res=args.da3_process_res)
        mem_da3 = MemoryReport(
            param_count=param_count(da3.model.backbone),
            peak_rss_delta_mb=poller.peak_delta_mb,
            peak_cuda_mb=snapshot_cuda_peak(args.device),
        )
        reset_cuda_peak(args.device)
        with RSSPoller() as poller:
            _ = ssm.backbone(rgb0.unsqueeze(0).unsqueeze(0).to(args.device))
        mem_ssm = MemoryReport(
            param_count=param_count(ssm.backbone),
            peak_rss_delta_mb=poller.peak_delta_mb,
            peak_cuda_mb=snapshot_cuda_peak(args.device),
        )
        print(f"  DA3   : {mem_da3.param_count/1e6:.2f} M params, "
              f"{mem_da3.peak_rss_delta_mb:.1f} MB RSS-delta")
        print(f"  SSM-3D: {mem_ssm.param_count/1e6:.2f} M params, "
              f"{mem_ssm.peak_rss_delta_mb:.1f} MB RSS-delta")

    print("[5/5] writing visualizations ...")
    for i in range(N):
        rgb = sample.images[i]
        gt = sample.gt_depth[i]
        valid = sample.valid_mask[i]
        if da3 is not None and torch.isfinite(da3_depth_preds[i]).any():
            ssm_pred = ssm_depth_preds[i] if i < len(ssm_depth_preds) else None
            save_depth_grid(
                rgb, gt, da3_depth_preds[i], valid,
                args.out_root / f"depth_grid_{i:02d}.png",
                pred_ssm=ssm_pred,
            )
            save_error_heatmap(
                da3_depth_preds[i], gt, valid,
                args.out_root / f"error_{i:02d}.png",
                pred_ssm=ssm_pred,
            )
        if len(da3_feats_all) > i:
            save_feature_comparison(
                rgb, da3_feats_all[i], da3_grid,
                ssm_feats_all[i], ssm_grid,
                args.out_root / f"features_{i:02d}.png",
            )

    if da3 is not None:
        save_depth_metric_bars(
            per_image_depth,
            args.out_root / "metric_bars_depth.png",
            per_image_ssm=per_image_depth_ssm if any(m for m in per_image_depth_ssm) else None,
        )
    save_repr_metric_bars(
        per_image_repr_da3, per_image_repr_ssm, args.out_root / "metric_bars_repr.png"
    )
    if da3 is not None:
        save_memory_bars(
            mem_da3.as_dict(), mem_ssm.as_dict(),
            args.out_root / "metric_bars_memory.png",
        )
    print("  rendering architecture diagrams ...")
    for p in render_arch_diagrams(args.out_root):
        print(f"    {p}")
    memory_note = (
        "**Memory note.** Parameters reported are **backbone only** so the comparison "
        "reflects the architectural change (DA3 softmax attention vs. SSM-3D "
        "Mamba-3 SSD). DA3's full published model is ~34 M because it ships "
        "additional DPT, cam-enc and cam-dec heads (~12 M). Peak RSS is the "
        "delta during one warm forward pass: DA3 uses its standard `inference()` "
        "path at process_res=504, SSM-3D uses `backbone.forward` at img_size "
        "(typically 224). Numbers are comparable as \"what the deployed "
        "inference path costs,\" not as an isolated activation-memory benchmark.\n\n"
    )
    ssm_has_trained_depth = any(m for m in per_image_depth_ssm)
    if ssm_has_trained_depth:
        depth_note = (
            "SSM-3D depth comes from the Phase-B-distilled backbone + Phase-C-"
            "fine-tuned DimBridge feeding DA3's frozen DualDPT. Predictions are "
            "scale-ambiguous, so both DA3 and SSM-3D are median-aligned to GT "
            "before scoring — standard MiDaS/DA3 eval convention."
        )
    else:
        depth_note = (
            "Depth numbers in this table are DA3 vs GT. SSM-3D's depth panel in "
            "`depth_grid_*.png` / `error_*.png` comes from a **shared-DPT smoke "
            "test**: DA3's pretrained DualDPT is bolted onto SSM-3D's 4 "
            "intermediate layers (384-dim features duplicated to 768-dim to match "
            "DA3's cat_token format). The SSM-3D side of those figures is therefore "
            "qualitative, not a trained depth head."
        )
    note = (
        memory_note +
        "DA3-SMALL features are 768-dim (`cat_token=True`); SSM-3D features are "
        "384-dim. Representation metrics are dim-invariant and compared as scores.\n\n"
        + depth_note + "\n\n"
        "Interpretation: **lower** `feat_cos_mean` is better (less token collapse); "
        "**higher** `effective_rank` and `cross_view_nn_agreement` are better. "
        "DA3's high `feat_cos_mean` is a known property of `cat_token=True`: each "
        "patch token is concatenated with a global pooled token, which pushes "
        "pairwise cosines up without hurting downstream depth accuracy. "
        "`cross_view_nn_agreement` (GT-warped across image i <-> (i+1) mod N) is "
        "the cleanest head-to-head feature-quality signal here."
    )
    if args.skip_da3:
        note = "Ran with --skip-da3: DA3 rows are empty. " + note
    write_summary_md(
        args.out_root / "summary.md",
        da3_depth=per_image_depth,
        da3_repr=per_image_repr_da3,
        ssm_repr=per_image_repr_ssm,
        note=note,
        memory_da3=mem_da3.as_dict() if da3 is not None else None,
        memory_ssm=mem_ssm.as_dict() if da3 is not None else None,
        ssm_depth=per_image_depth_ssm if ssm_has_trained_depth else None,
    )

    print("\ndone. Outputs:")
    for p in sorted(args.out_root.glob("*.png"))[:12]:
        print(f"  {p}")
    print(f"  {args.out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
