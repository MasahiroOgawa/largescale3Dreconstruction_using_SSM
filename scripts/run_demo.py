"""One-command end-to-end demo.

Produces four PNGs under `outputs/demo/`:

  1. outputs/demo/feature_pca_view{i}.png       - PCA of backbone features
  2. outputs/demo/depth_view{i}.png              - predicted depth map
  3. outputs/demo/cross_attention.png            - cross-view attention heatmap
  4. outputs/demo/seg_overlay_coco{i}.png        - instance-seg overlay on COCO-mini

Usage:
    uv run python scripts/run_demo.py --data-root data --out-root outputs/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ssm3d.data.coco_mini import load_coco_mini
from ssm3d.data.eth3d import download_eth3d_terrains, load_eth3d_scene
from ssm3d.model import SSM3DNet
from ssm3d.train.overfit import overfit_run
from ssm3d.viz import (
    TinyInstanceSegHead,
    save_cross_attention_heatmap,
    save_depth_colormap,
    save_feature_pca,
    save_seg_overlay,
    train_seg_head,
)
from ssm3d.weights import load_dinov2_backbone, warm_start_mamba3_from_qkv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_dinov2 import ensure_dinov2_vits14  # noqa: E402


def _patch_grid(img_size: int, patch_size: int) -> tuple[int, int]:
    s = img_size // patch_size
    return s, s


def collapse_smoke_check(
    net: SSM3DNet, images: torch.Tensor, cross_attn: torch.Tensor | None = None
) -> None:
    """Print diagnostic numbers flagged in PLAN.md §3. Warn when outputs look flat.

    - feat_cos_mean: mean off-diagonal cosine sim between patch tokens in view 0.
      > 0.7 means features have collapsed.
    - depth_std: std of predicted depth for view 0 over [0, 1].
      < 0.02 means depth is effectively constant.
    - cross_attn_row_max: max of the (softmaxed) cross-attn row for the centre query.
      < 2 / T_kv means the attention is flatter than 2x uniform.
    """
    net.eval()
    with torch.no_grad():
        out = net(images.unsqueeze(0))
    feats = out["features"][0, 0]  # (N, C)
    depth0 = out["depth"][0, 0]  # (1, H, W)

    fn = torch.nn.functional.normalize(feats, dim=-1)
    cos = fn @ fn.transpose(0, 1)
    n = cos.shape[0]
    off_diag = (cos.sum() - cos.diagonal().sum()) / (n * (n - 1))
    feat_cos_mean = float(off_diag)
    depth_std = float(depth0.std())

    warn = []
    print(f"  feat_cos_mean (patches): {feat_cos_mean:.3f}   (warn if > 0.7)")
    if feat_cos_mean > 0.7:
        warn.append("feat_cos_mean")
    print(f"  depth_std (view 0):      {depth_std:.4f}       (warn if < 0.02)")
    if depth_std < 0.02:
        warn.append("depth_std")
    if cross_attn is not None:
        T_kv = cross_attn.shape[-1]
        row = torch.softmax(cross_attn[0].mean(dim=0), dim=-1)  # (T_q, T_kv)
        cross_row_max = float(row.max())
        thresh = 2.0 / T_kv
        print(f"  cross_attn_row_max:      {cross_row_max:.4f}     (warn if < {thresh:.4f})")
        if cross_row_max < thresh:
            warn.append("cross_attn_row_max")
    if warn:
        print(f"[WARN] outputs are likely to look flat; see PLAN.md §3 ({', '.join(warn)})")


def run_feature_and_depth_visuals(
    net: SSM3DNet, images: torch.Tensor, out_root: Path
) -> None:
    """images: (S, 3, H, W). Runs net, saves feature_pca + depth per view."""
    net.eval()
    with torch.no_grad():
        x = images.unsqueeze(0)  # (1, S, 3, H, W)
        out = net(x)
    feats = out["features"][0]  # (S, N, C)
    h, w = out["grid_hw"]
    depth = out["depth"][0]  # (S, 1, H, W)
    img_h, img_w = images.shape[-2:]
    for i in range(feats.shape[0]):
        save_feature_pca(
            feats[i],
            out_root / f"feature_pca_view{i}.png",
            spatial_hw=(h, w),
            upsample_to=(img_h, img_w),
        )
        # Honest label (PLAN §9e): the depth head is randomly-initialised and
        # only trained on a self-supervised smoothness+variance surrogate, so
        # the output is not metric depth. Call it what it is.
        save_depth_colormap(depth[i], out_root / f"depth_head_activation_view{i}.png")


def run_cross_attention_visual(
    net: SSM3DNet, images: torch.Tensor, out_root: Path
) -> torch.Tensor:
    """Compute cross-view attention as direct cosine similarity between the
    backbone's patch features for view 0 (query) and view 1 (kv), and save
    the heatmap for a centre-ish query patch.

    Per PLAN §9e: the random-init `Mamba3CrossAttention` previously dominated
    the heatmap with the decay mask's raster-order shape rather than feature
    similarity. Direct cosine-sim *is* the quantity the viz wants to show,
    and the cross module itself is covered by unit tests.
    """
    assert images.shape[0] >= 2, "need at least 2 views for cross-view attention"
    net.eval()
    with torch.no_grad():
        out = net(images.unsqueeze(0))
    feats = out["features"][0]  # (S, N, C)
    h, w = out["grid_hw"]

    q = torch.nn.functional.normalize(feats[0], dim=-1)  # (N, C)
    kv = torch.nn.functional.normalize(feats[1], dim=-1)  # (N, C)
    sim = q @ kv.transpose(-2, -1)  # (T_q, T_kv)
    attn = sim.unsqueeze(0).unsqueeze(0)  # (B=1, H=1, T_q, T_kv)

    query_index = (h // 2) * w + (w // 2)
    save_cross_attention_heatmap(
        attn_map=attn,
        kv_image=images[1],
        kv_grid_hw=(h, w),
        query_index=query_index,
        path=out_root / "cross_attention.png",
    )
    return attn


def run_coco_seg_visual(
    net: SSM3DNet,
    data_root: Path,
    out_root: Path,
    num_images: int = 6,
    iters: int = 100,
) -> None:
    samples = load_coco_mini(data_root, num_images=num_images, image_size=net.backbone.img_size)
    in_c = net.backbone.embed_dim
    h, w = _patch_grid(net.backbone.img_size, net.backbone.patch_size)
    head = TinyInstanceSegHead(in_channels=in_c, hidden=64, image_size=net.backbone.img_size)

    def extractor(img: torch.Tensor) -> torch.Tensor:
        # img: (3, H, W) -> (C, h, w) backbone features
        x = img.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
        with torch.no_grad():
            grid = net.features_grid(x)  # (1, C, h, w)
        return grid[0]

    losses = train_seg_head(head, extractor, samples, iters=iters, lr=3e-3, device="cpu")
    print(f"seg-head loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    for i, sample in enumerate(samples[:3]):
        save_seg_overlay(head, extractor, sample, out_root / f"seg_overlay_coco{i}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out-root", type=Path, default=Path("outputs/demo"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="must be 14 to load DINOv2-small weights as-is",
    )
    ap.add_argument("--depth", type=int, default=12, help="number of transformer blocks")
    ap.add_argument("--overfit-iters", type=int, default=30)
    ap.add_argument("--seg-iters", type=int, default=300)
    ap.add_argument("--num-views", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="path to a DINOv2-compatible checkpoint. If omitted, auto-downloads DINOv2-small.",
    )
    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help="skip loading DINOv2 weights (random-init backbone; outputs will be worse).",
    )
    ap.add_argument(
        "--warm-start",
        action="store_true",
        help=(
            "cast DINOv2 QKV into Mamba-3 BCV projections. Off by default: "
            "empirically this *worsens* feat_cos_mean (0.14 → 0.97) because "
            "SSD attention ≠ softmax and the structured-but-wrong attention "
            "output corrupts DINOv2's MLP activation distribution. Useful as "
            "a starting point only once downstream training is added."
        ),
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)

    print("[1/5] downloading + loading ETH3D terrains ...")
    scene_dir = download_eth3d_terrains(args.data_root)
    sample = load_eth3d_scene(scene_dir, max_images=args.num_views, image_size=args.img_size)
    print(f"  -> loaded {sample.images.shape}")

    print("[2/5] building SSM3DNet with Mamba3 attention swapped in ...")
    net = SSM3DNet(
        size="small",
        img_size=args.img_size,
        patch_size=args.patch_size,
        depth=args.depth,
        head_hidden=64,
    )
    if not args.no_pretrained:
        ckpt = args.pretrained or ensure_dinov2_vits14()
        load_dinov2_backbone(net.backbone.vit, ckpt)
        if args.warm_start:
            warm_start_mamba3_from_qkv(net.backbone.vit, ckpt)

    print(f"[3/5] short overfit ({args.overfit_iters} iters, head-only) ...")
    batch = sample.images.unsqueeze(0)  # (1, S, 3, H, W)
    # Freeze the backbone per PLAN §9a: with no depth GT the anti-collapse
    # hinge otherwise destroys backbone features by re-wiring one axis to the
    # depth head.
    result = overfit_run(
        net, batch, iters=args.overfit_iters, lr=3e-3, device="cpu", trainable="head"
    )
    print(f"  -> loss {result.initial_loss:.4f} -> {result.final_loss:.4f}")

    print("[4/5] writing feature-PCA + depth + cross-view visuals ...")
    run_feature_and_depth_visuals(net, sample.images, args.out_root)
    attn = (
        run_cross_attention_visual(net, sample.images, args.out_root)
        if args.num_views >= 2
        else None
    )
    print("  collapse smoke-check:")
    collapse_smoke_check(net, sample.images, cross_attn=attn)

    print("[5/5] COCO-mini instance-seg demo ...")
    run_coco_seg_visual(net, args.data_root, args.out_root, num_images=15, iters=args.seg_iters)

    print("done. outputs:")
    for p in sorted(args.out_root.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
