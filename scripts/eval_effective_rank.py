"""Mean effective_rank of SSM-3D backbone features for one or more ckpts.

Used to diagnose where rank is being lost. PLAN §15.23/§15.24.

Example:
    uv run python scripts/eval_effective_rank.py \\
        --ckpts outputs/runs/depth_ft_cm24/ckpt_1000.pt \\
                outputs/runs/depth_ft_cm26/ckpt_1000.pt \\
        --state-dims 64 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_dinov2 import ensure_dinov2_vits14  # noqa: E402

from ssm3d.data.eth3d import download_eth3d_terrains, load_eth3d_scene
from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, da3_features, load_da3
from ssm3d.eval.metrics import effective_rank
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_dinov2_backbone


def _build_ssm(ckpt: Path, state_dim: int, device: str,
               img_size: int, patch_size: int, chunk_size: int) -> SSM3DNet:
    net = SSM3DNet(
        size="small", img_size=img_size, patch_size=patch_size,
        depth=12, chunk_size=chunk_size, mamba_state_dim=state_dim,
    )
    load_dinov2_backbone(net.backbone.vit, ensure_dinov2_vits14())
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(state["student"])
    net.to(device).eval()
    return net


@torch.inference_mode()
def measure(ckpt: Path, state_dim: int, images: torch.Tensor, device: str,
            img_size: int, patch_size: int, chunk_size: int) -> float:
    net = _build_ssm(ckpt, state_dim, device, img_size, patch_size, chunk_size)
    feats = net.backbone(images.unsqueeze(0)).features[0]
    ranks = [effective_rank(feats[i]) for i in range(feats.shape[0])]
    return sum(ranks) / len(ranks)


@torch.inference_mode()
def measure_per_layer(ckpt: Path, state_dim: int, images: torch.Tensor, device: str,
                      img_size: int, patch_size: int, chunk_size: int) -> list[float]:
    """Effective_rank at the output of each Mamba-3 block (12 blocks for size=small)."""
    net = _build_ssm(ckpt, state_dim, device, img_size, patch_size, chunk_size)
    captured: dict[int, torch.Tensor] = {}

    def hook(idx: int):
        def f(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[idx] = t.detach()
        return f

    handles = [blk.register_forward_hook(hook(i)) for i, blk in enumerate(net.backbone.vit.blocks)]
    try:
        _ = net.backbone(images.unsqueeze(0))
    finally:
        for h in handles:
            h.remove()

    per_layer: list[float] = []
    for i in sorted(captured):
        tokens = captured[i][:, 1:, :]  # strip cls; (N_img, 1296, C)
        ranks = [effective_rank(tokens[k]) for k in range(tokens.shape[0])]
        per_layer.append(sum(ranks) / len(ranks))
    return per_layer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=Path, nargs="*", default=[])
    ap.add_argument(
        "--state-dims", type=int, nargs="*", default=[],
        help="One state_dim per ckpt (in order).",
    )
    ap.add_argument(
        "--include-da3-teacher", action="store_true",
        help="Also measure DA3-SMALL teacher effective_rank on the same images "
             "(§15.24 probe 1).",
    )
    ap.add_argument(
        "--per-layer", action="store_true",
        help="Measure effective_rank at each Mamba-3 block output (§15.24 probe 2). "
             "Prints one row per layer, one column per ckpt.",
    )
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if len(args.ckpts) != len(args.state_dims):
        ap.error("--ckpts and --state-dims must have equal length")

    scene_dir = download_eth3d_terrains(args.data_root, scene="terrains", download_depth=False)
    sample = load_eth3d_scene(
        scene_dir, max_images=args.max_images, image_size=args.img_size, load_gt_depth=False
    )
    images = sample.images.to(args.device)

    if args.per_layer:
        per_ckpt: list[tuple[str, list[float]]] = []
        for ckpt, sd in zip(args.ckpts, args.state_dims):
            ranks = measure_per_layer(
                ckpt, sd, images, args.device, args.img_size, args.patch_size, args.chunk_size
            )
            per_ckpt.append((f"{ckpt.parent.name}/{ckpt.name} (sd={sd})", ranks))
        n_layers = len(per_ckpt[0][1]) if per_ckpt else 0
        col_w = 24
        header = "layer".ljust(8) + "".join(name.rjust(col_w) for name, _ in per_ckpt)
        print("\n" + header)
        print("-" * len(header))
        for k in range(n_layers):
            row = f"{k:<8}" + "".join(f"{r[k]:>{col_w}.2f}" for _, r in per_ckpt)
            print(row)
    else:
        print(f"\n{'source':<48} {'C':>4} {'state_dim':>10} {'eff_rank':>10}")
        print("-" * 78)
        for ckpt, sd in zip(args.ckpts, args.state_dims):
            er = measure(ckpt, sd, images, args.device, args.img_size, args.patch_size, args.chunk_size)
            print(f"{str(ckpt):<48} {384:>4} {sd:>10} {er:>10.2f}")

        if args.include_da3_teacher:
            da3 = load_da3(DEFAULT_HF_MODEL, device=args.device)
            with torch.inference_mode():
                tokens, _ = da3_features(da3, images, process_res=args.img_size)
            ranks = [effective_rank(tokens[i]) for i in range(tokens.shape[0])]
            er = sum(ranks) / len(ranks)
            C = tokens.shape[-1]
            print(f"{'DA3-SMALL teacher (last layer)':<48} {C:>4} {'-':>10} {er:>10.2f}")


if __name__ == "__main__":
    main()
