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
from ssm3d.eval.metrics import effective_rank
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_dinov2_backbone


@torch.inference_mode()
def measure(ckpt: Path, state_dim: int, images: torch.Tensor, device: str,
            img_size: int, patch_size: int, chunk_size: int) -> float:
    net = SSM3DNet(
        size="small", img_size=img_size, patch_size=patch_size,
        depth=12, chunk_size=chunk_size, mamba_state_dim=state_dim,
    )
    load_dinov2_backbone(net.backbone.vit, ensure_dinov2_vits14())
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(state["student"])
    net.to(device).eval()
    feats = net.backbone(images.unsqueeze(0)).features[0]
    ranks = [effective_rank(feats[i]) for i in range(feats.shape[0])]
    return sum(ranks) / len(ranks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--state-dims", type=int, nargs="+", required=True,
        help="One state_dim per ckpt (in order).",
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

    print(f"\n{'ckpt':<48} {'state_dim':>10} {'eff_rank':>10}")
    print("-" * 70)
    for ckpt, sd in zip(args.ckpts, args.state_dims):
        er = measure(ckpt, sd, images, args.device, args.img_size, args.patch_size, args.chunk_size)
        print(f"{str(ckpt):<48} {sd:>10} {er:>10.2f}")


if __name__ == "__main__":
    main()
