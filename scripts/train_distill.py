"""Phase B entrypoint: distill DA3 backbone features into SSM-3D Mamba-3 mixer.

Usage:
    uv run python scripts/train_distill.py \
        --steps 6000 --batch-size 4 --device cuda \
        --data-root data --out outputs/runs/distill
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ssm3d.data.eth3d_multi import (
    ETH3DMultiSceneDataset,
    TRAIN_SCENES,
    download_eth3d_scenes,
    infinite_sampler,
)
from ssm3d.eval.da3_reference import load_da3
from ssm3d.model import SSM3DNet
from ssm3d.train.distill import DistillConfig, DISTILL_LAYERS, distill
from ssm3d.weights import load_da3_backbone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_dinov2 import ensure_dinov2_vits14  # noqa: E402


def _build_iter(dataset: ETH3DMultiSceneDataset, seed: int):
    sampler_iter = infinite_sampler(dataset, seed=seed)
    while True:
        idx = next(sampler_iter)
        yield dataset[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/distill"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument(
        "--state-dim", type=int, default=64,
        help="CM11: Mamba-3 SSD recurrent state dim (default 64; CM11 uses 32).",
    )
    ap.add_argument(
        "--chunk-size", type=int, default=None,
        help="CM12: chunked SSD query-axis chunk size (None = full T x T mask). "
             "Needed when distilling at img_size>=504 to fit in 12 GB VRAM.",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-attn", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--scenes",
        nargs="+",
        default=list(TRAIN_SCENES),
        help="ETH3D training scenes; `terrains` is hard-rejected.",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="assume scenes are already extracted under data/eth3d/*",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if not args.skip_download:
        print(f"[1/4] downloading {len(args.scenes)} ETH3D scenes ...")
        download_eth3d_scenes(args.data_root, scenes=args.scenes)

    print("[2/4] building dataset ...")
    dataset = ETH3DMultiSceneDataset(
        args.data_root, scenes=args.scenes, image_size=args.img_size
    )
    print(f"  -> {len(dataset)} images across {len(dataset.scenes)} scenes")

    print("[3/4] building student (DA3 backbone loaded) + teacher ...")
    student = SSM3DNet(
        size="small",
        img_size=args.img_size,
        patch_size=args.patch_size,
        depth=12,
        mamba_state_dim=args.state_dim,
        chunk_size=args.chunk_size,
    )
    teacher = load_da3(device=args.device)
    load_da3_backbone(student.backbone.vit, teacher, verbose=True)

    print("[4/4] starting distillation ...")
    cfg = DistillConfig(
        steps=args.steps,
        ckpt_every=args.ckpt_every,
        batch_size=args.batch_size,
        lr_attn=args.lr_attn,
        weight_decay=args.weight_decay,
        amp_dtype=args.amp_dtype,
        device=args.device,
        layers=DISTILL_LAYERS,
    )
    data_iter = _build_iter(dataset, seed=args.seed)
    distill(student, teacher, data_iter, cfg, args.out, bridge=None)
    print(f"done. checkpoints in {args.out}")


if __name__ == "__main__":
    main()
