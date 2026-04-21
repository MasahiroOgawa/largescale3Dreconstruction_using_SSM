"""Phase C entrypoint: depth fine-tune the SSM-3D student on ETH3D GT depth.

Starts from a Phase-B distillation checkpoint. Trains Mamba-3 mixer + DimBridge
while DA3's DualDPT stays frozen.

Usage:
    uv run python scripts/train_depth.py \
        --init outputs/runs/distill/ckpt_6000.pt \
        --steps 2000 --device cuda \
        --data-root data --out outputs/runs/depth_ft
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ssm3d.bridge import DimBridgeStack
from ssm3d.data.eth3d_multi import (
    ETH3DMultiSceneDataset,
    TRAIN_SCENES,
    download_eth3d_scenes,
    infinite_sampler,
)
from ssm3d.eval.da3_reference import load_da3
from ssm3d.eval.dpt_adapter import SHARED_DPT_LAYERS
from ssm3d.model import SSM3DNet
from ssm3d.train.depth_ft import DepthFTConfig, depth_ft
from ssm3d.weights import load_da3_backbone


def _build_iter(dataset: ETH3DMultiSceneDataset, seed: int):
    sampler_iter = infinite_sampler(dataset, seed=seed)
    while True:
        idx = next(sampler_iter)
        yield dataset[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/depth_ft"))
    ap.add_argument("--init", type=Path, default=None, help="Phase-B checkpoint")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr-attn", type=float, default=1e-4)
    ap.add_argument("--lr-bridge", type=float, default=3e-4)
    ap.add_argument("--lambda-edge", type=float, default=0.1)
    ap.add_argument(
        "--freeze-mixer", action="store_true",
        help="CM14: freeze Mamba-3 attn params during Phase-C; train only DimBridge",
    )
    ap.add_argument(
        "--lambda-kd", type=float, default=0.0,
        help="CM17: weight on per-layer L2+cos KD loss against DA3 teacher features "
             "(0 = off, default). Requires mixer trainable to be effective.",
    )
    ap.add_argument(
        "--augment", action="store_true",
        help="CM9: enable random crop + hflip + color jitter on ETH3D (Phase-C only).",
    )
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", nargs="+", default=list(TRAIN_SCENES))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if not args.skip_download:
        print(f"[1/4] downloading {len(args.scenes)} ETH3D scenes (+ depth) ...")
        download_eth3d_scenes(args.data_root, scenes=args.scenes, download_depth=True)

    print("[2/4] building GT-depth dataset ...")
    dataset = ETH3DMultiSceneDataset(
        args.data_root,
        scenes=args.scenes,
        image_size=args.img_size,
        load_gt_depth=True,
        augment=args.augment,
        augment_seed=args.seed,
    )
    print(f"  -> {len(dataset)} images across {len(dataset.scenes)} scenes")

    print("[3/4] building student + teacher ...")
    student = SSM3DNet(
        size="small",
        img_size=args.img_size,
        patch_size=args.patch_size,
        depth=12,
    )
    teacher = load_da3(device=args.device)
    load_da3_backbone(student.backbone.vit, teacher, verbose=True)
    bridge = DimBridgeStack(num_layers=len(SHARED_DPT_LAYERS), in_dim=384)

    if args.init is not None:
        print(f"  loading Phase-B state from {args.init}")
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        student.load_state_dict(state["student"])
        if state.get("bridge") is not None:
            bridge.load_state_dict(state["bridge"])

    print("[4/4] starting depth fine-tune ...")
    cfg = DepthFTConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr_attn=args.lr_attn,
        lr_bridge=args.lr_bridge,
        lambda_edge=args.lambda_edge,
        amp_dtype=args.amp_dtype,
        device=args.device,
        layers=SHARED_DPT_LAYERS,
        freeze_mixer=args.freeze_mixer,
        lambda_kd=args.lambda_kd,
    )
    data_iter = _build_iter(dataset, seed=args.seed)
    depth_ft(student, teacher, data_iter, cfg, args.out, bridge=bridge)
    print(f"done. checkpoints in {args.out}")


if __name__ == "__main__":
    main()
