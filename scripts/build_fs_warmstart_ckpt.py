"""Build a 'warm-started' full-swap (alt_start=4, cat_token=True) ckpt that
loads DA3-SMALL non-attn weights and initialises Mamba-3 B/C/V from DA3 qkv.

No training. The output ckpt has the structure the existing eval scripts
(eval_ckpt_sweep.py, eval_recon_metrics.py, eval_ray_metrics.py) expect:
  state["student"]: backbone state dict
  state["bridge"]:  None (no learned bridge yet)
  state["dualdpt"]: None (use DA3 native head)

Used in PLAN § 15.45 to measure accuracy of the architectural swap before
any training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_da3_backbone, warm_start_mamba3_from_qkv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/fs_warmstart/ckpt_warmstart.pt"))
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    da3 = load_da3(DEFAULT_HF_MODEL, device=args.device)

    net = SSM3DNet(
        size="small",
        img_size=args.img_size,
        patch_size=args.patch_size,
        depth=12,
        chunk_size=args.chunk_size,
        mamba_state_dim=args.state_dim,
        alt_start=4,
        cat_token=True,
    )
    net.to(args.device).eval()

    print("[1/2] loading DA3-SMALL non-attention weights into full-swap backbone ...")
    load_da3_backbone(net.backbone.vit, da3, verbose=True)
    print("[2/2] warm-starting Mamba-3 B/C/V from DA3 qkv ...")
    da3_state = da3.model.backbone.pretrained.state_dict()
    warm_start_mamba3_from_qkv(
        net.backbone.vit, da3_state, prefix="",
        num_tokens=(args.img_size // args.patch_size) ** 2, verbose=True,
    )

    state = {"student": net.state_dict(), "bridge": None, "dualdpt": None}
    torch.save(state, args.out)
    print(f"\nckpt saved → {args.out}")


if __name__ == "__main__":
    main()
