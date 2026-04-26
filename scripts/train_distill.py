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
from ssm3d.train.distill import (
    DISTILL_LAYERS,
    DISTILL_LAYERS_LARGE,
    DistillConfig,
    distill,
)
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
        "--alt-start", type=int, default=-1,
        help="CM-FS (§ 15.43): DA3-style cross-view alternation start. "
             "-1 = legacy partial-swap (CM12 → CM30); 4 = full-swap mirror "
             "of DA3-SMALL.",
    )
    ap.add_argument(
        "--cat-token", action="store_true",
        help="CM-FS: required with --alt-start ≥ 0 for full-swap.",
    )
    ap.add_argument(
        "--use-fused-kernel", action="store_true",
        help="CM-FS: route Mamba-3 self-attention through the upstream "
             "Triton kernel (PLAN § 15.46/§ 15.47). Recommended for "
             "full-swap training to get the kernel's 30-150x speedup.",
    )
    ap.add_argument(
        "--lambda-dpt-depth", type=float, default=0.0,
        help="Step 5a-v2 (PLAN § 15.51): weight on DA3-style aleatoric ℓ1 "
             "depth loss (DA3 paper § 3.3 eq. 2). Set to 1.0 for full DA3 "
             "matching, 0 to disable.",
    )
    ap.add_argument(
        "--lambda-dpt-ray", type=float, default=0.0,
        help="Step 5a-v2: weight on aleatoric ℓ1 ray loss. Critical for pose.",
    )
    ap.add_argument(
        "--lambda-dpt-grad", type=float, default=0.0,
        help="Step 5a-v2: weight on depth-gradient ℓ1 loss (DA3 eq. 3). "
             "Preserves edges. DA3 sets α=1.",
    )
    ap.add_argument(
        "--lambda-dpt-conf-log", type=float, default=1.0,
        help="Step 5a-v2: λ_c in the aleatoric log-confidence penalty.",
    )
    ap.add_argument(
        "--no-aleatoric-dpt", action="store_true",
        help="Disable confidence weighting; use plain ℓ1 instead of "
             "aleatoric form.",
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
        "--teacher", type=str, default="depth-anything/DA3-SMALL",
        help="CM20: HF model ID for Phase-B teacher. Use "
             "'depth-anything/DA3-LARGE-1.1' to distill from the ViT-L teacher "
             "(triggers the student-side projector; CC BY-NC license).",
    )
    ap.add_argument(
        "--teacher-layers", type=int, nargs="+", default=None,
        help="CM20: layer indices to distill from the teacher. Defaults to "
             "(5,7,9,11) for 12-block teachers and (11,15,19,23) for 24-block "
             "teachers.",
    )
    ap.add_argument(
        "--student-layers", type=int, nargs="+", default=None,
        help="CM30: layer indices to extract from the student. Must have the "
             "same length as --teacher-layers. Defaults to --teacher-layers "
             "(only correct when teacher and student have the same depth).",
    )
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
        alt_start=args.alt_start,
        cat_token=args.cat_token,
        use_fused_kernel=args.use_fused_kernel,
    )
    teacher = load_da3(hf_model=args.teacher, device=args.device)
    teacher_blocks = len(teacher.model.backbone.pretrained.blocks)
    teacher_dim = teacher.model.backbone.pretrained.embed_dim
    if args.teacher_layers is not None:
        layers = tuple(args.teacher_layers)
    elif teacher_blocks >= 24:
        layers = DISTILL_LAYERS_LARGE
    else:
        layers = DISTILL_LAYERS
    print(
        f"  teacher={args.teacher}  blocks={teacher_blocks}  "
        f"embed_dim={teacher_dim}  layers={layers}"
    )
    # CM20: student stays ViT-S-shaped. When the distill teacher is a bigger
    # model, still init non-attn params from DA3-SMALL so the Phase-A transfer
    # (patch_embed / MLPs / norms) benefit is preserved.
    if teacher_dim == student.backbone.vit.embed_dim:
        load_da3_backbone(student.backbone.vit, teacher, verbose=True)
    else:
        print("  loading DA3-SMALL for student init (shape-compatible) ...")
        small = load_da3(hf_model="depth-anything/DA3-SMALL", device=args.device)
        load_da3_backbone(student.backbone.vit, small, verbose=True)
        del small

    print("[4/4] starting distillation ...")
    student_layers = tuple(args.student_layers) if args.student_layers is not None else None
    if student_layers is not None and len(student_layers) != len(layers):
        ap.error(
            f"--student-layers ({len(student_layers)}) and --teacher-layers "
            f"({len(layers)}) must have the same length"
        )
    cfg = DistillConfig(
        steps=args.steps,
        ckpt_every=args.ckpt_every,
        batch_size=args.batch_size,
        lr_attn=args.lr_attn,
        weight_decay=args.weight_decay,
        amp_dtype=args.amp_dtype,
        device=args.device,
        layers=layers,
        student_layers=student_layers,
        lambda_dpt_depth=args.lambda_dpt_depth,
        lambda_dpt_ray=args.lambda_dpt_ray,
        lambda_dpt_grad=args.lambda_dpt_grad,
        lambda_dpt_conf_log=args.lambda_dpt_conf_log,
        use_aleatoric_dpt=not args.no_aleatoric_dpt,
    )
    data_iter = _build_iter(dataset, seed=args.seed)
    distill(student, teacher, data_iter, cfg, args.out, bridge=None)
    print(f"done. checkpoints in {args.out}")


if __name__ == "__main__":
    main()
