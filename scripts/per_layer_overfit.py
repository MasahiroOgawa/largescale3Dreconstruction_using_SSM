"""Per-layer scene-overfit ablation.

For each flat layer index k in 0..N_layers-1, runs:
  1. Train:  install_mamba3 on layer k only, super=3 sub=1 (only that layer's
             mamba3 attention is trainable), scene-overfit terrains, real
             GT supervision (no feature distillation).
  2. Eval:   each saved ckpt is evaluated on the same held-out test split
             that V2/V4 used.

Outputs per layer go to `<out>/layer_KK/`:
  - split.json (training-time view split)
  - ckpt_*.pt at each ckpt-every step
  - train.log
  - eval_ckpt_*.log per ckpt

Designed to be resumable: skips trainings whose final ckpt already exists, and
evaluations whose log already exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(cmd)}\n  → {log_path}", flush=True)
    with log_path.open("w") as logf:
        return subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO).returncode


def _train_layer(out_dir: Path, layer_k: int, args) -> int:
    cmd = [
        "uv", "run", "python", "-m", "mamba3_attn.train.train_super",
        "--super", "3", "--sub", "1",
        "--scene-overfit", args.scene, "--scene-dataset", args.dataset,
        "--train-frac", str(args.train_frac),
        "--split-seed", str(args.split_seed),
        "--steps", str(args.steps),
        "--warmup-steps", str(args.warmup_steps),
        "--decay-steps", str(args.decay_steps),
        "--chunk-size", str(args.chunk_size),
        "--state-dim", str(args.state_dim),
        "--n-views", str(args.n_views),
        "--image-size", str(args.image_size),
        "--candidate-views", str(args.candidate_views),
        "--frame-stride", str(args.frame_stride),
        "--ckpt-every", str(args.ckpt_every),
        "--swap-layer", str(layer_k),
        "--out-dir", str(out_dir),
    ]
    if _needs_cam_posed(layer_k, args):
        cmd.append("--cam-posed")
    return _run(cmd, out_dir / "train.log")


def _needs_cam_posed(layer_k: int, args) -> bool:
    """Layers ≥ n_backbone_layers live in cam_enc.trunk and need extrinsics fed
    into student.model.forward to be on the loss path. Backbone layers don't
    need cam_posed and we keep them off so already-trained ckpts stay reusable.
    """
    if args.cam_posed_all:
        return True
    return layer_k >= args.n_backbone_layers


def _eval_layer_ckpt(out_dir: Path, layer_k: int, ckpt_step: int,
                      split_json: Path, args) -> int:
    cmd = [
        "uv", "run", "python", "-m", "mamba3_attn.eval.phase4_evaluator",
        "--scene-overfit", args.scene, "--scene-dataset", args.dataset,
        "--image-size", str(args.image_size),
        "--state-dim", str(args.state_dim),
        "--max-images", str(args.max_images),
        "--split-json", str(split_json),
        "--ckpt", str(out_dir / f"ckpt_{ckpt_step}.pt"),
        "--swap-layer", str(layer_k),
    ]
    return _run(cmd, out_dir / f"eval_ckpt_{ckpt_step}.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--layers", type=str, default=None,
                    help="Comma-separated subset of layer indices (default: 0..n_layers-1).")
    ap.add_argument("--scene", default="terrains")
    ap.add_argument("--dataset", default="eth3d", choices=["eth3d", "hiroom", "7scenes"])
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--decay-steps", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--candidate-views", type=int, default=256)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--n-backbone-layers", type=int, default=12,
                    help="Number of backbone (non-cam_enc) layers. Layers below this index "
                    "are trained image-only; layers ≥ this index live in cam_enc.trunk and "
                    "are auto-trained with --cam-posed.")
    ap.add_argument("--cam-posed-all", action="store_true",
                    help="Force --cam-posed on every layer (including backbone). Default "
                    "is to apply --cam-posed only to cam_enc.trunk layers.")
    args = ap.parse_args()

    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    layers = (
        [int(x) for x in args.layers.split(",")] if args.layers
        else list(range(args.n_layers))
    )
    ckpt_steps = list(range(args.ckpt_every, args.steps + 1, args.ckpt_every))

    # Phase 1: train each layer (skip if already done).
    for k in layers:
        layer_dir = args.out / f"layer_{k:02d}"
        final_ckpt = layer_dir / f"ckpt_{args.steps}.pt"
        if final_ckpt.exists():
            print(f"[per_layer] layer {k:02d}: {final_ckpt.name} exists, skipping training", flush=True)
            continue
        rc = _train_layer(layer_dir, k, args)
        if rc != 0:
            print(f"[per_layer] layer {k:02d} TRAIN FAILED (rc={rc})", flush=True)

    # Resolve canonical split (any layer's split.json — they're identical given the
    # same scene/seed/frac).
    canonical_split: Path | None = None
    for k in layers:
        sp = args.out / f"layer_{k:02d}" / "split.json"
        if sp.exists():
            canonical_split = sp
            break
    if canonical_split is None:
        print("[per_layer] no split.json found — eval phase aborted", flush=True)
        return

    # Phase 2: eval each ckpt for each layer (skip if log exists).
    for k in layers:
        layer_dir = args.out / f"layer_{k:02d}"
        if not (layer_dir / f"ckpt_{args.steps}.pt").exists():
            continue
        for s in ckpt_steps:
            log = layer_dir / f"eval_ckpt_{s}.log"
            if log.exists():
                print(f"[per_layer] layer={k:02d} ckpt_{s}: eval exists, skipping", flush=True)
                continue
            _eval_layer_ckpt(layer_dir, k, s, canonical_split, args)

    print(f"\n[per_layer] DONE. Outputs under {args.out}", flush=True)


if __name__ == "__main__":
    main()
