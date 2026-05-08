"""V4-init scene-overfit: assemble per-layer mamba3 init weights, then
joint all-unfreeze fine-tune.

Pipeline (PLAN §15.59.7):

  1. Build a fully-mamba3 DA3-SMALL student (all 16 layers swapped, warm-start).
  2. For each layer k ∈ 0..15, find the test-loss-minimum ckpt under
     `outputs/runs/per_layer_overfit/layer_KK/` (from `test_loss.json`) and
     copy that layer's `attn.*` keys into the warm-start state dict.
  3. Save the assembled state dict as `<out>/ckpt_init.pt`.
  4. Invoke `mamba3_attn.train.train_super` with
       --super 3 --sub 3
       --init-ckpt <ckpt_init.pt>
       --cam-posed
       (V2 low-LR recipe: lr_attn=1e-5, lr_head=5e-6, lr_other=1e-6,
        steps=200, ckpt-every=50, warmup=20, decay=50)
  5. Eval each saved ckpt against the same scene/split with
     `mamba3_attn.eval.phase4_evaluator`.

Inputs:
  --per-layer-out  outputs/runs/per_layer_overfit/  (test_loss.json + layer_KK ckpts)
  --out            outputs/runs/scene_overfit_perlayer_init_terrains/
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from mamba3_attn.eval.phase4_evaluator import build_patched_api

REPO = Path(__file__).resolve().parents[1]


def _layer_prefix(k: int, n_backbone: int) -> str:
    """State-dict key prefix for the attention module at flat layer index k."""
    if k < n_backbone:
        return f"backbone.pretrained.blocks.{k}.attn."
    return f"cam_enc.trunk.{k - n_backbone}.attn."


def _best_ckpt_step_per_layer(per_layer_out: Path) -> dict[int, int]:
    """Pick the test-loss-minimum step per layer from test_loss.json."""
    test_loss_path = per_layer_out / "test_loss.json"
    test_loss = json.loads(test_loss_path.read_text())
    return {
        int(k): int(min(v, key=lambda s: v[s]["test"]["total"]))
        for k, v in test_loss.items()
    }


def _assemble_init_ckpt(
    per_layer_out: Path, out_path: Path, n_backbone: int, state_dim: int, device: str,
) -> dict[int, int]:
    """Build all-mamba3 warm-start, splice per-layer trained mamba3 weights in,
    save to out_path. Returns the (layer -> source step) map.
    """
    print("[scene-init] building all-mamba3 warm-start student", flush=True)
    api_base = build_patched_api(
        ckpt_path=None, device=device, state_dim=state_dim,
        patched=True, swap_layers=None,
    )
    base_state: dict[str, torch.Tensor] = {
        k: v.detach().cpu() for k, v in api_base.model.state_dict().items()
    }
    del api_base

    best_steps = _best_ckpt_step_per_layer(per_layer_out)
    print(f"[scene-init] per-layer source steps: {best_steps}", flush=True)

    n_total_layers = max(best_steps) + 1
    for k in range(n_total_layers):
        if k not in best_steps:
            print(f"[scene-init] layer {k:02d}: no per-layer ckpt — keeping warm-start", flush=True)
            continue
        step = best_steps[k]
        ckpt = torch.load(
            per_layer_out / f"layer_{k:02d}" / f"ckpt_{step}.pt",
            map_location="cpu", weights_only=False,
        )
        layer_state = ckpt["model"]
        prefix = _layer_prefix(k, n_backbone)
        matched = {key: layer_state[key] for key in layer_state if key.startswith(prefix)}
        if not matched:
            raise RuntimeError(
                f"layer {k:02d}: no keys matched prefix {prefix!r} in "
                f"{per_layer_out / f'layer_{k:02d}' / f'ckpt_{step}.pt'}"
            )
        base_state.update(matched)
        print(f"[scene-init] layer {k:02d}: copied {len(matched):3d} keys "
              f"from step {step} (prefix {prefix})", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": -1, "model": base_state}, out_path)
    print(f"[scene-init] wrote {out_path}", flush=True)
    return best_steps


def _run_training(out_dir: Path, init_ckpt: Path, args) -> int:
    cmd = [
        "uv", "run", "python", "-m", "mamba3_attn.train.train_super",
        "--super", "3", "--sub", "3",
        "--init-ckpt", str(init_ckpt),
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
        "--lr-attn", str(args.lr_attn),
        "--lr-head", str(args.lr_head),
        "--lr-other", str(args.lr_other),
        "--cam-posed",
        "--out-dir", str(out_dir),
    ]
    log_path = out_dir / "train.log"
    print(f"\n$ {' '.join(cmd)}\n  → {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        return subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO).returncode


def _run_eval(out_dir: Path, ckpt_step: int, split_json: Path, args) -> int:
    cmd = [
        "uv", "run", "python", "-m", "mamba3_attn.eval.phase4_evaluator",
        "--scene-overfit", args.scene, "--scene-dataset", args.dataset,
        "--image-size", str(args.image_size),
        "--state-dim", str(args.state_dim),
        "--max-images", str(args.max_images),
        "--split-json", str(split_json),
        "--ckpt", str(out_dir / f"ckpt_{ckpt_step}.pt"),
    ]
    log_path = out_dir / f"eval_ckpt_{ckpt_step}.log"
    print(f"\n$ {' '.join(cmd)}\n  → {log_path}", flush=True)
    with log_path.open("w") as logf:
        return subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-layer-out", type=Path, required=True,
                    help="Directory from scripts/per_layer_overfit.py "
                    "(must contain test_loss.json + layer_KK/ckpt_*.pt).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir for the all-unfreeze run.")
    ap.add_argument("--scene", default="terrains")
    ap.add_argument("--dataset", default="eth3d", choices=["eth3d", "hiroom", "7scenes"])
    ap.add_argument("--n-backbone-layers", type=int, default=12)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    # V2 low-LR recipe (PLAN §15.59.4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--decay-steps", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--lr-attn", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=5e-6)
    ap.add_argument("--lr-other", type=float, default=1e-6)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--candidate-views", type=int, default=256)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--skip-train", action="store_true",
                    help="Re-run only the eval phase against existing ckpts.")
    args = ap.parse_args()

    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    init_ckpt = args.out / "ckpt_init.pt"

    if not args.skip_train:
        sources = _assemble_init_ckpt(
            args.per_layer_out, init_ckpt, args.n_backbone_layers,
            args.state_dim, args.device,
        )
        (args.out / "init_sources.json").write_text(json.dumps(sources, indent=2))

        rc = _run_training(args.out, init_ckpt, args)
        if rc != 0:
            print(f"[scene-init] TRAIN FAILED (rc={rc})", flush=True)
            return

    split_json = args.out / "split.json"
    if not split_json.exists():
        print(f"[scene-init] no split.json under {args.out}", flush=True)
        return
    ckpt_steps = list(range(args.ckpt_every, args.steps + 1, args.ckpt_every))
    for s in ckpt_steps:
        ckpt = args.out / f"ckpt_{s}.pt"
        if not ckpt.exists():
            print(f"[scene-init] {ckpt.name} missing, skipping eval", flush=True)
            continue
        log = args.out / f"eval_ckpt_{s}.log"
        if log.exists() and "MEAN" in log.read_text():
            print(f"[scene-init] eval_ckpt_{s}.log already MEAN-complete, skipping",
                  flush=True)
            continue
        _run_eval(args.out, s, split_json, args)

    print(f"\n[scene-init] DONE. Outputs under {args.out}", flush=True)


if __name__ == "__main__":
    main()
