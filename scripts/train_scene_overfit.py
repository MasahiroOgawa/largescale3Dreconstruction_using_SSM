"""End-to-end per-scene-overfit comparison (PLAN §15.59).

Trains four variants on the same view-level train/test split of one scene
and runs `phase4_evaluator` on the held-out test views for each:

    1. un-patched DA3-SMALL, full overfit (ceiling baseline)
    2. patched DA3 (Mamba-3 swap), full overfit
    3. patched DA3 (Mamba-3 swap), head-only (frozen attentions)
    4. un-patched DA3-SMALL, zero-shot (no training)

Writes a comparison table to `<out>/comparison.md`. The Triton kernel is
mandatory (PLAN §15.59 "Recommended approach"); there is no opt-out.

Usage:

    uv run python scripts/train_scene_overfit.py \\
        --scene terrains --dataset eth3d \\
        --train-frac 0.75 --split-seed 42 \\
        --steps 5000 --warmup-steps 200 --decay-steps 500 \\
        --img-size 504 --chunk-size 128 \\
        --out outputs/runs/scene_overfit_terrains
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Variant:
    name: str
    label: str
    train_args: list[str] | None  # None = no training (zero-shot)
    eval_args: list[str]  # appended to base eval args


def _build_variants(args) -> list[Variant]:
    """Define the four overfit-comparison variants."""
    base_train = [
        "--super", "3", "--sub", "3",
        "--scene-overfit", args.scene,
        "--scene-dataset", args.dataset,
        "--train-frac", str(args.train_frac),
        "--split-seed", str(args.split_seed),
        "--steps", str(args.steps),
        "--warmup-steps", str(args.warmup_steps),
        "--decay-steps", str(args.decay_steps),
        "--chunk-size", str(args.chunk_size),
        "--state-dim", str(args.state_dim),
        "--n-views", str(args.n_views),
        "--image-size", str(args.img_size),
        "--candidate-views", str(args.candidate_views),
        "--frame-stride", str(args.frame_stride),
        "--ckpt-every", str(args.ckpt_every),
    ]
    if args.no_augment:
        base_train += ["--no-augment"]
    if args.lr_attn:
        base_train += ["--lr-attn", str(args.lr_attn)]
    if args.lr_head:
        base_train += ["--lr-head", str(args.lr_head)]
    if args.lr_other:
        base_train += ["--lr-other", str(args.lr_other)]

    variants: list[Variant] = []

    # 1. Un-patched DA3-SMALL zero-shot reference (no training).
    # Pipeline-correctness check: should land near the DA3 paper's terrains
    # number; large gap implies our eval pipeline (split / image_size / metric
    # defs) is buggy and all other rows are untrustworthy.
    variants.append(Variant(
        name="unpatched_zeroshot",
        label="1. DA3-SMALL zero-shot (pipeline check)",
        train_args=None,
        eval_args=["--no-patch"],
    ))

    # 2. Un-patched DA3-SMALL, full scene-overfit (recipe ceiling).
    # Tells us how well DA3's architecture itself can fit this scene under our
    # training recipe; bounds the best-case row 4 should approach.
    variants.append(Variant(
        name="unpatched_overfit",
        label="2. DA3-SMALL un-patched, scene-overfit (recipe ceiling)",
        train_args=base_train + ["--no-mamba3-swap"],
        eval_args=["--no-patch"],
    ))

    # 3. Patched DA3 (Mamba-3), head-only (frozen attentions, just DPT + cam_dec adapt).
    # Cheapest mamba3 row: tests whether the post-swap mamba3 backbone can be
    # adapted to the scene by retraining only the depth+camera heads.
    # Replace `--sub 3` (all) with `--sub 2` (head); other args identical.
    head_train: list[str] = []
    i = 0
    while i < len(base_train):
        if base_train[i] == "--sub":
            head_train += ["--sub", "2"]
            i += 2
        else:
            head_train.append(base_train[i])
            i += 1
    variants.append(Variant(
        name="patched_head_only",
        label="3. mamba3 swap, head-only adapt",
        train_args=head_train,
        eval_args=[],
    ))

    # 4. Patched DA3 (Mamba-3), full scene-overfit — the row that actually
    # answers "does mamba3 swap match DA3?" (compared against row 2).
    variants.append(Variant(
        name="patched_overfit",
        label="4. mamba3 swap, full unfreeze (the comparison row)",
        train_args=base_train[:],
        eval_args=[],
    ))

    return variants


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(cmd)}\n  → {log_path}", flush=True)
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO)
    return proc.returncode


def _train_one(variant: Variant, out_dir: Path, steps: int) -> Path | None:
    """Returns the path to the final ckpt (or None for zero-shot)."""
    if variant.train_args is None:
        return None
    var_dir = out_dir / variant.name
    final_ckpt = var_dir / f"ckpt_{steps}.pt"
    if final_ckpt.exists():
        print(f"[scene_overfit] {variant.name}: {final_ckpt.name} already exists, skipping training", flush=True)
        return final_ckpt
    cmd = [
        "uv", "run", "python", "-m", "mamba3_attn.train.train_super",
        "--out-dir", str(var_dir),
    ] + variant.train_args
    rc = _run(cmd, var_dir / "train.log")
    if rc != 0:
        raise RuntimeError(f"training failed for {variant.name} (rc={rc}); see {var_dir}/train.log")
    # Latest ckpt
    ckpts = sorted(var_dir.glob("ckpt_*.pt"))
    if not ckpts:
        raise RuntimeError(f"no ckpt produced under {var_dir}")
    return ckpts[-1]


def _eval_one(variant: Variant, out_dir: Path, ckpt: Path | None,
              args, split_json: Path | None) -> Path:
    var_dir = out_dir / variant.name
    var_dir.mkdir(parents=True, exist_ok=True)
    eval_log = var_dir / "eval.log"
    cmd = [
        "uv", "run", "python", "-m", "mamba3_attn.eval.phase4_evaluator",
        "--scene-overfit", args.scene,
        "--scene-dataset", args.dataset,
        "--image-size", str(args.img_size),
        "--state-dim", str(args.state_dim),
        "--max-images", str(args.max_images),
    ]
    if split_json is not None:
        cmd += ["--split-json", str(split_json)]
    if ckpt is not None:
        cmd += ["--ckpt", str(ckpt)]
    cmd += variant.eval_args
    rc = _run(cmd, eval_log)
    if rc != 0:
        raise RuntimeError(f"eval failed for {variant.name} (rc={rc}); see {eval_log}")
    return eval_log


def _parse_metrics_from_eval_log(log_path: Path) -> dict[str, float]:
    """Pull the MEAN row from phase4's print_table output."""
    text = log_path.read_text()
    # Look for the line like:
    # MEAN                    0.0234  0.0089   0.1234   0.0987
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("MEAN"):
            parts = line.split()
            try:
                out["auc30"] = float(parts[-4])
                out["auc15"] = float(parts[-3])
                out["fscore_posed"] = float(parts[-2])
                out["fscore_unposed"] = float(parts[-1])
            except (ValueError, IndexError):
                continue
    return out


def _write_comparison(out_dir: Path, variants: list[Variant],
                      metrics: dict[str, dict[str, float]]) -> Path:
    md = ["# Per-scene overfit comparison (PLAN §15.59)\n"]
    md.append(f"\nScene: `{out_dir.name}`. Held-out test views; metrics are mean across views.\n")
    md.append(
        "\n| Variant | AUC@30° ↑ | AUC@15° ↑ | F-score posed ↑ | F-score unposed ↑ |\n"
        "|---|---|---|---|---|\n"
    )
    for v in variants:
        m = metrics.get(v.name, {})
        def fmt(k: str) -> str:
            x = m.get(k)
            return f"{x:.4f}" if x is not None else "n/a"
        md.append(
            f"| {v.label} | {fmt('auc30')} | {fmt('auc15')} | "
            f"{fmt('fscore_posed')} | {fmt('fscore_unposed')} |\n"
        )

    # Acceptance gates: row 4 (patched_overfit, mamba3-swap full) vs
    # row 2 (unpatched_overfit, DA3 architecture ceiling).
    ceiling = metrics.get("unpatched_overfit", {})
    patched = metrics.get("patched_overfit", {})
    if ceiling and patched:
        md.append("\n## Acceptance gates (row 4 vs row 2, ceiling = un-patched scene-overfit)\n")
        md.append("\n| Metric | Ceiling | Patched | Ratio | Gate (≥ 0.9 of ceiling) |\n|---|---|---|---|---|\n")
        for k, label in [
            ("auc30", "AUC@30°"), ("auc15", "AUC@15°"),
            ("fscore_posed", "F-score posed"), ("fscore_unposed", "F-score unposed"),
        ]:
            c = ceiling.get(k); p = patched.get(k)
            if c is None or p is None or c == 0:
                md.append(f"| {label} | n/a | n/a | n/a | n/a |\n")
                continue
            ratio = p / c
            gate = "✅ PASS" if ratio >= 0.9 else "❌ FAIL"
            md.append(f"| {label} | {c:.4f} | {p:.4f} | {ratio:.3f} | {gate} |\n")

    path = out_dir / "comparison.md"
    path.write_text("".join(md))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset", default="eth3d", choices=["eth3d", "hiroom", "7scenes"])
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--decay-steps", type=int, default=500)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--candidate-views", type=int, default=256)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--lr-attn", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--lr-other", type=float, default=1e-5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", type=str, default="all",
                    help="Comma-separated subset of "
                    "{unpatched_overfit,patched_overfit,patched_head_only,unpatched_zeroshot}, "
                    "or 'all' (default).")
    args = ap.parse_args()

    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    variants = _build_variants(args)
    if args.variants != "all":
        wanted = set(args.variants.split(","))
        variants = [v for v in variants if v.name in wanted]
        if not variants:
            sys.exit(f"no variants matched: {args.variants}")

    # Determine the canonical split.json. Train the first non-zero-shot variant
    # first so its split.json can be reused for the others (same scene + seed +
    # frac → identical split).
    canonical_split: Path | None = None
    for v in variants:
        if v.train_args is not None:
            canonical_split = args.out / v.name / "split.json"
            break

    metrics: dict[str, dict[str, float]] = {}

    for v in variants:
        try:
            ckpt = _train_one(v, args.out, args.steps)
        except Exception as e:
            print(f"[scene_overfit] train {v.name} failed: {e}", flush=True)
            continue

        # If this is the zero-shot variant, derive split from canonical_split.
        split_for_eval = canonical_split if canonical_split is not None and canonical_split.exists() else None
        try:
            eval_log = _eval_one(v, args.out, ckpt, args, split_for_eval)
            metrics[v.name] = _parse_metrics_from_eval_log(eval_log)
            print(f"[scene_overfit] {v.name}: {metrics[v.name]}", flush=True)
        except Exception as e:
            print(f"[scene_overfit] eval {v.name} failed: {e}", flush=True)

    md_path = _write_comparison(args.out, variants, metrics)
    print(f"\n[scene_overfit] wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
