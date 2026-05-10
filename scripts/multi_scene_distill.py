"""Random per-scene split + multi-scene supervised training + multi-scene eval.

PLAN §15.59.8 protocol — replaces the §15.59.x scene-overfit-on-terrains
setup with a true generalization test:

  1. Discover every extracted ETH3D scene that has GT depth.
  2. Random per-scene split with seed + train_frac (default 75/25).
  3. Train all-mamba3 student on the train scenes via `mamba3_attn.train.train_super`
     (super=3, GT-supervised). The mamba3 attention starts from `install_mamba3`
     init (no per-scene-overfit warmstart — testing whether multi-scene GT
     supervision alone reaches DA3-SMALL parity on unseen scenes).
  4. Evaluate every saved ckpt on the test scenes via `phase4_evaluator`,
     full multi-view eval (no scene-overfit split).
  5. Aggregate AUC@30°/AUC@15°/F_posed/F_unp across test scenes and dump a
     summary table.

Outputs land under `<out>/`:
  - split.json          — train/test scene assignment + seed
  - ckpt_<step>.pt      — saved checkpoints (every --ckpt-every steps)
  - train.log           — training stdout
  - eval_ckpt_<step>_<dataset>_<scene>.log  — per-test-scene eval log
  - summary.md          — aggregated test-scene metrics per ckpt

Usage:
  uv run python scripts/multi_scene_distill.py \\
    --out outputs/runs/multi_scene_distill_eth3d \\
    --steps 3000 --train-frac 0.75 --split-seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _discover_eth3d_scenes_with_depth(data_root: Path) -> list[str]:
    """List every ETH3D scene under `data_root/eth3d/` that has extracted GT depth."""
    eth3d_root = data_root / "eth3d"
    if not eth3d_root.exists():
        raise RuntimeError(f"missing {eth3d_root}")
    scenes: list[str] = []
    for sd in sorted(eth3d_root.iterdir()):
        if not sd.is_dir():
            continue
        depth_candidates = [
            sd / "ground_truth_depth" / "dslr_images",
            sd / sd.name / "ground_truth_depth" / "dslr_images",
        ]
        if any(p.exists() for p in depth_candidates):
            scenes.append(sd.name)
    return scenes


def _split_scenes(scenes: list[str], train_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = scenes[:]
    rng.shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * train_frac))
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def _scene_arg(scenes: list[str], dataset: str = "eth3d") -> str:
    return ",".join(f"{dataset}:{s}" for s in scenes)


def _run_training(out_dir: Path, train_scenes: list[str], args) -> int:
    cmd = [
        "uv", "run", "python", "-m", "mamba3_attn.train.train_super",
        "--super", "3", "--sub", "3",
        "--scenes", _scene_arg(train_scenes),
        "--steps", str(args.steps),
        "--warmup-steps", str(args.warmup_steps),
        "--decay-steps", str(args.decay_steps),
        "--chunk-size", str(args.chunk_size),
        "--state-dim", str(args.state_dim),
        "--n-views", str(args.n_views),
        "--image-size", str(args.image_size),
        "--ckpt-every", str(args.ckpt_every),
        "--lr-attn", str(args.lr_attn),
        "--lr-head", str(args.lr_head),
        "--lr-other", str(args.lr_other),
        "--out-dir", str(out_dir),
    ]
    log_path = out_dir / "train.log"
    print(f"\n$ {' '.join(cmd)}\n  → {log_path}", flush=True)
    with log_path.open("w") as logf:
        return subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO).returncode


def _run_eval(out_dir: Path, ckpt_step: int, scene: str, args) -> int:
    cmd = [
        "uv", "run", "python", "-m", "mamba3_attn.eval.phase4_evaluator",
        "--eval-scenes", f"eth3d:{scene}",
        "--image-size", str(args.image_size),
        "--state-dim", str(args.state_dim),
        "--max-images", str(args.max_images),
        "--ckpt", str(out_dir / f"ckpt_{ckpt_step}.pt"),
    ]
    log_path = out_dir / f"eval_ckpt_{ckpt_step}_eth3d_{scene}.log"
    print(f"\n$ {' '.join(cmd)}\n  → {log_path}", flush=True)
    with log_path.open("w") as logf:
        return subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO).returncode


_MEAN_RE = re.compile(
    r"^MEAN\s+(?P<auc30>[\d.\-naN]+)\s+(?P<auc15>[\d.\-naN]+)\s+"
    r"(?P<fp>[\d.\-naN]+)\s+(?P<fu>[\d.\-naN]+)"
)


def _parse_eval_log(log_path: Path) -> dict[str, float] | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text().splitlines():
        m = _MEAN_RE.match(line.strip())
        if m:
            def _f(s: str) -> float:
                try:
                    return float(s)
                except ValueError:
                    return float("nan")
            return {k: _f(m.group(k)) for k in ("auc30", "auc15", "fp", "fu")}
    return None


def _write_summary(out_dir: Path, ckpt_steps: list[int], test_scenes: list[str]) -> None:
    rows: list[str] = ["# §15.59.8 multi-scene supervised eval", ""]
    rows.append(f"Test scenes: {', '.join(test_scenes)} (random split)")
    rows.append("")
    rows.append("| ckpt | scene | AUC@30° | AUC@15° | F_posed | F_unp |")
    rows.append("|---|---|---|---|---|---|")
    aggregates: dict[int, list[dict[str, float]]] = {}
    for s in ckpt_steps:
        aggregates[s] = []
        for sc in test_scenes:
            log = out_dir / f"eval_ckpt_{s}_eth3d_{sc}.log"
            metrics = _parse_eval_log(log)
            if metrics is None:
                rows.append(f"| {s} | {sc} | (no MEAN) | — | — | — |")
                continue
            rows.append(
                f"| {s} | {sc} | {metrics['auc30']:.4f} | {metrics['auc15']:.4f} | "
                f"{metrics['fp']:.4f} | {metrics['fu']:.4f} |"
            )
            aggregates[s].append(metrics)
    rows += ["", "## Per-ckpt mean across test scenes", ""]
    rows.append("| ckpt | AUC@30° | AUC@15° | F_posed | F_unp |")
    rows.append("|---|---|---|---|---|")
    for s in ckpt_steps:
        agg = aggregates[s]
        if not agg:
            rows.append(f"| {s} | — | — | — | — |")
            continue
        def _mean(key: str) -> float:
            vals = [m[key] for m in agg if m[key] == m[key]]  # drop NaN
            return sum(vals) / len(vals) if vals else float("nan")
        rows.append(
            f"| {s} | {_mean('auc30'):.4f} | {_mean('auc15'):.4f} | "
            f"{_mean('fp'):.4f} | {_mean('fu'):.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(rows) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--split-seed", type=int, default=42)
    # Training recipe (PLAN §15.59.8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--decay-steps", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--lr-attn", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--lr-other", type=float, default=1e-5)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    scenes = _discover_eth3d_scenes_with_depth(args.data_root)
    if len(scenes) < 2:
        raise RuntimeError(f"need ≥2 ETH3D scenes with GT depth, found {len(scenes)}")
    train_scenes, test_scenes = _split_scenes(scenes, args.train_frac, args.split_seed)
    if not test_scenes:
        raise RuntimeError("split produced 0 test scenes; raise train_frac or add scenes")

    split = {
        "dataset": "eth3d",
        "all_scenes": scenes,
        "train_frac": args.train_frac,
        "seed": args.split_seed,
        "train": train_scenes,
        "test": test_scenes,
    }
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    print(f"[multi-scene] {len(scenes)} scenes total, "
          f"{len(train_scenes)} train / {len(test_scenes)} test "
          f"(seed={args.split_seed}, frac={args.train_frac})")
    print(f"[multi-scene] train: {train_scenes}")
    print(f"[multi-scene] test:  {test_scenes}")

    if not args.skip_train:
        rc = _run_training(args.out, train_scenes, args)
        if rc != 0:
            print(f"[multi-scene] TRAIN FAILED (rc={rc})", flush=True)
            return

    ckpt_steps = [int(p.stem.split("_")[1]) for p in sorted(args.out.glob("ckpt_*.pt"))
                  if p.stem.split("_")[1].isdigit()]
    if not ckpt_steps:
        print(f"[multi-scene] no ckpts under {args.out}, nothing to eval", flush=True)
        return
    for s in sorted(ckpt_steps):
        for sc in test_scenes:
            log = args.out / f"eval_ckpt_{s}_eth3d_{sc}.log"
            if log.exists() and "MEAN" in log.read_text():
                print(f"[multi-scene] {log.name} already MEAN-complete, skipping", flush=True)
                continue
            _run_eval(args.out, s, sc, args)

    _write_summary(args.out, sorted(ckpt_steps), test_scenes)
    print(f"\n[multi-scene] DONE. Summary at {args.out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
