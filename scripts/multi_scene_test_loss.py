"""Retroactive train/test loss diagnostic for multi-scene runs (PLAN §15.59.8).

Reads `<run>/split.json` and for each `<run>/ckpt_<step>.pt`, computes the
DA3 paper loss (same form used in train_super) on:

  - the **train scenes** (random n_views=4 batches × n_batches, no augmentation)
  - the **test scenes** (random n_views=4 batches × n_batches, no augmentation)

Outputs:
  - `<run>/test_loss.json`           — {step: {train: {total, L_D, ...}, test: {...}}}
  - `<run>/test_loss_summary.md`     — markdown table per step
  - `<run>/test_loss_curves.png`     — train vs test total loss over steps

Usage:
  uv run python scripts/multi_scene_test_loss.py \\
    --run outputs/runs/multi_scene_distill_eth3d \\
    --n-batches 6
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mamba3_attn.eval.phase4_evaluator import build_patched_api
from mamba3_attn.train.da3_loss import DA3LossWeights, da3_paper_loss
from mamba3_attn.train.multi_view import _slice_batch, load_full_scene_cache
from mamba3_attn.train.train_super import _student_forward, build_target


CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")


def _ckpt_step(path: Path) -> int | None:
    m = CKPT_RE.search(path.name)
    return int(m.group(1)) if m else None


@torch.no_grad()
def _scene_loss(api, caches: dict[str, object], n_views: int, n_batches: int,
                image_size: int, device, weights: DA3LossWeights, seed: int,
                amp_dtype: torch.dtype) -> dict[str, float]:
    """Average DA3 paper loss over `n_batches` random subsets per scene, then
    average across scenes. Returns total + per-component means."""
    rng = random.Random(seed)
    keys = ("L_D", "L_M", "L_grad", "L_P", "L_C")
    totals: list[float] = []
    comps: dict[str, list[float]] = {k: [] for k in keys}
    for scene, cache in caches.items():
        n_total = cache.images.shape[0]
        if n_total < n_views:
            continue
        all_idx = list(range(n_total))
        for _ in range(n_batches):
            picked = rng.sample(all_idx, n_views)
            batch = _slice_batch(cache, picked)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                target, gt_kwargs, _ = build_target(batch, None, image_size, device)
                s_out = _student_forward(api, batch.images.to(device))
                loss_out = da3_paper_loss(student=s_out, target=target,
                                          weights=weights, **gt_kwargs)
            totals.append(float(loss_out.total))
            comps["L_D"].append(float(loss_out.l_depth))
            comps["L_M"].append(float(loss_out.l_ray))
            comps["L_grad"].append(float(loss_out.l_grad))
            comps["L_P"].append(float(loss_out.l_point))
            comps["L_C"].append(float(loss_out.l_cam))
    def _mean(xs: list[float]) -> float:
        return sum(xs) / max(len(xs), 1)
    return {"total": _mean(totals), **{k: _mean(v) for k, v in comps.items()}}


def _load_caches(scenes: list[str], data_root: Path, image_size: int,
                 candidate_views: int) -> dict[str, object]:
    out = {}
    for sc in scenes:
        out[sc] = load_full_scene_cache(
            "eth3d", sc, data_root,
            image_size=image_size, candidate_views=candidate_views,
        )
    return out


def _plot(results: dict, out_path: Path) -> None:
    steps = sorted(int(s) for s in results)
    train_total = [results[str(s)]["train"]["total"] for s in steps]
    test_total = [results[str(s)]["test"]["total"] for s in steps]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_total, "o-", label="train scenes", color="C0", linewidth=1.5)
    ax.plot(steps, test_total, "s-", label="test scenes", color="C1", linewidth=1.5)
    ax.set_xlabel("ckpt step")
    ax.set_ylabel("DA3 paper loss (mean over scenes × batches)")
    ax.set_title("§15.59.8 train vs test loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_summary(results: dict, out_path: Path) -> None:
    lines = ["# §15.59.8 train vs test loss\n",
             "\nDA3 paper loss (same form as training) averaged over scenes × random "
             "n_views=4 batches, no augmentation. Negative loss indicates Kendall-Gal "
             "log-scale memorisation.\n",
             "\n## Total loss per ckpt\n",
             "\n| ckpt | train_loss | test_loss | gap (test−train) |\n",
             "|---|---|---|---|\n"]
    for s in sorted(int(x) for x in results):
        tr = results[str(s)]["train"]["total"]
        te = results[str(s)]["test"]["total"]
        lines.append(f"| {s} | {tr:.4f} | {te:.4f} | {te - tr:+.4f} |\n")
    lines.append("\n## Per-component (train/test)\n")
    lines.append("\n| ckpt | L_D tr/te | L_M tr/te | L_P tr/te | L_C tr/te |\n")
    lines.append("|---|---|---|---|---|\n")
    for s in sorted(int(x) for x in results):
        tr = results[str(s)]["train"]
        te = results[str(s)]["test"]
        lines.append(
            f"| {s} | {tr['L_D']:.3f}/{te['L_D']:.3f} | "
            f"{tr['L_M']:.3f}/{te['L_M']:.3f} | "
            f"{tr['L_P']:.3f}/{te['L_P']:.3f} | "
            f"{tr['L_C']:.3f}/{te['L_C']:.3f} |\n"
        )
    out_path.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="Run directory containing split.json and ckpt_*.pt")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--n-batches", type=int, default=6)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--candidate-views", type=int, default=256)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ckpts", type=str, default=None,
                    help="Comma-separated steps to evaluate (default: all in run dir).")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip ckpt evaluation; re-render summary + plot from existing JSON.")
    args = ap.parse_args()

    split = json.loads((args.run / "split.json").read_text())
    train_scenes = list(split["train"])
    test_scenes = list(split["test"])
    print(f"[test-loss] train: {train_scenes}", flush=True)
    print(f"[test-loss] test:  {test_scenes}", flush=True)

    out_json = args.run / "test_loss.json"
    if args.aggregate_only:
        results = json.loads(out_json.read_text())
    else:
        ckpts = sorted([p for p in args.run.glob("ckpt_*.pt") if _ckpt_step(p) is not None],
                       key=lambda p: _ckpt_step(p))
        if args.ckpts:
            wanted = {int(s) for s in args.ckpts.split(",")}
            ckpts = [c for c in ckpts if _ckpt_step(c) in wanted]
        if not ckpts:
            raise RuntimeError(f"no ckpts under {args.run}")
        print(f"[test-loss] {len(ckpts)} ckpts to evaluate", flush=True)

        device = torch.device(args.device)
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        print("[test-loss] loading train-scene caches", flush=True)
        train_caches = _load_caches(train_scenes, args.data_root, args.image_size,
                                    args.candidate_views)
        print("[test-loss] loading test-scene caches", flush=True)
        test_caches = _load_caches(test_scenes, args.data_root, args.image_size,
                                   args.candidate_views)

        weights = DA3LossWeights()  # default = matches train_super.py default

        results: dict = {}
        for ckpt_path in ckpts:
            step = _ckpt_step(ckpt_path)
            print(f"\n[test-loss] ckpt_{step}: building patched API...", flush=True)
            api = build_patched_api(str(ckpt_path), device=str(device),
                                    state_dim=args.state_dim, patched=True)
            api.model.eval()
            print(f"[test-loss] ckpt_{step}: train-scene loss...", flush=True)
            train_loss = _scene_loss(api, train_caches, args.n_views, args.n_batches,
                                     args.image_size, device, weights, args.seed, amp_dtype)
            print(f"[test-loss] ckpt_{step}: test-scene loss...", flush=True)
            test_loss = _scene_loss(api, test_caches, args.n_views, args.n_batches,
                                    args.image_size, device, weights, args.seed + 1, amp_dtype)
            print(f"[test-loss] ckpt_{step}: train={train_loss['total']:.4f}  "
                  f"test={test_loss['total']:.4f}  gap={test_loss['total']-train_loss['total']:+.4f}",
                  flush=True)
            results[str(step)] = {"train": train_loss, "test": test_loss}
            del api
            if device.type == "cuda":
                torch.cuda.empty_cache()

        out_json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out_json}", flush=True)

    _write_summary(results, args.run / "test_loss_summary.md")
    _plot(results, args.run / "test_loss_curves.png")
    print(f"wrote {args.run / 'test_loss_summary.md'}", flush=True)
    print(f"wrote {args.run / 'test_loss_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
