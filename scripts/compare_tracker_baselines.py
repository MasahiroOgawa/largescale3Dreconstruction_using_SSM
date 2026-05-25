"""Compare one or more of our tracker runs against published TAPVid-3D baselines.

Reads:
  - configs/tapvid3d_baselines.yaml — published 3D-AJ per subset (median scaling).
  - For each --eval-dir <path>:
      <path>/metric_results/{pstudio,drivetrack,adt}.json   per-clip results
      (the directory layout written by scripts/eval_mamba3_tracker.py).

Writes (to --out-dir):
  - comparison.md   markdown table: baselines + our runs vs subset 3D-AJ + mean
  - comparison.png  grouped-bar chart of 3D-AJ across all methods × subsets

Usage:
  uv run python scripts/compare_tracker_baselines.py \\
      --eval-dir outputs/eval_tracker/v14 \\
      --eval-dir outputs/eval_tracker/v17 \\
      --out-dir  outputs/eval_tracker/comparison_v14_v17
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


SUBSETS = ("pstudio", "drivetrack", "aria")
SUBSET_ON_DISK = {"pstudio": "pstudio", "drivetrack": "drivetrack", "aria": "adt"}


def _load_run(eval_dir: Path) -> dict[str, float | None]:
    """Return {subset: 3D-AJ as fraction in [0, 1]} for one of our runs."""
    out: dict[str, float | None] = {s: None for s in SUBSETS}
    for sub in SUBSETS:
        json_path = eval_dir / "metric_results" / f"{SUBSET_ON_DISK[sub]}.json"
        if not json_path.exists():
            continue
        per_clip = json.loads(json_path.read_text())
        ajs = [c["average_jaccard"] for c in per_clip
               if isinstance(c.get("average_jaccard"), (int, float))]
        out[sub] = sum(ajs) / len(ajs) if ajs else None
    return out


def _mean(vals: list[float | None]) -> float | None:
    finite = [v for v in vals if isinstance(v, (int, float))]
    return sum(finite) / len(finite) if finite else None


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}"


def _write_markdown(
    baselines: list[dict], ours: list[tuple[str, dict[str, float | None]]],
    out_path: Path,
) -> None:
    lines = [
        "# TAPVid-3D 3D-AJ comparison (median scaling, %)",
        "",
        "Per-clip mean 3D Average Jaccard × 100. Subsets are the TAPVid-3D",
        "test splits. Baselines from configs/tapvid3d_baselines.yaml.",
        "",
        "| method | pstudio | drivetrack | aria (adt) | mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in baselines:
        vals = [b.get(s) for s in SUBSETS]
        lines.append(
            f"| {b['name']} | {_fmt(vals[0])} | {_fmt(vals[1])} | "
            f"{_fmt(vals[2])} | {_fmt(_mean(vals))} |"
        )
    for label, m in ours:
        vals = [m.get(s) for s in SUBSETS]
        lines.append(
            f"| **{label}** | **{_fmt(vals[0])}** | **{_fmt(vals[1])}** | "
            f"**{_fmt(vals[2])}** | **{_fmt(_mean(vals))}** |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[compare] wrote {out_path}")


def _plot(
    baselines: list[dict], ours: list[tuple[str, dict[str, float | None]]],
    out_path: Path,
) -> None:
    methods = [(b["name"], {s: b.get(s) for s in SUBSETS}, "baseline") for b in baselines]
    methods.extend((label, m, "ours") for label, m in ours)

    x_labels = ["pstudio", "drivetrack", "aria", "mean"]
    n_methods = len(methods)
    n_groups = len(x_labels)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * n_methods + 4), 5.5))
    x = np.arange(n_groups)
    for i, (name, m, kind) in enumerate(methods):
        vals = [m.get(s) for s in SUBSETS] + [_mean([m.get(s) for s in SUBSETS])]
        ys = [0 if v is None else v * 100 for v in vals]
        color = "#d62728" if kind == "ours" else None
        alpha = 1.0 if kind == "ours" else 0.85
        bars = ax.bar(
            x + i * width - 0.4 + width / 2, ys, width, label=name,
            color=color, alpha=alpha,
            edgecolor="black" if kind == "ours" else "none", linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{v * 100:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("3D-AJ (%)")
    ax.set_title("TAPVid-3D 3D-AJ — published baselines vs our runs")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-dir", type=Path, action="append", default=[],
        help="Path to outputs/eval_tracker/<run>/. May be repeated.",
    )
    ap.add_argument(
        "--baselines", type=Path,
        default=Path("configs/tapvid3d_baselines.yaml"),
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--label", action="append", default=None,
        help="Display label per --eval-dir (in order). "
             "Default: parent directory name.",
    )
    args = ap.parse_args()

    if not args.eval_dir:
        ap.error("at least one --eval-dir is required")
    if args.label is not None and len(args.label) != len(args.eval_dir):
        ap.error("--label count must match --eval-dir count")

    baselines = yaml.safe_load(args.baselines.read_text())["baselines"]
    ours: list[tuple[str, dict[str, float | None]]] = []
    for i, ed in enumerate(args.eval_dir):
        label = args.label[i] if args.label else ed.name
        ours.append((label, _load_run(ed)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_markdown(baselines, ours, args.out_dir / "comparison.md")
    _plot(baselines, ours, args.out_dir / "comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
