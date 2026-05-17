"""Build a comparison table between this tracker and published baselines.

Reads:
  - configs/tapvid3d_baselines.yaml — published 3D-AJ per subset for the
    leaderboard methods.
  - outputs/eval_tracker/<run>/summary.md  (or metric_results/*.json) — this
    run's per-subset roll-up.

Writes:
  - outputs/eval_tracker/<run>/comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _load_ours(eval_dir: Path) -> dict[str, dict[str, float]]:
    """Return {subset: {average_jaccard, average_pts_within_thresh, occlusion_accuracy}}."""
    out: dict[str, dict[str, float]] = {}
    for json_path in (eval_dir / "metric_results").glob("*.json"):
        sub = json_path.stem
        per_clip = json.loads(json_path.read_text())
        if not per_clip:
            out[sub] = {}
            continue
        keys = ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy")
        agg = {}
        for k in keys:
            vals = [c[k] for c in per_clip if isinstance(c.get(k), (int, float))]
            agg[k] = sum(vals) / len(vals) if vals else float("nan")
        out[sub] = agg
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="outputs/eval_tracker/<run> containing metric_results/*.json")
    ap.add_argument("--baselines", type=Path,
                    default=Path("configs/tapvid3d_baselines.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.baselines.read_text())
    baselines = cfg["baselines"]
    ours = _load_ours(args.eval_dir)

    rows = [
        "# TAPVid-3D comparison\n",
        "## 3D-AJ per subset (higher is better)\n",
        "| method | aria | drivetrack | pstudio | mean |",
        "|---|---|---|---|---|",
    ]
    for b in baselines:
        vals = [b.get(s, float("nan")) for s in ("aria", "drivetrack", "pstudio")]
        finite = [v for v in vals if isinstance(v, (int, float))]
        mean = sum(finite) / len(finite) if finite else float("nan")
        rows.append(
            f"| {b['name']} | {b.get('aria', float('nan')):.3f} | "
            f"{b.get('drivetrack', float('nan')):.3f} | "
            f"{b.get('pstudio', float('nan')):.3f} | {mean:.3f} |"
        )

    # Ours
    aria = ours.get("adt", {}).get("average_jaccard", float("nan"))
    drv = ours.get("drivetrack", {}).get("average_jaccard", float("nan"))
    pst = ours.get("pstudio", {}).get("average_jaccard", float("nan"))
    finite_ours = [v for v in (aria, drv, pst) if isinstance(v, (int, float)) and v == v]
    mean_ours = sum(finite_ours) / len(finite_ours) if finite_ours else float("nan")
    rows.append(
        f"| **Mamba-3 tracker (this run)** | **{aria:.3f}** | **{drv:.3f}** | "
        f"**{pst:.3f}** | **{mean_ours:.3f}** |"
    )

    rows.append("\n## Per-subset detail of this run\n")
    rows.append("| subset | 3D-AJ | APD3D | OA |")
    rows.append("|---|---|---|---|")
    for sub in ("adt", "drivetrack", "pstudio"):
        m = ours.get(sub, {})
        rows.append(
            f"| {sub} | {m.get('average_jaccard', float('nan')):.4f} | "
            f"{m.get('average_pts_within_thresh', float('nan')):.4f} | "
            f"{m.get('occlusion_accuracy', float('nan')):.4f} |"
        )

    rows.append("")
    rows.append("Baseline numbers from configs/tapvid3d_baselines.yaml "
                "(TAPVid-3D paper + each method's released numbers).")

    out = args.eval_dir / "comparison.md"
    out.write_text("\n".join(rows) + "\n")
    print(f"[compare] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
