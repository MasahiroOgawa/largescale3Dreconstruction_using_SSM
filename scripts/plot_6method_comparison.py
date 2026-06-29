"""Generate two grouped-bar figures comparing 6 methods on TAPVid-3D minival.

Figure 1: TAPVid-3D normalised median-AJ (leaderboard metric, scale-invariant)
Figure 2: Absolute metric-AJ (fixed-metre thresholds, real 3D accuracy)

Methods:
  1. SEA-RAFT + DA3
  2. SEA-RAFT + MegaSAM
  3. TAPIP3D + DA3
  4. TAPIP3D + MegaSAM (image-only, no GT poses)
  5. Ours (DA3)      = v33 = SEA-RAFT + DA3 + Mamba3SSD
  6. Ours (MegaSAM)  = v34 = SEA-RAFT + MegaSAM + Mamba3SSD

Run after each new eval completes to update the figures.
NaN values are shown as hatched empty bars labelled "N/A".
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("result/figures")
REPO = Path(__file__).parent.parent

SUBSETS = ["drivetrack", "pstudio", "adt"]
SUBSET_LABELS = {"drivetrack": "Drivetrack", "pstudio": "Panoptic Studio", "adt": "ADT"}

METHODS = [
    "BootsTAPIR\n+ZoeDepth",
    "Spatial\nTracker",
    "SEA-RAFT\n+DA3",
    "SEA-RAFT\n+MegaSAM",
    "TAPIP3D\n+DA3",
    "TAPIP3D\n+MegaSAM",
    "Ours\n(DA3)",
    "Ours\n(MegaSAM)",
]

SUBSET_COLORS = {
    "drivetrack": "#4878CF",  # steel blue
    "pstudio": "#E06C2B",  # burnt orange
    "adt": "#3A9E5C",  # forest green
}

HATCH_PENDING = "///"


def _read_summary(path: Path) -> dict[str, dict]:
    """Parse summary.md → {subset: {3d_aj, metric_aj}}. Returns {} if not found."""
    if not path.exists():
        return {}
    txt = path.read_text()
    results = {}
    for line in txt.splitlines():
        m = re.match(
            r"\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
            r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
            line,
        )
        if m:
            subset = m.group(1)
            if subset in SUBSETS:
                results[subset] = {
                    "3d_aj": float(m.group(2)),
                    "metric_aj": float(m.group(5)),
                }
    return results


def build_data_tables() -> tuple[dict, dict]:
    """Return (norm_aj, abs_aj) tables: {method_idx: {subset: float_or_nan}}."""

    NaN = float("nan")

    # ── SEA-RAFT + DA3 (use the _fix run) ─────────────────────────────────────
    searaft_da3 = _read_summary(
        REPO / "result/metric3d_searaft_fix_20260618-1718/summary.md"
    )
    # ── v33 (use the _fix run) ────────────────────────────────────────────────
    v33 = _read_summary(REPO / "result/metric3d_v33_fix_20260618-1718/summary.md")
    # ── SEA-RAFT + MegaSAM ────────────────────────────────────────────────────
    searaft_msam = _read_summary(
        REPO / "result/metric3d_searaft_megasam_minival_20260626-0903/summary.md"
    )
    # ── v34 (may still be running) ─────────────────────────────────────────────
    v34_dirs = sorted(REPO.glob("result/metric3d_v34_megasam_minival_*/summary.md"))
    v34 = _read_summary(v34_dirs[-1]) if v34_dirs else {}

    # ── TAPIP3D + DA3 (official TAPIP3D eval JSONs for normalised; absolute eval for metric-AJ) ──
    tapip3d_da3_norm = {}
    tapip3d_da3_abs = {}
    tapip3d_dir = Path(
        "/home/mas/proj/study/TAPIP3D/outputs/auto_generated"
        "/tapip3d_kubric_24frames_384trajs_2026-06-24_18-55-53"
    )
    for s in SUBSETS:
        j = tapip3d_dir / f"metrics_{s}_da3_minival.json"
        if j.exists():
            d = json.loads(j.read_text())
            tapip3d_da3_norm[s] = d.get(
                f"{s}_da3_minival-metrics/mean/tapvid3d_average_jaccard_best", NaN
            )
    abs_dir = REPO / "result/tapip3d_absolute_eval_20260625-1318/metric_results"
    for s in SUBSETS:
        j = abs_dir / f"{s}.json"
        if j.exists():
            clips = json.loads(j.read_text())
            vals = [c["metric_average_jaccard"] for c in clips]
            tapip3d_da3_abs[s] = float(np.mean(vals)) if vals else NaN

    # ── TAPIP3D + MegaSAM (image-only, no GT poses) ───────────────────────────
    # drivetrack/pstudio DROID-SLAM depth failure → 0.00% / 0.27% normalised
    # ADT: overnight run pending
    tapip3d_msam_norm = {
        "drivetrack": 0.0000,
        "pstudio": 0.0027,
        "adt": NaN,
    }
    tapip3d_msam_abs = {
        "drivetrack": 0.0000,
        "pstudio": 0.0000,
        "adt": NaN,
    }

    # ── Published baselines (normalised AJ from TAPVid-3D paper) ─────────────
    bootstapir_norm = {"drivetrack": 0.051, "pstudio": 0.102, "adt": 0.086}
    bootstapir_abs = {}  # predictions not released; all N/A

    # SpatialTracker: normalised published; absolute drivetrack scored via our pipeline
    spatialtracker_norm = {"drivetrack": 0.058, "pstudio": 0.098, "adt": 0.092}
    spatialtracker_abs = {"drivetrack": 0.069}  # pstudio/adt predictions not released

    def _get(table: dict, subset: str, key: str) -> float:
        return table.get(subset, {}).get(key, NaN)

    # method index → {subset: value}
    norm_aj: dict[int, dict[str, float]] = {}
    abs_aj: dict[int, dict[str, float]] = {}
    for m_idx, (norm_src, abs_src) in enumerate(
        [
            (bootstapir_norm, bootstapir_abs),
            (spatialtracker_norm, spatialtracker_abs),
            (searaft_da3, searaft_da3),
            (searaft_msam, searaft_msam),
            (tapip3d_da3_norm, tapip3d_da3_abs),
            (tapip3d_msam_norm, tapip3d_msam_abs),
            (v33, v33),
            (v34, v34),
        ]
    ):
        norm_aj[m_idx] = {}
        abs_aj[m_idx] = {}
        for s in SUBSETS:
            # Methods with plain {subset: float} sources (not summary.md dicts):
            # 0=BootsTAPIR, 1=SpatialTracker, 4=TAPIP3D+DA3, 5=TAPIP3D+MegaSAM
            if m_idx in (0, 1, 4, 5):
                norm_aj[m_idx][s] = float(norm_src.get(s, NaN))
                abs_aj[m_idx][s] = float(abs_src.get(s, NaN))
            else:  # SEA-RAFT / v33 / v34: sources are summary.md {subset: {key: float}} dicts
                norm_aj[m_idx][s] = _get(norm_src, s, "3d_aj")
                abs_aj[m_idx][s] = _get(abs_src, s, "metric_aj")

    return norm_aj, abs_aj


def _percent(v: float) -> float:
    return v * 100.0


def make_figure(
    data: dict[int, dict[str, float]], title: str, ylabel: str, fname: str
) -> None:
    n_methods = len(METHODS)
    n_subsets = len(SUBSETS)
    bar_w = 0.22
    group_gap = 0.2
    group_w = n_subsets * bar_w + group_gap
    x_centers = np.arange(n_methods) * group_w

    fig, ax = plt.subplots(figsize=(12, 5))

    offsets = np.linspace(-(n_subsets - 1) / 2, (n_subsets - 1) / 2, n_subsets) * bar_w
    legend_handles = []

    for s_idx, subset in enumerate(SUBSETS):
        color = SUBSET_COLORS[subset]
        xs = x_centers + offsets[s_idx]
        heights = [
            _percent(data[m].get(subset, float("nan"))) for m in range(n_methods)
        ]

        for xi, hi in zip(xs, heights):
            if np.isnan(hi):
                # N/A bar
                ax.bar(
                    xi,
                    0.0,
                    width=bar_w,
                    color="none",
                    edgecolor="#888",
                    linewidth=0.8,
                    hatch=HATCH_PENDING,
                )
                ax.text(
                    xi,
                    0.5,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#888",
                    rotation=90,
                )
            else:
                ax.bar(
                    xi,
                    hi,
                    width=bar_w,
                    color=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                )
                label = f"{hi:.2f}" if hi < 1.0 else f"{hi:.1f}"
                if hi < 0.8:  # small bar: rotate label upward to avoid crowding
                    ax.text(
                        xi,
                        hi + 0.15,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color="#444",
                        rotation=90,
                    )
                else:
                    ax.text(
                        xi,
                        hi + 0.3,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=6.0,
                        color="#222",
                    )

        patch = mpatches.Patch(color=color, alpha=0.85, label=SUBSET_LABELS[subset])
        legend_handles.append(patch)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(METHODS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.85)
    ax.set_xlim(x_centers[0] - group_w / 2, x_centers[-1] + group_w / 2)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Vertical separators between method groups
    for xi in (x_centers[:-1] + x_centers[1:]) / 2:
        ax.axvline(xi, color="#ddd", linewidth=0.8, zorder=0)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def print_table(norm_aj, abs_aj):
    header = f"{'Method':<24}" + "".join(f"{s:>12}" for s in SUBSETS)
    print("\n=== Normalised median-AJ (%) ===")
    print(header)
    for m_idx, name in enumerate(METHODS):
        row = name.replace("\n", " ").ljust(24)
        for s in SUBSETS:
            v = _percent(norm_aj[m_idx].get(s, float("nan")))
            row += f"{'N/A':>12}" if np.isnan(v) else f"{v:>11.2f}%"
        print(row)

    print("\n=== Absolute metric-AJ (%) ===")
    print(header)
    for m_idx, name in enumerate(METHODS):
        row = name.replace("\n", " ").ljust(24)
        for s in SUBSETS:
            v = _percent(abs_aj[m_idx].get(s, float("nan")))
            row += f"{'N/A':>12}" if np.isnan(v) else f"{v:>11.2f}%"
        print(row)


def main():
    norm_aj, abs_aj = build_data_tables()
    print_table(norm_aj, abs_aj)

    make_figure(
        norm_aj,
        title="TAPVid-3D Normalised Median-AJ (leaderboard metric, higher = better)",
        ylabel="3D-AJ [%]  (normalised, scale-invariant)",
        fname="fig1_normalized_aj",
    )
    make_figure(
        abs_aj,
        title="Absolute Metric-AJ (fixed-metre thresholds, real 3D accuracy)",
        ylabel="Metric-AJ [%]  (absolute, 1cm–2.56m thresholds)",
        fname="fig2_absolute_aj",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
