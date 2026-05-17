"""Headline comparison figure — VSSD-DA3 vs DA3-SMALL on accuracy + efficiency.

Reads:
  - `<eval_root>/accuracy/<model>/<dataset>_metrics.json`  (one per dataset, per model)
       from the DA3 official evaluator output (`workspace.work_dir`)
  - `<eval_root>/efficiency/efficiency.json`
       from `scripts/bench_efficiency_patched.py`

Writes:
  - `<eval_root>/comparison.png` + `.pdf` — 2-panel figure
        top:    F-score (or AUC@30°) per dataset, clustered bars
        bottom: peak memory + latency vs cross-view T (long-context regime)
  - `<eval_root>/summary.md` — head-to-head table + gate verdict
        (gate: VSSD must reach DA3-SMALL accuracy AND beat its efficiency)

Usage:
    uv run python scripts/plot_da3_vs_vssd.py \\
        --eval-root outputs/eval_vssd_full \\
        --models depth-anything_DA3-SMALL outputs_runs_vssd_da3_stageB_ckpt_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# DA3 evaluator typically writes per-dataset summary JSONs containing
# precision / recall / F-score (recon) and AUC{30,15,5,3} (pose). The exact
# layout depends on the evaluator version; this loader is lenient about
# either flat {"f_score": ...} or nested {"recon_posed": {"f_score": ...}}.
RECON_KEYS = ("f_score", "fscore", "F@5cm", "F_score")
POSE_KEYS = ("auc_30", "auc30", "AUC@30")


def _flatten(obj, depth: int = 0):
    if depth > 4 or not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            yield from _flatten(v, depth + 1)
        else:
            yield k, v


def _pick(d: dict, candidates: tuple[str, ...]) -> float | None:
    flat = dict(_flatten(d))
    for k in candidates:
        if k in flat:
            try:
                return float(flat[k])
            except (TypeError, ValueError):
                pass
    return None


def load_accuracy(model_dir: Path, datasets: list[str]) -> dict[str, dict[str, float]]:
    """Return {dataset: {"f": F-score, "auc30": AUC@30°}} for the given model dir.

    Reads both *_recon_posed.json (for F-score) and *_pose.json (for AUC) per
    dataset, so the headline figure shows accuracy + pose side by side.
    """
    out: dict[str, dict[str, float]] = {}
    for ds in datasets:
        candidates = list(model_dir.rglob(f"*{ds}*.json"))
        if not candidates:
            print(f"[plot] no metrics json for {ds} under {model_dir}")
            continue
        f = a = None
        for path in candidates:
            try:
                data = json.loads(path.read_text())
            except Exception as e:
                print(f"[plot] failed to read {path}: {e}")
                continue
            name = path.name.lower()
            if f is None and ("recon" in name or _pick(data, RECON_KEYS) is not None):
                f = _pick(data, RECON_KEYS)
            if a is None and ("pose" in name or _pick(data, POSE_KEYS) is not None):
                a = _pick(data, POSE_KEYS)
        out[ds] = {"f": f if f is not None else float("nan"),
                   "auc30": a if a is not None else float("nan")}
        print(f"[plot] {model_dir.name}/{ds}: F={out[ds]['f']:.4f}, AUC@30°={out[ds]['auc30']:.4f}")
    return out


def load_efficiency(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def plot(
    eval_root: Path, acc_by_model: dict[str, dict[str, dict[str, float]]],
    eff_rows: list[dict], datasets: list[str], variant_a: str, variant_b: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))

    # ── Top: accuracy bars ────────────────────────────────────────────────
    ax = axes[0]
    n_ds = len(datasets)
    bar_w = 0.35
    x = np.arange(n_ds)
    f_a = [acc_by_model.get(variant_a, {}).get(d, {}).get("f", float("nan")) for d in datasets]
    f_b = [acc_by_model.get(variant_b, {}).get(d, {}).get("f", float("nan")) for d in datasets]
    ax.bar(x - bar_w / 2, f_a, width=bar_w, label=variant_a, color="tab:blue")
    ax.bar(x + bar_w / 2, f_b, width=bar_w, label=variant_b, color="tab:red")
    for i, (va, vb) in enumerate(zip(f_a, f_b)):
        if not np.isnan(va):
            ax.text(i - bar_w / 2, va, f"{va:.3f}", ha="center", va="bottom", fontsize=8)
        if not np.isnan(vb):
            ax.text(i + bar_w / 2, vb, f"{vb:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("F-score @ 5 cm (recon_posed)")
    ax.set_title("Accuracy on DA3 benchmark")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # ── Bottom: efficiency curves ─────────────────────────────────────────
    ax = axes[1]
    style = {
        "DA3-SMALL (transformer)": ("o-", "tab:blue"),
        "DA3-SMALL +VSSD (NC-SSD)": ("D-", "tab:red"),
        "DA3-SMALL +Mamba-3 (PyTorch)": ("s--", "tab:orange"),
        "DA3-SMALL +Mamba-3 (Triton kernel)": ("^-", "tab:green"),
    }
    by_label: dict[str, list[tuple[int, float, float, bool]]] = {}
    for r in eff_rows:
        T = r["cross_view_T"] if "cross_view_T" in r else r["n_views"] * (r["img_size"] // 14) ** 2
        by_label.setdefault(r["label"], []).append(
            (T, r.get("latency_ms", float("nan")), r.get("peak_MiB", float("nan")), r.get("OOM", False))
        )
    ax2 = ax.twinx()
    for label, rows in by_label.items():
        if label not in style:
            continue
        marker, color = style[label]
        rows.sort()
        Ts = [t for t, _, _, oom in rows if not oom]
        lats = [l for t, l, _, oom in rows if not oom]
        mems = [m for t, _, m, oom in rows if not oom]
        if Ts:
            ax.plot(Ts, mems, marker, color=color, label=label + " (mem)")
            ax2.plot(Ts, lats, marker.replace("-", ":"), color=color, alpha=0.6,
                     label=label + " (lat)")
        # Mark OOM points at peak T
        ooms = [t for t, _, _, oom in rows if oom]
        if ooms:
            ax.scatter(ooms, [0.0] * len(ooms), marker="x", color=color, s=80,
                       label=label + " OOM")
    ax.set_xlabel("cross-view sequence length T (tokens)")
    ax.set_ylabel("peak GPU memory (MiB)")
    ax2.set_ylabel("forward latency (ms, dotted)")
    ax.set_title("Efficiency — solid = peak mem, dotted = latency")
    ax.grid(True, alpha=0.3)
    handles, labels_ = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels_ + l2, fontsize=7, loc="upper left")

    plt.tight_layout()
    png = eval_root / "comparison.png"
    pdf = eval_root / "comparison.pdf"
    fig.savefig(png, dpi=120)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[plot] wrote {png} and {pdf}")


def write_summary(
    eval_root: Path, acc_by_model: dict[str, dict[str, dict[str, float]]],
    eff_rows: list[dict], datasets: list[str], variant_a: str, variant_b: str,
) -> None:
    rows: list[str] = ["# VSSD-DA3 vs DA3-SMALL — head-to-head\n"]
    rows.append("## Accuracy (DA3 official benchmark)\n")
    rows.append("| dataset | DA3-SMALL F@5cm | VSSD F@5cm | Δ | DA3-SMALL AUC@30° | VSSD AUC@30° | Δ |")
    rows.append("|---|---|---|---|---|---|---|")
    f_a_mean, f_b_mean = [], []
    a_a_mean, a_b_mean = [], []
    for d in datasets:
        fa = acc_by_model.get(variant_a, {}).get(d, {}).get("f", float("nan"))
        fb = acc_by_model.get(variant_b, {}).get(d, {}).get("f", float("nan"))
        aa = acc_by_model.get(variant_a, {}).get(d, {}).get("auc30", float("nan"))
        ab = acc_by_model.get(variant_b, {}).get(d, {}).get("auc30", float("nan"))
        if not np.isnan(fa): f_a_mean.append(fa)
        if not np.isnan(fb): f_b_mean.append(fb)
        if not np.isnan(aa): a_a_mean.append(aa)
        if not np.isnan(ab): a_b_mean.append(ab)
        rows.append(f"| {d} | {fa:.4f} | {fb:.4f} | {fb - fa:+.4f} | "
                    f"{aa:.4f} | {ab:.4f} | {ab - aa:+.4f} |")
    fa_m = sum(f_a_mean) / len(f_a_mean) if f_a_mean else float("nan")
    fb_m = sum(f_b_mean) / len(f_b_mean) if f_b_mean else float("nan")
    aa_m = sum(a_a_mean) / len(a_a_mean) if a_a_mean else float("nan")
    ab_m = sum(a_b_mean) / len(a_b_mean) if a_b_mean else float("nan")
    rows.append(f"| **mean** | **{fa_m:.4f}** | **{fb_m:.4f}** | **{fb_m - fa_m:+.4f}** | "
                f"**{aa_m:.4f}** | **{ab_m:.4f}** | **{ab_m - aa_m:+.4f}** |")

    rows.append("\n## Efficiency (full DA3 forward, B=1)\n")
    by_key = {(r["img_size"], r["n_views"], r["label"]): r for r in eff_rows}
    sizes = sorted({r["img_size"] for r in eff_rows})
    nvs = sorted({r["n_views"] for r in eff_rows})
    rows.append("| img² | S | T | DA3-SMALL mem/lat | VSSD mem/lat | mem ratio | lat ratio |")
    rows.append("|---|---|---|---|---|---|---|")
    for img in sizes:
        for nv in nvs:
            T = nv * (img // 14) ** 2
            a = by_key.get((img, nv, "DA3-SMALL (transformer)"))
            b = by_key.get((img, nv, "DA3-SMALL +VSSD (NC-SSD)"))
            def cell(r):
                if r is None: return "—"
                if r.get("OOM"): return "**OOM**"
                return f"{r['peak_MiB']:.0f} MiB / {r['latency_ms']:.0f} ms"
            mr = lr = "—"
            if a and b and not a.get("OOM") and not b.get("OOM"):
                mr = f"{b['peak_MiB'] / a['peak_MiB']:.2f}×"
                lr = f"{b['latency_ms'] / a['latency_ms']:.2f}×"
            elif a and a.get("OOM") and b and not b.get("OOM"):
                mr = lr = "attn OOM, VSSD ok"
            rows.append(f"| {img} | {nv} | {T} | {cell(a)} | {cell(b)} | {mr} | {lr} |")

    rows.append("\n## Gate verdict\n")
    rows.append("Gates from `memory/feedback_no_efficiency_only_paper.md`:")
    rows.append(f"- Accuracy parity: VSSD F-score within ε of DA3-SMALL mean? "
                f"({fb_m:.4f} vs {fa_m:.4f} → Δ={fb_m - fa_m:+.4f})")
    rows.append(f"- Efficiency win: VSSD mem/lat < 1× at the largest (img, S)? "
                f"see efficiency table.")
    rows.append(f"- Long-context survival: VSSD ok where DA3 OOMs? "
                f"see OOM column.")
    (eval_root / "summary.md").write_text("\n".join(rows) + "\n")
    print(f"[plot] wrote {eval_root / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--models", nargs=2, required=True,
                    metavar=("BASELINE_DIR", "VSSD_DIR"),
                    help="Two model output dirs under <eval-root>/accuracy/. "
                    "Both should contain per-dataset metrics JSONs from the DA3 evaluator.")
    ap.add_argument("--datasets", nargs="+",
                    default=["eth3d", "7scenes", "scannetpp", "hiroom", "dtu"])
    args = ap.parse_args()

    eval_root = args.eval_root.resolve()
    acc_root = eval_root / "accuracy"
    eff_path = eval_root / "efficiency" / "efficiency.json"

    name_a, name_b = args.models
    acc = {
        name_a: load_accuracy(acc_root / name_a, args.datasets),
        name_b: load_accuracy(acc_root / name_b, args.datasets),
    }
    eff = load_efficiency(eff_path) if eff_path.exists() else []
    if not eff:
        print(f"[plot] WARNING: no efficiency.json at {eff_path}; lower panel will be empty")

    plot(eval_root, acc, eff, args.datasets, name_a, name_b)
    write_summary(eval_root, acc, eff, args.datasets, name_a, name_b)


if __name__ == "__main__":
    main()
