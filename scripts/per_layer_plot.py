"""Plot per-layer scene-overfit results.

Reads `<out>/layer_KK/{train.log, eval_ckpt_*.log}` produced by
`scripts/per_layer_overfit.py` and writes:

  - `<out>/train_loss_curves.png`  — total loss vs step, one line per layer.
  - `<out>/eval_metric_curves.png` — 4 subplots (AUC30, AUC15, F_posed,
                                      F_unposed), one line per layer, x-axis
                                      = ckpt step.
  - `<out>/per_layer_summary.md`   — table of final-ckpt metrics per layer.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_TRAIN_LINE = re.compile(
    r"^\[3-1\] step\s+(\d+)/\d+\s+loss=([-\d.eE+]+)\s+L_D=([-\d.eE+]+)\s+L_M=([-\d.eE+]+)"
)
_MEAN_LINE = re.compile(
    r"^MEAN\s+([-\d.eE+nan]+)\s+([-\d.eE+nan]+)\s+([-\d.eE+nan]+)\s+([-\d.eE+nan]+)\s*$"
)


def _parse_train_log(path: Path) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        m = _TRAIN_LINE.match(line)
        if not m:
            continue
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


def _parse_eval_log(path: Path) -> tuple[float, float, float, float] | None:
    if not path.exists():
        return None
    for line in path.read_text(errors="ignore").splitlines():
        m = _MEAN_LINE.match(line)
        if m:
            def _f(s: str) -> float:
                try:
                    return float(s)
                except ValueError:
                    return float("nan")
            return tuple(_f(m.group(i)) for i in range(1, 5))  # type: ignore[return-value]
    return None


def _gather(out_dir: Path) -> dict[int, dict]:
    layer_dirs = sorted(out_dir.glob("layer_*"))
    out: dict[int, dict] = {}
    for ld in layer_dirs:
        if not ld.is_dir():
            continue
        try:
            k = int(ld.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        train = _parse_train_log(ld / "train.log")
        evals: dict[int, tuple[float, float, float, float]] = {}
        for log in sorted(ld.glob("eval_ckpt_*.log")):
            try:
                step = int(log.stem.split("_")[-1])
            except ValueError:
                continue
            m = _parse_eval_log(log)
            if m is not None:
                evals[step] = m
        out[k] = {"train": train, "evals": evals}
    return out


def _plot_train(data: dict[int, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.viridis
    n = max(len(data), 1)
    for i, (k, d) in enumerate(sorted(data.items())):
        rows = d["train"]
        if not rows:
            continue
        steps = [r[0] for r in rows]
        losses = [r[1] for r in rows]
        color = cmap(i / max(n - 1, 1))
        ax.plot(steps, losses, label=f"layer {k:02d}", color=color, linewidth=1.2, alpha=0.85)
    ax.set_xlabel("training step")
    ax.set_ylabel("total loss")
    ax.set_title("Per-layer training loss — only that layer's mamba3 attention is trainable")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_eval(data: dict[int, dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_names = ["AUC@30°", "AUC@15°", "F_posed", "F_unposed"]
    cmap = plt.cm.viridis
    n = max(len(data), 1)
    for mi, ax in enumerate(axes.flat):
        for i, (k, d) in enumerate(sorted(data.items())):
            evals = d["evals"]
            if not evals:
                continue
            steps = sorted(evals.keys())
            ys = [evals[s][mi] for s in steps]
            color = cmap(i / max(n - 1, 1))
            ax.plot(steps, ys, "o-", label=f"layer {k:02d}", color=color, linewidth=1.2,
                    markersize=4, alpha=0.85)
        ax.set_xlabel("ckpt step")
        ax.set_ylabel(metric_names[mi])
        ax.set_title(f"{metric_names[mi]} vs ckpt step (per swapped layer)")
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.legend(fontsize=6, ncol=2, loc="best")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


_V2_CEILING = (0.8015, 0.6356, 0.0071, 0.0645)  # §15.59.4 V2 (terrains)


def _write_summary(data: dict[int, dict], path: Path) -> None:
    a30_v2, a15_v2, fp_v2, fu_v2 = _V2_CEILING
    lines = [
        "# Per-layer scene-overfit summary (PLAN §15.59.6)\n",
        "\nEach row: only that layer's attention is mamba3; the other 15 layers are the original\n"
        "DA3-SMALL transformer. Recipe: scene-overfit on eth3d/terrains, super=3 sub=1, 200 steps,\n"
        f"ckpt-every=50, split-seed=42, train-frac=0.75. V2 ceiling (§15.59.4): "
        f"AUC@30°={a30_v2:.4f} / AUC@15°={a15_v2:.4f} / F_posed={fp_v2:.4f} / F_unposed={fu_v2:.4f}.\n",
        "\n## Final ckpt (step=200)\n",
        "\n| Layer | AUC@30° | AUC@15° | F_posed | F_unposed |\n",
        "|---|---|---|---|---|\n",
    ]
    for k, d in sorted(data.items()):
        evals = d["evals"]
        if not evals:
            lines.append(f"| {k:02d} | n/a | n/a | n/a | n/a |\n")
            continue
        last_step = max(evals)
        a30, a15, fp, fu = evals[last_step]
        lines.append(f"| {k:02d} | {a30:.4f} | {a15:.4f} | {fp:.4f} | {fu:.4f} |\n")

    lines.append(
        "\n## Peak AUC@30° across {50, 100, 150, 200}\n"
        "\nReports the ckpt and metrics at which AUC@30° is maximal — many layers peak\n"
        "early (50 or 100) and degrade with more steps, so the final-ckpt row understates\n"
        "the per-layer ceiling.\n"
        "\n| Layer | Best ckpt | AUC@30° | AUC@15° | F_posed | F_unposed |\n"
        "|---|---|---|---|---|---|\n"
    )
    for k, d in sorted(data.items()):
        evals = d["evals"]
        if not evals:
            lines.append(f"| {k:02d} | — | n/a | n/a | n/a | n/a |\n")
            continue
        best_step = max(evals, key=lambda s: evals[s][0])
        a30, a15, fp, fu = evals[best_step]
        lines.append(
            f"| {k:02d} | {best_step} | {a30:.4f} | {a15:.4f} | {fp:.4f} | {fu:.4f} |\n"
        )

    lines.append(
        "\n## Footnote — layers 12-15 (cam_enc.trunk) are unreachable in this recipe\n"
        "\nLayers 12-15 swap mamba3 attention into `cam_enc.trunk`, which is the input\n"
        "camera-conditioning trunk and is only active when extrinsics are passed at the\n"
        "DA3 forward (`src/mamba3_attn/patch.py:11-13`). Scene-overfit training feeds\n"
        "images only — no extrinsics — so with a layer-12-15 swap, no trainable parameter\n"
        "is on the loss path and autograd raises `RuntimeError: element 0 of tensors does\n"
        "not require grad and does not have a grad_fn` before any ckpt is saved. Fixing\n"
        "this would require a recipe that injects extrinsics, which deviates from the\n"
        "DA3 paper setup (ruled out per `feedback_stay_close_to_da3_paper.md`).\n"
    )
    path.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="Same dir passed to per_layer_overfit.py.")
    args = ap.parse_args()

    data = _gather(args.out)
    if not data:
        raise SystemExit(f"no layer_* dirs found under {args.out}")

    train_path = args.out / "train_loss_curves.png"
    eval_path = args.out / "eval_metric_curves.png"
    summary_path = args.out / "per_layer_summary.md"

    _plot_train(data, train_path)
    _plot_eval(data, eval_path)
    _write_summary(data, summary_path)

    print(f"wrote {train_path}")
    print(f"wrote {eval_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
