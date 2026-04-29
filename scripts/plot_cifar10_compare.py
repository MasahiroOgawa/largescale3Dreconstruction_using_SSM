"""Reusable figure generation for cifar10_compare.py runs.

Reads a `results.json` produced by `scripts/cifar10_compare.py` and writes:

    lr_curves.png              — LR (end of epoch) vs step, one line per variant
    loss_vs_steps.png          — train + test cross-entropy vs step, one line per variant
    efficiency_comparison.png  — params / latency / peak mem / s-per-epoch, bar per variant

Standalone CLI:

    uv run python scripts/plot_cifar10_compare.py outputs/<run>/results.json

Or import `make_all_figures(results_path, out_dir)` from `cifar10_compare.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANT_LABEL = {
    "cnn": "CNN (small ResNet)",
    "vit_attn": "ViT-Tiny + softmax",
    "vit_mamba3": "ViT-Tiny + Mamba-3 SSD",
}
VARIANT_COLOR = {
    "cnn": "#1f77b4",
    "vit_attn": "#ff7f0e",
    "vit_mamba3": "#2ca02c",
}
VARIANT_ORDER = ("cnn", "vit_attn", "vit_mamba3")
CIFAR10_TRAIN_N = 50_000


def _steps_per_epoch(cfg: dict) -> int:
    return int(cfg.get("steps_per_epoch") or CIFAR10_TRAIN_N // cfg["batch_size"])


def _recipe_subtitle(cfg: dict) -> str:
    sched = cfg.get("lr_schedule", "cosine")
    grad_clip = cfg.get("grad_clip", 0.0)
    return (
        f"AdamW peak={cfg['lr']:.0e}, {cfg.get('warmup_epochs', 5)}-ep warmup → {sched}, "
        f"grad_clip={'off' if grad_clip <= 0 else grad_clip}, "
        f"{cfg['epochs']} ep × {_steps_per_epoch(cfg)} steps"
    )


def _present_variants(variants: dict) -> list[str]:
    return [n for n in VARIANT_ORDER if n in variants]


def make_lr_curves(results_path: Path, out_path: Path) -> None:
    data = json.loads(results_path.read_text())
    cfg, variants = data["config"], data["variants"]
    spe = _steps_per_epoch(cfg)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name in _present_variants(variants):
        rows = variants[name]["epochs"]
        steps = [r["epoch"] * spe for r in rows]
        lrs = [r["lr_end"] for r in rows]
        ax.plot(steps, lrs, label=VARIANT_LABEL[name],
                color=VARIANT_COLOR[name], linewidth=2.0)

    warmup_step = cfg.get("warmup_epochs", 5) * spe
    ax.axvline(warmup_step, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_title(f"LR schedule — {results_path.parent.name}\n{_recipe_subtitle(cfg)}")
    ax.set_xlabel("step")
    ax.set_ylabel("LR (end of epoch)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_loss_vs_steps(results_path: Path, out_path: Path) -> None:
    data = json.loads(results_path.read_text())
    cfg, variants = data["config"], data["variants"]
    spe = _steps_per_epoch(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for name in _present_variants(variants):
        rows = variants[name]["epochs"]
        steps = [r["epoch"] * spe for r in rows]
        axes[0].plot(steps, [r["train_loss"] for r in rows],
                     label=VARIANT_LABEL[name], color=VARIANT_COLOR[name], linewidth=2.0)
        axes[1].plot(steps, [r["test_loss"] for r in rows],
                     label=VARIANT_LABEL[name], color=VARIANT_COLOR[name], linewidth=2.0)

    warmup_step = cfg.get("warmup_epochs", 5) * spe
    for ax, title in zip(axes, ["Train loss vs steps", "Test loss vs steps"]):
        ax.axvline(warmup_step, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Loss vs steps — {results_path.parent.name}\n{_recipe_subtitle(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_efficiency_comparison(results_path: Path, out_path: Path) -> None:
    data = json.loads(results_path.read_text())
    variants = data["variants"]
    names = _present_variants(variants)
    if not names:
        return

    labels = [VARIANT_LABEL[n] for n in names]
    colors = [VARIANT_COLOR[n] for n in names]
    params = [variants[n]["params"] / 1e6 for n in names]
    latency = [variants[n]["efficiency"]["latency_ms"] for n in names]
    peakmem = [variants[n]["efficiency"]["peak_mib"] for n in names]
    sec_ep = [variants[n]["mean_train_wall_s"] for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    panels = [
        (params, "Parameters (M)", "M params"),
        (latency, "Inference latency (ms)\nB=128, T=65", "ms"),
        (peakmem, "Peak GPU memory (MiB)\nB=128, T=65", "MiB"),
        (sec_ep, "Train wall-clock (s/epoch)", "s/epoch"),
    ]
    for ax, (vals, title, unit) in zip(axes, panels):
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.18)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"Efficiency — {results_path.parent.name} (B=128, T=65)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_all_figures(results_path: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    results_path = Path(results_path)
    out_dir = Path(out_dir) if out_dir is not None else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "lr_curves.png",
        out_dir / "loss_vs_steps.png",
        out_dir / "efficiency_comparison.png",
    ]
    make_lr_curves(results_path, paths[0])
    make_loss_vs_steps(results_path, paths[1])
    make_efficiency_comparison(results_path, paths[2])
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_json", type=Path, help="Path to results.json from cifar10_compare.py")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir for the 3 PNGs (default: same dir as results.json)")
    args = ap.parse_args()
    for p in make_all_figures(args.results_json, args.out_dir):
        print(f"saved {p}")


if __name__ == "__main__":
    main()
