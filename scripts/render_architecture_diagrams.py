"""Render DA3 / SSM-3D / diff architecture diagrams as PNGs.

Produces three figures:
  - arch_da3.png    : DA3 pipeline (softmax attention).
  - arch_ssm3d.png  : SSM-3D pipeline (Mamba-3 SSD attention).
  - arch_diff.png   : side-by-side with the changed attention block highlighted.

Block-chart style drawn with matplotlib patches — no graphviz/mermaid dep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


BOX_MUTED = "#dee2e6"
BOX_ACTIVE = "#cfe2f3"
BOX_DIFF = "#fad2cf"
EDGE_MUTED = "#adb5bd"
EDGE_DIFF = "#c0392b"
TEXT_MUTED = "#495057"
TEXT_DIFF = "#a92a1a"


def _box(ax, x, y, w, h, text, *, fill=BOX_MUTED, edge=EDGE_MUTED, lw=1.2,
         text_color="black", fontsize=9, bold=False):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, facecolor=fill, edgecolor=edge,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, weight=weight, wrap=True)


def _arrow(ax, x0, y0, x1, y1, *, label="", color=EDGE_MUTED, lw=1.4):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
    )
    if label:
        ax.text((x0 + x1) / 2 + 0.05, (y0 + y1) / 2, label,
                fontsize=7, color=color, ha="left", va="center")


def _pipeline(ax, title, *, attn_label, attn_fill, attn_edge, attn_bold,
              head_label, head_sub, muted=False, attn_text_color=None):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=10)

    fill = BOX_MUTED if muted else BOX_ACTIVE
    edge = EDGE_MUTED if muted else "#6c757d"
    tc = TEXT_MUTED if muted else "black"

    blocks = [
        (12.2, "Input images\n(B, S, 3, H, W)", fill, edge, 1.1),
        (10.7, "PatchEmbed\nConv2d 14×14", fill, edge, 1.1),
        (9.2, "[CLS] + [REG] tokens\n+ 2D RoPE", fill, edge, 1.1),
        (7.5, "12× ViT Block\n(LN → Attn → + , LN → MLP → +)", fill, edge, 1.2),
    ]
    for y, text, fc, ec, h in blocks:
        _box(ax, 2.0, y, 6.0, h, text, fill=fc, edge=ec, text_color=tc, fontsize=9)

    # Attention inset (highlighted on diff/arch_*)
    if attn_text_color is None:
        attn_text_color = TEXT_DIFF if (attn_bold and attn_edge == EDGE_DIFF) else tc
    _box(ax, 2.4, 5.4, 5.2, 1.3,
         attn_label, fill=attn_fill, edge=attn_edge,
         lw=2.2 if attn_bold else 1.4, bold=attn_bold,
         text_color=attn_text_color, fontsize=10)

    _box(ax, 2.0, 3.6, 6.0, 1.2,
         f"aux features at [5, 7, 9, 11]\n(B·S, N_patch, 384)",
         fill=fill, edge=edge, text_color=tc, fontsize=9)
    _box(ax, 2.0, 1.8, 6.0, 1.3, head_label, fill=fill, edge=edge,
         text_color=tc, fontsize=9)
    if head_sub:
        ax.text(5.0, 1.5, head_sub, ha="center", va="top",
                fontsize=7, color=tc, style="italic")
    _box(ax, 2.0, 0.3, 6.0, 1.0,
         "depth (N, H, W)", fill=fill, edge=edge,
         text_color=tc, fontsize=9)

    # Arrows between top blocks
    xs = [(13.3, 12.2), (12.2, 11.8), (10.7, 10.3), (9.2, 8.7)]
    _arrow(ax, 5.0, 12.2, 5.0, 11.8, color=edge)
    _arrow(ax, 5.0, 10.7, 5.0, 10.3, color=edge)
    _arrow(ax, 5.0, 9.2, 5.0, 8.7, color=edge)
    _arrow(ax, 5.0, 7.5, 5.0, 6.7, color=edge)  # into attention
    _arrow(ax, 5.0, 5.4, 5.0, 4.8, color=edge)  # out of attention
    _arrow(ax, 5.0, 3.6, 5.0, 3.1, color=edge)
    _arrow(ax, 5.0, 1.8, 5.0, 1.3, color=edge)


def _render_da3(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 10))
    _pipeline(
        ax, "Depth-Anything-3 (DA3)",
        attn_label="softmax attention\nqkv Linear → softmax(QKᵀ/√d)·V → proj",
        attn_fill=BOX_ACTIVE, attn_edge="#2b6cb0", attn_bold=False,
        head_label="DualDPT  (pretrained)",
        head_sub="LN → per-stage Conv1×1 → resize layers\n→ scratch + 4 fusion blocks → main/aux heads",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_ssm3d(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 10))
    _pipeline(
        ax, "SSM-3D (this repo)",
        attn_label="Mamba-3 SSD attention\nproj → B,C,V,Δ,A,λ → (L⊙CBᵀ)·V + reverse → proj",
        attn_fill=BOX_DIFF, attn_edge=EDGE_DIFF, attn_bold=True,
        head_label="DA3 DualDPT  (shared, smoke test)",
        head_sub="384-dim features duplicated → 768-dim\nfed to DA3's un-retrained head",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_diff(path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 10))

    _pipeline(
        axes[0], "DA3 (reference)",
        attn_label="softmax attention",
        attn_fill=BOX_ACTIVE, attn_edge="#2b6cb0", attn_bold=True,
        head_label="DualDPT (pretrained)",
        head_sub="",
        muted=True,
    )
    _pipeline(
        axes[1], "SSM-3D (ours)",
        attn_label="Mamba-3 SSD attention",
        attn_fill=BOX_DIFF, attn_edge=EDGE_DIFF, attn_bold=True,
        head_label="DualDPT (shared, smoke test)",
        head_sub="",
        muted=True,
    )

    fig.text(0.5, 0.48, "SWAP\nsoftmax → Mamba-3 SSD",
             ha="center", va="center", fontsize=12, weight="bold",
             color=EDGE_DIFF,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5",
                       edgecolor=EDGE_DIFF, lw=2.0))
    fig.text(0.5, 0.92,
             "Identical: PatchEmbed • MLP • norms • 2D-RoPE • DualDPT head",
             ha="center", va="center", fontsize=10, style="italic",
             color=TEXT_MUTED)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _render_da3(out_dir / "arch_da3.png"),
        _render_ssm3d(out_dir / "arch_ssm3d.png"),
        _render_diff(out_dir / "arch_diff.png"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    args = ap.parse_args()
    for p in render_all(args.out_dir):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
