"""Render DA3 / SSM-3D / diff architecture diagrams as PNGs.

Produces three figures:
  - arch_da3.png    : DA3 pipeline (softmax attention inside each ViT block).
  - arch_mamba3_attn.png  : SSM-3D pipeline (Mamba-3 SSD attention inside each ViT block).
  - arch_diff.png   : side-by-side with the changed sub-module highlighted.

The attention module is drawn *inside* the 12× ViT Block container because
`install_mamba3()` swaps `block.attn` for each of the 12 blocks — the rest of
the block (LN, MLP, residuals) is unchanged.

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
BOX_WHITE = "#ffffff"
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


def _arrow(ax, x0, y0, x1, y1, *, color=EDGE_MUTED, lw=1.4):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
    )


def _pipeline(ax, title, *, block_label, attn_short, attn_formula,
              attn_fill, attn_edge, attn_bold,
              head_label, head_sub, muted=False, show_formula=True):
    """Draw the DA3-style pipeline with the token-mixer sub-module nested
    inside the per-block container (that is where the swap lives)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=10)

    fill = BOX_MUTED if muted else BOX_ACTIVE
    edge = EDGE_MUTED if muted else "#6c757d"
    tc = TEXT_MUTED if muted else "black"
    attn_tc = TEXT_DIFF if (attn_bold and attn_edge == EDGE_DIFF) else tc

    def vbox(y, h, text, fontsize=9):
        _box(ax, 2.0, y, 6.0, h, text, fill=fill, edge=edge,
             text_color=tc, fontsize=fontsize)

    # --- Top vertical stack ---
    vbox(14.3, 1.0, "Input images\n(B, S, 3, H, W)")
    vbox(12.8, 1.0, "PatchEmbed\nConv2d 14×14")
    vbox(11.3, 1.0, "[CLS] + [REG] tokens\n+ 2D RoPE")

    # --- Per-block container (token-mixer lives INSIDE this block) ---
    cont_x, cont_y, cont_w, cont_h = 0.6, 6.4, 8.8, 4.4
    _box(ax, cont_x, cont_y, cont_w, cont_h, "",
         fill=fill, edge=edge, lw=1.4)
    ax.text(cont_x + cont_w / 2, cont_y + cont_h - 0.25,
            block_label,
            ha="center", va="top", fontsize=9, weight="bold", color=tc)

    # Inline residual block: x → LN → ATTN → ⊕ → LN → MLP → ⊕
    sub_y = cont_y + 1.9
    sub_h = 1.55
    items = [
        ("LN", 0.7, False),
        (attn_short, 2.0, True),
        ("⊕", 0.5, False),
        ("LN", 0.7, False),
        ("MLP", 0.85, False),
        ("⊕", 0.5, False),
    ]
    gap = 0.18
    total_w = sum(w for _, w, _ in items) + gap * (len(items) - 1)
    x = cont_x + (cont_w - total_w) / 2
    positions: list[tuple[float, float]] = []
    for label, w, is_attn in items:
        fc = attn_fill if is_attn else BOX_WHITE
        ec = attn_edge if is_attn else edge
        bold = attn_bold if is_attn else False
        lw = 2.0 if (is_attn and attn_bold) else 1.1
        _box(ax, x, sub_y, w, sub_h, label,
             fill=fc, edge=ec, bold=bold, lw=lw,
             text_color=attn_tc if is_attn else tc,
             fontsize=8 if is_attn else 10)
        positions.append((x, w))
        x += w + gap
    for (x0, w0), (x1, _) in zip(positions, positions[1:]):
        y_mid = sub_y + sub_h / 2
        ax.annotate("", xy=(x1, y_mid), xytext=(x0 + w0, y_mid),
                    arrowprops=dict(arrowstyle="->", color=edge, lw=1.0))

    if show_formula:
        ax.text(cont_x + cont_w / 2, cont_y + 1.15,
                f"ATTN ≡  {attn_formula}",
                ha="center", va="center",
                fontsize=8, style="italic", color=attn_tc,
                weight="bold" if attn_bold else "normal")

    # --- Downstream: aux features → DPT → depth ---
    vbox(5.0, 1.0, "aux features at [5, 7, 9, 11]\n(B·S, N_patch, 384)")
    vbox(3.3, 1.2, head_label)
    if head_sub:
        ax.text(5.0, 3.0, head_sub, ha="center", va="top",
                fontsize=7, color=tc, style="italic")
    vbox(1.3, 1.0, "depth (N, H, W)")

    # Arrows between vertical stages
    _arrow(ax, 5.0, 14.3, 5.0, 13.8, color=edge)
    _arrow(ax, 5.0, 12.8, 5.0, 12.3, color=edge)
    _arrow(ax, 5.0, 11.3, 5.0, 10.8, color=edge)  # → ViT container top
    _arrow(ax, 5.0, 6.4, 5.0, 6.0, color=edge)    # ViT container bottom → aux
    _arrow(ax, 5.0, 5.0, 5.0, 4.5, color=edge)
    _arrow(ax, 5.0, 3.3, 5.0, 2.3, color=edge)


def _render_da3(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 11))
    _pipeline(
        ax, "Depth-Anything-3 (DA3)",
        block_label="12× ViT Block   (×12 stack, residual stream — softmax attention inside)",
        attn_short="softmax\nattention",
        attn_formula="softmax(QKᵀ/√d) · V   (per-block, inside every ViT block)",
        attn_fill=BOX_ACTIVE, attn_edge="#2b6cb0", attn_bold=False,
        head_label="DualDPT  (pretrained)",
        head_sub="LN → per-stage Conv1×1 → resize layers\n→ scratch + 4 fusion blocks → main/aux heads",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_mamba3_attn(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 11))
    _pipeline(
        ax, "SSM-3D (this repo)",
        block_label="12× SSM Block   (×12 stack, residual stream — Mamba-3 SSD as token mixer)",
        attn_short="Mamba-3\nSSD",
        attn_formula="(L ⊙ CBᵀ) · V  +  reverse   [bidirectional Mamba-3 SSD, per-block]",
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 11))

    _pipeline(
        axes[0], "DA3 (reference)",
        block_label="12× ViT Block   (softmax attention)",
        attn_short="softmax",
        attn_formula="softmax(QKᵀ/√d)·V",
        attn_fill=BOX_ACTIVE, attn_edge="#2b6cb0", attn_bold=True,
        head_label="DualDPT (pretrained)",
        head_sub="",
        muted=True, show_formula=False,
    )
    _pipeline(
        axes[1], "SSM-3D (ours)",
        block_label="12× SSM Block   (Mamba-3 SSD token mixer)",
        attn_short="Mamba-3\nSSD",
        attn_formula="(L⊙CBᵀ)·V + reverse",
        attn_fill=BOX_DIFF, attn_edge=EDGE_DIFF, attn_bold=True,
        head_label="DualDPT (shared, smoke test)",
        head_sub="",
        muted=True, show_formula=False,
    )

    fig.text(0.5, 0.96,
             "SWAP (×12, per-block):  block.attn : softmax → Mamba-3 SSD",
             ha="center", va="center", fontsize=11, weight="bold",
             color=EDGE_DIFF,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5",
                       edgecolor=EDGE_DIFF, lw=2.0))
    fig.text(0.5, 0.02,
             "Identical: PatchEmbed • LN • MLP • residuals • 2D-RoPE • DualDPT head",
             ha="center", va="center", fontsize=10, style="italic",
             color=TEXT_MUTED)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _render_da3(out_dir / "arch_da3.png"),
        _render_mamba3_attn(out_dir / "arch_mamba3_attn.png"),
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
