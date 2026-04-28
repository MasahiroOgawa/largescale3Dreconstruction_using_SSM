"""Plot loss curves for one super-phase (3 sub-phases overlaid)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAT = re.compile(
    r"step\s+(\d+)/\d+\s+loss=([-\d.]+)\s+L_D=([-\d.]+)\s+L_M=([-\d.]+)\s+"
    r"L_grad=([-\d.]+)\s+L_P=([-\d.]+)\s+L_C=([-\d.]+)"
)


def parse(path: Path):
    s, l, d, m, g, p, c = [], [], [], [], [], [], []
    if not path.exists():
        return s, l, d, m, g, p, c
    for ln in path.read_text().splitlines():
        match = PAT.search(ln)
        if match:
            s.append(int(match.group(1)))
            l.append(float(match.group(2)))
            d.append(float(match.group(3)))
            m.append(float(match.group(4)))
            g.append(float(match.group(5)))
            p.append(float(match.group(6)))
            c.append(float(match.group(7)))
    return s, l, d, m, g, p, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--super", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--logs", nargs=3, type=Path,
                    help="Three log paths in sub-1 sub-2 sub-3 order.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    super_label = {1: "DA3-SMALL teacher", 2: "DA3-LARGE teacher", 3: "GT"}[args.super]
    sub_labels = ["sub 1 (attn only)", "sub 2 (head adapt)", "sub 3 (full unfreeze)"]
    colors = ["C0", "C1", "C2"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax_total = axes[0, 0]; ax_d = axes[0, 1]; ax_m = axes[0, 2]
    ax_g = axes[1, 0]; ax_p = axes[1, 1]; ax_c = axes[1, 2]

    cumulative = 0
    boundaries = []
    for log_path, sub_label, color in zip(args.logs, sub_labels, colors):
        s, l, d, m, g, p, c = parse(log_path)
        if not s:
            print(f"NO DATA: {log_path}")
            cumulative += 500
            boundaries.append(cumulative)
            continue
        s_g = [x + cumulative for x in s]
        ax_total.plot(s_g, l, label=sub_label, color=color)
        ax_d.plot(s_g, d, color=color)
        ax_m.plot(s_g, m, color=color)
        ax_g.plot(s_g, g, color=color)
        ax_p.plot(s_g, p, color=color)
        ax_c.plot(s_g, c, color=color)
        cumulative += max(s) + 1
        boundaries.append(cumulative)

    for ax, title in [(ax_total, "Total"), (ax_d, "L_D (depth)"),
                      (ax_m, "L_M (ray)"), (ax_g, "L_grad"),
                      (ax_p, "L_P (3D pts)"), (ax_c, "L_C (cam_dec)")]:
        ax.set_xlabel("global step (within super-phase)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        for b in boundaries[:-1]:
            ax.axvline(b, color="gray", linestyle=":", alpha=0.5)
    ax_total.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Super-phase {args.super} — {super_label}", fontsize=12)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
