"""Plot CM23 |relative_depth_error| vs training step to visualise the overfit boundary.

Also plots the CM22 vs CM23 cosine LR schedule so the step-1000 divergence
is obvious at a glance.
"""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt

steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
rel_err_cm23 = [0.0642, 0.0669, 0.0702, 0.0854, 0.0697, 0.0694, 0.0703, 0.0678]

CM22_1000 = 0.0531
DA3_SMALL = 0.0417
CM12_500 = 0.0676


def cosine_lr(t: int, T_max: int, peak: float = 1e-5, floor_frac: float = 0.1) -> float:
    eta_min = peak * floor_frac
    t = min(t, T_max)
    return eta_min + 0.5 * (peak - eta_min) * (1 + math.cos(math.pi * t / T_max))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(steps, rel_err_cm23, "o-", color="crimson", linewidth=2,
         markersize=8, label="CM23 (8 k schedule)")
ax1.plot([1000], [CM22_1000], "s", color="seagreen", markersize=12,
         label=f"CM22@1000 = {CM22_1000:.4f}")

peak_idx = int(min(range(len(rel_err_cm23)), key=rel_err_cm23.__getitem__))
ax1.annotate(
    f"CM23 peak @1000: {rel_err_cm23[peak_idx]:.4f}",
    xy=(steps[peak_idx], rel_err_cm23[peak_idx]),
    xytext=(steps[peak_idx] + 800, rel_err_cm23[peak_idx] + 0.002),
    fontsize=10, color="crimson",
    arrowprops=dict(arrowstyle="->", color="crimson"),
)
ax1.annotate(
    f"overfit spike @4000: 0.0854",
    xy=(4000, 0.0854), xytext=(4400, 0.082),
    fontsize=10, color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred"),
)
ax1.axhline(CM22_1000, color="seagreen", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.axhline(DA3_SMALL, color="steelblue", linestyle=":", linewidth=1.5,
            label=f"DA3-SMALL = {DA3_SMALL:.4f} (target)")
ax1.axhline(CM12_500, color="gray", linestyle=":", linewidth=1.0,
            label=f"CM12@500 = {CM12_500:.4f} (init)")
ax1.set_xlabel("training step")
ax1.set_ylabel("|relative_depth_error|  (lower is better)")
ax1.set_title("Depth accuracy on ETH3D `terrains`")
ax1.set_xticks(steps)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", fontsize=9)

t22 = list(range(0, 1001, 25))
lr22 = [cosine_lr(t, T_max=1000) for t in t22]
t23 = list(range(0, 8001, 100))
lr23 = [cosine_lr(t, T_max=8000) for t in t23]

ax2.plot(t22, lr22, color="seagreen", linewidth=2, label="CM22 (T_max=1000)")
ax2.plot(t23, lr23, color="crimson", linewidth=2, label="CM23 (T_max=8000)")

lr22_at1k = cosine_lr(1000, T_max=1000)
lr23_at1k = cosine_lr(1000, T_max=8000)
ax2.plot([1000], [lr22_at1k], "s", color="seagreen", markersize=12)
ax2.plot([1000], [lr23_at1k], "o", color="crimson", markersize=12)
ax2.annotate(
    f"CM22@1000\nLR = {lr22_at1k:.2e}\n(10 % floor)",
    xy=(1000, lr22_at1k), xytext=(2200, lr22_at1k + 1e-6),
    fontsize=9, color="seagreen",
    arrowprops=dict(arrowstyle="->", color="seagreen"),
)
ax2.annotate(
    f"CM23@1000\nLR = {lr23_at1k:.2e}\n(97 % of peak)",
    xy=(1000, lr23_at1k), xytext=(2200, lr23_at1k - 2.5e-6),
    fontsize=9, color="crimson",
    arrowprops=dict(arrowstyle="->", color="crimson"),
)
ax2.axvline(1000, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("training step")
ax2.set_ylabel("learning rate (attn/dpt group)")
ax2.set_title("Why step-1000 |relative_depth_error| differs: the LR schedule\n"
              "CosineAnnealingLR(T_max=cfg.steps, eta_min=peak·0.1)")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", fontsize=10)

plt.suptitle("CM22 vs CM23 — same init, same recipe, same LRs; "
             "only `--steps` differs", fontsize=11, y=1.02)
plt.tight_layout()
out = Path("outputs/runs/depth_ft_cm23/cm23_overfit.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
