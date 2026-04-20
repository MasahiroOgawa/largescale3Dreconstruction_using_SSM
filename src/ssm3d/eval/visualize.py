"""Visualizations for the SSM-3D vs. DA3 ETH3D comparison.

Three artifact types:
  - depth_grid_{i}.png       : RGB | GT | DA3 depth (shared colormap + clip).
  - error_{i}.png            : |DA3 − GT|, shared color scale.
  - features_{i}.png         : RGB | DA3 feat-PCA | SSM-3D feat-PCA.
  - metric_bars_depth.png    : DA3 metrics per image (abs_rel, δ<1.25, rmse).
  - metric_bars_repr.png     : feat_cos_mean + effective_rank, DA3 vs SSM-3D.
  - summary.md               : averaged metrics, stdev, per-image table.

SSM-3D has no trained depth head; we do NOT compare depth between models
(architectural mismatch prevents shared-DPT). Depth comparison is DA3 vs GT only.
Feature comparison is head-to-head on ETH3D patch features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from ssm3d.viz.feature_pca import feature_pca_image


def _percentile_clip(arr: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo_pct)), float(np.percentile(finite, hi_pct))


def _colorize(arr: np.ndarray, lo: float, hi: float, cmap: str = "turbo", invert: bool = True) -> np.ndarray:
    norm = np.clip((arr - lo) / max(hi - lo, 1e-8), 0, 1)
    if invert:
        norm = 1.0 - norm
    return (cm.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)


def save_depth_grid(
    rgb: Tensor,
    gt: Tensor,
    pred_da3: Tensor,
    valid: Tensor,
    path: Path,
) -> Path:
    """3-panel grid: RGB | GT depth | DA3 depth. Shared colormap + clip range."""
    rgb_np = (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    gt_np = gt.float().cpu().numpy()
    pred_np = pred_da3.float().cpu().numpy()
    valid_np = valid.bool().cpu().numpy()

    ref = gt_np.copy()
    ref[~valid_np] = np.nan
    lo, hi = _percentile_clip(ref)

    gt_rgb = _colorize(np.where(valid_np, gt_np, np.nan), lo, hi)
    gt_rgb[~valid_np] = (40, 40, 40)  # grey out invalid
    pred_rgb = _colorize(pred_np, lo, hi)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes, [rgb_np, gt_rgb, pred_rgb], ["RGB", "GT depth", "DA3 depth"]
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_error_heatmap(
    pred_da3: Tensor,
    gt: Tensor,
    valid: Tensor,
    path: Path,
) -> Path:
    """|pred − gt| absolute error heatmap, masked to valid pixels."""
    pred_np = pred_da3.float().cpu().numpy()
    gt_np = gt.float().cpu().numpy()
    valid_np = valid.bool().cpu().numpy()
    err = np.abs(pred_np - gt_np)
    err[~valid_np] = np.nan
    lo = 0.0
    hi = np.nanpercentile(err, 95) if np.isfinite(err).any() else 1.0
    err_rgb = _colorize(np.where(valid_np, err, 0), lo, hi, cmap="magma", invert=False)
    err_rgb[~valid_np] = (40, 40, 40)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.imshow(err_rgb)
    ax.set_title(f"|DA3 − GT|  (white=high, capped at p95={hi:.2f}m)", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_feature_comparison(
    rgb: Tensor,
    da3_feats: Tensor,
    da3_grid_hw: tuple[int, int],
    ssm_feats: Tensor,
    ssm_grid_hw: tuple[int, int],
    path: Path,
) -> Path:
    """3-panel grid: RGB | DA3 feat PCA | SSM-3D feat PCA."""
    img_h, img_w = rgb.shape[-2:]
    rgb_np = (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    da3_pca = feature_pca_image(da3_feats, spatial_hw=da3_grid_hw, upsample_to=(img_h, img_w))
    ssm_pca = feature_pca_image(ssm_feats, spatial_hw=ssm_grid_hw, upsample_to=(img_h, img_w))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes, [rgb_np, da3_pca, ssm_pca], ["RGB", "DA3 features (PCA)", "SSM-3D features (PCA)"]
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_depth_metric_bars(
    per_image: list[dict[str, float]],
    path: Path,
) -> Path:
    """Per-image bar chart of DA3 depth metrics (abs_rel, δ<1.25, rmse, log10)."""
    keys = ["abs_rel", "delta<1.25", "rmse", "log10"]
    n = len(per_image)
    x = np.arange(n)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.5), sharex=True)
    for ax, key in zip(axes, keys):
        vals = [m.get(key, float("nan")) for m in per_image]
        ax.bar(x, vals, color="#4C72B0")
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("image idx")
    plt.suptitle("DA3 depth metrics per image (ETH3D terrains)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_repr_metric_bars(
    per_image_da3: list[dict[str, float]],
    per_image_ssm: list[dict[str, float]],
    path: Path,
) -> Path:
    """Grouped bar chart comparing DA3 vs SSM-3D representation metrics."""
    keys = ["feat_cos_mean", "effective_rank", "cross_view_nn_agreement"]
    n = len(per_image_da3)
    x = np.arange(n)
    width = 0.4
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.5), sharex=True)
    for ax, key in zip(axes, keys):
        da3_vals = [m.get(key, float("nan")) for m in per_image_da3]
        ssm_vals = [m.get(key, float("nan")) for m in per_image_ssm]
        ax.bar(x - width / 2, da3_vals, width, label="DA3", color="#4C72B0")
        ax.bar(x + width / 2, ssm_vals, width, label="SSM-3D", color="#DD8452")
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("image idx")
    axes[0].legend(loc="upper right", fontsize=9)
    plt.suptitle("Representation metrics: DA3 vs SSM-3D (ETH3D terrains)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def write_summary_md(
    path: Path,
    da3_depth: list[dict[str, float]],
    da3_repr: list[dict[str, float]],
    ssm_repr: list[dict[str, float]],
    note: str = "",
) -> Path:
    lines = [
        "# SSM-3D vs. Depth-Anything-3 on ETH3D `terrains`",
        "",
        "Per-image means ± std across the evaluated views.",
        "",
        "## Depth (DA3 only — SSM-3D has no trained depth head)",
        "",
        "| Metric | Mean | Std |",
        "|---|---|---|",
    ]
    for key in ["abs_rel", "delta<1.25", "delta<1.25^2", "rmse", "log10"]:
        m, s = _mean_std([d.get(key, float("nan")) for d in da3_depth])
        lines.append(f"| {key} | {m:.4f} | {s:.4f} |")

    lines += [
        "",
        "## Representation metrics (head-to-head)",
        "",
        "| Metric | DA3 mean ± std | SSM-3D mean ± std |",
        "|---|---|---|",
    ]
    for key in ["feat_cos_mean", "effective_rank", "cross_view_nn_agreement"]:
        d_m, d_s = _mean_std([d.get(key, float("nan")) for d in da3_repr])
        s_m, s_s = _mean_std([d.get(key, float("nan")) for d in ssm_repr])
        lines.append(f"| {key} | {d_m:.4f} ± {d_s:.4f} | {s_m:.4f} ± {s_s:.4f} |")

    if note:
        lines += ["", "## Notes", "", note]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
