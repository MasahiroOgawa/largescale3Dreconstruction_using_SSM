"""Visualizations for the SSM-3D vs. DA3 ETH3D comparison.

Artifact types:
  - depth_grid_{i}.png       : 2x2 corners — Input | GT / DA3 | SSM-3D (shared clip).
  - error_{i}.png            : 1x2 — |DA3 − GT| | |SSM-3D − GT| (shared scale).
  - features_{i}.png         : RGB | DA3 feat-PCA | SSM-3D feat-PCA.
  - metric_bars_depth.png    : DA3 metrics per image (|relative_depth_error|, δ<1.25, rmse).
  - metric_bars_repr.png     : feat_cos_mean + effective_rank, DA3 vs SSM-3D.
  - summary.md               : averaged metrics, stdev, per-image table.

SSM-3D depth is the shared-DPT smoke test (DA3's pretrained DualDPT bolted
onto SSM-3D 384→768 duplicated features, not retrained). Figures label it so.
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


def _median_align(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pv = pred[valid]
    gv = gt[valid]
    if pv.size == 0 or gv.size == 0:
        return pred
    p_med = float(np.median(pv))
    if abs(p_med) < 1e-8:
        return pred
    return pred * (float(np.median(gv)) / p_med)


def save_depth_grid(
    rgb: Tensor,
    gt: Tensor,
    pred_da3: Tensor,
    valid: Tensor,
    path: Path,
    pred_ssm: Tensor | None = None,
) -> Path:
    """2x2 corner grid: Input | GT / DA3 | SSM-3D. Shared colormap + clip range.

    If `pred_ssm` is None, the bottom-right cell is drawn as a labelled
    "N/A" placeholder so the layout is stable across runs.
    """
    rgb_np = (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    gt_np = gt.float().cpu().numpy()
    pred_np = pred_da3.float().cpu().numpy()
    valid_np = valid.bool().cpu().numpy()

    ref = gt_np.copy()
    ref[~valid_np] = np.nan
    lo, hi = _percentile_clip(ref)

    gt_rgb = _colorize(np.where(valid_np, gt_np, np.nan), lo, hi)
    gt_rgb[~valid_np] = (40, 40, 40)
    da3_aligned = _median_align(pred_np, gt_np, valid_np)
    da3_rgb = _colorize(da3_aligned, lo, hi)

    if pred_ssm is not None:
        ssm_np = pred_ssm.float().cpu().numpy()
        ssm_aligned = _median_align(ssm_np, gt_np, valid_np)
        ssm_rgb = _colorize(ssm_aligned, lo, hi)
        ssm_title = "SSM-3D depth (shared-DPT smoke test, median-aligned)"
    else:
        ssm_rgb = np.full_like(rgb_np, 200)
        ssm_title = "SSM-3D depth (unavailable)"

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    panels = [
        (axes[0, 0], rgb_np, "Input"),
        (axes[0, 1], gt_rgb, "GT depth"),
        (axes[1, 0], da3_rgb, "DA3 depth (median-aligned)"),
        (axes[1, 1], ssm_rgb, ssm_title),
    ]
    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle(f"depth (clip {lo:.2f}–{hi:.2f} m)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_error_heatmap(
    pred_da3: Tensor,
    gt: Tensor,
    valid: Tensor,
    path: Path,
    pred_ssm: Tensor | None = None,
) -> Path:
    """1x2 side-by-side error heatmaps with a shared color scale.

    Left panel: |DA3 − GT|. Right panel: |SSM-3D − GT| (median-aligned first,
    since the shared-DPT smoke-test output is scale-ambiguous). The shared
    clip is p95 of the UNION of both error maps so the two halves are
    directly comparable.
    """
    gt_np = gt.float().cpu().numpy()
    valid_np = valid.bool().cpu().numpy()
    da3_aligned = _median_align(pred_da3.float().cpu().numpy(), gt_np, valid_np)
    err_da3 = np.abs(da3_aligned - gt_np)
    err_da3[~valid_np] = np.nan

    if pred_ssm is not None:
        ssm_np = pred_ssm.float().cpu().numpy()
        ssm_aligned = _median_align(ssm_np, gt_np, valid_np)
        err_ssm = np.abs(ssm_aligned - gt_np)
        err_ssm[~valid_np] = np.nan
        stacked = np.concatenate(
            [err_da3[np.isfinite(err_da3)], err_ssm[np.isfinite(err_ssm)]]
        )
        hi = float(np.percentile(stacked, 95)) if stacked.size else 1.0
    else:
        err_ssm = None
        hi = float(np.nanpercentile(err_da3, 95)) if np.isfinite(err_da3).any() else 1.0
    lo = 0.0

    def _panel(err: np.ndarray) -> np.ndarray:
        rgb = _colorize(np.where(valid_np, err, 0), lo, hi, cmap="magma", invert=False)
        rgb[~valid_np] = (40, 40, 40)
        return rgb

    da3_rgb = _panel(err_da3)
    da3_p95 = float(np.nanpercentile(err_da3, 95)) if np.isfinite(err_da3).any() else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(da3_rgb)
    axes[0].set_title(f"|DA3 − GT|  (median-aligned; p95={da3_p95:.2f} m)", fontsize=10)
    axes[0].axis("off")
    if err_ssm is not None:
        ssm_p95 = (
            float(np.nanpercentile(err_ssm, 95)) if np.isfinite(err_ssm).any() else float("nan")
        )
        axes[1].imshow(_panel(err_ssm))
        axes[1].set_title(
            f"|SSM-3D − GT| (shared-DPT, median-aligned; p95={ssm_p95:.2f} m)", fontsize=10
        )
    else:
        axes[1].imshow(np.full_like(da3_rgb, 200))
        axes[1].set_title("SSM-3D error (unavailable)", fontsize=10)
    axes[1].axis("off")
    plt.suptitle(f"absolute depth error (shared scale, capped at p95={hi:.2f} m)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    per_image_da3: list[dict[str, float]],
    path: Path,
    per_image_ssm: list[dict[str, float]] | None = None,
) -> Path:
    """Per-image bar chart of depth metrics (|relative_depth_error|, δ<1.25, rmse, log10).

    When `per_image_ssm` is provided, renders grouped DA3-vs-SSM-3D bars.
    """
    keys = ["|relative_depth_error|", "delta<1.25", "rmse", "log10"]
    titles = {
        "|relative_depth_error|": r"|relative_depth_error| = mean(|d̂−d|/d)   ↓ lower is better",
        "delta<1.25": r"δ<1.25 = frac{max(d̂/d, d/d̂) < 1.25}   ↑ higher is better",
        "rmse": r"rmse = √mean(d̂−d)²  [m]   ↓ lower is better",
        "log10": r"log10 = mean|log₁₀d̂ − log₁₀d|   ↓ lower is better",
    }
    n = len(per_image_da3)
    x = np.arange(n)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.5), sharex=True)
    if per_image_ssm is None:
        for ax, key in zip(axes, keys):
            vals = [m.get(key, float("nan")) for m in per_image_da3]
            ax.bar(x, vals, color="#4C72B0")
            ax.set_title(titles[key], fontsize=9)
            ax.set_xlabel("image idx")
        plt.suptitle("DA3 depth metrics per image (ETH3D terrains)", fontsize=11)
    else:
        width = 0.4
        for ax, key in zip(axes, keys):
            da3_vals = [m.get(key, float("nan")) for m in per_image_da3]
            ssm_vals = [m.get(key, float("nan")) for m in per_image_ssm]
            ax.bar(x - width / 2, da3_vals, width, label="DA3", color="#4C72B0")
            ax.bar(x + width / 2, ssm_vals, width, label="SSM-3D", color="#DD8452")
            ax.set_title(titles[key], fontsize=9)
            ax.set_xlabel("image idx")
        axes[0].legend(loc="upper right", fontsize=9)
        plt.suptitle("Depth metrics per image: DA3 vs SSM-3D (ETH3D terrains, median-aligned)", fontsize=11)
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
    titles = {
        "feat_cos_mean": "feat_cos_mean  (mean pairwise token cosine, ↓ = less collapse)",
        "effective_rank": "effective_rank  (exp(entropy of SVD spectrum), ↑ richer)",
        "cross_view_nn_agreement": "cross_view_nn_agreement  (GT-warped NN match frac, ↑)",
    }
    n = len(per_image_da3)
    x = np.arange(n)
    width = 0.4
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.5), sharex=True)
    for ax, key in zip(axes, keys):
        da3_vals = [m.get(key, float("nan")) for m in per_image_da3]
        ssm_vals = [m.get(key, float("nan")) for m in per_image_ssm]
        ax.bar(x - width / 2, da3_vals, width, label="DA3", color="#4C72B0")
        ax.bar(x + width / 2, ssm_vals, width, label="SSM-3D", color="#DD8452")
        ax.set_title(titles[key], fontsize=9)
        ax.set_xlabel("image idx")
    axes[0].legend(loc="upper right", fontsize=9)
    plt.suptitle("Representation metrics per image: DA3 vs SSM-3D (ETH3D terrains)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_memory_bars(
    da3_report: dict[str, float],
    ssm_report: dict[str, float],
    path: Path,
) -> Path:
    """Side-by-side bar chart: parameter count and peak RSS delta, DA3 vs SSM-3D.

    Separate subplots because the two axes differ by ~3 orders of magnitude
    (params in millions, RSS in MB) and a shared axis would squash one bar.
    If `peak_cuda_mb` is non-zero in either report, a third subplot is added.
    """
    labels = ["DA3", "SSM-3D"]
    params_m = [da3_report.get("param_count", 0.0) / 1e6,
                ssm_report.get("param_count", 0.0) / 1e6]
    rss_mb = [da3_report.get("peak_rss_delta_mb", 0.0),
              ssm_report.get("peak_rss_delta_mb", 0.0)]
    cuda_mb = [da3_report.get("peak_cuda_mb", 0.0),
               ssm_report.get("peak_cuda_mb", 0.0)]
    use_cuda = any(v > 0 for v in cuda_mb)

    n = 3 if use_cuda else 2
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    colors = ["#4C72B0", "#DD8452"]

    def _bar(ax, vals, title, ylabel):
        ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    _bar(axes[0], params_m, "Parameters (backbone only)", "M params")
    _bar(axes[1], rss_mb, "Peak host RSS delta (one fwd)", "MB")
    if use_cuda:
        _bar(axes[2], cuda_mb, "Peak CUDA memory (one fwd)", "MB")

    plt.suptitle("Memory footprint: DA3 vs SSM-3D (backbone-only params; peak deltas per inference call)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


_DEPTH_GATES: dict[str, tuple[str, float]] = {
    "|relative_depth_error|": ("le", 0.073),
    "delta<1.25": ("ge", 0.935),
    "rmse": ("le", 0.29),
    "log10": ("le", 0.031),
}

_REPR_GATES: dict[str, tuple[str, float]] = {
    "cross_view_nn_agreement": ("ge", 0.55),
    "effective_rank": ("ge", 150.0),
}


def _gate_ok(mean: float, direction: str, threshold: float) -> bool:
    if not np.isfinite(mean):
        return False
    return mean <= threshold if direction == "le" else mean >= threshold


def _md_escape(key: str) -> str:
    """Escape pipes so metric keys like `|relative_depth_error|` are safe
    in GFM table cells (raw `|` would be read as a column separator)."""
    return key.replace("|", r"\|")


def write_summary_md(
    path: Path,
    da3_depth: list[dict[str, float]],
    da3_repr: list[dict[str, float]],
    ssm_repr: list[dict[str, float]],
    note: str = "",
    memory_da3: dict[str, float] | None = None,
    memory_ssm: dict[str, float] | None = None,
    ssm_depth: list[dict[str, float]] | None = None,
) -> Path:
    lines = [
        "# SSM-3D vs. Depth-Anything-3 on ETH3D `terrains`",
        "",
        "Per-image means ± std across the evaluated views.",
        "",
    ]
    depth_keys = ["|relative_depth_error|", "delta<1.25", "delta<1.25^2", "rmse", "log10"]
    if ssm_depth is not None and any(len(m) > 0 for m in ssm_depth):
        lines += [
            "## Depth (median-aligned; head-to-head)",
            "",
            "| Metric | DA3 mean ± std | SSM-3D mean ± std |",
            "|---|---|---|",
        ]
        for key in depth_keys:
            d_m, d_s = _mean_std([d.get(key, float("nan")) for d in da3_depth])
            s_m, s_s = _mean_std([d.get(key, float("nan")) for d in ssm_depth])
            lines.append(f"| {_md_escape(key)} | {d_m:.4f} ± {d_s:.4f} | {s_m:.4f} ± {s_s:.4f} |")
    else:
        lines += [
            "## Depth (DA3 only)",
            "",
            "| Metric | Mean | Std |",
            "|---|---|---|",
        ]
        for key in depth_keys:
            m, s = _mean_std([d.get(key, float("nan")) for d in da3_depth])
            lines.append(f"| {_md_escape(key)} | {m:.4f} | {s:.4f} |")

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
        lines.append(f"| {_md_escape(key)} | {d_m:.4f} ± {d_s:.4f} | {s_m:.4f} ± {s_s:.4f} |")

    if ssm_depth is not None and any(len(m) > 0 for m in ssm_depth):
        lines += [
            "",
            "## Acceptance gates (PLAN §9)",
            "",
            "| Gate | Target | SSM-3D mean | Pass |",
            "|---|---|---|---|",
        ]
        for key, (direction, threshold) in _DEPTH_GATES.items():
            mean_val = _mean_std([d.get(key, float("nan")) for d in ssm_depth])[0]
            op = "≤" if direction == "le" else "≥"
            ok = "✅" if _gate_ok(mean_val, direction, threshold) else "❌"
            lines.append(f"| {_md_escape(key)} | {op} {threshold:.4f} | {mean_val:.4f} | {ok} |")
        for key, (direction, threshold) in _REPR_GATES.items():
            mean_val = _mean_std([d.get(key, float("nan")) for d in ssm_repr])[0]
            op = "≤" if direction == "le" else "≥"
            ok = "✅" if _gate_ok(mean_val, direction, threshold) else "❌"
            lines.append(f"| {_md_escape(key)} | {op} {threshold:.4f} | {mean_val:.4f} | {ok} |")

    if memory_da3 is not None and memory_ssm is not None:
        lines += [
            "",
            "## Memory usage",
            "",
            "| Metric | DA3 | SSM-3D |",
            "|---|---|---|",
            f"| Parameters (M) | {memory_da3.get('param_count', 0) / 1e6:.2f} | "
            f"{memory_ssm.get('param_count', 0) / 1e6:.2f} |",
            f"| Peak RSS delta during inference (MB) | "
            f"{memory_da3.get('peak_rss_delta_mb', 0):.1f} | "
            f"{memory_ssm.get('peak_rss_delta_mb', 0):.1f} |",
        ]
        if memory_da3.get("peak_cuda_mb", 0) > 0 or memory_ssm.get("peak_cuda_mb", 0) > 0:
            lines.append(
                f"| Peak CUDA memory (MB) | {memory_da3.get('peak_cuda_mb', 0):.1f} | "
                f"{memory_ssm.get('peak_cuda_mb', 0):.1f} |"
            )

    if note:
        lines += ["", "## Notes", "", note]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
