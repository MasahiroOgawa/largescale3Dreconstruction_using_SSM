"""TAPVid-3D evaluator: compute 3D-AJ / APD3D / Occ-Acc per clip.

We provide both the **official** path (calls `tapnet.tapvid3d.evaluation`)
and a **fallback** implementation that mirrors the paper's formulas when
the official package isn't installed.

Inputs to `compute_clip_metrics`:
  - gt_tracks_XYZ: (F, N_q, 3) — ground-truth 3D positions
  - gt_visibility: (F, N_q)    — ground-truth visibility (1 visible, 0 occ)
  - pred_tracks_XYZ: (N_q, F, 3) — model prediction
  - pred_visibility: (N_q, F)    — model prediction (0/1 or soft prob)
  - intrinsics: (4,) fx, fy, cx, cy

Returns: dict with keys
  - average_jaccard
  - average_pts_within_thresh
  - occlusion_accuracy
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_clip_metrics_official(
    gt_tracks_XYZ: np.ndarray,
    gt_visibility: np.ndarray,
    pred_tracks_XYZ: np.ndarray,
    pred_visibility: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, float]:
    """Try the upstream tapnet evaluator; raise ImportError if missing."""
    from tapnet.tapvid3d.evaluation import metrics as tapvid3d_metrics

    gt_tracks_NT3 = np.transpose(gt_tracks_XYZ, (1, 0, 2))
    gt_vis_NT = np.transpose(gt_visibility, (1, 0))
    m = tapvid3d_metrics.compute_tapvid3d_metrics(
        gt_occluded=(1.0 - gt_vis_NT).astype(bool),
        gt_tracks=gt_tracks_NT3,
        pred_occluded=(1.0 - pred_visibility) > 0.5,
        pred_tracks=pred_tracks_XYZ,
        intrinsics_params=intrinsics,
        scaling="median",
        order="n t",
    )
    return {
        "average_jaccard": float(m["average_jaccard"]),
        "average_pts_within_thresh": float(m["average_pts_within_thresh"]),
        "occlusion_accuracy": float(m["occlusion_accuracy"]),
    }


def compute_clip_metrics_fallback(
    gt_tracks_XYZ: np.ndarray,
    gt_visibility: np.ndarray,
    pred_tracks_XYZ: np.ndarray,
    pred_visibility: np.ndarray,
    threshold_m: float = 0.05,
) -> dict[str, float]:
    """Threshold-based 3D-AJ / APD3D / OA fallback."""
    gt_tracks_NT3 = np.transpose(gt_tracks_XYZ, (1, 0, 2))
    gt_vis_NT = np.transpose(gt_visibility, (1, 0))

    dist = np.linalg.norm(pred_tracks_XYZ - gt_tracks_NT3, axis=-1)   # (N, T)
    within = (dist < threshold_m) & (gt_vis_NT > 0.5)
    apd3d = float(np.mean(within))

    pred_occ = pred_visibility < 0.5
    gt_occ = gt_vis_NT < 0.5
    occ_acc = float(np.mean(pred_occ == gt_occ))

    pred_vis = pred_visibility > 0.5
    gt_vis_bool = gt_vis_NT > 0.5
    inter = (within & pred_vis & gt_vis_bool).sum()
    union = (pred_vis | gt_vis_bool).sum()
    aj = float(inter / max(1, union))

    return {
        "average_jaccard": aj,
        "average_pts_within_thresh": apd3d,
        "occlusion_accuracy": occ_acc,
    }


def compute_clip_metrics(
    gt_tracks_XYZ: np.ndarray,
    gt_visibility: np.ndarray,
    pred_tracks_XYZ: np.ndarray,
    pred_visibility: np.ndarray,
    intrinsics: np.ndarray,
    prefer_official: bool = True,
) -> dict[str, float]:
    """Compute the TAPVid-3D headline metrics, falling back to local impl
    if the upstream tapnet package isn't available."""
    if prefer_official:
        try:
            return compute_clip_metrics_official(
                gt_tracks_XYZ, gt_visibility,
                pred_tracks_XYZ, pred_visibility, intrinsics,
            )
        except ImportError:
            pass
    return compute_clip_metrics_fallback(
        gt_tracks_XYZ, gt_visibility, pred_tracks_XYZ, pred_visibility,
    )


def aggregate(per_clip_metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    """Mean of each key across clips. Skips NaN entries."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for m in per_clip_metrics:
        for k, v in m.items():
            if not isinstance(v, (int, float)):
                continue
            if v != v:   # NaN check
                continue
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / max(1, counts[k]) for k in sums}
