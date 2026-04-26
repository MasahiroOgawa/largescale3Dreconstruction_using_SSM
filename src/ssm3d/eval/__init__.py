"""Evaluation utilities for comparing SSM-3D against Depth-Anything-3 on ETH3D."""

from .metrics import (
    abs_relative_depth_error,
    delta_threshold,
    rmse,
    log10_metric,
    depth_metrics,
    align_scale_median,
    feat_cos_mean,
    effective_rank,
    cross_view_nn_agreement,
    gt_camera_rays,
    ray_angular_error,
)

__all__ = [
    "abs_relative_depth_error",
    "delta_threshold",
    "rmse",
    "log10_metric",
    "depth_metrics",
    "align_scale_median",
    "feat_cos_mean",
    "effective_rank",
    "cross_view_nn_agreement",
    "gt_camera_rays",
    "ray_angular_error",
]
