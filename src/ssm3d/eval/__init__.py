"""Evaluation utilities for comparing SSM-3D against Depth-Anything-3 on ETH3D."""

from .metrics import (
    abs_rel,
    delta_threshold,
    rmse,
    log10_metric,
    depth_metrics,
    align_scale_median,
    feat_cos_mean,
    effective_rank,
    cross_view_nn_agreement,
)

__all__ = [
    "abs_rel",
    "delta_threshold",
    "rmse",
    "log10_metric",
    "depth_metrics",
    "align_scale_median",
    "feat_cos_mean",
    "effective_rank",
    "cross_view_nn_agreement",
]
