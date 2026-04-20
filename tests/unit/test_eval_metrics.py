"""Sanity tests for eval metrics on toy data."""

from __future__ import annotations

import torch

from ssm3d.eval.metrics import (
    abs_rel,
    align_scale_median,
    cross_view_nn_agreement,
    delta_threshold,
    depth_metrics,
    effective_rank,
    feat_cos_mean,
    log10_metric,
    rmse,
)


def test_perfect_depth_gives_zero_error():
    gt = torch.full((8, 8), 2.0)
    valid = torch.ones_like(gt, dtype=torch.bool)
    pred = gt.clone()
    assert abs_rel(pred, gt, valid) == 0.0
    assert rmse(pred, gt, valid) == 0.0
    assert log10_metric(pred, gt, valid) == 0.0
    assert delta_threshold(pred, gt, valid, 1.25) == 1.0


def test_align_scale_median_matches_median():
    gt = torch.tensor([1.0, 2.0, 4.0, 8.0])
    pred = gt * 3.5  # off by a factor
    valid = torch.ones_like(gt, dtype=torch.bool)
    aligned = align_scale_median(pred, gt, valid)
    assert torch.isclose(torch.median(aligned), torch.median(gt))


def test_delta_threshold_on_known_ratios():
    gt = torch.tensor([1.0, 1.0, 1.0, 1.0])
    pred = torch.tensor([1.0, 1.1, 1.3, 2.0])  # ratios 1.0, 1.1, 1.3, 2.0
    valid = torch.ones_like(gt, dtype=torch.bool)
    d = delta_threshold(pred, gt, valid, 1.25)
    assert abs(d - 0.5) < 1e-5  # first two within 1.25


def test_depth_metrics_as_dict_fields():
    gt = torch.full((4, 4), 2.0)
    pred = gt * 1.05
    valid = torch.ones_like(gt, dtype=torch.bool)
    m = depth_metrics(pred, gt, valid, align=False).as_dict()
    for key in ["abs_rel", "delta<1.25", "delta<1.25^2", "rmse", "log10"]:
        assert key in m


def test_feat_cos_mean_collapsed_vs_orthogonal():
    # Collapsed: all tokens identical -> cos = 1
    C = 8
    N = 16
    collapsed = torch.ones(N, C)
    assert abs(feat_cos_mean(collapsed) - 1.0) < 1e-4
    # Orthogonal-ish: random gaussian tokens -> mean off-diag ~ 0
    torch.manual_seed(0)
    random_feats = torch.randn(64, 128)
    assert abs(feat_cos_mean(random_feats)) < 0.15


def test_effective_rank_bounds():
    # Rank-1: one direction -> effective rank ~ 1
    v = torch.randn(1, 64)
    rank1 = torch.cat([v * c for c in torch.linspace(1, 2, 32)], dim=0)
    assert effective_rank(rank1) < 1.5
    # Full-rank random -> high effective rank
    torch.manual_seed(0)
    full = torch.randn(128, 64)
    assert effective_rank(full) > 20


def test_cross_view_nn_agreement_identity_pair():
    """Two identical images + identity pose -> perfect agreement (1.0).

    We set depth to constant, intrinsic to a simple K, and the two extrinsics
    to identity. Features are a random (T, C) matrix reused for both views.
    """
    H, W = 8, 8
    T = H * W
    feats = torch.randn(T, 32)
    depth = torch.full((56, 56), 5.0)  # image-level
    K = torch.tensor([[50.0, 0, 28.0], [0, 50.0, 28.0], [0, 0, 1]])
    E = torch.eye(4)
    score = cross_view_nn_agreement(
        feats, feats,
        grid_hw=(H, W),
        depth_a=depth,
        intrinsic_a=K, intrinsic_b=K,
        extrinsic_a_w2c=E, extrinsic_b_w2c=E,
        image_hw_b=(56, 56),
        radius_px=5.0,
    )
    assert score > 0.95, f"identity pair should near-perfectly agree, got {score}"
