"""Sanity tests for the v8 tracking loss (velocity + position, Huber-clipped scale)."""

from __future__ import annotations

from pathlib import Path

import torch

from mamba3_tracker.model.heads import TrackerOutputs
from mamba3_tracker.train.config import load_config
from mamba3_tracker.train.loss import TrackingLoss


REPO = Path(__file__).resolve().parents[2]


def _toy_gt(B: int = 1, F: int = 4, N: int = 3, drift: float = 0.1):
    """GT with deterministic linear motion. Anchor at frame 0."""
    torch.manual_seed(0)
    anchor = torch.tensor([[1.0, 0.5, 5.0]]).expand(B, N, 3).contiguous()
    t = torch.arange(F, dtype=torch.float32).view(1, F, 1, 1)
    motion = torch.tensor([drift, drift, drift]).view(1, 1, 1, 3) * t
    tracks = anchor.unsqueeze(1) + motion  # (B, F, N, 3)
    vis = torch.ones(B, F, N, dtype=torch.bool)
    qmask = torch.ones(B, N, dtype=torch.bool)
    anchor_idx = torch.zeros(B, N, dtype=torch.long)
    K = torch.tensor([[500.0, 0.0, 224.0],
                      [0.0, 500.0, 224.0],
                      [0.0, 0.0, 1.0]]).view(1, 3, 3).expand(B, 3, 3).contiguous()
    return tracks, vis, qmask, anchor_idx, K


def _equal_weights() -> dict[str, float]:
    # Each term weight = 1/7 after normalisation.
    keys = ("vel_3D", "vel_2D", "pos_3D", "pos_2D", "smooth_3D", "smooth_2D", "vis")
    return {k: 1.0 / len(keys) for k in keys}


def test_loss_zero_when_pred_matches_gt():
    tracks, vis, qmask, anchor_idx, K = _toy_gt()
    delta = tracks - tracks[:, :1]                   # GT delta from anchor
    pred = TrackerOutputs(
        xyz=delta.clone(),
        vis_logits=torch.full(vis.shape, 10.0),      # sigmoid → 1
        spawn_logits=torch.zeros(vis.shape),
    )
    loss = TrackingLoss(weights=_equal_weights())
    out = loss(pred, tracks, vis, qmask, anchor_idx, K)
    # vel/pos/smooth should be ~0 (vis BCE has a small floor at logits=10, fine).
    for k in ("vel_3D", "vel_2D", "pos_3D", "pos_2D", "smooth_3D", "smooth_2D"):
        assert torch.allclose(getattr(out, k), torch.zeros(()), atol=1e-6), (
            f"{k}={getattr(out, k).item()} should be ~0 on perfect prediction"
        )


def test_static_zero_predictor_is_penalised_on_moving_gt():
    """Regression guard for v6 failure mode: predicting Δp̂ = 0 must NOT be free
    when GT has motion. v8's velocity term should fire."""
    tracks, vis, qmask, anchor_idx, K = _toy_gt(drift=0.5)  # ~0.5 m/frame motion
    pred = TrackerOutputs(
        xyz=torch.zeros_like(tracks),  # static-at-anchor predictor
        vis_logits=torch.full(vis.shape, 10.0),
        spawn_logits=torch.zeros(vis.shape),
    )
    loss = TrackingLoss(weights=_equal_weights())
    out = loss(pred, tracks, vis, qmask, anchor_idx, K)
    # Δ_3D = 0.05 m, GT v ≈ 0.5 m → s_3D ≈ 0.5, residual ≈ 1 → loss ≈ O(1).
    assert out.vel_3D.item() > 0.5, (
        f"vel_3D={out.vel_3D.item()} — static predictor should produce O(1) loss"
    )
    assert out.pos_3D.item() > 0.5, (
        f"pos_3D={out.pos_3D.item()} — static predictor should miss most frames"
    )


def test_backward_populates_xyz_grad():
    tracks, vis, qmask, anchor_idx, K = _toy_gt(drift=0.3)
    delta_pred = torch.zeros_like(tracks, requires_grad=True)
    pred = TrackerOutputs(
        xyz=delta_pred,
        vis_logits=torch.full(vis.shape, 10.0, requires_grad=True),
        spawn_logits=torch.zeros(vis.shape, requires_grad=True),
    )
    loss = TrackingLoss(weights=_equal_weights())
    out = loss(pred, tracks, vis, qmask, anchor_idx, K)
    out.total.backward()
    assert delta_pred.grad is not None
    # Visible (t ≥ 1) entries should have non-zero gradient from vel + pos.
    g = delta_pred.grad[:, 1:].abs().sum()
    assert g.item() > 0.0, "no gradient flowed into Δp̂ for t ≥ 1"


def test_config_loader_normalises_loss_weights():
    cfg = load_config(REPO / "configs" / "v8.yaml")
    s = sum(cfg["loss"]["weights"].values())
    assert abs(s - 1.0) < 1e-9, f"normalised weights sum to {s}, expected 1.0"
    # Raw weights survived for the snapshot.
    assert sum(cfg["loss"]["weights_raw"].values()) > 1.0
