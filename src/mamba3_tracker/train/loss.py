"""Tracking loss v8 — velocity-based, single Huber-clipped scale per (t, n).

v6/v7's scaled-position loss let the model collapse to "predict Δp̂ = 0":
`(Δp̂ − Δp*) / s` with per-clip median scale s ≈ 1 m and motion ≈ 5 cm
gives a residual of 0.05 — Smooth-L1(0.05) ≈ 0.00125. Predicting zero
already sits at the loss floor.

v8 replaces all relative-position terms (pos, mag, dir, reproj from v6)
with **velocity** residuals + a small absolute-position anchor, both
divided by a Huber-clipped per-(t, n) scale built from the GT velocity
magnitude. The position residual is written on the absolute prediction
`p̂ = p*_anchor + Δp̂` (numerically identical to `Δp̂ − Δp*`, but the
formulation mirrors what the TAPVid-3D evaluator measures).

Seven terms:
  * `vel_3D`     — ‖(v̂ − v*) / s_3D‖²  for visible (t≥1, n)
  * `vel_2D`     — ‖(û − u*) / s_2D‖²  for visible (t≥1, n) in pixel space
  * `pos_3D`     — ‖(p̂ − p*) / s_3D‖²  for visible (t, n)
  * `pos_2D`     — ‖(π(p̂) − π(p*)) / s_2D‖²  for visible (t, n)
  * `smooth_3D`  — ‖(v̂(t) − v̂(t−1)) / s_3D‖²  for visible (t≥2, n)
  * `smooth_2D`  — same in 2D
  * `vis`        — BCEWithLogits (surrogate for Occ-Acc)

Spawn is removed (not in TAPVid-3D eval). `pred.spawn_logits` survives
in the model output only as a no-op tensor; loss ignores it.

Final `total = Σ_i λ_i · loss_i` where `Σ_i λ_i = 1` (normalised by
the config loader in `train/config.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.heads import TrackerOutputs


_TERMS = ("vel_3D", "vel_2D", "pos_3D", "pos_2D", "smooth_3D", "smooth_2D", "vis")


@dataclass
class TrackingLossOutput:
    total: Tensor
    vel_3D: Tensor
    vel_2D: Tensor
    pos_3D: Tensor
    pos_2D: Tensor
    smooth_3D: Tensor
    smooth_2D: Tensor
    vis: Tensor


def _project(xyz: Tensor, K: Tensor) -> Tensor:
    """Pinhole projection. xyz: (B, ..., 3); K: (B, 3, 3); returns (B, ..., 2)."""
    extra_dims = xyz.dim() - 2
    view = (-1,) + (1,) * extra_dims
    fx = K[:, 0, 0].view(view)
    fy = K[:, 1, 1].view(view)
    cx = K[:, 0, 2].view(view)
    cy = K[:, 1, 2].view(view)
    Z = xyz[..., 2].clamp_min(1e-6)
    u = (xyz[..., 0] / Z) * fx + cx
    v = (xyz[..., 1] / Z) * fy + cy
    return torch.stack([u, v], dim=-1)


def _huber_scale(v: Tensor, delta: float) -> Tensor:
    """`s = sqrt(δ² + ‖v‖²)`. Detached; no gradient flows through s."""
    return (delta * delta + v.pow(2).sum(dim=-1)).clamp_min(1e-12).sqrt().detach()


def _weighted_sum_sq(x: Tensor, weight: Tensor) -> Tensor:
    """Weighted mean of `‖x‖²` (sum over last dim) over `weight > 0` entries."""
    sq = x.pow(2).sum(dim=-1)
    denom = weight.sum().clamp_min(1.0)
    return (sq * weight).sum() / denom


class TrackingLoss(nn.Module):
    """v8 loss. Construct with normalised weights + δ from `train/config.py`."""

    def __init__(
        self,
        weights: Mapping[str, float],
        delta_3d_m: float = 0.05,
        delta_2d_px: float = 1.0,
    ) -> None:
        super().__init__()
        missing = set(_TERMS) - set(weights)
        if missing:
            raise ValueError(f"TrackingLoss: missing weights for {sorted(missing)}")
        self.w = {k: float(weights[k]) for k in _TERMS}
        self.delta_3d_m = float(delta_3d_m)
        self.delta_2d_px = float(delta_2d_px)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,    # (B, F, N, 3)
        gt_visibility: Tensor,    # (B, F, N) bool
        gt_query_mask: Tensor,    # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long
        K: Tensor,                # (B, 3, 3)
    ) -> TrackingLossOutput:
        B, F_, N, _ = gt_tracks_XYZ.shape
        device = pred.xyz.device
        dtype = pred.xyz.dtype
        zero = torch.zeros((), device=device, dtype=dtype)

        # GT anchor 3-D position: p*_anchor(n)
        anchor_idx = gt_anchor_frame.clamp(min=0, max=F_ - 1)
        gt_anchor_xyz = gt_tracks_XYZ.gather(
            dim=1, index=anchor_idx.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)                                                       # (B, N, 3)

        # Absolute predicted 3-D position: p̂(t) = p*_anchor + Δp̂(t)
        p_pred = gt_anchor_xyz.unsqueeze(1) + pred.xyz                      # (B, F, N, 3)
        p_gt = gt_tracks_XYZ

        # 2-D projections (pixel space)
        uv_pred = _project(p_pred, K)                                       # (B, F, N, 2)
        uv_gt = _project(p_gt, K)

        # GT velocity v*(t) = p*(t) − p*(t−1) for t ≥ 1; padded zero at t=0
        v_gt_3d = torch.zeros_like(p_gt)
        u_gt_2d = torch.zeros_like(uv_gt)
        if F_ >= 2:
            v_gt_3d[:, 1:] = p_gt[:, 1:] - p_gt[:, :-1]
            u_gt_2d[:, 1:] = uv_gt[:, 1:] - uv_gt[:, :-1]

        # Per-(t, n) Huber-clipped scales. Floor at δ when v*=0 (incl. t=0).
        s_3D = _huber_scale(v_gt_3d, self.delta_3d_m)                       # (B, F, N)
        s_2D = _huber_scale(u_gt_2d, self.delta_2d_px)

        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm                                                  # visible (t, n)
        if F_ >= 2:
            vis_pair = vis_f[:, 1:] * vis_f[:, :-1] * qm[:, 1:]             # (B, F-1, N)
        else:
            vis_pair = torch.zeros(B, 0, N, device=device, dtype=dtype)

        # 1. Position residuals (all t).
        r_p3d = (p_pred - p_gt) / s_3D.unsqueeze(-1).clamp_min(1e-12)
        r_p2d = (uv_pred - uv_gt) / s_2D.unsqueeze(-1).clamp_min(1e-12)
        pos_3D = _weighted_sum_sq(r_p3d, w_pos)
        pos_2D = _weighted_sum_sq(r_p2d, w_pos)

        # 2. Velocity residuals (t ≥ 1). Predicted velocity comes from
        #    consecutive Δp̂ differences (the GT anchor cancels).
        if F_ >= 2:
            v_pred_3d = pred.xyz[:, 1:] - pred.xyz[:, :-1]                  # (B, F-1, N, 3)
            u_pred_2d = uv_pred[:, 1:] - uv_pred[:, :-1]
            r_v3d = (v_pred_3d - v_gt_3d[:, 1:]) / s_3D[:, 1:].unsqueeze(-1).clamp_min(1e-12)
            r_v2d = (u_pred_2d - u_gt_2d[:, 1:]) / s_2D[:, 1:].unsqueeze(-1).clamp_min(1e-12)
            vel_3D = _weighted_sum_sq(r_v3d, vis_pair)
            vel_2D = _weighted_sum_sq(r_v2d, vis_pair)
        else:
            v_pred_3d = torch.zeros(B, 0, N, 3, device=device, dtype=dtype)
            u_pred_2d = torch.zeros(B, 0, N, 2, device=device, dtype=dtype)
            vel_3D = zero
            vel_2D = zero

        # 3. Time smoothness on the *residual* velocity (t ≥ 2).
        # Penalise the second difference of (v̂ − v*) so that:
        #   * a perfect prediction yields exactly zero smoothness loss, and
        #   * the static-Δp̂=0 predictor still pays for its mismatch with the
        #     GT acceleration profile (the v6 failure mode we're trying to
        #     kill must not get a free smoothness pass).
        if F_ >= 3:
            res_v3d = v_pred_3d - v_gt_3d[:, 1:]                            # (B, F-1, N, 3)
            res_v2d = u_pred_2d - u_gt_2d[:, 1:]
            acc_res_3d = res_v3d[:, 1:] - res_v3d[:, :-1]                   # (B, F-2, N, 3)
            acc_res_2d = res_v2d[:, 1:] - res_v2d[:, :-1]
            # Triple-visible (t, t-1, t-2) for acceleration of velocity.
            triple_vis = vis_pair[:, 1:] * vis_f[:, :-2]
            s3d_acc = s_3D[:, 2:].unsqueeze(-1).clamp_min(1e-12)
            s2d_acc = s_2D[:, 2:].unsqueeze(-1).clamp_min(1e-12)
            smooth_3D = _weighted_sum_sq(acc_res_3d / s3d_acc, triple_vis)
            smooth_2D = _weighted_sum_sq(acc_res_2d / s2d_acc, triple_vis)
        else:
            smooth_3D = zero
            smooth_2D = zero

        # 4. Visibility BCE (Occ-Acc surrogate). Mean over per-frame qmask entries.
        vis_loss = F.binary_cross_entropy_with_logits(
            pred.vis_logits, vis_f, weight=qm, reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        total = (
            self.w["vel_3D"]    * vel_3D
            + self.w["vel_2D"]    * vel_2D
            + self.w["pos_3D"]    * pos_3D
            + self.w["pos_2D"]    * pos_2D
            + self.w["smooth_3D"] * smooth_3D
            + self.w["smooth_2D"] * smooth_2D
            + self.w["vis"]       * vis_loss
        )
        return TrackingLossOutput(
            total=total, vel_3D=vel_3D, vel_2D=vel_2D,
            pos_3D=pos_3D, pos_2D=pos_2D,
            smooth_3D=smooth_3D, smooth_2D=smooth_2D, vis=vis_loss,
        )
