"""Tracking loss v2 — relative motion + per-clip median scale normalisation.

Because the propagator's query-conditioned initialisation pins slot `n`
to GT track `n` by construction (`doc/attention/mamba3_attention.tex
§8.3`), there is no Hungarian matching. The position head predicts
Δp_n^(t) = p_n^(t) − p_n^query, and the loss compares it against the
GT motion Δp*_n^(t) after dividing by a per-clip median scale `s`.

Four terms (§8.6):
  - Scale-normalised relative position L1 on visible frames.
  - Visibility BCE.
  - Spawn BCE on the first-visible frame per track.
  - Temporal smoothness on the scaled Δp trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.heads import TrackerOutputs


@dataclass
class TrackingLossWeights:
    pos: float = 1.0
    vis: float = 0.5
    spawn: float = 0.5
    smooth: float = 0.1


@dataclass
class TrackingLossOutput:
    total: Tensor
    pos: Tensor
    vis: Tensor
    spawn: Tensor
    smooth: Tensor
    scale: Tensor   # per-clip median scale used (B,) — for logging


def _per_clip_median_scale(
    gt_anchor_xyz: Tensor,    # (B, N, 3)
    query_mask: Tensor,       # (B, N) bool
    eps: float = 1e-3,
) -> Tensor:
    """s_b = median over valid tracks of ||gt_anchor_xyz_n||₂.

    Falls back to `eps` when a clip has no valid tracks.
    """
    B = gt_anchor_xyz.shape[0]
    norms = gt_anchor_xyz.norm(dim=-1)                    # (B, N)
    s = torch.full((B,), eps, device=gt_anchor_xyz.device, dtype=gt_anchor_xyz.dtype)
    for b in range(B):
        keep = query_mask[b]
        if keep.any():
            s[b] = norms[b, keep].median()
    return s.clamp_min(eps)


def _first_visible_frame(vis: Tensor) -> Tensor:
    """(B, F, N) bool → (B, N) long: index of first True along F (or F if never)."""
    B, F_, N = vis.shape
    if F_ == 0:
        return torch.full((B, N), F_, dtype=torch.long, device=vis.device)
    idx = torch.arange(F_, device=vis.device).view(1, F_, 1).expand(B, F_, N)
    masked_idx = torch.where(vis, idx, torch.full_like(idx, F_))
    return masked_idx.min(dim=1).values


class TrackingLoss(nn.Module):
    def __init__(self, weights: TrackingLossWeights | None = None) -> None:
        super().__init__()
        self.w = weights or TrackingLossWeights()

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,    # (B, F, N, 3)
        gt_visibility: Tensor,    # (B, F, N) bool
        gt_query_mask: Tensor,    # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long
    ) -> TrackingLossOutput:
        B, F_, N, _ = gt_tracks_XYZ.shape
        device = pred.xyz.device

        # Gather GT anchor positions: p^*_n^(t_n^q).
        anchor_idx = gt_anchor_frame.clamp(min=0, max=F_ - 1)
        gt_anchor_xyz = gt_tracks_XYZ.gather(
            dim=1, index=anchor_idx.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)                                                  # (B, N, 3)

        # Per-clip median scale s_b.
        s = _per_clip_median_scale(gt_anchor_xyz, gt_query_mask)       # (B,)
        s_inv = (1.0 / s).view(B, 1, 1, 1)

        # GT motion Δp* = p* − p^q (broadcast anchor over the F axis).
        delta_gt = gt_tracks_XYZ - gt_anchor_xyz.unsqueeze(1)         # (B, F, N, 3)

        # Predicted Δp (head output is interpreted as relative motion).
        delta_pred = pred.xyz                                          # (B, F, N, 3)

        # Scale-normalised residual.
        r = (delta_pred - delta_gt) * s_inv                            # (B, F, N, 3)

        vis_f = gt_visibility.float()                                  # (B, F, N)
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()       # (B, F, N)

        # Position L1 on visible matched cells (slot = track by construction).
        l1 = r.abs().sum(dim=-1)                                       # (B, F, N)
        w_pos = vis_f * qm
        pos_loss = (l1 * w_pos).sum() / w_pos.sum().clamp_min(1.0)

        # Visibility BCE over all valid query slots.
        vis_loss = F.binary_cross_entropy_with_logits(
            pred.vis_logits, vis_f, weight=qm, reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        # Spawn BCE: positive at first visible frame per track.
        first_vis = _first_visible_frame(gt_visibility)                # (B, N)
        t_idx = torch.arange(F_, device=device).view(1, F_, 1).expand(B, F_, N)
        spawn_target = (t_idx == first_vis.unsqueeze(1)).float()
        ever_vis = gt_visibility.any(dim=1).float().unsqueeze(1)       # (B, 1, N)
        w_spawn = qm * ever_vis
        spawn_loss = F.binary_cross_entropy_with_logits(
            pred.spawn_logits, spawn_target, weight=w_spawn, reduction="sum",
        ) / w_spawn.sum().clamp_min(1.0)

        # Temporal smoothness on the scaled Δp trajectory.
        if F_ > 1:
            d_pred_scaled = delta_pred * s_inv                          # (B, F, N, 3)
            jerk = (d_pred_scaled[:, 1:] - d_pred_scaled[:, :-1]).pow(2).sum(dim=-1).sqrt()
            wsm = vis_f[:, 1:] * vis_f[:, :-1] * qm[:, 1:]
            smooth_loss = (jerk * wsm).sum() / wsm.sum().clamp_min(1.0)
        else:
            smooth_loss = torch.zeros((), device=device)

        total = (
            self.w.pos * pos_loss
            + self.w.vis * vis_loss
            + self.w.spawn * spawn_loss
            + self.w.smooth * smooth_loss
        )
        return TrackingLossOutput(
            total=total, pos=pos_loss, vis=vis_loss,
            spawn=spawn_loss, smooth=smooth_loss, scale=s,
        )
