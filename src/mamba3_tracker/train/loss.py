"""Tracking loss with Hungarian matching at the anchor frame.

Match predicted query slots to GT tracks once per clip (using the anchor
frame's GT 3D position), then accumulate four per-frame terms:
  - Position L1 on visible frames
  - Visibility BCE
  - Spawn BCE (positive at the first visible frame only)
  - Temporal smoothness penalty on consecutive visible frames
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
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


def _hungarian_match_anchor(
    pred_xyz_anchor: Tensor,        # (B, N, 3) prediction at frame 0
    gt_xyz_anchor: Tensor,          # (B, M, 3) GT at each query's own anchor frame
    gt_mask: Tensor,                # (B, M) True = real GT track
) -> Tensor:
    """Return (B, M) long: for each GT track m, the predicted slot index it matches.

    Uses anchor-frame L1 distance as cost. Predicted slots not chosen by any GT
    are simply ignored at loss time (they're free to predict "no-object" — but
    we don't currently train them to do so explicitly; the spawn-logit BCE on
    unmatched slots handles that implicitly).
    """
    B, N, _ = pred_xyz_anchor.shape
    M = gt_xyz_anchor.shape[1]
    out = torch.zeros(B, M, dtype=torch.long, device=pred_xyz_anchor.device)
    pred = pred_xyz_anchor.detach().float()
    gt_f = gt_xyz_anchor.float()
    for b in range(B):
        m_keep = gt_mask[b].nonzero(as_tuple=False).flatten()
        if m_keep.numel() == 0:
            continue
        cost = torch.cdist(gt_f[b, m_keep], pred[b], p=1)
        cost_cpu = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_cpu)
        for r, c in zip(row_ind, col_ind):
            out[b, m_keep[r]] = int(c)
    return out


class TrackingLoss(nn.Module):
    def __init__(self, weights: TrackingLossWeights | None = None) -> None:
        super().__init__()
        self.w = weights or TrackingLossWeights()

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,    # (B, F, M, 3)
        gt_visibility: Tensor,    # (B, F, M) bool
        gt_query_mask: Tensor,    # (B, M) bool
        gt_anchor_frame: Tensor,  # (B, M) long — per-track anchor frame index
    ) -> TrackingLossOutput:
        B, F_, M, _ = gt_tracks_XYZ.shape
        device = pred.xyz.device

        # GT position at each track's own anchor frame (gathered from F axis)
        anchor_idx = gt_anchor_frame.clamp(min=0, max=F_ - 1)
        gt_xyz_anchor = gt_tracks_XYZ.gather(
            dim=1, index=anchor_idx.view(B, 1, M, 1).expand(B, 1, M, 3),
        ).squeeze(1)                                              # (B, M, 3)

        # For matching we use frame-0 predictions (the bank state Q^(0)
        # has integrated only one frame and is closest to the anchor regime).
        pred_anchor = pred.xyz[:, 0]                              # (B, N, 3)
        assign = _hungarian_match_anchor(pred_anchor, gt_xyz_anchor, gt_query_mask)
        # `assign[b, m]` = predicted-slot index for GT track m of batch b.

        # Gather predictions onto the GT track indices: shape (B, F, M, *)
        idx_xyz = assign.view(B, 1, M, 1).expand(B, F_, M, 3)
        pred_xyz_matched = pred.xyz.gather(dim=2, index=idx_xyz)
        idx_logit = assign.view(B, 1, M).expand(B, F_, M)
        pred_vis_matched = pred.vis_logits.gather(dim=2, index=idx_logit)
        pred_spawn_matched = pred.spawn_logits.gather(dim=2, index=idx_logit)

        vis_f = gt_visibility.float()                              # (B, F, M)
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, M).float()   # (B, F, M)

        # Position L1 on visible matched pairs
        l1 = (pred_xyz_matched - gt_tracks_XYZ).abs().sum(dim=-1)  # (B, F, M)
        w_pos = vis_f * qm
        pos_loss = (l1 * w_pos).sum() / w_pos.sum().clamp_min(1.0)

        # Visibility BCE over all valid query slots
        vis_loss = F.binary_cross_entropy_with_logits(
            pred_vis_matched, vis_f, weight=qm, reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        # Spawn BCE: positive at the first visible frame for each track
        # spawn_target[b, t, m] = 1 iff t == argmin_{s : vis(b,s,m)=True} s
        first_vis = _first_visible_frame(gt_visibility)            # (B, M)
        t_idx = torch.arange(F_, device=device).view(1, F_, 1).expand(B, F_, M)
        spawn_target = (t_idx == first_vis.unsqueeze(1)).float()
        # Mask to only train spawn on tracks that *ever* appear
        ever_vis = gt_visibility.any(dim=1).float().unsqueeze(1)   # (B, 1, M)
        w_spawn = qm * ever_vis
        spawn_loss = F.binary_cross_entropy_with_logits(
            pred_spawn_matched, spawn_target, weight=w_spawn, reduction="sum",
        ) / w_spawn.sum().clamp_min(1.0)

        # Temporal smoothness on consecutive visible pairs
        if F_ > 1:
            delta = (pred_xyz_matched[:, 1:] - pred_xyz_matched[:, :-1]).pow(2).sum(dim=-1).sqrt()
            wsm = (vis_f[:, 1:] * vis_f[:, :-1] * qm[:, 1:])
            smooth_loss = (delta * wsm).sum() / wsm.sum().clamp_min(1.0)
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
            spawn=spawn_loss, smooth=smooth_loss,
        )


def _first_visible_frame(vis: Tensor) -> Tensor:
    """(B, F, M) bool → (B, M) long: index of first True along F (or F if never)."""
    B, F_, M = vis.shape
    sentinel = torch.full((B, M), F_, dtype=torch.long, device=vis.device)
    if F_ == 0:
        return sentinel
    idx = torch.arange(F_, device=vis.device).view(1, F_, 1).expand(B, F_, M)
    masked_idx = torch.where(vis, idx, torch.full_like(idx, F_))
    return masked_idx.min(dim=1).values
