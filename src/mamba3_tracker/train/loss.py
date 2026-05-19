"""Tracking loss v6 — Smooth-L1 + direction-cosine + scaled magnitude + 2D reproj.

v5's L1-only position loss let the model collapse to a "predict near-zero
motion" baseline: scale-normalised |Δp̂ − Δp*|/s gives the same constant
gradient ±1/s whether the residual is 1 cm or 1 m, so the optimiser had
no reason to push past the small-motion floor. v6 adds three terms that
attack that baseline:

  * Smooth-L1 (Huber, δ=1) on the scaled position residual — quadratic
    near zero, linear far away. Penalises *under*-prediction harder than
    plain L1 because the gradient grows with the residual.
  * **Direction cosine**: `1 − cos(Δp̂, Δp*)` per visible (t, n). Forces
    the model to get the *direction* of motion right even when its
    magnitude estimate is shaky.
  * **Scaled magnitude penalty**: `((‖Δp̂‖ − ‖Δp*‖) / s)²`. Tells the
    model "GT moved 10 % of the scene, you must also move ~10 %". The
    /s normalisation keeps it scale-invariant — monocular RGB can't
    recover absolute metres anyway.
  * **2D reprojection loss**: project the predicted absolute 3D position
    `p̂ = p_q^* + Δp̂` through the per-clip pinhole intrinsics `K` to
    pixel coords `(û, v̂)`; compare to the GT pixel coords
    `(u*, v*) = project(p*, K)`. Smooth-L1 in pixel space, normalised
    by `image_size` to ~[0, 1] range. Pixel-space supervision is the
    geometric signal that's natively in-distribution (queries are 2-D
    pixel locations after all) and has no scale ambiguity.

`doc/attention/mamba3_attention.tex §8.6` updated this commit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.heads import TrackerOutputs


@dataclass
class TrackingLossWeights:
    pos: float = 1.0       # Smooth-L1 on scaled relative position
    mag: float = 0.5       # scaled magnitude penalty
    dir: float = 0.5       # direction cosine
    reproj: float = 1.0    # 2D pixel-space reprojection
    vis: float = 0.5
    spawn: float = 0.5
    smooth: float = 0.1


@dataclass
class TrackingLossOutput:
    total: Tensor
    pos: Tensor
    mag: Tensor
    dir: Tensor
    reproj: Tensor
    vis: Tensor
    spawn: Tensor
    smooth: Tensor
    scale: Tensor          # per-clip median scale used (B,) — for logging


def _per_clip_median_scale(
    gt_anchor_xyz: Tensor,    # (B, N, 3)
    query_mask: Tensor,       # (B, N) bool
    eps: float = 1e-3,
) -> Tensor:
    B = gt_anchor_xyz.shape[0]
    norms = gt_anchor_xyz.norm(dim=-1)
    s = torch.full((B,), eps, device=gt_anchor_xyz.device, dtype=gt_anchor_xyz.dtype)
    for b in range(B):
        keep = query_mask[b]
        if keep.any():
            s[b] = norms[b, keep].median()
    return s.clamp_min(eps)


def _first_visible_frame(vis: Tensor) -> Tensor:
    B, F_, N = vis.shape
    if F_ == 0:
        return torch.full((B, N), F_, dtype=torch.long, device=vis.device)
    idx = torch.arange(F_, device=vis.device).view(1, F_, 1).expand(B, F_, N)
    masked_idx = torch.where(vis, idx, torch.full_like(idx, F_))
    return masked_idx.min(dim=1).values


def _project(xyz: Tensor, K: Tensor) -> Tensor:
    """Pinhole projection of batched 3-D points.

    Args:
        xyz: (B, F, N, 3) world coordinates (or any leading shape (B, ...)).
        K:   (B, 3, 3) per-clip intrinsics.
    Returns:
        uv: (B, F, N, 2) pixel coordinates.
    """
    # Pull scalars per-batch, then broadcast over the trailing (F, N, …) dims.
    extra_dims = xyz.dim() - 2          # number of dims to broadcast over (F, N, ...)
    view = (-1,) + (1,) * extra_dims
    fx = K[:, 0, 0].view(view)
    fy = K[:, 1, 1].view(view)
    cx = K[:, 0, 2].view(view)
    cy = K[:, 1, 2].view(view)
    Z = xyz[..., 2].clamp_min(1e-6)
    u = (xyz[..., 0] / Z) * fx + cx
    v = (xyz[..., 1] / Z) * fy + cy
    return torch.stack([u, v], dim=-1)


class TrackingLoss(nn.Module):
    def __init__(
        self,
        weights: TrackingLossWeights | None = None,
        image_size: int = 448,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.w = weights or TrackingLossWeights()
        self.image_size = image_size
        self.smooth_l1_beta = smooth_l1_beta

    def _smooth_l1(self, x: Tensor) -> Tensor:
        return F.smooth_l1_loss(x, torch.zeros_like(x),
                                reduction="none", beta=self.smooth_l1_beta)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,    # (B, F, N, 3)
        gt_visibility: Tensor,    # (B, F, N) bool
        gt_query_mask: Tensor,    # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long
        K: Tensor,                # (B, 3, 3) per-clip intrinsics
    ) -> TrackingLossOutput:
        B, F_, N, _ = gt_tracks_XYZ.shape
        device = pred.xyz.device

        # GT anchor 3-D positions: p^*_n^(t_n^q)
        anchor_idx = gt_anchor_frame.clamp(min=0, max=F_ - 1)
        gt_anchor_xyz = gt_tracks_XYZ.gather(
            dim=1, index=anchor_idx.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)                                                   # (B, N, 3)

        s = _per_clip_median_scale(gt_anchor_xyz, gt_query_mask)        # (B,)
        s_inv = (1.0 / s).view(B, 1, 1, 1)

        delta_gt = gt_tracks_XYZ - gt_anchor_xyz.unsqueeze(1)           # (B, F, N, 3)
        delta_pred = pred.xyz                                            # (B, F, N, 3)
        r = (delta_pred - delta_gt) * s_inv                              # (B, F, N, 3)

        vis_f = gt_visibility.float()                                    # (B, F, N)
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()         # (B, F, N)
        w_pos = vis_f * qm                                               # (B, F, N)
        w_pos_sum = w_pos.sum().clamp_min(1.0)

        # 1. Smooth-L1 on scaled position residual
        sl1 = self._smooth_l1(r).sum(dim=-1)                             # (B, F, N)
        pos_loss = (sl1 * w_pos).sum() / w_pos_sum

        # 2. Scaled magnitude penalty
        mag_pred = delta_pred.norm(dim=-1)                               # (B, F, N)
        mag_gt = delta_gt.norm(dim=-1)
        mag_diff = (mag_pred - mag_gt) * s_inv.squeeze(-1)
        mag_loss = ((mag_diff.pow(2)) * w_pos).sum() / w_pos_sum

        # 3. Direction cosine: 1 − cos(Δp̂, Δp*). Stable form with eps.
        eps = 1e-6
        cos_num = (delta_pred * delta_gt).sum(dim=-1)
        cos_den = mag_pred.clamp_min(eps) * mag_gt.clamp_min(eps)
        cos_sim = (cos_num / cos_den).clamp(-1.0, 1.0)
        dir_loss = ((1.0 - cos_sim) * w_pos).sum() / w_pos_sum

        # 4. 2-D reprojection. Project p̂ = p_q + Δp̂ and p* through K.
        p_pred = delta_pred + gt_anchor_xyz.unsqueeze(1)                 # (B, F, N, 3)
        p_gt = gt_tracks_XYZ                                              # (B, F, N, 3)
        uv_pred = _project(p_pred, K)
        uv_gt = _project(p_gt, K)
        uv_diff = (uv_pred - uv_gt) / float(self.image_size)
        reproj_sl1 = self._smooth_l1(uv_diff).sum(dim=-1)                # (B, F, N)
        # Also keep finite if Z went very small / negative
        reproj_finite = torch.isfinite(reproj_sl1).float()
        w_reproj = w_pos * reproj_finite
        reproj_loss = (torch.nan_to_num(reproj_sl1, nan=0.0, posinf=0.0, neginf=0.0)
                       * w_reproj).sum() / w_reproj.sum().clamp_min(1.0)

        # 5. Visibility BCE
        vis_loss = F.binary_cross_entropy_with_logits(
            pred.vis_logits, vis_f, weight=qm, reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        # 6. Spawn BCE on first-visible frame
        first_vis = _first_visible_frame(gt_visibility)
        t_idx = torch.arange(F_, device=device).view(1, F_, 1).expand(B, F_, N)
        spawn_target = (t_idx == first_vis.unsqueeze(1)).float()
        ever_vis = gt_visibility.any(dim=1).float().unsqueeze(1)
        w_spawn = qm * ever_vis
        spawn_loss = F.binary_cross_entropy_with_logits(
            pred.spawn_logits, spawn_target, weight=w_spawn, reduction="sum",
        ) / w_spawn.sum().clamp_min(1.0)

        # 7. Smoothness on scaled Δp trajectory
        if F_ > 1:
            d_pred_scaled = delta_pred * s_inv
            jerk = (d_pred_scaled[:, 1:] - d_pred_scaled[:, :-1]).pow(2).sum(dim=-1).sqrt()
            wsm = vis_f[:, 1:] * vis_f[:, :-1] * qm[:, 1:]
            smooth_loss = (jerk * wsm).sum() / wsm.sum().clamp_min(1.0)
        else:
            smooth_loss = torch.zeros((), device=device)

        total = (
            self.w.pos    * pos_loss
            + self.w.mag    * mag_loss
            + self.w.dir    * dir_loss
            + self.w.reproj * reproj_loss
            + self.w.vis    * vis_loss
            + self.w.spawn  * spawn_loss
            + self.w.smooth * smooth_loss
        )
        return TrackingLossOutput(
            total=total, pos=pos_loss, mag=mag_loss, dir=dir_loss,
            reproj=reproj_loss, vis=vis_loss, spawn=spawn_loss,
            smooth=smooth_loss, scale=s,
        )
