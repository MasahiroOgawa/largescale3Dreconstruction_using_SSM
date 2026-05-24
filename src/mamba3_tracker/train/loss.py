"""Tracking loss v11 — cumulative-sum trajectory + scale-normalised squared L2.

User-driven design after v6–v10's static-tracks failure mode was traced to two
sources: (1) the propagator track-memory accumulator blowing up across many
residual adds without LayerNorm (fixed in v10), and (2) the per-(t, n) Huber
loss scale producing a non-stationary cost surface that made the loss spiky
and non-monotone in training-step space.

v11 keeps the loss's *shape in residual space* simple and smooth — a parabola
in the prediction variable — and replaces the offset-from-anchor prediction
with a cumulative integration of per-frame motion.

Reconstruction:

    Δp̂(t, n) = pred.xyz[..., t, n, :]    model output per frame
    Δp̂(0, n) is IGNORED (the first frame's emitted value is discarded).
    p̂(0, n) = p*(0, n)                    initial position from GT
    p̂(t, n) = p̂(0, n) + Σ_{s=1..t} Δp̂(s, n)    for t ≥ 1

Per-clip scale (training only — eval uses TAPVid-3D's median scaling):

    s_3D = median over visible (t, n) of  ‖p*(t, n)‖₂          [m]
    s_2D = image_size                                            [px]

Three loss terms — all squared L2 in scale-normalised space:

    L_3D(t, n) = ‖( p̂(t, n) − p*(t, n) ) / s_3D‖²
    L_2D(t, n) = ‖( π(p̂(t, n), K) − π(p*(t, n), K) ) / s_2D‖²
    L_vis      = BCEWithLogits(pred.vis_logits, gt_visibility)

Means over visible (t, n) and over the batch. Weighted sum with
normalised weights `Σ λ_i = 1` from `configs/v11.yaml`.

No velocity term. No smoothness term. No threshold. No Huber per-(t, n)
clamp. Three terms, three numbers, smooth U-shape in the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.heads import TrackerOutputs


_TERMS = ("pos_3D", "pos_2D", "vis")


@dataclass
class TrackingLossOutput:
    total: Tensor
    pos_3D: Tensor
    pos_2D: Tensor
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


def reconstruct_trajectory(
    delta_pred: Tensor,
    init_xyz: Tensor,
    anchor_idx: Tensor,
) -> Tensor:
    """Bidirectional cumsum reconstruction anchored at each track's anchor frame.

    For each track n with anchor frame `a_n = anchor_idx[b, n]`:
        p̂(a_n, n) = init_xyz[b, n]
        p̂(t, n)   = init_xyz[b, n] + Σ_{s = a_n+1 .. t} Δp̂(s, n)       for t > a_n
        p̂(t, n)   = init_xyz[b, n] − Σ_{s = t+1   .. a_n} Δp̂(s, n)     for t < a_n

    Implemented as:
        cs[t] = Σ_{s = 0..t} Δp̂(s)               (cumsum with Δp̂(0) := 0)
        p̂(t) = init_xyz + (cs[t] − cs[a_n])      gather + subtract

    Args:
        delta_pred: (B, F, N, 3) — model's per-frame `Δp̂(t, n)`. The
            entry at t=0 is ignored (Δp̂(0) := 0 inside the cumsum).
        init_xyz: (B, N, 3) — initial 3-D position at each track's
            anchor frame `a_n` (v12 uses GT at `tracks_XYZ[a_n, n]`;
            future = feature-detector output).
        anchor_idx: (B, N) long — each track's anchor frame index in
            window coords. Clamped to [0, F-1].
    Returns:
        p_hat: (B, F, N, 3) with `p̂(a_n) = init_xyz` exactly.
    """
    B, F_, N, _ = delta_pred.shape
    if F_ == 0:
        return delta_pred
    if F_ == 1:
        return init_xyz.unsqueeze(1)
    # Zero out the t=0 entry so cumsum starts effectively at t=1.
    delta_zero_init = delta_pred.clone()
    delta_zero_init[:, 0] = 0.0
    cs = delta_zero_init.cumsum(dim=1)                      # (B, F, N, 3)
    a = anchor_idx.clamp(min=0, max=F_ - 1)                 # (B, N)
    cs_at_anchor = cs.gather(                               # (B, N, 3)
        dim=1, index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
    ).squeeze(1)
    p_hat = init_xyz.unsqueeze(1) + cs - cs_at_anchor.unsqueeze(1)
    return p_hat


def _per_clip_median_scale(
    gt_tracks_XYZ: Tensor,    # (B, F, N, 3)
    visible: Tensor,          # (B, F, N) bool/float
    eps: float = 1e-3,
) -> Tensor:
    """`s_3D = median(‖p*‖)` over visible (t, n), one scalar per clip."""
    B = gt_tracks_XYZ.shape[0]
    norms = gt_tracks_XYZ.norm(dim=-1)                # (B, F, N)
    s = torch.full((B,), eps, device=gt_tracks_XYZ.device, dtype=gt_tracks_XYZ.dtype)
    for b in range(B):
        v = visible[b].bool()
        if v.any():
            s[b] = norms[b][v].median()
    return s.clamp_min(eps)


def reconstruct_trajectory_v18(
    delta_tilde: Tensor,
    scale: Tensor,
    init_xyz: Tensor,
    anchor_idx: Tensor,
) -> Tensor:
    """v18 reconstruction (anchor-based with learnable per-clip scale).

    Same shape as v11–v17 — each track anchored at its own query
    frame and integrated bidirectionally — but each per-frame delta
    is multiplied by the model-predicted per-clip scalar `s` before
    accumulating:

        Δp̂(t, n) = s · Δp̃(t, n)
        p̂(a_n, n) = init_xyz[b, n]   (= GT at the anchor frame)
        p̂(t, n)   = init_xyz + (cs[t] − cs[a_n])     for t ≠ a_n
        cs[t]     = Σ_{τ=0..t}  Δp̂(τ, n)   with Δp̂(0) := 0

    Anchoring at GT keeps the predicted Z in a sensible (positive,
    scene-depth) range from step 0 — no projection blow-ups in the
    2D loss during early training, unlike pure-cumsum-from-origin.

    Args:
        delta_tilde: (B, F, N, 3) — raw model output, pre-scale.
        scale:       (B,)        — per-clip positive scalar from ScaleHead.
        init_xyz:    (B, N, 3)   — GT position at each track's anchor frame.
        anchor_idx:  (B, N) long — each track's anchor frame index.
    Returns:
        p_hat: (B, F, N, 3) absolute trajectory in world units.
    """
    s = scale.to(delta_tilde.dtype).view(-1, 1, 1, 1)
    return reconstruct_trajectory(s * delta_tilde, init_xyz, anchor_idx)


def _gt_path_length_from_anchor(
    gt_tracks_XYZ: Tensor,    # (B, F, N, 3)
    anchor_idx: Tensor,       # (B, N) long
    eps: float = 1e-3,
) -> Tensor:
    """Per-(t, n) cumulative 3D path length of GT between anchor a_n and t.

        cs[t]      = Σ_{τ=1..t} ‖p*(τ) − p*(τ−1)‖
        s_gt(t,n)  = |cs[t]  −  cs[a_n]|                       (path between)

    Symmetric with the anchor-based reconstruction: at t = a_n the
    GT motion budget is 0 (and the predicted residual is exactly 0
    by construction); growing in either temporal direction matches
    how far GT has actually moved. Floored at `eps` for safety;
    detached so no gradient flows through GT magnitudes.

    Returns: (B, F, N) positive scalars.
    """
    B, F_, N, _ = gt_tracks_XYZ.shape
    delta_gt = torch.zeros_like(gt_tracks_XYZ)
    if F_ > 1:
        delta_gt[:, 1:] = gt_tracks_XYZ[:, 1:] - gt_tracks_XYZ[:, :-1]
    step_norms = delta_gt.norm(dim=-1)                                       # (B, F, N), [:, 0] = 0
    cs = step_norms.cumsum(dim=1)                                            # (B, F, N)
    a = anchor_idx.clamp(min=0, max=F_ - 1).long()                           # (B, N)
    cs_at_anchor = cs.gather(
        dim=1, index=a.unsqueeze(1).expand(B, 1, N),
    ).squeeze(1)                                                             # (B, N)
    s_gt = (cs - cs_at_anchor.unsqueeze(1)).abs()                            # (B, F, N)
    return s_gt.clamp_min(eps).detach()


class TrackingLossV18(nn.Module):
    """v18 loss: anchor-based reconstruction with a learnable per-clip
    scale, squared L² in 3D normalised by GT path length from anchor,
    plus v11–v17 style 2D term and visibility BCE.

    Reconstruction (`reconstruct_trajectory_v18`):
        p̂(a_n, n) = p*_anchor(n)                                   (GT injected at anchor)
        p̂(t, n)   = p*_anchor + s · (cs[t] − cs[a_n])               (per-anchor cumsum)

    3D term (the v18 change vs v11–v17):

        L_3D(t,n) = ((p̂(t,n) − p*(t,n)) / s_gt(t,n))²   summed over xyz

    where `s_gt(t,n)` is the cumulative GT 3D path length BETWEEN
    the anchor and t (see `_gt_path_length_from_anchor`). Removes
    per-clip scale dependence (drivetrack ~ 30 m vs pstudio ~ 1 m)
    and structurally penalises the "predict static at anchor"
    collapse: `Δp̂ ≈ 0` gives residual = GT motion from anchor,
    s_gt = GT path length from anchor → normalised ≈ 1 per (t,n).

    2D term (unchanged from v11–v17):

        L_2D(t,n) = ((π(p̂) − π(p*)) / image_size)²   summed over uv

    `image_size` is the encoder input side length (architectural
    constant). No startup projection blow-ups because the GT anchor
    keeps `p̂_z ≈ scene_depth` from step 0.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        image_size: int = 224,
    ) -> None:
        super().__init__()
        missing = set(_TERMS) - set(weights)
        if missing:
            raise ValueError(f"TrackingLossV18: missing weights for {sorted(missing)}")
        self.w = {k: float(weights[k]) for k in _TERMS}
        self.image_size = float(image_size)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,    # (B, F, N, 3)
        gt_visibility: Tensor,    # (B, F, N) bool
        gt_query_mask: Tensor,    # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long
        K: Tensor,                # (B, 3, 3)
    ) -> TrackingLossOutput:
        if pred.scale is None:
            raise RuntimeError("TrackingLossV18 requires pred.scale; set model.predict_scale=True")

        B, F_, N, _ = gt_tracks_XYZ.shape
        a = gt_anchor_frame.clamp(min=0, max=F_ - 1).long()                  # (B, N)
        init_xyz = gt_tracks_XYZ.gather(
            dim=1, index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)                                                         # (B, N, 3)
        p_hat = reconstruct_trajectory_v18(pred.xyz, pred.scale, init_xyz, a)  # (B, F, N, 3)
        s_gt = _gt_path_length_from_anchor(gt_tracks_XYZ, a)                 # (B, F, N)

        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm
        denom = w_pos.sum().clamp_min(1.0)

        r_3D = (p_hat - gt_tracks_XYZ) / s_gt.unsqueeze(-1)                  # (B, F, N, 3)
        pos_3D = ((r_3D.pow(2).sum(dim=-1)) * w_pos).sum() / denom

        uv_hat = _project(p_hat, K)
        uv_gt = _project(gt_tracks_XYZ, K)
        r_2D = (uv_hat - uv_gt) / self.image_size
        finite_2D = torch.isfinite(r_2D).all(dim=-1).float()
        w_2D = w_pos * finite_2D
        denom_2D = w_2D.sum().clamp_min(1.0)
        pos_2D = (torch.nan_to_num(r_2D.pow(2).sum(dim=-1), nan=0.0,
                                   posinf=0.0, neginf=0.0) * w_2D).sum() / denom_2D

        vis_loss = F.binary_cross_entropy_with_logits(
            pred.vis_logits, vis_f, weight=qm, reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        total = (
            self.w["pos_3D"] * pos_3D
            + self.w["pos_2D"] * pos_2D
            + self.w["vis"]    * vis_loss
        )
        return TrackingLossOutput(
            total=total, pos_3D=pos_3D, pos_2D=pos_2D, vis=vis_loss,
        )


class TrackingLoss(nn.Module):
    """v11 loss. Construct with normalised weights from `train/config.py`."""

    def __init__(
        self,
        weights: Mapping[str, float],
        image_size: int = 224,
    ) -> None:
        super().__init__()
        missing = set(_TERMS) - set(weights)
        if missing:
            raise ValueError(f"TrackingLoss: missing weights for {sorted(missing)}")
        self.w = {k: float(weights[k]) for k in _TERMS}
        self.image_size = float(image_size)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,    # (B, F, N, 3)
        gt_visibility: Tensor,    # (B, F, N) bool
        gt_query_mask: Tensor,    # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long — kept in signature for backwards compat (unused in v11)
        K: Tensor,                # (B, 3, 3)
    ) -> TrackingLossOutput:
        B, F_, N, _ = gt_tracks_XYZ.shape

        # v12: each track is anchored at its own query frame `gt_anchor_frame[b, n]`,
        # not at clip-frame-0. p̂(a_n, n) = GT[a_n, n]; trajectory integrated
        # bidirectionally from there. This matches the TAPVid-3D evaluation
        # convention where queries can be at any frame and all visible frames
        # are scored. See `reconstruct_trajectory` for the cumsum-gather math.
        a = gt_anchor_frame.clamp(min=0, max=F_ - 1).long()                  # (B, N)
        init_xyz = gt_tracks_XYZ.gather(
            dim=1, index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)                                                          # (B, N, 3)
        p_hat = reconstruct_trajectory(pred.xyz, init_xyz, a)                 # (B, F, N, 3)

        # Mask: visible (t, n) AND real query slot.
        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm                                                  # (B, F, N)
        denom = w_pos.sum().clamp_min(1.0)

        # Per-clip 3-D scale (detached so no gradient flows back through GT magnitudes).
        s_3D = _per_clip_median_scale(gt_tracks_XYZ, w_pos).detach()        # (B,)
        s_3D_b = s_3D.view(B, 1, 1, 1)
        s_2D_b = self.image_size

        # L_3D — scale-normalised squared L2 in 3D.
        r_3D = (p_hat - gt_tracks_XYZ) / s_3D_b                             # (B, F, N, 3)
        pos_3D = ((r_3D.pow(2).sum(dim=-1)) * w_pos).sum() / denom

        # L_2D — pinhole-project both predicted and GT, then scale-normalised
        # squared L2 in pixel space.
        uv_hat = _project(p_hat, K)                                          # (B, F, N, 2)
        uv_gt = _project(gt_tracks_XYZ, K)                                   # (B, F, N, 2)
        r_2D = (uv_hat - uv_gt) / s_2D_b                                     # (B, F, N, 2)
        # Mask out non-finite projections (rare: Z near zero — clamp handles it,
        # but be defensive).
        finite_2D = torch.isfinite(r_2D).all(dim=-1).float()
        w_2D = w_pos * finite_2D
        denom_2D = w_2D.sum().clamp_min(1.0)
        pos_2D = ((torch.nan_to_num(r_2D.pow(2).sum(dim=-1), nan=0.0,
                                    posinf=0.0, neginf=0.0)) * w_2D).sum() / denom_2D

        # Visibility BCE — masked by qm (per-frame target weighted by query slot).
        vis_loss = F.binary_cross_entropy_with_logits(
            pred.vis_logits, vis_f, weight=qm, reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        total = (
            self.w["pos_3D"] * pos_3D
            + self.w["pos_2D"] * pos_2D
            + self.w["vis"]    * vis_loss
        )
        return TrackingLossOutput(
            total=total, pos_3D=pos_3D, pos_2D=pos_2D, vis=vis_loss,
        )
