"""Training-free point tracking by chaining SEA-RAFT optical flow.

Each query point is propagated from its anchor frame outward by bilinearly
sampling per-frame dense flow: forward to later frames, backward to earlier
ones. Visibility comes from forward-backward (cycle) consistency — a point
whose forward flow does not invert back onto itself is treated as occluded,
and once occluded it stays occluded along the chain.

All frames share the same flow fields, so every query is propagated in lockstep
with a per-frame "is this point placed yet" mask, rather than looping per query.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _sample(field: Tensor, pts: Tensor, image_size: int) -> Tensor:
    """Bilinearly sample a (1, 2, S, S) field at pts (N, 2) pixel coords -> (N, 2)."""
    N = pts.shape[0]
    grid = (2.0 * pts / (image_size - 1) - 1.0).view(1, N, 1, 2)
    s = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return s.view(2, N).transpose(0, 1)


def _consistent(
    p_from: Tensor, disp: Tensor, back: Tensor, alpha: float, beta: float,
) -> Tensor:
    """Cycle-consistency test for one hop p_from -> p_from+disp -> +back ≈ p_from."""
    err = (disp + back).norm(dim=-1)
    tol = alpha * (disp.norm(dim=-1) + back.norm(dim=-1)) + beta
    return err <= tol


@torch.no_grad()
def track_clip(
    flow_model,
    images: Tensor,        # (F, 3, S, S) float RGB in [0, 255] on flow_model.device
    queries_xy: Tensor,    # (N, 2) pixel coords in the S×S image space
    anchor_t: Tensor,      # (N,) long frame index of each query
    image_size: int,
    fb_alpha: float = 0.05,
    fb_beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return (uv (F, N, 2), vis (F, N) float in {0,1}) on CPU."""
    device = flow_model.device
    F_ = int(images.shape[0])
    N = int(queries_xy.shape[0])
    images = images.to(device)
    uv = torch.zeros(F_, N, 2, device=device)
    vis = torch.zeros(F_, N, dtype=torch.bool, device=device)
    idx = torch.arange(N, device=device)
    anchor_t = anchor_t.to(device).clamp(0, F_ - 1)
    uv[anchor_t, idx] = queries_xy.to(device)
    vis[anchor_t, idx] = True

    if F_ == 1:
        return uv.cpu(), vis.float().cpu()

    # Precompute consecutive forward/backward flow once per clip.
    fwd = [flow_model.flow(images[t : t + 1], images[t + 1 : t + 2])[0] for t in range(F_ - 1)]
    bwd = [flow_model.flow(images[t + 1 : t + 2], images[t : t + 1])[0] for t in range(F_ - 1)]
    fwd = [f.unsqueeze(0) for f in fwd]   # each (1, 2, S, S)
    bwd = [f.unsqueeze(0) for f in bwd]

    # Forward sweep: fill frames after each query's anchor.
    for t in range(F_ - 1):
        m = anchor_t <= t
        if not m.any():
            continue
        d = _sample(fwd[t], uv[t], image_size)            # t -> t+1
        cand = uv[t] + d
        b = _sample(bwd[t], cand, image_size)             # t+1 -> t
        ok = _consistent(uv[t], d, b, fb_alpha, fb_beta)
        uv[t + 1, m] = cand[m]
        vis[t + 1, m] = vis[t, m] & ok[m]

    # Backward sweep: fill frames before each query's anchor.
    for t in range(F_ - 1, 0, -1):
        m = anchor_t >= t
        if not m.any():
            continue
        d = _sample(bwd[t - 1], uv[t], image_size)        # t -> t-1
        cand = uv[t] + d
        f = _sample(fwd[t - 1], cand, image_size)         # t-1 -> t
        ok = _consistent(uv[t], d, f, fb_alpha, fb_beta)
        uv[t - 1, m] = cand[m]
        vis[t - 1, m] = vis[t, m] & ok[m]

    return uv.cpu(), vis.float().cpu()


@torch.no_grad()
def track_clip_with_flow(
    flow_model,
    images: Tensor,        # (F, 3, S, S) float RGB in [0, 255] on flow_model.device
    queries_xy: Tensor,    # (N, 2) pixel coords in the S×S image space
    anchor_t: Tensor,      # (N,) long frame index of each query
    image_size: int,
    fb_alpha: float = 0.05,
    fb_beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (uv (F,N,2), vis (F,N), flow_at_uv (F,N,2)) on CPU.

    flow_at_uv[t] = forward flow sampled at uv[t]; zeros at t = F-1.
    """
    device = flow_model.device
    F_ = int(images.shape[0])
    N = int(queries_xy.shape[0])
    images = images.to(device)
    uv = torch.zeros(F_, N, 2, device=device)
    vis = torch.zeros(F_, N, dtype=torch.bool, device=device)
    flow_at = torch.zeros(F_, N, 2, device=device)
    idx = torch.arange(N, device=device)
    anchor_t = anchor_t.to(device).clamp(0, F_ - 1)
    uv[anchor_t, idx] = queries_xy.to(device)
    vis[anchor_t, idx] = True

    if F_ == 1:
        return uv.cpu(), vis.float().cpu(), flow_at.cpu()

    fwd = [flow_model.flow(images[t : t + 1], images[t + 1 : t + 2])[0] for t in range(F_ - 1)]
    bwd = [flow_model.flow(images[t + 1 : t + 2], images[t : t + 1])[0] for t in range(F_ - 1)]
    fwd = [f.unsqueeze(0) for f in fwd]
    bwd = [f.unsqueeze(0) for f in bwd]

    for t in range(F_ - 1):
        m = anchor_t <= t
        if not m.any():
            continue
        d = _sample(fwd[t], uv[t], image_size)
        cand = uv[t] + d
        b = _sample(bwd[t], cand, image_size)
        ok = _consistent(uv[t], d, b, fb_alpha, fb_beta)
        uv[t + 1, m] = cand[m]
        vis[t + 1, m] = vis[t, m] & ok[m]

    for t in range(F_ - 1, 0, -1):
        m = anchor_t >= t
        if not m.any():
            continue
        d = _sample(bwd[t - 1], uv[t], image_size)
        cand = uv[t] + d
        f = _sample(fwd[t - 1], cand, image_size)
        ok = _consistent(uv[t], d, f, fb_alpha, fb_beta)
        uv[t - 1, m] = cand[m]
        vis[t - 1, m] = vis[t, m] & ok[m]

    for t in range(F_ - 1):
        flow_at[t] = _sample(fwd[t], uv[t], image_size)

    return uv.cpu(), vis.float().cpu(), flow_at.cpu()
