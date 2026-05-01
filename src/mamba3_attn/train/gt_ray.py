"""Build GT ray map from camera params for L_M supervision.

Per DA3 paper §3.1: per-pixel ray r ∈ R^6 = (t, d) where
  - t = camera origin in world frame (3-vector)
  - d = R_c2w @ K^-1 @ p (3-vector, magnitude-preserved)

For the model's ray output at resolution (h, w) and intrinsic K_h scaled
to that resolution, we build the dense ray map:
  M(u, v, :3) = t          (broadcast across all pixels)
  M(u, v, 3:) = d(u, v)
"""

from __future__ import annotations

import torch
from torch import Tensor


def gt_ray_map(
    gt_K: Tensor,        # (B, S, 3, 3) at original image resolution
    gt_w2c: Tensor,      # (B, S, 4, 4)
    image_hw: tuple[int, int],   # (H, W) of input images that K is calibrated for
    out_hw: tuple[int, int],     # (h, w) of ray map output
) -> Tensor:
    """Returns ray map (B, S, h, w, 6) [origin | direction] in world frame."""
    B, S = gt_K.shape[:2]
    H, W = image_hw
    h, w = out_hw

    sx = w / W
    sy = h / H
    K = gt_K.clone().float()
    K[..., 0, 0] *= sx; K[..., 0, 2] *= sx
    K[..., 1, 1] *= sy; K[..., 1, 2] *= sy

    R_w2c = gt_w2c[..., :3, :3].float()
    t_w2c = gt_w2c[..., :3, 3].float()
    R_c2w = R_w2c.transpose(-1, -2)
    t_c2w = -(R_c2w @ t_w2c.unsqueeze(-1)).squeeze(-1)  # (B, S, 3)

    device = gt_K.device
    v_grid, u_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u_grid = u_grid + 0.5
    v_grid = v_grid + 0.5
    fx = K[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = K[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

    x_n = (u_grid - cx) / fx
    y_n = (v_grid - cy) / fy
    z_n = torch.ones_like(x_n)
    p_cam = torch.stack([x_n, y_n, z_n], dim=-1)  # (B, S, h, w, 3)

    # d = R_c2w @ p_cam (per pixel)
    d = (R_c2w.unsqueeze(-3).unsqueeze(-3) @ p_cam.unsqueeze(-1)).squeeze(-1)

    # Origin t broadcast across pixels
    t = t_c2w.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, h, w, -1)

    return torch.cat([t, d], dim=-1)
