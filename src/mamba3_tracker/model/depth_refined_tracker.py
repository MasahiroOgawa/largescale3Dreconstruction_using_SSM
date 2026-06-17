"""Mamba-3 depth-along-ray refiner (v33).

Lesson from v32: refining the 2D position (a `delta_uv` residual on top of an
already-good SEA-RAFT flow track) *degrades* 3D accuracy — small nudges push
points across DA3 depth discontinuities and out of the AJ threshold band.

v33 obeys the constraint "the SSM only treats 3D positions, never touches the
SEA-RAFT 2D track". With the pixel position `(u, v)` frozen, the only 3D degree
of freedom that leaves the 2D projection invariant is the depth `z` along the
pixel ray:

    xyz = z * ((u - cx)/fx, (v - cy)/fy, 1) = z * (ray_x, ray_y, 1)

So a small causal Mamba-3 SSM ingests the per-track sequence of
`[ray_x, ray_y, z_raw/z_ref, vis]` and emits a multiplicative depth correction
`z = z_raw * exp(Δlog z)`. The reprojection of `xyz` is exactly `(u, v)` for
every frame — the 2D track is mathematically untouched.

The Δlog-z head is zero-initialised, so at step 0 `z = z_raw` and the model
reproduces the training-free SEA-RAFT+DA3 baseline exactly.

Notation follows doc/mamba3_3dpoint_tracking.tex §6.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mamba3_attn.mamba3.cross_attention import Mamba3CrossAttention
from .heads import TrackerOutputs


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class Mamba3DepthRefiner(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_log_correction: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_log_correction = float(max_log_correction)
        # Input: [ray_x, ray_y, z_raw/z_ref, vis] = 4
        self.embed = _mlp(4, dim, dim)
        self.layers = nn.ModuleList([
            Mamba3CrossAttention(
                dim_q=dim, dim_kv=dim, num_heads=num_heads,
                state_dim=state_dim, variant="B", bidirectional_mask=False,
            )
            for _ in range(num_layers)
        ])
        self.pre_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)

        self.dz_head = _mlp(dim, 64, 1)
        # Zero-init: Δlog z = 0 at step 0 → z = z_raw → SEA-RAFT+DA3 baseline.
        with torch.no_grad():
            self.dz_head[-1].weight.zero_()
            self.dz_head[-1].bias.zero_()

    def forward(
        self,
        ray: Tensor,          # (B, F, N, 2)  fixed pixel ray (u-cx)/fx, (v-cy)/fy
        z_raw: Tensor,        # (B, F, N)     DA3 depth sampled at the frozen uv
        vis: Tensor,          # (B, F, N)     SEA-RAFT FB-consistency flag (frozen)
        z_ref: float | None = None,
    ) -> TrackerOutputs:
        B, F_, N, _ = ray.shape
        zr = z_ref if z_ref is not None else float(z_raw.flatten().median().item()) + 1e-6
        feat = torch.cat([
            ray,                              # (B,F,N,2)
            (z_raw / zr).unsqueeze(-1),       # (B,F,N,1)
            vis.unsqueeze(-1),                # (B,F,N,1)
        ], dim=-1)                            # (B,F,N,4)

        x = self.embed(feat)                  # (B,F,N,D)
        x = x.permute(0, 2, 1, 3).reshape(B * N, F_, self.dim)   # (B*N, F, D)
        for pre_n, layer, post_n in zip(self.pre_norms, self.layers, self.post_norms):
            xn = pre_n(x)
            x = post_n(x + layer(xn, xn))
        x = self.out_norm(x)
        x = x.reshape(B, N, F_, self.dim).permute(0, 2, 1, 3)   # (B,F,N,D)

        dlog = self.dz_head(x).squeeze(-1)                       # (B,F,N)
        dlog = dlog.clamp(-self.max_log_correction, self.max_log_correction)
        z_pred = z_raw * torch.exp(dlog)                         # (B,F,N)

        xyz = torch.stack([ray[..., 0] * z_pred, ray[..., 1] * z_pred, z_pred], dim=-1)

        # uv / visibility are frozen SEA-RAFT outputs handled outside the model;
        # echo a zero vis_logits placeholder for TrackerOutputs structural compat.
        vis_logits = x.new_zeros(B, F_, N)
        return TrackerOutputs(
            xyz=xyz, uv=None, vis_logits=vis_logits, spawn_logits=vis_logits,
        )
