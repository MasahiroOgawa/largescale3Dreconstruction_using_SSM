"""Causal cross-frame propagator with query-conditioned bank initialisation.

Each track slot `n` is bound at construction time to a user-supplied
query `(x_n^q, y_n^q, t_n^q)`: a pixel location and an anchor frame
index. The initial bank state `q_n^(0)` is the encoder's finest
pyramid feature bilinear-sampled at that pixel and frame
(`doc/attention/mamba3_attention.tex §8.3, eq. eq:track-bank-init`).
This pins slot identity to a specific physical point, removing the
slot-flipping problem of a free-floating learnable bank.

For each subsequent frame `t = 1..F-1`, the bank is refined
coarse-to-fine across pyramid levels: a cross-SSD block at each level
reads from that level's frame-t token grid and adds a residual update
to `Q`. Per-frame loop carries temporal causality implicitly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mamba3_attn.mamba3.cross_attention import Mamba3CrossAttention


def _pyramid_to_flat_kv(level: Tensor) -> Tensor:
    """(B, D, h, w) → (B, h*w, D)."""
    return level.flatten(2).transpose(1, 2)


def _sample_query_features(
    pyramid_finest: Tensor,    # (B, F, D, h, w) — the finest pyramid level
    queries_xyt: Tensor,       # (B, N, 3) — (x, y, t) in input-image coords
    query_mask: Tensor,        # (B, N) bool
    image_size: int,
) -> Tensor:
    """Bilinear-sample the finest pyramid level at each query's (x, y, t).

    Returns (B, N, D). Masked slots (`query_mask=False`) are returned as zeros.
    Out-of-bound x/y are clamped to grid edges.
    """
    B, F_, D, h, w = pyramid_finest.shape
    N = queries_xyt.shape[1]
    device = pyramid_finest.device

    # Scale (x, y) in pixel coords → feature-grid coords in [0, 1] then
    # to torch.grid_sample's normalized [-1, 1].
    px = queries_xyt[..., 0].clamp(0, image_size - 1) / max(image_size - 1, 1)  # (B, N)
    py = queries_xyt[..., 1].clamp(0, image_size - 1) / max(image_size - 1, 1)
    gx = px * 2.0 - 1.0
    gy = py * 2.0 - 1.0
    # grid_sample expects (..., 2) with (x, y) last
    grid_xy = torch.stack([gx, gy], dim=-1)            # (B, N, 2)

    out = torch.zeros(B, N, D, device=device, dtype=pyramid_finest.dtype)
    for b in range(B):
        n_keep = query_mask[b].nonzero(as_tuple=False).flatten()
        if n_keep.numel() == 0:
            continue
        t_idx = queries_xyt[b, n_keep, 2].long().clamp(0, F_ - 1)   # (N_b,)
        # Group by frame to avoid one grid_sample per query
        for t_val in torch.unique(t_idx).tolist():
            sel = n_keep[t_idx == t_val]                       # (N_t,)
            if sel.numel() == 0:
                continue
            feat = pyramid_finest[b:b + 1, t_val]              # (1, D, h, w)
            grid = grid_xy[b, sel].view(1, -1, 1, 2)            # (1, N_t, 1, 2)
            sampled = F.grid_sample(
                feat, grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )                                                   # (1, D, N_t, 1)
            out[b, sel] = sampled.squeeze(-1).squeeze(0).transpose(0, 1)
    return out


class CausalCrossPropagator(nn.Module):
    """Query-conditioned bank refined coarse-to-fine across pyramid levels.

    For each frame `t` and each pyramid level `l` (coarse → fine), the update is
        Q ← Q + CrossAttn_l( LN(Q),  LN(Y^(t)_l) ).
    After the finest level, that frame's bank state Q is stored as Q^(t).

    `bank_init` is now derived from the input queries (no longer a learnable
    Parameter), so the slot count `N` is dynamic per batch / clip.
    """

    def __init__(
        self,
        dim: int = 384,
        num_pyramid_levels: int = 3,
        num_heads: int = 6,
        state_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_pyramid_levels < 1:
            raise ValueError("need at least 1 pyramid level")
        self.dim = dim
        self.num_pyramid_levels = num_pyramid_levels

        # Variant B (token-level): the per-(query, kv_patch) inner product
        # form, easier to inspect via attention maps for debugging. Note: in
        # this codebase variants A and B are *mathematically equivalent*
        # because the cross-mask `build_cross_mask` is rank-1 in the query
        # axis (decay depends only on kv index j). The real "model collapses
        # to zero motion" failure that initial debugging blamed on variant A
        # was actually a NaN propagation caused by nan_to_num in the heads
        # corrupting parameters via NaN gradients; see commit message and
        # heads.py for the actual fix. Variant B costs O(T_q · T_kv) memory
        # vs A's O(N_state · D); both are fine at our pyramid sizes (32², 64²).
        self.cross_levels = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=dim, dim_kv=dim, num_heads=num_heads,
                    state_dim=state_dim, variant="B",
                )
                for _ in range(num_pyramid_levels)
            ]
        )
        self.q_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(num_pyramid_levels)]
        )
        self.kv_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(num_pyramid_levels)]
        )

    def forward(
        self,
        pyramid: list[Tensor],
        queries_xyt: Tensor,       # (B, N, 3) — (x, y, t)
        query_mask: Tensor,        # (B, N) bool
        image_size: int,
    ) -> Tensor:
        """
        Args:
            pyramid: list of `num_pyramid_levels` tensors, each (B, F, D, h_l, w_l).
                Ordered coarse → fine.
            queries_xyt: (B, N, 3) per-track queries in pixel coords + anchor frame.
            query_mask: (B, N) True where the slot is a real query (False = padding).
            image_size: side length of the (square) input image — used to map
                pixel coords to feature-grid coords.

        Returns:
            Q_history: (B, F, N, D) — per-(frame, track) refined features.
        """
        if len(pyramid) != self.num_pyramid_levels:
            raise ValueError(
                f"pyramid has {len(pyramid)} levels, "
                f"propagator was built for {self.num_pyramid_levels}"
            )

        F_ = pyramid[0].shape[1]

        # Bank init: sample the finest pyramid at each query's (x, y, t).
        Q = _sample_query_features(
            pyramid[-1], queries_xyt, query_mask, image_size=image_size,
        )

        history: list[Tensor] = []
        for t in range(F_):
            for l, (cross, norm_q, norm_kv) in enumerate(
                zip(self.cross_levels, self.q_norms, self.kv_norms)
            ):
                kv_grid = pyramid[l][:, t]                  # (B, D, h_l, w_l)
                kv_tokens = norm_kv(_pyramid_to_flat_kv(kv_grid))   # (B, h*w, D)
                q_tokens = norm_q(Q)                         # (B, N, D)
                delta = cross(q_tokens, kv_tokens)           # (B, N, D)
                Q = Q + delta
            history.append(Q)

        return torch.stack(history, dim=1)
