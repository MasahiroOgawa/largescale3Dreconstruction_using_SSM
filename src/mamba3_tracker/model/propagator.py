"""Causal cross-frame propagator with coarse-to-fine pyramid refinement.

The persistent query bank `Q ∈ ℝ^(B, N, D)` represents tracked-point
identities. At each frame `t` it is refined coarse-to-fine across the
encoder's pyramid levels: a cross-SSD block at each level reads from that
level's frame-t token grid and adds a residual update to Q.

Temporal causality is enforced by construction — the loop processes frames
in order and only ever consumes the current frame's features. Track
information from earlier frames is carried implicitly by the residual
accumulation in `Q`, which serves as the Mamba-3 hidden state of §8.3 of
`doc/attention/mamba3_attention.tex`.

Output: per-(frame, track) feature `Q^(t) ∈ ℝ^(B, F, N, D)`.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from mamba3_attn.mamba3.cross_attention import Mamba3CrossAttention


def _pyramid_to_flat_kv(level: Tensor) -> Tensor:
    """(B, D, h, w) → (B, h*w, D)."""
    B, D, h, w = level.shape
    return level.flatten(2).transpose(1, 2)


class CausalCrossPropagator(nn.Module):
    """Persistent query bank refined coarse-to-fine across pyramid levels.

    For each frame `t` and each pyramid level `l` (coarse → fine), the update is
        Q ← Q + CrossAttn_l( LN(Q),  LN(Y^(t)_l) ).
    After the finest level, that frame's bank state Q is stored as Q^(t).
    """

    def __init__(
        self,
        dim: int = 384,
        num_tracks: int = 512,
        num_pyramid_levels: int = 3,
        num_heads: int = 6,
        state_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_pyramid_levels < 1:
            raise ValueError("need at least 1 pyramid level")
        self.dim = dim
        self.num_tracks = num_tracks
        self.num_pyramid_levels = num_pyramid_levels

        self.bank_init = nn.Parameter(torch.randn(num_tracks, dim) * 0.02)
        # Variant A (state-compressed): kv is collapsed to a single
        # (H, N_state, head_dim) summary before the per-query read-out, so
        # memory is independent of T_kv. Crucial for the pyramid: the finest
        # level has T_kv = h·w which grows quadratically with resolution.
        self.cross_levels = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=dim, dim_kv=dim, num_heads=num_heads,
                    state_dim=state_dim, variant="A",
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
        initial_bank: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            pyramid: list of `num_pyramid_levels` tensors, each (B, F, D, h_l, w_l).
                Ordered coarse → fine.
            initial_bank: optional (B, N, D) bank state to start from (for
                streaming inference across multiple windows). Defaults to
                `bank_init` broadcast to batch size.

        Returns:
            Q_history: (B, F, N, D) — per-(frame, track) refined features.
        """
        if len(pyramid) != self.num_pyramid_levels:
            raise ValueError(
                f"pyramid has {len(pyramid)} levels, "
                f"propagator was built for {self.num_pyramid_levels}"
            )

        B = pyramid[0].shape[0]
        F_ = pyramid[0].shape[1]

        if initial_bank is None:
            Q = self.bank_init.unsqueeze(0).expand(B, -1, -1).contiguous()
        else:
            Q = initial_bank

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
