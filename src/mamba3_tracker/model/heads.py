"""Per-track read-out heads.

Each updated query feature q_n^(t) ∈ ℝ^D passes through three small MLPs
producing one prediction per (frame, track):
  - xyz_head:   D → 128 → 3      (3D position)
  - vis_head:   D → 128 → 1      (visibility, sigmoid)
  - spawn_head: D → 128 → 1      (spawn confidence, sigmoid)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TrackerOutputs:
    xyz: Tensor          # (B, F, N, 3)
    vis_logits: Tensor   # (B, F, N)
    spawn_logits: Tensor # (B, F, N)


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class TrackHeads(nn.Module):
    def __init__(self, dim: int = 384, hidden: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.xyz_head = _mlp(dim, hidden, 3)
        self.vis_head = _mlp(dim, hidden, 1)
        self.spawn_head = _mlp(dim, hidden, 1)

    def forward(self, q_history: Tensor) -> TrackerOutputs:
        """
        Args:
            q_history: (B, F, N, D)

        Returns:
            TrackerOutputs with xyz (B,F,N,3) and logits (B,F,N) each.
        """
        x = self.norm(q_history)
        # No nan_to_num here — a previous attempt at "safety clamping" silently
        # masked an upstream bf16 overflow and produced NaN parameters in every
        # checkpoint from step ~250 onward. We'd rather a NaN crash the
        # training immediately than corrupt 20k steps' worth of weights without
        # warning. The training script's grad-NaN guard handles the
        # numerical-instability case at the right layer.
        return TrackerOutputs(
            xyz=self.xyz_head(x),
            vis_logits=self.vis_head(x).squeeze(-1),
            spawn_logits=self.spawn_head(x).squeeze(-1),
        )
