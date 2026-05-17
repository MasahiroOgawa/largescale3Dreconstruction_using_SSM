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
        # Untrained-network outputs in bf16 can drift to ±inf early in
        # training. Clamp xyz to a generous metric range (TAPVid-3D scenes
        # are typically ≤30 m) and the logits to ±15 so BCEWithLogits stays
        # finite. The clamp is a saturating non-linearity — gradients still
        # flow until the clamp activates, then offending parameters get
        # updated to bring outputs back in range.
        xyz = torch.nan_to_num(self.xyz_head(x), nan=0.0, posinf=100.0, neginf=-100.0)
        xyz = xyz.clamp(-100.0, 100.0)
        vis_logits = torch.nan_to_num(self.vis_head(x).squeeze(-1),
                                       nan=0.0, posinf=15.0, neginf=-15.0).clamp(-15.0, 15.0)
        spawn_logits = torch.nan_to_num(self.spawn_head(x).squeeze(-1),
                                         nan=0.0, posinf=15.0, neginf=-15.0).clamp(-15.0, 15.0)
        return TrackerOutputs(xyz=xyz, vis_logits=vis_logits, spawn_logits=spawn_logits)
