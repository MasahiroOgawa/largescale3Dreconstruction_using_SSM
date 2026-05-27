"""Per-track read-out heads.

Each updated query feature q_n^(t) ∈ ℝ^D passes through three small MLPs
producing one prediction per (frame, track):
  - xyz_head:   D → 128 → 3      (3D position)
  - vis_head:   D → 128 → 1      (visibility, sigmoid)
  - spawn_head: D → 128 → 1      (spawn confidence, sigmoid)

v18 adds an optional clip-level `ScaleHead` that produces one positive
scalar `s ∈ ℝ⁺` per clip from pooled per-frame CLS tokens. With v18 loss
the model emits raw deltas Δp̃, and the absolute trajectory is recovered
as `p̂(t,n) = Σ_{τ=0..t} s · Δp̃(τ,n)` — pure cumsum from zero, no GT
anchor. `s` is trained jointly with the rest of the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TrackerOutputs:
    xyz: Tensor          # (B, F, N, 3)
    vis_logits: Tensor   # (B, F, N)
    spawn_logits: Tensor # (B, F, N)
    scale: Tensor | None = None  # (B,) clip-level positive scalar, v18+; None for v11–v17


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


class ScaleHead(nn.Module):
    """Predicts one positive scalar `s` per clip from per-frame CLS tokens.

    Input  cls_per_frame: (B, F, D) — DINO CLS token at each frame
    Output                (B,)      — positive metric scale.

    The MLP emits a raw real number `z` (any value); a positive-mapping
    function turns it into `s`. Two parameterisations:

      * param="softplus" (v18/v19):  s = softplus(z), bias init so s≈1.
        Problem: ∂s/∂z = sigmoid(z) → 0 as s → 0, so once `s` drifts
        toward zero the gradient to climb back out is ~100× attenuated —
        the scale gets stuck at 0 (the v18/v19 pstudio/adt collapse).

      * param="exp" (v20+):  s = exp(z), bias init 0 so s = 1.
        ∂(log s)/∂z = 1 everywhere → learning is uniform across orders of
        magnitude (0.01–100 m ↔ z ∈ [−4.6, +4.6]); `s` never hard-zeros
        and can always recover. log-scale `z` is also the natural quantity
        to supervise directly (see TrackingLossV20 scale term).
    """

    def __init__(self, dim: int, hidden: int | None = None,
                 param: str = "softplus") -> None:
        super().__init__()
        if param not in ("softplus", "exp"):
            raise ValueError(f"ScaleHead param must be 'softplus' or 'exp', got {param!r}")
        self.param = param
        hidden = hidden or max(dim // 2, 32)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            final = self.mlp[-1]
            final.weight.zero_()
            # init so s = 1 at startup: softplus(0.5413)=1, exp(0)=1.
            final.bias.fill_(math.log(math.expm1(1.0)) if param == "softplus" else 0.0)

    def forward(self, cls_per_frame: Tensor) -> Tensor:
        x = self.norm(cls_per_frame.mean(dim=1))               # (B, D)
        z = self.mlp(x).squeeze(-1)                            # (B,) raw log-scale (exp) / pre-softplus
        if self.param == "exp":
            return torch.exp(z)
        return F.softplus(z)
