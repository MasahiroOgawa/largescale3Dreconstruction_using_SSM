"""Mamba-3 self-attention (SSD) as a drop-in replacement for softmax attention.

Output formula per head:
    Y_h = (L_h ⊙ (C_h · B_hᵀ)) · V_h     shape (T, head_dim)

Where L is the structured decay mask from mask.build_three_term_mask.

Bidirectional variant sums the forward and reversed SSD (paper eq. 552).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .mask import build_three_term_mask, build_two_term_mask
from .projections import AttentionProjections


def ssd_forward(
    B: Tensor,
    C: Tensor,
    V: Tensor,
    L: Tensor,
) -> Tensor:
    """Apply SSD output formula given projections and mask.

    Args:
        B: (batch, H, T, N_state)
        C: (batch, H, T, N_state)
        V: (batch, H, T, head_dim)
        L: (batch, H, T, T) lower-triangular decay mask

    Returns:
        Y: (batch, H, T, head_dim)
    """
    # (B, H, T, N) @ (B, H, N, T) -> (B, H, T, T)
    sim = torch.matmul(C, B.transpose(-2, -1))
    weighted = sim * L
    return torch.matmul(weighted, V)


class Mamba3SelfAttention(nn.Module):
    """Bidirectional Mamba-3 self-attention.

    Args:
        dim:          token feature dimension D
        num_heads:    H (D must be divisible by H)
        state_dim:    N_state per head (default 64)
        bidirectional: if True, sum forward and reverse SSD
        three_term:   if True, use Mamba-3 trapezoidal mask; else Mamba-2 two-term
        rope:         optional module implementing forward(tokens, positions)
        out_proj:     if True, apply a final linear projection (like Attention.proj)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        state_dim: int = 64,
        bidirectional: bool = True,
        three_term: bool = True,
        rope: Optional[nn.Module] = None,
        out_proj: bool = True,
        proj_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dim = state_dim
        self.bidirectional = bidirectional
        self.three_term = three_term
        self.rope = rope

        self.projections = AttentionProjections(dim, num_heads, state_dim)
        self.proj = nn.Linear(dim, dim, bias=proj_bias) if out_proj else nn.Identity()

    def _build_mask(self, delta: Tensor, A_log: Tensor, lam: Tensor) -> Tensor:
        if self.three_term:
            return build_three_term_mask(delta, A_log, lam)
        return build_two_term_mask(delta, A_log)

    def _one_direction(
        self,
        Bp: Tensor,
        Cp: Tensor,
        Vp: Tensor,
        delta: Tensor,
        A_log: Tensor,
        lam: Tensor,
    ) -> Tensor:
        L = self._build_mask(delta, A_log, lam)
        return ssd_forward(Bp, Cp, Vp, L)

    def forward(
        self,
        x: Tensor,
        pos: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:         (B, T, D)
            pos:       (B, T, 2) integer 2D positions for RoPE, or None
            attn_mask: (B, T, T) additive/boolean mask OR (B, T); True/finite → keep.
                       When provided, zeros-out columns of L before the SSD
                       (so masked tokens don't contribute regardless of decay).

        Returns:
            y: (B, T, D)
        """
        Bp, Cp, Vp, delta, A_log, lam = self.projections(x)

        if self.rope is not None and pos is not None:
            Bp = self.rope(Bp, pos)
            Cp = self.rope(Cp, pos)

        y = self._one_direction(Bp, Cp, Vp, delta, A_log, lam)

        if self.bidirectional:
            # Reverse along T
            Bp_r = Bp.flip(dims=(-2,))
            Cp_r = Cp.flip(dims=(-2,))
            Vp_r = Vp.flip(dims=(-2,))
            delta_r = delta.flip(dims=(-1,))
            A_log_r = A_log.flip(dims=(-1,))
            lam_r = lam.flip(dims=(-1,))
            y_rev = self._one_direction(Bp_r, Cp_r, Vp_r, delta_r, A_log_r, lam_r)
            y_rev = y_rev.flip(dims=(-2,))
            y = y + y_rev

        if attn_mask is not None:
            # Token-zero-out semantics: if the row-i column-j is masked, remove
            # contribution of kv-token j from query-token i. For simplicity we
            # accept (B, T) that masks out whole kv-tokens.
            if attn_mask.ndim == 2:
                keep = attn_mask.to(y.dtype)  # (B, T)
                y = y * keep[:, None, :, None]

        # Merge heads: (B, H, T, head_dim) -> (B, T, D)
        Bsz, H, T, hd = y.shape
        y = y.transpose(1, 2).contiguous().view(Bsz, T, H * hd)
        return self.proj(y)
