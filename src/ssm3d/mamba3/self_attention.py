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

from .mask import (
    build_three_term_mask,
    build_three_term_mask_rows,
    build_two_term_mask,
    build_two_term_mask_rows,
)
from .projections import AttentionProjections


def ssd_forward(
    B: Tensor,
    C: Tensor,
    V: Tensor,
    L: Tensor,
    row_renorm: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Apply SSD output formula given projections and mask.

    Args:
        B: (batch, H, T, N_state)
        C: (batch, H, T, N_state)
        V: (batch, H, T, head_dim)
        L: (batch, H, T, T) lower-triangular decay mask
        row_renorm: if True, divide each row of (L ⊙ (C·Bᵀ)) by the sum of the
            row's magnitudes before multiplying V. Gives softmax-like "row sums
            to 1" contract — needed when a downstream softmax-trained head
            (e.g. DA3 DualDPT) consumes these features.

    Returns:
        Y: (batch, H, T, head_dim)
    """
    sim = torch.matmul(C, B.transpose(-2, -1))
    weighted = sim * L
    if row_renorm:
        denom = weighted.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
        weighted = weighted / denom
    return torch.matmul(weighted, V)


def ssd_forward_chunked(
    B: Tensor,
    C: Tensor,
    V: Tensor,
    delta: Tensor,
    A_log: Tensor,
    lam: Tensor | None = None,
    three_term: bool = True,
    row_renorm: bool = False,
    chunk_size: int = 128,
    eps: float = 1e-6,
) -> Tensor:
    """Memory-efficient SSD forward: builds mask rows in chunks of `chunk_size`.

    Equivalent to `ssd_forward(B, C, V, build_*_mask(delta, A_log[, lam]), ...)`
    but never materializes the full (..., T, T) mask — peak memory per chunk is
    O(chunk_size * T) instead of O(T * T). Needed for high-resolution inference
    (R7 in PLAN §9) where T = (H/patch_size)² exceeds a few hundred tokens.

    Args mirror `ssd_forward`; `three_term` picks the Mamba-3 trapezoidal mask
    (requires `lam`) vs. the Mamba-2 two-term mask.
    """
    T = B.shape[-2]
    if chunk_size is None or chunk_size >= T:
        if three_term:
            assert lam is not None, "three_term=True requires lam"
            L = build_three_term_mask(delta, A_log, lam)
        else:
            L = build_two_term_mask(delta, A_log)
        return ssd_forward(B, C, V, L, row_renorm=row_renorm, eps=eps)

    out_chunks: list[Tensor] = []
    for q_start in range(0, T, chunk_size):
        q_end = min(q_start + chunk_size, T)
        if three_term:
            assert lam is not None, "three_term=True requires lam"
            L_rows = build_three_term_mask_rows(delta, A_log, lam, q_start, q_end)
        else:
            L_rows = build_two_term_mask_rows(delta, A_log, q_start, q_end)
        C_chunk = C[..., q_start:q_end, :]
        sim = torch.matmul(C_chunk, B.transpose(-2, -1))  # (..., chunk, T)
        weighted = sim * L_rows
        if row_renorm:
            denom = weighted.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
            weighted = weighted / denom
        out_chunks.append(torch.matmul(weighted, V))
    return torch.cat(out_chunks, dim=-2)


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
        row_renorm: bool = True,
        post_norm: bool = True,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.state_dim = state_dim
        self.bidirectional = bidirectional
        self.three_term = three_term
        self.rope = rope
        self.row_renorm = row_renorm
        self.chunk_size = chunk_size

        self.projections = AttentionProjections(dim, num_heads, state_dim)
        # Per-head pre-tanh gate for the reverse SSD stream. Zero-init means
        # tanh(0)=0, so at init the layer behaves as forward-only (reverse noise
        # disabled). Training opens the gate as needed (R2 in PLAN §9).
        self.rev_gate = nn.Parameter(torch.zeros(num_heads)) if bidirectional else None
        self.post_norm = nn.LayerNorm(dim) if post_norm else nn.Identity()
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
        if self.chunk_size is not None and self.chunk_size < Bp.shape[-2]:
            return ssd_forward_chunked(
                Bp, Cp, Vp, delta, A_log, lam,
                three_term=self.three_term,
                row_renorm=self.row_renorm,
                chunk_size=self.chunk_size,
            )
        L = self._build_mask(delta, A_log, lam)
        return ssd_forward(Bp, Cp, Vp, L, row_renorm=self.row_renorm)

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
            gate = torch.tanh(self.rev_gate)[None, :, None, None]  # (1, H, 1, 1)
            y = y + gate * y_rev

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
        y = self.post_norm(y)
        return self.proj(y)
