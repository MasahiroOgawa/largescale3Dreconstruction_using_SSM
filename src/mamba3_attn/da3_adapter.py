"""Drop-in adapter so Mamba-3 attention can be used wherever DA3 expects its
`Attention` class.

DA3's Block constructs attention via:
    attn_class(dim, num_heads=..., qkv_bias=..., proj_bias=...,
               attn_drop=..., proj_drop=..., qk_norm=..., rope=...)

and calls it as:
    attn(x, pos=pos, attn_mask=attn_mask)

Our Mamba3Attention matches that signature exactly, ignoring kwargs that don't
apply (qkv_bias, attn_drop, proj_drop, qk_norm) and using `rope` if provided.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .mamba3 import Mamba3SelfAttention, Mamba3VSSDAttention


class Mamba3Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,  # ignored
        proj_bias: bool = True,
        attn_drop: float = 0.0,  # ignored
        proj_drop: float = 0.0,  # applied as nn.Dropout on output
        norm_layer: nn.Module = nn.LayerNorm,  # ignored (BCNorm handles our norm)
        qk_norm: bool = False,  # ignored
        fused_attn: bool = True,  # ignored
        rope: Optional[nn.Module] = None,
        state_dim: int = 64,
        bidirectional: bool = True,
        three_term: bool = True,
        chunk_size: Optional[int] = None,
        use_fused_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.inner = Mamba3SelfAttention(
            dim=dim,
            num_heads=num_heads,
            state_dim=state_dim,
            bidirectional=bidirectional,
            three_term=three_term,
            rope=rope,
            out_proj=True,
            proj_bias=proj_bias,
            chunk_size=chunk_size,
            use_fused_kernel=use_fused_kernel,
        )
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        y = self.inner(x, pos=pos, attn_mask=attn_mask)
        return self.proj_drop(y)


class Mamba3VSSDAdapter(nn.Module):
    """DA3-shaped wrapper around :class:`Mamba3VSSDAttention`.

    Same ``__init__`` signature as :class:`Mamba3Attention` so
    ``install_mamba3(variant="vssd")`` can construct it with identical
    keyword arguments. The SSD-only flags (``bidirectional``, ``three_term``,
    ``chunk_size``, ``use_fused_kernel``) are accepted-and-ignored by NC-SSD.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope: Optional[nn.Module] = None,
        state_dim: int = 64,
        bidirectional: bool = True,
        three_term: bool = True,
        chunk_size: Optional[int] = None,
        use_fused_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.inner = Mamba3VSSDAttention(
            dim=dim,
            num_heads=num_heads,
            state_dim=state_dim,
            rope=rope,
            out_proj=True,
            proj_bias=proj_bias,
        )
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        y = self.inner(x, pos=pos, attn_mask=attn_mask)
        return self.proj_drop(y)
