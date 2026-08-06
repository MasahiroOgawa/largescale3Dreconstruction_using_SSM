"""Mamba-3 SSD attention — the research contribution.

A drop-in replacement for transformer self/cross attention based on the
Structured State-space Duality (SSD) form of Mamba-3. Designed to be
**model-agnostic**: install into any architecture that consumes a
multi-head attention module with the standard `(B, T, C)` signature.

Core formula:  Y = (L ⊙ (C Bᵀ)) V

where C, B are query / key projections, L is a structured causal
(or bidirectional) decay mask, and V is the value projection.

References:
- `mamba3_doc/attention/mamba3_attention.tex` — derivation
- `mamba3_doc/original_paper/mamba3.pdf` — Dao et al. Mamba-3 paper
- Triton kernel: `mamba_ssm.ops.triton.mamba3.mamba3_siso_combined`

Public API:

>>> from mamba3_attn.mamba3 import Mamba3SelfAttention, Mamba3CrossAttention
>>> attn = Mamba3SelfAttention(dim=384, num_heads=6, state_dim=64,
...                             bidirectional=True, three_term=True)
>>> y = attn(x)  # x: (B, T, dim) → y: (B, T, dim)

Integration patterns:
- **DA3 swap** — see `mamba3_attn.patch.install_mamba3` for in-place attention
  replacement in a real Depth-Anything-3 model. The `mamba3_attn.da3_adapter`
  module provides a DA3-shaped wrapper.
- **Generic ViT swap** — instantiate `Mamba3SelfAttention(dim, num_heads)`
  with the same signature as `nn.MultiheadAttention` and replace
  `block.attn` directly. Many ViTs work without further adaptation.

Modules:
- `Mamba3SelfAttention`  — SISO bidirectional self-attention.
- `Mamba3CrossAttention` — query / kv-stream variant (for encoder-decoder).
- `RoPE2D`               — 2D rotary position embedding (DA3-compatible).
- `AttentionProjections` — Q/K/V/B/C/Δ projection bundle.
- `BCNorm`               — channel-wise normalization on the C/B streams.
- `build_two_term_mask`, `build_three_term_mask`, `build_cross_mask` —
  structured decay masks (causal / bidirectional / three-term variants).
"""

from .cross_attention import Mamba3CrossAttention
from .mask import build_cross_mask, build_three_term_mask, build_two_term_mask
from .projections import AttentionProjections, BCNorm
from .rope2d import RoPE2D
from .self_attention import Mamba3SelfAttention
from .vssd_attention import (Mamba3VSSDAttention, Mamba3VSSDBetaGammaAttention,
                             vssd_beta_gamma_forward, vssd_forward)

__all__ = [
    "Mamba3SelfAttention",
    "Mamba3VSSDAttention",
    "Mamba3CrossAttention",
    "RoPE2D",
    "AttentionProjections",
    "BCNorm",
    "build_two_term_mask",
    "build_three_term_mask",
    "build_cross_mask",
    "vssd_forward",
]
