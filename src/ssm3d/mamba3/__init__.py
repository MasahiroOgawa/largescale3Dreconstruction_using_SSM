"""Mamba-3 SSD attention building blocks.

Reference: /home/mas/proj/study/mamba3_doc/attention/mamba3_attention.tex
Core formula:  Y = (L ⊙ (C Bᵀ)) V
"""

from .mask import build_two_term_mask, build_three_term_mask, build_cross_mask
from .projections import AttentionProjections, BCNorm
from .rope2d import RoPE2D
from .self_attention import Mamba3SelfAttention
from .cross_attention import Mamba3CrossAttention

__all__ = [
    "build_two_term_mask",
    "build_three_term_mask",
    "build_cross_mask",
    "AttentionProjections",
    "BCNorm",
    "RoPE2D",
    "Mamba3SelfAttention",
    "Mamba3CrossAttention",
]
