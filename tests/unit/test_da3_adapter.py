"""Unit tests for ssm3d.da3_adapter.Mamba3Attention"""

from __future__ import annotations

import torch
from torch import nn

from ssm3d.da3_adapter import Mamba3Attention
from ssm3d.mamba3.rope2d import RoPE2D


def test_accepts_full_da3_kwargs():
    # Should not raise on any of the kwargs DA3's Block passes.
    attn = Mamba3Attention(
        dim=32,
        num_heads=4,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        norm_layer=nn.LayerNorm,
        qk_norm=True,
        fused_attn=False,
        rope=None,
    )
    x = torch.randn(2, 10, 32)
    y = attn(x)
    assert y.shape == x.shape


def test_forward_without_pos_or_mask():
    attn = Mamba3Attention(dim=16, num_heads=2)
    x = torch.randn(1, 8, 16)
    y = attn(x)
    assert y.shape == x.shape


def test_forward_with_pos_runs():
    rope = RoPE2D()
    attn = Mamba3Attention(dim=16, num_heads=2, rope=rope)
    x = torch.randn(1, 9, 16)
    pos = torch.cartesian_prod(torch.arange(3), torch.arange(3)).view(1, 9, 2)
    y = attn(x, pos=pos)
    assert y.shape == x.shape


def test_forward_with_attn_mask_runs():
    attn = Mamba3Attention(dim=16, num_heads=2)
    x = torch.randn(1, 8, 16)
    mask = torch.ones(1, 8)
    mask[:, 4:] = 0
    y = attn(x, attn_mask=mask)
    assert y.shape == x.shape


def test_inside_da3_block():
    """Swap attn_class on DA3's Block and confirm the residual path flows."""
    from depth_anything_3.model.dinov2.layers.block import Block

    block = Block(dim=32, num_heads=4, attn_class=Mamba3Attention)
    x = torch.randn(2, 16, 32)
    y = block(x)
    assert y.shape == x.shape
    assert type(block.attn).__name__ == "Mamba3Attention"
