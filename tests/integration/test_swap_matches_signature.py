"""Verify install_mamba3 swaps every backbone attention without breaking shapes."""

from __future__ import annotations

import torch

from ssm3d.da3_adapter import Mamba3Attention
from ssm3d.model import SSM3DBackbone
from ssm3d.patch import count_mamba3_attn


def test_backbone_constructed_with_mamba3_block_fn():
    # SSM3DBackbone already plumbs Mamba3Attention through block_fn.
    bb = SSM3DBackbone(size="small", img_size=224, patch_size=16, depth=2)
    n_m3 = count_mamba3_attn(bb)
    n_blocks = len(bb.vit.blocks)
    assert n_m3 == n_blocks, f"expected {n_blocks} Mamba3Attention modules, got {n_m3}"
    for block in bb.vit.blocks:
        assert isinstance(block.attn, Mamba3Attention)


def test_backbone_forward_shapes():
    torch.manual_seed(0)
    bb = SSM3DBackbone(size="small", img_size=224, patch_size=16, depth=2)
    bb.eval()
    x = torch.randn(1, 2, 3, 224, 224)
    with torch.no_grad():
        out = bb(x)
    B, S, N, C = out.features.shape
    h, w = out.grid_hw
    assert (B, S) == (1, 2)
    assert N == h * w == (224 // 16) ** 2
    assert C == bb.embed_dim
    assert torch.isfinite(out.features).all()
