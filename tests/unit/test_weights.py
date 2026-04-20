"""Weight-loader filters .attn.* and shape mismatches, loads compatible tensors."""

from __future__ import annotations

import torch

from ssm3d.model import SSM3DBackbone
from ssm3d.weights import load_dinov2_backbone, warm_start_mamba3_from_qkv


def test_load_dinov2_filters_attn_and_mismatches():
    backbone = SSM3DBackbone(size="small", img_size=112, patch_size=16, depth=2)
    target = backbone.vit.state_dict()

    fake = {}
    for k, v in target.items():
        if ".attn." in k:
            fake[k] = torch.zeros_like(v)
        else:
            fake[k] = torch.ones_like(v)
    fake["patch_embed.proj.weight"] = torch.zeros(10, 3, 4, 4)
    fake["nonexistent.key"] = torch.zeros(3)

    counters = load_dinov2_backbone(backbone.vit, fake, verbose=False)
    assert counters["skipped_attn"] > 0, "expected some .attn. keys in the fake ckpt"
    assert counters["shape_mismatch"] >= 1, "patch_embed shape mismatch should be counted"
    assert counters["loaded"] > 0

    got = backbone.vit.state_dict()
    for k, v in got.items():
        if ".attn." in k:
            continue
        if "patch_embed.proj.weight" in k:
            continue
        assert torch.allclose(v, torch.ones_like(v)), f"{k} did not load"


def test_warm_start_copies_qkv_into_bcv():
    """K→B, Q→C, V→V per block, output-proj copied too."""
    depth = 3
    backbone = SSM3DBackbone(size="small", img_size=112, patch_size=14, depth=depth)
    D = backbone.embed_dim
    H = backbone.vit.blocks[0].attn.inner.projections.num_heads
    N = backbone.vit.blocks[0].attn.inner.projections.state_dim
    hd = D // H
    assert hd >= N, "test assumes head_dim >= state_dim"

    fake = {}
    # Use structured non-trivial values so copies can be detected byte-for-byte.
    for i in range(depth):
        qkv = torch.randn(3 * D, D)
        fake[f"blocks.{i}.attn.qkv.weight"] = qkv
        fake[f"blocks.{i}.attn.proj.weight"] = torch.randn(D, D)
        fake[f"blocks.{i}.attn.proj.bias"] = torch.randn(D)

    counters = warm_start_mamba3_from_qkv(backbone.vit, fake, verbose=False)
    assert counters["warmed"] == depth
    assert counters["out_warmed"] == depth

    for i in range(depth):
        qkv = fake[f"blocks.{i}.attn.qkv.weight"]
        Q_expected = qkv[0:D].view(H, hd, D)[:, :N, :].reshape(H * N, D)
        K_expected = qkv[D : 2 * D].view(H, hd, D)[:, :N, :].reshape(H * N, D)
        V_expected = qkv[2 * D : 3 * D]

        proj_w = backbone.vit.blocks[i].attn.inner.projections.proj.weight
        s1 = H * N
        assert torch.equal(proj_w[:s1], K_expected), f"block {i}: B rows should equal K"
        assert torch.equal(proj_w[s1 : 2 * s1], Q_expected), f"block {i}: C rows should equal Q"
        assert torch.equal(proj_w[2 * s1 : 2 * s1 + D], V_expected), f"block {i}: V rows should equal V"

        out_proj = backbone.vit.blocks[i].attn.inner.proj
        assert torch.equal(out_proj.weight, fake[f"blocks.{i}.attn.proj.weight"])
        assert torch.equal(out_proj.bias, fake[f"blocks.{i}.attn.proj.bias"])


def test_warm_start_skips_when_head_dim_too_small():
    """If state_dim > head_dim, block is skipped cleanly."""
    backbone = SSM3DBackbone(
        size="small", img_size=112, patch_size=14, depth=2, mamba_state_dim=1024
    )
    # head_dim = 384/6 = 64; 1024 > 64, so all blocks should be skipped.
    D = backbone.embed_dim
    fake = {f"blocks.{i}.attn.qkv.weight": torch.zeros(3 * D, D) for i in range(2)}
    counters = warm_start_mamba3_from_qkv(backbone.vit, fake, verbose=False)
    assert counters["warmed"] == 0
    assert counters["skipped"] == 2
