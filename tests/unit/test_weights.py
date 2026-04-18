"""Weight-loader filters .attn.* and shape mismatches, loads compatible tensors."""

from __future__ import annotations

import torch

from ssm3d.model import SSM3DBackbone
from ssm3d.weights import load_dinov2_backbone


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
