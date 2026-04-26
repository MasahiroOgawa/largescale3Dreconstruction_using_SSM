"""End-to-end forward pass on SSM3DNet with random inputs."""

from __future__ import annotations

import torch

from ssm3d.model import SSM3DNet


def test_ssm3dnet_forward_random():
    torch.manual_seed(0)
    net = SSM3DNet(size="small", img_size=224, patch_size=16, depth=2, head_hidden=32)
    net.eval()
    x = torch.randn(1, 2, 3, 224, 224)
    with torch.no_grad():
        out = net(x)
    assert set(out.keys()) == {"features", "depth", "grid_hw"}
    assert out["features"].shape == (1, 2, (224 // 16) ** 2, net.backbone.embed_dim)
    assert out["depth"].shape == (1, 2, 1, 224, 224)
    assert torch.isfinite(out["depth"]).all()
    # softplus keeps depth positive
    assert (out["depth"] >= 0).all()


def test_ssm3dnet_grad_flows():
    torch.manual_seed(0)
    net = SSM3DNet(size="small", img_size=224, patch_size=16, depth=2, head_hidden=32)
    net.train()
    x = torch.randn(1, 2, 3, 224, 224, requires_grad=False)
    out = net(x)
    loss = out["depth"].mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradient reached any parameter"


def test_ssm3dnet_full_swap_forward():
    """alt_start + cat_token mirrors DA3's hybrid self+cross alternation pattern.

    DA3 sets `alt_start=4` for 12-block ViT-S so layers 5/7/9/11 run on the
    `(B, S*N, C)` concatenated multi-view sequence (cross-view). Our Mamba-3
    self-attention block handles both per-view and cross-view input shapes
    transparently — it is the same module, applied to a longer or shorter
    token sequence depending on `alt_start`. § 15.43.
    """
    torch.manual_seed(0)
    # depth=4 with alt_start=2 means layers 2, 3 alternate (layer 3 → cross-view).
    net = SSM3DNet(
        size="small", img_size=224, patch_size=16, depth=4, head_hidden=32,
        alt_start=2, cat_token=True,
    )
    net.eval()
    assert net.backbone.vit.alt_start == 2
    assert hasattr(net.backbone.vit, "camera_token")
    # multi-view input — layer 3 (cross-view) feeds the same Mamba-3 block
    # the (B, S*N, C) concatenated stream.
    x = torch.randn(1, 3, 3, 224, 224)
    with torch.no_grad():
        out = net(x)
    feat = out["features"]
    # cat_token=True at tap layer doubles channel dim ([local_x ‖ current_x]).
    assert feat.shape[:3] == (1, 3, (224 // 16) ** 2)
    assert feat.shape[-1] == 2 * net.backbone.embed_dim
    assert torch.isfinite(feat).all()
