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
