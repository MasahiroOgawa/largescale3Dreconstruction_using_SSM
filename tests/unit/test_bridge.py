"""Tests for ssm3d.bridge.DimBridge — identity-at-init, shapes, training path."""

from __future__ import annotations

import torch

from ssm3d.bridge import DimBridge, DimBridgeStack


def test_dim_bridge_identity_at_init_reproduces_cat_duplicate():
    torch.manual_seed(0)
    bridge = DimBridge(in_dim=384)
    x = torch.randn(2, 7, 384)
    y = bridge(x)
    expected = torch.cat([x, x], dim=-1)
    assert torch.allclose(y, expected, atol=1e-5)


def test_dim_bridge_stack_matches_cat_duplicate_at_init():
    torch.manual_seed(0)
    stack = DimBridgeStack(num_layers=4, in_dim=384)
    feats = [torch.randn(1, 16, 384) for _ in range(4)]
    bridged = stack(feats)
    assert len(bridged) == 4
    for f, b in zip(feats, bridged):
        assert b.shape == (1, 16, 768)
        assert torch.allclose(b, torch.cat([f, f], dim=-1), atol=1e-5)


def test_dim_bridge_receives_gradients():
    bridge = DimBridge(in_dim=16)
    x = torch.randn(1, 4, 16, requires_grad=True)
    y = bridge(x)
    y.sum().backward()
    assert bridge.linear.weight.grad is not None
    assert bridge.linear.weight.grad.abs().sum() > 0


def test_dim_bridge_rejects_non_double_out_dim():
    import pytest

    with pytest.raises(ValueError):
        DimBridge(in_dim=384, out_dim=512)
