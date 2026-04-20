"""Shape/plumbing test for the shared-DPT adapter.

We don't want this test to require DA3 weights, so we stub a mock DualDPT
that validates the feature list format (4 layers, 768-dim tokens, wrapped
in 1-tuples) and returns a depth tensor with the documented shape.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ssm3d.eval.dpt_adapter import SHARED_DPT_LAYERS, shared_dpt_depth
from ssm3d.model import SSM3DNet


class _MockDualDPT(nn.Module):
    head_main = "depth"

    def forward(self, feats, H: int, W: int, patch_start_idx: int = 0):
        assert len(feats) == 4, f"expected 4 layers, got {len(feats)}"
        for i, f in enumerate(feats):
            assert isinstance(f, tuple) and len(f) == 1, (
                f"layer {i} must be a 1-tuple, got {type(f).__name__}"
            )
            tensor = f[0]
            assert tensor.shape[-1] == 768, (
                f"layer {i} expected 768-dim (384 duplicated), got {tensor.shape[-1]}"
            )
        B, S = feats[0][0].shape[:2]
        depth = torch.zeros(B, S, 1, H, W)
        return {"depth": depth}


@pytest.fixture(scope="module")
def _tiny_ssm_net() -> SSM3DNet:
    net = SSM3DNet(size="small", img_size=56, patch_size=14, depth=12)
    net.eval()
    return net


def test_shared_dpt_produces_expected_shape(_tiny_ssm_net):
    images = torch.randn(2, 3, 56, 56)
    mock_head = _MockDualDPT()
    depth = shared_dpt_depth(_tiny_ssm_net, mock_head, images)
    assert depth.shape == (2, 56, 56)
    assert torch.isfinite(depth).all()


def test_shared_dpt_rejects_wrong_layer_count(_tiny_ssm_net):
    with pytest.raises(ValueError):
        shared_dpt_depth(_tiny_ssm_net, _MockDualDPT(), torch.zeros(1, 3, 56, 56),
                         layers=(5, 7, 9))


def test_shared_dpt_layers_match_da3_out_layers():
    # Guards against someone changing SHARED_DPT_LAYERS without rechecking
    # DA3-SMALL's `out_layers: [5, 7, 9, 11]`.
    assert SHARED_DPT_LAYERS == (5, 7, 9, 11)
