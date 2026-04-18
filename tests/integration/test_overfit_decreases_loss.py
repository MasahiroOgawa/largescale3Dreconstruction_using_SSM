"""Short overfit run must reduce the self-consistency loss substantially."""

from __future__ import annotations

import torch

from ssm3d.model import SSM3DNet
from ssm3d.train.overfit import overfit_run


def test_overfit_self_consistency_decreases():
    torch.manual_seed(0)
    net = SSM3DNet(size="small", img_size=112, patch_size=16, depth=2, head_hidden=32)
    # two views of distinct random images
    images = torch.randn(1, 2, 3, 112, 112)
    result = overfit_run(net, images, iters=30, lr=3e-3, device="cpu")
    assert result.final_loss < 0.5 * result.initial_loss, (
        f"final {result.final_loss:.4f} not < 0.5 * initial {result.initial_loss:.4f}"
    )
