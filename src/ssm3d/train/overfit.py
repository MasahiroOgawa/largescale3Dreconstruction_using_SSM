"""Short overfit training loop for the demo.

Inputs: a handful of multi-view image tuples and (optional) pseudo-depth
targets. The loss is simple L1 on depth. When no GT depth is provided we use
a self-consistency target: the mean depth across views should equal the
per-view depth (a trivial but non-degenerate objective that demonstrates
gradients flow through the swapped attention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.optim import AdamW


@dataclass
class OverfitResult:
    losses: list[float] = field(default_factory=list)
    initial_loss: float = 0.0
    final_loss: float = 0.0


def _scale_invariant_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # median-align pred to target to remove global scale ambiguity, then L1
    with torch.no_grad():
        scale = target.median() / pred.median().clamp_min(eps)
    return torch.abs(pred * scale - target).mean()


def overfit_run(
    net: torch.nn.Module,
    images: torch.Tensor,  # (B, S, 3, H, W)
    gt_depth: Optional[torch.Tensor] = None,  # (B, S, 1, H, W) or None
    iters: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> OverfitResult:
    net.to(device)
    net.train()
    images = images.to(device)
    if gt_depth is not None:
        gt_depth = gt_depth.to(device)

    opt = AdamW(net.parameters(), lr=lr)
    losses: list[float] = []
    for it in range(iters):
        out = net(images)
        pred = out["depth"]  # (B, S, 1, H, W)
        if gt_depth is not None:
            loss = _scale_invariant_l1(pred, gt_depth)
        else:
            # Self-consistency: minimize variance across views of predicted depth
            # (each view's depth should be close to the multi-view mean).
            mean_depth = pred.mean(dim=1, keepdim=True)
            loss = (pred - mean_depth).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    return OverfitResult(losses=losses, initial_loss=losses[0], final_loss=losses[-1])
