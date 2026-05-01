"""Tiny instance-segmentation head + 100-iter COCO-mini training + overlay demo.

The point of this module is not to match Mask-RCNN — it's to prove the
Mamba-3-swapped backbone produces instance-discriminative features. We train
a 2-layer conv head to do binary foreground segmentation on the union of
instance masks in each image. The resulting overlays should roughly follow
object boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class TinyInstanceSegHead(nn.Module):
    """2-layer conv head: feature map -> per-pixel foreground logit.

    Expects inputs shaped (B, C, h, w) where (h, w) is the patch grid.
    Upsamples to (B, 1, image_size, image_size) at the end.
    """

    def __init__(self, in_channels: int, hidden: int = 64, image_size: int = 224) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, 1, kernel_size=1)
        self.image_size = image_size

    def forward(self, features_grid: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(features_grid))
        x = self.conv2(x)  # (B, 1, h, w)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x


def train_seg_head(
    head: TinyInstanceSegHead,
    feature_extractor,
    samples: List,
    iters: int = 100,
    lr: float = 3e-3,
    device: str = "cpu",
) -> list[float]:
    """Train `head` on the given COCO-mini samples using binary foreground masks.

    `feature_extractor(image)` must return a (C, h, w) tensor of backbone features
    for a single (3, H, W) image input (batched internally).
    """
    head.train()
    head.to(device)
    optim = torch.optim.Adam(head.parameters(), lr=lr)
    losses: list[float] = []

    images = torch.stack([s.image for s in samples]).to(device)  # (N, 3, H, W)
    # Any-instance mask per image as binary target.
    targets = torch.stack([s.masks.any(dim=0).float() for s in samples]).to(device)  # (N, H, W)

    with torch.no_grad():
        feats = []
        for img in images:
            feats.append(feature_extractor(img))
        feats = torch.stack(feats).to(device)  # (N, C, h, w)

    pos_weight = ((targets.numel() - targets.sum()) / targets.sum().clamp_min(1)).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for it in range(iters):
        logits = head(feats).squeeze(1)  # (N, H, W)
        loss = loss_fn(logits, targets)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    return losses


@torch.no_grad()
def save_seg_overlay(
    head: TinyInstanceSegHead,
    feature_extractor,
    sample,
    path: Path,
    alpha: float = 0.6,
    cmap: str = "viridis",
) -> Path:
    """Overlay the per-pixel foreground probability (not a binary mask)."""
    head.eval()
    feats = feature_extractor(sample.image).unsqueeze(0)
    logits = head(feats).squeeze()  # (H, W)
    prob = torch.sigmoid(logits).cpu().numpy()

    img = sample.image.cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
    heat = cm.get_cmap(cmap)(prob)[..., :3]
    weight = alpha * prob[..., None]
    overlay = (1 - weight) * img + weight * heat
    overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)
    return path
