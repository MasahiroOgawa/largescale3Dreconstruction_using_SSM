"""Depth map visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


def save_depth_colormap(depth: torch.Tensor, path: Path, cmap: str = "turbo", invert: bool = True) -> Path:
    """Save a colormap visualization of a depth / disparity map.

    Args:
        depth: (H, W) or (1, H, W) or (B, 1, H, W) float tensor
        path: output image path
        cmap: matplotlib colormap name
        invert: if True, closer = brighter (common for disparity-style views)
    """
    d = depth.detach().cpu().float()
    if d.ndim == 4:
        d = d[0, 0]
    elif d.ndim == 3:
        d = d[0]
    d_np = d.numpy()

    lo, hi = np.percentile(d_np, 2), np.percentile(d_np, 98)
    d_norm = np.clip((d_np - lo) / max(hi - lo, 1e-8), 0, 1)
    if invert:
        d_norm = 1.0 - d_norm

    img = (cm.get_cmap(cmap)(d_norm)[..., :3] * 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path
