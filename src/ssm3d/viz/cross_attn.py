"""Cross-view attention map visualization.

Picks one query token index, takes that row from the Mamba-3 cross-attention
similarity matrix, reshapes to the kv patch grid, and overlays on the kv image.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


def save_cross_attention_heatmap(
    attn_map: torch.Tensor,
    kv_image: torch.Tensor,
    kv_grid_hw: tuple[int, int],
    query_index: int,
    path: Path,
    alpha: float = 0.6,
    cmap: str = "inferno",
) -> Path:
    """
    Args:
        attn_map: (B=1, H_heads, T_q, T_kv) — similarity * mask from Mamba3CrossAttention
        kv_image: (3, H_img, W_img) in [0, 1]
        kv_grid_hw: (h, w) of the kv patch grid so h*w = T_kv
        query_index: which query token to visualize
        path: output image path
    """
    assert attn_map.ndim == 4 and attn_map.shape[0] == 1
    attn = attn_map.detach().cpu().float()
    row = attn[0, :, query_index, :].mean(dim=0)  # average over heads -> (T_kv,)
    h, w = kv_grid_hw
    assert row.numel() == h * w, f"expected {h*w} kv tokens, got {row.numel()}"

    heat = row.view(h, w).numpy()
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-8)

    img = kv_image.detach().cpu().float().permute(1, 2, 0).numpy()  # (H, W, 3)
    H_img, W_img = img.shape[:2]
    heat_resized = np.array(
        Image.fromarray((heat * 255).astype(np.uint8)).resize((W_img, H_img), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    heat_rgb = cm.get_cmap(cmap)(heat_resized)[..., :3]  # (H, W, 3)

    overlay = (1 - alpha) * img + alpha * heat_rgb
    overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)
    return path
