"""Shared-DPT smoke test: run DA3's pretrained DualDPT on SSM-3D features.

DA3-SMALL concatenates two 384-d attention streams (`cat_token=True`) into
768-d features, then feeds 4 intermediate layers `[5,7,9,11]` into its DualDPT.
SSM-3D has a single 384-d stream. To route SSM-3D features through the same
head for a visual comparison, we duplicate each layer along the channel axis
(`cat([f, f], -1)`) before calling DualDPT.

This is an **un-retrained** head-on-drifted-features smoke test. The resulting
depth is visually informative but not a fair apples-to-apples benchmark —
consumers should label it accordingly in figures and summaries.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ssm3d.model import SSM3DNet


SHARED_DPT_LAYERS: tuple[int, ...] = (5, 7, 9, 11)


def get_dualdpt(da3_model) -> torch.nn.Module:
    """Return a reference to the DualDPT head living at `model.model.head`."""
    return da3_model.model.head


@torch.inference_mode()
def shared_dpt_depth(
    ssm_net: SSM3DNet,
    dualdpt: torch.nn.Module,
    images: Tensor,
    layers: Sequence[int] = SHARED_DPT_LAYERS,
) -> Tensor:
    """Run SSM-3D backbone → duplicate channel dim → DA3 DualDPT → depth.

    Args:
        ssm_net: SSM3DNet with DINOv2 weights loaded.
        dualdpt: the `DualDPT` instance from a loaded DA3 model.
        images: (N, 3, H, W) in [0, 1].
        layers: 4 backbone layer indices to feed the head. Must match the
            intermediate_layer_idx contract of DualDPT (default (5,7,9,11)).

    Returns:
        (N, H, W) depth tensor on CPU. NaN-free if the pipeline is healthy.
    """
    if len(layers) != 4:
        raise ValueError(f"DualDPT expects exactly 4 layers, got {len(layers)}")
    x = images.unsqueeze(0)  # (B=1, S=N, 3, H, W)
    out = ssm_net.backbone(x, export_feat_layers=list(layers))
    if len(out.aux_features) != 4:
        raise RuntimeError(
            f"backbone returned {len(out.aux_features)} aux layers; expected 4"
        )
    duplicated = [torch.cat([f, f], dim=-1) for f in out.aux_features]
    feats_for_dpt = [(f,) for f in duplicated]  # DualDPT does feats[i][0]
    H_img, W_img = images.shape[-2], images.shape[-1]
    result = dualdpt(feats_for_dpt, H_img, W_img, patch_start_idx=0)
    depth_main_name = getattr(dualdpt, "head_main", "depth")
    depth = result[depth_main_name]  # (B, S, H/dr, W/dr) after squeeze(-1) in head
    if depth.dim() == 5:
        depth = depth.squeeze(2)  # drop channel axis if present
    depth = depth[0]  # unwrap B=1 -> (N, H, W)
    return depth.detach().cpu().float()
