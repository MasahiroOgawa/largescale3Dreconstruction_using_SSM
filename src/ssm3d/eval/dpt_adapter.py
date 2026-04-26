"""Shared-DPT bridge: run DA3's pretrained DualDPT on SSM-3D features.

DA3-SMALL concatenates two 384-d attention streams (`cat_token=True`) into
768-d features, then feeds 4 intermediate layers `[5,7,9,11]` into its DualDPT.
SSM-3D has a single 384-d stream.

At init, the `DimBridge` (when provided) reproduces the legacy `cat([f, f], -1)`
duplication, so existing shape tests pass. Training the bridge in Phase C lets
the two 384-d halves of the DualDPT input carry complementary information
instead of redundant copies (PLAN §9 R5).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from ssm3d.bridge import DimBridgeStack
from ssm3d.model import SSM3DNet


SHARED_DPT_LAYERS: tuple[int, ...] = (5, 7, 9, 11)


def get_dualdpt(da3_model) -> torch.nn.Module:
    """Return a reference to the DualDPT head living at `model.model.head`."""
    return da3_model.model.head


def _bridge_feats(
    feats: list[Tensor], bridge: Optional[DimBridgeStack]
) -> list[Tensor]:
    if bridge is None:
        return [torch.cat([f, f], dim=-1) for f in feats]
    return bridge(feats)


def shared_dpt_depth(
    ssm_net: SSM3DNet,
    dualdpt: torch.nn.Module,
    images: Tensor,
    layers: Sequence[int] = SHARED_DPT_LAYERS,
    bridge: Optional[DimBridgeStack] = None,
) -> Tensor:
    """Run SSM-3D backbone → channel bridge → DA3 DualDPT → depth.

    Args:
        ssm_net: SSM3DNet with backbone weights loaded.
        dualdpt: the `DualDPT` instance from a loaded DA3 model.
        images: (N, 3, H, W) in [0, 1].
        layers: 4 backbone layer indices to feed the head. Must match the
            intermediate_layer_idx contract of DualDPT (default (5,7,9,11)).
        bridge: optional `DimBridgeStack`. When None, falls back to the legacy
            `cat([f, f], -1)` duplicate (deterministic smoke test).

    Returns:
        (N, H, W) depth tensor on CPU. NaN-free if the pipeline is healthy.
    """
    return shared_dpt_outputs(ssm_net, dualdpt, images, layers, bridge)["depth"]


def shared_dpt_outputs(
    ssm_net: SSM3DNet,
    dualdpt: torch.nn.Module,
    images: Tensor,
    layers: Sequence[int] = SHARED_DPT_LAYERS,
    bridge: Optional[DimBridgeStack] = None,
) -> dict[str, Tensor]:
    """Run the full SSM-3D + DualDPT pipeline; return depth + ray + confidences.

    DualDPT emits a dict keyed by head names ('depth', 'ray' for DA3-SMALL).
    `output_conv2` produces 2 channels → depth (1) + depth_conf (1).
    `output_conv2_aux` produces 7 channels → ray (6: 3 direction + 3 origin) +
    ray_conf (1). See `dualdpt.py:230-264`.

    Returns dict with CPU float tensors:
        depth:      (N, H, W)
        depth_conf: (N, H, W)
        ray:        (N, H, W, 6)
        ray_conf:   (N, H, W)
    """
    if len(layers) != 4:
        raise ValueError(f"DualDPT expects exactly 4 layers, got {len(layers)}")
    with torch.inference_mode():
        x = images.unsqueeze(0)  # (B=1, S=N, 3, H, W)
        out = ssm_net.backbone(x, export_feat_layers=list(layers))
        if len(out.aux_features) != 4:
            raise RuntimeError(
                f"backbone returned {len(out.aux_features)} aux layers; expected 4"
            )
        bridged = _bridge_feats(out.aux_features, bridge)
        feats_for_dpt = [(f,) for f in bridged]
        H_img, W_img = images.shape[-2], images.shape[-1]
        result = dualdpt(feats_for_dpt, H_img, W_img, patch_start_idx=0)
        head_main = getattr(dualdpt, "head_main", "depth")
        head_aux = getattr(dualdpt, "head_aux", "ray")

        def _drop_view_axis(t: Tensor) -> Tensor:
            return t.squeeze(2) if t.dim() == 5 else t

        out_dict: dict[str, Tensor] = {}

        def _maybe(key: str, drop_view: bool):
            if key not in result:
                return None
            t = result[key]
            if drop_view:
                t = _drop_view_axis(t)
            return t[0].detach().cpu().float()

        out_dict["depth"] = _maybe(head_main, True)
        out_dict["depth_conf"] = _maybe(f"{head_main}_conf", True)
        out_dict["ray"] = _maybe(head_aux, False)
        out_dict["ray_conf"] = _maybe(f"{head_aux}_conf", False)
        return out_dict
