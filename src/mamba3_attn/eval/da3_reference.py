"""Thin wrapper around DA3's own API for use as a reference model.

DA3-SMALL is published on HF (verified 2026-04-20). We call `from_pretrained`
once and reuse the object for both depth inference and feature extraction.

Feature note: DA3-SMALL is configured with `cat_token=True, alt_start=4,
out_layers=[5,7,9,11]`, so its output features are 768-dim (concat of two
384-dim streams). SSM-3D features are 384-dim. Representation metrics
(`feat_cos_mean`, `effective_rank`, `cross_view_nn_agreement`) are
dim-invariant; they can be computed independently on each model's own
feature space and compared as scores.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Import order: mamba3_attn first to register DA3 sys.path.
import mamba3_attn  # noqa: F401

DEFAULT_HF_MODEL = "depth-anything/DA3-SMALL"


def _stub_export_module() -> None:
    """Pre-inject a stub `depth_anything_3.utils.export` into sys.modules.

    DA3's api.py does `from depth_anything_3.utils.export import export`, which
    transitively pulls in pycolmap, moviepy.editor, gsplat, trimesh — none of
    which we need for depth inference + feature extraction. Registering a stub
    module with a no-op `export` attr lets api.py import cleanly without
    editing the submodule (forbidden by project rules).
    """
    import sys
    import types
    name = "depth_anything_3.utils.export"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.export = lambda *a, **kw: None
    sys.modules[name] = stub


def load_da3(hf_model: str = DEFAULT_HF_MODEL, device: str = "cpu"):
    """Load a pretrained DA3 via HuggingFace Hub.

    Imports DA3 lazily because it depends on heavy packages (open3d, cv2).
    """
    _stub_export_module()
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(hf_model)
    model = model.to(device)
    model.eval()
    model.device = device
    return model


def _to_numpy_rgb_list(images: Tensor) -> list[np.ndarray]:
    """(N, 3, H, W) float [0,1] -> list of (H, W, 3) uint8 arrays for DA3."""
    arr = (images.clamp(0, 1) * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [arr[i] for i in range(arr.shape[0])]


@torch.inference_mode()
def da3_depth(model, images: Tensor, process_res: int = 504) -> Tensor:
    """Run DA3 inference and return a depth tensor at the INPUT image size.

    Args:
        images: (N, 3, H, W) float32 in [0, 1].
        process_res: DA3 processing resolution (square-bounded resize).

    Returns:
        (N, H, W) float32 depth (whatever DA3 emits, typically metric-ish).
    """
    image_list = _to_numpy_rgb_list(images)
    pred = model.inference(
        image_list,
        process_res=process_res,
        export_dir=None,
    )
    depth_np = pred.depth  # (N, H_da3, W_da3)
    depth = torch.from_numpy(np.asarray(depth_np)).float()
    # Resize to input size for direct comparison with GT.
    N, H_in, W_in = images.shape[0], images.shape[-2], images.shape[-1]
    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(H_in, W_in), mode="bilinear", align_corners=False
    ).squeeze(1)
    return depth_resized


@torch.inference_mode()
def da3_features(
    model,
    images: Tensor,
    layer: int = -1,
    process_res: int = 504,
) -> tuple[Tensor, tuple[int, int]]:
    """Return DA3 final-layer patch tokens + grid size.

    We run the backbone directly (skipping the DPT head) at the same processing
    resolution DA3 uses for inference, then reshape to (N, T, C).
    """
    from depth_anything_3.utils.io.input_processor import InputProcessor

    processor = InputProcessor() if getattr(model, "input_processor", None) is None else model.input_processor
    image_list = _to_numpy_rgb_list(images)
    imgs_cpu, _, _ = model._preprocess_inputs(
        image_list, None, None, process_res, "upper_bound_resize"
    )
    imgs, _, _ = model._prepare_model_inputs(imgs_cpu, None, None)
    imgs = imgs.to(model.device)

    backbone = model.model.backbone
    feats, _aux = backbone(imgs, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
    # `feats` layout depends on DA3 version; standard is list[tuple(tokens, aux_tokens)]
    # We want the LAST layer patch tokens.
    last = feats[layer] if isinstance(feats, (list, tuple)) else feats
    if isinstance(last, (list, tuple)):
        last = last[0]
    tokens = last  # (B, S, T, C) or (B*S, T, C)
    if tokens.dim() == 4:
        B, S, T, C = tokens.shape
        tokens = tokens.reshape(B * S, T, C)
    # Infer grid size from T (assume square)
    import math
    side = int(round(math.sqrt(tokens.shape[1])))
    return tokens.detach().cpu().float(), (side, side)
