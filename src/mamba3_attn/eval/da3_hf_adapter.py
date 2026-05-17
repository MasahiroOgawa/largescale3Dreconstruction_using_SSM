"""DA3-compatible loader for our patched .pt checkpoints.

DA3's official evaluator (`python -m depth_anything_3.bench.evaluator`) ends
with::

    from depth_anything_3.api import DepthAnything3
    api = DepthAnything3.from_pretrained(model_path)

That `from_pretrained` comes from `huggingface_hub.PyTorchModelHubMixin` and
only understands two things: a local directory containing
``config.json + model.safetensors``, or an HF repo ID. Our checkpoints are
plain `.pt` files holding ``{"model": state_dict, "cfg": {...}}`` produced by
``train_super.py``. The cfg includes ``variant`` (``mamba3`` | ``vssd``) and
``state_dim``, so we can rebuild the exact patched model.

This module exposes :func:`install_pt_loader` which monkey-patches
``DepthAnything3.from_pretrained`` to detect ``.pt`` paths and route them
through :func:`load_patched_da3`. Behaviour for HF repo IDs and HF-style
local directories is unchanged.

Usage (preferred — via the wrapper script ``scripts/run_da3_bench_eval.py``)::

    install_pt_loader()
    runpy.run_module("depth_anything_3.bench.evaluator", run_name="__main__")

The patch survives only within the parent Python process. The evaluator's
multi-GPU path subprocess-spawns ``python -m depth_anything_3.bench.evaluator``
which bypasses the patch — force single-GPU mode with
``CUDA_VISIBLE_DEVICES=0`` when running patched checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..patch import install_mamba3
from .da3_reference import DEFAULT_HF_MODEL, load_da3

_PATCHED = False
_ORIG_FROM_PRETRAINED = None


def load_patched_da3(ckpt_path: str | Path, device: str = "cuda"):
    """Build a Mamba-3 / VSSD-patched DA3-SMALL and load ``ckpt_path`` into it.

    Reads ``cfg["variant"]`` and ``cfg["state_dim"]`` from the checkpoint to
    decide which operator family to install (with sensible defaults if the
    keys are missing). Returns the DA3 `api` object with the patched model in
    eval mode on `device`.
    """
    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("cfg") if isinstance(state, dict) else None
    if not isinstance(cfg, dict):
        cfg = {}
    variant = cfg.get("variant", "mamba3")
    state_dim = int(cfg.get("state_dim", 64))

    api = load_da3(DEFAULT_HF_MODEL, device="cpu")
    install_mamba3(
        api.model, which="all", variant=variant, state_dim=state_dim,
        use_fused_kernel=(variant == "mamba3"), chunk_size=128,
    )
    missing, unexpected = api.model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"[da3_hf_adapter] WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"[da3_hf_adapter] WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    api = api.to(device)
    api.model.eval()
    api.device = device
    print(f"[da3_hf_adapter] loaded patched DA3-SMALL (variant={variant}, state_dim={state_dim}) "
          f"from {ckpt_path}")
    return api


def install_pt_loader() -> None:
    """Monkey-patch ``DepthAnything3.from_pretrained`` to recognise our .pt ckpts.

    Idempotent — safe to call multiple times. Wraps the original class method
    so behaviour for HF repo IDs and HF-style local directories is unchanged.
    """
    global _PATCHED, _ORIG_FROM_PRETRAINED
    if _PATCHED:
        return

    # DA3's `utils.export/__init__.py` eagerly imports gs (moviepy.editor),
    # glb (trimesh), and colmap (pycolmap). None of those are needed for the
    # `mini_npz` export path the benchmark evaluator uses. Stub each module
    # individually so the rest of `utils.export/__init__.py` loads normally
    # and `export_to_mini_npz` writes `results.npz` for the fuse3d step.
    # This must run before `depth_anything_3.api` is imported.
    import sys
    import types

    def _stub(name: str, attrs: dict) -> None:
        if name in sys.modules:
            return
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    _noop = lambda *a, **kw: None
    _stub("depth_anything_3.utils.export.gs",
          {"export_to_gs_ply": _noop, "export_to_gs_video": _noop})
    _stub("depth_anything_3.utils.export.glb",
          {"export_to_glb": _noop, "_depths_to_world_points_with_colors": _noop})
    _stub("depth_anything_3.utils.export.colmap",
          {"export_to_colmap": _noop})

    from depth_anything_3.api import DepthAnything3

    _ORIG_FROM_PRETRAINED = DepthAnything3.from_pretrained

    def _is_pt_path(p: Any) -> bool:
        if not isinstance(p, (str, Path)):
            return False
        s = str(p)
        return s.endswith(".pt") and Path(s).is_file()

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if _is_pt_path(pretrained_model_name_or_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return load_patched_da3(pretrained_model_name_or_path, device=device)
        return _ORIG_FROM_PRETRAINED.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

    DepthAnything3.from_pretrained = _patched_from_pretrained
    _PATCHED = True
    print("[da3_hf_adapter] DepthAnything3.from_pretrained patched to recognise .pt checkpoints")
