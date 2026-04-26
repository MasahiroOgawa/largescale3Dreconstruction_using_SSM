"""ssm3d — Mamba-3 attention swapped into Depth-Anything-3 for large-scale 3D reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

_THIRD_PARTY = Path(__file__).resolve().parent.parent.parent / "third_party"

_DA3_SRC = _THIRD_PARTY / "depth-anything-3" / "src"
if _DA3_SRC.exists() and str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

# Official Mamba-3 (state-spaces/mamba) — for the SISO Triton kernel
# (`mamba_ssm.ops.triton.mamba3.mamba3_siso_combined`) used in the
# efficiency benchmark and as a reference implementation. Per-token
# A + lambda gating + three-term mask are all in `Mamba3` directly.
#
# Mamba-3 only needs Triton kernels (no C++/CUDA compilation), but
# `mamba_ssm/__init__.py` imports `selective_scan_cuda` (Mamba-1's
# compiled extension). Stub it in sys.modules so the package init
# succeeds without building C++ extensions.
_MAMBA_SSM_SRC = _THIRD_PARTY / "mamba-ssm"
if _MAMBA_SSM_SRC.exists() and str(_MAMBA_SSM_SRC) not in sys.path:
    import types
    if "selective_scan_cuda" not in sys.modules:
        sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
    if "causal_conv1d_cuda" not in sys.modules:
        sys.modules["causal_conv1d_cuda"] = types.ModuleType("causal_conv1d_cuda")
    sys.path.insert(0, str(_MAMBA_SSM_SRC))

__all__ = ["_DA3_SRC", "_MAMBA_SSM_SRC"]
