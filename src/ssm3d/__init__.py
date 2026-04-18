"""ssm3d — Mamba-3 attention swapped into Depth-Anything-3 for large-scale 3D reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

_DA3_SRC = Path(__file__).resolve().parent.parent.parent / "third_party" / "depth-anything-3" / "src"
if _DA3_SRC.exists() and str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

__all__ = ["_DA3_SRC"]
