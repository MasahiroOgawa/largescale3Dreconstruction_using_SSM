"""7Scenes (DA3-BENCH) loader — multi-view eval format.

Layout (from `depth-anything/DA3-BENCH/7scenes.zip`):

    7scenes/
    └── 7Scenes/
        ├── {scene}/seq-01/         (or seq-02 for `stairs`)
        │   ├── frame-XXXXXX.color.png   # 640x480 RGB
        │   ├── frame-XXXXXX.depth.png   # 16-bit, mm; 65535 = invalid
        │   └── frame-XXXXXX.pose.txt    # camera-to-world (need to invert for w2c)
        └── meshes/{scene}.ply           # GT mesh (recon target)

Fixed intrinsics for all images: fx=fy=585, cx=320, cy=240 (DA3 constants).
Returns the same `ETH3DSample` / `ETH3DCams` shapes as `eth3d.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .eth3d import ETH3DSample, _center_crop_resize
from ..eval.eth3d_gt import ETH3DCams


SEVENSCENES_FX = 585.0
SEVENSCENES_FY = 585.0
SEVENSCENES_CX = 320.0
SEVENSCENES_CY = 240.0
SEVENSCENES_DEPTH_INVALID = 65535
DEFAULT_DATA_DIR = "da3_bench/7scenes"
SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")


def list_sevenscenes_scenes() -> tuple[str, ...]:
    return SCENES


def _seq_dir(data_root: Path, scene: str) -> Path:
    seq = "seq-02" if scene == "stairs" else "seq-01"
    return Path(data_root) / DEFAULT_DATA_DIR / "7Scenes" / scene / seq


def _gt_mesh_path(data_root: Path, scene: str) -> Path:
    return Path(data_root) / DEFAULT_DATA_DIR / "7Scenes" / "meshes" / f"{scene}.ply"


def _shared_intrinsic() -> np.ndarray:
    return np.array(
        [[SEVENSCENES_FX, 0.0, SEVENSCENES_CX],
         [0.0, SEVENSCENES_FY, SEVENSCENES_CY],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_sevenscenes_scene(
    data_root: Path,
    scene: str,
    max_images: int = 12,
    image_size: int = 504,
    load_gt_depth: bool = True,
    frame_stride: int = 1,
) -> ETH3DSample:
    """Load up to `max_images` frames from a 7Scenes sequence."""
    sd = _seq_dir(data_root, scene)
    n_total = 500 if scene == "stairs" else 1000

    image_paths: list[Path] = []
    for i in range(0, n_total, frame_stride):
        img = sd / f"frame-{i:06d}.color.png"
        pose = sd / f"frame-{i:06d}.pose.txt"
        if img.exists() and pose.exists():
            image_paths.append(img)
        if len(image_paths) >= max_images:
            break

    imgs: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    orig_sizes: list[tuple[int, int]] = []

    for p in image_paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"))
        h, w = arr.shape[:2]
        orig_sizes.append((h, w))
        rgb_resized, _ = _center_crop_resize(arr, image_size)
        imgs.append(torch.from_numpy(rgb_resized.astype(np.float32) / 255.0).permute(2, 0, 1))

        if not load_gt_depth:
            continue
        d_path = p.with_name(p.name.replace(".color.", ".depth."))
        if not d_path.exists():
            depths.append(torch.full((image_size, image_size), float("nan")))
            valid_masks.append(torch.zeros(image_size, image_size, dtype=torch.bool))
            continue
        with Image.open(d_path) as dim:
            d_raw = np.asarray(dim).astype(np.float32)
        d_raw[d_raw == SEVENSCENES_DEPTH_INVALID] = 0.0
        d_raw = d_raw / 1000.0  # mm → m
        d_resized, _ = _center_crop_resize(d_raw, image_size)
        d_resized = np.ascontiguousarray(d_resized)
        valid = np.isfinite(d_resized) & (d_resized > 0)
        depths.append(torch.from_numpy(d_resized))
        valid_masks.append(torch.from_numpy(valid))

    stack = torch.stack(imgs, dim=0) if imgs else torch.empty(0, 3, image_size, image_size)
    gt_depth = torch.stack(depths, dim=0) if depths else None
    valid_mask = torch.stack(valid_masks, dim=0) if valid_masks else None

    return ETH3DSample(
        name=scene,
        images=stack,
        image_paths=image_paths,
        gt_depth=gt_depth,
        valid_mask=valid_mask,
        orig_sizes=orig_sizes,
    )


def load_sevenscenes_cams(
    data_root: Path,
    scene: str,
    image_size: int,
    image_names: Optional[list[str]] = None,
) -> ETH3DCams:
    """Per-image intrinsics (rescaled) and extrinsics (w2c) for 7Scenes."""
    sd = _seq_dir(data_root, scene)
    K_shared = _shared_intrinsic()

    if image_names is None:
        n_total = 500 if scene == "stairs" else 1000
        image_names = []
        for i in range(n_total):
            p = sd / f"frame-{i:06d}.color.png"
            if p.exists() and (sd / f"frame-{i:06d}.pose.txt").exists():
                image_names.append(p.name)

    # Shared image dims (640x480) → crop=480, scale = image_size / 480
    h_orig, w_orig = 480, 640
    s = min(h_orig, w_orig)
    crop_left = (w_orig - s) // 2
    crop_top = (h_orig - s) // 2
    scale = image_size / s

    K = K_shared.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = (K[0, 2] - crop_left) * scale
    K[1, 2] = (K[1, 2] - crop_top) * scale

    intrinsics: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    orig_sizes: dict[str, tuple[int, int]] = {}

    for name in image_names:
        stem = Path(name).stem  # e.g. frame-000000.color
        idx_str = stem.replace(".color", "")
        pose_path = sd / f"{idx_str}.pose.txt"
        c2w = np.loadtxt(pose_path).astype(np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"Unexpected pose shape {c2w.shape} at {pose_path}")
        E = np.linalg.inv(c2w).astype(np.float32)  # w2c

        intrinsics[name] = K.copy()
        extrinsics[name] = E
        orig_sizes[name] = (h_orig, w_orig)

    return ETH3DCams(intrinsics=intrinsics, extrinsics=extrinsics, orig_sizes=orig_sizes)
