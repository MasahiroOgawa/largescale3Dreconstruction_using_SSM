"""HiRoom (DA3-BENCH) loader — multi-view eval format.

Layout (from `depth-anything/DA3-BENCH/hiroom.zip`):

    hiroom/
    ├── data/{date}/{capture}/{cam_dir}/
    │   ├── image/{stem}.jpg
    │   ├── depth/{stem}.png         # 16-bit, pixel/65535 * 100 → meters
    │   ├── pose/{stem}.npy          # (4, 4) world-to-camera
    │   ├── cam_K.npy                # (3, 3) shared intrinsics
    │   └── aliasing_mask/{stem}.png
    ├── fused_pcd/{capture}-{cam_dir}.ply  # GT fused point cloud (also alt scene-name forms)
    └── selected_scene_list_val.txt        # 29 scenes, one path per line

Returns the same `ETH3DSample` / `ETH3DCams` shapes as `eth3d.py` so the
existing eval scripts can dispatch on `--dataset hiroom` without other
changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .eth3d import ETH3DSample, _center_crop_resize
from ..eval.eth3d_gt import ETH3DCams


HIROOM_DEPTH_SCALE = 100.0 / 65535.0  # 16-bit PNG → meters
DEFAULT_DATA_DIR = "da3_bench/hiroom"


def list_hiroom_scenes(data_root: Path) -> list[str]:
    """Read the val scene list (one path per line, e.g. `20241230/828738/cam_sampled_08`)."""
    list_path = Path(data_root) / DEFAULT_DATA_DIR / "selected_scene_list_val.txt"
    return [s for s in list_path.read_text().splitlines() if s.strip()]


def _scene_dir(data_root: Path, scene: str) -> Path:
    return Path(data_root) / DEFAULT_DATA_DIR / "data" / scene


def _gt_pcd_path(data_root: Path, scene: str) -> Path:
    """DA3 reference forms the PLY name from the last 3 path segments joined by `-`."""
    parts = scene.split("/")[-3:]
    name = "-".join(parts)
    return Path(data_root) / DEFAULT_DATA_DIR / "fused_pcd" / f"{name}.ply"


def load_hiroom_scene(
    data_root: Path,
    scene: str,
    max_images: int = 12,
    image_size: int = 504,
    load_gt_depth: bool = True,
) -> ETH3DSample:
    """Load up to `max_images` frames from a HiRoom scene at `image_size`."""
    sd = _scene_dir(data_root, scene)
    image_dir = sd / "image"
    depth_dir = sd / "depth"
    pose_dir = sd / "pose"

    # Frames where pose exists (DA3 reference filter). Sort by stem to keep
    # neighbouring frames adjacent — relevant for multi-view consistency.
    image_paths = sorted(p for p in image_dir.iterdir() if (pose_dir / f"{p.stem}.npy").exists())
    image_paths = image_paths[:max_images]

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
        d_path = depth_dir / f"{p.stem}.png"
        if not d_path.exists():
            depths.append(torch.full((image_size, image_size), float("nan")))
            valid_masks.append(torch.zeros(image_size, image_size, dtype=torch.bool))
            continue
        with Image.open(d_path) as dim:
            d_raw = np.asarray(dim).astype(np.float32) * HIROOM_DEPTH_SCALE
        d_resized, _ = _center_crop_resize(d_raw, image_size)
        d_resized = np.ascontiguousarray(d_resized)
        valid = np.isfinite(d_resized) & (d_resized > 0)
        depths.append(torch.from_numpy(d_resized))
        valid_masks.append(torch.from_numpy(valid))

    stack = torch.stack(imgs, dim=0) if imgs else torch.empty(0, 3, image_size, image_size)
    gt_depth = torch.stack(depths, dim=0) if depths else None
    valid_mask = torch.stack(valid_masks, dim=0) if valid_masks else None

    return ETH3DSample(
        name=scene.replace("/", "_"),
        images=stack,
        image_paths=image_paths,
        gt_depth=gt_depth,
        valid_mask=valid_mask,
        orig_sizes=orig_sizes,
    )


def load_hiroom_cams(
    data_root: Path,
    scene: str,
    image_size: int,
    image_names: Optional[list[str]] = None,
) -> ETH3DCams:
    """Per-image intrinsics (rescaled for center-crop+resize) and extrinsics (w2c)."""
    sd = _scene_dir(data_root, scene)
    K_shared = np.load(sd / "cam_K.npy").astype(np.float32)

    pose_dir = sd / "pose"
    image_dir = sd / "image"
    if image_names is None:
        image_names = sorted(p.name for p in image_dir.iterdir() if (pose_dir / f"{p.stem}.npy").exists())

    intrinsics: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    orig_sizes: dict[str, tuple[int, int]] = {}

    for name in image_names:
        stem = Path(name).stem
        img_path = image_dir / name
        with Image.open(img_path) as im:
            w, h = im.size
        s = min(h, w)
        crop_left = (w - s) // 2
        crop_top = (h - s) // 2
        scale = image_size / s

        K = K_shared.copy()
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] = (K[0, 2] - crop_left) * scale
        K[1, 2] = (K[1, 2] - crop_top) * scale

        E = np.load(pose_dir / f"{stem}.npy").astype(np.float32)
        if E.shape != (4, 4):
            raise ValueError(f"Unexpected pose shape {E.shape} at {pose_dir / f'{stem}.npy'}")

        intrinsics[name] = K
        extrinsics[name] = E
        orig_sizes[name] = (h, w)

    return ETH3DCams(intrinsics=intrinsics, extrinsics=extrinsics, orig_sizes=orig_sizes)
