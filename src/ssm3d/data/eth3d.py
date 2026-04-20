"""ETH3D low-res multi-view downloader + loader.

Uses the `terrains` scene (~0.2 GB undistorted images). Optional depth download
(`terrains_dslr_depth.7z`) provides per-DSLR-image GT depth aligned with the
RGB images. GT depth format is ETH3D-standard: raw float32 row-major binaries
at `ground_truth_depth/dslr_images/{image_name}`, one file per image with the
same filename as the JPG (confirmed from DA3's bench loader and ETH3D docs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import py7zr
import torch
from PIL import Image

from ._util import download, require_free

BASE = "https://www.eth3d.net/data"
DEFAULT_SCENE = "terrains"


@dataclass
class ETH3DSample:
    name: str
    images: torch.Tensor  # (N, 3, H, W) float32 in [0, 1]
    image_paths: list[Path]
    gt_depth: Optional[torch.Tensor] = None  # (N, H, W) float32, metres, 0/NaN=invalid
    valid_mask: Optional[torch.Tensor] = None  # (N, H, W) bool, True=valid
    orig_sizes: list[tuple[int, int]] = field(default_factory=list)  # (H, W) per image


def _extract_7z(archive: Path, out_dir: Path) -> Path:
    """Extract a 7z archive to `out_dir`. Returns the root of the extracted content.

    Uses a per-archive marker so extracting a second archive (e.g. GT depth) into
    the same scene dir is not skipped by an earlier archive's marker.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f".extracted_{archive.stem}"
    if marker.exists():
        return out_dir
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(path=out_dir)
    marker.write_text("ok")
    return out_dir


def download_eth3d_terrains(
    data_root: Path,
    scene: str = DEFAULT_SCENE,
    download_depth: bool = False,
) -> Path:
    """Download + extract a single ETH3D low-res scene (undistorted images only).

    When `download_depth=True`, also fetches `{scene}_rig_depth.7z` and extracts
    it under the same scene directory so GT depth maps sit alongside the images.

    Returns the extracted scene directory (containing the `images/` tree).
    """
    data_root = Path(data_root)
    budget = (3 if download_depth else 2) * 2**30
    require_free(data_root, budget)

    archive_url = f"{BASE}/{scene}_dslr_undistorted.7z"
    archive_path = data_root / "eth3d" / f"{scene}_dslr_undistorted.7z"
    archive_path = download(archive_url, archive_path, min_bytes=10_000_000)

    extract_root = data_root / "eth3d" / scene
    _extract_7z(archive_path, extract_root)

    if download_depth:
        depth_url = f"{BASE}/{scene}_dslr_depth.7z"
        depth_archive = data_root / "eth3d" / f"{scene}_dslr_depth.7z"
        depth_archive = download(depth_url, depth_archive, min_bytes=100_000)
        _extract_7z(depth_archive, extract_root)

    return extract_root


def _find_images_root(scene_dir: Path) -> Path:
    candidates = [
        scene_dir / "images",
        scene_dir / scene_dir.name / "images",
        scene_dir,
    ]
    for c in candidates:
        if c.exists() and any(c.rglob("*.JPG")):
            return c
    raise FileNotFoundError(f"No .JPG images found under {scene_dir}")


def _center_crop_resize(arr: np.ndarray, image_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Center-square crop then resize. Returns resized array + crop box (l,t,r,b) in original pixels."""
    h, w = arr.shape[:2]
    s = min(h, w)
    left, top = (w - s) // 2, (h - s) // 2
    right, bottom = left + s, top + s
    cropped = arr[top:bottom, left:right]
    if cropped.ndim == 3:
        im = Image.fromarray(cropped)
        im = im.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(im), (left, top, right, bottom)
    else:
        # nearest for depth (preserves invalid 0/inf markers)
        im = Image.fromarray(cropped, mode="F")
        im = im.resize((image_size, image_size), Image.NEAREST)
        return np.asarray(im, dtype=np.float32), (left, top, right, bottom)


def _infer_depth_shape(n_floats: int, jpg_hw: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Solve H*W == n_floats with aspect ratio matching the JPG.

    ETH3D GT depth is stored at the pre-undistortion sensor resolution (e.g.
    4032x6048 for Nikon 24MP), while the saved undistorted JPG has slightly
    different dims (undistortion padding). Aspect ratio is preserved, so we
    solve for the 3:2-ish integer pair that matches `n_floats`.
    """
    jpg_h, jpg_w = jpg_hw
    if n_floats == jpg_h * jpg_w:
        return jpg_hw
    ratio = jpg_w / jpg_h
    h = int(round((n_floats / ratio) ** 0.5))
    for dh in range(-4, 5):
        hh = h + dh
        if hh <= 0 or n_floats % hh:
            continue
        ww = n_floats // hh
        if abs(ww / hh - ratio) < 0.02:
            return (hh, ww)
    return None


def load_eth3d_scene(
    scene_dir: Path,
    max_images: int = 4,
    image_size: int = 224,
    load_gt_depth: bool = False,
) -> ETH3DSample:
    """Load up to `max_images` resized images (optionally with GT depth).

    GT depth layout (per ETH3D):
        {scene_dir}/ground_truth_depth/dslr_images/{image_name.JPG}  (float32 raw)

    Depth files are row-major float32 at the camera-model's (W, H) from
    `cameras.txt`, which differs slightly from the saved JPG dimensions due to
    undistortion padding. We reshape depth at the camera plane and center-square
    crop + resize both RGB and depth independently to `image_size`; they remain
    pixel-aligned to within the undistortion padding (a few pixels at full
    resolution, sub-pixel at 224 resolution).
    Invalid depths (0 or non-finite) become `valid_mask=False`.
    """
    scene_dir = Path(scene_dir)
    img_root = _find_images_root(scene_dir)

    paths = sorted(img_root.rglob("*.JPG"))[:max_images]
    imgs: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    orig_sizes: list[tuple[int, int]] = []

    depth_root_candidates = [
        scene_dir / "ground_truth_depth" / "dslr_images",
        scene_dir / scene_dir.name / "ground_truth_depth" / "dslr_images",
    ]
    depth_root = next((c for c in depth_root_candidates if c.exists()), None)

    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size
        orig_sizes.append((orig_h, orig_w))
        arr = np.asarray(Image.open(p).convert("RGB"))
        rgb_resized, _ = _center_crop_resize(arr, image_size)
        imgs.append(torch.from_numpy(rgb_resized.astype(np.float32) / 255.0).permute(2, 0, 1))

        if load_gt_depth and depth_root is not None:
            gt_path = depth_root / p.name
            if gt_path.exists():
                raw = np.fromfile(gt_path, dtype=np.float32)
                shape = _infer_depth_shape(raw.size, (orig_h, orig_w))
                if shape is None:
                    depths.append(torch.full((image_size, image_size), float("nan")))
                    valid_masks.append(torch.zeros(image_size, image_size, dtype=torch.bool))
                    continue
                depth_full = raw.reshape(*shape)
                depth_full = np.where(np.isfinite(depth_full), depth_full, 0.0).astype(np.float32)
                depth_resized, _ = _center_crop_resize(depth_full, image_size)
                depth_resized = np.ascontiguousarray(depth_resized)
                valid = np.isfinite(depth_resized) & (depth_resized > 0)
                depths.append(torch.from_numpy(depth_resized))
                valid_masks.append(torch.from_numpy(valid))
            else:
                depths.append(torch.full((image_size, image_size), float("nan")))
                valid_masks.append(torch.zeros(image_size, image_size, dtype=torch.bool))

    stack = torch.stack(imgs, dim=0) if imgs else torch.empty(0, 3, image_size, image_size)
    gt_depth = torch.stack(depths, dim=0) if depths else None
    valid_mask = torch.stack(valid_masks, dim=0) if valid_masks else None

    return ETH3DSample(
        name=scene_dir.name,
        images=stack,
        image_paths=paths,
        gt_depth=gt_depth,
        valid_mask=valid_mask,
        orig_sizes=orig_sizes,
    )
