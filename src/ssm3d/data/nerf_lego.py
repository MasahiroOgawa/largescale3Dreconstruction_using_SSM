"""NeRF-Synthetic `lego` scene — fallback multi-view source (~100 MB).

Provides a simple per-scene interface even though the only guaranteed direct
URL is for the full archive. We pull only `lego` out of the zip to keep disk
usage minimal.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ._util import download, require_free

# Mirrored on HuggingFace Hub: nerf-synthetic individual-file tarballs.
# The public GDrive link from the NeRF repo isn't browseable without auth.
# We fall back to the HF dataset mirror (single scene zip is available).
LEGO_URL = (
    "https://huggingface.co/datasets/cmudrc/nerf_synthetic/resolve/main/lego.zip"
)


@dataclass
class NeRFSample:
    name: str
    images: torch.Tensor  # (N, 3, H, W)
    c2w: torch.Tensor  # (N, 4, 4)


def download_nerf_lego(data_root: Path) -> Path:
    data_root = Path(data_root)
    require_free(data_root, 2 * 2**30)

    archive = data_root / "nerf_synthetic" / "lego.zip"
    archive = download(LEGO_URL, archive, min_bytes=10_000_000)

    out_dir = data_root / "nerf_synthetic" / "lego"
    if not (out_dir / ".extracted").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
        (out_dir / ".extracted").write_text("ok")
    # NeRF-synthetic lego often unzips to a nested lego/ dir
    nested = out_dir / "lego"
    return nested if nested.exists() else out_dir


def load_nerf_lego_scene(scene_dir: Path, split: str = "train", max_images: int = 4, image_size: int = 224) -> NeRFSample:
    scene_dir = Path(scene_dir)
    meta_path = scene_dir / f"transforms_{split}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    meta = json.loads(meta_path.read_text())

    imgs, poses = [], []
    for frame in meta["frames"][:max_images]:
        # NeRF uses relative path without extension; files are .png
        rel = frame["file_path"]
        img_path = scene_dir / f"{rel}.png"
        if not img_path.exists():
            img_path = scene_dir / rel
            if img_path.suffix == "":
                img_path = img_path.with_suffix(".png")
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
        poses.append(torch.tensor(frame["transform_matrix"], dtype=torch.float32))
    return NeRFSample(
        name="lego",
        images=torch.stack(imgs) if imgs else torch.empty(0, 3, image_size, image_size),
        c2w=torch.stack(poses) if poses else torch.empty(0, 4, 4),
    )
