"""Multi-scene ETH3D loader for distillation / depth fine-tune training.

Wraps the single-scene helpers in `eth3d.py` so a training loop can iterate
over many scenes without the single-scene loader's cap on `max_images`.

The `terrains` scene is strictly held-out as the evaluation test set. Any
code path that would include it raises a ValueError — this is enforced at
both download and iterate time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .eth3d import (
    DEFAULT_SCENE,
    _center_crop_resize,
    _crop_resize_box,
    _find_images_root,
    _infer_depth_shape,
    _random_square_crop_box,
    download_eth3d_terrains,
    load_eth3d_scene,
)


def _find_depth_root(scene_dir: Path) -> Path | None:
    """Probe the two ETH3D layouts for `ground_truth_depth/dslr_images`.

    Returns `None` if the scene lacks extracted GT depth. Cached per scene at
    `__init__` so `__getitem__` skips redundant stat() calls.
    """
    candidates = [
        scene_dir / "ground_truth_depth" / "dslr_images",
        scene_dir / scene_dir.name / "ground_truth_depth" / "dslr_images",
    ]
    return next((c for c in candidates if c.exists()), None)

HELD_OUT = "terrains"

TRAIN_SCENES: tuple[str, ...] = (
    "courtyard",
    "facade",
    "office",
    "delivery_area",
    "kicker",
    "pipes",
    "electro",
    "playground",
    "relief",
    "relief_2",
)


def _assert_no_heldout(scenes: Iterable[str]) -> None:
    if HELD_OUT in tuple(scenes):
        raise ValueError(
            f"Scene {HELD_OUT!r} is reserved as eval-only test set "
            f"(ETH3D terrains). Remove it from the training list."
        )


def download_eth3d_scenes(
    data_root: Path,
    scenes: Iterable[str] = TRAIN_SCENES,
    download_depth: bool = False,
) -> dict[str, Path]:
    """Download + extract each scene; return a dict {scene: extract_dir}."""
    scene_list = list(scenes)
    _assert_no_heldout(scene_list)
    out: dict[str, Path] = {}
    for s in scene_list:
        out[s] = download_eth3d_terrains(
            Path(data_root), scene=s, download_depth=download_depth
        )
    return out


@dataclass
class _SceneIndex:
    name: str
    scene_dir: Path
    image_paths: list[Path]
    depth_root: Path | None = None  # cached at __init__; None if scene has no GT depth


class ETH3DMultiSceneDataset(Dataset):
    """Yields random images from a pool of ETH3D scenes.

    Each __getitem__ returns a dict with:
        images: (1, 3, H, W) — a single image reshaped for the SSM3D backbone
            which expects (B, S, 3, H, W). The caller stacks B items into
            (B, 1, 3, H, W).
        scene: scene name
        path: absolute image path

    The dataset length is the total number of images across all scenes. For
    training we iterate indefinitely via a sampler, so __len__ is mainly
    informational.
    """

    def __init__(
        self,
        data_root: Path,
        scenes: Iterable[str] = TRAIN_SCENES,
        image_size: int = 224,
        load_gt_depth: bool = False,
        cache_in_memory: bool = True,
        augment: bool = False,
        augment_seed: int = 0,
    ) -> None:
        scene_list = list(scenes)
        _assert_no_heldout(scene_list)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.load_gt_depth = load_gt_depth
        self.augment = augment
        self._aug_rng = random.Random(augment_seed) if augment else None
        # When augmenting, caching the first sample would freeze the RNG draw.
        self.cache_in_memory = cache_in_memory and not augment
        self._cache: dict[int, dict] = {}
        self.scenes: list[_SceneIndex] = []
        for s in scene_list:
            scene_dir = self.data_root / "eth3d" / s
            if not scene_dir.exists():
                continue
            try:
                img_root = _find_images_root(scene_dir)
            except FileNotFoundError:
                continue
            paths = sorted(img_root.rglob("*.JPG"))
            if not paths:
                continue
            depth_root = _find_depth_root(scene_dir) if load_gt_depth else None
            self.scenes.append(_SceneIndex(s, scene_dir, paths, depth_root))
        if not self.scenes:
            raise RuntimeError(
                f"No ETH3D scenes found under {self.data_root / 'eth3d'}. "
                "Call download_eth3d_scenes first."
            )
        self.flat = [
            (si, p) for si in self.scenes for p in si.image_paths
        ]

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, idx: int) -> dict:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]
        si, path = self.flat[idx]
        from PIL import Image
        import numpy as np

        with Image.open(path) as img:
            orig_w, orig_h = img.size
            rgb_arr = np.asarray(img.convert("RGB"))

        aug = self._sample_aug(orig_h, orig_w) if self.augment else None
        rgb_resized = self._crop_and_resize_rgb(rgb_arr, aug)
        tensor = torch.from_numpy(rgb_resized.astype("float32") / 255.0).permute(2, 0, 1)
        if aug is not None:
            if aug["flip"]:
                tensor = torch.flip(tensor, dims=[-1])
            tensor = _apply_color_jitter(tensor, aug["color"])

        out = {
            "images": tensor.unsqueeze(0),  # (1, 3, H, W) → S=1 view
            "scene": si.name,
            "path": str(path),
        }
        if self.load_gt_depth and si.depth_root is not None:
            gt_path = si.depth_root / path.name
            if gt_path.exists():
                raw = np.fromfile(gt_path, dtype=np.float32)
                shape = _infer_depth_shape(raw.size, (orig_h, orig_w))
                if shape is not None:
                    depth_full = raw.reshape(*shape)
                    depth_full = np.where(
                        np.isfinite(depth_full), depth_full, 0.0
                    ).astype("float32")
                    depth_resized = self._crop_and_resize_depth(
                        depth_full, aug, (orig_h, orig_w)
                    )
                    depth_resized = np.ascontiguousarray(depth_resized)
                    depth_t = torch.from_numpy(depth_resized.copy()).unsqueeze(0)
                    if aug is not None and aug["flip"]:
                        depth_t = torch.flip(depth_t, dims=[-1])
                    valid_t = (torch.isfinite(depth_t) & (depth_t > 0))
                    out["gt_depth"] = depth_t
                    out["valid_mask"] = valid_t
        if self.cache_in_memory:
            self._cache[idx] = out
        return out

    def _sample_aug(self, rgb_h: int, rgb_w: int) -> dict:
        rng = self._aug_rng
        assert rng is not None
        box = _random_square_crop_box(rgb_h, rgb_w, rng, scale_range=(0.6, 1.0))
        return {
            "box_rgb": box,
            "flip": rng.random() < 0.5,
            "color": {
                "brightness": rng.uniform(0.6, 1.4),
                "contrast": rng.uniform(0.6, 1.4),
                "saturation": rng.uniform(0.6, 1.4),
                "hue": rng.uniform(-0.1, 0.1),
            },
        }

    def _crop_and_resize_rgb(
        self, rgb_arr: "np.ndarray", aug: dict | None
    ) -> "np.ndarray":
        if aug is None:
            out, _ = _center_crop_resize(rgb_arr, self.image_size)
            return out
        out, _ = _crop_resize_box(rgb_arr, aug["box_rgb"], self.image_size)
        return out

    def _crop_and_resize_depth(
        self,
        depth_full: "np.ndarray",
        aug: dict | None,
        rgb_hw: tuple[int, int],
    ) -> "np.ndarray":
        if aug is None:
            out, _ = _center_crop_resize(depth_full, self.image_size)
            return out
        dh, dw = depth_full.shape[:2]
        rh, rw = rgb_hw
        l, t, r, b = aug["box_rgb"]
        sx, sy = dw / rw, dh / rh
        box_d = (
            int(round(l * sx)),
            int(round(t * sy)),
            int(round(r * sx)),
            int(round(b * sy)),
        )
        out, _ = _crop_resize_box(depth_full, box_d, self.image_size)
        return out


def _apply_color_jitter(img: torch.Tensor, factors: dict) -> torch.Tensor:
    """Brightness / contrast / saturation / hue jitter on a (3, H, W) tensor in [0, 1]."""
    from torchvision.transforms.functional import (
        adjust_brightness,
        adjust_contrast,
        adjust_hue,
        adjust_saturation,
    )
    img = adjust_brightness(img, factors["brightness"])
    img = adjust_contrast(img, factors["contrast"])
    img = adjust_saturation(img, factors["saturation"])
    img = adjust_hue(img, factors["hue"])
    return img.clamp(0.0, 1.0)


def infinite_sampler(dataset: Dataset, seed: int = 0) -> Iterable[int]:
    """Yield shuffled indices forever — for training with a fixed step budget."""
    rng = random.Random(seed)
    n = len(dataset)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        yield from perm
