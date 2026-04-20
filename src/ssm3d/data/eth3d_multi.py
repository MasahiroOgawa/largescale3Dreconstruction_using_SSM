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
    _find_images_root,
    download_eth3d_terrains,
    load_eth3d_scene,
)

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
    ) -> None:
        scene_list = list(scenes)
        _assert_no_heldout(scene_list)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.load_gt_depth = load_gt_depth
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
            self.scenes.append(_SceneIndex(s, scene_dir, paths))
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
        si, path = self.flat[idx]
        # Reuse load_eth3d_scene with a one-image slice — preserves the same
        # center-crop + resize pipeline used at eval time.
        sample = load_eth3d_scene(
            si.scene_dir,
            max_images=1,
            image_size=self.image_size,
            load_gt_depth=self.load_gt_depth,
        )
        # load_eth3d_scene uses sorted(rglob) — to pick a specific image we
        # rerun with a stricter slice rather than paging. For training
        # randomness, the caller shuffles indices; the inner loader's "first
        # max_images=1" is stable across calls, so we remap by binding the
        # full list and reading exactly the intended path.
        from PIL import Image
        import numpy as np
        from .eth3d import _center_crop_resize

        arr = np.asarray(Image.open(path).convert("RGB"))
        rgb_resized, _ = _center_crop_resize(arr, self.image_size)
        tensor = torch.from_numpy(rgb_resized.astype("float32") / 255.0).permute(2, 0, 1)

        out = {
            "images": tensor.unsqueeze(0),  # (1, 3, H, W) → S=1 view
            "scene": si.name,
            "path": str(path),
        }
        if self.load_gt_depth and sample.gt_depth is not None:
            # We loaded only one image from the scene above; pick the depth
            # for this specific path if it matches. If the scene dir has depth,
            # look it up directly for reliability:
            from .eth3d import _infer_depth_shape
            depth_root_candidates = [
                si.scene_dir / "ground_truth_depth" / "dslr_images",
                si.scene_dir / si.scene_dir.name / "ground_truth_depth" / "dslr_images",
            ]
            depth_root = next((c for c in depth_root_candidates if c.exists()), None)
            if depth_root is not None:
                gt_path = depth_root / path.name
                if gt_path.exists():
                    import numpy as np
                    orig = Image.open(path)
                    orig_w, orig_h = orig.size
                    orig.close()
                    raw = np.fromfile(gt_path, dtype=np.float32)
                    shape = _infer_depth_shape(raw.size, (orig_h, orig_w))
                    if shape is not None:
                        depth_full = raw.reshape(*shape)
                        depth_full = np.where(
                            np.isfinite(depth_full), depth_full, 0.0
                        ).astype("float32")
                        depth_resized, _ = _center_crop_resize(
                            depth_full, self.image_size
                        )
                        depth_resized = np.ascontiguousarray(depth_resized)
                        valid = (np.isfinite(depth_resized) & (depth_resized > 0))
                        out["gt_depth"] = torch.from_numpy(depth_resized).unsqueeze(0)
                        out["valid_mask"] = torch.from_numpy(valid).unsqueeze(0)
        return out


def infinite_sampler(dataset: Dataset, seed: int = 0) -> Iterable[int]:
    """Yield shuffled indices forever — for training with a fixed step budget."""
    rng = random.Random(seed)
    n = len(dataset)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        yield from perm
