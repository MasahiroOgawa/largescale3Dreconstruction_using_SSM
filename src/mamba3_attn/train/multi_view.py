"""Multi-view scene sampler unified across ETH3D / HiRoom / 7Scenes.

Yields one batch = one scene's S views, with GT depth + intrinsics +
extrinsics where available. Designed for DA3 paper-style training where
the network consumes (B, S, 3, H, W) batches.

Train splits (PLAN §15.54):
- ETH3D: all non-terrains scenes (10 scenes)
- HiRoom: first 25 of selected_scene_list_val.txt
- 7Scenes: chess, fire, heads, office (sub-sample frames)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from ..data.bench import load_bench_cams, load_bench_scene
from ..data.eth3d_multi import _apply_color_jitter
from ..data.eth3d_multi import TRAIN_SCENES as ETH3D_TRAIN_SCENES
from ..data.hiroom import list_hiroom_scenes


SEVENSCENES_TRAIN = ("chess", "fire", "heads", "office")
SEVENSCENES_EVAL = ("pumpkin", "redkitchen", "stairs")


@dataclass
class MultiViewBatch:
    images: Tensor          # (S, 3, H, W), float in [0, 1]
    gt_depth: Tensor        # (S, H, W), meters
    gt_K: Tensor            # (S, 3, 3) at image_size
    gt_w2c: Tensor          # (S, 4, 4)
    valid: Tensor           # (S, H, W) bool
    dataset: str
    scene: str


def _build_train_scene_list(data_root: Path) -> list[tuple[str, str]]:
    """Return list of (dataset, scene) pairs in the train split."""
    out: list[tuple[str, str]] = []
    out += [("eth3d", s) for s in ETH3D_TRAIN_SCENES]
    try:
        hiroom = list_hiroom_scenes(data_root)[:25]
        out += [("hiroom", s) for s in hiroom]
    except FileNotFoundError:
        pass
    out += [("7scenes", s) for s in SEVENSCENES_TRAIN]
    return out


def _stack_cams(cams, names: list[str]) -> tuple[Tensor, Tensor]:
    Ks = torch.from_numpy(np.stack([cams.intrinsics[n] for n in names])).float()
    Es = torch.from_numpy(np.stack([cams.extrinsics[n] for n in names])).float()
    return Ks, Es


def sample_scene(
    dataset: str,
    scene: str,
    data_root: Path,
    n_views: int,
    image_size: int,
    rng: random.Random,
) -> MultiViewBatch | None:
    """Load one batch from one scene. Returns None if scene unusable.

    For 7Scenes (long sequences), picks a random window-stride to vary the
    multi-view configuration each sample.
    """
    frame_stride = 1
    if dataset == "7scenes":
        frame_stride = rng.randint(20, 100)
    sample = load_bench_scene(
        dataset, scene, data_root, max_images=n_views,
        image_size=image_size, load_gt_depth=True, frame_stride=frame_stride,
    )
    if sample.images.shape[0] < n_views:
        return None
    if sample.gt_depth is None or sample.valid_mask is None:
        return None
    names = [p.name for p in sample.image_paths]
    cams = load_bench_cams(dataset, scene, data_root, image_size=image_size, image_names=names)
    if any(n not in cams.intrinsics or n not in cams.extrinsics for n in names):
        return None
    Ks, Es = _stack_cams(cams, names)
    return MultiViewBatch(
        images=sample.images,
        gt_depth=sample.gt_depth,
        gt_K=Ks,
        gt_w2c=Es,
        valid=sample.valid_mask,
        dataset=dataset,
        scene=scene,
    )


def multi_view_iterator(
    data_root: Path,
    n_views: int = 4,
    image_size: int = 504,
    seed: int = 0,
    require_gt: bool = True,
) -> Iterator[MultiViewBatch]:
    """Yield batches forever, cycling through train scenes in shuffled order.

    Skips scenes that can't produce a valid batch (insufficient views, no GT).
    """
    rng = random.Random(seed)
    scenes = _build_train_scene_list(data_root)
    if not scenes:
        raise RuntimeError(f"No train scenes found under {data_root}")
    print(f"[multi_view] {len(scenes)} train scenes "
          f"(eth3d:{sum(1 for d,_ in scenes if d=='eth3d')}, "
          f"hiroom:{sum(1 for d,_ in scenes if d=='hiroom')}, "
          f"7scenes:{sum(1 for d,_ in scenes if d=='7scenes')})")
    while True:
        order = scenes[:]
        rng.shuffle(order)
        for ds, sc in order:
            try:
                batch = sample_scene(ds, sc, data_root, n_views, image_size, rng)
            except Exception as e:
                print(f"[multi_view] skip {ds}/{sc}: {e}")
                continue
            if batch is None:
                continue
            if require_gt and (batch.gt_depth is None or batch.valid.float().mean() < 0.05):
                continue
            yield batch


def load_full_scene_cache(
    dataset: str,
    scene: str,
    data_root: Path,
    image_size: int,
    candidate_views: int = 256,
    frame_stride: int = 1,
) -> MultiViewBatch:
    """Load *all* candidate views of one scene once into an in-memory `MultiViewBatch`.

    For ETH3D / HiRoom, `candidate_views` should be ≥ the scene's view count
    (≈ 42 for ETH3D `terrains`, ≤ 30 per HiRoom scene). For 7Scenes (1000 frames)
    pass a `frame_stride` ≥ 20 to subsample down to ~50 candidates before split.
    """
    sample = load_bench_scene(
        dataset, scene, data_root, max_images=candidate_views,
        image_size=image_size, load_gt_depth=True, frame_stride=frame_stride,
    )
    if sample.gt_depth is None or sample.valid_mask is None:
        raise RuntimeError(f"{dataset}/{scene}: no GT depth — required for per-scene overfit")
    names = [p.name for p in sample.image_paths]
    cams = load_bench_cams(dataset, scene, data_root, image_size=image_size, image_names=names)
    if any(n not in cams.intrinsics or n not in cams.extrinsics for n in names):
        raise RuntimeError(f"{dataset}/{scene}: missing per-view intrinsics or extrinsics")
    Ks, Es = _stack_cams(cams, names)
    return MultiViewBatch(
        images=sample.images,
        gt_depth=sample.gt_depth,
        gt_K=Ks,
        gt_w2c=Es,
        valid=sample.valid_mask,
        dataset=dataset,
        scene=scene,
    )


def _slice_batch(cache: MultiViewBatch, indices: list[int]) -> MultiViewBatch:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return MultiViewBatch(
        images=cache.images.index_select(0, idx),
        gt_depth=cache.gt_depth.index_select(0, idx),
        gt_K=cache.gt_K.index_select(0, idx),
        gt_w2c=cache.gt_w2c.index_select(0, idx),
        valid=cache.valid.index_select(0, idx),
        dataset=cache.dataset,
        scene=cache.scene,
    )


def _photometric_aug(images: Tensor, rng: random.Random) -> Tensor:
    """Per-view independent color jitter on `(S, 3, H, W)` images.

    Geometric augmentation (hflip / random crop) would invalidate the cached
    `gt_K` / `gt_w2c` consumed by L_C / L_P in DA3's §3.3, so this stays
    photometric-only.
    """
    factors_list = [
        {
            "brightness": rng.uniform(0.6, 1.4),
            "contrast": rng.uniform(0.6, 1.4),
            "saturation": rng.uniform(0.6, 1.4),
            "hue": rng.uniform(-0.1, 0.1),
        }
        for _ in range(images.shape[0])
    ]
    return torch.stack(
        [_apply_color_jitter(images[i], f) for i, f in enumerate(factors_list)],
        dim=0,
    )


def iter_single_scene(
    cache: MultiViewBatch,
    train_indices: list[int],
    n_views: int,
    augment: bool = True,
    seed: int = 0,
) -> Iterator[MultiViewBatch]:
    """Yield batches forever, sampling `n_views` indices from `train_indices` per step.

    `cache` is produced by `load_full_scene_cache(...)`. `train_indices` is the
    train half of `view_split.split_views(...)`. With augmentation enabled, each
    yielded batch's images receive per-view color jitter; intrinsics/extrinsics
    are passed through unchanged so L_C / L_P remain valid.
    """
    if len(train_indices) < n_views:
        raise ValueError(
            f"train_indices has {len(train_indices)} views, need ≥ n_views={n_views}"
        )
    rng = random.Random(seed)
    while True:
        picked = rng.sample(train_indices, n_views)
        batch = _slice_batch(cache, picked)
        if augment:
            batch = MultiViewBatch(
                images=_photometric_aug(batch.images, rng),
                gt_depth=batch.gt_depth, gt_K=batch.gt_K, gt_w2c=batch.gt_w2c,
                valid=batch.valid, dataset=batch.dataset, scene=batch.scene,
            )
        yield batch
