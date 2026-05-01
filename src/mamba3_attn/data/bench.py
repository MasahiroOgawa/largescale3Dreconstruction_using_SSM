"""Dataset dispatcher for multi-view benchmark eval.

Lets eval scripts pick `--dataset {eth3d,hiroom,7scenes} --scene <name>` and
get back the same `ETH3DSample` / `ETH3DCams` shapes regardless of source.

Default scenes:
- eth3d: `terrains` (the held-out scene used throughout PLAN §15.x)
- hiroom: first entry of `selected_scene_list_val.txt`
- 7scenes: `chess`
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..eval.eth3d_gt import ETH3DCams, load_eth3d_cams
from .eth3d import ETH3DSample, download_eth3d_terrains, load_eth3d_scene
from .hiroom import list_hiroom_scenes, load_hiroom_cams, load_hiroom_scene
from .sevenscenes import load_sevenscenes_cams, load_sevenscenes_scene


DATASETS = ("eth3d", "hiroom", "7scenes")


def default_scene(dataset: str, data_root: Path) -> str:
    if dataset == "eth3d":
        return "terrains"
    if dataset == "hiroom":
        return list_hiroom_scenes(data_root)[0]
    if dataset == "7scenes":
        return "chess"
    raise ValueError(f"Unknown dataset {dataset!r}")


def load_bench_scene(
    dataset: str,
    scene: str,
    data_root: Path,
    max_images: int,
    image_size: int,
    load_gt_depth: bool,
    frame_stride: int = 1,
) -> ETH3DSample:
    if dataset == "eth3d":
        scene_dir = download_eth3d_terrains(data_root, scene=scene, download_depth=load_gt_depth)
        return load_eth3d_scene(
            scene_dir, max_images=max_images, image_size=image_size, load_gt_depth=load_gt_depth,
        )
    if dataset == "hiroom":
        return load_hiroom_scene(
            data_root, scene, max_images=max_images, image_size=image_size,
            load_gt_depth=load_gt_depth,
        )
    if dataset == "7scenes":
        return load_sevenscenes_scene(
            data_root, scene, max_images=max_images, image_size=image_size,
            load_gt_depth=load_gt_depth, frame_stride=frame_stride,
        )
    raise ValueError(f"Unknown dataset {dataset!r}")


def load_bench_cams(
    dataset: str,
    scene: str,
    data_root: Path,
    image_size: int,
    image_names: Optional[list[str]] = None,
) -> ETH3DCams:
    if dataset == "eth3d":
        scene_dir = download_eth3d_terrains(data_root, scene=scene, download_depth=False)
        return load_eth3d_cams(scene_dir, image_size=image_size, image_names=image_names)
    if dataset == "hiroom":
        return load_hiroom_cams(data_root, scene, image_size=image_size, image_names=image_names)
    if dataset == "7scenes":
        return load_sevenscenes_cams(data_root, scene, image_size=image_size, image_names=image_names)
    raise ValueError(f"Unknown dataset {dataset!r}")
