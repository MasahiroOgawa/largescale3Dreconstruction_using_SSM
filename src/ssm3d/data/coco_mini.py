"""Tiny COCO val subset for the instance-segmentation demo.

Downloads:
  * COCO val2017 annotations (~240 MB zip, but we extract only instances_val2017.json)
  * A fixed list of ~20 val2017 image IDs from the COCO image CDN (~<20 MB).

We deliberately avoid downloading the full val2017 image zip (~6 GB).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from ._util import download, require_free

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


@dataclass
class COCOMiniSample:
    image: torch.Tensor  # (3, H, W) float32 [0,1]
    image_id: int
    masks: torch.Tensor  # (K, H, W) bool
    category_ids: list[int]
    file_name: str


def _coco_image_url(file_name: str) -> str:
    return f"http://images.cocodataset.org/val2017/{file_name}"


def download_coco_mini(data_root: Path, num_images: int = 20) -> tuple[Path, list[int]]:
    """Returns (annotation_json_path, list_of_image_ids) — downloads per-image PNGs lazily in load_coco_mini."""
    data_root = Path(data_root)
    require_free(data_root, 2 * 2**30)

    ann_root = data_root / "coco_mini"
    ann_zip = ann_root / "annotations_trainval2017.zip"
    ann_json = ann_root / "annotations" / "instances_val2017.json"

    if not ann_json.exists():
        ann_zip = download(ANN_URL, ann_zip, min_bytes=10_000_000)
        with zipfile.ZipFile(ann_zip) as z:
            for name in z.namelist():
                if name.endswith("instances_val2017.json"):
                    z.extract(name, ann_root)
                    break
        if ann_zip.exists():
            ann_zip.unlink()

    # Pick the first num_images that have at least one instance annotation.
    with open(ann_json) as f:
        data = json.load(f)
    anns_by_img: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    chosen: list[int] = []
    for img in data["images"]:
        if img["id"] in anns_by_img:
            chosen.append(img["id"])
            if len(chosen) >= num_images:
                break
    return ann_json, chosen


def _decode_mask(seg, height: int, width: int) -> np.ndarray:
    """Decode COCO polygon or RLE to a (H, W) bool mask."""
    from pycocotools import mask as coco_mask

    if isinstance(seg, list):
        # polygons
        rles = coco_mask.frPyObjects(seg, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            rle = coco_mask.frPyObjects(seg, height, width)
        else:
            rle = seg
    else:
        return np.zeros((height, width), dtype=bool)
    return coco_mask.decode(rle).astype(bool)


def load_coco_mini(
    data_root: Path,
    num_images: int = 20,
    image_size: int = 224,
    max_masks_per_image: int = 8,
) -> List[COCOMiniSample]:
    ann_json, ids = download_coco_mini(data_root, num_images=num_images)
    with open(ann_json) as f:
        data = json.load(f)
    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    img_root = Path(ann_json).parent.parent / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    samples: list[COCOMiniSample] = []
    for iid in ids:
        meta = images_by_id[iid]
        file_name = meta["file_name"]
        img_path = img_root / file_name
        if not img_path.exists():
            download(_coco_image_url(file_name), img_path, min_bytes=1024)

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size
            im_r = im.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(im_r, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(arr).permute(2, 0, 1)

        masks, cats = [], []
        for a in anns_by_img[iid][:max_masks_per_image]:
            m = _decode_mask(a["segmentation"], orig_h, orig_w)
            m_resized = np.array(Image.fromarray(m).resize((image_size, image_size), Image.NEAREST)).astype(bool)
            masks.append(m_resized)
            cats.append(a["category_id"])
        if not masks:
            masks = [np.zeros((image_size, image_size), dtype=bool)]
            cats = [-1]
        samples.append(
            COCOMiniSample(
                image=image_t,
                image_id=iid,
                masks=torch.from_numpy(np.stack(masks)),
                category_ids=cats,
                file_name=file_name,
            )
        )
    return samples
