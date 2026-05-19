"""PyTorch Dataset wrappers for TAPVid-3D training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .tapvid3d import SUBSETS, list_clips, load_clip, peek_clip_F


@dataclass
class TrackingBatch:
    """One training batch. Variable N_q is padded to N_q_max with `query_mask`."""
    images: torch.Tensor          # (B, F, 3, H, W)
    queries_xyt: torch.Tensor     # (B, N_q_max, 3)
    tracks_XYZ: torch.Tensor      # (B, F, N_q_max, 3)
    visibility: torch.Tensor      # (B, F, N_q_max) bool
    query_mask: torch.Tensor      # (B, N_q_max) bool — True where the query slot is real
    K: torch.Tensor               # (B, 3, 3)
    clip_ids: list[str]
    subsets: list[str]


class TAPVid3DDataset(Dataset):
    """Yields fixed-length temporal windows sampled from full clips.

    For training: window length F (e.g. 24), random start within the clip,
    photometric augmentation only (no geometric augmentation — would
    invalidate `tracks_XYZ`).

    For evaluation: set `window_size=None` and the full clip is returned.
    """

    def __init__(
        self,
        clip_paths: Sequence[Path],
        window_size: int | None = 24,
        seed: int = 0,
        max_queries: int = 512,
        augment: bool = False,
        image_size: int = 448,
    ) -> None:
        self.clip_paths = list(clip_paths)
        self.window_size = window_size
        self.max_queries = max_queries
        self.augment = augment
        self.image_size = image_size
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clip_paths)

    def __getitem__(self, idx: int) -> dict:
        # Pick the window first, then decode only those JPEG frames. Decoding
        # the whole clip up-front and slicing afterwards burns ~566 MB per
        # drivetrack __getitem__ call (1280×1920×3 float32 × 24 frames) and
        # leaks into PyTorch's caching allocator when persistent_workers=True,
        # which is what killed v7 via systemd-oomd at ~step 200.
        path = self.clip_paths[idx]
        F_total = peek_clip_F(path)
        if self.window_size is not None and self.window_size < F_total:
            start = self._rng.randint(0, F_total - self.window_size)
            end = start + self.window_size
        else:
            start, end = 0, F_total

        clip = load_clip(path, frames=(start, end))
        images = clip.images
        tracks = clip.tracks_XYZ
        vis = clip.visibility
        H_orig, W_orig = int(clip.H), int(clip.W)

        # Resize every clip to a common (image_size, image_size) so collate
        # can stack across subsets with different native resolutions
        # (drivetrack 1280×1920, adt 512×512, pstudio 360×640). 3D GT tracks
        # are in camera-frame metres and are unaffected by image resize.
        # Query (x, y) coords ARE in pixel space, so they must be scaled to
        # match the resized image (the propagator's bilinear-sample step in
        # §8.3 of doc/attention/mamba3_attention.tex uses these to seed
        # each track's initial bank slot).
        sx = self.image_size / float(W_orig)
        sy = self.image_size / float(H_orig)
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = torch.nn.functional.interpolate(
                images, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )

        if self.augment:
            images = _photometric_aug(images, self._rng)

        # Subselect queries if there are more than `max_queries`.
        N_q = clip.N_q
        if N_q > self.max_queries:
            picked = sorted(self._rng.sample(range(N_q), self.max_queries))
            queries = clip.queries_xyt[picked]
            tracks = tracks[:, picked]
            vis = vis[:, picked]
            N_q = self.max_queries
        else:
            queries = clip.queries_xyt

        # Shift query frame indices to the window's reference; only keep
        # queries whose anchor frame falls inside [start, end). If the cropped
        # window contains zero anchors, re-cast every query's anchor to
        # frame 0 of the window so we still have GT supervision.
        keep = (queries[:, 2].long() >= start) & (queries[:, 2].long() < end)
        if keep.sum().item() == 0:
            queries = queries.clone()
            queries[:, 2] = float(start)
        else:
            queries = queries[keep].clone()
            tracks = tracks[:, keep]
            vis = vis[:, keep]
        queries[:, 2] -= start
        # Scale (x, y) into the resized image's pixel coords.
        queries[:, 0] *= sx
        queries[:, 1] *= sy

        # Scale K to the resized-image pixel coords too, so the 2D
        # reprojection loss can compare predicted/GT pixel coords directly
        # (v6 addition; see src/mamba3_tracker/train/loss.py).
        K = clip.K.clone()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        return {
            "images": images,
            "queries_xyt": queries,
            "tracks_XYZ": tracks,
            "visibility": vis,
            "K": K,
            "clip_id": clip.clip_id,
            "subset": clip.subset,
        }


def _photometric_aug(images: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Same per-frame color jitter applied across the whole window."""
    brightness = rng.uniform(0.7, 1.3)
    contrast = rng.uniform(0.7, 1.3)
    out = images * brightness
    mean = out.mean(dim=(-1, -2, -3), keepdim=True)
    out = (out - mean) * contrast + mean
    return out.clamp_(0.0, 1.0)


def collate_tracking(items: list[dict]) -> TrackingBatch:
    """Pad variable-N_q clips into a single batch."""
    B = len(items)
    Nmax = max(it["queries_xyt"].shape[0] for it in items)
    F, _, H, W = items[0]["images"].shape

    images = torch.stack([it["images"] for it in items], dim=0)
    K = torch.stack([it["K"] for it in items], dim=0)

    queries = torch.zeros(B, Nmax, 3, dtype=torch.float32)
    tracks = torch.zeros(B, F, Nmax, 3, dtype=torch.float32)
    vis = torch.zeros(B, F, Nmax, dtype=torch.bool)
    qmask = torch.zeros(B, Nmax, dtype=torch.bool)
    for b, it in enumerate(items):
        n = it["queries_xyt"].shape[0]
        queries[b, :n] = it["queries_xyt"]
        tracks[b, :, :n] = it["tracks_XYZ"]
        vis[b, :, :n] = it["visibility"]
        qmask[b, :n] = True

    return TrackingBatch(
        images=images,
        queries_xyt=queries,
        tracks_XYZ=tracks,
        visibility=vis,
        query_mask=qmask,
        K=K,
        clip_ids=[it["clip_id"] for it in items],
        subsets=[it["subset"] for it in items],
    )


def split_clips(
    clip_paths: Sequence[Path],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Random 90/10 train/val split, stratified by subset."""
    rng = random.Random(seed)
    by_subset: dict[str, list[Path]] = {s: [] for s in SUBSETS}
    for p in clip_paths:
        for s in SUBSETS:
            if s in p.parts:
                by_subset[s].append(p)
                break

    train: list[Path] = []
    val: list[Path] = []
    for sub_clips in by_subset.values():
        shuffled = sub_clips[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_frac)))
        val += shuffled[:n_val]
        train += shuffled[n_val:]
    return train, val


def default_train_val(
    data_root: str | Path = "~/data",
    subsets: Iterable[str] = SUBSETS,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    return split_clips(list_clips(data_root, subsets), val_frac=val_frac, seed=seed)
