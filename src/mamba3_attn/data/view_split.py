"""Deterministic view-level train/test split for per-scene overfit.

PLAN §15.59 (per-scene-overfit pivot): once we drop cross-scene generalization
and deliberately overfit to a single scene, the train/test split is *across
views* of that scene rather than across scenes. `split_views` produces a
deterministic, disjoint (train, test) index pair so eval can verify recovery
on views the optimizer never saw.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


def split_views(
    num_views: int, train_frac: float = 0.75, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Return (train, test) view indices for a scene with `num_views` images.

    Both lists are sorted. Their union covers `[0, num_views)` exactly once.
    Train size is `ceil(num_views * train_frac)`; test gets the rest.
    """
    if num_views <= 1:
        raise ValueError(f"need ≥ 2 views to split, got {num_views}")
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    rng = random.Random(seed)
    shuffled = list(range(num_views))
    rng.shuffle(shuffled)

    n_train = max(1, min(num_views - 1, math.ceil(num_views * train_frac)))
    train = sorted(shuffled[:n_train])
    test = sorted(shuffled[n_train:])
    return train, test


def write_split(
    out_dir: Path, num_views: int, train_frac: float, seed: int,
    train: list[int], test: list[int],
) -> Path:
    """Persist the split to `<out_dir>/split.json` for later eval reproducibility."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "split.json"
    payload = {
        "num_views": num_views,
        "train_frac": train_frac,
        "seed": seed,
        "train": list(train),
        "test": list(test),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_split(path: Path) -> tuple[list[int], list[int]]:
    """Read a split written by `write_split`. Returns (train, test)."""
    payload = json.loads(Path(path).read_text())
    return list(payload["train"]), list(payload["test"])
