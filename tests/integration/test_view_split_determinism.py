"""Determinism + invariants for the view-level train/test split.

PLAN §15.59: per-scene overfit relies on a deterministic view split so the
held-out eval can be reproduced from the same `--split-seed` and `--train-frac`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamba3_attn.data.view_split import read_split, split_views, write_split


def test_split_is_deterministic_for_same_seed():
    a = split_views(42, train_frac=0.75, seed=42)
    b = split_views(42, train_frac=0.75, seed=42)
    assert a == b


def test_train_test_are_disjoint_and_cover_all_views():
    train, test = split_views(42, train_frac=0.75, seed=42)
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(42))


def test_train_size_matches_ceil_train_frac():
    train, test = split_views(42, train_frac=0.75, seed=42)
    # ceil(42 * 0.75) = 32, so 10 in test
    assert len(train) == 32
    assert len(test) == 10


def test_different_seeds_produce_different_splits():
    a = split_views(42, train_frac=0.75, seed=42)
    b = split_views(42, train_frac=0.75, seed=43)
    assert a != b


def test_lists_are_sorted():
    train, test = split_views(42, train_frac=0.75, seed=42)
    assert train == sorted(train)
    assert test == sorted(test)


def test_at_least_one_view_in_each_split():
    train, test = split_views(2, train_frac=0.75, seed=0)
    assert len(train) >= 1 and len(test) >= 1


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        split_views(1)
    with pytest.raises(ValueError):
        split_views(10, train_frac=0.0)
    with pytest.raises(ValueError):
        split_views(10, train_frac=1.0)


def test_write_then_read_split_roundtrips(tmp_path: Path):
    train, test = split_views(42, train_frac=0.75, seed=42)
    write_split(tmp_path, num_views=42, train_frac=0.75, seed=42,
                train=train, test=test)
    payload = json.loads((tmp_path / "split.json").read_text())
    assert payload["num_views"] == 42
    assert payload["train_frac"] == 0.75
    assert payload["seed"] == 42
    rt_train, rt_test = read_split(tmp_path / "split.json")
    assert rt_train == train
    assert rt_test == test
