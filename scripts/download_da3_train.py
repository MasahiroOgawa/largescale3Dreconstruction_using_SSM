"""Audit / download the DA3-style training corpus.

Today this is intentionally lean — the primary corpus is the 10 ETH3D train
scenes (`courtyard, facade, office, delivery_area, kicker, pipes, electro,
playground, relief, relief_2`), all of which the local download flow already
fetches with GT depth via `mamba3_attn.data.eth3d_multi.download_eth3d_scenes`.

The DA3 paper Table 1 lists 22 datasets; only a handful are downloadable
from HuggingFace without auth and parse cleanly into a (RGB, depth, K, w2c)
pipeline. Each extra dataset that gets added below should:

1. Have its own loader under `src/mamba3_attn/data/<name>.py` exposing
   `download_*` + `load_*_scene` that returns the same `ETH3DSample` shape.
2. Be registered in `src/mamba3_attn/data/bench.py` (`DATASETS`,
   `default_scene`, `load_bench_scene`, `load_bench_cams`).
3. Have entries in `configs/da3_train_core.yaml` so the training launcher
   picks them up via `train_super.SuperPhaseConfig.scenes`.

Usage:
    uv run python scripts/download_da3_train.py            # audit + ETH3D
    uv run python scripts/download_da3_train.py --skip-eth3d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mamba3_attn.data._util import free_bytes


# Stubs for datasets we want to add but haven't wired loaders for yet. Each
# entry is a one-line note describing the source + approximate size; the
# function it points to either downloads or prints a NotImplementedError.
DEFERRED_DATASETS: dict[str, str] = {
    "blendedmvs": (
        "HarrisonPENG/blendedmvs (~82 GB, WebDataset). "
        "TODO: tar-shard → per-scene image bank loader under src/mamba3_attn/data/blendedmvs.py."
    ),
    "megadepth": (
        "Phototourism MegaDepth-v1 (~140 GB, tarballs). "
        "TODO: tarball loader + Colmap-pose parser under src/mamba3_attn/data/megadepth.py."
    ),
    "dl3dv": (
        "DL3DV-10K (~600 GB full, ~80 GB sample, HF gated). "
        "TODO: snapshot_download + transforms.json reader under src/mamba3_attn/data/dl3dv.py."
    ),
    "hypersim": (
        "apple/hypersim (~1.5 TB full, ~300 GB for 50-scene subset). "
        "TODO: per-scene tarball fetch + tonemap loader under src/mamba3_attn/data/hypersim.py."
    ),
}


def _audit_eth3d(data_root: Path) -> tuple[list[str], list[str]]:
    """Return (have_gt, missing_gt) train-scene names."""
    from mamba3_attn.data.eth3d_multi import TRAIN_SCENES, _find_depth_root

    have_gt: list[str] = []
    missing_gt: list[str] = []
    for s in TRAIN_SCENES:
        scene_dir = data_root / "eth3d" / s
        if not scene_dir.exists():
            missing_gt.append(s)
            continue
        if _find_depth_root(scene_dir) is None:
            missing_gt.append(s)
        else:
            have_gt.append(s)
    return have_gt, missing_gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--skip-eth3d", action="store_true",
                    help="Skip ETH3D download (audit only).")
    args = ap.parse_args()

    root = args.data_root.resolve()
    print(f"[da3_train] data root: {root}")
    print(f"[da3_train] free space: {free_bytes(root) / 2**30:.1f} GiB")

    print("\n[da3_train] Phase 1 — ETH3D train scenes")
    have, missing = _audit_eth3d(root)
    print(f"  on-disk with GT depth ({len(have)}): {have}")
    print(f"  missing ({len(missing)}):           {missing}")

    if missing and not args.skip_eth3d:
        from mamba3_attn.data.eth3d_multi import download_eth3d_scenes
        print(f"\n[da3_train] downloading {len(missing)} missing ETH3D scenes with GT depth ...")
        download_eth3d_scenes(root, scenes=missing, download_depth=True)

    print("\n[da3_train] Phase 2 — deferred datasets (need loader implementation):")
    for name, note in DEFERRED_DATASETS.items():
        print(f"  - {name}: {note}")

    print("\n[da3_train] done.")
    print(f"[da3_train] primary training corpus: ETH3D 10 train scenes, "
          f"images + GT depth + intrinsics + extrinsics.")
    print(f"[da3_train] see configs/da3_train_core.yaml for the scene list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
