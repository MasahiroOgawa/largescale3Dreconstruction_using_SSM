"""Download the ADT (Aria Digital Twin) subset of TAPVid-3D.

The official Google release does not redistribute the ADT .npz files (Aria
license requires generating them locally via aria_dataset_downloader after
accepting the licence). The HF mirror at `ZhengGuangze/TAPVid-3D` re-hosts
them pre-generated as 10 tarball batches of ~17 GB each.

This script pulls one or more batches and extracts the .npz files directly
under `~/data/tapvid3d/adt/`. After extraction, file names match the
`tapvid3d_*.npz` pattern used by the official MINIVAL_FILES split.

Usage:
    uv run python scripts/download_tapvid3d_adt.py            # batch 0 only
    uv run python scripts/download_tapvid3d_adt.py --batches 0 1 2
    uv run python scripts/download_tapvid3d_adt.py --batches all   # all 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


HF_BASE = "https://huggingface.co/datasets/ZhengGuangze/TAPVid-3D/resolve/main"
NUM_BATCHES = 10  # adt_batch_0 .. adt_batch_9


def _streamed_download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"[adt] {dest.name} already present ({dest.stat().st_size / 1e9:.1f} GB), skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))


def _extract_to(archive: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".npz"):
                continue
            target = dest / Path(m.name).name
            if target.exists() and target.stat().st_size > 0:
                continue
            extracted = tf.extractfile(m)
            if extracted is None:
                continue
            with open(target, "wb") as out:
                shutil.copyfileobj(extracted, out)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path.home() / "data" / "tapvid3d")
    ap.add_argument("--batches", nargs="+", default=["0"],
                    help="Batch indices to download (or 'all').")
    ap.add_argument("--keep-tarballs", action="store_true",
                    help="Keep .tar.gz files after extraction.")
    args = ap.parse_args()

    if args.batches == ["all"]:
        batches = list(range(NUM_BATCHES))
    else:
        batches = [int(b) for b in args.batches]

    adt_dir = args.data_root / "adt"
    cache_dir = adt_dir / "_tarballs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[adt] target: {adt_dir}")
    print(f"[adt] batches: {batches}")

    total_npz = 0
    for n in batches:
        url = f"{HF_BASE}/adt_batch_{n}.tar.gz"
        tar_path = cache_dir / f"adt_batch_{n}.tar.gz"
        print(f"\n[adt] batch {n}: {url}")
        _streamed_download(url, tar_path)
        print(f"[adt] extracting {tar_path.name} ...")
        n_extracted = _extract_to(tar_path, adt_dir)
        total_npz += n_extracted
        print(f"[adt] +{n_extracted} .npz files")
        if not args.keep_tarballs:
            tar_path.unlink()
            print(f"[adt] removed {tar_path.name} (use --keep-tarballs to retain)")

    if cache_dir.exists() and not any(cache_dir.iterdir()):
        cache_dir.rmdir()

    print(f"\n[adt] done. {total_npz} new .npz files under {adt_dir}")
    print(f"[adt] total .npz now: {len(list(adt_dir.glob('*.npz')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
