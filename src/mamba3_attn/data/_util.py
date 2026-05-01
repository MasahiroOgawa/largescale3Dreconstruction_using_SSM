"""Small helpers shared by downloaders."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def require_free(path: Path, need_bytes: int) -> None:
    have = free_bytes(path)
    if have < need_bytes:
        raise RuntimeError(
            f"Not enough free space under {path}: have {have // 2**30} GiB, need ~{need_bytes // 2**30} GiB. "
            "Free up space or pick a smaller dataset option."
        )


def download(
    url: str,
    dest: Path,
    min_bytes: int = 1024,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Path:
    """Stream-download `url` into `dest`. Skips if dest already exists & is non-empty."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar, open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
                if progress_cb:
                    progress_cb(pbar.n, total)
    return dest
