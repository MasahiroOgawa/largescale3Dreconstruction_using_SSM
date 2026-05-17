"""Download DA3-BENCH (eval suite) into data/da3_bench/.

DA3-BENCH lives on Hugging Face at `depth-anything/DA3-BENCH` and ships 6 zip
archives totalling ~38 GB:

    eth3d.zip       ~14.1 GB   high-res MVS (indoor/outdoor)
    scannetpp.zip   ~10.1 GB   high-quality indoor RGB-D
    dtu.zip          ~8.3 GB   MVS (22 scenes × 49 views, recon)
    7scenes.zip      ~3.3 GB   indoor RGB-D
    dtu64.zip        ~1.7 GB   DTU subset for pose eval (13 scenes × 64 views)
    hiroom.zip       ~0.7 GB   high-res indoor

After download we unzip into `data/da3_bench/<name>/` and create a
`workspace/benchmark_dataset` symlink at the project root pointing at
`data/da3_bench`, so DA3's evaluator (which hardcodes paths like
`workspace/benchmark_dataset/eth3d` in `utils/constants.py`) finds them when
run from this directory.

Usage:
    uv run python scripts/download_da3_bench.py
    uv run python scripts/download_da3_bench.py --only hiroom dtu64   # subset
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from mamba3_attn.data._util import free_bytes, require_free


REPO_ID = "depth-anything/DA3-BENCH"
# Approximate uncompressed sizes (zip + extracted) in GiB, used for the
# free-space precheck. ~3× zip size is a safe upper bound for extraction.
DATASETS: dict[str, float] = {
    "eth3d": 14.1,
    "scannetpp": 10.1,
    "dtu": 8.3,
    "7scenes": 3.3,
    "dtu64": 1.7,
    "hiroom": 0.7,
}


def _ensure_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        print(f"[bench] removing stale link/dir at {link}")
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            raise RuntimeError(f"refusing to remove non-symlink at {link}; resolve manually")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    print(f"[bench] symlinked {link} → {target}")


def _extract(zip_path: Path, out_dir: Path) -> Path:
    marker = out_dir / "_extracted_ok"
    if marker.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] extracting {zip_path.name} → {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.write_text("ok\n")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--only", nargs="*", default=None, choices=list(DATASETS),
                    help="Subset of benchmark datasets to fetch. Default: all 6.")
    ap.add_argument("--keep-zip", action="store_true",
                    help="Keep the .zip files after successful extraction.")
    args = ap.parse_args()

    root = args.data_root.resolve()
    bench_dir = root / "da3_bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    wanted = args.only or list(DATASETS)
    total_gib = sum(DATASETS[k] for k in wanted) * 2.5
    need_bytes = int(total_gib * (1024 ** 3))
    print(f"[bench] target: {wanted}\n[bench] free-space budget (zip + extract): ~{total_gib:.1f} GiB")
    require_free(root, need_bytes)
    print(f"[bench] free under {root}: {free_bytes(root) / 2**30:.1f} GiB — OK")

    allow_patterns = [f"{name}.zip" for name in wanted]
    print(f"[bench] snapshot_download from HF: {REPO_ID} ({allow_patterns})")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(bench_dir / "_hf"),
        allow_patterns=allow_patterns,
    )
    local = Path(local)
    print(f"[bench] HF cache at {local}")

    for name in wanted:
        zip_path = local / f"{name}.zip"
        if not zip_path.exists():
            print(f"[bench] [skip] {zip_path} not found in snapshot")
            continue
        _extract(zip_path, bench_dir / name)
        if not args.keep_zip:
            zip_path.unlink()
            print(f"[bench] removed {zip_path.name} (use --keep-zip to retain)")

    # DA3 evaluator hardcodes `workspace/benchmark_dataset/...` as relative
    # paths in src/depth_anything_3/utils/constants.py. Linking it here means
    # `python -m depth_anything_3.bench.evaluator` works when invoked from the
    # project root.
    proj_root = Path.cwd()
    _ensure_symlink(bench_dir, proj_root / "workspace" / "benchmark_dataset")

    print("\n[bench] done.")
    print(f"[bench] eval data:  {bench_dir}")
    print(f"[bench] DA3 expects: {proj_root / 'workspace' / 'benchmark_dataset'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
