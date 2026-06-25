"""Run DepthAnything-3 on TAPVid-3D clips and save depth.npz per clip.

Output layout (mirrors what TAPIP3D's make_da3_annotations.py expects):
    ~/data/tapvid3d_da3_out/<subset>/<clip_stem>/depth.npz
      depth      (T, Hd, Wd)  float32 metric metres
      intrinsics (T, 3, 3)    float32 at depth resolution
      extrinsics (T, 4, 4)    float32 (DA3-estimated world→camera poses)

Usage:
    uv run python scripts/run_da3_tapvid3d.py --split minival --subsets drivetrack pstudio adt
    uv run python scripts/run_da3_tapvid3d.py --split minival --subsets drivetrack
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

# DA3 needs the moviepy stub before any mamba3_tracker import
sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
        / "third_party"
        / "depth-anything-3"
        / "src"
    ),
)

from depth_anything_3.api import DepthAnything3  # noqa: E402
from PIL import Image  # noqa: E402

TAPVID_ROOT = Path("~/data/tapvid3d").expanduser()
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "outputs" / "tapvid3d_da3"


def list_seqs(subset: str, split: str) -> list[str]:
    if split == "sample":
        return sorted(p.name for p in (TAPVID_ROOT / subset).glob("*.npz"))
    from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES

    files = MINIVAL_FILES[subset] if split == "minival" else FULL_EVAL_FILES[subset]
    return sorted(files)


def decode_jpeg(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_da3_clip(
    da3: DepthAnything3, npz_path: Path, process_res: int, chunk_size: int = 0
) -> dict:
    """Return dict(depth, intrinsics, extrinsics) at depth resolution.

    chunk_size > 0: process in windows of that many frames, collecting depth only
    and using identity extrinsics (avoids OOM on long clips at the cost of global pose).
    chunk_size == 0: process entire clip at once (needs full VRAM).
    """
    # allow_pickle: TAPVid-3D npz files (official Google DeepMind dataset) store
    # images_jpeg_bytes as a Python object array — no user-supplied files here.
    npz = np.load(npz_path, allow_pickle=True)
    frames = [Image.fromarray(decode_jpeg(b)) for b in npz["images_jpeg_bytes"]]
    npz.close()

    T = len(frames)
    if chunk_size <= 0 or T <= chunk_size:
        pred = da3.inference(frames, process_res=process_res, export_format="mini_npz")
        depth = np.asarray(pred.depth, dtype=np.float32)
        Hd, Wd = depth.shape[1], depth.shape[2]
        if pred.intrinsics is not None:
            ki = np.asarray(pred.intrinsics, dtype=np.float32)
            if ki.ndim == 2:
                ki = np.broadcast_to(ki[None], (T, 3, 3)).copy()
        else:
            f = float(max(Hd, Wd))
            K = np.array([[f, 0, Wd / 2], [0, f, Hd / 2], [0, 0, 1]], np.float32)
            ki = np.broadcast_to(K[None], (T, 3, 3)).copy()
        if pred.extrinsics is not None:
            ei = np.asarray(pred.extrinsics, dtype=np.float32)
            if ei.ndim == 2:
                ei = np.broadcast_to(ei[None], (T, 4, 4)).copy()
        else:
            ei = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        return {"depth": depth, "intrinsics": ki, "extrinsics": ei}

    # Chunked: run DA3 independently on each window; concatenate depth + intrinsics.
    # Use identity extrinsics — each chunk's poses are relative to its own first frame
    # so a global pose chain can't be formed; identity is consistent with the GT
    # which also uses dummy identity extrinsics in TAPVid3dProvider.
    depths, intrinsics = [], []
    for start in range(0, T, chunk_size):
        chunk = frames[start : start + chunk_size]
        n = len(chunk)
        pred = da3.inference(chunk, process_res=process_res, export_format="mini_npz")
        depths.append(np.asarray(pred.depth, dtype=np.float32))  # (n, Hd, Wd)
        if pred.intrinsics is not None:
            ki = np.asarray(pred.intrinsics, dtype=np.float32)
            if ki.ndim == 2:  # (3,3) → broadcast to (n,3,3)
                ki = np.broadcast_to(ki[None], (n, 3, 3)).copy()
            intrinsics.append(ki)
        else:
            # Estimate from depth spatial dims: f = max(Hd, Wd), centre principal pt
            Hd, Wd = depths[-1].shape[1], depths[-1].shape[2]
            f = float(max(Hd, Wd))
            K = np.array([[f, 0, Wd / 2], [0, f, Hd / 2], [0, 0, 1]], np.float32)
            intrinsics.append(np.broadcast_to(K[None], (n, 3, 3)).copy())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    depth_all = np.concatenate(depths, axis=0)  # (T, Hd, Wd)
    intr_all = np.concatenate(intrinsics, axis=0)  # (T, 3, 3)
    extr_all = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))  # (T, 4, 4) identity
    return {"depth": depth_all, "intrinsics": intr_all, "extrinsics": extr_all}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split", default="minival", choices=["minival", "full_eval", "sample"]
    )
    ap.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    ap.add_argument("--da3-model", default="da3metric-large")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="frames per DA3 call (0=all-at-once); use >0 to avoid OOM on long clips",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[da3-tapvid3d] device={device}  model={args.da3_model}  split={args.split}")

    da3 = DepthAnything3.from_pretrained(f"depth-anything/{args.da3_model}").to(device)
    da3.eval()

    for subset in args.subsets:
        seqs = list_seqs(subset, args.split)
        print(f"[da3-tapvid3d] {subset}: {len(seqs)} clips")
        for i, fname in enumerate(seqs):
            stem = fname[:-4]
            out_dir = OUT_ROOT / subset / stem
            out_path = out_dir / "depth.npz"
            if out_path.exists() and not args.overwrite:
                print(f"  [{i + 1}/{len(seqs)}] skip (exists): {stem}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = TAPVID_ROOT / subset / fname
            if not npz_path.exists():
                print(f"  [{i + 1}/{len(seqs)}] MISSING: {npz_path}")
                continue
            print(f"  [{i + 1}/{len(seqs)}] {stem} ...", end=" ", flush=True)
            with torch.no_grad():
                result = run_da3_clip(da3, npz_path, args.process_res, args.chunk_size)
            np.savez_compressed(out_path, **result)
            print(f"depth{result['depth'].shape}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("[da3-tapvid3d] done.")


if __name__ == "__main__":
    main()
