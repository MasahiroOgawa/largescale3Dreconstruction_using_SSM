"""Convert TAPIP3D `save_preds` pickle dumps into our external-prediction npz layout.

TAPIP3D's train_eval.py (with +train.save_preds=true) writes one pickle per clip:
    <pred_dir>/<sample_id>.pkl  ->  {"preds": {"coords": (T,N,3), "visibs": (T,N)}, "metrics": ...}
where `sample_id` is the index into the provider's sorted split="sample" .npz list
(datasets/providers/tapvid3d_provider.py). `coords` are per-frame-camera metric 3D
(same convention as TAPVid-3D GT tracks_XYZ and the released SpatialTracker preds);
`visibs` are raw visibility logits.

We map sample_id -> clip stem by reproducing that sorted ordering from the GT npz
directory, then write `<out>/<subset>/<clip>.npz` with keys tracks_XYZ (F,N,3) and
visibility (F,N) — exactly what `eval_metric3d.py --method external` consumes.

Usage:
    uv run python scripts/convert_tapip3d_preds.py \
        --pred-dir <pred_dir_from_other_pc> --subset drivetrack \
        --out-dir ~/data/tapvid3d_baseline_preds/tapip3d
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True, help="dir of <sample_id>.pkl")
    ap.add_argument("--subset", type=str, default="drivetrack")
    ap.add_argument("--gt-root", type=Path, default=Path("~/data/tapvid3d"))
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="<out>/<subset>/<clip>.npz will be written")
    args = ap.parse_args()

    gt_dir = args.gt_root.expanduser() / args.subset
    # Reproduce TAPVid3dProvider(split="sample") ordering: sorted .npz names.
    seqs = sorted(p.name for p in gt_dir.glob("*.npz"))
    out_dir = args.out_dir.expanduser() / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(args.pred_dir.expanduser().glob("*.pkl"))
    print(f"[convert] {len(pkls)} pkls, {len(seqs)} GT seqs in {args.subset}")
    n = 0
    for pp in pkls:
        try:
            sid = int(pp.stem)
        except ValueError:
            print(f"[convert] skip non-integer pkl name: {pp.name}")
            continue
        if sid >= len(seqs):
            print(f"[convert] sample_id {sid} out of range ({len(seqs)} seqs) — skip")
            continue
        # Trusted source: these pkls are our own TAPIP3D save_preds output, not
        # third-party data. (numpy arrays + plain dict written by train_eval.py.)
        with open(pp, "rb") as f:
            d = pickle.load(f)
        preds = d["preds"]
        coords = np.asarray(preds["coords"], dtype=np.float32)        # (T,N,3)
        visibs = np.asarray(preds["visibs"], dtype=np.float32)        # (T,N) logits
        vis_prob = 1.0 / (1.0 + np.exp(-visibs))                       # sigmoid
        clip_stem = seqs[sid][:-4]
        np.savez(out_dir / f"{clip_stem}.npz",
                 tracks_XYZ=coords, visibility=vis_prob.astype(np.float32))
        n += 1
    print(f"[convert] wrote {n} npz -> {out_dir}")
    print(f"[convert] now score: uv run python scripts/eval_metric3d.py --method external "
          f"--pred-dir {args.out_dir.expanduser()} --subsets {args.subset} --split minival "
          f"--out-dir outputs/metric3d_tapip3d_{args.subset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
