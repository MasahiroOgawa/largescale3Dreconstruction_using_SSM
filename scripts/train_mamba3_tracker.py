"""Train the Mamba-3 3D point tracker on TAPVid-3D (v8 unified-config CLI).

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python scripts/train_mamba3_tracker.py \\
            --config configs/v8.yaml \\
            --out-dir outputs/runs/mamba3_tracker_v8

Smoke run (override knobs via CLI on top of the config):
    uv run python scripts/train_mamba3_tracker.py \\
        --config configs/v8.yaml \\
        --out-dir outputs/runs/mamba3_tracker_v8_smoke \\
        --steps 50 --window 4 --batch 1 --val-every 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mamba3_tracker.data.dataset import (
    TAPVid3DDataset, collate_tracking, default_train_val, minival_split,
)
from mamba3_tracker.data.tapvid3d import load_clip
from mamba3_tracker.model.tracker import Mamba3Tracker
from mamba3_tracker.train.config import dump_resolved, load_config
from mamba3_tracker.train.loss import TrackingLoss, TrackingLossOutput
from mamba3_tracker.train.schedule import wsd


_LOSS_KEYS = ("total", "pos_3D", "pos_2D", "vis")


def _save_ckpt(out_dir: Path, step: int, model, optim, sched, history, cfg) -> Path:
    """Save an atomic checkpoint that captures everything needed to resume:
    trainable params + optimiser + scheduler + step counter + loss history + cfg.

    Frozen `encoder.backbone.*` weights (v14+ DINOv2 backbone) are NOT saved —
    they reload from the HuggingFace Hub via `from_pretrained` at resume.
    Keeps ckpts at ~5 MB per step instead of ~90 MB.
    """
    path = out_dir / f"ckpt_{step}.pt"
    tmp = out_dir / f"ckpt_{step}.pt.tmp"
    model_sd = {k: v for k, v in model.state_dict().items()
                if not k.startswith("encoder.backbone.")}
    torch.save({
        "step": step,
        "model": model_sd,
        "optim": optim.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "history": history,
        "cfg": cfg,
    }, tmp)
    tmp.replace(path)
    return path


def _find_latest_ckpt(out_dir: Path) -> Path | None:
    cands = []
    for p in out_dir.glob("ckpt_*.pt"):
        try:
            cands.append((int(p.stem.split("_", 1)[1]), p))
        except ValueError:
            continue
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]


def _loss_to_dict(out: TrackingLossOutput) -> dict[str, float]:
    return {
        "total": float(out.total.item()),
        "pos_3D": float(out.pos_3D.item()),
        "pos_2D": float(out.pos_2D.item()),
        "vis": float(out.vis.item()),
    }


def _fmt_loss_row(d: dict[str, float]) -> str:
    return (f"loss={d['total']:.4f}  "
            f"p3D={d['pos_3D']:.4f} p2D={d['pos_2D']:.4f}  "
            f"vis={d['vis']:.4f}")


@torch.no_grad()
def _validate(model, val_ds, loss_fn, device, amp_dtype, n_clips: int = 5) -> dict:
    model.eval()
    totals: dict[str, list[float]] = defaultdict(list)
    for i in range(min(n_clips, len(val_ds))):
        batch = collate_tracking([val_ds[i]])
        queries = batch.queries_xyt.to(device)
        qmask = batch.query_mask.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            pred = model(batch.images.to(device), queries, qmask)
        out = loss_fn(
            pred,
            batch.tracks_XYZ.to(device), batch.visibility.to(device),
            qmask, queries[..., 2].long(),
            batch.K.to(device),
        )
        d = _loss_to_dict(out)
        for k in _LOSS_KEYS:
            totals[k].append(d[k])
    model.train()
    return {k: sum(v) / len(v) for k, v in totals.items()}


_SUBSETS_KNOWN = ("pstudio", "drivetrack", "adt")


def _which_subset(path: Path) -> str:
    for s in _SUBSETS_KNOWN:
        if s in path.parts:
            return s
    return "unknown"


@torch.no_grad()
def _motion_check(
    model,
    val_paths: list,
    device,
    amp_dtype,
    max_frames: int = 32,
    n_clips: int = 5,
) -> dict[str, float]:
    """In-training motion-ratio diagnostic. For each of the first `n_clips`
    val clips, run inference on the first `max_frames` frames, reconstruct
    the absolute trajectory with the v12 per-anchor cumsum, project to 2D
    pixel coords (in the clip's ORIGINAL image space — uses `clip.K`
    unscaled), and accumulate per-subset:

        ratio(subset) = Σ d̂(t, n)  /  Σ d*(t, n)

    where `d̂` is predicted pixel travel from each track's anchor frame to
    frame t, `d*` is GT pixel travel, summed over visible (t, n) pairs
    where the track is visible at both `a_n` and `t`.

    Returns {subset: ratio} for the subsets that appeared in val_paths.
    """
    model.eval()
    img_size = model.image_size
    pred_sums: dict[str, float] = defaultdict(float)
    gt_sums:   dict[str, float] = defaultdict(float)
    for path in val_paths[:n_clips]:
        sub = _which_subset(path)
        clip = load_clip(path)
        F_ = min(max_frames, clip.F)
        sx = img_size / float(clip.W); sy = img_size / float(clip.H)
        images = Fn.interpolate(
            clip.images[:F_], size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        ).unsqueeze(0).to(device)
        N_q = clip.queries_xyt.shape[0]
        queries = clip.queries_xyt.clone()
        queries[:, 0] *= sx; queries[:, 1] *= sy
        queries[:, 2] = queries[:, 2].clamp(max=F_ - 1)
        qmask = torch.ones(1, N_q, dtype=torch.bool, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            pred = model(images, queries.unsqueeze(0).to(device), qmask)
        delta = pred.xyz[0].float().cpu()                      # (F, N_q, 3)

        # Per-anchor cumsum reconstruction (matches loss.py / eval / render).
        a_n = clip.queries_xyt[:, 2].long().clamp(min=0, max=F_ - 1)
        init = clip.tracks_XYZ[a_n, torch.arange(N_q)]
        delta_zero = delta.clone(); delta_zero[0] = 0.0
        cs = delta_zero.cumsum(dim=0)                          # (F, N_q, 3)
        cs_at_anchor = cs[a_n, torch.arange(N_q)]              # (N_q, 3)
        p_hat = init.unsqueeze(0) + cs - cs_at_anchor.unsqueeze(0)  # (F, N_q, 3)
        p_gt  = clip.tracks_XYZ[:F_]                            # (F, N_q, 3)

        # Project both to 2D in ORIGINAL image coords (clip.K is unscaled).
        # Track-level mask: drop tracks where the *predicted* Z goes
        # unreasonably small at any frame (Z < 5 cm). This avoids one
        # bad track dominating the per-subset ratio via projection blow-up
        # (typical at random init: 0.1 m³ random output sometimes lands
        # near the camera plane).
        K = clip.K.numpy()
        Z_pred = p_hat.numpy()[..., 2]                          # (F, N_q)
        bad_track = (Z_pred < 0.05).any(axis=0)                 # (N_q,)
        def _proj(xyz_np: np.ndarray) -> np.ndarray:
            Z = np.clip(xyz_np[..., 2:3], 1e-6, None)
            return (xyz_np[..., :2] / Z) * np.array([K[0,0], K[1,1]]) + np.array([K[0,2], K[1,2]])
        uv_pred = _proj(p_hat.numpy())                          # (F, N_q, 2)
        uv_gt   = _proj(p_gt.numpy())

        # Pixel travel from each track's own anchor frame.
        a_idx = a_n.numpy()
        track_idx = np.arange(N_q)
        ref_pred = uv_pred[a_idx, track_idx]                    # (N_q, 2)
        ref_gt   = uv_gt[a_idx, track_idx]
        diff_pred = uv_pred - ref_pred[None, :, :]              # (F, N_q, 2)
        diff_gt   = uv_gt   - ref_gt[None, :, :]
        travel_pred = np.linalg.norm(diff_pred, axis=-1)        # (F, N_q)
        travel_gt   = np.linalg.norm(diff_gt,   axis=-1)

        # Mask: visible at frame t AND visible at anchor frame, finite,
        # and predicted Z stayed in a sane range across the clip.
        vis = clip.visibility[:F_].numpy()                      # (F, N_q)
        vis_anchor = vis[a_idx, track_idx]                      # (N_q,)
        finite = np.isfinite(travel_pred) & np.isfinite(travel_gt)
        mask = (vis & vis_anchor[None, :] & finite & ~bad_track[None, :])

        pred_sums[sub] += float((travel_pred * mask).sum())
        gt_sums[sub]   += float((travel_gt   * mask).sum())

    model.train()
    return {sub: pred_sums[sub] / max(gt_sums[sub], 1.0) for sub in pred_sums}


def _fmt_motion_row(m: dict[str, float]) -> str:
    if not m:
        return "(no clips)"
    return "  ".join(f"{sub}={ratio*100:5.1f}%" for sub, ratio in sorted(m.items()))


def _build_overrides(args: argparse.Namespace) -> dict:
    """Map CLI args to nested overrides for `load_config`."""
    train_o = {
        k: getattr(args, k) for k in (
            "steps", "warmup", "decay", "ckpt_every", "val_every", "log_every",
            "lr", "weight_decay", "grad_clip", "batch", "window",
            "amp", "num_workers", "seed",
        )
    }
    data_o = {"subsets": args.subsets, "image_size": args.image_size,
              "num_tracks": args.num_tracks}
    return {
        k: v for k, v in {
            "train": {k: v for k, v in train_o.items() if v is not None},
            "data":  {k: v for k, v in data_o.items() if v is not None},
        }.items() if v
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="Unified ablation YAML (e.g. configs/v8.yaml).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--init-ckpt", type=Path, default=None)
    # Optional CLI overrides over the YAML (None = use YAML value).
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--decay", type=int, default=None)
    ap.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=None)
    ap.add_argument("--val-every", dest="val_every", type=int, default=None)
    ap.add_argument("--log-every", dest="log_every", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    ap.add_argument("--grad-clip", dest="grad_clip", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--image-size", dest="image_size", type=int, default=None)
    ap.add_argument("--num-tracks", dest="num_tracks", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()

    cfg = load_config(args.config, overrides=_build_overrides(args))
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    print(f"[train] loaded config {args.config} (version={cfg['version']})")
    print(f"[train] raw loss weights:  {loss_cfg['weights_raw']}")
    print(f"[train] norm loss weights: {loss_cfg['weights']}")

    torch.manual_seed(int(train_cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[train_cfg["amp"]]
    use_amp = train_cfg["amp"] != "fp32"

    split_cfg = data_cfg.get("split", {"source": "legacy"})
    source = split_cfg.get("source", "legacy")
    if source == "minival":
        train_clips, val_clips, test_clips = minival_split(
            args.data_root,
            subsets=data_cfg["subsets"],
            n_train=int(split_cfg.get("n_train", 40)),
            n_val=int(split_cfg.get("n_val", 5)),
            n_test=int(split_cfg.get("n_test", 5)),
            seed=int(split_cfg.get("seed", 42)),
        )
        print(f"[train] split=minival   "
              f"{len(train_clips)} train / {len(val_clips)} val / {len(test_clips)} test  "
              f"(subsets={data_cfg['subsets']})")
    elif source == "legacy":
        train_clips, val_clips = default_train_val(
            args.data_root, subsets=data_cfg["subsets"], val_frac=0.1, seed=42,
        )
        print(f"[train] split=legacy(random 90/10)   "
              f"{len(train_clips)} train / {len(val_clips)} val  "
              f"(subsets={data_cfg['subsets']})")
    else:
        raise ValueError(f"data.split.source = {source!r} not in {{minival, legacy}}")
    print(f"[train] data root: {args.data_root}")

    train_ds = TAPVid3DDataset(train_clips, window_size=int(train_cfg["window"]),
                               augment=True, seed=int(train_cfg["seed"]),
                               max_queries=int(data_cfg["num_tracks"]),
                               image_size=int(data_cfg["image_size"]))
    val_ds = TAPVid3DDataset(val_clips, window_size=int(train_cfg["window"]),
                             augment=False, seed=0,
                             max_queries=int(data_cfg["num_tracks"]),
                             image_size=int(data_cfg["image_size"]))
    # `persistent_workers=False` so the worker process is torn down each epoch
    # and PyTorch's caching allocator inside it is freed. See
    # memory/feedback_tapvid_dataloader_window_only.md.
    loader = DataLoader(
        train_ds, batch_size=int(train_cfg["batch"]), shuffle=True,
        num_workers=int(train_cfg["num_workers"]), collate_fn=collate_tracking,
        pin_memory=True, persistent_workers=False,
    )

    model = Mamba3Tracker(
        dim=int(model_cfg["dim"]), num_heads=int(model_cfg["num_heads"]),
        state_dim=int(model_cfg["state_dim"]),
        level_sizes=tuple(model_cfg.get("level_sizes", [32, 64])),
        num_iters=int(model_cfg.get("num_iters", 1)),
        use_correlation=bool(model_cfg.get("use_correlation", False)),
        encoder_kind=str(model_cfg.get("encoder_kind", "pyramid")),
        dinov2_model=str(model_cfg.get("dinov2_model", "facebook/dinov2-small")),
        dinov2_image_size=int(model_cfg.get("dinov2_image_size", 448)),
        dinov2_fuse_layers=model_cfg.get("dinov2_fuse_layers"),
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] tracker params: trainable={n_trainable:.2f}M  total={n_total:.2f}M  "
          f"encoder_kind={model_cfg.get('encoder_kind', 'pyramid')}, "
          f"num_iters={model_cfg.get('num_iters', 1)}, "
          f"use_correlation={model_cfg.get('use_correlation', False)}")

    if args.init_ckpt is not None:
        state = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        print(f"[train] loaded init weights from {args.init_ckpt}")

    loss_fn = TrackingLoss(
        weights=loss_cfg["weights"],
        image_size=int(data_cfg["image_size"]),
    ).to(device)
    optim = AdamW(model.parameters(),
                  lr=float(train_cfg["lr"]),
                  weight_decay=float(train_cfg["weight_decay"]))
    n_steps = int(train_cfg["steps"])
    warmup = int(train_cfg["warmup"])
    decay = int(train_cfg["decay"])
    sched = LambdaLR(optim, lr_lambda=lambda s: wsd(s, warmup, decay, n_steps))

    history: list[dict] = []
    motion_history: list[dict] = []
    # In-training motion check (v13+): runs on the first N val clip paths in
    # full-clip (max_frames-window) inference. Cheap; logs [motion] line per val.
    motion_cfg = train_cfg.get("motion_check", {})
    motion_max_frames = int(motion_cfg.get("max_frames", 32))
    motion_n_clips    = int(motion_cfg.get("n_clips", 5))
    val_clips_for_motion = val_clips
    # cfg.json: resolved config + the CLI we used to launch (for repro).
    cfg_snapshot = {**cfg, "_launch": {
        "config_path": str(args.config),
        "out_dir": str(args.out_dir),
        "data_root": str(args.data_root),
        "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
    }}
    dump_resolved(cfg_snapshot, args.out_dir / "cfg.json")

    # Auto-resume from the latest ckpt_*.pt in --out-dir if one exists.
    # Restores model + optimiser + scheduler state + step counter + history,
    # so a killed run picks up at most `ckpt_every` steps behind where it died.
    start_step = 0
    latest = _find_latest_ckpt(args.out_dir)
    if latest is not None:
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optim.load_state_dict(state["optim"])
        if state.get("sched") is not None:
            sched.load_state_dict(state["sched"])
        start_step = int(state["step"])
        history = list(state.get("history", []))
        # Restore motion_history from disk so the run's [motion] log is contiguous.
        mh_path = args.out_dir / "motion_history.json"
        if mh_path.exists():
            try:
                motion_history = json.loads(mh_path.read_text())
            except json.JSONDecodeError:
                pass
        print(f"[train] RESUMED from {latest} at step {start_step}", flush=True)

    grad_clip = float(train_cfg["grad_clip"])
    log_every = int(train_cfg["log_every"])
    val_every = int(train_cfg["val_every"])
    ckpt_every = int(train_cfg["ckpt_every"])

    model.train()
    t0 = time.perf_counter()
    step = start_step
    loader_iter = iter(loader)
    while step < n_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        queries = batch.queries_xyt.to(device, non_blocking=True)
        qmask = batch.query_mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(
                batch.images.to(device, non_blocking=True),
                queries, qmask,
            )
        loss_out = loss_fn(
            pred,
            batch.tracks_XYZ.to(device, non_blocking=True),
            batch.visibility.to(device, non_blocking=True),
            qmask,
            queries[..., 2].long(),
            batch.K.to(device, non_blocking=True),
        )
        loss_out.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # NaN-guard: skip optimiser step on non-finite grads (single NaN
        # corrupts every parameter via NaN + x = NaN otherwise — that's
        # how v2 silently ran 30k steps on a dead model).
        if not torch.isfinite(grad_norm):
            print(f"[train] step {step:6d}: non-finite grad_norm={grad_norm.item()} — "
                  f"skipping optimizer step", flush=True)
        elif not torch.isfinite(loss_out.total):
            print(f"[train] step {step:6d}: non-finite loss={loss_out.total.item()} — "
                  f"skipping optimizer step", flush=True)
        else:
            optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        if step % log_every == 0:
            lr = sched.get_last_lr()[0]
            dt = time.perf_counter() - t0
            row = _loss_to_dict(loss_out)
            print(f"[train] step {step:6d}/{n_steps}  "
                  f"{_fmt_loss_row(row)}  lr={lr:.2e}  elapsed={dt:.0f}s",
                  flush=True)
            history.append({"step": step, "lr": lr, **row})

        if step > 0 and step % val_every == 0:
            v = _validate(model, val_ds, loss_fn, device, amp_dtype)
            print(f"[train] step {step:6d}  VAL     {_fmt_loss_row(v)}", flush=True)
            history.append({"step": step, "val": v})
            # In-training motion check (v13+): 2D pixel travel ratio per subset.
            # Cheap (~5 val clips × ~32 frames), runs on val_paths the user
            # passed in. Logs as [motion] line and persists to motion_history.json.
            m = _motion_check(model, val_clips_for_motion, device, amp_dtype,
                              max_frames=motion_max_frames,
                              n_clips=motion_n_clips)
            print(f"[train] step {step:6d}  MOTION  {_fmt_motion_row(m)}", flush=True)
            motion_history.append({"step": step, **{f"{k}_ratio": v for k, v in m.items()}})
            (args.out_dir / "motion_history.json").write_text(
                json.dumps(motion_history, indent=2)
            )

        if step > 0 and step % ckpt_every == 0:
            p = _save_ckpt(args.out_dir, step, model, optim, sched, history, cfg_snapshot)
            print(f"[train] saved {p}", flush=True)

        (args.out_dir / "loss_history.json").write_text(json.dumps(history, indent=2))
        step += 1

    final = _save_ckpt(args.out_dir, n_steps, model, optim, sched, history, cfg_snapshot)
    print(f"[train] DONE — {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
