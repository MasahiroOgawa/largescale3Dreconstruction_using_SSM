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

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mamba3_tracker.data.dataset import (
    TAPVid3DDataset, collate_tracking, default_train_val,
)
from mamba3_tracker.model.tracker import Mamba3Tracker
from mamba3_tracker.train.config import dump_resolved, load_config
from mamba3_tracker.train.loss import TrackingLoss, TrackingLossOutput
from mamba3_tracker.train.schedule import wsd


_LOSS_KEYS = ("total", "vel_3D", "vel_2D", "pos_3D", "pos_2D",
              "smooth_3D", "smooth_2D", "vis")


def _save_ckpt(out_dir: Path, step: int, model, optim, cfg: dict) -> Path:
    path = out_dir / f"ckpt_{step}.pt"
    torch.save({"step": step, "model": model.state_dict(),
                "optim": optim.state_dict(), "cfg": cfg}, path)
    return path


def _loss_to_dict(out: TrackingLossOutput) -> dict[str, float]:
    return {
        "total": float(out.total.item()),
        "vel_3D": float(out.vel_3D.item()),
        "vel_2D": float(out.vel_2D.item()),
        "pos_3D": float(out.pos_3D.item()),
        "pos_2D": float(out.pos_2D.item()),
        "smooth_3D": float(out.smooth_3D.item()),
        "smooth_2D": float(out.smooth_2D.item()),
        "vis": float(out.vis.item()),
    }


def _fmt_loss_row(d: dict[str, float]) -> str:
    return (f"loss={d['total']:.4f}  "
            f"v3D={d['vel_3D']:.4f} v2D={d['vel_2D']:.4f}  "
            f"p3D={d['pos_3D']:.4f} p2D={d['pos_2D']:.4f}  "
            f"s3D={d['smooth_3D']:.4f} s2D={d['smooth_2D']:.4f}  "
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

    train_clips, val_clips = default_train_val(
        args.data_root, subsets=data_cfg["subsets"], val_frac=0.1, seed=42,
    )
    print(f"[train] data root: {args.data_root}")
    print(f"[train] {len(train_clips)} train clips / {len(val_clips)} val clips "
          f"(subsets={data_cfg['subsets']})")

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
        level_sizes=tuple(model_cfg["level_sizes"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] tracker params: {n_params:.2f}M, level_sizes={model_cfg['level_sizes']}")

    if args.init_ckpt is not None:
        state = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        print(f"[train] loaded init weights from {args.init_ckpt}")

    loss_fn = TrackingLoss(
        weights=loss_cfg["weights"],
        delta_3d_m=float(loss_cfg["delta_3d_m"]),
        delta_2d_px=float(loss_cfg["delta_2d_px"]),
    ).to(device)
    optim = AdamW(model.parameters(),
                  lr=float(train_cfg["lr"]),
                  weight_decay=float(train_cfg["weight_decay"]))
    n_steps = int(train_cfg["steps"])
    warmup = int(train_cfg["warmup"])
    decay = int(train_cfg["decay"])
    sched = LambdaLR(optim, lr_lambda=lambda s: wsd(s, warmup, decay, n_steps))

    history: list[dict] = []
    # cfg.json: resolved config + the CLI we used to launch (for repro).
    cfg_snapshot = {**cfg, "_launch": {
        "config_path": str(args.config),
        "out_dir": str(args.out_dir),
        "data_root": str(args.data_root),
        "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
    }}
    dump_resolved(cfg_snapshot, args.out_dir / "cfg.json")

    grad_clip = float(train_cfg["grad_clip"])
    log_every = int(train_cfg["log_every"])
    val_every = int(train_cfg["val_every"])
    ckpt_every = int(train_cfg["ckpt_every"])

    model.train()
    t0 = time.perf_counter()
    step = 0
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
            print(f"[train] step {step:6d}  VAL  {_fmt_loss_row(v)}", flush=True)
            history.append({"step": step, "val": v})

        if step > 0 and step % ckpt_every == 0:
            p = _save_ckpt(args.out_dir, step, model, optim, cfg_snapshot)
            print(f"[train] saved {p}", flush=True)

        (args.out_dir / "loss_history.json").write_text(json.dumps(history, indent=2))
        step += 1

    final = _save_ckpt(args.out_dir, n_steps, model, optim, cfg_snapshot)
    print(f"[train] DONE — {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
