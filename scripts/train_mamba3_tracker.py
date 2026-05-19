"""Train the Mamba-3 3D point tracker on TAPVid-3D.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python scripts/train_mamba3_tracker.py \\
            --out-dir outputs/runs/mamba3_tracker_v1 \\
            --steps 30000 --window 8 --batch 2 \\
            --lr 2e-4 --level-sizes 32 64

For a smoke run:
    uv run python scripts/train_mamba3_tracker.py \\
        --out-dir outputs/runs/mamba3_tracker_smoke \\
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
from mamba3_tracker.train.loss import TrackingLoss, TrackingLossWeights
from mamba3_tracker.train.schedule import wsd


def _save_ckpt(out_dir: Path, step: int, model, optim, cfg: dict) -> Path:
    path = out_dir / f"ckpt_{step}.pt"
    torch.save({"step": step, "model": model.state_dict(),
                "optim": optim.state_dict(), "cfg": cfg}, path)
    return path


_LOSS_KEYS = ("total", "pos", "mag", "dir", "reproj", "vis", "spawn", "smooth")


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
        for k in _LOSS_KEYS:
            v = getattr(out, k, None)
            if v is not None:
                totals[k].append(float(v.item()))
    model.train()
    return {k: sum(v) / len(v) for k, v in totals.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--subsets", nargs="+",
                    default=["pstudio", "drivetrack"],
                    help="Which TAPVid-3D subsets to use (e.g. add 'adt' once it's extracted).")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--decay", type=int, default=3000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=448,
                    help="All clips are resized to (image_size, image_size) "
                    "before collation. 448 = 32 patches × 14 patch size.")
    ap.add_argument("--num-tracks", type=int, default=256)
    ap.add_argument("--level-sizes", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--init-ckpt", type=Path, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp]
    use_amp = args.amp != "fp32"

    train_clips, val_clips = default_train_val(
        args.data_root, subsets=args.subsets, val_frac=0.1, seed=42,
    )
    print(f"[train] data root: {args.data_root}")
    print(f"[train] {len(train_clips)} train clips / {len(val_clips)} val clips "
          f"(subsets={args.subsets})")

    train_ds = TAPVid3DDataset(train_clips, window_size=args.window,
                               augment=True, seed=args.seed,
                               max_queries=args.num_tracks,
                               image_size=args.image_size)
    val_ds = TAPVid3DDataset(val_clips, window_size=args.window,
                             augment=False, seed=0,
                             max_queries=args.num_tracks,
                             image_size=args.image_size)
    # `persistent_workers=False` so the worker process is torn down each epoch
    # and PyTorch's caching allocator inside it is freed. With `True`, the
    # worker held decoded clip residue across all 30k steps and crossed
    # systemd-oomd's PSI threshold around step 200 (v7 first/second launches).
    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_tracking,
        pin_memory=True, persistent_workers=False,
    )

    model = Mamba3Tracker(
        dim=args.dim, num_heads=args.num_heads, state_dim=args.state_dim,
        level_sizes=tuple(args.level_sizes),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] tracker params: {n_params:.2f}M, level_sizes={args.level_sizes}")

    if args.init_ckpt is not None:
        state = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        print(f"[train] loaded init weights from {args.init_ckpt}")

    loss_fn = TrackingLoss(TrackingLossWeights()).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = LambdaLR(optim, lr_lambda=lambda s: wsd(s, args.warmup, args.decay, args.steps))

    history: list[dict] = []
    cfg = vars(args).copy()
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    (args.out_dir / "cfg.json").write_text(json.dumps(cfg, indent=2))

    model.train()
    t0 = time.perf_counter()
    step = 0
    loader_iter = iter(loader)
    while step < args.steps:
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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # Skip the optimizer step on non-finite grads. Without this guard, a
        # single NaN gradient corrupts every parameter (NaN + anything = NaN)
        # and the rest of training silently runs on a dead model — exactly
        # the failure mode the v2 30k run hit.
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

        if step % args.log_every == 0:
            lr = sched.get_last_lr()[0]
            dt = time.perf_counter() - t0
            print(f"[train] step {step:6d}/{args.steps}  "
                  f"loss={loss_out.total.item():.4f}  "
                  f"pos={loss_out.pos.item():.4f}  mag={loss_out.mag.item():.4f}  "
                  f"dir={loss_out.dir.item():.4f}  reproj={loss_out.reproj.item():.4f}  "
                  f"vis={loss_out.vis.item():.4f}  "
                  f"spawn={loss_out.spawn.item():.4f}  smooth={loss_out.smooth.item():.4f}  "
                  f"lr={lr:.2e}  elapsed={dt:.0f}s", flush=True)
            history.append({
                "step": step, "lr": lr,
                "loss": float(loss_out.total.item()),
                "pos": float(loss_out.pos.item()),
                "mag": float(loss_out.mag.item()),
                "dir": float(loss_out.dir.item()),
                "reproj": float(loss_out.reproj.item()),
                "vis": float(loss_out.vis.item()),
                "spawn": float(loss_out.spawn.item()),
                "smooth": float(loss_out.smooth.item()),
            })

        if step > 0 and step % args.val_every == 0:
            v = _validate(model, val_ds, loss_fn, device, amp_dtype)
            print(f"[train] step {step:6d}  VAL  total={v['total']:.4f}  "
                  f"pos={v.get('pos', 0):.4f}  mag={v.get('mag', 0):.4f}  "
                  f"dir={v.get('dir', 0):.4f}  reproj={v.get('reproj', 0):.4f}  "
                  f"vis={v.get('vis', 0):.4f}  "
                  f"spawn={v.get('spawn', 0):.4f}  smooth={v.get('smooth', 0):.4f}",
                  flush=True)
            history.append({"step": step, "val": v})

        if step > 0 and step % args.ckpt_every == 0:
            p = _save_ckpt(args.out_dir, step, model, optim, cfg)
            print(f"[train] saved {p}", flush=True)

        (args.out_dir / "loss_history.json").write_text(json.dumps(history, indent=2))
        step += 1

    final = _save_ckpt(args.out_dir, args.steps, model, optim, cfg)
    print(f"[train] DONE — {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
