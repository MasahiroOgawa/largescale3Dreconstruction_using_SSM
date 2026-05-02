"""CIFAR-10 sanity check — Mamba-3 attention vs softmax attention vs CNN.

See `doc/PLAN_cifar10.md` for motivation and the acceptance gate. All three variants
share the same recipe (AdamW, lr 1e-3, wd 0.05, 5-ep warmup → cosine, batch 128,
50 epochs, RandomCrop+Flip, bf16 autocast on CUDA) and a matched parameter
budget (~2.7 M).

The two ViT variants share the same skeleton; `vit_mamba3` is built by calling
`mamba3_attn.patch.install_mamba3(net, which="backbone_only")` after construction —
the same swap path used by the DA3 depth pipeline.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_efficiency_patched import count_params, measure  # noqa: E402
from plot_cifar10_compare import make_all_figures  # noqa: E402

from mamba3_attn.patch import install_mamba3  # noqa: E402

# Stream per-epoch lines to log files in real time (block-buffering would hide
# progress until the buffer fills or the process exits).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(line_buffering=True)

# Args whose default type is Path; YAML loads them as str, so coerce on load.
_PATH_ARGS = ("out", "data_dir", "config")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_TRAIN_N = 50_000

VARIANTS = ("cnn", "vit_attn", "vit_mamba3")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class SmallResNet(nn.Module):
    """CIFAR-style ResNet: 3×3 stem + 3 stages × 2 BasicBlocks at {64, 128, 256}."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.stage2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.stage3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.gap(x).flatten(1)


class VanillaAttention(nn.Module):
    """Timm-style multi-head softmax attention.

    Exposes `qkv: nn.Linear(dim, 3*dim)`, `proj: nn.Linear(dim, dim)`,
    and `num_heads` so `mamba3_attn.patch.install_mamba3` can swap it in-place.
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer block. `self.attn` is the swap target."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VanillaAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """ViT-Tiny@d6: 4×4 patch embed (32→8×8=64 tokens), CLS, learnable pos-emb."""

    def __init__(self, img_size: int = 32, patch: int = 4, dim: int = 192,
                 depth: int = 6, num_heads: int = 3, mlp_ratio: float = 4.0):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        n_patches = (img_size // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.feat_dim = dim
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(variant: str, patch_size: int = 4, mamba_chunk_size: int | None = None,
                num_classes: int = 10) -> nn.Module:
    if variant == "cnn":
        return Classifier(SmallResNet(), num_classes)
    if variant == "vit_attn":
        return Classifier(ViTTiny(patch=patch_size), num_classes)
    if variant == "vit_mamba3":
        model = Classifier(ViTTiny(patch=patch_size), num_classes)
        n_swapped = install_mamba3(model, which="backbone_only", chunk_size=mamba_chunk_size)
        if n_swapped == 0:
            raise RuntimeError("install_mamba3 swapped 0 modules; backbone.blocks not found")
        chunk_str = "naive O(T²)" if mamba_chunk_size is None else f"chunked, chunk={mamba_chunk_size}"
        print(f"  [vit_mamba3] swapped {n_swapped} attention modules → Mamba3Attention ({chunk_str})")
        return model
    raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Data, schedule, train/eval
# ---------------------------------------------------------------------------


def make_loaders(batch_size: int, data_dir: Path, num_workers: int, device: torch.device):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    data_dir.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_tf)
    pin = device.type == "cuda"
    persistent = num_workers > 0
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=True, persistent_workers=persistent,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=persistent,
    )
    return train_dl, test_dl


def make_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class WarmupCosineStrategy:
    """Per-step linear warmup → cosine decay (LambdaLR wrapper)."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int):
        self.optimizer = optimizer
        self.scheduler = make_scheduler(optimizer, total_steps, warmup_steps)

    def step_per_step(self) -> None:
        self.scheduler.step()

    def step_per_epoch(self, train_loss: float) -> None:
        pass

    @property
    def last_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]


class WarmupPlateauStrategy:
    """Per-step linear warmup, then per-epoch ReduceLROnPlateau on EMA(train_loss).

    After warmup completes, end-of-epoch we update an EMA of train loss and feed
    it to ReduceLROnPlateau, which halves the LR (factor) when no improvement
    > threshold is seen for `patience` consecutive epochs.
    """

    def __init__(self, optimizer, peak_lr: float, warmup_steps: int,
                 factor: float, patience: int, threshold: float, min_lr: float,
                 ema_alpha: float):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.ema_alpha = ema_alpha
        self.step_count = 0
        self.loss_ema: float | None = None
        self.plateau = ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience,
            threshold=threshold, min_lr=min_lr,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = 0.0

    def step_per_step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.peak_lr * self.step_count / max(1, self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def step_per_epoch(self, train_loss: float) -> None:
        if self.loss_ema is None:
            self.loss_ema = train_loss
        else:
            self.loss_ema = self.ema_alpha * train_loss + (1.0 - self.ema_alpha) * self.loss_ema
        if self.step_count > self.warmup_steps:
            self.plateau.step(self.loss_ema)

    @property
    def last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def build_lr_strategy(args, optimizer, total_steps: int, warmup_steps: int):
    if args.lr_schedule == "plateau":
        return WarmupPlateauStrategy(
            optimizer, args.lr, warmup_steps,
            factor=args.plateau_factor, patience=args.plateau_patience,
            threshold=args.plateau_threshold, min_lr=args.plateau_min_lr,
            ema_alpha=args.plateau_loss_ema_alpha,
        )
    return WarmupCosineStrategy(optimizer, total_steps, warmup_steps)


def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_one_epoch(model, loader, optimizer, lr_strategy, device, grad_clip: float = 0.0):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    t0 = time.perf_counter()
    last_lr = lr_strategy.last_lr
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        lr_strategy.step_per_step()
        last_lr = lr_strategy.last_lr
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(-1) == y).sum().item()
        total_n += bs
    wall = time.perf_counter() - t0
    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
        "lr_end": last_lr,
        "wall_s": wall,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(-1) == y).sum().item()
        total_n += bs
    return {"loss": total_loss / max(1, total_n), "acc": total_correct / max(1, total_n)}


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------


def run_variant(variant: str, args, device: torch.device, train_dl, test_dl) -> dict:
    print(f"\n=== variant: {variant} ===")
    set_seed(args.seed)
    model = build_model(variant, patch_size=args.patch_size,
                        mamba_chunk_size=args.mamba_chunk_size).to(device)
    n_params = count_params(model)
    print(f"  params: {n_params/1e6:.2f} M ({n_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * min(args.warmup_epochs, args.epochs)
    lr_strategy = build_lr_strategy(args, optimizer, total_steps, warmup_steps)

    epochs_log: list[dict] = []
    best_acc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer, lr_strategy, device, args.grad_clip)
        ev = evaluate(model, test_dl, device)
        lr_strategy.step_per_epoch(tr["loss"])
        row = {
            "epoch": ep,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "test_loss": ev["loss"], "test_acc": ev["acc"],
            "lr_end": tr["lr_end"], "train_wall_s": tr["wall_s"],
        }
        epochs_log.append(row)
        print(f"  ep {ep:3d}  trL {tr['loss']:.4f} trA {tr['acc']*100:5.2f}  "
              f"teL {ev['loss']:.4f} teA {ev['acc']*100:5.2f}  "
              f"lr {tr['lr_end']:.2e}  {tr['wall_s']:.1f}s")
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path = args.out / f"ckpt_{variant}.pt"
    torch.save({"variant": variant, "state_dict": best_state, "best_test_acc": best_acc},
               ckpt_path)
    print(f"  saved best ckpt → {ckpt_path}  (test_acc {best_acc*100:.2f})")

    model.load_state_dict(best_state)
    model.eval()
    x = torch.randn(128, 3, 32, 32, device=device)
    with torch.inference_mode():
        eff = measure(lambda inp: model(inp), x, device, warmup=3, repeats=10)
    print(f"  efficiency: latency {eff['latency_ms']:.2f} ms  "
          f"peak {eff['peak_mib']:.1f} MiB (B=128, T=65)")

    train_walls = [r["train_wall_s"] for r in epochs_log]
    return {
        "variant": variant,
        "params": n_params,
        "best_test_acc": best_acc,
        "final_train_acc": epochs_log[-1]["train_acc"],
        "final_test_acc": epochs_log[-1]["test_acc"],
        "mean_train_wall_s": float(np.mean(train_walls)),
        "efficiency": eff,
        "epochs": epochs_log,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_results_json(out: Path, results: dict, args) -> None:
    spe = CIFAR10_TRAIN_N // args.batch_size
    cfg = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": 0.05, "seed": args.seed, "device": args.device,
        "warmup_epochs": args.warmup_epochs, "grad_clip": args.grad_clip,
        "lr_schedule": args.lr_schedule, "steps_per_epoch": spe,
        "patch_size": args.patch_size,
        "mamba_chunk_size": args.mamba_chunk_size,
    }
    if args.lr_schedule == "plateau":
        cfg.update({
            "plateau_factor": args.plateau_factor,
            "plateau_patience": args.plateau_patience,
            "plateau_threshold": args.plateau_threshold,
            "plateau_min_lr": args.plateau_min_lr,
            "plateau_loss_ema_alpha": args.plateau_loss_ema_alpha,
        })
    payload = {"config": cfg, "variants": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2))


def _fmt_mib(v: float) -> str:
    return "n/a (CPU)" if not math.isfinite(v) else f"{v:.1f}"


def write_efficiency_table_md(out: Path, results: dict) -> None:
    lines = [
        "# Efficiency table — CIFAR-10 compare (B=128, T=65)",
        "",
        "| Variant | Params (M) | Latency (ms) | Peak (MiB) | Train s/epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, r in results.items():
        lines.append(
            f"| {variant} | {r['params']/1e6:.2f} | "
            f"{r['efficiency']['latency_ms']:.2f} | "
            f"{_fmt_mib(r['efficiency']['peak_mib'])} | "
            f"{r['mean_train_wall_s']:.1f} |"
        )
    (out / "efficiency_table.md").write_text("\n".join(lines) + "\n")


def evaluate_gate(results: dict) -> dict:
    a = results.get("vit_attn")
    m = results.get("vit_mamba3")
    if a is None or m is None:
        return {"runnable": False,
                "reason": "gate requires both vit_attn and vit_mamba3 results"}
    acc_gap_pp = (m["best_test_acc"] - a["best_test_acc"]) * 100.0
    mem_a = a["efficiency"]["peak_mib"]
    mem_m = m["efficiency"]["peak_mib"]
    mem_ratio = mem_m / mem_a if (math.isfinite(mem_a) and mem_a > 0
                                  and math.isfinite(mem_m)) else float("nan")
    acc_pass = acc_gap_pp >= -2.0
    mem_pass = (not math.isfinite(mem_ratio)) or (mem_ratio <= 1.1)
    return {
        "runnable": True,
        "acc_gap_pp": acc_gap_pp, "acc_pass": acc_pass,
        "mem_ratio": mem_ratio, "mem_pass": mem_pass,
        "verdict": "PASS" if (acc_pass and mem_pass) else "FAIL",
    }


def write_summary_md(out: Path, results: dict, args) -> None:
    label = {"cnn": "CNN (small ResNet)", "vit_attn": "ViT-Tiny + softmax",
             "vit_mamba3": "ViT-Tiny + Mamba-3 SSD"}
    lines = [
        "# CIFAR-10 compare — summary",
        "",
        f"Recipe: AdamW lr={args.lr}, wd=0.05, {args.warmup_epochs}-ep warmup → {args.lr_schedule}, "
        f"grad_clip={args.grad_clip if args.grad_clip > 0 else 'off'}, "
        f"batch={args.batch_size}, epochs={args.epochs}, seed={args.seed}, device={args.device}.",
        "",
        "## Head-to-head",
        "",
        "| Variant | Params (M) | Train Acc | Test Acc | Train s/epoch | Test lat (ms, B=128) | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("cnn", "vit_attn", "vit_mamba3"):
        if variant not in results:
            continue
        r = results[variant]
        lines.append(
            f"| {label[variant]} | {r['params']/1e6:.2f} | "
            f"{r['final_train_acc']*100:.2f} | {r['best_test_acc']*100:.2f} | "
            f"{r['mean_train_wall_s']:.1f} | {r['efficiency']['latency_ms']:.2f} | "
            f"{_fmt_mib(r['efficiency']['peak_mib'])} |"
        )
    lines += [
        "",
        "## Reference leaderboard (orientation only — published numbers)",
        "",
        "| Reference | Test Acc | Notes |",
        "|---|---:|---|",
        "| ResNet-20 (He et al. 2015) | ~91.7% | 0.27 M params, 200 ep |",
        "| ResNet-110 | ~93.6% | 1.7 M params |",
        "| ViT-Tiny from scratch (Steiner et al. 2021) | ~85–88% | similar size, no strong aug |",
        "| ViT-Tiny + Mixup/Cutmix | ~92% | strong aug |",
        "| ViT-Huge (JFT-pretrained, fine-tuned) | >99% | not comparable; orientation only |",
        "",
        "Live: <https://paperswithcode.com/sota/image-classification-on-cifar-10>",
        "",
        "## Acceptance gate (`PLAN_cifar10.md §5`)",
        "",
    ]
    gate = evaluate_gate(results)
    if not gate["runnable"]:
        lines.append(f"_Not evaluable_: {gate['reason']}.")
    else:
        mem_str = (f"{gate['mem_ratio']:.2f}×" if math.isfinite(gate["mem_ratio"])
                   else "n/a (CPU)")
        lines += [
            f"- Acc gap (mamba3 − softmax): **{gate['acc_gap_pp']:+.2f} pp** "
            f"(threshold ≥ −2.00 pp) → {'PASS' if gate['acc_pass'] else 'FAIL'}",
            f"- Mem ratio (mamba3 / softmax): **{mem_str}** "
            f"(threshold ≤ 1.10×) → {'PASS' if gate['mem_pass'] else 'FAIL'}",
            "",
            f"**Verdict: {gate['verdict']}**",
        ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    """Load a YAML config and coerce known Path-typed keys to Path objects."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a YAML mapping, got {type(cfg).__name__}")
    for k in _PATH_ARGS:
        if k in cfg and cfg[k] is not None and not isinstance(cfg[k], Path):
            cfg[k] = Path(cfg[k])
    return cfg


def dump_config(args: argparse.Namespace, path: Path) -> None:
    """Write the resolved args namespace to a YAML file for recovery."""
    cfg: dict = {}
    for k, v in vars(args).items():
        if k == "config":
            continue
        cfg[k] = str(v) if isinstance(v, Path) else v
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    config_defaults = load_config(pre_args.config) if pre_args.config else {}

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config with defaults; CLI args override. Resolved "
                         "config is auto-dumped to <out>/config.yaml on every run.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("outputs/cifar10_compare"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--device", choices=["cuda", "cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    ap.add_argument("--patch-size", type=int, default=4,
                    help="ViT patch side length (px). 32x32 image: patch=4→T=65 (default), "
                         "patch=2→T=257, patch=1→T=1025. Ignored for cnn variant.")
    ap.add_argument("--mamba-chunk-size", type=int, default=None,
                    help="vit_mamba3 SSD mask chunk size. None=naive O(T²) (default; "
                         "fine for small T but OOMs at T≥1k). Set to 64 or 128 for long sequences.")
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="Linear-warmup length in epochs before cosine decay (default 5; §9.7 uses 10)")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="Max grad norm for clip_grad_norm_; 0 disables (default 0; §9.7 uses 1.0)")
    ap.add_argument("--lr-schedule", choices=["cosine", "plateau"], default="cosine",
                    help="Post-warmup LR decay: cosine (default) or ReduceLROnPlateau on EMA(train_loss)")
    ap.add_argument("--plateau-factor", type=float, default=0.5,
                    help="Multiplier applied to LR when plateau detected (default 0.5)")
    ap.add_argument("--plateau-patience", type=int, default=5,
                    help="Epochs of no-improvement (> threshold) before LR is reduced (default 5)")
    ap.add_argument("--plateau-threshold", type=float, default=1e-3,
                    help="Min loss improvement to count as 'still going down' (default 1e-3)")
    ap.add_argument("--plateau-min-lr", type=float, default=1e-6,
                    help="Floor for plateau LR reductions (default 1e-6)")
    ap.add_argument("--plateau-loss-ema-alpha", type=float, default=0.3,
                    help="EMA alpha for train-loss smoothing fed to plateau scheduler (default 0.3)")
    if config_defaults:
        unknown = set(config_defaults) - {a.dest for a in ap._actions}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        ap.set_defaults(**config_defaults)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dump_config(args, args.out / "config.yaml")
    print(f"[config] resolved → {args.out / 'config.yaml'}")
    device = torch.device(args.device)

    if device.type == "cpu" and args.epochs > 5:
        print("[smoke test] --device cpu → shortening epochs 50 → 5")
        args.epochs = 5

    set_seed(args.seed)
    train_dl, test_dl = make_loaders(args.batch_size, args.data_dir, args.num_workers, device)
    print(f"[data] CIFAR-10: {len(train_dl.dataset)} train / {len(test_dl.dataset)} test, "
          f"batch_size={args.batch_size}, steps/epoch={len(train_dl)}")

    results: dict[str, dict] = {}
    for variant in args.variants:
        results[variant] = run_variant(variant, args, device, train_dl, test_dl)

    write_results_json(args.out, results, args)
    write_efficiency_table_md(args.out, results)
    write_summary_md(args.out, results, args)
    fig_paths = make_all_figures(args.out / "results.json", args.out)
    for p in fig_paths:
        print(f"  figure → {p}")
    print(f"\n[done] artifacts → {args.out}")

    gate = evaluate_gate(results)
    if gate["runnable"]:
        print(f"[gate] {gate['verdict']}  "
              f"acc_gap={gate['acc_gap_pp']:+.2f}pp  "
              f"mem_ratio={gate['mem_ratio']:.2f}×")


if __name__ == "__main__":
    main()
