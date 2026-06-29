"""Phase 1: distill patched DA3-SMALL student against DA3-LARGE teacher.

Trainable: Mamba-3 attention modules (12 backbone + 4 cam_enc, 16 total).
Frozen: everything else (DPT head, cam_dec, cam_enc projections, MLPs,
norms, patch embed) — loaded from DA3-SMALL pretrained.

Loss: DA3 § 3.3 with target = teacher predictions.

DA3's `_process_camera_estimation` deletes ray output before returning,
so we register a forward hook on `model.head` (DualDPT) to capture the
raw {depth, depth_conf, ray, ray_conf} dict. cam_dec extrinsics come
from the final output dict.

Saves full DA3 state_dict each ckpt (~1 GB; small for ~3 ckpts).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ..mamba3 import Mamba3SelfAttention
from ..patch import count_mamba3_attn, install_mamba3
from .da3_loss import DA3LossWeights, da3_paper_loss
from .multi_view import multi_view_iterator


TEACHER_HF_MODEL = "depth-anything/DA3-LARGE"
STUDENT_HF_MODEL = DEFAULT_HF_MODEL  # DA3-SMALL


@dataclass
class Phase1Config:
    steps: int = 2000
    n_views: int = 4
    image_size: int = 504
    state_dim: int = 64
    use_fused_kernel: bool = True
    chunk_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_steps: int = 100
    decay_steps: int = 400
    log_every: int = 25
    ckpt_every: int = 500
    amp_dtype: str = "bf16"
    device: str = "cuda"
    seed: int = 0
    weights: DA3LossWeights = field(default_factory=DA3LossWeights)


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _capture_dpt_hook(model: nn.Module, captured: dict) -> torch.utils.hooks.RemovableHandle:
    """Hook on `model.head` (DualDPT) capturing raw output dict."""
    def hook(_m, _inp, out):
        for k in ("depth", "depth_conf", "ray", "ray_conf"):
            if k in out:
                captured[k] = out[k]
    return model.head.register_forward_hook(hook)


def _set_only_mamba3_trainable(da3) -> tuple[list[nn.Parameter], int]:
    """Freeze everything; unfreeze only Mamba3SelfAttention parameters.

    Returns (trainable_params, count).
    """
    for p in da3.model.parameters():
        p.requires_grad_(False)
    trainables: list[nn.Parameter] = []
    for m in da3.model.modules():
        if isinstance(m, Mamba3SelfAttention):
            for p in m.parameters():
                p.requires_grad_(True)
                trainables.append(p)
    return trainables, sum(p.numel() for p in trainables)


def _wsd_lambda(step: int, warmup: int, decay: int, total: int, floor: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    stable_end = total - decay
    if step < stable_end:
        return 1.0
    import math
    prog = (step - stable_end) / max(1, decay)
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * prog))


@torch.inference_mode()
def _teacher_forward(teacher, images: Tensor) -> dict[str, Tensor]:
    """Forward DA3-LARGE teacher and return {depth, depth_conf, ray, ray_conf, extrinsics}."""
    captured: dict = {}
    h = _capture_dpt_hook(teacher.model, captured)
    try:
        out = teacher.model(images.unsqueeze(0))
    finally:
        h.remove()
    captured["extrinsics"] = out.extrinsics  # (1, S, 3, 4) cam_dec
    return captured


def _student_forward(student, images: Tensor) -> dict[str, Tensor]:
    captured: dict = {}
    h = _capture_dpt_hook(student.model, captured)
    try:
        out = student.model(images.unsqueeze(0))
    finally:
        h.remove()
    captured["extrinsics"] = out.extrinsics
    return captured


def train(cfg: Phase1Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f"[phase1] loading teacher {TEACHER_HF_MODEL}")
    teacher = load_da3(TEACHER_HF_MODEL, device=cfg.device)
    teacher.model.eval()
    for p in teacher.model.parameters():
        p.requires_grad_(False)

    print(f"[phase1] loading student {STUDENT_HF_MODEL} + Mamba-3 swap")
    student = load_da3(STUDENT_HF_MODEL, device=cfg.device)
    n_swap = install_mamba3(
        student.model, which="all", state_dim=cfg.state_dim,
        use_fused_kernel=cfg.use_fused_kernel, chunk_size=cfg.chunk_size,
    )
    print(f"[phase1] swapped {n_swap} attentions; total Mamba3: {count_mamba3_attn(student.model)}")
    student = student.to(cfg.device)
    student.model.train()

    trainables, n_params = _set_only_mamba3_trainable(student)
    print(f"[phase1] trainable params: {n_params/1e6:.2f} M ({len(trainables)} tensors)")

    opt = AdamW(trainables, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = LambdaLR(
        opt,
        lambda s: _wsd_lambda(s, cfg.warmup_steps, cfg.decay_steps, cfg.steps),
    )

    use_amp = device.type == "cuda" and cfg.amp_dtype != "fp32"
    amp_dtype = _amp_dtype(cfg.amp_dtype)

    data = multi_view_iterator(
        data_root=Path("data"),
        n_views=cfg.n_views,
        image_size=cfg.image_size,
        seed=cfg.seed,
        require_gt=False,  # Phase 1 doesn't need GT
    )

    log_lines: list[str] = []
    for step in range(cfg.steps):
        batch = next(data)
        images = batch.images.to(device)  # (S, 3, H, W)

        with torch.no_grad():
            t_out = _teacher_forward(teacher, images)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            s_out = _student_forward(student, images)
            loss_out = da3_paper_loss(
                student=s_out,
                target=t_out,
                weights=cfg.weights,
            )
            loss = loss_out.total

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainables, cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            line = (
                f"[phase1] step {step:5d}/{cfg.steps}  "
                f"loss={loss.item():.4f}  "
                f"L_D={loss_out.l_depth.item():.4f}  "
                f"L_M={loss_out.l_ray.item():.4f}  "
                f"L_grad={loss_out.l_grad.item():.4f}  "
                f"L_P={loss_out.l_point.item():.4f}  "
                f"L_C={loss_out.l_cam.item():.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}  "
                f"[{batch.dataset}/{batch.scene}]"
            )
            print(line, flush=True)
            log_lines.append(line)

        if (step + 1) % cfg.ckpt_every == 0 or step == cfg.steps - 1:
            ckpt_path = out_dir / f"ckpt_{step + 1}.pt"
            torch.save(
                {
                    "step": step,
                    "model": student.model.state_dict(),
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"[phase1] saved {ckpt_path}", flush=True)
            (out_dir / "log.txt").write_text("\n".join(log_lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--out-dir", type=Path, default=Path("result/runs/phase1_distill"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp-dtype", type=str, default="bf16")
    ap.add_argument("--lambda-depth", type=float, default=1.0)
    ap.add_argument("--lambda-ray", type=float, default=1.0)
    ap.add_argument("--lambda-grad", type=float, default=1.0)
    ap.add_argument("--lambda-point", type=float, default=1.0)
    ap.add_argument("--lambda-cam", type=float, default=1.0)
    args = ap.parse_args()

    cfg = Phase1Config(
        steps=args.steps,
        n_views=args.n_views,
        image_size=args.image_size,
        state_dim=args.state_dim,
        lr=args.lr,
        ckpt_every=args.ckpt_every,
        device=args.device,
        amp_dtype=args.amp_dtype,
        weights=DA3LossWeights(
            lambda_depth=args.lambda_depth,
            lambda_ray=args.lambda_ray,
            lambda_grad=args.lambda_grad,
            lambda_point=args.lambda_point,
            lambda_cam=args.lambda_cam,
        ),
    )
    train(cfg, args.out_dir)


if __name__ == "__main__":
    main()
