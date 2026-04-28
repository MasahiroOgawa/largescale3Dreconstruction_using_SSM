"""Unified training script for the 3×3 super-phase / sub-phase grid.

Three super-phases differ in **supervision target**:
- super=1 → DA3-SMALL teacher (warmstart, dim-matched, easiest gradient)
- super=2 → DA3-LARGE teacher (push toward LARGE quality)
- super=3 → GT supervision (real ground truth)

Three sub-phases differ in **trainable scope**:
- sub=1 → only Mamba-3 attentions (the swap target)
- sub=2 → top fusion blocks of DPT + entire cam_dec (head adapt)
- sub=3 → everything (full unfreeze, low LR)

So every (super, sub) pair has a constant supervision target (no
discontinuity within a super-phase) and a fixed scope progression.
Sub 1 → 2 → 3 within a super-phase forms a chain of init → ckpt.

Usage:
    uv run python -m ssm3d.train.train_super \\
        --super 1 --sub 1 \\
        --steps 500 \\
        --out-dir outputs/runs/sp1_sub1 \\
        --init-ckpt outputs/runs/<previous>/ckpt_500.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ..mamba3 import Mamba3SelfAttention
from ..patch import install_mamba3, count_mamba3_attn
from .da3_loss import DA3LossWeights, da3_paper_loss
from .gt_ray import gt_ray_map
from .multi_view import multi_view_iterator


TEACHER_HF = {
    1: "depth-anything/DA3-SMALL",
    2: "depth-anything/DA3-LARGE",
    3: None,  # GT — no teacher loaded
}


@dataclass
class SuperPhaseConfig:
    super_phase: int = 1
    sub_phase: int = 1
    init_ckpt: Optional[str] = None
    steps: int = 500
    n_views: int = 4
    image_size: int = 504
    state_dim: int = 64
    use_fused_kernel: bool = True
    chunk_size: int = 128
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_steps: int = 50
    decay_steps: int = 100
    log_every: int = 25
    ckpt_every: int = 250
    amp_dtype: str = "bf16"
    device: str = "cuda"
    seed: int = 0
    n_dpt_top_fusion_unfrozen: int = 2
    weights: DA3LossWeights = field(default_factory=DA3LossWeights)


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _wsd_lambda(step: int, warmup: int, decay: int, total: int, floor: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    stable_end = total - decay
    if step < stable_end:
        return 1.0
    prog = (step - stable_end) / max(1, decay)
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * prog))


def _capture_dpt_hook(model: nn.Module, captured: dict):
    def hook(_m, _inp, out):
        for k in ("depth", "depth_conf", "ray", "ray_conf"):
            if k in out:
                captured[k] = out[k]
    return model.head.register_forward_hook(hook)


@torch.inference_mode()
def _teacher_forward(teacher, images: Tensor) -> dict:
    captured: dict = {}
    h = _capture_dpt_hook(teacher.model, captured)
    try:
        out = teacher.model(images.unsqueeze(0))
    finally:
        h.remove()
    captured["extrinsics"] = out.extrinsics
    return captured


def _student_forward(student, images: Tensor) -> dict:
    captured: dict = {}
    h = _capture_dpt_hook(student.model, captured)
    try:
        out = student.model(images.unsqueeze(0))
    finally:
        h.remove()
    captured["extrinsics"] = out.extrinsics
    return captured


def set_trainables(student, scope: str, n_top_fusion: int) -> tuple[list[nn.Parameter], list[dict], dict]:
    """Set requires_grad per scope. Returns (params_flat, param_groups, info)."""
    for p in student.model.parameters():
        p.requires_grad_(False)

    info = {"attn": 0, "dpt": 0, "cam_dec": 0, "other": 0}
    attn_params: list[nn.Parameter] = []
    dpt_params: list[nn.Parameter] = []
    cam_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    # Mamba-3 attentions
    if scope in ("attn", "all"):
        for m in student.model.modules():
            if isinstance(m, Mamba3SelfAttention):
                for p in m.parameters():
                    p.requires_grad_(True)
                    attn_params.append(p)
        info["attn"] = sum(p.numel() for p in attn_params)

    # Top fusion blocks of DPT + cam_dec (head adapt)
    if scope in ("head", "all"):
        head = student.model.head
        fusion_module_names: list[str] = []
        for name, _ in head.named_modules():
            if "fusion" in name.lower() and name.count(".") <= 2:
                fusion_module_names.append(name)
        top_fusion = fusion_module_names[-(2 * n_top_fusion):] if fusion_module_names else []

        for name, p in head.named_parameters():
            if "output_conv" in name or "output_layer" in name:
                p.requires_grad_(True)
                dpt_params.append(p)
                continue
            if any(name.startswith(fb + ".") or name == fb for fb in top_fusion):
                p.requires_grad_(True)
                dpt_params.append(p)
        info["dpt"] = sum(p.numel() for p in dpt_params)

        if hasattr(student.model, "cam_dec") and student.model.cam_dec is not None:
            for p in student.model.cam_dec.parameters():
                p.requires_grad_(True)
                cam_params.append(p)
            info["cam_dec"] = sum(p.numel() for p in cam_params)

    # All — also unfreeze MLPs / norms / embed / etc.
    if scope == "all":
        for name, p in student.model.named_parameters():
            if not p.requires_grad:
                p.requires_grad_(True)
                other_params.append(p)
        info["other"] = sum(p.numel() for p in other_params)

    flat = attn_params + dpt_params + cam_params + other_params

    # Per-group LRs depending on scope
    groups: list[dict] = []
    if scope == "attn":
        groups = [{"params": attn_params, "lr": 3e-4, "tag": "attn"}]
    elif scope == "head":
        if dpt_params:
            groups.append({"params": dpt_params, "lr": 5e-5, "tag": "dpt"})
        if cam_params:
            groups.append({"params": cam_params, "lr": 1e-4, "tag": "cam_dec"})
    elif scope == "all":
        groups = [{"params": flat, "lr": 1e-5, "tag": "all"}]
    else:
        raise ValueError(f"Unknown scope {scope!r}")
    for g in groups:
        g.setdefault("weight_decay", 0.05)
    return flat, groups, info


def build_target(batch, teacher, image_size: int, device) -> tuple[dict, dict]:
    """Compose the target dict and (optional) GT-derived inputs for L_P/L_C.

    Returns (target_dict, gt_kwargs). gt_kwargs is passed to da3_paper_loss
    only when teacher is None (GT mode).
    """
    images = batch.images.to(device)

    if teacher is not None:
        # Teacher mode: target = teacher predictions
        t_out = _teacher_forward(teacher, images)
        target = {
            "depth": t_out["depth"],
            "ray": t_out["ray"],
            "depth_conf": t_out.get("depth_conf"),
            "ray_conf": t_out.get("ray_conf"),
            "extrinsics": t_out["extrinsics"],
        }
        return target, {}

    # GT mode
    gt_depth = batch.gt_depth.unsqueeze(0).to(device)
    gt_K = batch.gt_K.unsqueeze(0).to(device)
    gt_w2c = batch.gt_w2c.unsqueeze(0).to(device)
    gt_valid = batch.valid.unsqueeze(0).to(device)
    ray_hw = (image_size // 14 * 8, image_size // 14 * 8)
    gt_ray = gt_ray_map(gt_K, gt_w2c, (image_size, image_size), ray_hw)
    target = {
        "depth": gt_depth,
        "ray": gt_ray,
        "extrinsics": gt_w2c[..., :3, :].contiguous(),
    }
    return target, {
        "gt_depth": gt_depth, "gt_intrinsics": gt_K,
        "gt_w2c": gt_w2c, "gt_valid": gt_valid,
    }


def train(cfg: SuperPhaseConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    super_label = {1: "SMALL", 2: "LARGE", 3: "GT"}[cfg.super_phase]
    sub_label = {1: "attn", 2: "head", 3: "all"}[cfg.sub_phase]
    label = f"{cfg.super_phase}-{cfg.sub_phase} ({super_label} teacher / {sub_label} scope)"
    print(f"[train_super] {label}", flush=True)

    teacher = None
    if cfg.super_phase in (1, 2):
        hf = TEACHER_HF[cfg.super_phase]
        print(f"[train_super] loading teacher {hf}", flush=True)
        teacher = load_da3(hf, device=cfg.device)
        teacher.model.eval()
        for p in teacher.model.parameters():
            p.requires_grad_(False)

    print(f"[train_super] loading student DA3-SMALL + Mamba-3 swap", flush=True)
    student = load_da3(DEFAULT_HF_MODEL, device=cfg.device)
    install_mamba3(
        student.model, which="all", state_dim=cfg.state_dim,
        use_fused_kernel=cfg.use_fused_kernel, chunk_size=cfg.chunk_size,
    )
    print(f"[train_super] swapped {count_mamba3_attn(student.model)} attentions", flush=True)
    student = student.to(cfg.device)

    if cfg.init_ckpt is not None:
        print(f"[train_super] loading init from {cfg.init_ckpt}", flush=True)
        state = torch.load(cfg.init_ckpt, map_location=cfg.device, weights_only=False)
        student.model.load_state_dict(state["model"])
    student.model.train()

    flat, groups, info = set_trainables(student, sub_label, cfg.n_dpt_top_fusion_unfrozen)
    n_total = sum(info.values())
    print(f"[train_super] trainable: attn={info['attn']/1e6:.2f}M, "
          f"dpt={info['dpt']/1e6:.2f}M, cam_dec={info['cam_dec']/1e6:.2f}M, "
          f"other={info['other']/1e6:.2f}M (total {n_total/1e6:.2f}M)", flush=True)

    opt = AdamW(groups)
    sched = LambdaLR(opt, lambda s: _wsd_lambda(s, cfg.warmup_steps, cfg.decay_steps, cfg.steps))
    use_amp = device.type == "cuda" and cfg.amp_dtype != "fp32"
    amp_dtype = _amp_dtype(cfg.amp_dtype)

    require_gt = cfg.super_phase == 3
    data = multi_view_iterator(
        Path("data"), n_views=cfg.n_views, image_size=cfg.image_size,
        seed=cfg.seed, require_gt=require_gt,
    )

    log_lines: list[str] = []
    for step in range(cfg.steps):
        batch = next(data)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            target, gt_kwargs = build_target(batch, teacher, cfg.image_size, device)
            s_out = _student_forward(student, batch.images.to(device))
            loss_out = da3_paper_loss(student=s_out, target=target, weights=cfg.weights, **gt_kwargs)
            loss = loss_out.total

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(flat, cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            line = (
                f"[{cfg.super_phase}-{cfg.sub_phase}] step {step:5d}/{cfg.steps}  "
                f"loss={loss.item():.4f}  L_D={loss_out.l_depth.item():.4f}  "
                f"L_M={loss_out.l_ray.item():.4f}  L_grad={loss_out.l_grad.item():.4f}  "
                f"L_P={loss_out.l_point.item():.4f}  L_C={loss_out.l_cam.item():.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}  [{batch.dataset}/{batch.scene}]"
            )
            print(line, flush=True)
            log_lines.append(line)

        if (step + 1) % cfg.ckpt_every == 0 or step == cfg.steps - 1:
            ckpt_path = out_dir / f"ckpt_{step + 1}.pt"
            torch.save({"step": step, "model": student.model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print(f"[train_super] saved {ckpt_path}", flush=True)
            (out_dir / "log.txt").write_text("\n".join(log_lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--super", type=int, choices=[1, 2, 3], required=True, dest="super_phase")
    ap.add_argument("--sub", type=int, choices=[1, 2, 3], required=True, dest="sub_phase")
    ap.add_argument("--init-ckpt", type=str, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--ckpt-every", type=int, default=250)
    args = ap.parse_args()

    cfg = SuperPhaseConfig(
        super_phase=args.super_phase,
        sub_phase=args.sub_phase,
        init_ckpt=args.init_ckpt,
        steps=args.steps,
        n_views=args.n_views,
        image_size=args.image_size,
        ckpt_every=args.ckpt_every,
    )
    train(cfg, args.out_dir)


if __name__ == "__main__":
    main()
