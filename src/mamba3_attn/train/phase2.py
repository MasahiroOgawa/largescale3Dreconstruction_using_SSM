"""Phase 2: GT-supervised head adaptation.

Trainable: top several layers of DPT (last fusion blocks + final output
convs) + entire cam_dec MLP. Mamba-3 attention frozen (carrying Phase 1
distillation). Everything else frozen at DA3-SMALL pretrained.

Loss: DA3 § 3.3 against GT (depth + ray + 3D points + cam_dec extrinsics).

Loads from Phase 1 ckpt (`outputs/runs/phase1_distill/ckpt_<N>.pt`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ..patch import install_mamba3
from .da3_loss import DA3LossWeights, da3_paper_loss
from .gt_ray import gt_ray_map
from .multi_view import multi_view_iterator
from .phase1 import _capture_dpt_hook, _wsd_lambda, _amp_dtype


@dataclass
class Phase2Config:
    init_ckpt: str = "outputs/runs/phase1_distill/ckpt_1000.pt"
    steps: int = 500
    n_views: int = 4
    image_size: int = 504
    state_dim: int = 64
    use_fused_kernel: bool = True
    chunk_size: int = 128
    lr_dpt: float = 5e-5
    lr_cam: float = 1e-4
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


def _set_phase2_trainables(da3, n_top_fusion: int) -> tuple[list[nn.Parameter], list[nn.Parameter], int]:
    """Freeze everything; unfreeze top fusion blocks of DPT + cam_dec.

    DA3 DualDPT structure: 4 reassemble + 2 sets of 4 fusion blocks +
    2 output convs (one per branch: depth, ray). Top-N fusion = the last
    N fusion blocks of *each* branch (depth + ray), plus output convs.

    Returns (dpt_params, cam_params, total_count_M).
    """
    for p in da3.model.parameters():
        p.requires_grad_(False)

    dpt_params: list[nn.Parameter] = []
    cam_params: list[nn.Parameter] = []

    head = da3.model.head  # DualDPT
    # Look for fusion blocks in head; structure differs slightly per DA3 version
    fusion_module_names: list[str] = []
    for name, _ in head.named_modules():
        if "fusion" in name.lower() and name.count(".") <= 2:
            fusion_module_names.append(name)
    # Heuristic: top-N fusions per branch = last 2N entries of fusion_module_names
    top_fusion = fusion_module_names[-(2 * n_top_fusion):] if fusion_module_names else []

    for name, p in head.named_parameters():
        # Output conv layers always unfrozen
        if "output_conv" in name or "output_layer" in name:
            p.requires_grad_(True)
            dpt_params.append(p)
            continue
        # Top-N fusion blocks unfrozen
        if any(name.startswith(fb + ".") or name == fb for fb in top_fusion):
            p.requires_grad_(True)
            dpt_params.append(p)
            continue

    # cam_dec: tiny MLP, unfreeze everything
    if hasattr(da3.model, "cam_dec") and da3.model.cam_dec is not None:
        for p in da3.model.cam_dec.parameters():
            p.requires_grad_(True)
            cam_params.append(p)

    n = sum(p.numel() for p in dpt_params + cam_params)
    return dpt_params, cam_params, n


def _student_forward_capture(student, images: Tensor) -> dict:
    captured: dict = {}
    h = _capture_dpt_hook(student.model, captured)
    try:
        out = student.model(images.unsqueeze(0))
    finally:
        h.remove()
    captured["extrinsics"] = out.extrinsics
    return captured


def train(cfg: Phase2Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f"[phase2] loading DA3-SMALL + Mamba-3 swap")
    student = load_da3(DEFAULT_HF_MODEL, device=cfg.device)
    install_mamba3(
        student.model, which="all", state_dim=cfg.state_dim,
        use_fused_kernel=cfg.use_fused_kernel, chunk_size=cfg.chunk_size,
    )
    student = student.to(cfg.device)

    print(f"[phase2] loading Phase 1 ckpt {cfg.init_ckpt}")
    state = torch.load(cfg.init_ckpt, map_location=cfg.device, weights_only=False)
    student.model.load_state_dict(state["model"])
    student.model.train()

    dpt_p, cam_p, n_params = _set_phase2_trainables(student, cfg.n_dpt_top_fusion_unfrozen)
    print(f"[phase2] trainable: dpt={sum(p.numel() for p in dpt_p)/1e6:.2f}M, cam_dec={sum(p.numel() for p in cam_p)/1e6:.2f}M (total {n_params/1e6:.2f}M)")

    param_groups = []
    if dpt_p:
        param_groups.append({"params": dpt_p, "lr": cfg.lr_dpt, "weight_decay": cfg.weight_decay})
    if cam_p:
        param_groups.append({"params": cam_p, "lr": cfg.lr_cam, "weight_decay": cfg.weight_decay})
    if not param_groups:
        raise RuntimeError("No trainable params in Phase 2 — check head structure.")
    opt = AdamW(param_groups)
    sched = LambdaLR(opt, lambda s: _wsd_lambda(s, cfg.warmup_steps, cfg.decay_steps, cfg.steps))

    use_amp = device.type == "cuda" and cfg.amp_dtype != "fp32"
    amp_dtype = _amp_dtype(cfg.amp_dtype)

    data = multi_view_iterator(
        Path("data"), n_views=cfg.n_views, image_size=cfg.image_size,
        seed=cfg.seed, require_gt=True,
    )

    log_lines: list[str] = []
    for step in range(cfg.steps):
        batch = next(data)
        images = batch.images.to(device)             # (S, 3, H, W)
        gt_depth = batch.gt_depth.unsqueeze(0).to(device)   # (1, S, H, W)
        gt_K = batch.gt_K.unsqueeze(0).to(device)           # (1, S, 3, 3)
        gt_w2c = batch.gt_w2c.unsqueeze(0).to(device)       # (1, S, 4, 4)
        gt_valid = batch.valid.unsqueeze(0).to(device)      # (1, S, H, W)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            s_out = _student_forward_capture(student, images)

            # GT targets
            ray_hw = s_out["ray"].shape[-3:-1]
            gt_ray = gt_ray_map(gt_K, gt_w2c, (cfg.image_size, cfg.image_size), ray_hw)
            target = {
                "depth": gt_depth,
                "ray": gt_ray,
                "extrinsics": gt_w2c[..., :3, :].contiguous(),  # (1, S, 3, 4) w2c
            }
            loss_out = da3_paper_loss(
                student=s_out, target=target, weights=cfg.weights,
                gt_depth=gt_depth, gt_intrinsics=gt_K, gt_w2c=gt_w2c, gt_valid=gt_valid,
            )
            loss = loss_out.total

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(dpt_p + cam_p, cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            line = (
                f"[phase2] step {step:5d}/{cfg.steps}  "
                f"loss={loss.item():.4f}  D={loss_out.l_depth.item():.4f}  "
                f"M={loss_out.l_ray.item():.4f}  grad={loss_out.l_grad.item():.4f}  "
                f"P={loss_out.l_point.item():.4f}  C={loss_out.l_cam.item():.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}  [{batch.dataset}/{batch.scene}]"
            )
            print(line, flush=True)
            log_lines.append(line)

        if (step + 1) % cfg.ckpt_every == 0 or step == cfg.steps - 1:
            ckpt_path = out_dir / f"ckpt_{step + 1}.pt"
            torch.save({"step": step, "model": student.model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print(f"[phase2] saved {ckpt_path}", flush=True)
            (out_dir / "log.txt").write_text("\n".join(log_lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="outputs/runs/phase1_distill/ckpt_1000.pt")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/runs/phase2_gt"))
    ap.add_argument("--lr-dpt", type=float, default=5e-5)
    ap.add_argument("--lr-cam", type=float, default=1e-4)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--n-dpt-top-fusion-unfrozen", type=int, default=2)
    args = ap.parse_args()
    cfg = Phase2Config(
        init_ckpt=args.init, steps=args.steps,
        lr_dpt=args.lr_dpt, lr_cam=args.lr_cam,
        ckpt_every=args.ckpt_every,
        n_dpt_top_fusion_unfrozen=args.n_dpt_top_fusion_unfrozen,
    )
    train(cfg, args.out_dir)


if __name__ == "__main__":
    main()
