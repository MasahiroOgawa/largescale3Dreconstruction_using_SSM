"""Phase 3: full-unfreeze co-adaptation.

Trainable: everything (Mamba-3 attention + heads + MLPs + norms + embed).
Low LR (1e-5 across all groups), short schedule (~500 steps), ckpts every
100 steps for rollback.

Loss: DA3 § 3.3 against GT (same as Phase 2).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ..patch import install_mamba3
from .da3_loss import DA3LossWeights, da3_paper_loss
from .gt_ray import gt_ray_map
from .multi_view import multi_view_iterator
from .phase1 import _amp_dtype, _wsd_lambda
from .phase2 import _student_forward_capture


@dataclass
class Phase3Config:
    init_ckpt: str = "outputs/runs/phase2_gt/ckpt_500.pt"
    steps: int = 500
    n_views: int = 4
    image_size: int = 504
    state_dim: int = 64
    use_fused_kernel: bool = True
    chunk_size: int = 128
    lr: float = 1e-5
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_steps: int = 50
    decay_steps: int = 100
    log_every: int = 25
    ckpt_every: int = 100
    amp_dtype: str = "bf16"
    device: str = "cuda"
    seed: int = 0
    weights: DA3LossWeights = field(default_factory=DA3LossWeights)


def train(cfg: Phase3Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f"[phase3] loading DA3-SMALL + Mamba-3 swap")
    student = load_da3(DEFAULT_HF_MODEL, device=cfg.device)
    install_mamba3(
        student.model, which="all", state_dim=cfg.state_dim,
        use_fused_kernel=cfg.use_fused_kernel, chunk_size=cfg.chunk_size,
    )
    student = student.to(cfg.device)

    print(f"[phase3] loading Phase 2 ckpt {cfg.init_ckpt}")
    state = torch.load(cfg.init_ckpt, map_location=cfg.device, weights_only=False)
    student.model.load_state_dict(state["model"])
    student.model.train()

    # Unfreeze everything
    for p in student.model.parameters():
        p.requires_grad_(True)
    trainables = [p for p in student.model.parameters() if p.requires_grad]
    print(f"[phase3] trainable: {sum(p.numel() for p in trainables)/1e6:.2f}M")

    opt = AdamW(trainables, lr=cfg.lr, weight_decay=cfg.weight_decay)
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
        images = batch.images.to(device)
        gt_depth = batch.gt_depth.unsqueeze(0).to(device)
        gt_K = batch.gt_K.unsqueeze(0).to(device)
        gt_w2c = batch.gt_w2c.unsqueeze(0).to(device)
        gt_valid = batch.valid.unsqueeze(0).to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            s_out = _student_forward_capture(student, images)
            ray_hw = s_out["ray"].shape[-3:-1]
            gt_ray = gt_ray_map(gt_K, gt_w2c, (cfg.image_size, cfg.image_size), ray_hw)
            target = {
                "depth": gt_depth, "ray": gt_ray,
                "extrinsics": gt_w2c[..., :3, :].contiguous(),
            }
            loss_out = da3_paper_loss(
                student=s_out, target=target, weights=cfg.weights,
                gt_depth=gt_depth, gt_intrinsics=gt_K, gt_w2c=gt_w2c, gt_valid=gt_valid,
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
                f"[phase3] step {step:5d}/{cfg.steps}  "
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
            print(f"[phase3] saved {ckpt_path}", flush=True)
            (out_dir / "log.txt").write_text("\n".join(log_lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default="outputs/runs/phase2_gt/ckpt_500.pt")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/runs/phase3_unfreeze"))
    ap.add_argument("--ckpt-every", type=int, default=100)
    args = ap.parse_args()
    cfg = Phase3Config(init_ckpt=args.init, steps=args.steps, lr=args.lr, ckpt_every=args.ckpt_every)
    train(cfg, args.out_dir)


if __name__ == "__main__":
    main()
