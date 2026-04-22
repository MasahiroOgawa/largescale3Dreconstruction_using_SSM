"""Feature distillation: train SSM-3D mixer params to match DA3 backbone features.

Pipeline (PLAN §9 Phase B):
  1. Teacher = frozen DA3 backbone. Exports 4 intermediate-layer tensors at
     indices DISTILL_LAYERS (single-stream 384-d for DA3-SMALL).
  2. Student = SSM-3D backbone with DA3 weights loaded (non-attn) and Mamba-3
     mixer params trainable. Exports the same 4 layers.
  3. Loss  per layer = λ_l2 · ||f_s − f_t||² / C + λ_cos · (1 − cos).
  4. Optimizer: AdamW. bf16 autocast on CUDA, fp32 on CPU smoke runs.

This trains the mixer to reproduce DA3's feature trajectory, closing the R1 gap
(random-init SSD attention). DimBridge is NOT trained here — it trains in
Phase C alongside the depth objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

DISTILL_LAYERS: tuple[int, ...] = (5, 7, 9, 11)
DISTILL_LAYERS_LARGE: tuple[int, ...] = (11, 15, 19, 23)


class DistillProjector(nn.Module):
    """CM20: per-layer Linear 384 -> teacher_dim. Phase-B only; discarded at Phase-C."""

    def __init__(self, num_layers: int, student_dim: int = 384, teacher_dim: int = 1024):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Linear(student_dim, teacher_dim) for _ in range(num_layers)]
        )

    def forward(self, layer_idx: int, f_s: Tensor) -> Tensor:
        return self.projs[layer_idx](f_s)


@dataclass
class DistillConfig:
    steps: int = 6000
    lr_attn: float = 3e-4
    lr_bridge: float = 1e-3
    weight_decay: float = 0.05
    lambda_l2: float = 1.0
    lambda_cos: float = 1.0
    batch_size: int = 4
    grad_clip: float = 1.0
    layers: tuple[int, ...] = DISTILL_LAYERS
    amp_dtype: str = "bf16"
    log_every: int = 50
    ckpt_every: int = 1000
    device: str = "cuda"


@dataclass
class DistillLog:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    loss_l2: list[float] = field(default_factory=list)
    loss_cos: list[float] = field(default_factory=list)


def _split_param_groups(
    student, bridge: nn.Module | None, cfg: DistillConfig
) -> list[dict]:
    """Partition student trainables into attn (low LR) + bridge (higher LR).

    Only parameters under blocks[*].attn.* are unfrozen in the student
    (patch_embed, MLPs, norms stay frozen — they were just loaded from DA3).
    """
    attn_params: list[nn.Parameter] = []
    for name, p in student.named_parameters():
        if ".attn." in name:
            p.requires_grad_(True)
            attn_params.append(p)
        else:
            p.requires_grad_(False)

    groups = [
        {"params": attn_params, "lr": cfg.lr_attn, "weight_decay": cfg.weight_decay},
    ]
    if bridge is not None:
        bridge_params = [p for p in bridge.parameters() if p.requires_grad]
        if bridge_params:
            groups.append(
                {"params": bridge_params, "lr": cfg.lr_bridge, "weight_decay": cfg.weight_decay}
            )
    return groups


def _teacher_features(
    da3_model, images: Tensor, layers: Iterable[int]
) -> list[Tensor]:
    """Run DA3 backbone, return aux features at `layers` as [B*S, T, C].

    The DA3 wrapper `DinoV2` forwards to `self.pretrained.get_intermediate_layers`
    with its baked-in `out_layers=[5,7,9,11]`; we need to call the inner DINOv2
    directly to pass our own `export_feat_layers` list.
    """
    _, aux = da3_model.model.backbone.pretrained.get_intermediate_layers(
        images, n=1, export_feat_layers=list(layers), ref_view_strategy="first"
    )
    # aux is a list of tensors, each shape (B, S, T, C). Merge B*S axes.
    return [a.reshape(-1, a.shape[-2], a.shape[-1]) for a in aux]


def _student_features(
    student, images: Tensor, layers: Iterable[int]
) -> list[Tensor]:
    out = student.backbone(images, export_feat_layers=list(layers))
    return [a.reshape(-1, a.shape[-2], a.shape[-1]) for a in out.aux_features]


def _per_layer_loss(
    f_s: Tensor,
    f_t: Tensor,
    patch_start_idx: int,
    cfg: DistillConfig,
    projector: DistillProjector | None = None,
    layer_idx: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """L2 + (1 − cos) on patch tokens only (drop cls+register).

    CM20: when ``projector`` is supplied, project student features to the
    teacher's embed dim before computing the match (enables DA3-LARGE teacher
    with a 384-dim student).
    """
    s = f_s[:, patch_start_idx:]
    if projector is not None:
        s = projector(layer_idx, s)
    s = s.float()
    t = f_t[:, patch_start_idx:].float().detach()
    C = s.shape[-1]
    l2 = (s - t).pow(2).mean() / max(C, 1)
    s_n = F.normalize(s, dim=-1)
    t_n = F.normalize(t, dim=-1)
    cos_sim = (s_n * t_n).sum(dim=-1).mean()
    cos_loss = 1.0 - cos_sim
    return cfg.lambda_l2 * l2 + cfg.lambda_cos * cos_loss, l2.detach(), cos_loss.detach()


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def distill(
    student,
    da3_model,
    data_iter: Iterator,
    cfg: DistillConfig,
    out_dir: Path,
    bridge: nn.Module | None = None,
) -> DistillLog:
    """Main distillation loop. `data_iter` yields dicts with "images" key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    student.to(device).train()
    da3_model.eval()
    for p in da3_model.parameters():
        p.requires_grad_(False)
    if bridge is not None:
        bridge.to(device).train()

    student_dim = int(student.backbone.vit.embed_dim)
    teacher_dim = int(da3_model.model.backbone.pretrained.embed_dim)
    projector: DistillProjector | None = None
    if teacher_dim != student_dim:
        projector = DistillProjector(
            num_layers=len(cfg.layers),
            student_dim=student_dim,
            teacher_dim=teacher_dim,
        ).to(device).train()
        print(
            f"[distill] CM20 projector enabled: {student_dim} -> {teacher_dim} "
            f"over {len(cfg.layers)} layers"
        )

    param_groups = _split_param_groups(student, bridge, cfg)
    if projector is not None:
        param_groups.append(
            {
                "params": list(projector.parameters()),
                "lr": cfg.lr_attn,
                "weight_decay": cfg.weight_decay,
            }
        )
    opt = AdamW(param_groups)
    sched = CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr_attn * 0.1)
    use_amp = device.type == "cuda" and cfg.amp_dtype != "fp32"
    amp_dtype = _amp_dtype(cfg.amp_dtype)
    patch_start_idx = int(getattr(da3_model.model.backbone, "patch_start_idx", 1))

    log = DistillLog()
    for step in range(cfg.steps):
        batch_images = []
        for _ in range(cfg.batch_size):
            batch = next(data_iter)
            batch_images.append(batch["images"])
        images = torch.stack(batch_images, dim=0).to(device)  # (B, S=1, 3, H, W)

        with torch.no_grad():
            teacher_feats = _teacher_features(da3_model, images, cfg.layers)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            student_feats = _student_features(student, images, cfg.layers)
            loss = torch.zeros((), device=device)
            loss_l2 = torch.zeros((), device=device)
            loss_cos = torch.zeros((), device=device)
            for li, (f_s, f_t) in enumerate(zip(student_feats, teacher_feats)):
                lk, l2k, cosk = _per_layer_loss(
                    f_s, f_t, patch_start_idx, cfg,
                    projector=projector, layer_idx=li,
                )
                loss = loss + lk
                loss_l2 = loss_l2 + l2k
                loss_cos = loss_cos + cosk
            loss = loss / len(cfg.layers)
            loss_l2 = loss_l2 / len(cfg.layers)
            loss_cos = loss_cos / len(cfg.layers)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        trainables = [
            p for grp in opt.param_groups for p in grp["params"] if p.grad is not None
        ]
        if cfg.grad_clip > 0 and trainables:
            torch.nn.utils.clip_grad_norm_(trainables, cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            log.step.append(step)
            log.loss.append(float(loss.item()))
            log.loss_l2.append(float(loss_l2.item()))
            log.loss_cos.append(float(loss_cos.item()))
            print(
                f"[distill] step {step:5d}/{cfg.steps}  "
                f"loss={loss.item():.4f}  l2={loss_l2.item():.4f}  "
                f"cos={loss_cos.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}"
            )

        if (step + 1) % cfg.ckpt_every == 0 or step == cfg.steps - 1:
            state = {
                "step": step,
                "student": student.state_dict(),
                "bridge": bridge.state_dict() if bridge is not None else None,
                "projector": (
                    projector.state_dict() if projector is not None else None
                ),
                "cfg": cfg.__dict__,
                "log": log.__dict__,
            }
            ckpt_path = out_dir / f"ckpt_{step + 1}.pt"
            torch.save(state, ckpt_path)
            print(f"[distill] saved {ckpt_path}")

    return log
