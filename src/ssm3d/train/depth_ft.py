"""Depth fine-tune: SSM-3D mixer + DimBridge → frozen DA3 DualDPT → depth.

Pipeline (PLAN §9 Phase C):
  1. Student starts from the Phase-B distillation checkpoint.
  2. DA3 DualDPT is loaded and **frozen** — we trust its pretrained mapping
     from 768-d features to metric depth.
  3. Trainables: SSM-3D mixer (blocks.*.attn.*) + DimBridge (per-layer 384→768).
  4. Loss: scale-invariant log-RMSE (SILog) + edge-aware smoothness on ETH3D
     non-terrains GT depth.
  5. AdamW, bf16 autocast on CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ssm3d.bridge import DimBridgeStack
from ssm3d.eval.dpt_adapter import SHARED_DPT_LAYERS, get_dualdpt
from ssm3d.train.distill import _amp_dtype
from ssm3d.train.overfit import _edge_aware_smoothness


@dataclass
class DepthFTConfig:
    steps: int = 2000
    lr_attn: float = 1e-4
    lr_bridge: float = 3e-4
    weight_decay: float = 0.05
    lambda_edge: float = 0.1
    batch_size: int = 2
    grad_clip: float = 1.0
    layers: tuple[int, ...] = SHARED_DPT_LAYERS
    amp_dtype: str = "bf16"
    log_every: int = 25
    ckpt_every: int = 500
    device: str = "cuda"
    silog_lambda: float = 0.85  # variance-term weight in SILog (Eigen 2014)


@dataclass
class DepthFTLog:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    loss_silog: list[float] = field(default_factory=list)
    loss_edge: list[float] = field(default_factory=list)


def silog_loss(
    pred: Tensor, target: Tensor, valid: Tensor, lam: float = 0.85, eps: float = 1e-6
) -> Tensor:
    """Scale-invariant log-RMSE loss (Eigen et al. 2014).

    Uses only valid GT pixels. `pred` and `target` must be positive where valid;
    we clamp before the log to avoid NaNs in mixed-precision.
    """
    valid = valid & (target > 0)
    if not valid.any():
        return pred.sum() * 0.0
    d = torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps))
    d = d[valid]
    mse = d.pow(2).mean()
    bias = d.mean().pow(2)
    return mse - lam * bias


def _prepare_bridge(layers: tuple[int, ...]) -> DimBridgeStack:
    return DimBridgeStack(num_layers=len(layers), in_dim=384)


def _set_trainables(
    student,
    bridge: DimBridgeStack,
    da3_model,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    attn_params: list[nn.Parameter] = []
    for name, p in student.named_parameters():
        if ".attn." in name:
            p.requires_grad_(True)
            attn_params.append(p)
        else:
            p.requires_grad_(False)
    bridge_params = list(bridge.parameters())
    for p in bridge_params:
        p.requires_grad_(True)
    for p in da3_model.parameters():
        p.requires_grad_(False)
    return attn_params, bridge_params


def _run_dpt(
    student,
    bridge: DimBridgeStack,
    dualdpt: nn.Module,
    images: Tensor,
    layers: tuple[int, ...],
) -> Tensor:
    """Student backbone → DimBridge (trainable) → frozen DualDPT → depth.

    Args:
        images: (B, S, 3, H, W).

    Returns:
        depth: (B, S, 1, H, W) predicted by DualDPT's main head.
    """
    out = student.backbone(images, export_feat_layers=list(layers))
    bridged = bridge(out.aux_features)  # list of (B, S, T, 768)
    feats_for_dpt = [(f,) for f in bridged]
    H, W = images.shape[-2], images.shape[-1]
    result = dualdpt(feats_for_dpt, H, W, patch_start_idx=0)
    depth = result[getattr(dualdpt, "head_main", "depth")]  # (B, S, 1 or C, Hd, Wd)
    if depth.dim() == 5 and depth.shape[2] > 1:
        depth = depth[:, :, :1]
    elif depth.dim() == 4:
        depth = depth.unsqueeze(2)
    if depth.shape[-2:] != (H, W):
        B, S = depth.shape[0], depth.shape[1]
        depth = F.interpolate(
            depth.reshape(B * S, 1, depth.shape[-2], depth.shape[-1]),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, S, 1, H, W)
    return depth.clamp_min(1e-4)


def depth_ft(
    student,
    da3_model,
    data_iter: Iterator,
    cfg: DepthFTConfig,
    out_dir: Path,
    bridge: DimBridgeStack | None = None,
) -> tuple[DepthFTLog, DimBridgeStack]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    if bridge is None:
        bridge = _prepare_bridge(cfg.layers)
    bridge.to(device).train()
    student.to(device).train()
    dualdpt = get_dualdpt(da3_model).to(device)
    dualdpt.eval()

    attn_params, bridge_params = _set_trainables(student, bridge, da3_model)
    opt = AdamW(
        [
            {"params": attn_params, "lr": cfg.lr_attn, "weight_decay": cfg.weight_decay},
            {"params": bridge_params, "lr": cfg.lr_bridge, "weight_decay": cfg.weight_decay},
        ]
    )
    sched = CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr_attn * 0.1)
    use_amp = device.type == "cuda" and cfg.amp_dtype != "fp32"
    amp_dtype = _amp_dtype(cfg.amp_dtype)

    log = DepthFTLog()
    for step in range(cfg.steps):
        batch_images: list[Tensor] = []
        batch_depth: list[Tensor] = []
        batch_valid: list[Tensor] = []
        # Build batch; skip samples without GT.
        while len(batch_images) < cfg.batch_size:
            sample = next(data_iter)
            if "gt_depth" not in sample or "valid_mask" not in sample:
                continue
            batch_images.append(sample["images"])
            batch_depth.append(sample["gt_depth"])
            batch_valid.append(sample["valid_mask"])
        images = torch.stack(batch_images, dim=0).to(device)  # (B, 1, 3, H, W)
        gt = torch.stack(batch_depth, dim=0).to(device).unsqueeze(2)  # (B, 1, 1, H, W)
        valid = torch.stack(batch_valid, dim=0).to(device).unsqueeze(2)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            pred = _run_dpt(student, bridge, dualdpt, images, cfg.layers)
            pred_f = pred.float()
            gt_f = gt.float()
            l_silog = silog_loss(
                pred_f, gt_f, valid, lam=cfg.silog_lambda
            )
            # edge-aware smoothness on flattened batch (drop the S=1 axis)
            B = pred_f.shape[0]
            l_edge = _edge_aware_smoothness(
                pred_f.reshape(B, 1, pred_f.shape[-2], pred_f.shape[-1]),
                images.float().reshape(B, 3, images.shape[-2], images.shape[-1]),
            )
            loss = l_silog + cfg.lambda_edge * l_edge

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(attn_params + bridge_params, cfg.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            log.step.append(step)
            log.loss.append(float(loss.item()))
            log.loss_silog.append(float(l_silog.item()))
            log.loss_edge.append(float(l_edge.item()))
            print(
                f"[depth_ft] step {step:5d}/{cfg.steps}  "
                f"loss={loss.item():.4f}  silog={l_silog.item():.4f}  "
                f"edge={l_edge.item():.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )

        if (step + 1) % cfg.ckpt_every == 0 or step == cfg.steps - 1:
            ckpt_path = out_dir / f"ckpt_{step + 1}.pt"
            torch.save(
                {
                    "step": step,
                    "student": student.state_dict(),
                    "bridge": bridge.state_dict(),
                    "cfg": cfg.__dict__,
                    "log": log.__dict__,
                },
                ckpt_path,
            )
            print(f"[depth_ft] saved {ckpt_path}")

    return log, bridge
