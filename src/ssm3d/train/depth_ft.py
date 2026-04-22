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
from ssm3d.train.distill import _amp_dtype, _teacher_features
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
    # CM14: freeze the Mamba-3 mixer so only DimBridge trains in Phase-C.
    # Directly counters the CM3/CM5 overfit pattern (capacity vs 374 images).
    freeze_mixer: bool = False
    # CM17: add Phase-B feature-KD as a regulariser during Phase-C.
    # lambda_kd > 0 requires mixer trainable (KD gradient flows into attn params).
    lambda_kd: float = 0.0
    kd_patch_start_idx: int = 1  # drop cls token before KD (DINOv2 convention)
    # CM18: drop the learnable DimBridge; fall back to the static cat([f, f])
    # duplicate that DimBridge was initialised to mimic.
    no_bridge: bool = False
    # CM21: unfreeze DA3's DualDPT and train it at its own (low) LR. Targets
    # the CM20 diagnosis that the frozen head is the bottleneck. Deployed
    # param count is unchanged — DualDPT already ships with DA3.
    unfreeze_dpt: bool = False
    lr_dpt: float = 1e-5


@dataclass
class DepthFTLog:
    step: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    loss_silog: list[float] = field(default_factory=list)
    loss_edge: list[float] = field(default_factory=list)
    loss_kd: list[float] = field(default_factory=list)


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
    bridge: DimBridgeStack | None,
    da3_model,
    dualdpt: nn.Module,
    freeze_mixer: bool = False,
    unfreeze_dpt: bool = False,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    attn_params: list[nn.Parameter] = []
    for name, p in student.named_parameters():
        if ".attn." in name and not freeze_mixer:
            p.requires_grad_(True)
            attn_params.append(p)
        else:
            p.requires_grad_(False)
    bridge_params: list[nn.Parameter] = []
    if bridge is not None:
        bridge_params = list(bridge.parameters())
        for p in bridge_params:
            p.requires_grad_(True)
    for p in da3_model.parameters():
        p.requires_grad_(False)
    dpt_params: list[nn.Parameter] = []
    if unfreeze_dpt:
        dpt_params = list(dualdpt.parameters())
        for p in dpt_params:
            p.requires_grad_(True)
        dualdpt.train()
    return attn_params, bridge_params, dpt_params


def _kd_loss(
    student_feats: list[Tensor],
    teacher_feats: list[Tensor],
    patch_start_idx: int = 1,
) -> Tensor:
    """Per-layer L2 + (1 − cos) on patch tokens, averaged across layers.

    Mirrors `distill._per_layer_loss` with λ_l2 = λ_cos = 1. Teacher is
    detached (no grad flows into DA3).
    """
    total = student_feats[0].new_zeros(())
    for f_s, f_t in zip(student_feats, teacher_feats):
        s = f_s[:, patch_start_idx:].float()
        t = f_t[:, patch_start_idx:].float().detach()
        C = s.shape[-1]
        l2 = (s - t).pow(2).mean() / max(C, 1)
        s_n = F.normalize(s, dim=-1)
        t_n = F.normalize(t, dim=-1)
        cos_loss = 1.0 - (s_n * t_n).sum(dim=-1).mean()
        total = total + l2 + cos_loss
    return total / max(len(student_feats), 1)


def _run_dpt(
    student,
    bridge: DimBridgeStack | None,
    dualdpt: nn.Module,
    images: Tensor,
    layers: tuple[int, ...],
) -> tuple[Tensor, list[Tensor]]:
    """Student backbone → (DimBridge or cat-duplicate) → frozen DualDPT → depth.

    Args:
        images: (B, S, 3, H, W).

    Returns:
        depth: (B, S, 1, H, W) predicted by DualDPT's main head.
        aux_features: raw student features at `layers`, same shape as the
            backbone produced — kept for CM17 KD loss without a second forward.
    """
    out = student.backbone(images, export_feat_layers=list(layers))
    if bridge is None:
        bridged = [torch.cat([f, f], dim=-1) for f in out.aux_features]
    else:
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
    return depth.clamp_min(1e-4), out.aux_features


def depth_ft(
    student,
    da3_model,
    data_iter: Iterator,
    cfg: DepthFTConfig,
    out_dir: Path,
    bridge: DimBridgeStack | None = None,
) -> tuple[DepthFTLog, DimBridgeStack | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    if cfg.no_bridge:
        bridge = None
    elif bridge is None:
        bridge = _prepare_bridge(cfg.layers)
    if bridge is not None:
        bridge.to(device).train()
    student.to(device).train()
    dualdpt = get_dualdpt(da3_model).to(device)
    dualdpt.eval()

    attn_params, bridge_params, dpt_params = _set_trainables(
        student, bridge, da3_model, dualdpt,
        freeze_mixer=cfg.freeze_mixer, unfreeze_dpt=cfg.unfreeze_dpt,
    )
    if cfg.freeze_mixer and cfg.lambda_kd > 0:
        print(
            "[depth_ft] warn: lambda_kd>0 with freeze_mixer=True — KD gradient "
            "has no trainable path into the backbone; set freeze_mixer=False "
            "for CM17 to be active."
        )
    param_groups: list[dict] = []
    if attn_params:
        param_groups.append(
            {"params": attn_params, "lr": cfg.lr_attn, "weight_decay": cfg.weight_decay}
        )
    if bridge_params:
        param_groups.append(
            {"params": bridge_params, "lr": cfg.lr_bridge, "weight_decay": cfg.weight_decay}
        )
    if dpt_params:
        param_groups.append(
            {"params": dpt_params, "lr": cfg.lr_dpt, "weight_decay": cfg.weight_decay}
        )
        print(
            f"[depth_ft] CM21 DualDPT unfrozen: {sum(p.numel() for p in dpt_params)/1e6:.2f} M params "
            f"at lr={cfg.lr_dpt:.1e}"
        )
    if not param_groups:
        raise RuntimeError(
            "depth_ft: no trainable parameters (freeze_mixer + no_bridge leaves nothing)."
        )
    opt = AdamW(param_groups)
    lr_ref = cfg.lr_attn if attn_params else cfg.lr_bridge
    sched = CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=lr_ref * 0.1)
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

        teacher_feats: list[Tensor] | None = None
        if cfg.lambda_kd > 0:
            with torch.no_grad():
                teacher_feats = _teacher_features(da3_model, images, cfg.layers)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            pred, student_aux = _run_dpt(student, bridge, dualdpt, images, cfg.layers)
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
            l_kd = pred_f.new_zeros(())
            if teacher_feats is not None:
                student_flat = [
                    a.reshape(-1, a.shape[-2], a.shape[-1]) for a in student_aux
                ]
                l_kd = _kd_loss(
                    student_flat, teacher_feats, patch_start_idx=cfg.kd_patch_start_idx
                )
            loss = l_silog + cfg.lambda_edge * l_edge + cfg.lambda_kd * l_kd

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                attn_params + bridge_params + dpt_params, cfg.grad_clip
            )
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            log.step.append(step)
            log.loss.append(float(loss.item()))
            log.loss_silog.append(float(l_silog.item()))
            log.loss_edge.append(float(l_edge.item()))
            log.loss_kd.append(float(l_kd.item()))
            print(
                f"[depth_ft] step {step:5d}/{cfg.steps}  "
                f"loss={loss.item():.4f}  silog={l_silog.item():.4f}  "
                f"edge={l_edge.item():.4f}  kd={l_kd.item():.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )

        if (step + 1) % cfg.ckpt_every == 0 or step == cfg.steps - 1:
            ckpt_path = out_dir / f"ckpt_{step + 1}.pt"
            torch.save(
                {
                    "step": step,
                    "student": student.state_dict(),
                    "bridge": bridge.state_dict() if bridge is not None else None,
                    "dualdpt": (
                        dualdpt.state_dict() if cfg.unfreeze_dpt else None
                    ),
                    "cfg": cfg.__dict__,
                    "log": log.__dict__,
                },
                ckpt_path,
            )
            print(f"[depth_ft] saved {ckpt_path}")

    return log, bridge
