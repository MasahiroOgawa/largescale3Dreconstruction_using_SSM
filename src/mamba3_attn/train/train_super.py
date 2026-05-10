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
    uv run python -m mamba3_attn.train.train_super \\
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

from ..data.view_split import split_views, write_split
from ..eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ..mamba3 import Mamba3SelfAttention
from ..patch import install_mamba3, count_mamba3_attn
from .da3_loss import DA3LossWeights, da3_paper_loss
from .gt_ray import gt_ray_map
from .multi_view import iter_single_scene, load_full_scene_cache, multi_view_iterator


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
    # When set, install_mamba3 swaps only these flat layer indices
    # (0..N_bb-1 backbone, N_bb..N_bb+N_cam-1 cam_enc).
    swap_layers: Optional[list[int]] = None
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
    lambda_feat: float = 1.0  # feature-distillation weight; auto-disabled when teacher dim differs
    # Skip the install_mamba3 swap so the student stays as un-patched DA3-SMALL.
    # Used for the per-scene-overfit "ceiling" baseline (PLAN §15.59 row 1) where
    # we want the same training recipe applied to the un-patched architecture.
    no_mamba3_swap: bool = False
    # Per-scene overfit (PLAN §15.59). When `scene_overfit` is set, training pulls
    # `n_views` per step from the train half of `split_views(...)` instead of cycling
    # across multiple scenes. The held-out test indices are written to `split.json`.
    scene_overfit: Optional[str] = None
    scene_dataset: str = "eth3d"
    train_frac: float = 0.75
    split_seed: int = 42
    candidate_views: int = 256
    frame_stride: int = 1
    augment: bool = True
    # Group-wise LRs for the `all` scope under scene-overfit. The default of `None`
    # keeps the legacy single-group lr=1e-5 path so existing super-1/2/3 runs are
    # unaffected.
    lr_attn: Optional[float] = None
    lr_head: Optional[float] = None
    lr_other: Optional[float] = None
    # Pass GT extrinsics + intrinsics into student.model.forward so cam_enc.trunk
    # runs and its mamba3 attention (flat layers 12..15 in DA3-SMALL) is on the
    # loss path. Required to train per-layer init weights for those layers under
    # the scene-overfit recipe (PLAN §15.59.6 / §15.59.5 Stage A). Only valid
    # with super_phase=3 (GT path), since super_phase=1/2 (teacher distillation)
    # has no GT extrinsics on hand.
    cam_posed: bool = False
    # Explicit (dataset, scene) list for the multi-scene training iterator.
    # When None, falls back to the hardcoded TRAIN_SCENES (eth3d non-terrains
    # + hiroom train + 7scenes train). Used by PLAN §15.59.8 random-split
    # protocol where train scenes are drawn from a runtime random partition.
    scenes: Optional[list[tuple[str, str]]] = None


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


FEAT_LAYERS = (5, 7, 9, 11)


@torch.inference_mode()
def _teacher_forward(teacher, images: Tensor, export_layers: Optional[tuple[int, ...]] = None) -> dict:
    captured: dict = {}
    h = _capture_dpt_hook(teacher.model, captured)
    try:
        if export_layers:
            out = teacher.model(images.unsqueeze(0), export_feat_layers=list(export_layers))
        else:
            out = teacher.model(images.unsqueeze(0))
    finally:
        h.remove()
    captured["extrinsics"] = out.extrinsics
    if export_layers and hasattr(out, "aux"):
        captured["feats"] = [out.aux[f"feat_layer_{i}"] for i in export_layers]
    return captured


def _student_forward(
    student, images: Tensor,
    export_layers: Optional[tuple[int, ...]] = None,
    extrinsics: Optional[Tensor] = None,
    intrinsics: Optional[Tensor] = None,
) -> dict:
    captured: dict = {}
    h = _capture_dpt_hook(student.model, captured)
    fwd_kw: dict = {}
    if extrinsics is not None:
        fwd_kw["extrinsics"] = extrinsics
    if intrinsics is not None:
        fwd_kw["intrinsics"] = intrinsics
    if export_layers:
        fwd_kw["export_feat_layers"] = list(export_layers)
    try:
        out = student.model(images.unsqueeze(0), **fwd_kw)
    finally:
        h.remove()
    captured["extrinsics"] = out.extrinsics
    if export_layers and hasattr(out, "aux"):
        captured["feats"] = [out.aux[f"feat_layer_{i}"] for i in export_layers]
    return captured


def _feature_distill_loss(student_feats: list[Tensor], teacher_feats: list[Tensor]) -> Tensor:
    """Per-layer ℓ2/C + (1 − cos) on patch features. Both inputs (B,S,Hp,Wp,C)."""
    total = student_feats[0].new_zeros(())
    for f_s, f_t in zip(student_feats, teacher_feats):
        s = f_s.float()
        t = f_t.float().detach()
        c = s.shape[-1]
        l2 = (s - t).pow(2).mean() / max(c, 1)
        s_n = torch.nn.functional.normalize(s, dim=-1)
        t_n = torch.nn.functional.normalize(t, dim=-1)
        cos_loss = 1.0 - (s_n * t_n).sum(dim=-1).mean()
        total = total + l2 + cos_loss
    return total / max(len(student_feats), 1)


def set_trainables(
    student, scope: str, n_top_fusion: int,
    lr_attn: Optional[float] = None,
    lr_head: Optional[float] = None,
    lr_other: Optional[float] = None,
) -> tuple[list[nn.Parameter], list[dict], dict]:
    """Set requires_grad per scope. Returns (params_flat, param_groups, info).

    When `scope == "all"` and any of `lr_attn / lr_head / lr_other` is set, the
    optimizer is built with three groups (attn / dpt+cam_dec / other) using the
    supplied LRs. This is the per-scene-overfit recipe (PLAN §15.59) where the
    Mamba-3 attentions need to move farther from warm-start than the DA3-pretrained
    heads or MLPs.
    """
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

    # cam_dec only — for T3 (PLAN §15.58 / CM-A): freeze everything except
    # net.cam_dec.* to test hypothesis 2 (cam_dec mismatch) in isolation.
    if scope == "cam_dec_only":
        if hasattr(student.model, "cam_dec") and student.model.cam_dec is not None:
            for p in student.model.cam_dec.parameters():
                p.requires_grad_(True)
                cam_params.append(p)
            info["cam_dec"] = sum(p.numel() for p in cam_params)

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
        if lr_attn is None and lr_head is None and lr_other is None:
            groups = [{"params": flat, "lr": 1e-5, "tag": "all"}]
        else:
            head_params = dpt_params + cam_params
            if attn_params:
                groups.append({"params": attn_params, "lr": lr_attn or 1e-5, "tag": "attn"})
            if head_params:
                groups.append({"params": head_params, "lr": lr_head or 1e-5, "tag": "head"})
            if other_params:
                groups.append({"params": other_params, "lr": lr_other or 1e-5, "tag": "other"})
    elif scope == "cam_dec_only":
        if cam_params:
            groups.append({"params": cam_params, "lr": 1e-4, "tag": "cam_dec"})
    else:
        raise ValueError(f"Unknown scope {scope!r}")
    for g in groups:
        g.setdefault("weight_decay", 0.05)
    return flat, groups, info


def build_target(batch, teacher, image_size: int, device,
                 export_feat_layers: Optional[tuple[int, ...]] = None) -> tuple[dict, dict, list[Tensor] | None]:
    """Compose the target dict, GT-derived inputs, and teacher's aux features.

    Returns (target_dict, gt_kwargs, teacher_feats).
    """
    images = batch.images.to(device)

    if teacher is not None:
        t_out = _teacher_forward(teacher, images, export_layers=export_feat_layers)
        target = {
            "depth": t_out["depth"],
            "ray": t_out["ray"],
            "depth_conf": t_out.get("depth_conf"),
            "ray_conf": t_out.get("ray_conf"),
            "extrinsics": t_out["extrinsics"],
        }
        return target, {}, t_out.get("feats")

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
    }, None


def train(cfg: SuperPhaseConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    super_label = {1: "SMALL", 2: "LARGE", 3: "GT"}[cfg.super_phase]
    sub_label = {1: "attn", 2: "head", 3: "all", 4: "cam_dec_only"}[cfg.sub_phase]
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

    print(f"[train_super] loading student DA3-SMALL"
          f"{' + Mamba-3 swap' if not cfg.no_mamba3_swap else ' (un-patched, no swap)'}",
          flush=True)
    student = load_da3(DEFAULT_HF_MODEL, device=cfg.device)
    if not cfg.no_mamba3_swap:
        install_mamba3(
            student.model, which="all", state_dim=cfg.state_dim,
            use_fused_kernel=cfg.use_fused_kernel, chunk_size=cfg.chunk_size,
            layer_indices=cfg.swap_layers,
        )
        n_swapped = count_mamba3_attn(student.model)
        layer_tag = f" (layers={cfg.swap_layers})" if cfg.swap_layers is not None else ""
        print(f"[train_super] swapped {n_swapped} attentions{layer_tag}", flush=True)
    if cfg.no_mamba3_swap and cfg.sub_phase == 1:
        raise ValueError(
            "no_mamba3_swap with sub=1 (attn_only) trains zero parameters: "
            "un-patched DA3 has no Mamba-3 attentions to unfreeze. Use sub=3 (all) instead."
        )
    student = student.to(cfg.device)

    if cfg.init_ckpt is not None:
        print(f"[train_super] loading init from {cfg.init_ckpt}", flush=True)
        state = torch.load(cfg.init_ckpt, map_location=cfg.device, weights_only=False)
        student.model.load_state_dict(state["model"])
    student.model.train()

    flat, groups, info = set_trainables(
        student, sub_label, cfg.n_dpt_top_fusion_unfrozen,
        lr_attn=cfg.lr_attn, lr_head=cfg.lr_head, lr_other=cfg.lr_other,
    )
    n_total = sum(info.values())
    print(f"[train_super] trainable: attn={info['attn']/1e6:.2f}M, "
          f"dpt={info['dpt']/1e6:.2f}M, cam_dec={info['cam_dec']/1e6:.2f}M, "
          f"other={info['other']/1e6:.2f}M (total {n_total/1e6:.2f}M)", flush=True)

    opt = AdamW(groups)
    sched = LambdaLR(opt, lambda s: _wsd_lambda(s, cfg.warmup_steps, cfg.decay_steps, cfg.steps))
    use_amp = device.type == "cuda" and cfg.amp_dtype != "fp32"
    amp_dtype = _amp_dtype(cfg.amp_dtype)

    require_gt = cfg.super_phase == 3
    if cfg.scene_overfit is not None:
        if cfg.super_phase != 3:
            raise ValueError(
                "scene_overfit requires super=3 (GT supervision). Per-scene "
                "overfit drops distillation entirely (PLAN §15.59 root-cause #2)."
            )
        print(
            f"[train_super] scene-overfit on {cfg.scene_dataset}/{cfg.scene_overfit}",
            flush=True,
        )
        cache = load_full_scene_cache(
            cfg.scene_dataset, cfg.scene_overfit, Path("data"),
            image_size=cfg.image_size, candidate_views=cfg.candidate_views,
            frame_stride=cfg.frame_stride,
        )
        n_views_total = cache.images.shape[0]
        train_idx, test_idx = split_views(
            n_views_total, train_frac=cfg.train_frac, seed=cfg.split_seed,
        )
        write_split(
            out_dir, num_views=n_views_total, train_frac=cfg.train_frac,
            seed=cfg.split_seed, train=train_idx, test=test_idx,
        )
        print(
            f"[train_super] scene has {n_views_total} views; "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"(seed={cfg.split_seed}, frac={cfg.train_frac})",
            flush=True,
        )
        data = iter_single_scene(
            cache, train_idx, n_views=cfg.n_views, augment=cfg.augment, seed=cfg.seed,
        )
    else:
        data = multi_view_iterator(
            Path("data"), n_views=cfg.n_views, image_size=cfg.image_size,
            seed=cfg.seed, require_gt=require_gt, scenes=cfg.scenes,
        )

    # Feature distillation enabled only for super-phase 1 (DA3-SMALL teacher,
    # dim-matched). LARGE teacher (super 2) has 1024-dim features that don't
    # match the student's 384-dim — dimensions can't align without a projector.
    feat_distill = cfg.super_phase == 1 and cfg.lambda_feat > 0
    export_layers = FEAT_LAYERS if feat_distill else None
    if feat_distill:
        print(f"[train_super] feature distillation ENABLED at layers {FEAT_LAYERS}", flush=True)

    log_lines: list[str] = []
    for step in range(cfg.steps):
        batch = next(data)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            target, gt_kwargs, t_feats = build_target(
                batch, teacher, cfg.image_size, device, export_feat_layers=export_layers,
            )
            student_extrinsics = gt_kwargs.get("gt_w2c") if cfg.cam_posed else None
            student_intrinsics = gt_kwargs.get("gt_intrinsics") if cfg.cam_posed else None
            s_out = _student_forward(
                student, batch.images.to(device),
                export_layers=export_layers,
                extrinsics=student_extrinsics,
                intrinsics=student_intrinsics,
            )
            loss_out = da3_paper_loss(student=s_out, target=target, weights=cfg.weights, **gt_kwargs)
            loss = loss_out.total

            l_feat = loss.new_zeros(())
            if feat_distill and t_feats is not None and "feats" in s_out:
                l_feat = _feature_distill_loss(s_out["feats"], t_feats)
                loss = loss + cfg.lambda_feat * l_feat

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
                f"L_feat={l_feat.item():.4f}  "
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
    ap.add_argument("--sub", type=int, choices=[1, 2, 3, 4], required=True, dest="sub_phase",
                    help="1=attn, 2=head, 3=all, 4=cam_dec_only (T3 / CM-A)")
    ap.add_argument("--init-ckpt", type=str, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--decay-steps", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=64)
    # Per-scene overfit (PLAN §15.59) — when --scene-overfit is set, super must be 3.
    ap.add_argument("--scene-overfit", type=str, default=None,
                    help="Scene name (e.g. 'terrains'). Activates per-scene overfit mode.")
    ap.add_argument("--scene-dataset", type=str, default="eth3d",
                    choices=["eth3d", "hiroom", "7scenes"])
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--candidate-views", type=int, default=256,
                    help="Cap on views loaded from the scene before splitting.")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Frame stride for 7Scenes long sequences.")
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable per-view photometric jitter (debugging only).")
    ap.add_argument("--lr-attn", type=float, default=None)
    ap.add_argument("--lr-head", type=float, default=None)
    ap.add_argument("--lr-other", type=float, default=None)
    ap.add_argument("--no-mamba3-swap", action="store_true",
                    help="Train un-patched DA3-SMALL (per-scene-overfit ceiling baseline).")
    ap.add_argument("--no-kendall-gal", action="store_true",
                    help="Use the legacy DA3 aleatoric form `c·|err| − λ·log(c)`. Default "
                    "(Kendall-Gal log-scale, PLAN §15.59.1) prices overconfidence "
                    "exponentially; legacy form suffers confidence collapse on overfit.")
    ap.add_argument("--swap-layer", type=int, action="append", default=None,
                    help="Per-layer ablation: restrict mamba3 swap to these flat layer "
                    "indices (0..N_bb-1 backbone, N_bb..N_bb+N_cam-1 cam_enc). Repeat to "
                    "swap multiple layers, e.g. --swap-layer 0 --swap-layer 5. When omitted, "
                    "all layers under `which` are swapped (existing behavior).")
    ap.add_argument("--cam-posed", action="store_true",
                    help="Pass GT extrinsics + intrinsics into student.model.forward so "
                    "cam_enc.trunk runs (and its mamba3 attention at flat layers 12..15 "
                    "is on the loss path). Required to train per-layer init for those "
                    "layers under scene-overfit. Only valid with --super 3 (GT path).")
    ap.add_argument("--scenes", type=str, default=None,
                    help="Explicit comma-separated (dataset:scene) list for the multi-scene "
                    "iterator, e.g. 'eth3d:courtyard,eth3d:facade,7scenes:chess'. Overrides "
                    "the hardcoded TRAIN_SCENES split. Used by PLAN §15.59.8 random scene "
                    "split. Mutually exclusive with --scene-overfit.")
    args = ap.parse_args()
    if args.cam_posed and args.super_phase != 3:
        ap.error("--cam-posed requires --super 3 (GT extrinsics needed)")
    if args.scenes and args.scene_overfit:
        ap.error("--scenes and --scene-overfit are mutually exclusive")
    scenes_list: Optional[list[tuple[str, str]]] = None
    if args.scenes:
        scenes_list = [tuple(s.split(":", 1)) for s in args.scenes.split(",")]  # type: ignore[misc]

    cfg = SuperPhaseConfig(
        super_phase=args.super_phase,
        sub_phase=args.sub_phase,
        init_ckpt=args.init_ckpt,
        steps=args.steps,
        n_views=args.n_views,
        image_size=args.image_size,
        ckpt_every=args.ckpt_every,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        chunk_size=args.chunk_size,
        state_dim=args.state_dim,
        scene_overfit=args.scene_overfit,
        scene_dataset=args.scene_dataset,
        train_frac=args.train_frac,
        split_seed=args.split_seed,
        candidate_views=args.candidate_views,
        frame_stride=args.frame_stride,
        augment=not args.no_augment,
        lr_attn=args.lr_attn,
        lr_head=args.lr_head,
        lr_other=args.lr_other,
        no_mamba3_swap=args.no_mamba3_swap,
        swap_layers=args.swap_layer,
        cam_posed=args.cam_posed,
        scenes=scenes_list,
    )
    cfg.weights.use_kendall_gal = not args.no_kendall_gal
    train(cfg, args.out_dir)


if __name__ == "__main__":
    main()
