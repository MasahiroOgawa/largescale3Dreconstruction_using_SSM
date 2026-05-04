"""DA3 paper § 3.3 training loss.

Implements the full loss from Depth-Anything-3 (Eqs. 1–3 in §3.3):

    L = L_D(D̂, D)              # aleatoric ℓ1 on depth
      + L_M(R̂, M)              # aleatoric ℓ1 on ray map (origin + direction)
      + L_P(D̂⊙d + t, P)        # 3D point ℓ1 in world space
      + β · L_C(ĉ, v)          # ℓ1 on cam_dec extrinsics
      + α · L_grad(D̂, D)       # ℓ1 on depth gradients (∂x, ∂y)

All terms are ℓ1-based (DA3 paper: "All loss terms are based on the ℓ1 norm").

Key shapes (DA3-SMALL at 504² input):
- depth, depth_conf:  (B, S, H,    W)        H=W=504
- ray, ray_conf:      (B, S, H/k, W/k, 6)    H/k=W/k=288 typically
- extrinsics (cam_dec): (B, S, 3, 4)         w2c, last row implicit
- intrinsics:         (B, S, 3, 3)

Ray channels: M[..., :3] = origin t (world), M[..., 3:] = direction d
(world frame, magnitude-preserved). Per the paper:

    P = t + D(u,v) · d

For Phase 1 distillation: target = teacher predictions (no GT needed).
For Phase 2/3 GT supervision: target derived from GT depth + GT cam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DA3LossWeights:
    """Per-term weights. DA3 paper uses α = β = 1, λ_c = 1."""
    lambda_depth: float = 1.0
    lambda_ray: float = 1.0
    lambda_grad: float = 1.0       # α
    lambda_point: float = 1.0
    lambda_cam: float = 1.0        # β
    lambda_conf_log: float = 1.0   # λ_c (aleatoric log-penalty, legacy form only)
    use_aleatoric: bool = True
    # Default OFF: original DA3 loss (`c·|err| − λ·log(c)`). The §15.59.1 OOM
    # cascade that motivated the Kendall-Gal pivot is now solved by the
    # phase4_evaluator TSDF guard — collapse no longer crashes the host —
    # so we keep the original DA3 paper setup for direct comparability.
    # The KG branch remains for ablation; flip this to True to enable it.
    use_kendall_gal: bool = False


@dataclass
class DA3LossOut:
    total: Tensor
    l_depth: Tensor
    l_ray: Tensor
    l_grad: Tensor
    l_point: Tensor
    l_cam: Tensor


def _l1_aleatoric(
    pred: Tensor, target: Tensor, conf: Optional[Tensor],
    valid: Optional[Tensor], lambda_log: float, use_aleatoric: bool,
    use_kendall_gal: bool = False,
) -> Tensor:
    """Heteroscedastic ℓ1 loss with per-pixel `conf` (aleatoric).

    Uses `valid` mask to ignore zero/invalid GT (NaN/inf) where present.
    `conf` may be lower rank than the per-pixel error (e.g. (B,S,H,W) for
    a (B,S,H,W,6) ray error); broadcast by appending singleton channels.

    Two forms (PLAN §15.59.1):

    - `use_kendall_gal=True` (default): heteroscedastic Laplace via the
      Kendall-Gal log-scale parameterization. Treats `b = c - 1` as the
      Laplace scale (≥ 0 by DA3's `expp1` head) and `s = log b` as the
      learned log-scale:

          L = (|err| / b) + s = exp(−s)·|err| + s

      Overconfidence (`b → 0`) is priced exponentially via `|err|/b`, so
      the model can only achieve negative loss by genuinely fitting the
      data — no confidence collapse.

    - `use_kendall_gal=False`: legacy DA3 form `c·|err| − λ·log(c)`. Kept
      for ablation/regression checks only; suffers confidence collapse on
      per-scene overfit (§15.59.1 ckpts hit `L_M = −19/−21`).
    """
    err = (pred.float() - target.float()).abs()
    if valid is not None:
        v = valid.float()
        while v.dim() < err.dim():
            v = v.unsqueeze(-1)
        err = err * v
        denom = v.sum().clamp_min(1.0) * (err.shape[-1] if err.dim() > v.dim() else 1)
    else:
        denom = torch.tensor(float(err.numel()), device=err.device)
    if conf is None or not use_aleatoric:
        return err.sum() / denom
    if use_kendall_gal:
        b = (conf.float() - 1.0).clamp_min(1e-6)
        while b.dim() < err.dim():
            b = b.unsqueeze(-1)
        s = torch.log(b)
        weighted = err / b + s
    else:
        c = conf.float().clamp_min(1e-6)
        while c.dim() < err.dim():
            c = c.unsqueeze(-1)
        weighted = c * err - lambda_log * torch.log(c)
    if valid is not None:
        v = valid.float()
        while v.dim() < weighted.dim():
            v = v.unsqueeze(-1)
        weighted = weighted * v
    return weighted.sum() / denom


def _depth_grad_l1(pred: Tensor, target: Tensor, valid: Optional[Tensor]) -> Tensor:
    """ℓ1 on depth gradients ∂x, ∂y. Eq. 3 of DA3 paper."""
    p, t = pred.float(), target.float()
    sx = p[..., 1:] - p[..., :-1]
    tx = t[..., 1:] - t[..., :-1]
    sy = p[..., 1:, :] - p[..., :-1, :]
    ty = t[..., 1:, :] - t[..., :-1, :]
    if valid is not None:
        vx = (valid[..., 1:].float() * valid[..., :-1].float())
        vy = (valid[..., 1:, :].float() * valid[..., :-1, :].float())
        lx = ((sx - tx).abs() * vx).sum() / vx.sum().clamp_min(1.0)
        ly = ((sy - ty).abs() * vy).sum() / vy.sum().clamp_min(1.0)
        return lx + ly
    return (sx - tx).abs().mean() + (sy - ty).abs().mean()


def _depth_to_ray_resolution(depth: Tensor, ray_hw: tuple[int, int]) -> Tensor:
    """Bilinear-resize depth (B, S, H, W) to ray's (h, w)."""
    B, S, H, W = depth.shape
    if (H, W) == ray_hw:
        return depth
    d = depth.reshape(B * S, 1, H, W).float()
    d = F.interpolate(d, size=ray_hw, mode="bilinear", align_corners=False)
    return d.reshape(B, S, ray_hw[0], ray_hw[1])


def _world_points(depth: Tensor, ray: Tensor) -> Tensor:
    """P = t + D · d. depth: (B,S,h,w). ray: (B,S,h,w,6) [origin|direction].

    Returns world points (B, S, h, w, 3).
    """
    t = ray[..., :3]
    d = ray[..., 3:]
    return t + depth.unsqueeze(-1) * d


def _gt_world_points_from_camera(
    gt_depth: Tensor,        # (B, S, H, W)
    gt_K: Tensor,            # (B, S, 3, 3) at depth resolution
    gt_w2c: Tensor,          # (B, S, 4, 4) world-to-camera
    out_hw: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Back-project GT depth to world-frame 3D points at `out_hw`.

    Returns:
        P_world: (B, S, out_h, out_w, 3) world-frame 3D points.
        valid:   (B, S, out_h, out_w) bool mask of pixels with GT depth.
    """
    B, S, H, W = gt_depth.shape
    out_h, out_w = out_hw

    # Resize GT depth to ray resolution; rescale K accordingly.
    if (H, W) != out_hw:
        d = gt_depth.reshape(B * S, 1, H, W).float()
        d = F.interpolate(d, size=out_hw, mode="nearest")
        d = d.reshape(B, S, out_h, out_w)
    else:
        d = gt_depth.float()

    sx = out_w / W
    sy = out_h / H
    K = gt_K.clone().float()
    K[..., 0, 0] *= sx; K[..., 0, 2] *= sx
    K[..., 1, 1] *= sy; K[..., 1, 2] *= sy

    # Pixel grid (with 0.5 center convention).
    device = gt_depth.device
    v, u = torch.meshgrid(
        torch.arange(out_h, device=device, dtype=torch.float32),
        torch.arange(out_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u = u + 0.5
    v = v + 0.5
    fx = K[..., 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
    fy = K[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    x_cam = (u - cx) / fx * d
    y_cam = (v - cy) / fy * d
    z_cam = d
    P_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (B, S, h, w, 3)

    # Transform to world: P_world = R_c2w @ P_cam + t_c2w
    R_w2c = gt_w2c[..., :3, :3]
    t_w2c = gt_w2c[..., :3, 3]
    R_c2w = R_w2c.transpose(-1, -2)
    t_c2w = -(R_c2w @ t_w2c.unsqueeze(-1)).squeeze(-1)

    # Apply per-pixel: (B,S,3,3) @ (B,S,h,w,3,1)
    P_world = (R_c2w.unsqueeze(-3).unsqueeze(-3) @ P_cam.unsqueeze(-1)).squeeze(-1)
    P_world = P_world + t_c2w.unsqueeze(-2).unsqueeze(-2)

    valid = torch.isfinite(d) & (d > 0)
    return P_world, valid


def _extrinsic_l1(pred_se3: Tensor, target_se3: Tensor) -> Tensor:
    """ℓ1 on extrinsic 3x4 (rotation + translation flattened).

    pred_se3, target_se3: (B, S, 3, 4) — w2c.
    """
    return (pred_se3.float() - target_se3.float()).abs().mean()


def da3_paper_loss(
    student: dict,             # {depth, depth_conf, ray, ray_conf, extrinsics}
    target: dict,              # same keys; "extrinsics" required for L_C
    weights: DA3LossWeights,
    gt_depth: Optional[Tensor] = None,        # (B, S, H, W) for L_P with GT cam
    gt_intrinsics: Optional[Tensor] = None,   # (B, S, 3, 3)
    gt_w2c: Optional[Tensor] = None,          # (B, S, 4, 4)
    gt_valid: Optional[Tensor] = None,        # (B, S, H, W) mask
    gt_world_points: Optional[Tensor] = None, # (B, S, h, w, 3) at ray hw if pre-derived
) -> DA3LossOut:
    """Compute DA3 § 3.3 loss.

    Two modes:
    - Phase 1 (teacher target): pass `target` dict with teacher's depth/ray/
      extrinsics. P_target derived from teacher depth+ray.
    - Phase 2/3 (GT target): pass GT_depth + GT_K + GT_w2c. `target` still
      provides extrinsics (the cam_dec target = GT w2c) and may pass GT
      depth/ray as the depth/ray targets.
    """
    device = student["depth"].device
    z = lambda: torch.zeros((), device=device)
    s_depth = student["depth"]
    s_ray = student["ray"]
    s_dconf = student.get("depth_conf")
    s_rconf = student.get("ray_conf")
    s_ext = student.get("extrinsics")

    t_depth = target.get("depth")
    t_ray = target.get("ray")
    t_ext = target.get("extrinsics")

    valid_depth = gt_valid

    # L_D: aleatoric ℓ1 on depth.
    l_depth = z()
    if weights.lambda_depth > 0 and t_depth is not None and s_depth.shape == t_depth.shape:
        l_depth = _l1_aleatoric(
            s_depth, t_depth, s_dconf, valid_depth,
            weights.lambda_conf_log, weights.use_aleatoric,
            weights.use_kendall_gal,
        )

    # L_M: aleatoric ℓ1 on ray (6 channels).
    l_ray = z()
    if weights.lambda_ray > 0 and t_ray is not None and s_ray.shape == t_ray.shape:
        # Resize valid to ray resolution if needed.
        valid_ray = None
        if valid_depth is not None and valid_depth.shape[-2:] != s_ray.shape[-3:-1]:
            v = valid_depth.float().unsqueeze(2)  # (B,S,1,H,W)
            B, S = v.shape[0], v.shape[1]
            v = v.reshape(B * S, 1, *v.shape[-2:])
            v = F.interpolate(v, size=s_ray.shape[-3:-1], mode="nearest")
            valid_ray = v.reshape(B, S, *s_ray.shape[-3:-1]) > 0.5
        elif valid_depth is not None:
            valid_ray = valid_depth
        l_ray = _l1_aleatoric(
            s_ray, t_ray, s_rconf, valid_ray,
            weights.lambda_conf_log, weights.use_aleatoric,
            weights.use_kendall_gal,
        )

    # L_grad: ℓ1 on depth gradients.
    l_grad = z()
    if weights.lambda_grad > 0 and t_depth is not None and s_depth.shape == t_depth.shape:
        l_grad = _depth_grad_l1(s_depth, t_depth, valid_depth)

    # L_P: ℓ1 on world 3D points.
    l_point = z()
    if weights.lambda_point > 0:
        # Student point cloud: combine student depth (resized to ray hw) with
        # student ray. Always use student's d/t for the prediction side.
        ray_hw = s_ray.shape[-3:-1]
        s_depth_at_ray = _depth_to_ray_resolution(s_depth, ray_hw)
        P_pred = _world_points(s_depth_at_ray, s_ray)

        # Target points: prefer pre-derived, else build from teacher or GT.
        if gt_world_points is not None:
            P_target = gt_world_points
            v_pt = None
            if valid_depth is not None:
                v = valid_depth.float().unsqueeze(2)
                B, S = v.shape[0], v.shape[1]
                v = v.reshape(B * S, 1, *v.shape[-2:])
                v = F.interpolate(v, size=ray_hw, mode="nearest")
                v_pt = v.reshape(B, S, *ray_hw) > 0.5
        elif gt_depth is not None and gt_intrinsics is not None and gt_w2c is not None:
            P_target, v_pt = _gt_world_points_from_camera(gt_depth, gt_intrinsics, gt_w2c, ray_hw)
        elif t_depth is not None and t_ray is not None:
            t_depth_at_ray = _depth_to_ray_resolution(t_depth, ray_hw)
            P_target = _world_points(t_depth_at_ray, t_ray)
            v_pt = None
        else:
            P_target = None

        if P_target is not None and P_target.shape == P_pred.shape:
            err = (P_pred - P_target).abs()
            if v_pt is not None:
                v = v_pt.float().unsqueeze(-1)
                l_point = (err * v).sum() / (v.sum().clamp_min(1.0) * 3)
            else:
                l_point = err.mean()

    # L_C: ℓ1 on cam_dec extrinsics.
    l_cam = z()
    if weights.lambda_cam > 0 and s_ext is not None and t_ext is not None and s_ext.shape == t_ext.shape:
        l_cam = _extrinsic_l1(s_ext, t_ext)

    total = (
        weights.lambda_depth * l_depth
        + weights.lambda_ray * l_ray
        + weights.lambda_grad * l_grad
        + weights.lambda_point * l_point
        + weights.lambda_cam * l_cam
    )
    return DA3LossOut(total=total, l_depth=l_depth, l_ray=l_ray, l_grad=l_grad, l_point=l_point, l_cam=l_cam)
