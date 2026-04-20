"""Metrics for the SSM-3D vs. DA3 ETH3D comparison.

Two families:

1. Depth metrics — standard monocular depth evaluation computed on valid GT
   pixels. `align_scale_median` matches prediction scale to GT via the median
   ratio before scoring (DA3 / MiDaS convention).
2. Representation metrics — measure the quality of backbone features without
   GT. `feat_cos_mean` catches literal collapse; `effective_rank` catches
   sneakier low-rank collapse; `cross_view_nn_agreement` measures whether
   matching patches across views actually map to each other in feature space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Depth metrics
# -----------------------------------------------------------------------------


def align_scale_median(pred: Tensor, gt: Tensor, valid: Tensor) -> Tensor:
    """Scale `pred` so that median(pred_valid) == median(gt_valid).

    Scale-ambiguous predictions (e.g. raw DPT output) are routinely median-aligned
    before scoring — see MiDaS / DA3 eval conventions.
    """
    pv = pred[valid]
    gv = gt[valid]
    if pv.numel() == 0 or gv.numel() == 0:
        return pred
    scale = torch.median(gv) / torch.median(pv).clamp_min(1e-8)
    return pred * scale


def abs_rel(pred: Tensor, gt: Tensor, valid: Tensor) -> float:
    pv, gv = pred[valid], gt[valid]
    if pv.numel() == 0:
        return float("nan")
    return float(torch.mean(torch.abs(pv - gv) / gv.clamp_min(1e-8)))


def rmse(pred: Tensor, gt: Tensor, valid: Tensor) -> float:
    pv, gv = pred[valid], gt[valid]
    if pv.numel() == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean((pv - gv) ** 2)))


def log10_metric(pred: Tensor, gt: Tensor, valid: Tensor) -> float:
    pv, gv = pred[valid].clamp_min(1e-8), gt[valid].clamp_min(1e-8)
    if pv.numel() == 0:
        return float("nan")
    return float(torch.mean(torch.abs(torch.log10(pv) - torch.log10(gv))))


def delta_threshold(pred: Tensor, gt: Tensor, valid: Tensor, threshold: float = 1.25) -> float:
    pv, gv = pred[valid].clamp_min(1e-8), gt[valid].clamp_min(1e-8)
    if pv.numel() == 0:
        return float("nan")
    ratio = torch.maximum(pv / gv, gv / pv)
    return float((ratio < threshold).float().mean())


@dataclass
class DepthMetrics:
    abs_rel: float
    delta_1_25: float
    delta_1_25_sq: float
    rmse: float
    log10: float

    def as_dict(self) -> dict[str, float]:
        return {
            "abs_rel": self.abs_rel,
            "delta<1.25": self.delta_1_25,
            "delta<1.25^2": self.delta_1_25_sq,
            "rmse": self.rmse,
            "log10": self.log10,
        }


def depth_metrics(pred: Tensor, gt: Tensor, valid: Tensor, align: bool = True) -> DepthMetrics:
    """All standard depth metrics in one call. Optionally median-aligns first."""
    if align:
        pred = align_scale_median(pred, gt, valid)
    return DepthMetrics(
        abs_rel=abs_rel(pred, gt, valid),
        delta_1_25=delta_threshold(pred, gt, valid, 1.25),
        delta_1_25_sq=delta_threshold(pred, gt, valid, 1.25 ** 2),
        rmse=rmse(pred, gt, valid),
        log10=log10_metric(pred, gt, valid),
    )


# -----------------------------------------------------------------------------
# Representation metrics
# -----------------------------------------------------------------------------


def feat_cos_mean(feats: Tensor) -> float:
    """Mean off-diagonal cosine similarity between patch tokens.

    Args:
        feats: (N, C) patch features for ONE view/image.
    Returns:
        scalar in [-1, 1]. > 0.7 = collapse.
    """
    fn = torch.nn.functional.normalize(feats, dim=-1)
    cos = fn @ fn.transpose(0, 1)
    n = cos.shape[0]
    if n < 2:
        return float("nan")
    off = (cos.sum() - cos.diagonal().sum()) / (n * (n - 1))
    return float(off)


def effective_rank(feats: Tensor, eps: float = 1e-12) -> float:
    """exp(H(singular-value distribution)) — effective participating dimensions.

    For a (N, C) matrix this is `exp(-sum p_i log p_i)` where
    p_i = sigma_i / sum(sigma). Ranges in [1, min(N,C)]; higher = richer feature
    space. Detects low-rank collapse invisible to `feat_cos_mean`.
    """
    f = feats - feats.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(f.float())
    except RuntimeError:
        return float("nan")
    p = s / s.sum().clamp_min(eps)
    p = p[p > eps]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


def _depth_to_cam(
    depth_grid: Tensor,
    intrinsic: Tensor,
    image_hw: tuple[int, int],
) -> Tensor:
    """Back-project a grid-resolution depth map to camera-frame 3D points.

    Args:
        depth_grid: (H_grid, W_grid) metric depth at the feature-grid resolution.
        intrinsic: (3, 3) at image resolution (fx, fy, cx, cy in image pixels).
        image_hw: (H_img, W_img) of the original image the intrinsic refers to.

    Returns:
        (H_grid, W_grid, 3) xyz in camera frame. Invalid (depth<=0) → NaN.
    """
    H, W = depth_grid.shape
    img_h, img_w = image_hw
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    # Grid-cell center → image pixel center
    px_per_cell_x = img_w / W
    px_per_cell_y = img_h / H
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=depth_grid.dtype, device=depth_grid.device) + 0.5,
        torch.arange(W, dtype=depth_grid.dtype, device=depth_grid.device) + 0.5,
        indexing="ij",
    )
    u = xs * px_per_cell_x
    v = ys * px_per_cell_y
    z = depth_grid
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = torch.stack([x, y, z], dim=-1)
    xyz[~torch.isfinite(z) | (z <= 0)] = float("nan")
    return xyz


def _project(xyz_cam: Tensor, intrinsic: Tensor) -> Tensor:
    """Project camera-frame xyz to pixel (u, v). Returns (H, W, 2) float pixels."""
    x, y, z = xyz_cam[..., 0], xyz_cam[..., 1], xyz_cam[..., 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = fx * x / z.clamp_min(1e-8) + cx
    v = fy * y / z.clamp_min(1e-8) + cy
    return torch.stack([u, v], dim=-1)


def cross_view_nn_agreement(
    feats_a: Tensor,
    feats_b: Tensor,
    grid_hw: tuple[int, int],
    depth_a: Tensor,
    intrinsic_a: Tensor,
    intrinsic_b: Tensor,
    extrinsic_a_w2c: Tensor,
    extrinsic_b_w2c: Tensor,
    image_hw_b: tuple[int, int],
    radius_px: float = 32.0,
) -> float:
    """Fraction of view-A patches whose best feature-NN in view B lies near the
    GT-warped pixel.

    Pipeline:
      1. Back-project each A patch centre to camera-A xyz via `depth_a`.
      2. Transform to camera-B via `ext_a_inv @ ext_b` (both w2c), then project.
      3. Compute feature cosine sim A↔B (per-patch) and find argmax per query.
      4. Measure distance from argmax patch's image-pixel centre to GT-warped
         pixel; fraction within `radius_px` is the score.

    Args:
        feats_a, feats_b: (T, C) patch features per view.
        grid_hw: (H_grid, W_grid) patch grid for both views.
        depth_a: (H_img, W_img) GT depth in view A (metres).
        intrinsic_a, intrinsic_b: (3, 3) at image resolution.
        extrinsic_a_w2c, extrinsic_b_w2c: (4, 4) world-to-camera.
        image_hw_b: image resolution of view B (for bounds checking + pixel scale).
        radius_px: neighbourhood radius in B's pixel space.

    Returns:
        fraction in [0, 1]. Higher = features are 3D-consistent across views.
    """
    H, W = grid_hw
    img_h_b, img_w_b = image_hw_b
    img_h_a, img_w_a = depth_a.shape

    # 1. Back-project A to cam-A
    # Downsample depth to grid resolution (use nearest to preserve validity)
    depth_grid = torch.nn.functional.interpolate(
        depth_a.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
    ).squeeze(0).squeeze(0)
    xyz_a = _depth_to_cam(depth_grid, intrinsic_a, (img_h_a, img_w_a))  # (H, W, 3)

    # 2. A_cam -> world -> B_cam
    ext_a_c2w = torch.linalg.inv(extrinsic_a_w2c)
    ext_a2b = extrinsic_b_w2c @ ext_a_c2w  # (4, 4)
    ones = torch.ones_like(xyz_a[..., :1])
    xyz_a_h = torch.cat([xyz_a, ones], dim=-1)  # (H, W, 4)
    xyz_b = (ext_a2b @ xyz_a_h.reshape(-1, 4).transpose(0, 1)).transpose(0, 1).reshape(H, W, 4)[..., :3]
    uv_b = _project(xyz_b, intrinsic_b)  # (H, W, 2) in B pixel coords

    # 3. Feature NN
    fa = torch.nn.functional.normalize(feats_a, dim=-1)
    fb = torch.nn.functional.normalize(feats_b, dim=-1)
    sim = fa @ fb.transpose(0, 1)  # (T_a, T_b)
    nn_idx_b = sim.argmax(dim=-1)  # (T_a,)

    # Map patch idx -> pixel centre in B
    sx = img_w_b / W
    sy = img_h_b / H
    patch_rows = nn_idx_b // W
    patch_cols = nn_idx_b % W
    nn_px_b = torch.stack(
        [(patch_cols.float() + 0.5) * sx, (patch_rows.float() + 0.5) * sy], dim=-1
    )  # (T_a, 2)

    # 4. Compare with GT-warped
    uv_flat = uv_b.reshape(-1, 2)  # (T_a, 2)
    valid = (
        torch.isfinite(uv_flat).all(dim=-1)
        & (uv_flat[:, 0] >= 0) & (uv_flat[:, 0] < img_w_b)
        & (uv_flat[:, 1] >= 0) & (uv_flat[:, 1] < img_h_b)
        & (xyz_b[..., 2].reshape(-1) > 0)
    )
    if not valid.any():
        return float("nan")
    dist = torch.norm(nn_px_b[valid] - uv_flat[valid], dim=-1)
    return float((dist < radius_px).float().mean())
