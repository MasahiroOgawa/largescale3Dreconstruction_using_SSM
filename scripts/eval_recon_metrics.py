"""3D reconstruction F-score / Chamfer for SSM-3D ckpts vs DA3 teacher.

PLAN § 15.36 (eval-expansion task 2). Reconstruction-posed mode: use GT
camera intrinsics + extrinsics + predicted depth to back-project
per-view 3D points into the world frame. Concatenate across views to
form a predicted point cloud, compare to a GT point cloud (built the
same way from GT depth) via DA3's `evaluate_3d_reconstruction`.

Note: reconstruction-unposed mode (using predicted poses) is *also*
implementable, but § 15.35 showed that pose AUC@30° ≈ 0.04 for our
checkpoints — predicted-pose reconstruction will be uninformative
until the ray problem is fixed. Posed mode isolates the depth-quality
contribution to 3D consistency.

Example:
    uv run python scripts/eval_recon_metrics.py \\
        --ckpts outputs/runs/depth_ft_cm24/ckpt_1000.pt \\
                outputs/runs/depth_ft_cm30/ckpt_1000.pt \\
        --threshold 0.05
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_dinov2 import ensure_dinov2_vits14  # noqa: E402

from ssm3d.bridge import DimBridgeStack
from ssm3d.data.eth3d import download_eth3d_terrains, load_eth3d_scene
from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, da3_depth, load_da3
from ssm3d.eval.dpt_adapter import SHARED_DPT_LAYERS, get_dualdpt, shared_dpt_outputs
from ssm3d.eval.eth3d_gt import load_eth3d_cams
from ssm3d.eval.metrics import align_scale_median
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_dinov2_backbone


def backproject_depth_to_world(
    depth: torch.Tensor,           # (H, W)
    K: np.ndarray,                 # (3, 3) image-resolution intrinsic
    extrinsic_w2c: np.ndarray,     # (4, 4) world-to-camera
    image_hw: tuple[int, int],     # (H_img, W_img)
) -> np.ndarray:
    """Back-project a depth map into world-frame xyz points."""
    H_d, W_d = depth.shape
    H_img, W_img = image_hw
    sx, sy = W_d / W_img, H_d / H_img
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = K[0, 2] * sx, K[1, 2] * sy
    v, u = torch.meshgrid(
        torch.arange(H_d, dtype=torch.float32),
        torch.arange(W_d, dtype=torch.float32),
        indexing="ij",
    )
    z = depth.float()
    x_cam = (u + 0.5 - cx) / fx * z
    y_cam = (v + 0.5 - cy) / fy * z
    xyz_cam = torch.stack([x_cam, y_cam, z], dim=-1).reshape(-1, 3).numpy()    # (N, 3)
    valid = (xyz_cam[:, 2] > 1e-6) & np.isfinite(xyz_cam).all(axis=1)
    xyz_cam = xyz_cam[valid]
    R_w2c = extrinsic_w2c[:3, :3]
    t_w2c = extrinsic_w2c[:3, 3]
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    xyz_world = (R_c2w @ xyz_cam.T).T + t_c2w
    return xyz_world


def view_pcd(
    depth_per_view: list[torch.Tensor],
    K_per_view: list[np.ndarray],
    extr_per_view: list[np.ndarray],
    image_hw: tuple[int, int],
    voxel_down: float | None = 0.02,
) -> np.ndarray:
    """Concatenate world points from all views into one cloud, optionally down-sampled.

    Simpler than TSDF fusion; usable when RGB images aren't needed and per-view
    coverage is comparable.
    """
    pts = [
        backproject_depth_to_world(d, K, E, image_hw)
        for d, K, E in zip(depth_per_view, K_per_view, extr_per_view)
    ]
    cloud = np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
    if voxel_down is not None and voxel_down > 0:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd = pcd.voxel_down_sample(voxel_down)
        cloud = np.asarray(pcd.points)
    return cloud


def tsdf_fused_pcd(
    depth_per_view: list[torch.Tensor],
    rgb_per_view: list[torch.Tensor],   # (3, H_img, W_img) in [0, 1]
    K_per_view: list[np.ndarray],
    extr_per_view: list[np.ndarray],
    image_hw: tuple[int, int],
    max_depth: float = 30.0,
    voxel_length: float = 4.0 / 512.0,
    sdf_trunc: float = 0.04,
    sample_points: int = 1_000_000,
) -> np.ndarray:
    """DA3 official protocol: TSDF-fuse depth + RGB across views into a mesh,
    then sample points uniformly. Closer to DA3-BENCH's recon scoring."""
    from depth_anything_3.bench.utils import (
        create_tsdf_volume, fuse_depth_to_tsdf, sample_points_from_mesh,
    )
    H_img, W_img = image_hw
    depths_np = np.stack([
        F.interpolate(
            d.unsqueeze(0).unsqueeze(0).float(), size=(H_img, W_img),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()
        for d in depth_per_view
    ]).astype(np.float32)
    rgb_np = np.stack([
        np.ascontiguousarray(
            (r.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        )
        for r in rgb_per_view
    ])
    # DA3's fuse_depth_to_tsdf wants per-view K already at depth resolution.
    Ks = np.stack(K_per_view).astype(np.float32)
    Es = np.stack(extr_per_view).astype(np.float32)

    volume = create_tsdf_volume(voxel_length=voxel_length, sdf_trunc=sdf_trunc)
    mesh = fuse_depth_to_tsdf(volume, depths_np, rgb_np, Ks, Es, max_depth=max_depth)
    pcd = sample_points_from_mesh(mesh, num_points=sample_points)
    return np.asarray(pcd.points)


def build_ssm(ckpt: Path, state_dim: int, img_size: int, patch_size: int,
              chunk_size: int, device: str, alt_start: int = -1,
              cat_token: bool = False, use_fused_kernel: bool = False):
    net = SSM3DNet(
        size="small", img_size=img_size, patch_size=patch_size, depth=12,
        chunk_size=chunk_size, mamba_state_dim=state_dim,
        alt_start=alt_start, cat_token=cat_token,
        use_fused_kernel=use_fused_kernel,
    )
    load_dinov2_backbone(net.backbone.vit, ensure_dinov2_vits14())
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(state["student"])
    bridge = None
    if state.get("bridge") is not None:
        bridge = DimBridgeStack(num_layers=len(SHARED_DPT_LAYERS), in_dim=384)
        bridge.load_state_dict(state["bridge"])
        bridge.to(device).eval()
    return net.to(device).eval(), bridge, state.get("dualdpt")


def aligned_depth(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Median-align predicted depth to GT (same as our depth eval)."""
    valid = torch.isfinite(gt) & (gt > 0) & torch.isfinite(pred) & (pred > 0)
    if not valid.any():
        return pred
    return align_scale_median(pred, gt, valid)


def upsample_depth(depth: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly upsample (H, W) depth to target_hw."""
    if depth.shape == target_hw:
        return depth
    return F.interpolate(
        depth.unsqueeze(0).unsqueeze(0).float(), size=target_hw,
        mode="bilinear", align_corners=False,
    ).squeeze()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=Path, nargs="*", default=[])
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="F-score distance threshold (meters); ETH3D outdoor scenes "
                         "are scale-ambiguous so this is meaningful only after median "
                         "alignment.")
    ap.add_argument("--voxel-down", type=float, default=0.02,
                    help="Voxel downsample size for back-project mode; 0 to disable.")
    ap.add_argument("--alt-start", type=int, default=-1,
                    help="DA3-style alternation start. -1 = legacy partial-swap; "
                         "4 = full-swap mirror of DA3-SMALL.")
    ap.add_argument("--cat-token", action="store_true",
                    help="Required with --alt-start ≥ 0 for full-swap.")
    ap.add_argument("--use-fused-kernel", action="store_true",
                    help="Route Mamba-3 self-attention through the upstream "
                         "Triton kernel (PLAN § 15.46).")
    ap.add_argument("--mode", choices=["backproject", "tsdf"], default="tsdf",
                    help="`tsdf` matches DA3's official recon protocol "
                         "(volume fusion + mesh sample); `backproject` is simpler.")
    ap.add_argument("--max-depth", type=float, default=30.0,
                    help="TSDF max-depth truncation in meters.")
    args = ap.parse_args()

    from depth_anything_3.bench.utils import evaluate_3d_reconstruction

    scene_dir = download_eth3d_terrains(args.data_root, scene="terrains", download_depth=True)
    sample = load_eth3d_scene(
        scene_dir, max_images=args.max_images, image_size=args.img_size, load_gt_depth=True
    )
    images = sample.images.to(args.device)
    image_names = [p.name for p in sample.image_paths]
    cams = load_eth3d_cams(scene_dir, image_size=args.img_size, image_names=image_names)
    Ks = [cams.intrinsics[n] for n in image_names]
    Es = [cams.extrinsics[n] for n in image_names]
    image_hw = (args.img_size, args.img_size)

    gt_depths = [sample.gt_depth[i] for i in range(len(image_names))]
    rgb_per_view = [sample.images[i] for i in range(len(image_names))]
    if args.mode == "tsdf":
        gt_cloud = tsdf_fused_pcd(
            gt_depths, rgb_per_view, Ks, Es, image_hw, max_depth=args.max_depth,
        )
    else:
        gt_cloud = view_pcd(gt_depths, Ks, Es, image_hw, voxel_down=args.voxel_down)
    print(f"[GT cloud] {len(gt_cloud):,} points (mode={args.mode})")

    da3 = load_da3(DEFAULT_HF_MODEL, device=args.device)
    shared_dpt_base = get_dualdpt(da3)

    print(f"\n{'source':<48} {'F@5cm':>8} {'prec':>7} {'recall':>7} {'acc(m)':>8} {'comp(m)':>9}")
    print("-" * 92)

    teacher_d = da3_depth(da3, images, process_res=args.img_size).cpu()
    teacher_d_aligned = []
    for i in range(len(image_names)):
        d = upsample_depth(teacher_d[i], image_hw)
        teacher_d_aligned.append(aligned_depth(d, gt_depths[i]))
    if args.mode == "tsdf":
        teacher_cloud = tsdf_fused_pcd(
            teacher_d_aligned, rgb_per_view, Ks, Es, image_hw, max_depth=args.max_depth,
        )
    else:
        teacher_cloud = view_pcd(teacher_d_aligned, Ks, Es, image_hw, voxel_down=args.voxel_down)
    m = evaluate_3d_reconstruction(teacher_cloud, gt_cloud, threshold=args.threshold)
    print(f"{'DA3-SMALL teacher':<48} {m['fscore']:>8.4f} {m['precision']:>7.4f} "
          f"{m['recall']:>7.4f} {m['acc']:>8.4f} {m['comp']:>9.4f}")

    for ckpt_path in args.ckpts:
        ssm, bridge, tuned_dpt_state = build_ssm(
            ckpt_path, args.state_dim, args.img_size, args.patch_size,
            args.chunk_size, args.device,
            alt_start=args.alt_start, cat_token=args.cat_token,
            use_fused_kernel=args.use_fused_kernel,
        )
        if tuned_dpt_state is not None:
            shared_dpt = copy.deepcopy(shared_dpt_base)
            shared_dpt.load_state_dict(tuned_dpt_state)
            shared_dpt.to(args.device).eval()
        else:
            shared_dpt = shared_dpt_base
        out = shared_dpt_outputs(ssm, shared_dpt, images, bridge=bridge)
        d_per = []
        for i in range(len(image_names)):
            d = upsample_depth(out["depth"][i], image_hw)
            d_per.append(aligned_depth(d, gt_depths[i]))
        if args.mode == "tsdf":
            cloud = tsdf_fused_pcd(
                d_per, rgb_per_view, Ks, Es, image_hw, max_depth=args.max_depth,
            )
        else:
            cloud = view_pcd(d_per, Ks, Es, image_hw, voxel_down=args.voxel_down)
        m = evaluate_3d_reconstruction(cloud, gt_cloud, threshold=args.threshold)
        print(f"{str(ckpt_path):<48} {m['fscore']:>8.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['acc']:>8.4f} {m['comp']:>9.4f}")


if __name__ == "__main__":
    main()
