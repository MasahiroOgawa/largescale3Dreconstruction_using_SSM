"""Ray + pose evaluation for SSM-3D ckpts vs DA3 teacher.

PLAN § 15.34 / § 15.35. Two modes:

- `--mode ray`: per-pixel angular error of the raw camray direction
  channels vs GT camera rays. Diagnostic only — absolute scale not
  comparable to DA3's published AUC because the raw channels feed
  RANSAC-based pose extraction.
- `--mode pose` (default): pipe predicted camrays through
  `get_extrinsic_from_camray` to obtain pred SE(3), then call DA3's
  own `compute_pose(pred, gt)` for AUC@3/5/15/30 — the metric DA3
  publishes on the official benchmark.

Example:
    uv run python scripts/eval_ray_metrics.py \\
        --ckpts outputs/runs/depth_ft_cm24/ckpt_1000.pt \\
                outputs/runs/depth_ft_cm30/ckpt_1000.pt \\
        --state-dim 64
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_dinov2 import ensure_dinov2_vits14  # noqa: E402

from ssm3d.bridge import DimBridgeStack
from ssm3d.data.bench import DATASETS, default_scene, load_bench_cams, load_bench_scene
from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ssm3d.eval.dpt_adapter import SHARED_DPT_LAYERS, get_dualdpt, shared_dpt_outputs
from ssm3d.eval.metrics import gt_camera_rays, ray_angular_error
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_dinov2_backbone


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


def per_image_ray_metrics(pred_ray_dir, conf, intrinsics_dict, image_names, img_size):
    """pred_ray_dir: (N, H_pred, W_pred, 3) on CPU.

    Predicted ray resolution may be smaller than img_size (DualDPT downsamples).
    GT rays are computed at the predicted resolution with scaled intrinsics.
    """
    H_pred, W_pred = pred_ray_dir.shape[1], pred_ray_dir.shape[2]
    sx, sy = W_pred / img_size, H_pred / img_size
    per_image = []
    for i, name in enumerate(image_names):
        K = torch.from_numpy(intrinsics_dict[name]).float().clone()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        gt = gt_camera_rays(K, (H_pred, W_pred))
        per_image.append(ray_angular_error(
            pred_ray_dir[i], gt, conf=conf[i] if conf is not None else None,
        ))
    keys = per_image[0].keys()
    return {k: sum(d[k] for d in per_image) / len(per_image) for k in keys}


@torch.inference_mode()
def teacher_ray_metrics(da3, images, intrinsics_dict, image_names, img_size):
    """Capture DA3's raw ray output (pre-RANSAC pose extraction) via forward hook."""
    dpt = get_dualdpt(da3)
    head_aux = getattr(dpt, "head_aux", "ray")
    captured: dict = {}

    def _hook(_m, _inp, output):
        captured["ray"] = output[head_aux].detach()
        captured["ray_conf"] = output[f"{head_aux}_conf"].detach()

    handle = dpt.register_forward_hook(_hook)
    try:
        _ = da3.model(images.unsqueeze(0))
    finally:
        handle.remove()
    ray = captured["ray"][0]                # (N, H, W, 6)
    conf = captured["ray_conf"][0]          # (N, H, W)
    return per_image_ray_metrics(
        ray[..., :3].cpu().float(), conf.cpu().float(),
        intrinsics_dict, image_names, img_size,
    )


@torch.inference_mode()
def teacher_camray(da3, images):
    """Capture DA3 teacher's raw 6-channel camray + conf via forward hook."""
    dpt = get_dualdpt(da3)
    head_aux = getattr(dpt, "head_aux", "ray")
    captured: dict = {}

    def _hook(_m, _inp, output):
        captured["ray"] = output[head_aux].detach()
        captured["ray_conf"] = output[f"{head_aux}_conf"].detach()

    handle = dpt.register_forward_hook(_hook)
    try:
        _ = da3.model(images.unsqueeze(0))
    finally:
        handle.remove()
    return captured["ray"], captured["ray_conf"]   # (B, S, h, w, 6) and (B, S, h, w)


def gt_se3_w2c(extrinsics_dict, image_names) -> torch.Tensor:
    import numpy as np
    arr = np.stack([extrinsics_dict[n] for n in image_names])  # (N, 4, 4)
    return torch.from_numpy(arr).float()


@torch.inference_mode()
def pose_auc(pred_camray, pred_conf, gt_se3) -> dict[str, float]:
    """Run DA3's pose-AUC pipeline. pred_camray: (B, S, h, w, 6); gt_se3: (S, 4, 4)."""
    from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray
    from depth_anything_3.bench.utils import compute_pose

    pred_se3, _f, _pp = get_extrinsic_from_camray(
        pred_camray, pred_conf, pred_camray.shape[-3], pred_camray.shape[-2],
    )
    pred_se3 = pred_se3[0].cpu().float()           # (S, 4, 4)
    metrics = compute_pose(pred_se3, gt_se3)
    return {k: float(metrics[k]) for k in ("auc03", "auc05", "auc15", "auc30")}


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
    ap.add_argument("--mode", choices=["ray", "pose", "both"], default="pose")
    ap.add_argument("--alt-start", type=int, default=-1,
                    help="DA3-style alternation start. -1 = legacy partial-swap; "
                         "4 = full-swap mirror of DA3-SMALL.")
    ap.add_argument("--cat-token", action="store_true",
                    help="Required with --alt-start ≥ 0 for full-swap.")
    ap.add_argument("--use-fused-kernel", action="store_true",
                    help="Route Mamba-3 self-attention through the upstream "
                         "Triton kernel (PLAN § 15.46).")
    ap.add_argument("--dataset", choices=DATASETS, default="eth3d")
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Stride for 7Scenes frame sampling.")
    args = ap.parse_args()

    scene = args.scene or default_scene(args.dataset, args.data_root)
    sample = load_bench_scene(
        args.dataset, scene, args.data_root, max_images=args.max_images,
        image_size=args.img_size, load_gt_depth=False, frame_stride=args.frame_stride,
    )
    images = sample.images.to(args.device)
    image_names = [p.name for p in sample.image_paths]
    cams = load_bench_cams(args.dataset, scene, args.data_root,
                           image_size=args.img_size, image_names=image_names)
    gt_se3 = gt_se3_w2c(cams.extrinsics, image_names)
    print(f"[eval_ray] dataset={args.dataset} scene={scene} N={len(image_names)}")

    da3 = load_da3(DEFAULT_HF_MODEL, device=args.device)
    shared_dpt_base = get_dualdpt(da3)

    if args.mode in ("ray", "both"):
        print(f"\n=== Per-pixel ray angular error (camera-frame, raw channels) ===")
        print(f"{'source':<48} {'mean_deg':>10} {'median':>10} {'auc_3°':>9} {'auc_30°':>9}")
        print("-" * 90)
        teacher_m = teacher_ray_metrics(da3, images, cams.intrinsics, image_names, args.img_size)
        print(f"{'DA3-SMALL teacher':<48} {teacher_m['mean_deg']:>10.3f} "
              f"{teacher_m['median_deg']:>10.3f} {teacher_m['auc_3']:>9.4f} {teacher_m['auc_30']:>9.4f}")

    if args.mode in ("pose", "both"):
        print(f"\n=== Pose AUC (DA3 official metric: rotation+translation joint AUC) ===")
        print(f"{'source':<48} {'AUC@3':>8} {'AUC@5':>8} {'AUC@15':>8} {'AUC@30':>8}")
        print("-" * 88)
        ray, conf = teacher_camray(da3, images)
        m = pose_auc(ray, conf, gt_se3)
        print(f"{'DA3-SMALL teacher':<48} {m['auc03']:>8.4f} {m['auc05']:>8.4f} "
              f"{m['auc15']:>8.4f} {m['auc30']:>8.4f}")

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

        if args.mode in ("ray", "both"):
            m = per_image_ray_metrics(
                out["ray"][..., :3], out["ray_conf"], cams.intrinsics, image_names, args.img_size,
            )
            print(f"{str(ckpt_path):<48} {m['mean_deg']:>10.3f} {m['median_deg']:>10.3f} "
                  f"{m['auc_3']:>9.4f} {m['auc_30']:>9.4f}")

        if args.mode in ("pose", "both"):
            ray = out["ray"].unsqueeze(0).to(args.device)        # (1, S, h, w, 6)
            conf = out["ray_conf"].unsqueeze(0).to(args.device)  # (1, S, h, w)
            m = pose_auc(ray, conf, gt_se3)
            print(f"{str(ckpt_path):<48} {m['auc03']:>8.4f} {m['auc05']:>8.4f} "
                  f"{m['auc15']:>8.4f} {m['auc30']:>8.4f}")


if __name__ == "__main__":
    main()
