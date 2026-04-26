"""Per-pixel ray angular error for SSM-3D ckpts vs DA3 teacher.

PLAN § 15.34 (eval-expansion task 1). DA3's DualDPT auxiliary head emits a
per-pixel 6-channel "camray": first 3 channels are the camera-frame ray
direction, last 3 are the camera origin (for downstream pose extraction).
This script evaluates the first-3 against GT camera rays derived from
ETH3D intrinsics.

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
from ssm3d.data.eth3d import download_eth3d_terrains, load_eth3d_scene
from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ssm3d.eval.dpt_adapter import SHARED_DPT_LAYERS, get_dualdpt, shared_dpt_outputs
from ssm3d.eval.eth3d_gt import load_eth3d_cams
from ssm3d.eval.metrics import gt_camera_rays, ray_angular_error
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_dinov2_backbone


def build_ssm(ckpt: Path, state_dim: int, img_size: int, patch_size: int,
              chunk_size: int, device: str):
    net = SSM3DNet(
        size="small", img_size=img_size, patch_size=patch_size, depth=12,
        chunk_size=chunk_size, mamba_state_dim=state_dim,
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
    args = ap.parse_args()

    scene_dir = download_eth3d_terrains(args.data_root, scene="terrains", download_depth=False)
    sample = load_eth3d_scene(
        scene_dir, max_images=args.max_images, image_size=args.img_size, load_gt_depth=False
    )
    images = sample.images.to(args.device)
    cams = load_eth3d_cams(scene_dir, image_size=args.img_size, image_names=[p.name for p in sample.image_paths])

    da3 = load_da3(DEFAULT_HF_MODEL, device=args.device)
    shared_dpt_base = get_dualdpt(da3)

    print(f"\n{'source':<48} {'mean_deg':>10} {'median':>10} {'auc_3°':>9} {'auc_30°':>9}")
    print("-" * 90)

    teacher_m = teacher_ray_metrics(da3, images, cams.intrinsics, [p.name for p in sample.image_paths], args.img_size)
    print(f"{'DA3-SMALL teacher (cat-duplicate bridge)':<48} {teacher_m['mean_deg']:>10.3f} "
          f"{teacher_m['median_deg']:>10.3f} {teacher_m['auc_3']:>9.4f} {teacher_m['auc_30']:>9.4f}")

    for ckpt_path in args.ckpts:
        ssm, bridge, tuned_dpt_state = build_ssm(
            ckpt_path, args.state_dim, args.img_size, args.patch_size,
            args.chunk_size, args.device,
        )
        if tuned_dpt_state is not None:
            shared_dpt = copy.deepcopy(shared_dpt_base)
            shared_dpt.load_state_dict(tuned_dpt_state)
            shared_dpt.to(args.device).eval()
        else:
            shared_dpt = shared_dpt_base
        out = shared_dpt_outputs(ssm, shared_dpt, images, bridge=bridge)
        m = per_image_ray_metrics(
            out["ray"][..., :3], out["ray_conf"],
            cams.intrinsics, [p.name for p in sample.image_paths], args.img_size,
        )
        print(f"{str(ckpt_path):<48} {m['mean_deg']:>10.3f} {m['median_deg']:>10.3f} "
              f"{m['auc_3']:>9.4f} {m['auc_30']:>9.4f}")


if __name__ == "__main__":
    main()
