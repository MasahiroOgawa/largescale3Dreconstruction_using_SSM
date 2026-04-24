"""CM23: sweep depth metrics over multiple Phase-C ckpts to locate the overfit peak.

Loads ETH3D `terrains` + DA3 once, then iterates ckpts, prints a table of
(step, |relative_depth_error|, delta<1.25, rmse, log10) for SSM-3D. No visuals,
no per-ckpt summary. Intended for a single ckpt-family sweep only.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_dinov2 import ensure_dinov2_vits14  # noqa: E402

from ssm3d.bridge import DimBridgeStack
from ssm3d.data.eth3d import download_eth3d_terrains, load_eth3d_scene
from ssm3d.eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ssm3d.eval.dpt_adapter import SHARED_DPT_LAYERS, get_dualdpt, shared_dpt_depth
from ssm3d.eval.metrics import depth_metrics
from ssm3d.model import SSM3DNet
from ssm3d.weights import load_dinov2_backbone


def _maybe_resize_pos_embed(state: dict, model: torch.nn.Module) -> None:
    key = "backbone.vit.pos_embed"
    if key not in state:
        return
    src = state[key]
    dst = model.state_dict().get(key)
    if dst is None or src.shape == dst.shape:
        return
    cls_src, patch_src = src[:, :1], src[:, 1:]
    n_src, n_dst = patch_src.shape[1], dst.shape[1] - 1
    g_src, g_dst = int(math.sqrt(n_src)), int(math.sqrt(n_dst))
    dim = patch_src.shape[-1]
    grid = patch_src.reshape(1, g_src, g_src, dim).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid.float(), size=(g_dst, g_dst), mode="bicubic", align_corners=False
    ).to(patch_src.dtype)
    patch_dst = grid.permute(0, 2, 3, 1).reshape(1, g_dst * g_dst, dim)
    state[key] = torch.cat([cls_src, patch_dst], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    torch.manual_seed(0)

    print("[1/3] data + DA3 ...")
    scene_dir = download_eth3d_terrains(args.data_root, scene="terrains", download_depth=True)
    sample = load_eth3d_scene(
        scene_dir, max_images=args.max_images, image_size=args.img_size, load_gt_depth=True
    )
    N = sample.images.shape[0]
    da3 = load_da3(DEFAULT_HF_MODEL, device=args.device)
    shared_dpt_base = get_dualdpt(da3)

    print("[2/3] building SSM-3D (reused per ckpt) ...")
    ssm = SSM3DNet(
        size="small", img_size=args.img_size, patch_size=args.patch_size,
        depth=12, chunk_size=args.chunk_size,
    )
    load_dinov2_backbone(ssm.backbone.vit, ensure_dinov2_vits14())
    ssm.to(args.device).eval()

    print(f"[3/3] sweeping {len(args.ckpts)} ckpts ...")
    print(f"\n{'ckpt':<32} {'|rel_err|':>10} {'d<1.25':>8} {'rmse':>8} {'log10':>8}")
    print("-" * 72)

    rows: list[tuple[str, float, float, float, float]] = []
    for ckpt_path in args.ckpts:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _maybe_resize_pos_embed(state["student"], ssm)
        ssm.load_state_dict(state["student"])
        bridge = None
        if state.get("bridge") is not None:
            bridge = DimBridgeStack(num_layers=len(SHARED_DPT_LAYERS), in_dim=384)
            bridge.load_state_dict(state["bridge"])
            bridge.to(args.device).eval()

        tuned_dpt_state = state.get("dualdpt")
        if tuned_dpt_state is not None:
            shared_dpt = copy.deepcopy(shared_dpt_base)
            shared_dpt.load_state_dict(tuned_dpt_state)
            shared_dpt.to(args.device).eval()
        else:
            shared_dpt = shared_dpt_base

        rel_err_values, d125s, rmses, log10s = [], [], [], []
        for i in range(N):
            rgb = sample.images[i]
            gt = sample.gt_depth[i]
            valid = sample.valid_mask[i]
            with torch.inference_mode():
                dep = shared_dpt_depth(
                    ssm, shared_dpt, rgb.unsqueeze(0).to(args.device), bridge=bridge,
                )[0]
            dm = depth_metrics(dep.detach().cpu(), gt, valid, align=True).as_dict()
            rel_err_values.append(dm["|relative_depth_error|"])
            d125s.append(dm["delta<1.25"])
            rmses.append(dm["rmse"])
            log10s.append(dm["log10"])

        rel_err = sum(rel_err_values) / N
        d125 = sum(d125s) / N
        rmse = sum(rmses) / N
        log10 = sum(log10s) / N
        name = ckpt_path.name
        print(f"{name:<32} {rel_err:>10.4f} {d125:>8.4f} {rmse:>8.4f} {log10:>8.4f}")
        rows.append((name, rel_err, d125, rmse, log10))

    print("\nbest by |relative_depth_error|:")
    best = min(rows, key=lambda r: r[1])
    print(f"  {best[0]}: |relative_depth_error|={best[1]:.4f}, d<1.25={best[2]:.4f}")


if __name__ == "__main__":
    main()
