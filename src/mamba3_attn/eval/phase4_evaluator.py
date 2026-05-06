"""Phase 4 accuracy eval — patched DA3 ckpt vs un-patched DA3 reference.

Uses DA3's full inference pipeline (`api.inference()` with the
`saddle_balanced` reference view strategy and Umeyama scale alignment
where applicable) on our existing ETH3D / HiRoom / 7Scenes loaders, then
calls DA3's own metric functions (`compute_pose`, `evaluate_3d_reconstruction`,
TSDF utilities) — the same metric definitions DA3 paper reports.

Eval split:
- ETH3D `terrains`
- HiRoom: last 4 of `selected_scene_list_val.txt`
- 7Scenes: `pumpkin`, `redkitchen`, `stairs`

Metrics per scene:
- pose AUC@30°/15°/5°/3° (rotation+translation joint)
- F-score@5cm + precision + recall + Chamfer in `recon_posed` mode
  (GT cams + pred depth fused via TSDF)
- F-score@5cm + precision + recall + Chamfer in `recon_unposed` mode
  (pred cams Umeyama-aligned to GT, then pred depth fused)

Usage:
    uv run python -m mamba3_attn.eval.phase4_evaluator \\
        --ckpt outputs/runs/phase3_unfreeze/ckpt_500.pt
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch

# Dump a Python traceback on SIGSEGV/SIGABRT/etc. so CUDA-level crashes (which
# bypass Python's try/except) leave a diagnosable stack instead of a silent rc=1.
faulthandler.enable(file=sys.stderr, all_threads=True)

from ..data.bench import load_bench_cams, load_bench_scene
from ..data.hiroom import list_hiroom_scenes
from ..data.view_split import read_split
from ..eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ..patch import install_mamba3


EVAL_SPLIT_ETH3D = ("terrains",)
EVAL_SPLIT_7SCENES = ("pumpkin", "redkitchen", "stairs")


@dataclass
class SceneMetrics:
    dataset: str
    scene: str
    auc30: float = float("nan")
    auc15: float = float("nan")
    auc05: float = float("nan")
    auc03: float = float("nan")
    fscore_posed: float = float("nan")
    prec_posed: float = float("nan")
    recall_posed: float = float("nan")
    fscore_unposed: float = float("nan")
    prec_unposed: float = float("nan")
    recall_unposed: float = float("nan")


def build_patched_api(ckpt_path: str | None, device: str = "cuda", state_dim: int = 64,
                      patched: bool = True):
    """Load DA3-SMALL → optionally install_mamba3 → optionally load ckpt → return api.

    Combinations supported:
      - `patched=True,  ckpt_path=...`  : Mamba-3-patched DA3 with trained weights.
      - `patched=True,  ckpt_path=None` : Mamba-3-patched DA3 at warm-start (no training).
      - `patched=False, ckpt_path=...`  : un-patched DA3-SMALL with overfit weights.
      - `patched=False, ckpt_path=None` : un-patched DA3-SMALL zero-shot reference.
    """
    api = load_da3(DEFAULT_HF_MODEL, device=device)
    if patched:
        install_mamba3(api.model, which="all", state_dim=state_dim,
                       use_fused_kernel=True, chunk_size=128)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        api.model.load_state_dict(state["model"])
    api = api.to(device)
    api.model.eval()
    return api


def _eval_split_hiroom(data_root: Path) -> tuple[str, ...]:
    return tuple(list_hiroom_scenes(data_root)[-4:])


def _scene_iter(data_root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    out += [("eth3d", s) for s in EVAL_SPLIT_ETH3D]
    try:
        out += [("hiroom", s) for s in _eval_split_hiroom(data_root)]
    except FileNotFoundError:
        pass
    out += [("7scenes", s) for s in EVAL_SPLIT_7SCENES]
    return out


def _to_se3(extr_3x4: torch.Tensor) -> torch.Tensor:
    """Pad (..., 3, 4) → (..., 4, 4) by appending [0,0,0,1]."""
    pad = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=extr_3x4.dtype, device=extr_3x4.device)
    pad = pad.expand(*extr_3x4.shape[:-2], 1, 4)
    return torch.cat([extr_3x4, pad], dim=-2)


def _eval_pose(pred_extr: torch.Tensor, gt_w2c: torch.Tensor) -> dict:
    from depth_anything_3.bench.utils import compute_pose
    pred = pred_extr if pred_extr.shape[-2] == 4 else _to_se3(pred_extr)
    return compute_pose(pred.cpu().float(), gt_w2c.cpu().float())


def _resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize (S, H, W) depth to (S, target_h, target_w) via bilinear."""
    import cv2
    H, W = target_hw
    if depth.shape[-2:] == target_hw:
        return depth.astype(np.float32)
    out = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in depth])
    return out.astype(np.float32)


def _depth_to_world_points(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Back-project (H, W) depth + (3,3) K + (4,4) w2c → (N, 3) world points."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    u = u + 0.5
    v = v + 0.5
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth.astype(np.float32)
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    P_cam = np.stack([x_cam, y_cam, z], axis=-1).reshape(-1, 3)
    valid = (z > 0).reshape(-1) & np.isfinite(P_cam).all(axis=1)
    P_cam = P_cam[valid]
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    return (R_c2w @ P_cam.T).T + t_c2w


def _sanitize_depth(depths: np.ndarray, max_depth: float) -> np.ndarray:
    """Replace non-finite with 0 and clamp to [0, max_depth].

    Open3D's `ScalableTSDFVolume` block-allocates around the deepest finite
    pixel; a single NaN/inf or outlier from a degenerate ckpt explodes the
    volume bounds and OOMs the host (PLAN §15.59.1). This guard keeps eval
    runnable regardless of model output quality — bad depths just produce
    empty/sparse fused clouds, which is the *correct* signal anyway.
    """
    out = np.where(np.isfinite(depths), depths, 0.0).astype(np.float32, copy=False)
    return np.clip(out, 0.0, max_depth)


def _release_to_os() -> None:
    """Force glibc to return freed memory to the OS.

    Python's `del` and `gc.collect()` mark Open3D's TSDF blocks as freed, but
    glibc's allocator keeps them in its arena pool — so the next call in the
    same process sees inflated RSS even though the memory is logically free.
    `malloc_trim(0)` flushes the pool. No-op on non-glibc platforms.
    """
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _estimate_world_surface_m2(depths: np.ndarray, Ks: np.ndarray) -> float:
    """Cheap upper bound on the world-space surface integrated by TSDF.

    Per-pixel world area at depth z is z²/(fx·fy); summed across valid pixels
    in all views gives a back-of-envelope surface area. Healthy ETH3D outdoor
    scenes land around 600–2000 m²; a confidence-collapsed ckpt that saturates
    near max_depth crosses 8000+ m². TSDF block memory is roughly linear in
    this number, so it's an effective early-out before Open3D allocates.
    """
    total = 0.0
    for d, K in zip(depths, Ks):
        valid = (d > 0) & np.isfinite(d)
        if not valid.any():
            continue
        z = d[valid].astype(np.float64, copy=False)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        if fx <= 0 or fy <= 0:
            continue
        total += float(np.sum(z * z) / (fx * fy))
    return total


def _tsdf_fuse(depths: np.ndarray, rgbs: np.ndarray, Ks: np.ndarray, Es: np.ndarray,
               max_depth: float = 30.0) -> np.ndarray:
    """TSDF-fuse + sample 1M points. depths (S,H,W), rgbs (S,H,W,3), Ks (S,3,3), Es (S,4,4).

    Wrapped in two host-RAM guards (PLAN §15.59.1):
      1. World-space surface-area pre-flight: degenerate ckpts that saturate
         depth near max_depth produce wildly large estimated surfaces; abort
         before Open3D allocates a single block.
      2. Per-view RSS check inside the integrate loop: trips even when the
         pre-flight underestimates (e.g., bimodal degenerate depth).

    Both raise MemoryError so the existing `_eval_recon` try/except logs a
    clean failure and lets the eval continue — instead of letting the kernel
    SIGKILL the process and take tmux/claude down with it.
    """
    import open3d as o3d
    from depth_anything_3.bench.utils import create_tsdf_volume, sample_points_from_mesh

    depths = _sanitize_depth(depths, max_depth)
    Ks_f = Ks.astype(np.float32)
    Es_f = Es.astype(np.float32)
    rgbs_u8 = rgbs.astype(np.uint8)

    surf_m2 = _estimate_world_surface_m2(depths, Ks_f)
    surf_budget_m2 = float(os.environ.get("PHASE4_SURFACE_M2", "8000"))
    if surf_m2 > surf_budget_m2:
        raise MemoryError(
            f"TSDF pre-flight: estimated world surface {surf_m2:.0f} m² > "
            f"{surf_budget_m2:.0f} m² budget; depth likely degenerate, skipping recon."
        )

    rss_budget_gb = float(os.environ.get("PHASE4_RSS_LIMIT_GB", "16"))
    proc = psutil.Process()

    # Adaptive voxel size: paper-default 4/512m, halved per retry so a
    # catastrophically dispersed depth field (e.g. saturating max_depth)
    # still produces a measurable F-score at coarser resolution rather than
    # NaN. F-score numbers across retries are slightly less precise but
    # comparable — beats no number at all.
    voxel_length = 4.0 / 512.0
    max_retries = int(os.environ.get("PHASE4_TSDF_MAX_RETRIES", "2"))
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        rss_start_gb = proc.memory_info().rss / (1024 ** 3)
        volume = create_tsdf_volume(voxel_length=voxel_length, sdf_trunc=0.04)
        try:
            for i in range(len(depths)):
                d = depths[i]
                h, w = d.shape[:2]
                depth_o3d = o3d.geometry.Image(d)
                color_o3d = o3d.geometry.Image(rgbs_u8[i])
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d,
                    depth_trunc=max_depth,
                    convert_rgb_to_intensity=False,
                    depth_scale=1.0,
                )
                K = Ks_f[i]
                ixt_o3d = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
                volume.integrate(rgbd, ixt_o3d, Es_f[i])
                rss_gb = proc.memory_info().rss / (1024 ** 3)
                if rss_gb > rss_budget_gb:
                    raise MemoryError(
                        f"TSDF integrate (voxel={voxel_length:.4f}m): RSS {rss_gb:.1f} GB "
                        f"> {rss_budget_gb:.1f} GB budget after view {i + 1}/{len(depths)} "
                        f"(started at {rss_start_gb:.1f} GB)."
                    )
            mesh = volume.extract_triangle_mesh()
            pcd = sample_points_from_mesh(mesh, num_points=1_000_000)
            # Copy out, then release the TSDF blocks back to the OS so the
            # next `_tsdf_fuse` call (e.g., gt_cloud right after pred_cloud
            # in `_eval_recon`) starts at clean RSS — without this, glibc
            # holds the freed blocks and the next call retries at coarser
            # voxel for no good reason. Copy is ~24 MB, negligible.
            points = np.array(pcd.points, copy=True)
            del volume, mesh, pcd
            _release_to_os()
            if attempt > 0:
                print(f"  [tsdf_fuse] succeeded at voxel_length={voxel_length:.4f}m "
                      f"(attempt {attempt + 1}/{max_retries + 1})", flush=True)
            return points
        except MemoryError as e:
            last_error = e
            del volume
            _release_to_os()
            if attempt < max_retries:
                voxel_length *= 2
                print(f"  [tsdf_fuse] retrying at coarser voxel_length={voxel_length:.4f}m: {e}",
                      flush=True)
            else:
                break
    assert last_error is not None
    raise MemoryError(
        f"TSDF fuse exhausted {max_retries + 1} retries up to voxel_length={voxel_length:.4f}m; "
        f"last error: {last_error}"
    )


def _eval_recon(depths: np.ndarray, rgbs: np.ndarray, Ks: np.ndarray, Es_pred: np.ndarray,
                 gt_depths: np.ndarray, gt_Ks: np.ndarray, gt_Es: np.ndarray,
                 threshold: float = 0.05, max_depth: float = 30.0) -> dict:
    from depth_anything_3.bench.utils import evaluate_3d_reconstruction
    pred_cloud = _tsdf_fuse(depths, rgbs, Ks, Es_pred, max_depth=max_depth)
    gt_cloud = _tsdf_fuse(gt_depths, rgbs, gt_Ks, gt_Es, max_depth=max_depth)
    return evaluate_3d_reconstruction(pred_cloud, gt_cloud, threshold=threshold)


@torch.inference_mode()
def evaluate_one_scene(api, dataset: str, scene: str, data_root: Path,
                       max_images: int = 12, image_size: int = 504,
                       frame_stride: int = 80, recon: bool = True,
                       view_indices: list[int] | None = None) -> SceneMetrics:
    """Evaluate one scene. If `view_indices` is given, restrict to those positional
    indices into the loaded sample (held-out test views from split.json). Loads the
    full candidate set (≥ max(view_indices) + 1 views) before slicing so the
    indices are stable wrt the train-time split.
    """
    load_max = max_images if view_indices is None else max(max_images, max(view_indices) + 1)
    print(f"  [stage] load_bench_scene (load_max={load_max}, image_size={image_size})...", flush=True)
    sample = load_bench_scene(dataset, scene, data_root,
                               max_images=load_max, image_size=image_size,
                               load_gt_depth=True,
                               frame_stride=frame_stride if dataset == "7scenes" else 1)
    print(f"  [stage] loaded {sample.images.shape[0]} views, images={tuple(sample.images.shape)}", flush=True)
    if view_indices is not None:
        if max(view_indices) >= sample.images.shape[0]:
            raise ValueError(
                f"view_indices max={max(view_indices)} but only {sample.images.shape[0]} views loaded"
            )
        idx = list(view_indices)
        sample.images = sample.images.index_select(0, torch.tensor(idx, dtype=torch.long))
        if sample.gt_depth is not None:
            sample.gt_depth = sample.gt_depth.index_select(0, torch.tensor(idx, dtype=torch.long))
        if sample.valid_mask is not None:
            sample.valid_mask = sample.valid_mask.index_select(0, torch.tensor(idx, dtype=torch.long))
        sample.image_paths = [sample.image_paths[i] for i in idx]
    if sample.images.shape[0] < 2:
        return SceneMetrics(dataset=dataset, scene=scene)
    image_paths = [str(p) for p in sample.image_paths]
    names = [p.name for p in sample.image_paths]
    print(f"  [stage] load_bench_cams (n={len(names)})...", flush=True)
    cams = load_bench_cams(dataset, scene, data_root, image_size=image_size, image_names=names)
    gt_w2c = torch.from_numpy(np.stack([cams.extrinsics[n] for n in names])).float()
    gt_K = np.stack([cams.intrinsics[n] for n in names])

    sm = SceneMetrics(dataset=dataset, scene=scene)

    print(f"  [stage] api.inference unposed (S={len(image_paths)}, res={image_size})...", flush=True)
    pred_unposed = api.inference(
        image_paths,
        process_res=image_size,
        ref_view_strategy="saddle_balanced",
        export_dir=None,
    )
    print(f"  [stage] inference unposed done", flush=True)
    pred_extr_unposed = torch.from_numpy(np.asarray(pred_unposed.extrinsics)).float()
    pose_m = _eval_pose(pred_extr_unposed, gt_w2c)
    sm.auc30 = float(pose_m.auc30)
    sm.auc15 = float(pose_m.auc15)
    sm.auc05 = float(pose_m.auc05)
    sm.auc03 = float(pose_m.auc03)

    if not recon:
        return sm

    print(f"  [stage] api.inference posed...", flush=True)
    pred_posed = api.inference(
        image_paths,
        extrinsics=gt_w2c.numpy(),
        intrinsics=gt_K,
        process_res=image_size,
        ref_view_strategy="saddle_balanced",
        export_dir=None,
    )
    print(f"  [stage] inference posed done", flush=True)

    # Build per-view depth + RGB at original (image_size) resolution for TSDF.
    pred_depth_posed = np.asarray(pred_posed.depth).astype(np.float32)
    if pred_depth_posed.ndim == 4:
        pred_depth_posed = pred_depth_posed.squeeze(0)  # (S, H_p, W_p)
    rgbs = np.ascontiguousarray(
        (sample.images.permute(0, 2, 3, 1).contiguous().numpy() * 255).astype(np.uint8)
    )
    gt_depths = np.ascontiguousarray(sample.gt_depth.numpy().astype(np.float32))
    H_gt, W_gt = gt_depths.shape[-2:]
    pred_depth_posed = _resize_depth(pred_depth_posed, (H_gt, W_gt))
    max_depth = 30.0 if dataset == "eth3d" else 10.0

    pred_unposed_extr = np.asarray(pred_unposed.extrinsics).copy()
    pred_unposed_depth = np.asarray(pred_unposed.depth).astype(np.float32)
    pred_unposed_K = np.asarray(pred_unposed.intrinsics).astype(np.float32)
    del pred_unposed, pred_posed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  [stage] recon posed (TSDF fuse + eval)...", flush=True)
    try:
        m_posed = _eval_recon(
            pred_depth_posed, rgbs, gt_K, gt_w2c.numpy(),
            gt_depths, gt_K, gt_w2c.numpy(),
            max_depth=max_depth,
        )
        sm.fscore_posed = float(m_posed["fscore"])
        sm.prec_posed = float(m_posed["precision"])
        sm.recall_posed = float(m_posed["recall"])
    except Exception as e:
        traceback.print_exc()
        print(f"  [recon_posed FAILED] {dataset}/{scene}: {e}", flush=True)
    del pred_depth_posed
    _release_to_os()

    print(f"  [stage] recon unposed (Umeyama align + TSDF)...", flush=True)
    try:
        from depth_anything_3.utils.pose_align import align_poses_umeyama
        _, _, scale, aligned = align_poses_umeyama(
            gt_w2c.numpy().copy(),
            pred_unposed_extr,
            return_aligned=True, ransac=True, random_state=42,
        )
        pred_depth_unposed = pred_unposed_depth
        if pred_depth_unposed.ndim == 4:
            pred_depth_unposed = pred_depth_unposed.squeeze(0)
        pred_depth_unposed = _resize_depth(pred_depth_unposed, (H_gt, W_gt)) * scale
        pred_depth_unposed = np.ascontiguousarray(pred_depth_unposed)

        pred_K_unposed = pred_unposed_K
        if pred_K_unposed.ndim == 4:
            pred_K_unposed = pred_K_unposed.squeeze(0)

        aligned_t = torch.from_numpy(aligned).float()
        if aligned_t.shape[-2:] == (3, 4):
            aligned_t = _to_se3(aligned_t)
        m_unposed = _eval_recon(
            pred_depth_unposed, rgbs, pred_K_unposed, aligned_t.numpy(),
            gt_depths, gt_K, gt_w2c.numpy(),
            max_depth=max_depth,
        )
        sm.fscore_unposed = float(m_unposed["fscore"])
        sm.prec_unposed = float(m_unposed["precision"])
        sm.recall_unposed = float(m_unposed["recall"])
    except Exception as e:
        traceback.print_exc()
        print(f"  [recon_unposed FAILED] {dataset}/{scene}: {e}", flush=True)

    return sm


def _print_table(rows: list[SceneMetrics], label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'dataset':<10} {'scene':<40} {'AUC30':>7} {'AUC15':>7} {'F_posed':>9} {'F_unp':>8}")
    print("-" * 85)
    for r in rows:
        print(f"{r.dataset:<10} {r.scene:<40} {r.auc30:>7.4f} {r.auc15:>7.4f} "
              f"{r.fscore_posed:>9.4f} {r.fscore_unposed:>8.4f}")
    if not rows:
        return
    means = {
        "auc30": np.nanmean([r.auc30 for r in rows]),
        "auc15": np.nanmean([r.auc15 for r in rows]),
        "fp": np.nanmean([r.fscore_posed for r in rows]),
        "fu": np.nanmean([r.fscore_unposed for r in rows]),
    }
    print("-" * 85)
    print(f"{'MEAN':<10} {'':<40} {means['auc30']:>7.4f} {means['auc15']:>7.4f} "
          f"{means['fp']:>9.4f} {means['fu']:>8.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Ckpt path. Omit for the un-patched DA3 zero-shot reference.")
    ap.add_argument("--no-patch", action="store_true",
                    help="Skip install_mamba3; eval an un-patched DA3 model "
                    "(reference or un-patched-overfit ckpt).")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--max-images", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--frame-stride", type=int, default=80)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--also-reference", action="store_true",
                    help="Also evaluate un-patched DA3 reference for direct comparison.")
    ap.add_argument("--pose-only", action="store_true",
                    help="Skip recon (F-score) eval, run only pose AUC.")
    # Per-scene-overfit eval (PLAN §15.59). When set, eval is restricted to one
    # scene's held-out view indices (typically loaded from a training run's split.json).
    ap.add_argument("--scene-overfit", type=str, default=None,
                    help="Scene name; restricts eval to one scene's held-out views.")
    ap.add_argument("--scene-dataset", type=str, default="eth3d",
                    choices=["eth3d", "hiroom", "7scenes"])
    ap.add_argument("--split-json", type=Path, default=None,
                    help="Path to split.json from training; uses its 'test' indices.")
    ap.add_argument("--view-indices", type=int, nargs="+", default=None,
                    help="Manual override: explicit positional view indices to eval.")
    args = ap.parse_args()

    if args.scene_overfit is not None:
        if args.split_json is not None:
            _, view_indices = read_split(args.split_json)
        elif args.view_indices is not None:
            view_indices = list(args.view_indices)
        else:
            ap.error("--scene-overfit requires either --split-json or --view-indices")
        scenes = [(args.scene_dataset, args.scene_overfit)]
        print(f"[phase4] scene-overfit eval on {args.scene_dataset}/{args.scene_overfit}, "
              f"held-out test indices = {view_indices}")
    else:
        view_indices = None
        scenes = _scene_iter(args.data_root)
        print(f"[phase4] eval scenes: {len(scenes)}")
        for d, s in scenes:
            print(f"  {d}/{s}")

    label_main = "un-patched DA3" if args.no_patch else "patched DA3"
    src = args.ckpt if args.ckpt is not None else "DA3-SMALL pretrained (no ckpt)"
    print(f"\n[phase4] loading {label_main} from {src}")
    student = build_patched_api(args.ckpt, device=args.device, state_dim=args.state_dim,
                                patched=not args.no_patch)

    student_rows: list[SceneMetrics] = []
    for ds, sc in scenes:
        print(f"\n[student] {ds}/{sc}", flush=True)
        try:
            sm = evaluate_one_scene(
                student, ds, sc, args.data_root,
                max_images=args.max_images, image_size=args.image_size,
                frame_stride=args.frame_stride, recon=not args.pose_only,
                view_indices=view_indices,
            )
            student_rows.append(sm)
            print(f"  AUC30={sm.auc30:.4f}  F_posed={sm.fscore_posed:.4f}  F_unp={sm.fscore_unposed:.4f}", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"  [FAILED] {ds}/{sc}: {e}", flush=True)

    _print_table(student_rows, "STUDENT (patched DA3 + Mamba-3)")

    if args.also_reference:
        print(f"\n[phase4] loading un-patched DA3 reference")
        del student
        torch.cuda.empty_cache()
        ref = build_patched_api(None, device=args.device, patched=False)
        ref_rows: list[SceneMetrics] = []
        for ds, sc in scenes:
            print(f"\n[reference] {ds}/{sc}", flush=True)
            try:
                sm = evaluate_one_scene(
                    ref, ds, sc, args.data_root,
                    max_images=args.max_images, image_size=args.image_size,
                    frame_stride=args.frame_stride, recon=not args.pose_only,
                    view_indices=view_indices,
                )
                ref_rows.append(sm)
                print(f"  AUC30={sm.auc30:.4f}  F_posed={sm.fscore_posed:.4f}  F_unp={sm.fscore_unposed:.4f}", flush=True)
            except Exception as e:
                print(f"  [FAILED] {ds}/{sc}: {e}", flush=True)
        _print_table(ref_rows, "REFERENCE (un-patched DA3-SMALL)")


if __name__ == "__main__":
    main()
