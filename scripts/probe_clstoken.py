"""T1+T2 probe: forward sanity + per-block cls/cam-token diff between
unpatched DA3-SMALL, patched (random init), and patched (Step-C ckpt).

See `doc/PLAN.md §15.58`. Goal:
- T1: detect a cls-token bug in the swap (NaN/Inf, shape collapse, magnitude
  step-jump at one block).
- T2: quantify how close the trained Mamba-3 cls/cam-tokens are to the
  un-patched transformer's, layer-by-layer.

Usage:
    uv run python scripts/probe_clstoken.py \\
        --ckpt outputs/runs/sp1_sub1_long/ckpt_5000.pt \\
        --out-dir outputs/probes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Triggers the legacy `ssm3d.*` → `mamba3_attn.*` aliasing in __init__.py
# so old ckpts (pre-rename) can be unpickled.
import mamba3_attn  # noqa: E402, F401


def fixed_batch(image_size: int, n_views: int, dataset: str, scene: str,
                seed: int) -> torch.Tensor:
    """Load a fixed batch of multi-view images at the model's input format.

    Returns a (n_views, 3, H, W) float32 tensor in [0, 1].
    """
    from mamba3_attn.data.bench import load_bench_scene
    torch.manual_seed(seed)
    sample = load_bench_scene(
        dataset, scene, data_root=Path("data"),
        max_images=n_views, image_size=image_size, load_gt_depth=False,
    )
    return sample.images[:n_views]


def preprocess(api, images: torch.Tensor, process_res: int) -> torch.Tensor:
    """Run DA3's preprocessing → (B=1, S, 3, H', W') ready for backbone."""
    arr = (images.clamp(0, 1) * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
    image_list = [arr[i] for i in range(arr.shape[0])]
    imgs_cpu, _, _ = api._preprocess_inputs(
        image_list, None, None, process_res, "upper_bound_resize"
    )
    imgs, _, _ = api._prepare_model_inputs(imgs_cpu, None, None)
    return imgs.to(api.device)


class BlockTokenCapture:
    """Forward hook capturing `output[..., 0, :]` (the cls/cam-token at index
    0 of every per-view sequence) on each backbone block."""

    def __init__(self, blocks):
        self.handles = []
        self.captured: list[torch.Tensor | None] = [None] * len(blocks)
        for i, block in enumerate(blocks):
            self.handles.append(block.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.captured[idx] = output[..., 0, :].detach().float().cpu()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def run_variant(api, imgs: torch.Tensor) -> dict:
    blocks = list(api.model.backbone.pretrained.blocks)
    cap = BlockTokenCapture(blocks)
    try:
        feats, _aux = api.model.backbone(
            imgs, cam_token=None, export_feat_layers=[], ref_view_strategy="first",
        )
    finally:
        cap.remove()
    has_nan = any(t is not None and not torch.isfinite(t).all() for t in cap.captured)
    shapes = [tuple(t.shape) if t is not None else None for t in cap.captured]
    return {
        "tokens": cap.captured,
        "has_nan_or_inf": has_nan,
        "shapes": shapes,
        "n_blocks": len(blocks),
    }


def per_block_diff(ref: list[torch.Tensor], cmp: list[torch.Tensor]) -> dict:
    rel_l2 = []
    cos = []
    for r, c in zip(ref, cmp):
        if r is None or c is None or r.shape != c.shape:
            rel_l2.append(float("nan"))
            cos.append(float("nan"))
            continue
        r_f = r.reshape(-1, r.shape[-1]).float()
        c_f = c.reshape(-1, c.shape[-1]).float()
        rel_l2.append(((r_f - c_f).norm(dim=-1) / (r_f.norm(dim=-1) + 1e-8)).mean().item())
        cos.append(F.cosine_similarity(r_f, c_f, dim=-1).mean().item())
    return {"rel_l2": rel_l2, "cos": cos}


def heatmap(diffs: dict, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ["random vs xfmr (cos)", "step_c vs xfmr (cos)",
            "random vs xfmr (rel L2)", "step_c vs xfmr (rel L2)"]
    data = np.array([
        diffs["rand_vs_xfmr"]["cos"],
        diffs["step_c_vs_xfmr"]["cos"],
        diffs["rand_vs_xfmr"]["rel_l2"],
        diffs["step_c_vs_xfmr"]["rel_l2"],
    ])
    n_blocks = data.shape[1]
    fig, ax = plt.subplots(figsize=(0.7 * n_blocks + 2, 3.5))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels([f"b{i}" for i in range(n_blocks)], rotation=0)
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="w" if v < 0.5 else "k", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def evaluate_t1(xfmr: dict, rand: dict, step_c: dict) -> dict:
    """T1 — structural sanity. Returns PASS/FAIL with reasons."""
    failures = []
    for name, r in (("rand", rand), ("step_c", step_c)):
        if r["has_nan_or_inf"]:
            failures.append(f"{name}: NaN/Inf in captured tokens")
        for i, (ref_s, our_s) in enumerate(zip(xfmr["shapes"], r["shapes"])):
            if ref_s != our_s:
                failures.append(f"{name}: block {i} shape {our_s} != xfmr {ref_s}")
                break
    return {"pass": not failures, "failures": failures}


def evaluate_t2(diffs: dict, feat_layers: tuple[int, ...] = (5, 7, 9, 11)) -> dict:
    """T2 — feature-quality verdict at the layers used for feat-distill."""
    cos_at = {i: diffs["step_c_vs_xfmr"]["cos"][i] for i in feat_layers}
    cos_min = min(cos_at.values())
    if cos_min > 0.9:
        verdict = "features_close"  # downstream is the consumer that fails
    elif cos_min > 0.5:
        verdict = "features_partial"
    else:
        verdict = "features_lost"  # cls-tokens lost most of their content
    spike = False
    cos_seq = diffs["step_c_vs_xfmr"]["cos"]
    for i in range(1, len(cos_seq) - 1):
        if not all(np.isfinite([cos_seq[i - 1], cos_seq[i], cos_seq[i + 1]])):
            continue
        neighbors_avg = (cos_seq[i - 1] + cos_seq[i + 1]) / 2
        if cos_seq[i] < neighbors_avg - 0.3:
            spike = True
            break
    return {"verdict": verdict, "cos_at_feat_layers": cos_at,
            "cos_min_at_feat_layers": cos_min, "has_spike": spike}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Step-C / sp1_sub1_long ckpt path for the patched-trained variant")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/probes"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--image-size", type=int, default=504)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--dataset", type=str, default="eth3d")
    ap.add_argument("--scene", type=str, default="courtyard")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from mamba3_attn.eval.phase4_evaluator import build_patched_api

    print("[probe] building three model variants", flush=True)
    api_xfmr = build_patched_api(ckpt_path=None, device=args.device, patched=False)
    api_rand = build_patched_api(ckpt_path=None, device=args.device, patched=True)
    api_step_c = build_patched_api(ckpt_path=args.ckpt, device=args.device, patched=True)

    print(f"[probe] loading fixed batch: {args.dataset}/{args.scene} "
          f"({args.n_views} views @ {args.image_size}²)", flush=True)
    images = fixed_batch(args.image_size, args.n_views, args.dataset, args.scene, args.seed)

    imgs_xfmr = preprocess(api_xfmr, images, args.image_size)
    imgs_rand = preprocess(api_rand, images, args.image_size)
    imgs_step_c = preprocess(api_step_c, images, args.image_size)

    print("[probe] running forward + capturing per-block tokens", flush=True)
    out_xfmr = run_variant(api_xfmr, imgs_xfmr)
    out_rand = run_variant(api_rand, imgs_rand)
    out_step_c = run_variant(api_step_c, imgs_step_c)

    diffs = {
        "rand_vs_xfmr": per_block_diff(out_xfmr["tokens"], out_rand["tokens"]),
        "step_c_vs_xfmr": per_block_diff(out_xfmr["tokens"], out_step_c["tokens"]),
    }

    t1 = evaluate_t1(out_xfmr, out_rand, out_step_c)
    t2 = evaluate_t2(diffs)

    payload = {
        "config": {"ckpt": args.ckpt, "image_size": args.image_size,
                   "n_views": args.n_views, "dataset": args.dataset,
                   "scene": args.scene, "seed": args.seed},
        "n_blocks": out_xfmr["n_blocks"],
        "shapes": {"xfmr": out_xfmr["shapes"], "rand": out_rand["shapes"],
                   "step_c": out_step_c["shapes"]},
        "diffs": diffs,
        "T1": t1,
        "T2": t2,
    }
    json_path = args.out_dir / "cls_token_diff.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[probe] wrote {json_path}", flush=True)

    heatmap(diffs, args.out_dir / "cls_token_diff.png",
            title=f"cls/cam-token diff per block — {args.dataset}/{args.scene}")
    print(f"[probe] wrote {args.out_dir / 'cls_token_diff.png'}", flush=True)

    print("\n=== T1 — Forward sanity ===")
    if t1["pass"]:
        print("PASS — no NaN/Inf, all shapes match xfmr reference.")
    else:
        print("FAIL")
        for f in t1["failures"]:
            print(f"  - {f}")

    print("\n=== T2 — Feature quality (cos vs xfmr at feat-distill layers 5/7/9/11) ===")
    for i, c in t2["cos_at_feat_layers"].items():
        print(f"  block {i:2d}  cos={c:+.4f}")
    print(f"  min cos = {t2['cos_min_at_feat_layers']:+.4f}  →  {t2['verdict']}")
    if t2["has_spike"]:
        print("  WARNING: cosine has a single-block dip ≥ 0.3 below neighbors "
              "(possible cls-token bug at one specific layer).")

    print("\n=== T2 — Per-block summary (step_c vs xfmr) ===")
    print("  blk |   cos   | rel_l2")
    for i, (c, l) in enumerate(zip(diffs["step_c_vs_xfmr"]["cos"],
                                   diffs["step_c_vs_xfmr"]["rel_l2"])):
        print(f"  {i:3d} | {c:+.4f} | {l:.4f}")


if __name__ == "__main__":
    main()
