"""Phase 4: eval our patched DA3 ckpt via DA3's official Evaluator.

Wraps the patched DA3-SMALL model in the standard `DepthAnything3` api
interface, then runs `bench.evaluator.Evaluator` over the eval split:
- ETH3D `terrains`
- HiRoom val (4 held-out scenes — last 4 of selected_scene_list_val.txt)
- 7Scenes test (`pumpkin`, `redkitchen`, `stairs`)

Modes: pose, recon_posed, recon_unposed (DA3 paper's full benchmark).

Usage:
    uv run python -m ssm3d.eval.phase4_evaluator \\
        --ckpt outputs/runs/phase3_unfreeze/ckpt_500.pt \\
        --work-dir outputs/runs/phase4_eval
"""

from __future__ import annotations

import argparse

import torch

from ..eval.da3_reference import DEFAULT_HF_MODEL, load_da3
from ..patch import install_mamba3


def build_patched_api(ckpt_path: str, device: str = "cuda", state_dim: int = 64):
    """Load DA3-SMALL → install_mamba3 → load our ckpt → return api."""
    api = load_da3(DEFAULT_HF_MODEL, device=device)
    install_mamba3(
        api.model, which="all", state_dim=state_dim,
        use_fused_kernel=True, chunk_size=128,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    api.model.load_state_dict(state["model"])
    api = api.to(device)
    api.model.eval()
    return api


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--work-dir", type=str, default="outputs/runs/phase4_eval")
    ap.add_argument(
        "--datasets", nargs="+",
        default=["eth3d", "hiroom", "7scenes"],
    )
    ap.add_argument(
        "--modes", nargs="+",
        default=["pose", "recon_posed", "recon_unposed"],
    )
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="Restrict to these scene names (e.g., terrains chess)")
    args = ap.parse_args()

    from depth_anything_3.bench.evaluator import Evaluator

    print(f"[phase4] loading patched DA3 from {args.ckpt}")
    api = build_patched_api(args.ckpt, state_dim=args.state_dim)

    evaluator = Evaluator(
        work_dir=args.work_dir,
        datas=args.datasets,
        modes=args.modes,
        scenes=args.scenes,
        max_frames=args.max_frames,
    )
    evaluator.infer(api)
    metrics = evaluator.eval()
    evaluator.print_metrics(metrics)


if __name__ == "__main__":
    main()
