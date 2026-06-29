"""Run DA3's official benchmark evaluator on a Mamba-3 / VSSD-patched checkpoint.

Installs the `.pt`-aware `DepthAnything3.from_pretrained` shim from
:mod:`mamba3_attn.eval.da3_hf_adapter`, then re-execs the upstream evaluator
via :func:`runpy.run_module` so all DA3 CLI flags pass through unchanged.

Examples::

    # 1) Baseline sanity: un-patched DA3-SMALL via its HF repo ID. Numbers
    #    must match the published table before trusting any patched run.
    uv run python scripts/run_da3_bench_eval.py \\
        model.path=depth-anything/DA3-SMALL \\
        eval.datasets=[hiroom] \\
        eval.modes=[recon_posed]

    # 2) Patched VSSD-DA3 ckpt. CUDA_VISIBLE_DEVICES=0 keeps the evaluator
    #    in single-GPU mode (multi-GPU subprocess-spawns and would bypass
    #    the from_pretrained shim).
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_da3_bench_eval.py \\
        model.path=result/runs/vssd_da3_stageB/ckpt_final.pt \\
        eval.datasets=[eth3d,7scenes,scannetpp,hiroom,dtu] \\
        eval.modes=[pose,recon_posed,recon_unposed] \\
        workspace.work_dir=result/eval_vssd_full/accuracy
"""

from __future__ import annotations

import os
import runpy
import sys

from mamba3_attn.eval.da3_hf_adapter import install_pt_loader


def main() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # Multi-GPU path subprocess-spawns `python -m depth_anything_3.bench.evaluator`
    # which loses the monkey-patch. Warn loudly so a patched-ckpt run doesn't
    # silently fall back to a vanilla DA3 model.
    if "," in visible:
        sys.stderr.write(
            "[run_da3_bench_eval] WARNING: CUDA_VISIBLE_DEVICES has multiple GPUs "
            f"({visible!r}); DA3's evaluator will subprocess-spawn workers that\n"
            "                   bypass the .pt loader shim. Set CUDA_VISIBLE_DEVICES=0 "
            "for patched-ckpt runs.\n"
        )

    install_pt_loader()

    # Hand argv off to DA3's `if __name__ == '__main__'` block by running the
    # module under `__main__`. argv[0] needs to look like the module entry,
    # the rest is what we want OmegaConf to see as `model.path=...` etc.
    if sys.argv and sys.argv[0].endswith(".py"):
        sys.argv[0] = "depth_anything_3.bench.evaluator"
    runpy.run_module("depth_anything_3.bench.evaluator", run_name="__main__")


if __name__ == "__main__":
    main()
