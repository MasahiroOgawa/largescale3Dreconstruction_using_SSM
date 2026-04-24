# largescale3Dreconstruction_using_SSM

## 1. About this repository

This project is an ablation study that replaces the softmax self-attention
blocks in **Depth-Anything-3** (DA3) with **Mamba-3 SSD** mixers, to ask
whether a state-space backbone of matched parameter budget can reach the
same monocular-depth quality as an attention backbone. The depth head
(DualDPT) is kept frozen from the DA3-SMALL checkpoint for most of
training, so the comparison isolates the backbone.

- Architecture: DINOv2-ViT-S/14 backbone (~22 M params, matched to
  DA3-SMALL) with its attention layers patched at import time via
  `ssm3d.patch.install_mamba3`. See
  `outputs/eval_cm22_1000/arch_{da3,ssm3d,diff}.png` for the block-level
  diagrams.
- Training is two-phase: **Phase B** distils backbone features from a
  frozen DA3-SMALL teacher into the SSM student with a per-layer
  `DimBridge`; **Phase C** fine-tunes for depth on ETH3D `terrains` with
  DualDPT unfrozen.
- Current best: **CM22@1000**
  (`|relative_depth_error|` = **0.0531**, δ<1.25 = **0.9972**) on ETH3D
  `terrains`, median-aligned. DA3-SMALL reference on the same views is
  **0.0417**, so the gap is **1.27×**. See `doc/PLAN.md §15.13` for the
  full recipe and `outputs/eval_cm22_1000/summary.md` for head-to-head
  numbers.
- The DA3 submodule is treated as read-only upstream. Swaps happen at
  runtime via the patch module; see `src/ssm3d/patch.py`.

## 2. How to set up

The venv is managed by **`uv`**. `pip` is not used in this project.

```bash
git clone --recurse-submodules <repo-url>
cd largescale3Dreconstruction_using_SSM
uv sync
```

`uv sync` installs the pinned dependency set from `pyproject.toml` +
`uv.lock`, including the local editable submodule at
`third_party/depth-anything-3`.

Requirements:

- CUDA-capable GPU (training uses bf16; inference works on CPU but is
  slow). Mamba-3 SSD requires the CUDA kernels from `mamba-ssm` installed
  by `uv sync`.
- First run downloads **ETH3D `terrains`** into `data/` and the
  **DINOv2-ViT-S/14** backbone weights via Hugging Face; both are cached.
- See `CLAUDE.md` for the full packaging rules. In particular: never
  invoke `pip` directly — always go through `uv`.

## 3. How to use

### Quick demo (a few views, no training)

```bash
uv run python scripts/run_demo.py
```

Produces patch-feature PCA visualisations, SSM-3D depth predictions, and
a collapse-smoke-check report (see `doc/PLAN.md §3`). Artifacts land in
`outputs/demo/`.

### Train (Phase B distil → Phase C depth FT, CM22 recipe)

```bash
# Phase B: distil DA3-SMALL teacher features into the SSM student
uv run python scripts/train_phase_b.py \
    --img-size 504 --patch-size 14 --chunk-size 128 \
    --steps 20000 --bs 1

# Phase C: depth fine-tune on ETH3D `terrains` (DualDPT unfrozen)
uv run python scripts/train_phase_c.py \
    --init outputs/runs/phase_b/ckpt_final.pt \
    --img-size 504 --patch-size 14 --chunk-size 128 \
    --steps 1000 --bs 1 \
    --lr-attn 1e-5 --lr-bridge 3e-5 --lr-dpt 1e-5 \
    --augment
```

The exact CM22 recipe (and why each hyperparameter is set this way) is
in `doc/PLAN.md §15.13`.

### Evaluate (SSM-3D vs DA3-SMALL, head-to-head on ETH3D)

```bash
uv run python scripts/eval_ssm3d_vs_da3.py \
    --ckpt outputs/runs/depth_ft_cm22/ckpt_1000.pt \
    --out  outputs/eval_cm22_1000
```

Writes per-metric bar plots, head-to-head grids, `summary.md` with the
acceptance-gate table, and architecture diagrams. See
`doc/evaluation.md` for metric definitions and gate thresholds.

## 4. References

### Upstream papers / repos

- **Depth Anything 3** — backbone + DualDPT head we patch.
  <https://github.com/ByteDance-Seed/DepthAnything>
- **Mamba / Mamba-3 SSD** — state-space mixer that replaces softmax
  attention. See Dao & Gu, *"Transformers are SSMs: Generalized Models
  and Efficient Algorithms Through Structured State Space Duality"*
  (ICML 2024).
- **DINOv2** — shared ViT-S/14 initialisation.
  <https://github.com/facebookresearch/dinov2>
- **ETH3D** benchmark — evaluation dataset (`terrains` scene, 42 views
  used for the held-out comparison).
  <https://www.eth3d.net/>

### Internal docs

- `doc/PLAN.md` — experiment log, acceptance gates, candidate-modification
  (CM) index, and reverted / kept decisions.
- `doc/evaluation.md` — metric definitions, median-alignment convention,
  and the head-to-head snapshot at CM22@1000.
- `CLAUDE.md` — project-local rules (uv / DA3 submodule / memory /
  sudo).
- `MEMORY.md` — index into `memory/*.md` for per-topic project memory.
