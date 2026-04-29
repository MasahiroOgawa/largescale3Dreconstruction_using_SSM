# PLAN — CIFAR-10 sanity check: Mamba-3 attention vs softmax attention vs CNN

> **See also:** [`doc/PLAN_mamba3_DA3.md`](PLAN_mamba3_DA3.md) — the prior DA3 ↔ Mamba-3 distillation plan (CM01–CM22, super-phase / sub-phase, etc.). All section numbers cited as `§15.x` from earlier commits live there.

## §1. Context — why this side-track

The depth pipeline (`PLAN_mamba3_DA3.md §15.13`) currently shows a **1.27×** gap between the Mamba-3-patched DA3-SMALL backbone and the un-patched DA3-SMALL transformer on ETH3D `terrains` (`|relative_depth_error|` 0.0531 vs 0.0417). Before continuing to invest training cycles in the Mamba-3 swap, we want a **clean, isolated sanity check** that answers:

> *On real images, with matched parameter budget and matched recipe, can our Mamba-3 attention swap reach the same classification accuracy as softmax attention from scratch?*

If the answer is "yes, parity within ~2 pp", then the depth gap is most plausibly a training-recipe / distillation issue we can keep working on. If "no", the depth gap likely reflects a real mixing-quality limitation of Mamba-3 SSD at this scale, and the project's load-bearing assumption needs revisiting.

CIFAR-10 is the right minimum. MNIST saturates at ~99 % for any reasonable classifier and won't differentiate the mixers. CIFAR-10 with 4×4 patches gives **64 patch tokens + 1 CLS = 65 tokens**, long enough for token-mixing differences to show up but short enough that one full run finishes in well under an hour on a single GPU.

## §2. Variants (all from scratch, matched parameter budget ≈ 2.7 M)

| # | Name (`--variants` key) | Architecture | Notes |
|---|---|---|---|
| 1 | `cnn` | Small ResNet — stem + 3 stages × 2 BasicBlocks, widths {64, 128, 256} | Sanity floor; CNNs are extremely well-tuned on CIFAR-10. |
| 2 | `vit_attn` | ViT-Tiny: depth=6, dim=192, heads=3, MLP×4, 4×4 patch embed, learnable absolute pos-emb, CLS token | Manual timm-style multi-head softmax attention (`VanillaAttention`) with explicit `qkv = nn.Linear(dim, 3*dim)` and `proj = nn.Linear(dim, dim)` submodules — required by `install_mamba3._infer_dim` / `_infer_num_heads`. |
| 3 | `vit_mamba3` | Same skeleton as #2, then `ssm3d.patch.install_mamba3(net, which="backbone_only")` | Uses the **same swap path** the DA3 depth project uses (`src/ssm3d/patch.py`). |

The wrapper for #2 must expose `qkv: Linear`, `proj: Linear`, and `num_heads` so `install_mamba3._infer_dim` / `_infer_num_heads` succeed when applied to #3.

The classifier wrapper (`Classifier(nn.Module)`) exposes `self.backbone.blocks` so `patch._backbone_blocks` finds the transformer block list.

## §3. Training recipe (identical across all 3 variants)

| Hyperparam | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-3 |
| Weight decay | 0.05 |
| Schedule | 5-epoch linear warmup → cosine decay to 0 |
| Batch size | 128 |
| Epochs | 50 (full); 5 if `--device cpu` (smoke test) |
| Augmentation | RandomCrop(32, padding=4) + HorizontalFlip + Normalize(CIFAR-10 mean/std) |
| Mixed precision | bf16 autocast on CUDA |
| Seed | 42 (single seed; documented caveat) |
| Loss | Cross-entropy |

CIFAR-10 = 50 000 train / 10 000 test, cached under `data/cifar10/`. 50 ep × 391 step/ep ≈ 20 k steps; expected ~10–20 min per variant on a single CUDA GPU.

## §4. Metrics & outputs

Per epoch, per variant, log: train top-1 acc, train loss, test top-1 acc, test loss, epoch wall-clock (s), LR. Keep best-test-acc checkpoint per variant.

After training, run a **fixed efficiency benchmark** per variant (reusing `measure(...)` and `count_params(...)` from `scripts/bench_efficiency_patched.py`):

```python
x = torch.randn(128, 3, 32, 32, device=device)
with torch.inference_mode():
    eff = measure(lambda inp: model(inp), x, device, warmup=3, repeats=10)
# {"latency_ms": ..., "peak_mib": ...}
```

Artifacts under `outputs/cifar10_compare/`:

```
results.json              # per-variant final + per-epoch metrics
curves.png                # train/test acc + train loss vs epoch (3 lines each)
efficiency_table.md       # params, latency, peak mem, train s/epoch
summary.md                # head-to-head + leaderboard refs + acceptance gate verdict
ckpt_<variant>.pt         # best test-acc checkpoint per variant
```

`summary.md` head-to-head table:

| Variant | Params | Train Acc | Test Acc | Train s/epoch | Test lat (ms, B=128) | Peak MiB |
|---|---|---|---|---|---|---|
| CNN (small ResNet) |  |  |  |  |  |  |
| ViT-Tiny + softmax |  |  |  |  |  |  |
| ViT-Tiny + Mamba-3 SSD |  |  |  |  |  |  |

Reference leaderboard (orientation only — published numbers, not from this run):

| Reference | Test Acc | Notes |
|---|---|---|
| ResNet-20 (He et al. 2015, standard recipe) | ~91.7 % | 0.27 M params, 200 ep |
| ResNet-110 | ~93.6 % | 1.7 M params |
| ViT-Tiny from scratch (Steiner et al. 2021) | ~85–88 % | similar size, no strong aug |
| ViT-Tiny + Mixup/Cutmix | ~92 % | strong aug |
| ViT-Huge (JFT-pretrained, fine-tuned) | >99 % | not comparable; orientation only |

Live leaderboard: <https://paperswithcode.com/sota/image-classification-on-cifar-10>.

## §5. Acceptance gate

The script answers: **"is Mamba-3 attention as good as softmax attention from scratch on real images, at matched parameter budget?"**

- **Pass** = `test_acc(vit_mamba3) ≥ test_acc(vit_attn) − 2 pp` AND `peak_mib(vit_mamba3) ≤ peak_mib(vit_attn) × 1.1` at this short sequence length (T=65; attention is essentially free here, so memory parity is the floor — not a "win" target).
- **Fail** = mamba3 is more than 2 pp behind softmax. That would suggest the depth-task gap (1.27×) reflects a real Mamba-3 mixing-quality limitation, not just a training-recipe issue, and would justify reconsidering the swap before further investment.

The CNN row is a sanity floor: both ViT variants should beat random (10 %) and approach the CNN; if either doesn't, the recipe (not the mixer) is broken — fix the recipe before drawing conclusions.

## §6. CLI

```bash
uv run python scripts/cifar10_compare.py \
    --epochs 50 --batch-size 128 --lr 1e-3 \
    --out outputs/cifar10_compare \
    --variants cnn vit_attn vit_mamba3 \
    --device cuda --seed 42
```

`--variants` allows running a subset for quick smoke tests. `--device cpu` auto-shortens to 5 epochs and prints a "smoke test" banner.

## §7. Out of scope (deliberately)

- CIFAR-100 (only run if v1 results are inconclusive).
- Multiple seeds (single seed v1; multi-seed only if the gap is small).
- Mixup / Cutmix (adds confound; vanilla aug only for the v1 comparison).
- FLOPs counting (wall-clock + peak memory is enough for a sanity check).
- Modifying `src/ssm3d/` (no project-code changes — pure additive script).

## §8. References

- **Mamba-3 / Mamba-2 SSD** — Dao & Gu, *"Transformers are SSMs"* (ICML 2024). Drop-in replacement for softmax attention via `ssm3d.patch.install_mamba3`.
- **ViT** — Dosovitskiy et al. 2020.
- **Steiner et al. 2021** — *"How to train your ViT"* (data, augmentation, regularization on small ViTs).
- **He et al. 2015** — ResNet (CIFAR-10 baselines).
- **paperswithcode CIFAR-10** — <https://paperswithcode.com/sota/image-classification-on-cifar-10>.
