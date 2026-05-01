# PLAN — CIFAR-10 sanity check: Mamba-3 attention vs softmax attention vs CNN

> **See also:** [`doc/PLAN_mamba3_DA3.md`](PLAN_mamba3_DA3.md) — the prior DA3 ↔ Mamba-3 distillation plan (CM01–CM22, super-phase / sub-phase, etc.). All section numbers cited as `§15.x` from earlier commits live there.

> **Status (2026-04-29):** §9.7 Mamba-friendly recipe rerun finished — `vit_mamba3` test acc **70.04 %**, +25.45 pp over §9.6's 44.59 % (no collapse), but still **−12.46 pp** behind the softmax baseline. Recipe was a real confound, mixing-quality gap is also real. See [§9.8](#98-results--mamba-friendly-recipe-rerun-97). **Next: §9.9 plateau-scheduler 3-variant rerun** using `ReduceLROnPlateau` on EMA(train_loss) — see [§9.9](#99-next-action--plateau-scheduler-rerun-all-3-variants).

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
| 3 | `vit_mamba3` | Same skeleton as #2, then `mamba3_attn.patch.install_mamba3(net, which="backbone_only")` | Uses the **same swap path** the DA3 depth project uses (`src/mamba3_attn/patch.py`). |

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
- Modifying `src/mamba3_attn/` (no project-code changes — pure additive script).

## §8. References

- **Mamba-3 / Mamba-2 SSD** — Dao & Gu, *"Transformers are SSMs"* (ICML 2024). Drop-in replacement for softmax attention via `mamba3_attn.patch.install_mamba3`.
- **ViT** — Dosovitskiy et al. 2020.
- **Steiner et al. 2021** — *"How to train your ViT"* (data, augmentation, regularization on small ViTs).
- **He et al. 2015** — ResNet (CIFAR-10 baselines).
- **paperswithcode CIFAR-10** — <https://paperswithcode.com/sota/image-classification-on-cifar-10>.

## §9. Implementation status

### §9.1. Script

`scripts/cifar10_compare.py` — single self-contained file, ~430 lines, landed in commit `a723abf` (2026-04-29). Reuses `measure(...)` and `count_params(...)` from `scripts/bench_efficiency_patched.py`; no edits under `src/mamba3_attn/`.

### §9.2. Param-budget verification (matched ≈ 2.7 M)

| Variant | Measured params | vs target |
|---|---:|---|
| `cnn` (SmallResNet) | 2,777,674 (2.78 M) | +0.08 M |
| `vit_attn` (ViT-Tiny@d6 + softmax) | 2,693,578 (2.69 M) | −0.01 M |
| `vit_mamba3` (same skeleton, 6/6 attn → Mamba3) | 2,707,474 (2.71 M) | +0.01 M |

`install_mamba3(model, which="backbone_only")` swaps **all 6/6** transformer blocks on the bare ViT classifier — the `_backbone_blocks` fallback to `net.backbone.blocks` works as designed. Mamba-3 SSD adds ~14 k params over softmax at this size; well within "matched".

### §9.3. Smoke-test results (wiring verified)

**CPU smoke** — `cnn`, `vit_attn`, 1 epoch, seed 42:

| Variant | Train acc | Test acc | Wall (s/epoch) | Latency @ B=128 |
|---|---:|---:|---:|---:|
| `cnn` | 47.67 % | 51.67 % | 306.7 | 220.0 ms |
| `vit_attn` | 33.24 % | 45.11 % | 161.2 | 127.0 ms |

Both variants train without error, all four artifacts (`results.json`, `curves.png`, `efficiency_table.md`, `summary.md`) are written, and the gate path correctly reports "_not evaluable_" when `vit_mamba3` is absent.

**CUDA sanity** — `vit_mamba3`, 2 epochs, RTX 4080 Laptop:

| Epoch | Train acc | Test acc | Wall (s) |
|---:|---:|---:|---:|
| 1 | 10.75 % | 16.31 % | 18.4 |
| 2 | 23.91 % | 29.93 % | 17.5 |

Efficiency at B=128, T=65: latency 13.2 ms, peak 153.5 MiB. No NaNs; learning is in progress. Per-epoch wall-clock projects to **~15 min/variant** for the full 50-ep run, **~45 min total** on this GPU.

### §9.4. Next action

Run the full 50-ep experiment, all three variants, single seed (per `§3` and `§7`):

```bash
uv run python scripts/cifar10_compare.py \
    --device cuda --epochs 50 --batch-size 128 --lr 1e-3 \
    --variants cnn vit_attn vit_mamba3 \
    --out outputs/cifar10_compare --seed 42
```

After the run completes, read `outputs/cifar10_compare/summary.md` for the head-to-head table and the §5 PASS/FAIL acceptance verdict. The smoke-test outputs at `outputs/cifar10_compare_smoke/` and `outputs/cifar10_compare_cuda_smoke/` can be deleted — they're no longer needed once the full run lands.

### §9.5. Branch decisions after the full run

- **PASS** (mamba3 within 2 pp of softmax, peak mem ≤ 1.1×): the depth-task 1.27× gap is a recipe / distillation issue. Resume `PLAN_mamba3_DA3.md` super-phase / feature-distill work (`§15.56`–`§15.57`).
- **FAIL** (mamba3 > 2 pp behind softmax): real Mamba-3 mixing-quality limitation at this scale; reconsider the swap before further investment. Possible follow-ups (only if FAIL): multi-seed re-run with `--seed {0,1,2}` to bound variance, then CIFAR-100 to confirm at slightly harder difficulty.

### §9.6. Results — full 50-ep run

Single seed=42, AdamW lr=1e-3 wd=0.05, 5-ep linear warmup → cosine, batch=128, RTX 4080 Laptop. Total wall-clock 23.5 min (cnn 5.2 min + vit_attn 4.8 min + vit_mamba3 12.5 min). All artifacts under `outputs/cifar10_compare/`.

**Head-to-head:**

| Variant | Params (M) | Train Acc | **Test Acc** | s/epoch | Latency (ms, B=128) | Peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| `cnn` (small ResNet) | 2.78 | 99.94 | **93.69** | 6.2 | 7.68 | 221.4 |
| `vit_attn` (ViT-Tiny + softmax) | 2.69 | 99.23 | **82.50** | 5.8 | 6.73 | 126.9 |
| `vit_mamba3` (ViT-Tiny + Mamba-3 SSD) | 2.71 | 41.10 | **44.59** | 15.0 | 13.73 | 153.5 |

Both `cnn` and `vit_attn` are well in line with the published references in §4: ResNet-20 ~91.7 %, ViT-Tiny from-scratch 85–88 %. Recipe is sound for softmax models.

**vit_mamba3 trajectory — instability, not stagnation:**

| Phase | Epochs | Test acc behavior |
|---|---|---|
| Slow warmup | 1–5 | 10.00 → 37.90 (climbing, LR ramps 0 → 1e-3) |
| Brief peak | 6 | **44.59** (best ckpt — saved here) |
| **Collapse** | 7–11 | 22.13 → 16.38 → 10.00 → 13.49 → 14.53 (back to ~random) |
| Slow recovery | 12–35 | 20.34 → 34.12 (re-learning from collapsed state) |
| Steady-state | 36–50 | 34.44 → 40.51 (final) |

The collapse begins exactly at epoch 7 — one epoch after LR finishes its 5-ep warmup and reaches the peak 1e-3. This is the canonical signature of **LR-too-high for an SSD/Mamba block**: SSD's recurrent state dynamics are stiffer than softmax attention, so a peak LR that is fine for ViT (1e-3) drives the SSD parameters out of the stable region. The model never fully recovers from the collapse within 50 epochs, but it is still descending at epoch 50, suggesting a longer schedule + lower LR would converge — which is consistent with how Mamba-2/3 papers train (typically lr ≤ 5e-4 with longer warmup and grad-clip).

**Acceptance gate (§5):** acc gap −37.91 pp (threshold ≥ −2.00 pp) → FAIL; mem ratio 1.21× (threshold ≤ 1.10×) → FAIL.

But §9.5's interpretation of FAIL ("*real Mamba-3 mixing-quality limitation*") presumes the recipe wasn't itself the confound. The collapse trajectory shows the recipe **is** the confound. So the run does not yet answer §1's question; it answered a different one ("does this exact recipe transfer?" — no).

### §9.7. Next action — Mamba-friendly recipe rerun

Re-run **only** `vit_mamba3` (cnn and vit_attn results stand) with the standard SSD-friendly recipe knobs:

| Knob | Old | New | Reason |
|---|---|---|---|
| Peak LR | 1e-3 | **3e-4** | SSD recurrence is stiffer; literature consensus for Mamba-2/3 from-scratch training. |
| Warmup epochs | 5 | **10** | Smoother ramp gives the recurrence time to stabilize before the LR peak. |
| Grad clip | none | **1.0** | Bounds the spike that triggers the ep-7 collapse. |
| Epochs | 50 | **80** | Lower LR ⇒ slower convergence; matches typical Mamba schedules at this size. |
| Everything else | unchanged | unchanged | Same data, aug, batch, optimizer, seed=42, same script. |

Implementation: add a `--lr-schedule mamba` flag (or simply `--peak-lr`, `--warmup-epochs`, `--grad-clip`) to `scripts/cifar10_compare.py`; do not touch `src/mamba3_attn/`. Output to `outputs/cifar10_compare_mamba_recipe/` so the original §9.6 numbers stay pristine.

**New acceptance gate** (same threshold, recipe-fair):

- **PASS-recipe** = `test_acc(vit_mamba3 @ Mamba-friendly recipe) ≥ 80.5 %` (i.e., within 2 pp of `vit_attn`'s 82.50). → recipe was the issue; depth-task 1.27 × gap is also recipe; resume `PLAN_mamba3_DA3.md`.
- **FAIL-recipe** = `< 80.5 %` even with Mamba-friendly recipe. → genuine mixing-quality gap; §9.5's FAIL interpretation now applies; reconsider the swap.

Estimated wall-clock: 80 ep × 15 s/ep ≈ 20 min on the same GPU.

### §9.8. Results — Mamba-friendly recipe rerun (§9.7)

Single seed=42, AdamW lr=3e-4 wd=0.05, 10-ep linear warmup → cosine, grad_clip=1.0, batch=128, 80 epochs, RTX 4080 Laptop. Wall-clock 20.6 min for the single variant. Artifacts under `outputs/cifar10_compare_mamba_recipe/`.

| Variant | Test Acc | Δ vs §9.6 same variant | Δ vs §9.6 vit_attn |
|---|---:|---:|---:|
| `vit_mamba3` @ §9.7 recipe | **70.04 %** (best, ep 74) | **+25.45 pp** | **−12.46 pp** |

**Trajectory — clean monotonic descent, no collapse.** Test acc climbs 10.00 → 70.04 across ep 1–74, then plateaus 69.9–70.0 at ep 74–80; train acc plateaus at 75.5. **Confirms the §9.6 ep-7 collapse was a real LR-too-high artifact.** Train loss bottoms at ~0.68 at ep 80 — still slowly descending; modest gains (≤ 1 pp) likely available with more epochs.

**Acceptance gate (§9.7 PASS-recipe ≥ 80.5 %):** **FAIL-recipe** by **−10.46 pp**.

**Interpretation:** the recipe was a real confound (+25 pp recovered just by fixing it), so §9.6's FAIL was not informative about Mamba-3's actual capability. But a **−12.46 pp gap survives proper SSD hyperparameters** — that's no longer attributable to recipe. This is a credible signal of a genuine mixing-quality / inductive-bias gap between SSD recurrence and softmax attention at T=65 from scratch, which likely accounts for a chunk of the depth-task 1.27× gap.

**Caveats before triggering §9.5's FAIL branch:**

- Single seed (multi-seed would bound variance to ±~1 pp at this scale).
- Only 3 SSD-friendly knobs adjusted; the post-warmup decay schedule is still cosine — could also try plateau (loss-driven) scheduling.
- T=65 is Mamba's weak regime; depth=6 is shallow.

These caveats motivate one more variant (§9.9) before closing the door.

### §9.9. Next action — plateau-scheduler rerun (all 3 variants)

Re-run **all 3 variants** with `ReduceLROnPlateau` on EMA(train_loss) replacing the cosine decay phase. Implemented in `scripts/cifar10_compare.py` via `--lr-schedule plateau`; see `WarmupPlateauStrategy`.

| Knob | Cosine (§9.7) | Plateau (§9.9) | Reason |
|---|---|---|---|
| Post-warmup schedule | half-cosine to 0 | `ReduceLROnPlateau` | Adapts LR to actual loss progress instead of a fixed shape. |
| LR-reduction factor | n/a | **0.5** | Halve LR on plateau (PyTorch default). |
| Patience (epochs no-improvement) | n/a | **5** | Long enough to ride out noise, short enough to react before stalling. |
| Improvement threshold | n/a | **1e-3** | Loss must drop by > 0.001 to count as "going down". |
| Min LR floor | 0 (cosine ends here) | **1e-6** | Avoids dropping into numerical-noise territory. |
| Loss EMA α | n/a | **0.3** | Smooths per-epoch train loss before feeding plateau. |
| Peak LR / warmup / clip / epochs | 3e-4 / 10 / 1.0 / 80 | unchanged | Apples-to-apples vs §9.7 except for the decay phase. |
| Variants | 1 (`vit_mamba3` only) | **all 3** | Self-contained 3-line / 3-bar comparison; ignore §9.6's misleading vit_mamba3. |

Output to `outputs/cifar10_compare_plateau/`. The script auto-emits the three figures via `scripts/plot_cifar10_compare.py`:

- `lr_curves.png` — 3 lines, post-warmup divergence shows when each variant hit a plateau
- `loss_vs_steps.png` — train + test loss vs steps, 3 lines
- `efficiency_comparison.png` — 4-panel bar chart (params / latency / peak mem / s-per-epoch)

**Run command:**

```bash
uv run python scripts/cifar10_compare.py \
    --device cuda --epochs 80 --batch-size 128 --lr 3e-4 \
    --warmup-epochs 10 --grad-clip 1.0 \
    --lr-schedule plateau \
    --plateau-factor 0.5 --plateau-patience 5 \
    --plateau-threshold 1e-3 --plateau-min-lr 1e-6 \
    --plateau-loss-ema-alpha 0.3 \
    --variants cnn vit_attn vit_mamba3 \
    --out outputs/cifar10_compare_plateau --seed 42
```

Estimated wall-clock: cnn 8 min + vit_attn 8 min + vit_mamba3 20 min ≈ **36 min total**.

**Branch decisions (revised, replacing §9.5 verdict):**

- **PASS-plateau** = `test_acc(vit_mamba3) ≥ test_acc(vit_attn) − 2 pp` under plateau scheduling. → both schedule and recipe were confounds; resume `PLAN_mamba3_DA3.md`.
- **FAIL-plateau** = gap remains > 2 pp under both cosine and plateau. → mixing-quality gap is robust to schedule choice; trigger §9.5 FAIL branch (reconsider swap), with §9.7 + §9.9 cosine + plateau as the supporting evidence.
