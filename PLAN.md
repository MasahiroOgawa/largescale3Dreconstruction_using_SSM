# SSM-3D — Visual Quality Fix Plan

Status: the end-to-end pipeline runs and all 42 tests pass, but the four `outputs/*.png` artifacts are visually poor. This file diagnoses why and lays out concrete fixes for the next iteration.

## 1. What the current outputs look like

| File | Observation |
|---|---|
| `feature_pca_view{0,1}.png` | 14×14 near-uniform yellow/green blob. No spatial structure. |
| `depth_view{0,1}.png` | Colorful noise. Does not track the scene (ground/building/sky) at all. |
| `cross_attention.png` | Near-uniform pale wash over the kv image. No localized blob. |
| `seg_overlay_coco{0..2}.png` | Pale warm tint covering most of the image; no mask tracks object boundaries. |

## 2. Diagnostic results

Measured on the current demo configuration (`SSM3DNet(size="small", depth=2)`, random init, 20 self-consistency iters):

```
per-view predicted-depth std  = 0.012   on [0, 1]    ← essentially constant
feature std across tokens     = 1.00    ← values spread, but ...
mean off-diag cosine (patches)= 0.918   ← all 196 patches look 91.8% identical
```

Varying depth (blocks) alone:

```
depth= 2  mean off-diag cosine = 0.918   (collapsed)
depth= 6  mean off-diag cosine = 0.891   (collapsed)
depth=12  mean off-diag cosine = 0.415   (discriminative)
```

Depth variance is ~0.01 on a [0, 1] scale; the visualization then stretches by the 2/98 percentile, amplifying numerical noise into a full turbo colormap. That is why `depth_view*.png` looks dramatic but is pure noise.

## 3. Root causes

1. **Demo uses only `--depth 2` transformer blocks.** ViT-Small ships with 12. With 2 random-init Mamba-3 blocks the token representations barely de-mix from the patch embedding, so every patch looks ~92% the same. Features, depth, cross-attention, and seg all inherit this collapse.

2. **No pretrained weights are loaded.** The approved plan (`/home/mas/.claude/plans/implement-the-code-which-snuggly-bubble.md`) explicitly deferred DA3 weight loading ("attention weight shapes don't match; skipping per the scope choice"). But the *non-attention* DA3/DINOv2 weights (patch embed, LayerNorms, MLPs, register/cls tokens, 2D-RoPE params) do match and can be loaded. Without them, the backbone is random ≠ useful features.

3. **Self-consistency loss is degenerate.** `overfit.py` minimizes `|pred - mean_over_views(pred)|`. A constant output — every pixel = 0.7 — satisfies this loss trivially. The network collapses toward a constant, which is exactly what we measure (std=0.012). The loss going `0.0035 → 0.0000` is therefore *meaningless*.

4. **Feature-PCA image is saved at the patch-grid resolution (14×14).** Correct information, unreadable size. Browsers render it as a 14-pixel blur.

5. **Cross-attention visualization has no normalization.** `attn = L ⊙ (C Bᵀ)` contains signed similarities times positive decay; averaging over heads cancels sign and yields a near-flat map. There is no softmax, no sharpening, no max-over-heads.

6. **Seg overlay uses a binary threshold + `spring` cmap.** With a random-init backbone the logits barely differentiate fg/bg, so the mask is near-all-on or near-all-off. `spring` (magenta→yellow) over a natural image is also hard to see.

## 4. Proposed fixes — do in this order

### Fix 1 — Use a realistic backbone depth in the demo (trivial, biggest visual win)

Drop `--depth 2` from the default. Use the full `depth=12` for ViT-Small. Current run time on CPU is dominated by token count, not block count, so this is affordable.

- `scripts/run_demo.py`: change `ap.add_argument("--depth", type=int, default=2)` → `default=12`.
- `tests/integration/test_overfit_decreases_loss.py`: keep `depth=2` there, it's about loss mechanics not visual quality.

### Fix 2 — Load DA3 DINOv2 non-attention pretrained weights

DA3 ships with a DINOv2-small checkpoint compatible with `patch_size=16`. The attention sub-modules are incompatible with our Mamba-3 swap, but everything else loads cleanly.

- Add `ssm3d.weights.load_dinov2_backbone(backbone, checkpoint_path)` that:
  - loads the state dict,
  - filters keys containing `.attn.` (our Mamba-3 blocks don't match them),
  - calls `backbone.vit.load_state_dict(filtered, strict=False)` and asserts the `missing_keys` set is **only** `.attn.*` parameters.
- Expose `--pretrained path/to/dinov2_vits14.pth` on `run_demo.py` (default: auto-download via `huggingface_hub` if not present).
- Acceptance: mean off-diag patch cosine < 0.5 at depth=12 before any overfit.

### Fix 3 — Replace the self-consistency loss with something non-degenerate

Keep the existing branch (GT-depth L1) for when ETH3D depth is downloaded. For the default no-GT path, swap the variance-collapse loss for one of these, in order of preference:

1. **Add ETH3D depth GT**: download `terrains_rig_depth.7z` (~0.6 GB, within the 2 GB budget). `src/ssm3d/data/eth3d.py` already has the scaffolding. Then `overfit_run` uses the existing `_scale_invariant_l1`.
2. **Photometric reprojection** between two views using predicted depth and pose (ETH3D provides poses). Non-trivial (~80 lines), but the right self-supervised signal.
3. **Anti-collapse regularizer fallback** if neither above is wired yet:
   - primary: `smoothness(pred) + 0.1 * variance_across_views(pred)`
   - penalize low variance: `+ λ · max(0, σ_target − pred.std())` where `σ_target` ≈ 0.1.

### Fix 4 — Upsample feature-PCA to image resolution before saving

`src/ssm3d/viz/feature_pca.py::feature_pca_image` currently returns the raw 14×14 array. Add an `upsample_to: tuple[int, int] | None = None` arg; when set, resize via `PIL.Image.NEAREST` *after* min-max normalization (so PCA math stays at patch resolution but the saved PNG is 224×224).

- `run_demo.py` and the integration test pass `upsample_to=(img_size, img_size)`.

### Fix 5 — Sharpen the cross-attention visualization

In `save_cross_attention_heatmap`:

- take **max** across heads instead of mean, so localized per-head responses survive,
- apply a row-wise softmax to the `CBᵀ` matrix *before* Hadamard with `L` so the row is a probability distribution,
- optionally plot only the top-k% of attention weights (mask the rest to zero in the overlay).

Either change `Mamba3CrossAttention.forward` to return `(softmax(CBᵀ) * L, y)` when `return_attn=True`, or do the softmax inside the viz function — the viz-side fix is less invasive.

### Fix 6 — Seg head: train longer and visualize probability, not binary mask

`seg_head.py::save_seg_overlay`:

- remove the `mask > threshold` hard gate; overlay `prob` directly as a `viridis` heatmap with `alpha * prob`.
- show per-pixel confidence instead of "in/out" so the viewer sees structure even at 60 iters.
- `run_demo.py`: bump default `--seg-iters` 60 → 300 and `num_images` 6 → 15.

### Fix 7 — Add a collapse smoke-check to `run_demo.py`

Before saving any visual, compute and print:

```
feat_cos_mean   (patches): {value:.3f}   (warn if > 0.7)
depth_std (view 0): {value:.4f}          (warn if < 0.02)
cross_attn_row_max: {value:.3f}          (warn if < 2/T_kv)
```

If any warning fires, print `"[WARN] outputs are likely to look flat; see PLAN.md §3"` so future debugging starts from a known place.

## 5. Acceptance criteria after fixes

Run `uv run python scripts/run_demo.py` with defaults. All four should hold:

- `feature_pca_view0.png` is 224×224 and shows visible scene structure (grass / building edges / sky).
- `depth_view0.png` has per-pixel depth std > 0.05 on [0, 1] and tracks the scene layout (closer → one end of colormap).
- `cross_attention.png` shows a **localized** blob near the query patch's correspondent in the kv image (not a global wash).
- `seg_overlay_coco{0,1,2}.png` shows heat concentrated on the annotated instance(s).

Numeric sanity-check target (reported by Fix 7):

```
feat_cos_mean < 0.5
depth_std > 0.05
cross_attn_row_max > 5 * (1 / T_kv)
```

## 6. Algorithm review findings (separate from visual-quality fixes)

Reviewing `/home/mas/proj/study/mamba3_doc/attention/mamba3_attention.tex`
against the implementation surfaced three items:

- **Implementation bug (already fixed):** `build_three_term_mask` had an
  off-by-two index when shifting $\beta$ into the second band. All existing
  tests set $\lambda\equiv 1$ (so $\beta=0$) and therefore never exercised
  this path. Added `test_three_term_mask_matches_direct_recurrence` which
  compares the closed-form mask to an independent direct iteration of the
  three-term recurrence with $\lambda \in (0,1)$ — it catches the off-by-one.
- **Tex clarification (applied):** `mamba3_attention.tex` now states the
  boundary conventions $\beta_1 := 0$ and $\beta_{T+1} := 0$ explicitly,
  and warns that the $\lambda\equiv 1$ sanity test is insufficient.
- **Tex clarification (applied):** added a paragraph on the raster-order
  recency bias of $\bm{L}^{\text{cross}}$ and a bidirectional
  cross-attention mask \eqref{eq:Lcross-bi} that removes this bias.
  This is directly relevant to Fix~5 above: once the bidirectional cross
  mask is adopted, the heat-map in `cross_attention.png` should become
  genuinely localised rather than drifting toward the end of the scan.

## 7. Out of scope (do not do in this iteration)

- Full DA3 training schedule.
- Loading the DA3 depth/ray heads (those expect specific backbone statistics we won't reproduce without training).
- Any GPU-specific kernels or `torch.compile`.
