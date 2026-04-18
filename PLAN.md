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

## 8. Post-Fix diagnosis — outputs are still bad, and the Fix-3 loss made things worse

After applying Fixes 1, 3, 4, 5, 6, 7 (Fix 2 mechanism is in place but no
compatible checkpoint is loaded by default), `outputs/*.png` remains
unusable. Running the same cosine diagnostic reveals that the overfit step
**destroys** the modest feature structure that was present at init:

| stage                                    | feat_cos_mean (patches) |
|------------------------------------------|--------------------------|
| depth=12, random init, real ETH3D image  | **0.584** (modest structure) |
| same, after 15 iters of Fix-3 overfit    | **0.999** (fully collapsed)  |
| same, but backbone frozen during overfit | **0.584** (preserved)        |

Depth-vs-luminance Pearson correlation is `-0.06`: the "predicted depth"
has no relationship to the input image. The depth map is a random
amplification of whatever axis the anti-collapse hinge latched onto.

Root causes newly identified:

1. **Fix-3's anti-collapse hinge `relu(0.1 − pred.std())` back-propagates
   into the backbone.** The cheapest way for the network to satisfy the
   hinge is to re-wire one axis of backbone variance to drive the depth
   head, which inevitably collapses all tokens onto that axis. The
   remedy is to *not* train the backbone during the demo — freeze it,
   train only the head (or skip the demo overfit entirely).

2. **`run_cross_attention_visual` constructs `Mamba3CrossAttention` with
   random weights and never trains it.** The heatmap is driven by the
   decay mask $\bm{L}^{\text{cross}}$'s raster-order shape, not by the
   similarity $C^q (B^{kv})^\top$. With random $B^{kv}$ the similarity
   has zero mean and the mask's corner-weighting dominates — exactly
   what `cross_attention.png` shows (hotspots at image corners, not at
   the query's correspondent point). Making the mask bidirectional in
   Fix 5 only redistributes the artifact, it does not fix it. The fix is
   either to warm-start the cross module by copying backbone B/C/V from
   a DINOv2-equivalent pretrained state, or to use DINOv2's attention
   layer directly for the cross-view viz (the point of the viz is "show
   which kv patch the query attends to"; it doesn't have to run through
   Mamba-3).

3. **Feature PCA at `feat_cos_mean = 0.58` is blocky and uninformative
   even after `upsample_to`.** Nearest-neighbor upsampling of a 14×14
   grid keeps the visible blockiness. The information bottleneck is 196
   tokens — we cannot visualize sub-patch structure this way. Upgrading
   to `patch_size=14` on a 224 image yields 16×16 = 256 tokens (tiny
   improvement); the real gain requires either smaller patches
   (patch_size=8 → 28×28 = 784 tokens, 4× compute) or bilinear
   upsampling *of the PCA output* (smoother, but no extra information).

4. **Mamba-3 blocks are random-init, and Fix 2 does not fix this.**
   `load_dinov2_backbone` correctly loads patch_embed, norms, MLPs,
   cls/register/mask tokens, and RoPE freqs — but the attention
   parameters (`B, C, V, Δ, A, λ` projections) have no counterpart in
   DINOv2's qkv attention, so 12 attention blocks remain random. Every
   MLP output feeds the next block's random attention, which destroys
   the DINOv2 MLP structure by block 2 or 3. **Partial-load Fix 2 alone
   is not sufficient.**

5. **Architecture mismatch blocks even partial Fix 2.** DINOv2-small
   ships with `patch_size=14`. Our demo uses `patch_size=16`. The
   `patch_embed.proj.weight` shape differs (384×3×16×16 vs 384×3×14×14),
   so the single most important weight — the embedding that turns pixels
   into tokens — is always shape-mismatched and silently skipped.

## 9. Fix plan v2 — what to do next, in order of impact

### 9a. Freeze the backbone during the demo overfit (≈ 5 lines, biggest immediate win)

`overfit.py` currently trains everything. Change `scripts/run_demo.py` to
either skip `overfit_run` entirely, or (preferred) call it with
`trainable="head"` and pass `net.depth_head.parameters()` to AdamW.
Expected effect: `feat_cos_mean` stays at 0.58 instead of collapsing to
0.999, and feature-PCA/cross-attn viz show the same structure-at-init
that the network produced before training. This is a negative-impact
*removal*, not a positive gain — the overfit step was hurting us.

Acceptance: `feat_cos_mean` after the demo ≤ 0.60 (equal to init).

### 9b. Match DINOv2's patch size (`patch_size=14`, `img_size=224`)

Change the demo's `patch_size=16` default to `14`. With 16×16 = 256
tokens we stay cheap on CPU, the patch grid resolution improves 14%, and
— critically — `patch_embed.proj.weight` now matches DINOv2-small's
shipped shape. This unblocks loading DINOv2 weights as-is.

Follow-up: update any tests that hard-code `patch_size=16` (at least
`test_swap_matches_signature.py` does not).

### 9c. Actually load a DINOv2-small checkpoint in the demo (Fix 2 activated)

The code path exists. Add `scripts/download_dinov2.py` that fetches
`dinov2_vits14_pretrain.pth` via `torch.hub.download_url_to_file` from
`https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`
(≈ 84 MB), and wire `--pretrained` in `run_demo.py` to auto-download on
first run. Confirm via the unit test that `counters["loaded"]` is ≥ 50.

Acceptance: after `uv run python scripts/run_demo.py`, `feat_cos_mean`
drops below **0.5** and feature-PCA tracks visible scene boundaries
(grass / rocks / sky). This is what the original §5 target was aiming
for.

### 9d. Warm-start Mamba-3 attention from DINOv2 qkv

This is the fix that actually makes `feat_cos_mean < 0.3` and produces
usable depth/cross-attn outputs without any end-to-end training.

The Mamba-3 $B, C$ projections are rank-N approximations of keys and
queries; the $V$ projection is literally a value projection. Initialise:

- `B_proj.weight ← K_proj.weight[:state_dim]` (first N rows of DINOv2
  K-projection, per head).
- `C_proj.weight ← Q_proj.weight[:state_dim]`.
- `V_proj.weight ← V_proj.weight` (unchanged).
- `A_log, Δ, λ` get small-random init so that on day 0 the SSD output is
  close to $(Q K^\top) V$ with mild recency bias.

This is an explicit cast from softmax attention to SSD attention;
accuracy is approximate but the representation is "DINOv2-like" from
step 0 instead of random. Add `ssm3d.weights.warm_start_mamba3_from_qkv`.

Acceptance: `feat_cos_mean` < 0.30, and `feature_pca_view0.png` clearly
tracks the scene layout.

### 9e. Honest visualisations where the demo cannot predict without training

- **Depth map**: with no depth GT and a random depth head, stop
  pretending the output is "predicted depth". Either:
  (a) remove the depth visualization, or
  (b) re-label it as "depth-head activation" in the filename and
      caption, so the viewer knows it's the head's untrained response,
      or
  (c) replace the depth head with a simple "feature magnitude" or
      "DINOv2 first-PCA channel" map — visually plausible and honest.
- **Cross-attention**: replace the random `Mamba3CrossAttention` in
  `run_cross_attention_visual` with a direct query→kv similarity using
  backbone features (cosine-sim of `feats[query_index]` against all kv
  tokens, softmax over kv). This *is* the quantity we wanted to
  visualize. The Mamba-3 cross module itself is proven out by the
  existing unit tests; it does not need to be in the demo.

### 9f. Regenerate the PCA-depth-attn outputs at native image resolution via bilinear

Change `save_feature_pca`'s upsample from `NEAREST` to `BILINEAR`. PCA
output is continuous; `NEAREST` artificially block-quantizes it. This
is a one-line fix in `src/ssm3d/viz/feature_pca.py`.

## 10. Revised acceptance criteria (after §9 fixes)

- `feat_cos_mean` (post-demo) < 0.5 (random init is already 0.58; the
  loss should not make it worse). With §9d warm-start: < 0.3.
- `feature_pca_view0.png` visibly traces grass / building edges / sky
  on the ETH3D terrain image.
- `cross_attention.png` shows a localised blob **at the correspondent
  scene patch** of the query, not at the image corners.
- `depth_view0.png` either (a) correlates with image luminance or
  structure at |ρ| > 0.3, or (b) is honestly re-labeled as an
  activation map.
- `seg_overlay_coco*.png` with backbone frozen and 500+ iters head
  training shows heat concentrated on annotated instances.
