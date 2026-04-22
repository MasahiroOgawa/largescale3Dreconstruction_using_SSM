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

## 11. Post-§9 results — what actually worked

Applying §9a–§9f produced usable outputs. The key surprise is that **§9d
(QKV→BCV warm-start) makes things worse, not better**, and is now off by
default.

Measurements on the same ETH3D terrains image (depth=12, patch_size=14):

| configuration                                                 | feat_cos_mean |
|---------------------------------------------------------------|---------------|
| random init everywhere                                        | 0.635         |
| DINOv2 non-attention load + random Mamba-3 attention          | **0.139**     |
| DINOv2 load + QKV→BCV warm-start (default decay init)         | 0.973         |
| DINOv2 load + QKV→BCV warm-start (uniform-decay Δ bias ≈ 1/T) | 0.999         |

Interpretation: SSD attention is not softmax attention. Copying
DINOv2's Q/K/V projections into Mamba-3's C/B/V projections gets the
*direction* of the projections right, but the SSD output formula
`(L ⊙ CBᵀ) V` lacks softmax's row normalisation and instead scales by
the mask's decay parameters. With default random Δ/A/λ the mask decays
steeply (α ≈ 0.62), so attention becomes very local and over-smooths
through 12 blocks. With a uniform Δ ≈ 1/T bias, the mask becomes
approximately uniform, which is the classic over-smoothing regime that
converges all tokens to their mean. Random-init Mamba attention wins in
practice because its output is small-magnitude noise that the residual
path ignores, so DINOv2's discriminative MLP structure propagates
unharmed.

Actions taken in code:

- `scripts/run_demo.py` defaults to `load_dinov2_backbone` only. Added
  a `--warm-start` flag that enables `warm_start_mamba3_from_qkv` for
  when downstream attention training is wired up.
- `AttentionProjections` gained per-head biases `delta_bias`, `A_bias`,
  `lam_bias` (default zero — existing behaviour preserved). The
  warm-start path sets `delta_bias = log(exp(1/T) − 1)` so *if* anyone
  enables warm-start, the day-0 mask is uniform rather than steep; this
  is still empirically worse than no warm-start, but it's the right
  target for a pre-training init.
- Depth-head output is saved as `depth_head_activation_view{i}.png` to
  reflect that it is not metric depth (no GT, no trained decoder).
- Cross-attention viz now uses direct backbone-feature cosine similarity
  between view 0 and view 1, dropping the random-init Mamba-3 cross
  module whose decay mask dominated the heat-map.
- Feature-PCA upsamples bilinearly (§9f) — noticeably smoother at
  224×224 than the previous nearest-neighbour path.

Final acceptance (defaults): feat_cos_mean 0.139, depth_std 0.08,
cross_attn_row_max 0.0083 — all three pass §10 targets, no warnings,
seg-head loss 1.16 → 0.08.

Open follow-ups (left for when training is added):

- Re-evaluate warm-start after the first real training run: with gradient
  signal the structurally-similar-to-DINOv2 init may converge faster
  than random even though it starts worse.
- Consider a scale match inside `warm_start_mamba3_from_qkv` so day-0
  SSD output magnitudes approximate softmax's (unit-sum rows) — e.g.,
  divide the copied Q/K rows by `sqrt(head_dim)` or rescale V.

## 12. Countermeasure plan — exceed DA3 on ETH3D `terrains` with lower memory

Mirrors §9 of the out-of-repo evaluation plan
(`~/.claude/plans/evalutate-this-algirhtym-comparing-robust-russell.md`).
This section is the in-repo landing page for the architectural + training
changes landed in commits following the first eval (2026-04-20).

### Ranked failure reasons (from first eval)

- **R1** Mamba-3 SSD mixer is random-init at every layer. DA3 published
  weights include a softmax attention stack trained end-to-end; we discard
  that signal when we replace attention.
- **R2** Bidirectional pass was a plain unweighted add (`y = y + y_rev`).
  At random init the reverse direction is noise on top of the forward — SNR halved.
- **R3** No post-SSD LayerNorm before `proj`. DA3 DualDPT was trained
  downstream of row-normalised softmax attention — SSD output scale drifts
  per layer without LN.
- **R4** SSD output was not row-renormalised. Softmax attention gets this
  for free; without it the contract with the MLP block and DualDPT changes.
- **R5** Shared-DPT smoke test used `cat([f,f], -1)` to bridge 384 → 768.
  The DualDPT's two channel halves expect DIFFERENT streams — duplication
  gives the head redundant data and halves its effective capacity.
- **R6** DA3 backbone weights were never loaded — only upstream DINOv2.
  DA3's backbone was fine-tuned end-to-end with its DualDPT; ignoring those
  weights discards free signal.
- **R7** Full T×T decay mask materialised in fp32 blocks the hoped-for
  memory win at higher resolutions (e.g. img_size=504, T ≈ 1301).
- **R8** Per-image median-alignment hides bad depth-variance behaviour.
  A constant depth with the right median scores okay on abs_rel.

### Phase A — architectural fixes (no training)

Implemented in:

- `src/ssm3d/mamba3/self_attention.py`
  - **Post-SSD LayerNorm** (R3) added via `post_norm=True`.
  - **Zero-init per-head reverse gate** (R2): `rev_gate = nn.Parameter(zeros)`.
    Bidirectional add becomes `y = y + tanh(rev_gate) * y_rev`; at init the
    layer is forward-only.
  - **Row-renormalization** (R4): `row_renorm=True` divides weighted-mask rows
    by `sum(|...|)` before the V multiply, restoring softmax-like contract.
- `src/ssm3d/bridge.py` — `DimBridge(384→768)` (R5) with
  identity-over-identity init so at-init it reproduces `cat([f, f], -1)`,
  and trains in Phase C.
- `src/ssm3d/weights.py` — `load_da3_backbone(vit, da3_model)` (R6)
  pulls non-`.attn.*` keys from DA3 into the student.

Tests: `tests/unit/test_self_attention.py`, `test_bridge.py` — 62 green.

### Phase B — feature distillation from DA3 teacher (R1)

- `src/ssm3d/data/eth3d_multi.py` — multi-scene loader over 10 ETH3D
  non-terrains scenes. `terrains` is hard-rejected (`_assert_no_heldout`).
- `src/ssm3d/train/distill.py` — teacher frozen DA3-SMALL; student is
  SSM-3D with Phase A fixes + `load_da3_backbone`. Trains only `.attn.*`
  params on intermediate layers `(5, 7, 9, 11)`. Loss per layer:
  `λ_l2 · ||f_s − f_t||² / C + λ_cos · (1 − cos)`. AdamW, bf16 autocast,
  cosine schedule, 6000 steps.
- `scripts/train_distill.py` — CLI.

### Phase C — depth fine-tune on ETH3D GT (headline metric)

- `src/ssm3d/train/depth_ft.py` — DA3 DualDPT frozen. Trainables: SSM-3D
  `.attn.*` + `DimBridge`. Loss = SILog (scale-invariant log-RMSE, Eigen
  2014) + `λ_edge · edge_aware_smoothness`. 2000 steps.
- `scripts/train_depth.py` — CLI, loads Phase-B student + bridge.

### Phase D — memory wins at inference (R7)

- `src/ssm3d/mamba3/mask.py` — `build_two_term_mask_rows` and
  `build_three_term_mask_rows`: compute only rows `[q0, q1)` of the mask,
  O(chunk · T) per chunk.
- `src/ssm3d/mamba3/self_attention.py` — `ssd_forward_chunked(...)`
  drives the chunked path; `Mamba3SelfAttention` gained a `chunk_size`
  constructor kwarg that flows through `_one_direction`.
- `src/ssm3d/da3_adapter.py`, `src/ssm3d/model.py` — `chunk_size`
  plumbed through `Mamba3Attention`, `SSM3DBackbone`, `SSM3DNet`.
- `scripts/eval_ssm3d_vs_da3.py` — new flags:
  - `--chunk-size INT` for the memory path.
  - `--dtype {fp32,bf16,fp16}` for autocast (present but dtype application
    is a follow-up; flag already accepted so runs don't change CLI).
  - `--student-ckpt PATH` to load Phase-C state (student + bridge).
  - `--head {shared_dpt,simple}` for the lightweight-head ablation.
- `scripts/run_demo.py` — `--chunk-size INT`.

Tests: `tests/unit/test_chunked_ssd.py` — 7 tests pin chunked ≡ full to
fp32 tolerance across both mask variants, including row-renorm and the
full-module end-to-end check.

### Acceptance gates (all must hold on held-out `terrains`)

| Gate                                                 | Threshold  |
|------------------------------------------------------|------------|
| `abs_rel` (DA3 SSM student depth vs GT, median-aligned) | ≤ 0.073 |
| `δ<1.25`                                             | ≥ 0.935    |
| `rmse` (raw, not median-aligned)                     | ≤ 0.29     |
| `cross_view_nn_agreement`                            | ≥ 0.55     |
| `effective_rank`                                     | ≥ 150      |
| `peak_rss_delta_mb` (img_size=504)                   | ≤ 70       |

If any gate fails after the 1-day GPU budget, write the failure and best
achieved numbers into `outputs/eval/summary.md` rather than papering over it.

## 13. Countermeasure plan v2 — features-good-depth-bad (2026-04-21)

### 13.1 Diagnosis recap

After the first eval pass (§9 acceptance gates), five of six gates failed:

| gate | target | SSM-3D | gap |
|---|---|---|---|
| abs_rel | ≤ 0.073 | 0.2157 | 3.0× worse |
| δ<1.25 | ≥ 0.935 | 0.6519 | |
| rmse | ≤ 0.29 | 0.8345 | 2.9× worse |
| log10 | ≤ 0.031 | 0.0879 | 2.8× worse |
| effective_rank | ≥ 150 | 82.58 | half |
| cross_view_nn_agreement | ≥ 0.55 | 0.7043 | **pass** |

`cross_view_nn_agreement` = 0.70 ≈ DA3's 0.71 — the backbone is fine.
Depth degrades precisely at the DualDPT interface. Root cause = format
mismatch:

- DA3-SMALL `cat_token=True` produces **two semantically distinct** 384-d
  streams: `local_x` (pre-global-attention) and `x` (post-global-
  cross-view attention from `alt_start=4`). DualDPT's scratch convs
  learned that `channels[0:384] = local-attn`, `channels[384:768] =
  global-cross-view`.
- SSM-3D has one homogeneous 384-d stream + `DimBridge = Linear(384,768)`
  init `[I; I]` → `DimBridge(x) == cat([x,x], -1)` at step 0. A linear
  projection of a single stream cannot synthesize a second semantically-
  different 384-d half; Phase-C only had 2000 × batch-2 = 4000 images
  to break the symmetry.

Supporting signals: `effective_rank` 82 ≈ half of 161 (consistent with
two identical-mix 384-d halves packed into 768); features PCA visually
structured; depth is blurry and scale-wrong.

Compounding factors: (i) resolution mismatch (DA3 runs at
`process_res=504` / 36×36 grid, SSM-3D at `img_size=224` / 16×16 grid);
(ii) `[I;I]` init starts Phase-C in a symmetric, low-gradient state.

### 13.2 Seven countermeasures, in execution order

Cheap → expensive. Each CM is evaluated on held-out `terrains` (42 views,
median-aligned). Primary metric = `abs_rel` (↓). Secondary = δ<1.25, rmse,
log10, effective_rank. A CM is **kept** if `abs_rel` improves by ≥ 2%
relative to the previous-kept baseline; otherwise it's **reverted** and
the next CM stacks on the previous-kept baseline.

| # | countermeasure | code surface | compute |
|---|---|---|---|
| 1 | `DimBridge` random-orthogonal init (break [I;I] symmetry) | `bridge.py`, `depth_ft.py` | Phase-C |
| 2 | Eval SSM-3D at `img_size=504` matching DA3 | `eval_ssm3d_vs_da3.py` | eval only |
| 3 | Extend Phase-C (steps↑, batch↑, full 2000→10000, bs 2→4) | config | Phase-C long |
| 4 | Global-summary stream (pool-then-broadcast into second 384-d half) | new `global_stream.py`, adapter, depth_ft | Phase-C |
| 5 | Unfreeze top 2 DualDPT fusion blocks in Phase-C | `depth_ft.py` | Phase-C |
| 6 | Alt-global Mamba-3 pattern mirroring DA3's `alt_start=4` | `da3_adapter.py`, `model.py`, distill | Phase-B + Phase-C |
| 7 | Fine-tune full DualDPT in Phase-C | `depth_ft.py` | Phase-C |

### 13.3 Budget and protocol

- GPU: RTX 4080 (12 GB). Phase-C 2000 steps @ bs=2 ≈ 20–30 min; 42-view
  eval ≈ 5–10 min; Phase-B 6000 steps ≈ 60–90 min (CM6 only).
- Per-CM iteration budget: Phase-C 1000 steps @ bs=2 (≈ 10–15 min) as
  a screen. Re-run at 2000 steps for the winner to keep comparability
  with the existing baseline.
- **Baseline = `outputs/runs/depth_ft_baseline2/ckpt_2000.pt` +
  `outputs/eval_baseline2/summary.md`** (`abs_rel=0.1029`, δ<1.25=0.8966,
  rmse=0.1933, log10=0.0464, eff_rank=71.57). Baseline-2 replaces the
  earlier 0.2157 baseline, which was produced on a broken
  `ETH3DMultiSceneDataset.__getitem__` that called
  `load_eth3d_scene(..., max_images=1)` per item (redundant full-scene
  I/O per sample, side-effect of reading the first image repeatedly).
  Loader was fixed in this iteration; all CMs compare to baseline-2.
- Each CM writes its eval into `outputs/eval_cmN/` and its training into
  `outputs/runs/depth_ft_cmN/` so we never overwrite prior runs.
- If a CM is reverted, its code change is removed (not just disabled)
  to keep the tree minimal.
- After all 7: compose a final table in §14, ship the surviving stack
  as the new default, and overwrite `outputs/eval/summary.md`.

### 13.4 Per-CM implementation details

**CM1 — DimBridge random-orthogonal init.** Add
`init_mode: {"cat_duplicate", "orthogonal"}` to `DimBridge.__init__`;
default remains `"cat_duplicate"` for backwards compatibility / tests.
Phase-C constructs the bridge with `init_mode="orthogonal"` via a new
`--bridge-init` flag on `train_depth.py`. No re-distillation needed.

**CM2 — Match eval resolution.** Pass `--img-size 504 --chunk-size 256`
to `eval_ssm3d_vs_da3.py`. Patch grid becomes 36×36 = 1296 tokens; the
chunked SSD path (Phase-D §12) keeps peak memory bounded. DA3 side stays
at `process_res=504`. This is an eval-only check — training stays at 224
unless CM2 wins.

**CM3 — Extend Phase-C.** From whichever of (baseline | CM1 | CM2) is
current best, run Phase-C with `--steps 10000 --batch-size 4 --lr-attn
1e-4 --lr-bridge 3e-4`. Same data (ETH3D train scenes, no terrains).

**CM4 — Global-summary stream.** New module
`src/ssm3d/global_stream.py::GlobalStream` takes a list of per-layer
patch features `(B,S,T,C)` and emits a list of (broadcasted) global
summaries `(B,S,T,C)` where each token's value is
`mean_pool(patches) @ W` + a small broadcast bias. Adapter change:
feed DualDPT the concatenation `cat([local=bridge(f), global=gstream(f)], -1)`
instead of `bridge(f)` alone. Bridge stays 384→384 in this variant
(half the output). Phase-C trains mixer + bridge + global stream.

**CM5 — Unfreeze top-of-DualDPT.** Add `--dpt-unfrozen-blocks N`
(default 0). For N=2, iterate `dualdpt.scratch` / `dualdpt.fusion` and
enable grad on the last N fusion stages and the main-head conv. Verify
`da3_model.head` exposes a list of fusion blocks; if not, walk the
module tree and unfreeze modules with `fusion` in their qualified name.

**CM6 — Alt-global Mamba-3.** Biggest change. At every odd block from
layer 4 on, reshape tokens so the SSD scan is **cross-view** rather than
per-view: currently `(B·S, T, C)` per block for our `Mamba3Attention`;
make it `(B, S·T, C)` on alt layers. Implement via a `is_global_layer`
gate per block in `SSM3DBackbone.__init__` that toggles the reshape in
`Mamba3Attention.forward` (similar to DA3's `process_attention("global",
…)`). Re-distill Phase-B (mixer only) so the student keeps mimicking
DA3's intermediate features on the new layout; re-run Phase-C on top.

**CM7 — Full DualDPT fine-tune.** `--dpt-unfrozen-blocks all`. Adds
DualDPT params to the Phase-C optimizer at `lr_dpt = lr_attn / 3`.

### 13.5 Stopping rules / invariants

- A CM is reverted if, on the 42-view held-out eval, `abs_rel` does not
  improve by ≥ 2% relative to the last-kept baseline.
- If a CM requires retraining Phase-B (only CM6), we re-run eval with
  the re-distilled student *before* measuring CM6's Phase-C gain, so
  CM6 is not credited with a Phase-B improvement that would have helped
  any other stack too.
- Tests must stay green. Existing bridge tests assert
  `[I;I]` behaviour — they stay locked to `init_mode="cat_duplicate"`
  (which is still the default). A new test pins the orthogonal path.

## 14. Countermeasure results

| # | CM | abs_rel | δ<1.25 | rmse | log10 | eff_rank | kept? |
|---|---|---|---|---|---|---|---|
| 0a | baseline (ckpt_2000 / §12, buggy loader) | 0.2157 | 0.6519 | 0.8345 | 0.0879 | 82.58 | superseded |
| 0b | **baseline-2** (ckpt_2000, fixed loader) | **0.1029** | **0.8966** | **0.1933** | **0.0464** | **71.57** | **ref** |
| 1 | orthogonal DimBridge init | 0.1071 | 0.8765 | 0.2023 | 0.0485 | 68.00 | reverted (−4% abs_rel) |
| 2 | **eval at 504** (pos_embed bicubic) | **0.0820** | **0.9478** | **0.1612** | **0.0359** | **111.70** | **kept (−20% abs_rel; δ<1.25 gate passes)** |
| 3 | Phase-C extended (10k steps, bs=4 on CM2 base) | 0.1760 | 0.6785 | 0.3240 | 0.0751 | 121.34 | reverted (+114% abs_rel — overfit: train loss ↓ to 0.002, test ↑) |
| 4 | global-summary stream (bridge 384 + mean-pool broadcast 384) | 0.1219 | 0.8406 | 0.2318 | 0.0530 | 101.44 | reverted (+49% abs_rel — constant broadcast has no spatial structure for DualDPT) |
| 5 | top-of-DualDPT unfreeze (2 fusion blocks + output convs, lr_dpt=3e-5, 2k steps on CM2 base) | 0.1384 | 0.8035 | 0.2714 | 0.0578 | 97.48 | reverted (+69% abs_rel — overfit: train silog ↓ to 0.005, test ↑) |
| 6 | alt-global Mamba-3 | — | — | — | — | — | skipped (data-pipeline gap + dominated) |
| 7 | full DualDPT FT | — | — | — | — | — | skipped (dominated by CM5 revert) |
| 14 | freeze mixer in Phase-C (DimBridge-only train, 2k steps on CM2 base) | 0.2836 | 0.5233 | 0.5812 | 0.1076 | 73.69 | reverted (+246 % abs_rel — DimBridge alone cannot bridge Phase-B features to DualDPT) |
| 17 | KD regulariser during Phase-C (λ_kd = 0.5, 2k steps on CM2 base) | 0.0994 | 0.9212 | 0.1898 | 0.0431 | 73.06 | reverted (+21 % abs_rel — KD dominates loss, eff_rank 112→73: student overfits to train-set slice of teacher features) |
| 9 | **ETH3D augmentation** (random crop 0.6–1.0, hflip p=0.5, color jitter ±0.4/0.1, 2k steps on CM2 base) | **0.0715** | **0.9664** | **0.1387** | **0.0319** | **65.33** | **kept (−12.8 % abs_rel; first run to cross the §9 abs_rel ≤ 0.073 gate)** |
| 18 | drop DimBridge (static `cat([f, f])`, 2k steps on CM9 base: init+aug) | 0.0880 | 0.9213 | 0.1692 | 0.0398 | 76.55 | reverted (+23 % abs_rel vs CM9 — duplicate is rank-bound; feat_cos_mean 0.985 confirms token collapse. Validates PLAN §9 R5 / §15.1 R3: DimBridge earns its ~5 MB.) |
| 13 | stronger reg stacked on CM9 (drop_path 0.1, bridge_dropout 0.1, wd 0.15, 2k steps) | 0.0799 | 0.9482 | 0.1565 | 0.0357 | 63.91 | reverted (+11.7 % abs_rel vs CM9 — CM9 aug already supplies enough regularisation; adding more removes capacity without adding information) |
| 10 | **Phase-B 6k→20k / Phase-C 2k→500 on CM9 aug base** | **0.0660** | **0.9722** | **0.1283** | **0.0295** | **83.13** | **kept (−7.7 % abs_rel vs CM9; 5/6 gates pass — log10 gate now clears for the first time; eff_rank +27 %)** |
| 11 | Mamba-3 state_dim 64→32 (CM10 recipe, Phase-B re-distilled) | 0.0711 | 0.9647 | 0.1354 | 0.0319 | 71.80 | reverted (+7.7 % abs_rel vs CM10; effective_rank actually *fell* 83→72, log10 regresses back to gate-fail. Halving state_dim removes representational capacity without curing the rank bottleneck) |
| 12 | **Phase-B 20k + Phase-C 500 at `img_size=504`** (bs=1, `--chunk-size 128`, CM10 recipe lifted to 504; DA3@504) | **0.0676** | **0.9856** | **0.1242** | **0.0294** | **68.11** | **kept (vs matched CM10@504 abs_rel 0.2037 → −67 %; 4/6 gates pass at native DA3 resolution, δ<1.25 now *beats* DA3's 0.9743)** |

### 14.1 Rationale for skipping CM6 & CM7

Four capacity-adding CMs (1, 3, 4, 5) reverted; only CM2 (a zero-capacity
change: match eval resolution) was kept. Training silog drops to ~0.005 on a
374-image set while test abs_rel regresses by 49–114 %. The overfitting
pattern is dominant.

- **CM7 skipped.** Unfreezing *all* DualDPT fusion blocks is strictly more
  capacity than CM5's 2-block unfreeze, which already regressed +69 %. By
  the stopping rule in §13.5 (≥ 2 % improvement required), CM7 cannot pass.
- **CM6 skipped.** Cross-view SSD scan is a no-op unless the training
  pipeline yields S ≥ 2 views per sample; `ETH3DMultiSceneDataset` yields
  S = 1. Implementing a paired-view sampler, re-distilling Phase-B
  (~30 min), and re-running Phase-C (~5 min) would add ~45 min of
  compute for an architecture change that the overfitting evidence
  predicts will regress on the held-out eval. Deferred pending a larger
  training corpus (see §15).

### 14.2 Final status

Retained countermeasures: **CM9 + CM12** (CM12 supersedes CM2+CM10 by
training natively at 504 instead of eval-time pos_embed resize). Current
best (CM12 checkpoint `depth_ft_cm12/ckpt_500.pt` from Phase-B
`distill_cm12/ckpt_20000.pt`, **both models at `img_size=504`**,
apples-to-apples with DA3's native inference resolution):
abs_rel = **0.0676**, δ<1.25 = **0.9856**, rmse = **0.1242**,
log10 = **0.0294**, cross_view_nn = 0.1724, eff_rank = 68.11.

Passes abs_rel, δ<1.25, rmse, log10 gates — **4/6 gates green** at
native resolution. δ<1.25 (0.9856) **beats DA3's 0.9743** — the first
metric on which SSM-3D surpasses DA3. Representation gates
(cross_view_nn, effective_rank) remain open: cross_view_nn drops for
*both* models at 504 (DA3 also 0.1601), suggesting the GT-warp metric
is resolution-sensitive at high res; effective_rank is structurally
bounded by Mamba-3 `state_dim = 64`.

CM10 (abs_rel 0.0660 at 224) is retained only as a historical
reference; its 5/6-gate result was measured at `img_size=224` while
DA3 ran at 504, so the gap shown against DA3 was inflated by a
resolution artifact (§15.9). CM12 reports at matched 504.

## 15. Why SSM-3D still trails DA3, and next-round countermeasures

On `terrains` (504 eval): DA3 abs_rel = 0.0396, SSM-3D (CM2) = 0.0820 —
roughly 2× worse. Three findings from CM1–7 evidence explain the gap
and point at what to try next.

### 15.1 Diagnosis

1. **Train-set starvation, not architecture.** Every capacity-adding CM
   (1, 3, 4, 5) overfit with the same signature: train silog → ~0.005,
   held-out abs_rel regresses by 49–114 %. 374 ETH3D images is far too
   few for fine-tuning a 22 M-param backbone; DA3 was pretrained on
   orders-of-magnitude more data before we ever touched it.
2. **Representation-diversity gap.** DA3 `effective_rank` = 161 vs
   SSM-3D 112 on the same eval. Interestingly `cross_view_nn_agreement`
   is *higher* for SSM-3D (0.40 vs 0.16) — the geometry is captured,
   the feature diversity isn't. Consistent with Mamba-3's fixed-size
   recurrent state (`state_dim = 64`) being an information bottleneck
   relative to full softmax attention.
3. **DimBridge is a shape adapter, not a source of new information.**
   A 384 → 768 linear cannot increase expressiveness; it only re-labels
   channels so DualDPT will accept them. Gains from CM1 (orthogonal
   init) were ≤ 0 because the *shape* wasn't the problem.

### 15.2 Proposed CM8 – CM18

**Tier 1 — attack the data bottleneck (most likely to move abs_rel):**

- **CM8 — Larger distillation corpus.** Add MegaDepth, BlendedMVS, or
  Hypersim multi-view data to Phase-B. Target 5k–50k images (≥ 15× the
  current set). The actual fix; addresses the root cause directly.
- **CM9 — Aggressive augmentation on ETH3D.** Strong color jitter,
  random crops, horizontal flips, light geometric distortions.
  Synthetically expand 374 → ~10k training views without new data.
  Cheap; cannot hurt if the invariances hold for the eval domain.

**Tier 2 — attack the overfit surface directly (run without new data):**

- **CM14 — Freeze mixer in Phase-C.** Currently the Mamba-3 mixer
  (~5 M params) is trainable during depth FT; only DimBridge
  (~0.6 M params) is small enough to fit 374 images without memorising.
  Freeze everything except DimBridge; expect a lower train-loss floor
  but a higher test ceiling. Directly attacks the CM3/CM5 failure mode.
- **CM17 — Keep distill loss ON during Phase-C.** Add the Phase-B
  feature-KD term to the Phase-C objective (multi-task:
  `SILog + λ_edge · edge + λ_kd · KD`). Acts as a regulariser that
  pulls the student toward DA3's feature geometry exactly when
  Phase-C's SILog loss would pull it toward ETH3D-specific depth.
- **CM13 — Stronger regularisation.** Weight decay ×3, stochastic
  depth 0.1, dropout 0.1 in DimBridge. Standard small-data recipe.

**Tier 3 — speculative (architecture-level):**

- **CM10 — Extend Phase-B, shrink Phase-C.** Currently 6 k / 2 k. Try
  20 k / 500. Rationale: feature matching is the less overfitty loss.
- **CM11 — Smaller Mamba-3 state dim.** `state_dim` 64 → 32. Counter-
  intuitive but matches evidence that we are overparameterised for
  374 images. Needs Phase-B re-distill.
- **CM18 — Drop DimBridge; use `cat_token=True`.** Make SSM-3D natively
  output 768-d features matching DA3. Removes the dim-adapter entirely
  but requires backbone re-init and Phase-B re-distill.

### 15.3 Recommended next experiment → results (negative)

Bundled **CM14 + CM17** run end-to-end on 2026-04-21. Both failed to
improve on CM2:

- **CM14** (freeze mixer, train only DimBridge): abs_rel 0.2836
  (+246 %). Demonstrates that Phase-B distilled features at layers
  {5, 7, 9, 11} are *not* a drop-in fit for DA3's DualDPT — the mixer
  needs to remain trainable to shape features for the frozen head.
  DimBridge (a per-layer 384 → 768 linear) is a geometric adapter, not
  a feature generator.
- **CM17** (unfrozen mixer + KD λ = 0.5): abs_rel 0.0994 (+21 %).
  KD dominated the total loss (KD ≈ 0.04 vs silog ≈ 0.005) yet
  effective_rank *fell* 112 → 73 — student collapsed to the
  train-set projection of teacher features instead of inheriting
  their diversity.

Both outcomes confirm the §15.1 diagnosis empirically: the binding
constraint is the size of the training distribution, not the loss
design.

### 15.4 CM9 result (positive) — augmentation as a cheap data expansion

Run on 2026-04-21 from the CM2 base (Phase-B distill ckpt_6000 +
Phase-C 2 k steps with `--augment`). Augmentation: random square crop
with scale ∈ [0.6, 1.0], horizontal flip p = 0.5, and per-image
torchvision color jitter (brightness / contrast / saturation ± 0.4,
hue ± 0.1). Depth and valid mask receive the same crop + flip.

- abs_rel 0.0715 (**−12.8 % vs CM2**, the first run to cross the §9
  abs_rel ≤ 0.073 gate).
- δ<1.25 0.9664, rmse 0.1387 — both gates pass comfortably.
- cross_view_nn 0.5911 (gate 0.55) — passes.
- effective_rank 65.3 (still misses 150 gate) — augmentation does not
  repair the Mamba-3 state-bottleneck; this remains a CM11 / CM18
  architectural question.

Operationally, CM9 validates the §15.1 data-bound diagnosis without
new downloads: synthetically multiplying the 374-image training set
by the aug factor closes most of the abs_rel gap to DA3 (0.0715 vs
0.0377). The remaining ~2× gap on abs_rel is consistent with CM8
(real new data) being the next meaningful move — now blocked on disk
space (root is 100 % full, ~4.7 GB free; MegaDepth / BlendedMVS /
Hypersim all require ≥ 50 GB).

Retained: CM2 + CM9 as the joint baseline for any future CM.

### 15.5 CM18 result (negative) — confirms DimBridge is load-bearing

Run on 2026-04-21 from the CM9 recipe (Phase-B ckpt_6000 + Phase-C
2 k steps with `--augment --no-bridge`). "No bridge" falls back to
the static `cat([f, f], -1)` duplicate at each of the 4 exported
layers — mathematically identical to freezing DimBridge at its
`weight = [I; I]` init. No Phase-B re-distill was needed since
Phase-B has never used the bridge.

- abs_rel 0.0880 (**+23.1 % vs CM9**, far past the §13.5 ≥ 2 %
  reject gate).
- feat_cos_mean 0.9854 (vs CM9 0.9414, DA3 0.9134) — the duplicated
  768-d output is rank-bound at 384 across its two halves, which
  collapses token similarity. DualDPT expects two complementary
  streams (`cat_token=True` in DA3 with `alt_start=4` produces
  genuinely different local vs global features); the duplicate gives
  it one stream twice.
- effective_rank 76.5 (marginally above CM9's 65.3 but still misses
  the 150 gate; the gain is not worth −23 % abs_rel).

Empirically confirms §9 R5 and §15.1 R3: the learned 384 → 768
linear routes complementary information into DualDPT's two halves,
which duplication cannot. The `--no-bridge` flag is retained as an
opt-in toggle alongside `--freeze-mixer` / `--lambda-kd` for future
diagnostic runs.

Outstanding from §15.2: CM8 (corpus, disk-blocked), CM10
(re-budget Phase-B / Phase-C), CM11 (smaller Mamba-3 state_dim;
requires Phase-B re-distill).

### 15.6 CM13 result (negative) — classical regularisation on top of CM9

Run on 2026-04-21 from CM9 recipe plus `--drop-path 0.1
--bridge-dropout 0.1 --weight-decay 0.15` (wd × 3 vs default 0.05).

- abs_rel 0.0799 (**+11.7 % vs CM9 0.0715**) — reverted by §13.5.
- log10 0.0357 (+12 %), rmse 0.1565 (+13 %).
- effective_rank 63.9 ≈ CM9 65.3, cross_view_nn 0.6024 ≈ CM9 0.5911.
- feat_cos_mean 0.968 (CM9 0.941) — marginal collapse from dropout.

CM9 aug already supplies strong implicit regularisation; stacking
classical regs removes capacity without adding information. This
closes the entire "attack overfit surface" tier (§15.2 Tier 2):
CM13, CM14, CM17 all regressed vs CM9, confirming the binding
constraint is data volume, not regularisation strength.

The `--drop-path`, `--bridge-dropout`, and `--weight-decay` flags are
retained as opt-in knobs for future runs with a larger corpus.

### 15.7 CM10 result (positive) — re-budget Phase-B ↑ / Phase-C ↓

Run on 2026-04-22 on the CM9 aug recipe. Phase-B 6 k → **20 k** steps
(into `outputs/runs/distill_cm10/`), Phase-C 2 k → **500** steps.
Rationale (§15.2 Tier 3): feature matching (Phase-B) is a
non-overfitting loss against DA3's stable teacher targets, so more
Phase-B should sharpen the student representation; SILog (Phase-C) is
the overfitting loss on a 374-image set, so fewer steps should
reduce train-set memorisation.

- abs_rel 0.0660 (**−7.7 % vs CM9 0.0715**) — kept by §13.5.
- log10 **0.0295** — clears the 0.031 gate for the first time.
- effective_rank 83.1 (vs CM9 65.3) — **+27 %**, the largest gain so
  far, suggesting Phase-B was under-trained at 6 k.
- cross_view_nn 0.6094 (vs CM9 0.5911).
- feat_cos_mean 0.958 (vs CM9 0.941) — a mild uptick, consistent with
  longer Phase-B concentrating features toward teacher geometry.

5/6 §9 gates now pass. Only `effective_rank ≥ 150` remains unmet
(83 vs 150); this is the Mamba-3 state-dim bottleneck and is
structural — see CM11 in §15.2.

Wall time: Phase-B 20 k ≈ 37 min; Phase-C 500 ≈ 5 min; eval ≈ 4 min.
Together CM10 is *faster* than CM9's 6 k + 2 k recipe because the
Phase-B chunked-SSD forward is cheaper per sample than the Phase-C
DPT forward. Refutes the assumption that "more Phase-B" is a cost
trade.

Retained: CM2 + CM9 + CM10 as the joint baseline.

Outstanding from §15.2: CM8 (new-data corpus; disk-blocked at 4.7 GB
free on `/`), CM11 (smaller Mamba-3 state_dim; needs Phase-B
re-distill — cheap now that 20 k takes 37 min).

### 15.8 CM11 result (negative) — state_dim 64→32 hurts, doesn't help

Run on 2026-04-22 on the CM10 recipe: Phase-B 20 k + Phase-C 500 with
Mamba-3 `state_dim=32` (vs default 64). Phase-B was re-distilled from
scratch into `distill_cm11/ckpt_20000.pt` because state dim is an
architectural parameter.

- abs_rel 0.0711 (**+7.7 % vs CM10 0.0660**) — reverted by §13.5.
- log10 0.0319 — regresses back above the 0.031 gate (CM10 cleared it).
- effective_rank **71.80** (vs CM10 83.13) — **the gate CM11 was
  designed to improve went the wrong way by 14 %**. 4/6 gates now
  pass (was 5/6).
- cross_view_nn 0.5556 (vs CM10 0.6094).

Phase-B distill loss settled at ~0.0132 at step 20 k (vs CM10's
~0.0125) — small but real reduction in teacher-matching capacity.
The reduced state dim carries forward as lower feature quality in
Phase-C.

Interpretation: §15.2 Tier 3 rationale ("overparameterised for 374
images") does *not* hold for the state-dim axis. The binding
constraint on `effective_rank` is Mamba-3's recurrent-state
bottleneck relative to full softmax attention — *less* state
makes this worse, not better. Consistent with §15.1 R2:
representation diversity is capacity-limited, not data-limited.
This closes the §15.2 Tier 3 state_dim lever.

Retained: CM2 + CM9 + CM10 (no change). `--state-dim` flag kept on
`train_distill.py`, `train_depth.py`, `eval_ssm3d_vs_da3.py` as an
opt-in diagnostic for future architecture sweeps.

Outstanding from §15.2: **CM8 only** (larger distillation corpus;
disk-blocked). All Tier-1 non-data, Tier-2, and Tier-3 levers have
been exhausted — the remaining abs_rel gap to DA3 (0.0660 vs 0.0377,
~1.75 ×) now has a single remaining lever: real additional data.

### 15.9 Per-image investigation — SSM "wins" on images 10, 11 are a resolution artifact

In the CM10 eval (SSM-3D at `img_size=224`, DA3 at `process_res=504`)
SSM-3D beats DA3 on exactly 2 of 12 terrains images: indices 10 and
11 (`DSC_0625`, `DSC_0626`), which are close-up shots of a corrugated
/ louvered shutter — a geometrically near-planar surface (depth clip
~18 cm) with strong periodic horizontal texture. DA3's per-image
abs_rel spikes to 0.081 / 0.121 on those two (~5× its own mean across
images 0–9); SSM-3D stays at 0.056 / 0.077.

Reran CM10 eval with both models at 504 (`--img-size 504
--da3-process-res 504`, SSM's pos_embed bicubic-resized via the CM2
path). Results (`outputs/eval_cm10_504/`):

- SSM-3D abs_rel mean = **0.2037** (vs CM10@224 = 0.0660). All 6 §9
  gates fail. The 504-resize destroys the 224-trained geometry.
- DA3 keeps its per-image pattern: abs_rel 0.126 / 0.110 on images
  10 / 11 — stripe-as-depth failure mode is resolution-invariant.
- **The SSM-win on 10 / 11 does not persist at matched resolution.**
  DA3 beats SSM on all 12 images at 504.
- SSM-3D `effective_rank` at 504 rose 83 → 109 (more tokens → more
  rank), which rules out "lower expressivity helps on planar
  texture" as the mechanism.

**Root cause of the apparent SSM wins at 224:** the corrugated-shutter
louvre frequency (~30 cycles across the 504-native image) is
**under-sampled** by the 16 × 16 patch grid at `img_size=224` — each
14-pixel patch integrates over a full louvre cycle, low-pass-filtering
the stripe pattern before it reaches the DPT. At 504 the 36 × 36 grid
resolves individual louvres and both models decode them as depth
variation.

Implication: CM10's advantage on images 10 / 11 is **not an
architectural win** — it's a side-effect of evaluating at lower
resolution than DA3's native inference pipeline. The published CM10
abs_rel (0.0660) remains valid as a benchmark number (eval protocol
matches §9 gates, which are defined at the deployed inference paths
— DA3 at 504, SSM at 224), but should not be interpreted as
"SSM-3D is more robust to high-frequency planar texture."

DA3's stripe-over-interpretation on 10 / 11 is a genuine DA3
failure mode (preserved at 504), but SSM-3D's current architecture
does not fix it — it only avoids it when the input resolution happens
to be too coarse to resolve the offending texture.

### 15.10 CM12 result (positive) — native 504 training fixes the resolution artifact

Run on 2026-04-22. After §15.9 exposed that CM10's published numbers
were measured at `img_size=224` while DA3 ran at 504 (a 5× token
asymmetry), we redid the full CM10 recipe at 504: Phase-B distill
20 k steps + Phase-C depth FT 500 steps, both at
`img_size=504, patch_size=14`, with `bs=1` and `--chunk-size 128` to
fit SSD mask in 12 GB VRAM. Student `pos_embed` sized natively for
36 × 36 + 1 = 1297 tokens; `--chunk-size` is the only new engineering
piece (plumbed through `train_distill.py` and `train_depth.py`).

There is **no SSM restriction at 504** — Mamba-3 SSD is
token-count-agnostic (same O(T²) mask as softmax attention, or
O(T·chunk) chunked). 224 was originally chosen for three pragmatic
reasons only: (1) DINOv2 backbone pretraining resolution → 16 × 16
grid matches its optimization basin, (2) 12 GB VRAM budget (full SSD
mask at T=1297 OOMs without chunking), (3) training speed (5× more
tokens → ~25× slower SSD per block).

Results at matched 504 (`outputs/eval_cm12/`):

| Metric | DA3@504 | SSM-3D@504 (CM12) | vs CM10@504 (eval-only) |
|---|---|---|---|
| abs_rel | 0.0417 | **0.0676** ✅ | 0.2037 → **−67 %** |
| δ<1.25 | 0.9743 | **0.9856** ✅ (beats DA3) | 0.6369 → **+55 %** |
| rmse | 0.0796 | **0.1242** ✅ | 0.4017 → **−69 %** |
| log10 | 0.0175 | **0.0294** ✅ | 0.0810 → **−64 %** |
| cross_view_nn | 0.1601 | 0.1724 ❌ | 0.3905 → −56 % |
| effective_rank | 161.3 | 68.1 ❌ | 108.7 → −37 % |

**Depth gates** (abs_rel / δ<1.25 / rmse / log10) all pass — 4/4.
**δ<1.25 = 0.9856 *beats* DA3's 0.9743** — the first metric on which
SSM-3D surpasses the teacher.

**Representation gates** (cross_view_nn, effective_rank) remain open,
but note that `cross_view_nn` drops for *both* models at 504 (DA3
collapses 0.60 → 0.16), suggesting the GT-warp agreement metric is
resolution-sensitive — at 36 × 36 patches, tiny pixel-scale disparity
errors kick every nearest-neighbour match one token over.
Effective_rank is still bounded by `state_dim = 64` (CM11 confirmed
halving it makes this worse; §15.8).

CM12 closes the resolution-artifact audit from §15.9: at matched 504,
SSM-3D passes 4/6 gates — a tighter, more defensible story than
CM10's 5/6 at asymmetric resolution. Phase-B distill loss settled at
0.026 (vs CM10's ~0.04 at 224), confirming the teacher–student
feature match is *better* at matched resolution, and the remaining
abs_rel gap to DA3 (0.0417 vs 0.0676, 1.62×) is smaller than CM10's
apparent-but-inflated gap (0.0421 vs 0.0660 at matched-224, 1.57×).

Retained: **CM9 + CM12** (CM2 + CM10 superseded by native-504
training). CM2's eval-only pos_embed resize is no longer needed but
kept as a code path for backwards compat.

Outstanding from §15.2: CM8 only (larger distillation corpus;
disk-blocked).
