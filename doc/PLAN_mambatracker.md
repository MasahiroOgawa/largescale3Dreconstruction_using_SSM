# PLAN — Mamba-3 SSD Tracker for Streaming 3D Point Tracking

Status: idea dump (2026-05-06). Not yet committed as the next research direction. Successor candidate to §15.59 (which is fighting on transformer's home turf — short T, parallelizable, FlashAttention closes the memory gap).

## Why tracking, not reconstruction

The mamba3 swap's structural advantages over transformer attention split into two regimes:

| Regime | Memory | Latency |
|---|---|---|
| Offline encode of fixed sequence (DA3 §15.59) | parity with FlashAttention | 1.25× → 2× faster, growing with T |
| Streaming / autoregressive with growing prefix | **mamba O(state_dim)** vs transformer O(T_seen · D) KV cache | mamba O(1) per step vs transformer O(T_seen) |

DA3 is the first regime — memory plot is flat, only latency wins. Streaming 3D point tracking is the second regime, where mamba wins on **both** axes and the gap grows without bound as the video gets longer.

## Task

**Input**: video stream, frames `f_1, f_2, ..., f_T` arriving one at a time.
**Query**: set of N 3D query points `q_1, ..., q_N` specified at frame 1.
**Output per frame**: each query's updated 3D position `(x_i^t, y_i^t, z_i^t)`.

## Why mamba3's structure fits tracking natively

SSD's recurrent form is

    s_t = A · s_{t-1} + B · x_t       # state update from new observation
    y_t = C · s_t                     # readout

This is exactly the structure of a **learned Kalman filter** on a hidden latent state. Tracking is the canonical task for that structure:

- noisy per-frame observations of where a point appears
- want a smoothed, history-aware 3D estimate
- state `s_i` learns "where this point has been, how it's moving, appearance signature" — no hand-designed dynamics model

Our existing Mamba-3 Triton kernel (`mamba3_attn/kernels/`) supports both modes:
- **chunk-scan**: parallel over time, used during training
- **recurrent**: one step at a time, used during streaming inference

Same kernel, different invocation. No new low-level work.

## Core architecture (sketch)

```
                   ┌──────────────┐
   frame f_t  ───▶│ DA3 encoder  │──▶ frame features F_t  (per-token)
                   └──────────────┘
                          │
                          ▼
   for each query q_i with hidden state s_i:
       # (1) cross-attend query feature to frame tokens to localize obs at frame t
       obs_i^t = cross_attn(query=q_i_feat, keys/values=F_t)        # (D,)

       # (2) SSD recurrent step — fold obs_i^t into per-query tracking state
       s_i_new = A_i · s_i + B_i · obs_i^t                          # (state_dim,)

       # (3) read out 3D position from state
       (x, y, z)_i^t = MLP(C_i · s_i_new)

       s_i ← s_i_new
```

Three architectural pieces:

1. **Per-frame encoder** (no recurrence): existing DA3 encoder with our mamba3 cross-view attention swap. Produces per-token features for frame `f_t`.

2. **Per-query SSD tracker** (the new core piece): mamba3 SSD layer (or shallow stack) running per query in recurrent mode. State `s_i` accumulates history. Per-query trackers run in parallel batches across queries.

3. **3D position head**: small MLP that reads the SSD state and emits `(x, y, z)`.

## Asymptotic comparison vs transformer (temporal aggregation across T frames)

|  | Transformer + FlashAttention | mamba3 SSD |
|---|---|---|
| Memory at frame T | O(T · D) — KV cache grows with frames | O(state_dim) — fixed |
| Compute per frame | O(T · D) — attend to all past KV | O(state_dim²) — fixed |
| Memory at T=10,000 frames (D=384, fp16) | ~tens of GB | tens of MB |
| Latency at T=10,000 | ~10,000× slower than at T=1 | 1× — constant |

This is the asymptotic regime mamba was designed for. Both memory **and** latency stay bounded as the video grows. Track a 1-hour video at 30 fps (T=108,000) on the same budget as tracking a single frame.

## Implementation scope

Three pieces of work, in increasing scope:

1. **`Mamba3Tracker` module** (~200 LOC): per-query SSD stack wrapping the existing kernel in recurrent mode. Inputs: cross-attention output `obs_t`. Outputs: 3D position. State carries between calls.

2. **Tracking head + dataset loader** (~500 LOC): a video dataset (candidates: TAP-Vid-3D, PointOdyssey, Kubric MOVi-F) yielding (frame, queries, GT tracks). Per-frame position loss + temporal smoothness loss.

3. **Training loop** (~300 LOC): exposes both chunk-scan training (parallel over time, fast on GPU) and recurrent inference (streaming, low memory). Trained jointly with DA3's depth supervision so the encoder learns features useful for both reconstruction and tracking.

The kernel itself is unchanged.

## Why this paper would actually win on both panels

Reconstruction (§15.59) fights on transformer's home turf — short T, parallelizable, FlashAttention closes the memory gap, so the headline reduces to "2× faster at long T." Tracking is the opposite scenario:

- **Long T natively**: videos are commonly 1k–100k frames. Transformer's KV cache OOMs at T~1k on consumer GPUs; mamba sails through T=100k.
- **Streaming requirement**: mamba's recurrent mode is native; transformer needs awkward cache management and latency blows up.
- **Per-query independence**: per-query SSD states naturally encode object identity. Batched parallelism across queries is straightforward.

The memory plot is dramatic (transformer OOM at T~1k, mamba constant). The latency plot is steeper (mamba constant per frame, transformer linear-growing per frame).

## Open questions / risks

- **Cross-attention cost per frame**: localising a query in a frame's tokens via attention is still O(tokens_per_frame), per query. With many queries and high-res frames, this can dominate. Mitigations: shared cross-attn keys/values across queries (compute once per frame), or replace cross-attn itself with a mamba3 spatial scan over frame tokens.
- **Per-query state vs shared state**: independent SSD per query is the cleanest formulation but doesn't share information across queries (other points moving similarly). Hybrid: a shared backbone state + per-query head. Adds complexity, may or may not pay.
- **Initialization**: how do queries get their initial state at frame 1? Either from a single-frame encoder pass over `f_1` at the query locations, or from a learned prior.
- **Ground-truth supervision quality**: synthetic data (Kubric, PointOdyssey) has clean GT but may not transfer; real-data 3D tracking labels are sparse. Plan: train on synthetic, fine-tune on whatever real labels exist (TAP-Vid-3D), evaluate on held-out real.
- **Dynamics breaks**: what happens at object disocclusion / re-appearance? SSD state must encode "I lost this point" without diverging. May need an explicit gating signal.

## Decision points before committing to this direction

- **(a) 1-day toy prototype first**: implement the SSD tracker on synthetic 2D point tracking (no DA3, just 2D position regression on Kubric or a simple synthetic). Validate that SSD's Kalman-filter-shape actually produces stable tracks. Cost: 1 day. Decisive: if tracks diverge or fail to learn, the architecture story is broken before any DA3 integration.
- **(b) Drop §15.59 to focus here, vs run both in parallel**: §15.59's accuracy gap (V4 < V1) is a serious problem; this tracking direction is greenfield. Choosing both consumes more time than choosing one. Most likely route: park §15.59 at the documented "efficiency wins, accuracy gap" state and shift the next training cycle to tracking.
- **(c) Benchmark target**: TAP-Vid-3D for 3D tracking metrics, PointOdyssey for long-T stress test. Need to confirm both are accessible.

## Tie-back to the project's standing constraints

- `feedback_no_efficiency_only_paper.md`: the tracking-paper framing reaches accuracy parity with transformer-baseline trackers (CoTracker, TAPIR-3D) is the necessary bar. Efficiency advantage is the *additional* result on top.
- `feedback_efficiency_and_accuracy_together.md`: every variant comparison reports both axes — same here. Tracking accuracy (e.g., AJ@5px, OA, F-score on TAP-Vid-3D) + latency/memory at increasing T.
- `feedback_stay_close_to_da3_paper.md`: the tracking head is *additive* — DA3 encoder unchanged, just adds a temporal head. Same posture as before: don't deviate from DA3 internals; build on top.

---

## Findings from the v6–v10 implementation (2026-05-19 to 2026-05-20)

What was actually built: a from-scratch tracker with the encoder/propagator/heads from §8 of `doc/attention/mamba3_attention.tex` (not the streaming SSD-per-query design above — see the implementation in `src/mamba3_tracker/`). The propagator carries one 384-dim "track memory" vector `Q[b, n, :]` per (clip, tracked-point), updated each frame by two cross-attention branches (`Mamba3CrossAttention` SSD + RAFT-style cosine `CorrelationCrossAttention`). v6 → v10 is the recorded ablation chain that landed on a load-bearing architectural lesson; results in `results/ablation/v{6,7,8}/`.

### The story in one paragraph

v6/v7/v8/v9 all hit the same visible symptom: **predicted tracks barely move on the rendered video** (a few-pixel oscillation at most). 3D-AJ numbers improved across variants (v6 0.005 → v7 0.044 → v8 0.049 on (pstudio + drivetrack) mean) but the qualitative videos stayed static. We chased it through three different loss redesigns (v6 multi-term, v7 + correlation arch, v8 velocity+position with Huber scale, v9a position weight raised to 1.0) before realising the loss wasn't the bottleneck.

### The actual bug

In `src/mamba3_tracker/model/propagator.py:198–206`, the per-frame update was

```python
for t in range(F_):
    for _ in range(self.num_iters):                # 3
        for l in range(self.num_pyramid_levels):   # 2
            ...
            Q = Q + delta_ssd + delta_corr         # ← unbounded accumulator
    history.append(Q)
```

That's 6 residual additions per frame × 32 frames = 192 additions into `Q` with **no LayerNorm between them**. The optimiser, free to learn cross-attention weights of any magnitude, drove the deltas large enough that `Q`'s L2 norm reached **10⁵–10⁶** by the end of a clip (healthy networks: ~1–10). Because every per-frame delta also points in a similar direction (slow-motion scene → similar weighted-average of frame patches each frame), the *direction* of `Q` was dominated by the running accumulator: `cos(Q(t), Q(0)) ≈ 1.0` across all frames. The head's input LayerNorm scrubbed the magnitude away but couldn't restore directional variation, so the head saw the same direction every frame and output the same Δp̂ every frame.

Diagnostic comparing the three trained checkpoints on one pstudio clip:

|  | mean ‖Q‖ per frame | cos(Q(15), Q(0)) |
|---|---|---|
| v7 ckpt_30000 | 250,979 | 0.26 |
| v8 ckpt_30000 | 3,626,264 | 0.9999 |
| v9 ckpt_1000 | 774,423 | 0.9999 |

v7 happened to converge with smaller `‖Q‖` and a little directional spread surviving → the "few pixels of motion" users saw. v8 and v9 saturated harder under their stronger per-frame loss gradients.

### The v10 fix

Add a post-update LayerNorm so `Q`'s magnitude is bounded after every residual addition:

```python
Q = self.out_norms[l](Q + delta_ssd + delta_corr)
```

This is standard transformer-block hygiene that the propagator was missing. With `Q` re-normalised, each new per-frame delta has *visible influence* on direction — the residual updates can do their job and the position/velocity losses can finally pull the prediction toward GT motion. At random init the patched model already produces clearly time-varying Δp̂ across frames where the old propagator collapsed instantly.

Implementation diff is one new `nn.ModuleList` of `LayerNorm`s on `CausalCrossPropagator.__init__` plus the one-line change in the inner loop. See commit landing `configs/v10.yaml` + the propagator change.

### Lessons for future variants (and for the streaming-SSD direction at the top of this file)

- **Residual-stack hygiene is non-negotiable in any propagator/recurrence.** The streaming-SSD design above is also a per-query recurrence; whoever implements `Mamba3Tracker` per the original sketch must put LayerNorm (or an explicit gating mechanism) on the state path, not only on the inputs to attention. Pre-norm alone is insufficient when the state itself is the residual accumulator.
- **Diagnose state-saturation before tuning losses.** Three failed loss redesigns cost ~12 hours of training compute before the track-memory diagnostic (`‖Q‖`, cos similarity across t) was run. Add this as a first-line check whenever the qualitative result diverges from the quantitative score.
- **`q_norm` pre-norm ≠ residual safety.** The existing code already had `q_norms[l](Q)` as the input to `ssd_levels[l]` and `corr_levels[l]`. That bounded the *attention input*, not the *state*. The state still drifted.
- **TAPVid-3D 3D-AJ can improve while the tracker is silently broken.** Predict-near-anchor satisfies the threshold-based metric for short-clip slow-motion subsets (pstudio). A correct architecture is necessary for the metric to actually measure tracking quality rather than anchor-snapping. Future ablations should report both the metric *and* a qualitative-motion diagnostic (e.g. ‖Δp̂(F-1) − Δp̂(0)‖ vs ‖Δp*(F-1) − Δp*(0)‖) per variant.
- **Loss-vs-arch attribution requires arch-side diagnostics.** "Why doesn't the model learn?" can have an architectural answer that no loss tweak will reach. v6→v9 was four loss recipes hitting the same arch failure; v10 is one arch fix that should unblock whichever of those losses is actually best.

