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
