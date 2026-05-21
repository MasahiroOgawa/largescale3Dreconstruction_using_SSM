"""Top-level Mamba-3 tracker (v2 — query-conditioned bank).

Pipeline (matches `doc/attention/mamba3_attention.tex §8` after the
2026-05-18 update):

  video        (B, F, 3, H, W)
  queries_xyt  (B, N, 3) in input-image pixel coords
  query_mask   (B, N) bool
        ↓
  PyramidEncoder.forward_video → list of (B, F, D, h_l, w_l)
        ↓                                                             ↓
  CausalCrossPropagator(pyramid, queries_xyt, mask, image_size)
        ↓
  q_history    (B, F, N, D)
        ↓
  TrackHeads
        ↓
  TrackerOutputs(delta_xyz, vis_logits, spawn_logits)

`delta_xyz` is motion since the query anchor; absolute positions are
recovered as `delta_xyz + p_query`.
"""

from __future__ import annotations

from torch import Tensor, nn

from .encoder import PyramidEncoder
from .heads import TrackerOutputs, TrackHeads
from .propagator import CausalCrossPropagator


class Mamba3Tracker(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        state_dim: int = 64,
        level_sizes: tuple[int, ...] = (32, 64),
        blocks_per_level: int = 2,
        patch: int = 14,
        num_iters: int = 1,
        use_correlation: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = PyramidEncoder(
            dim=dim, num_heads=num_heads, state_dim=state_dim, patch=patch,
            level_sizes=level_sizes, blocks_per_level=blocks_per_level,
        )
        self.propagator = CausalCrossPropagator(
            dim=dim, num_pyramid_levels=len(level_sizes),
            num_heads=num_heads, state_dim=state_dim,
            num_iters=num_iters, use_correlation=use_correlation,
        )
        self.heads = TrackHeads(dim=dim)
        self.image_size = self.encoder.coarse_image_size

    def forward(
        self,
        video: Tensor,
        queries_xyt: Tensor,
        query_mask: Tensor,
    ) -> TrackerOutputs:
        """
        Args:
            video: (B, F, 3, H, W). Resized to encoder.coarse_image_size² inside.
            queries_xyt: (B, N, 3) — (x, y, t) per track, in the SAME pixel coord
                system as `video` (i.e. already scaled by the dataset to match
                the encoder's input resolution).
            query_mask: (B, N) bool — True at real query slots, False at padding.
        Returns:
            TrackerOutputs with `xyz` = predicted Δp (motion since query anchor).
        """
        pyramid = self.encoder.forward_video(video)
        q_history = self.propagator(
            pyramid, queries_xyt, query_mask, image_size=self.image_size,
        )
        return self.heads(q_history)
