"""Top-level Mamba-3 tracker.

Pipeline (matches `doc/attention/mamba3_attention.tex §8`):
  video (B, F, 3, H, W)
    → PyramidEncoder.forward_video → list of (B, F, D, h_l, w_l)
    → CausalCrossPropagator         → (B, F, N, D)
    → TrackHeads                    → TrackerOutputs(xyz, vis, spawn)
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
        num_tracks: int = 512,
        level_sizes: tuple[int, ...] = (32, 64, 128),
        blocks_per_level: int = 2,
        patch: int = 14,
    ) -> None:
        super().__init__()
        self.encoder = PyramidEncoder(
            dim=dim, num_heads=num_heads, state_dim=state_dim, patch=patch,
            level_sizes=level_sizes, blocks_per_level=blocks_per_level,
        )
        self.propagator = CausalCrossPropagator(
            dim=dim, num_tracks=num_tracks,
            num_pyramid_levels=len(level_sizes),
            num_heads=num_heads, state_dim=state_dim,
        )
        self.heads = TrackHeads(dim=dim)

    def forward(self, video: Tensor) -> TrackerOutputs:
        """
        Args:
            video: (B, F, 3, H, W).
        Returns:
            TrackerOutputs (per the dataclass).
        """
        pyramid = self.encoder.forward_video(video)
        q_history = self.propagator(pyramid)
        return self.heads(q_history)
