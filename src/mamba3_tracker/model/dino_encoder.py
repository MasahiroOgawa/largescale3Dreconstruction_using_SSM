"""Frozen DINOv2 backbone for v14.

Drop-in replacement for `PyramidEncoder` when the upstream patch-embed +
multi-block VSSD pyramid was the model-quality bottleneck (v6–v13 failure
mode: encoder couldn't resolve sub-patch motion because it never saw the
image at finer than 14-px granularity, and the multi-level "feature
pyramid" was just bilinear-upsampled coarse features).

Design:

  * One frozen forward pass through DINOv2 per frame.
  * Output: a single feature grid per frame at the native DINO patch
    resolution (32×32 for 448×448 input at patch=14).
  * Output dim = 384 — same as the propagator's `dim`, so no projection
    layer.
  * No gradients flow through DINO. The encoder's `.parameters()` are
    marked `requires_grad=False`.

Compare to v13's `PyramidEncoder`:
  * v13 trained ~1.4M params from scratch on 80 clips → could not learn
    drivetrack motion (motion ratio ~2-8% across 6000 steps).
  * v14 uses ~22M pretrained-on-1.4B-images DINO params, frozen → strong
    per-patch appearance features without us spending gradient on them.

Memory at inference is unchanged from v13 since we ditched DINO's
gradient bookkeeping (no `backward()` through it).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 image encoder.

    Args:
        model_name: HuggingFace hub id. Default `facebook/dinov2-small`
            (22M params, hidden_size=384, patch=14). For 448×448 input
            this produces a 32×32 patch token grid plus a CLS token.
        image_size: side length the image gets resized to before feeding
            DINO. Must be a multiple of `patch=14`. Default 448 → 32×32
            grid.

    Output of `.forward_video(video)`:
        list of one tensor with shape (B, F, D, h, w) where D = 384,
        h = w = image_size // patch_size. Returned as a single-element
        list so it matches the existing propagator interface.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        image_size: int = 448,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.patch_size = int(self.backbone.config.patch_size)
        if image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be a multiple of patch_size={self.patch_size}"
            )
        self.image_size = int(image_size)
        self.grid_size = self.image_size // self.patch_size
        self.dim = int(self.backbone.config.hidden_size)

        # DINOv2 was trained with ImageNet mean/std normalisation.
        self.register_buffer("imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # `coarse_image_size` exposed for compatibility with PyramidEncoder.
    @property
    def coarse_image_size(self) -> int:
        return self.image_size

    @torch.no_grad()
    def _forward_one_image_batch(self, image: Tensor) -> Tensor:
        """Args:  image (B, 3, H, W) in [0, 1].
        Returns: (B, dim, grid, grid) feature map.
        """
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(
                image, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        image = (image - self.imagenet_mean) / self.imagenet_std

        # DINOv2 returns last_hidden_state of shape (B, 1 + grid*grid, D)
        # — first token is CLS, rest are patch tokens in row-major order.
        out = self.backbone(pixel_values=image)
        tokens = out.last_hidden_state[:, 1:, :]                  # (B, P, D)
        B, P, D = tokens.shape
        if P != self.grid_size * self.grid_size:
            raise RuntimeError(
                f"DINOv2 returned {P} patch tokens; expected "
                f"{self.grid_size * self.grid_size} for image_size={self.image_size}, "
                f"patch={self.patch_size}"
            )
        feat = tokens.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)
        return feat                                                # (B, D, h, w)

    def forward(self, image: Tensor) -> list[Tensor]:
        """Single-image forward (matches PyramidEncoder.forward signature).
        Returns a one-level "pyramid".
        """
        feat = self._forward_one_image_batch(image)
        return [feat]

    def forward_video(self, video: Tensor) -> list[Tensor]:
        """Args:  video (B, F, 3, H, W).
        Returns: one-element list of (B, F, D, grid, grid).
        """
        B, F_, C, H, W = video.shape
        flat = video.reshape(B * F_, C, H, W)
        feat = self._forward_one_image_batch(flat)
        D, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
        return [feat.reshape(B, F_, D, h, w)]

    def forward_frame(self, image: Tensor) -> Tensor:
        """Streaming-inference path: one frame in, one feature grid out.
        Args:  image (B, 3, H, W).
        Returns: (B, D, grid, grid).
        """
        return self._forward_one_image_batch(image)
