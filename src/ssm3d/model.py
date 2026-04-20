"""SSM-3D model: DA3's DINOv2 backbone with Mamba-3 self-attention + a small
depth head for the demo.

We construct the backbone programmatically so we can inject Mamba3Attention
via `block_fn=partial(Block, attn_class=Mamba3Attention)`, which is the one
place in DA3 where the attn class is plumbable without patching upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import order matters: import `ssm3d` first to add DA3 to sys.path.
import ssm3d  # noqa: F401

from depth_anything_3.model.dinov2.layers.block import Block  # noqa: E402
from depth_anything_3.model.dinov2.vision_transformer import (  # noqa: E402
    DinoVisionTransformer,
    vit_small,
    vit_base,
)

from .da3_adapter import Mamba3Attention


@dataclass
class BackboneOutput:
    features: torch.Tensor  # (B, S, N_patch, C) final-layer patch tokens
    aux_features: list[torch.Tensor]  # any intermediate-layer (B, S, N_all, C) feats
    grid_hw: tuple[int, int]  # (h, w) patch grid such that h*w = N_patch


class SSM3DBackbone(nn.Module):
    """DA3 DINOv2 backbone with Mamba-3 self-attention."""

    def __init__(
        self,
        size: str = "small",
        img_size: int = 224,
        patch_size: int = 16,
        depth: Optional[int] = None,
        export_feat_layers: tuple[int, ...] = (),
        mamba_state_dim: int = 64,
        mamba_bidirectional: bool = True,
        mamba_three_term: bool = True,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.export_feat_layers = list(export_feat_layers)

        attn_class = partial(
            Mamba3Attention,
            state_dim=mamba_state_dim,
            bidirectional=mamba_bidirectional,
            three_term=mamba_three_term,
            chunk_size=chunk_size,
        )
        block_fn = partial(Block, attn_class=attn_class)

        ctor = {"small": vit_small, "base": vit_base}[size]
        kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            block_fn=block_fn,
            cat_token=False,  # don't concatenate local+global (alt_start is off)
        )
        if depth is not None:
            kwargs["depth"] = depth
        self.vit: DinoVisionTransformer = ctor(**kwargs)

    @property
    def embed_dim(self) -> int:
        return int(self.vit.embed_dim)

    @property
    def grid_hw(self) -> tuple[int, int]:
        s = self.img_size // self.patch_size
        return (s, s)

    def forward(
        self,
        x: torch.Tensor,
        export_feat_layers: Optional[list[int]] = None,
    ) -> BackboneOutput:
        """
        Args:
            x: (B, S, 3, H, W)
            export_feat_layers: per-call override of the constructor value. Lets
                callers request intermediate-layer features without rebuilding
                the backbone (used by the shared-DPT eval adapter).
        """
        layers = self.export_feat_layers if export_feat_layers is None else list(export_feat_layers)
        outs, aux = self.vit.get_intermediate_layers(
            x, n=1, export_feat_layers=layers
        )
        # outs is a tuple of (output_tokens, camera_tokens)
        tokens, _cls = outs[0]
        # tokens: (B, S, N_patch, C), patch tokens only (cls/registers stripped)
        return BackboneOutput(features=tokens, aux_features=list(aux), grid_hw=self.grid_hw)


class SimpleDepthHead(nn.Module):
    """Very small depth head: 1x1 conv -> upsample -> 3x3 conv -> softplus.

    Outputs a single-channel depth map at image resolution.
    """

    def __init__(self, in_channels: int, hidden: int = 64, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, features_grid: torch.Tensor) -> torch.Tensor:
        # features_grid: (B, C, h, w)
        x = F.relu(self.conv1(features_grid))
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return F.softplus(x)  # positive depth


class SSM3DNet(nn.Module):
    """Backbone + depth head. Operates on multi-view input (B, S, 3, H, W)."""

    def __init__(
        self,
        size: str = "small",
        img_size: int = 224,
        patch_size: int = 16,
        depth: Optional[int] = None,
        head_hidden: int = 64,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = SSM3DBackbone(
            size=size, img_size=img_size, patch_size=patch_size,
            depth=depth, chunk_size=chunk_size,
        )
        self.depth_head = SimpleDepthHead(
            in_channels=self.backbone.embed_dim, hidden=head_hidden, image_size=img_size
        )

    def features_grid(self, images: torch.Tensor) -> torch.Tensor:
        """Return (B*S, C, h, w) feature grid."""
        out = self.backbone(images)
        feats = out.features  # (B, S, N, C)
        B, S, N, C = feats.shape
        h, w = out.grid_hw
        return feats.reshape(B * S, h, w, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(images)
        B, S, N, C = out.features.shape
        h, w = out.grid_hw
        grid = out.features.reshape(B * S, h, w, C).permute(0, 3, 1, 2).contiguous()
        depth = self.depth_head(grid).reshape(B, S, 1, self.backbone.img_size, self.backbone.img_size)
        return {"features": out.features, "depth": depth, "grid_hw": (h, w)}
