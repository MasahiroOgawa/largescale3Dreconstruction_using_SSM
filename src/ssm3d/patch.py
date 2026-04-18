"""Monkey-patch helpers for installing Mamba3 self-attention into an existing
Depth-Anything-3 network without modifying DA3 source.

We walk the backbone's Block list and replace each `block.attn` module with a
matching Mamba3Attention. RoPE references are preserved from the original
attention. Heads and camera encoder/decoder are left untouched by default.
"""

from __future__ import annotations

from typing import Literal

from torch import nn

from .da3_adapter import Mamba3Attention


def _infer_num_heads(attn: nn.Module) -> int:
    if hasattr(attn, "num_heads"):
        return int(attn.num_heads)
    raise AttributeError("Cannot infer num_heads from existing attention module")


def _infer_dim(attn: nn.Module) -> int:
    # DA3 Attention: self.qkv = nn.Linear(dim, dim * 3, bias=...)
    if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
        return int(attn.qkv.in_features)
    # Camera-encoder Attention (from utils/attention.py) has similar structure
    if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
        return int(attn.proj.in_features)
    raise AttributeError("Cannot infer dim from existing attention module")


def _swap_attn(block: nn.Module, *, state_dim: int = 64, bidirectional: bool = True, three_term: bool = True) -> None:
    """Replace `block.attn` in-place with a Mamba3Attention of matching (dim, num_heads)."""
    if not hasattr(block, "attn"):
        return
    old = block.attn
    dim = _infer_dim(old)
    num_heads = _infer_num_heads(old)
    rope = getattr(old, "rope", None)
    proj_bias = True
    if hasattr(old, "proj") and isinstance(old.proj, nn.Linear):
        proj_bias = old.proj.bias is not None
    new = Mamba3Attention(
        dim=dim,
        num_heads=num_heads,
        proj_bias=proj_bias,
        rope=rope,
        state_dim=state_dim,
        bidirectional=bidirectional,
        three_term=three_term,
    )
    new.to(next(old.parameters()).device if any(p.requires_grad for p in old.parameters()) else "cpu")
    # Match dtype of the module being replaced
    try:
        dtype = next(old.parameters()).dtype
        new = new.to(dtype=dtype)
    except StopIteration:
        pass
    block.attn = new


def install_mamba3(
    net: nn.Module,
    which: Literal["backbone_only", "all"] = "backbone_only",
    state_dim: int = 64,
    bidirectional: bool = True,
    three_term: bool = True,
) -> int:
    """Swap self-attention to Mamba-3 across the DA3 network.

    Args:
        net: a DA3 net (or anything with a .backbone.blocks list of Blocks).
        which:
            - "backbone_only": swap only the DINOv2 backbone blocks.
            - "all": additionally swap any other nn.Module named "attn" found
              anywhere in the net (covers the camera encoder's second attention
              stack in DA3's utils/ path).
        state_dim, bidirectional, three_term: forwarded to Mamba3Attention.

    Returns:
        number of attention modules swapped.
    """
    count = 0

    if hasattr(net, "backbone") and hasattr(net.backbone, "blocks"):
        for block in net.backbone.blocks:
            _swap_attn(block, state_dim=state_dim, bidirectional=bidirectional, three_term=three_term)
            count += 1

    if which == "all":
        # Walk every submodule and replace any `.attn` that is NOT already Mamba3Attention
        for module in net.modules():
            attn = getattr(module, "attn", None)
            if attn is None or isinstance(attn, Mamba3Attention):
                continue
            # skip the ones we already did in backbone (they're Mamba3Attention now)
            try:
                _swap_attn(module, state_dim=state_dim, bidirectional=bidirectional, three_term=three_term)
                count += 1
            except AttributeError:
                pass

    return count


def count_mamba3_attn(net: nn.Module) -> int:
    return sum(1 for m in net.modules() if isinstance(m, Mamba3Attention))
