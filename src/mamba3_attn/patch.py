"""Monkey-patch helpers for installing Mamba3 self/cross attention into an
existing Depth-Anything-3 network without modifying DA3 source.

DA3 uses the **same `block.attn` module** for both per-view ("local") and
cross-view ("global") attention — `process_attention(blk, "local"/"global")`
just rearranges tokens before the call. So swapping `block.attn` covers both
modes with a single replacement.

Locations of attention in DA3-SMALL:
  - `net.model.backbone.pretrained.blocks[i].attn`  — 12 backbone blocks
    (used for both per-view and cross-view via process_attention)
  - `net.model.cam_enc.trunk[i].attn`              — 4 camera-encoder blocks
    (input camera-conditioning, only active when extrinsics are passed)

The `cam_dec` head is a pure MLP, no attention to swap.
"""

from __future__ import annotations

from typing import Literal

from torch import nn

from .da3_adapter import Mamba3Attention, Mamba3VSSDAdapter

# Map variant name → DA3-shaped attention class. Adding a new variant means
# extending this dict and (if needed) the inner operator in mamba3/.
_VARIANT_CLASSES: dict[str, type[nn.Module]] = {
    "mamba3": Mamba3Attention,
    "vssd": Mamba3VSSDAdapter,
}


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


def _swap_attn(
    block: nn.Module, *,
    variant: Literal["mamba3", "vssd"] = "mamba3",
    state_dim: int = 64, bidirectional: bool = True, three_term: bool = True,
    use_fused_kernel: bool = True, chunk_size: int | None = None,
) -> None:
    """Replace `block.attn` in-place with the variant attention class of matching (dim, num_heads)."""
    if not hasattr(block, "attn"):
        return
    old = block.attn
    dim = _infer_dim(old)
    num_heads = _infer_num_heads(old)
    rope = getattr(old, "rope", None)
    proj_bias = True
    if hasattr(old, "proj") and isinstance(old.proj, nn.Linear):
        proj_bias = old.proj.bias is not None
    cls = _VARIANT_CLASSES[variant]
    new = cls(
        dim=dim,
        num_heads=num_heads,
        proj_bias=proj_bias,
        rope=rope,
        state_dim=state_dim,
        bidirectional=bidirectional,
        three_term=three_term,
        use_fused_kernel=use_fused_kernel,
        chunk_size=chunk_size,
    )
    new.to(next(old.parameters()).device if any(p.requires_grad for p in old.parameters()) else "cpu")
    # Match dtype of the module being replaced
    try:
        dtype = next(old.parameters()).dtype
        new = new.to(dtype=dtype)
    except StopIteration:
        pass
    block.attn = new


def _backbone_blocks(net: nn.Module) -> list[nn.Module] | None:
    """Find the backbone's `blocks` list across DA3 / re-impl variants.

    Real DA3:    net.backbone.pretrained.blocks
    Bare ViT:    net.backbone.blocks
    """
    if not hasattr(net, "backbone"):
        return None
    bb = net.backbone
    inner = getattr(bb, "pretrained", None)
    if inner is not None and hasattr(inner, "blocks"):
        return list(inner.blocks)
    if hasattr(bb, "blocks"):
        return list(bb.blocks)
    return None


def install_mamba3(
    net: nn.Module,
    which: Literal["backbone_only", "all"] = "all",
    variant: Literal["mamba3", "vssd"] = "mamba3",
    state_dim: int = 64,
    bidirectional: bool = True,
    three_term: bool = True,
    use_fused_kernel: bool = True,
    chunk_size: int | None = None,
    layer_indices: list[int] | None = None,
) -> int:
    """Swap self/cross attention to a Mamba-3-family operator across the DA3 network.

    Args:
        net: typically `da3.model` (a DA3 net), or any net with a backbone
            blocks list. `cam_enc.trunk[i].attn` is also covered when which="all".
        which:
            - "backbone_only": swap only backbone blocks (12 in DA3-SMALL).
            - "all" (default): also swap `cam_enc.trunk[i].attn` (4 more blocks
              in DA3-SMALL — camera-encoder self-attention on input cam tokens).
        variant: operator family to install.
            - "mamba3" (default): full Mamba-3 SSD (`Mamba3Attention`) with
              bidirectional scans and trapezoidal three-term mask.
            - "vssd": Non-Causal SSD (`Mamba3VSSDAdapter`); the SSD-only flags
              `bidirectional`, `three_term`, `chunk_size`, `use_fused_kernel`
              are silently ignored.
        state_dim, bidirectional, three_term: forwarded to the variant class.
        layer_indices: when set, restricts the swap to these flat indices.
            Numbering covers backbone first (0..N_bb-1), then cam_enc trunk
            (N_bb..N_bb+N_cam-1). When None, all layers covered by `which` are
            swapped.

    Returns:
        number of attention modules swapped.
    """
    if variant not in _VARIANT_CLASSES:
        raise ValueError(f"unknown variant: {variant!r}; expected one of {list(_VARIANT_CLASSES)}")

    count = 0
    target_cls = _VARIANT_CLASSES[variant]
    kw = dict(variant=variant, state_dim=state_dim, bidirectional=bidirectional,
              three_term=three_term, use_fused_kernel=use_fused_kernel,
              chunk_size=chunk_size)

    indices = set(layer_indices) if layer_indices is not None else None

    blocks = _backbone_blocks(net)
    if blocks is not None:
        for i, block in enumerate(blocks):
            if indices is None or i in indices:
                _swap_attn(block, **kw)
                count += 1

    if which == "all":
        cam_enc = getattr(net, "cam_enc", None)
        if cam_enc is not None and hasattr(cam_enc, "trunk"):
            n_bb = len(blocks) if blocks is not None else 0
            for j, block in enumerate(cam_enc.trunk):
                flat_idx = n_bb + j
                if indices is not None and flat_idx not in indices:
                    continue
                if hasattr(block, "attn") and not isinstance(block.attn, target_cls):
                    _swap_attn(block, **kw)
                    count += 1

    return count


def count_mamba3_attn(net: nn.Module) -> int:
    return sum(
        1 for m in net.modules()
        if isinstance(m, tuple(_VARIANT_CLASSES.values()))
    )
