"""Learned 384 → 768 channel bridge for the shared-DPT adapter.

DA3-SMALL's DualDPT expects 768-dim features (`cat_token=True` concatenates
two 384-dim attention streams before handing off to the head). SSM-3D has a
single 384-dim stream.

The fallback — `cat([f, f], -1)` — feeds the head redundant data, effectively
halving its capacity (PLAN §9 R5). `DimBridge` replaces that static duplicate
with a learnable `nn.Linear(384, 768)` per exported layer, initialised to the
cat-duplicate identity so Phase A behaviour at init is unchanged. Phase C
depth fine-tuning trains the bridge to route complementary information into
the DualDPT's two 384-dim halves.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DimBridge(nn.Module):
    """Learnable linear map from `in_dim` to `out_dim = 2 * in_dim`.

    Initialised so that weight = stack([I, I]) and bias = 0, i.e.
    `DimBridge(x) == cat([x, x], -1)` at step 0 — matches the Phase-A smoke
    test and is the starting point the DualDPT was trained against.
    """

    def __init__(
        self, in_dim: int = 384, out_dim: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        out_dim = out_dim if out_dim is not None else 2 * in_dim
        if out_dim != 2 * in_dim:
            raise ValueError(
                f"DimBridge expects out_dim == 2 * in_dim; got {in_dim}→{out_dim}"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        with torch.no_grad():
            eye = torch.eye(in_dim)
            self.linear.weight.copy_(torch.cat([eye, eye], dim=0))
            self.linear.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.dropout(x))


class DimBridgeStack(nn.Module):
    """One `DimBridge` per exported layer for the shared-DPT adapter.

    Stores bridges indexed by exported-layer position (0..N-1), not the
    absolute backbone layer index. Keeps Phase-B/C checkpoints small — only
    the bridge params ride along with the trained mixer.
    """

    def __init__(
        self, num_layers: int, in_dim: int = 384, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.bridges = nn.ModuleList(
            [DimBridge(in_dim=in_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        if len(feats) != len(self.bridges):
            raise ValueError(
                f"DimBridgeStack has {len(self.bridges)} bridges but got {len(feats)} features"
            )
        return [bridge(f) for bridge, f in zip(self.bridges, feats)]
