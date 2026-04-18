"""Unit tests for ssm3d.mamba3.mask"""

from __future__ import annotations

import torch

from ssm3d.mamba3.mask import build_two_term_mask, build_three_term_mask, build_cross_mask


def _random_inputs(batch=2, heads=3, T=8, seed=0):
    torch.manual_seed(seed)
    delta = torch.rand(batch, heads, T) * 0.5 + 0.1  # > 0
    A_log = -(torch.rand(batch, heads, T) * 2.0 + 0.1)  # < 0
    lam = torch.rand(batch, heads, T)  # in [0,1]
    return delta, A_log, lam


def test_two_term_mask_shape_and_lower_triangular():
    delta, A_log, _ = _random_inputs(T=6)
    L = build_two_term_mask(delta, A_log)
    assert L.shape == (2, 3, 6, 6)
    # Upper triangle (strictly above diagonal) must be exactly zero.
    upper = L.triu(diagonal=1)
    assert torch.all(upper == 0)
    # Lower triangle must be strictly positive given Δ>0, A<0.
    lower_mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    assert torch.all(L[..., lower_mask] > 0)


def test_two_term_mask_diagonal_equals_gamma():
    # L[t, t] = γ_t · ∏ over empty = γ_t = Δ_t
    delta, A_log, _ = _random_inputs(T=5)
    L = build_two_term_mask(delta, A_log)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)  # (B, H, T)
    assert torch.allclose(diag, delta, atol=1e-6)


def test_three_term_reduces_to_two_term_when_lambda_one():
    delta, A_log, _ = _random_inputs(T=7)
    lam = torch.ones_like(delta)  # → β = 0, γ = Δ (same as two-term)
    L2 = build_two_term_mask(delta, A_log)
    L3 = build_three_term_mask(delta, A_log, lam)
    assert torch.allclose(L3, L2, atol=1e-5)


def test_three_term_mask_lower_triangular():
    delta, A_log, lam = _random_inputs(T=6)
    L = build_three_term_mask(delta, A_log, lam)
    assert torch.all(L.triu(diagonal=1) == 0)


def test_three_term_extra_band_contributes_when_lambda_less_than_one():
    delta, A_log, lam = _random_inputs(T=6)
    L_with = build_three_term_mask(delta, A_log, lam)
    L_without = build_three_term_mask(delta, A_log, torch.ones_like(lam))
    # Off-diagonal lower entries should differ where λ<1 introduces β band.
    diff = (L_with - L_without).abs()
    # The band starts at entries [t, t-1] and below (j <= t-1)
    strict_lower = torch.tril(torch.ones(6, 6, dtype=torch.bool), diagonal=-1)
    assert diff[..., strict_lower].max() > 0


def test_cross_mask_shape_and_rank_one():
    delta, A_log, _ = _random_inputs(T=9)
    T_q = 5
    Lc = build_cross_mask(delta, A_log, T_q)
    assert Lc.shape == (2, 3, T_q, 9)
    # All rows identical (rank-1 broadcast).
    first = Lc[..., 0, :]
    for i in range(1, T_q):
        assert torch.allclose(Lc[..., i, :], first, atol=1e-6)


def test_cross_mask_column_decay_monotonic_from_right():
    # Rightmost column: γ_{T_kv} * prod over empty = γ_{T_kv}.
    # Leftmost column: γ_1 * prod of α over entire sequence, decayed by many factors.
    delta, A_log, _ = _random_inputs(T=12)
    Lc = build_cross_mask(delta, A_log, T_q=1)
    row = Lc[..., 0, :]  # (B, H, T_kv)
    # Right-most entry equals γ_{T_kv} = Δ_{T_kv}
    assert torch.allclose(row[..., -1], delta[..., -1], atol=1e-5)


def test_mask_underflow_graceful_with_large_decay():
    # Very steep decay shouldn't produce NaNs/Infs.
    delta = torch.full((1, 1, 100), 0.5)
    A_log = torch.full((1, 1, 100), -50.0)
    lam = torch.full((1, 1, 100), 0.5)
    L = build_three_term_mask(delta, A_log, lam)
    assert torch.all(torch.isfinite(L))
