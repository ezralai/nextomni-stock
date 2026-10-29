"""Tests for Marcos López de Prado's Memory-Preserving Fractional Differentiation Engine (FFD)."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.features.fracdiff import (
    augmented_dickey_fuller_test,
    evaluate_fracdiff_profile,
    find_optimal_d,
    fractional_differentiation_ffd,
    get_fractional_weights,
    FracDiffReceipt,
)
from src.cli import fracdiff_command


def test_fractional_weights_mathematical_properties():
    """Verify analytical recursive properties of binomial expansion weights."""
    # d = 0 (identity / raw levels)
    w_0 = get_fractional_weights(d=0.0)
    assert len(w_0) == 1
    assert np.isclose(w_0[0], 1.0)

    # d = 1 (standard first-difference / returns)
    w_1 = get_fractional_weights(d=1.0)
    assert len(w_1) == 2
    assert np.isclose(w_1[0], 1.0)
    assert np.isclose(w_1[1], -1.0)

    # Fractional 0 < d < 1
    d = 0.4
    tau = 1e-4
    w_frac = get_fractional_weights(d=d, threshold=tau)
    assert len(w_frac) > 2
    assert np.isclose(w_frac[0], 1.0)
    assert np.isclose(w_frac[1], -0.4)

    # All weights for k >= 1 must be strictly negative for 0 < d < 1
    assert np.all(w_frac[1:] < 0.0)

    # Weights must decay monotonically in magnitude
    abs_weights = np.abs(w_frac[1:])
    assert np.all(np.diff(abs_weights) <= 0.0)

    # Final retained weight must be bounded near cutoff threshold tau
    assert abs(w_frac[-1]) >= tau or len(w_frac) == 2000
    assert abs(w_frac[-1]) < tau * 1.5


def test_fractional_differentiation_ffd_alignment_and_zero_leakage():
    """Verify causal alignment, fixed window length, and integer boundary consistency."""
    np.random.seed(42)
    n = 100
    prices = pd.Series(100.0 + np.cumsum(np.random.normal(0.0, 1.0, size=n)))

    # d = 1 should match first difference (after first entry)
    ffd_1 = fractional_differentiation_ffd(prices, d=1.0)
    assert len(ffd_1) == n
    assert np.isnan(ffd_1.iloc[0])
    np.testing.assert_allclose(ffd_1.dropna().values, prices.diff().dropna().values, atol=1e-8)

    # d = 0 should match raw prices
    ffd_0 = fractional_differentiation_ffd(prices, d=0.0)
    assert len(ffd_0) == n
    np.testing.assert_allclose(ffd_0.values, prices.values, atol=1e-8)

    # Fractional d = 0.5
    ffd_half = fractional_differentiation_ffd(prices, d=0.5, threshold=1e-3)
    weights = get_fractional_weights(d=0.5, threshold=1e-3)
    l_win = len(weights)
    # Exactly l_win - 1 initial NaNs
    assert ffd_half.iloc[:l_win - 1].isna().all()
    assert not ffd_half.iloc[l_win - 1:].isna().any()


def test_augmented_dickey_fuller_test_synthetic_series():
    """Verify ADF unit-root test distinguishes stationary vs non-stationary series."""
    np.random.seed(42)
    n = 300

    # 1. Stationary series: AR(1) with phi = 0.2
    eps = np.random.normal(0, 1, size=n)
    stat_series = np.zeros(n)
    for t in range(1, n):
        stat_series[t] = 0.2 * stat_series[t - 1] + eps[t]

    stat, pval, is_stat = augmented_dickey_fuller_test(stat_series, lags=1)
    assert is_stat is True
    assert pval < 0.05
    assert stat < -2.863

    # 2. Non-stationary series: Random walk with drift
    rw_series = 100.0 + np.cumsum(np.random.normal(0.05, 1.0, size=n))
    stat_rw, pval_rw, is_stat_rw = augmented_dickey_fuller_test(rw_series, lags=1)
    assert is_stat_rw is False
    assert pval_rw > 0.05
    assert stat_rw > -2.863

    # 3. Degenerate flat series
    flat_series = np.full(50, 100.0)
    stat_f, pval_f, is_stat_f = augmented_dickey_fuller_test(flat_series)
    assert is_stat_f is False
    assert pval_f == 1.0


def test_find_optimal_d_memory_preservation():
    """Verify that optimal d* preserves substantially more memory than integer difference."""
    np.random.seed(42)
    n = 250
    # Simulate geometric Brownian motion price series
    log_ret = np.random.normal(0.0005, 0.015, size=n)
    prices = pd.Series(100.0 * np.exp(np.cumsum(log_ret)))

    res = find_optimal_d(prices, threshold=1e-3, d_step=0.05)

    assert "optimal_d" in res
    assert 0.0 <= res["optimal_d"] <= 1.0
    assert res["is_stationary"] is True
    # Fractional d* retains superior correlation to raw prices than d=1.0 returns
    assert res["retained_memory_corr"] > res["raw_to_diff1_corr"]
    assert res["retained_memory_corr"] > 0.50
    assert len(res["scan_points"]) > 5


def test_fracdiff_receipt_signature_and_tamper_detection():
    """Verify HMAC-SHA256 signature generation and tamper resilience."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, 150)))
    receipt = evaluate_fracdiff_profile(
        ticker="AAPL",
        prices=prices,
        threshold=1e-3,
        auto_d=True,
    )

    assert receipt.ticker == "AAPL"
    assert len(receipt.signature) == 64
    assert receipt.verify_integrity() is True

    # Tampering with optimal_d must invalidate signature
    receipt.optimal_d = 0.999
    assert receipt.verify_integrity() is False


def test_cli_fracdiff_command(capsys):
    """Verify CLI command prints human-readable format and JSON output."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    mock_df = pd.DataFrame({
        "session_date": dates,
        "adj_close": 150.0 + np.cumsum(np.random.normal(0.02, 1.2, size=n)),
        "close": 150.0 + np.cumsum(np.random.normal(0.02, 1.2, size=n)),
    })

    with patch("src.data.db.get_observations", return_value=mock_df):
        # 1. Human-readable output
        args_text = argparse.Namespace(
            ticker="MSFT",
            threshold=1e-3,
            d=None,
            auto_d=True,
            json=False,
        )
        fracdiff_command(args_text)
        out = capsys.readouterr().out
        assert "Memory-Preserving Fractional Differentiation Engine (FFD)" in out
        assert "Security / Ticker:            MSFT" in out
        assert "Optimal Stationary Order (d*):" in out
        assert "Memory Retained Correlation" in out

        # 2. JSON output
        args_json = argparse.Namespace(
            ticker="MSFT",
            threshold=1e-3,
            d=0.35,
            auto_d=False,
            json=True,
        )
        fracdiff_command(args_json)
        json_out = capsys.readouterr().out
        data = json.loads(json_out)
        assert data["ticker"] == "MSFT"
        assert data["selected_d"] == 0.35
        assert "retained_memory_corr" in data
        assert "signature" in data


def test_fracdiff_edge_cases_and_empty_series(capsys):
    """Verify robust degradation under short or empty histories."""
    # 1. Very short series (< 30)
    short_series = pd.Series([100.0, 101.0, 100.5, 102.0])
    res_short = find_optimal_d(short_series)
    assert res_short["optimal_d"] == 1.0
    assert res_short["is_stationary"] is False

    # 2. evaluate_fracdiff_profile with specific target_d
    receipt = evaluate_fracdiff_profile("NVDA", pd.Series(range(60)), auto_d=False, target_d=0.6)
    assert receipt.selected_d == 0.6
    assert receipt.ticker == "NVDA"

    # 3. CLI empty observations
    with patch("src.data.db.get_observations", return_value=pd.DataFrame()):
        args_empty = argparse.Namespace(
            ticker="NVDA",
            threshold=1e-4,
            d=None,
            auto_d=True,
            json=False,
        )
        fracdiff_command(args_empty)
        out = capsys.readouterr().out
        assert "No observation history found" in out
