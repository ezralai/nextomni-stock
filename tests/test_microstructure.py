"""Unit and integration tests for market microstructure liquidity feature store."""

import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.features.builder import (
    ALL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    build_feature_matrix,
)
from src.features.microstructure import (
    LiquidityReceipt,
    compute_amihud_illiquidity,
    compute_corwin_schultz_spread,
    compute_kyle_lambda_proxy,
    evaluate_liquidity_profile,
)


@pytest.fixture
def sample_ohlcv_df():
    """Generate 60 sessions of synthetic OHLCV data."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    returns = np.random.normal(0.0005, 0.012, n)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1.0 + r))

    prices = np.array(prices)
    highs = prices * (1.0 + np.abs(np.random.normal(0.005, 0.003, n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0.005, 0.003, n)))
    volumes = np.random.randint(50000, 200000, n).astype(float)

    return pd.DataFrame({
        "session_date": dates,
        "open": prices * 0.999,
        "high": highs,
        "low": lows,
        "close": prices,
        "adj_close": prices,
        "volume": volumes,
    })


def test_compute_amihud_illiquidity_bounds(sample_ohlcv_df):
    """Verify Amihud illiquidity is strictly non-negative and finite."""
    prices = sample_ohlcv_df["adj_close"]
    returns = prices.pct_change().fillna(0.0)
    volumes = sample_ohlcv_df["volume"]

    illiq = compute_amihud_illiquidity(returns, prices, volumes, window=15)
    assert len(illiq) == len(prices)
    assert (illiq >= 0.0).all()
    assert np.isfinite(illiq).all()


def test_compute_amihud_illiquidity_zero_volume_guard():
    """Verify zero volume does not produce ZeroDivisionError or NaNs."""
    returns = pd.Series([0.01, -0.02, 0.0])
    prices = pd.Series([100.0, 98.0, 98.0])
    volumes = pd.Series([0.0, 0.0, 0.0])

    illiq = compute_amihud_illiquidity(returns, prices, volumes, window=2)
    assert len(illiq) == 3
    assert np.isfinite(illiq).all()


def test_compute_corwin_schultz_spread_bounds(sample_ohlcv_df):
    """Verify Corwin-Schultz spread values are bounded within [0, 1]."""
    high = sample_ohlcv_df["high"]
    low = sample_ohlcv_df["low"]

    spread = compute_corwin_schultz_spread(high, low, window=15)
    assert len(spread) == len(high)
    assert (spread >= 0.0).all()
    assert (spread <= 1.0).all()
    assert np.isfinite(spread).all()


def test_compute_corwin_schultz_spread_identical_high_low():
    """Verify zero spread when High equals Low."""
    high = pd.Series([100.0] * 30)
    low = pd.Series([100.0] * 30)

    spread = compute_corwin_schultz_spread(high, low, window=10)
    assert np.isclose(spread.iloc[-1], 0.0)


def test_compute_kyle_lambda_proxy(sample_ohlcv_df):
    """Verify Kyle's lambda proxy calculates valid positive sensitivity."""
    prices = sample_ohlcv_df["adj_close"]
    returns = prices.pct_change().fillna(0.0)
    volumes = sample_ohlcv_df["volume"]

    k_lambda = compute_kyle_lambda_proxy(returns, volumes, window=15)
    assert len(k_lambda) == len(prices)
    assert (k_lambda >= 0.0).all()


def test_evaluate_liquidity_profile(sample_ohlcv_df):
    """Verify complete liquidity diagnostic receipt calculation and classification."""
    receipt = evaluate_liquidity_profile(sample_ohlcv_df, ticker="JPM")
    assert isinstance(receipt, LiquidityReceipt)
    assert receipt.ticker == "JPM"
    assert receipt.sample_size == 60
    assert receipt.latest_corwin_schultz_spread_pct >= 0.0
    assert receipt.latest_amihud_illiq >= 0.0
    assert receipt.liquidity_regime in {"HIGH_LIQUIDITY", "MODERATE_LIQUIDITY", "LOW_LIQUIDITY_DRYUP"}


def test_feature_builder_microstructure_backward_compatibility(sample_ohlcv_df):
    """Verify build_feature_matrix supports both base 12 features and opt-in 14 features."""
    # Default: base 12 features for backward compatibility
    df_base, cols_base = build_feature_matrix(sample_ohlcv_df, include_microstructure=False)
    assert cols_base == FEATURE_COLUMNS
    assert len(cols_base) == 12

    # Opt-in: all 14 features including microstructure
    df_all, cols_all = build_feature_matrix(sample_ohlcv_df, include_microstructure=True)
    assert cols_all == ALL_FEATURE_COLUMNS
    assert len(cols_all) == 14
    for m_col in MICROSTRUCTURE_COLUMNS:
        assert m_col in cols_all
        assert m_col in df_all.columns
        assert not df_all[m_col].isna().any()


def test_cli_liquidity_report_command(capsys, sample_ohlcv_df):
    """Verify CLI nextomni liquidity-report outputs structured JSON."""
    from src.cli import main
    import sys

    with patch("src.data.db.get_observations", return_value=sample_ohlcv_df), \
         patch.object(sys, "argv", ["nextomni", "liquidity-report", "--ticker", "JPM", "--json"]):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ticker"] == "JPM"
    assert "latest_amihud_illiq" in data
    assert "latest_corwin_schultz_spread_pct" in data
    assert "liquidity_regime" in data
