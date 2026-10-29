"""Test suite for Kyle's (1985) Informed Trading Lambda & Price Impact Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.price_impact import (
    PriceImpactReceipt,
    compute_square_root_impact,
    estimate_kyle_lambda,
    evaluate_price_impact_profile,
)
from src.cli import price_impact_command


@pytest.fixture
def synthetic_microstructure_series():
    """Create reproducible synthetic daily price and volume series."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Base price random walk with drift
    ret = np.random.normal(0.0005, 0.015, size=n)
    prices = 100.0 * np.exp(np.cumsum(ret))
    # Volume with volume spikes on high returns
    base_vol = 1_000_000 + (np.abs(ret) * 10_000_000)
    volumes = np.random.normal(base_vol, 200_000)
    volumes = np.maximum(volumes, 100_000)

    s_prices = pd.Series(prices, index=dates)
    s_volumes = pd.Series(volumes, index=dates)
    return s_prices, s_volumes


def test_estimate_kyle_lambda_mathematical_properties(synthetic_microstructure_series):
    """Verify Kyle's lambda slope non-negativity, depth inverse, and R-squared bounds."""
    prices, volumes = synthetic_microstructure_series
    lam_pct, lam_abs, r2, depth = estimate_kyle_lambda(prices, volumes, window=60)

    assert lam_pct >= 0.0
    assert lam_abs >= 0.0
    assert 0.0 <= r2 <= 1.0
    assert depth > 0.0


def test_compute_square_root_impact_scaling():
    """Verify Bouchaud et al. square-root law of market impact scaling."""
    sigma = 0.02
    adv = 1_000_000.0

    impact_1pct = compute_square_root_impact(sigma, adv, 0.01)
    impact_4pct = compute_square_root_impact(sigma, adv, 0.04)

    # 4x volume fraction must result in approximately 2x impact (sqrt(4) = 2)
    ratio = impact_4pct / impact_1pct
    assert abs(ratio - 2.0) < 0.05
    assert impact_1pct > 0.0


def test_evaluate_price_impact_receipt_and_tamper(synthetic_microstructure_series):
    """Verify receipt creation, regime categorization, and HMAC signature tamper resistance."""
    prices, volumes = synthetic_microstructure_series
    receipt = evaluate_price_impact_profile("AAPL", prices, volumes, window=50)

    assert isinstance(receipt, PriceImpactReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.sample_sessions == 100
    assert receipt.rolling_window == 50
    assert receipt.adverse_selection_regime in (
        "LOW_ADVERSE_SELECTION_HIGH_LIQUIDITY",
        "MODERATE_IMPACT_NORMAL_LIQUIDITY",
        "HIGH_ADVERSE_SELECTION_FRAGILE",
    )
    assert receipt.verify_integrity() is True

    # Tamper with lambda
    tampered_dict = receipt.to_dict()
    tampered_dict["kyle_lambda_pct_per_million"] += 0.5
    tampered_receipt = PriceImpactReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False


def test_price_impact_security_and_edge_cases():
    """Verify input validation, zero volume, and insufficient observations."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_price_impact_profile("BAD_TICKER$$$", pd.Series([10.0]), pd.Series([100.0]))

    # Insufficient observations
    short_p = pd.Series([10.0, 11.0, 12.0])
    short_v = pd.Series([100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="Insufficient aligned price and volume observations"):
        evaluate_price_impact_profile("AAPL", short_p, short_v)

    # Flat prices/volume handling in estimate_kyle_lambda
    flat_p = pd.Series([50.0] * 30)
    flat_v = pd.Series([1000.0] * 30)
    lam_pct, lam_abs, r2, depth = estimate_kyle_lambda(flat_p, flat_v, window=25)
    assert lam_pct == 0.0
    assert lam_abs == 0.0


def test_cli_price_impact_command_stdout_and_json(synthetic_microstructure_series, monkeypatch):
    """Verify CLI price-impact command stdout formatting and JSON output."""
    prices, volumes = synthetic_microstructure_series
    df_obs = pd.DataFrame({"close": prices, "volume": volumes})

    monkeypatch.setattr("src.data.db.get_observations", lambda ticker, db_path=None: df_obs)

    # 1. Standard stdout mode
    args_std = argparse.Namespace(ticker="AAPL", window=50, json=False)
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        price_impact_command(args_std)
        output = fake_out.getvalue()
        assert "Kyle's (1985) Informed Trading Lambda & Price Impact Engine" in output
        assert "Security / Ticker:            AAPL" in output
        assert "Square-Root Law Projected Execution Price Impact:" in output
        assert "Audit Digest (HMAC):" in output

    # 2. JSON mode
    args_json = argparse.Namespace(ticker="AAPL", window=50, json=True)
    with patch("sys.stdout", new=io.StringIO()) as fake_json_out:
        price_impact_command(args_json)
        payload = json.loads(fake_json_out.getvalue())
        assert payload["ticker"] == "AAPL"
        assert "kyle_lambda_pct_per_million" in payload
        assert "market_depth_millions" in payload
        assert "signature" in payload
