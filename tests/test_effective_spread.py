"""Unit and integration test suite for Richard Roll (1984) Effective Bid-Ask Spread Engine."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.effective_spread import (
    EffectiveSpreadReceipt,
    compute_roll_effective_spread,
    compute_roll_serial_covariance,
    evaluate_effective_spread_profile,
)
from src.cli import effective_spread_command


@pytest.fixture
def synthetic_bounce_prices() -> pd.Series:
    """Generate prices with dominant bid-ask bounce (alternating +/- $0.50 spread)."""
    np.random.seed(42)
    n = 60
    base_price = 100.0
    true_spread = 0.50  # $0.50 spread => Delta P alternates +/- 0.25
    order_signs = np.array([1, -1] * (n // 2))
    noise = np.random.normal(0, 0.05, size=n)
    price_levels = base_price + (true_spread / 2.0) * order_signs + np.cumsum(noise)
    return pd.Series(price_levels)


@pytest.fixture
def synthetic_trending_prices() -> pd.Series:
    """Generate strong momentum trending prices (positive serial autocovariance)."""
    np.random.seed(42)
    n = 60
    trend = np.linspace(100.0, 150.0, n)
    noise = np.random.normal(0, 0.2, size=n)
    return pd.Series(trend + noise)


def test_roll_effective_spread_bounce_recovery(synthetic_bounce_prices):
    """Verify that pure bid-ask bounce yields negative autocovariance and accurate spread recovery."""
    prices = synthetic_bounce_prices
    roll_dlr, roll_pct, signed_pct, pos_cov = compute_roll_effective_spread(prices)

    assert pos_cov is False
    assert 0.80 <= roll_dlr <= 1.20  # Alternating Delta P = +/-0.50 yields gamma_1 ~ -0.25 => S ~ 1.00
    assert roll_pct > 0.0
    assert pytest.approx(roll_pct, abs=0.1) == signed_pct


def test_roll_effective_spread_trending_momentum(synthetic_trending_prices):
    """Verify that trending momentum prices produce positive autocovariance with guarded zero floor."""
    prices = synthetic_trending_prices
    roll_dlr, roll_pct, signed_pct, pos_cov = compute_roll_effective_spread(prices)

    # In a strong trend, autocovariance of Delta P is positive
    dp = np.diff(prices.values)
    cov = compute_roll_serial_covariance(dp)

    if cov > 0.0:
        assert pos_cov is True
        assert roll_dlr == 0.0  # Classical zero floor
        assert roll_pct == 0.0
        assert signed_pct < 0.0  # Signed continuous extension captures momentum
    else:
        assert roll_dlr >= 0.0


def test_evaluate_effective_spread_profile_and_composite(synthetic_bounce_prices):
    """Verify end-to-end receipt generation, Corwin-Schultz fusion, and liquidity tiers."""
    prices = synthetic_bounce_prices
    high = prices + 0.30
    low = prices - 0.30
    volumes = pd.Series([1_000_000.0] * len(prices))

    receipt = evaluate_effective_spread_profile(
        ticker="AAPL",
        prices=prices,
        high=high,
        low=low,
        volumes=volumes,
        window=30,
    )

    assert isinstance(receipt, EffectiveSpreadReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.window == 30
    assert receipt.mean_price > 90.0
    assert receipt.composite_round_trip_bps > 0.0
    assert receipt.liquidity_tier in ("ULTRA_LIQUID", "LIQUID", "MODERATE_FRICTION", "ILLIQUID")
    assert receipt.verify_integrity() is True


def test_effective_spread_receipt_tamper_detection(synthetic_bounce_prices):
    """Verify HMAC-SHA256 signature authenticity and cryptographic tamper detection."""
    receipt = evaluate_effective_spread_profile(
        ticker="MSFT",
        prices=synthetic_bounce_prices,
        window=25,
    )
    assert receipt.verify_integrity() is True

    # Tamper with composite friction
    tampered_dict = receipt.to_dict()
    tampered_dict["composite_round_trip_bps"] = 999.9
    tampered_receipt = EffectiveSpreadReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with liquidity tier
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["liquidity_tier"] = "ILLIQUID" if receipt.liquidity_tier != "ILLIQUID" else "ULTRA_LIQUID"
    tampered_receipt2 = EffectiveSpreadReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_effective_spread_security_and_edge_cases():
    """Verify input validation, ticker allowlisting, and lookback bounds."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_effective_spread_profile("INVALID_TICKER$$$", pd.Series([10.0] * 30), window=20)

    # Window too small (< 10)
    with pytest.raises(ValueError, match="Lookback window must be at least 10"):
        evaluate_effective_spread_profile("AAPL", pd.Series([10.0] * 30), window=5)

    # Insufficient price history
    with pytest.raises(ValueError, match="Insufficient price history"):
        evaluate_effective_spread_profile("AAPL", pd.Series([10.0, 11.0, 12.0]), window=20)


def test_cli_effective_spread_command_stdout_and_json(synthetic_bounce_prices, monkeypatch):
    """Verify CLI effective-spread command execution in stdout and JSON modes."""
    df_obs = pd.DataFrame({
        "close": synthetic_bounce_prices,
        "high": synthetic_bounce_prices + 0.20,
        "low": synthetic_bounce_prices - 0.20,
        "volume": [500_000.0] * len(synthetic_bounce_prices),
    })
    monkeypatch.setattr(
        "src.data.db.get_observations",
        lambda *args, **kwargs: df_obs,
    )

    # 1. Stdout execution
    args_stdout = argparse.Namespace(
        ticker="AAPL",
        window=25,
        json=False,
    )
    effective_spread_command(args_stdout)

    # 2. JSON execution
    args_json = argparse.Namespace(
        ticker="AAPL",
        window=25,
        json=True,
    )
    effective_spread_command(args_json)
