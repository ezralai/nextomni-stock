"""Unit and integration test suite for Pástor & Stambaugh (2003) Systematic Liquidity Risk Engine."""

import argparse
import io
import json
import math
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.pastor_stambaugh import (
    PastorStambaughReceipt,
    classify_pastor_stambaugh_regime,
    compute_liquidity_beta,
    compute_stock_liquidity_gamma,
    evaluate_pastor_stambaugh_profile,
    extract_liquidity_innovations,
)
from src.cli import liquidity_beta_command


@pytest.fixture
def sample_liquidity_market_data():
    """Generate 80 sessions of synthetic asset and benchmark prices and volumes."""
    np.random.seed(42)
    n = 80
    r_mkt = np.random.normal(0.0004, 0.010, n)
    p_mkt = 400.0 * np.exp(np.cumsum(r_mkt))
    v_mkt = np.random.uniform(5000000, 15000000, n)

    # Asset with true beta_M = 1.25 and true beta_L = 0.85
    u_fake = np.random.normal(0.0, 1.0, n)
    eps = np.random.normal(0.0, 0.006, n)
    r_asset = 0.0003 + 1.25 * r_mkt + 0.85 * (u_fake * 0.01) + eps
    p_asset = 120.0 * np.exp(np.cumsum(r_asset))
    v_asset = np.random.uniform(500000, 2000000, n)

    return p_asset, v_asset, p_mkt, v_mkt


def test_compute_stock_liquidity_gamma_reversal() -> None:
    """Verify that order flow price impact produces negative reversal gamma (gamma < 0)."""
    np.random.seed(123)
    n = 60
    # Simulate return reversal where large buying is followed by negative next-day return
    r_m = np.random.normal(0.0002, 0.008, n)
    v = np.random.uniform(100000, 500000, n)
    p = np.full(n, 50.0)

    # Induce explicit reversal: r_{t+1}^e = -0.05 * sign(r_t^e) * v_norm
    r_excess = np.random.normal(0.0, 0.01, n)
    v_norm = v / np.mean(v)
    for t in range(n - 1):
        r_excess[t + 1] = -0.04 * np.sign(r_excess[t]) * v_norm[t] + np.random.normal(0.0, 0.002)

    r_stock = r_excess + r_m
    gamma = compute_stock_liquidity_gamma(
        returns=r_stock,
        market_returns=r_m,
        volumes=v,
        prices=p,
    )

    assert gamma < -0.01


def test_extract_liquidity_innovations_properties() -> None:
    """Verify that AR(2) filtering extracts standardized innovations."""
    np.random.seed(456)
    n = 60
    # Persistent AR(1) process L_t = 0.7 L_{t-1} + e_t
    L = np.zeros(n)
    for t in range(1, n):
        L[t] = 0.7 * L[t - 1] + np.random.normal(0.0, 0.5)

    innovations = extract_liquidity_innovations(L)

    assert len(innovations) == n
    # Scaled innovations should have std roughly 1.0
    assert np.std(innovations[2:], ddof=1) == pytest.approx(1.0, rel=0.15)
    assert abs(float(np.mean(innovations[2:]))) < 0.20


def test_compute_liquidity_beta_known_recovery() -> None:
    """Verify that two-factor OLS accurately recovers known beta_M and beta_L."""
    np.random.seed(789)
    n = 150
    rm = np.random.normal(0.0003, 0.012, n)
    u = np.random.normal(0.0, 1.0, n)

    true_alpha = 0.0004
    true_beta_m = 1.10
    true_beta_l = 0.75
    noise = np.random.normal(0.0, 0.002, n)

    rf_daily = 0.04 / 252.0
    y = rf_daily + true_alpha + true_beta_m * (rm - rf_daily) + true_beta_l * (u * 0.01) + noise

    alpha_ann, b_m, b_l, t_stat, is_sig, p_val, r_sq = compute_liquidity_beta(
        asset_returns=y,
        market_returns=rm,
        liquidity_innovations=u * 0.01,
        risk_free_rate_ann=0.04,
    )

    assert b_m == pytest.approx(true_beta_m, rel=0.08)
    assert b_l == pytest.approx(true_beta_l, rel=0.12)
    assert r_sq > 0.80
    assert is_sig is True
    assert p_val < 0.05


def test_classify_pastor_stambaugh_regime() -> None:
    """Verify systematic liquidity risk regime classification and premium calculation."""
    # High liquidity beta (> 0.50)
    regime, prem = classify_pastor_stambaugh_regime(0.80)
    assert regime == "HIGH_LIQUIDITY_BETA_PREMIUM"
    assert prem == pytest.approx(6.00, abs=1e-2)

    # Neutral liquidity profile (-0.20 <= beta <= 0.50)
    regime, prem = classify_pastor_stambaugh_regime(0.20)
    assert regime == "NEUTRAL_LIQUIDITY_PROFILE"
    assert prem == pytest.approx(1.50, abs=1e-2)

    # Defensive liquidity hedge (< -0.20)
    regime, prem = classify_pastor_stambaugh_regime(-0.40)
    assert regime == "LIQUIDITY_HEDGE_DEFENSIVE"
    assert prem == pytest.approx(-3.00, abs=1e-2)


def test_evaluate_pastor_stambaugh_profile_integration(
    sample_liquidity_market_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify complete end-to-end Pástor-Stambaugh evaluation workflow."""
    p_a, v_a, p_m, v_m = sample_liquidity_market_data

    receipt = evaluate_pastor_stambaugh_profile(
        ticker="AAPL",
        asset_prices=p_a,
        asset_volumes=v_a,
        benchmark_ticker="SPY",
        benchmark_prices=p_m,
        benchmark_volumes=v_m,
        lookback_window=60,
    )

    assert isinstance(receipt, PastorStambaughReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.benchmark_ticker == "SPY"
    assert receipt.lookback_window == 60
    assert receipt.market_beta > 0.5
    assert isinstance(receipt.liquidity_beta, float)
    assert isinstance(receipt.liquidity_risk_regime, str)
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_detection(
    sample_liquidity_market_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    p_a, v_a, p_m, v_m = sample_liquidity_market_data

    receipt = evaluate_pastor_stambaugh_profile(
        ticker="NVDA",
        asset_prices=p_a,
        asset_volumes=v_a,
        benchmark_ticker="SPY",
        benchmark_prices=p_m,
        benchmark_volumes=v_m,
    )

    assert receipt.verify_integrity() is True

    # Tamper with liquidity beta
    receipt.liquidity_beta += 0.5
    assert receipt.verify_integrity() is False

    # Reset and tamper with expected premium
    receipt.liquidity_beta -= 0.5
    receipt.expected_liquidity_premium_pct += 2.0
    assert receipt.verify_integrity() is False


def test_security_input_validation() -> None:
    """Verify input validation against invalid tickers, non-finite parameters, and short history."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_pastor_stambaugh_profile(
            ticker="INVALID!TICKER",
            asset_prices=[100.0] * 30,
            asset_volumes=[1000.0] * 30,
        )

    # Insufficient observations
    with pytest.raises(ValueError, match="Insufficient asset price/volume history"):
        evaluate_pastor_stambaugh_profile(
            ticker="AAPL",
            asset_prices=[100.0, 101.0],
            asset_volumes=[1000.0, 1000.0],
        )

    # Non-finite parameter
    with pytest.raises(ValueError, match="lookback_window must be finite"):
        evaluate_pastor_stambaugh_profile(
            ticker="AAPL",
            asset_prices=[100.0] * 30,
            asset_volumes=[1000.0] * 30,
            benchmark_prices=[200.0] * 30,
            benchmark_volumes=[5000.0] * 30,
            lookback_window=float("nan"),
        )


def test_cli_liquidity_beta_command_stdout() -> None:
    """Verify CLI command standard formatted output."""
    np.random.seed(42)
    n = 70
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, n)))
    volumes = np.random.uniform(1000000, 3000000, n)
    df_obs = pd.DataFrame({"adj_close": prices, "volume": volumes})

    args = argparse.Namespace(
        ticker="AAPL",
        benchmark="SPY",
        lookback=60,
        json=False,
    )

    with patch("src.data.db.get_observations", return_value=df_obs):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            liquidity_beta_command(args)
            output = mock_stdout.getvalue()

            assert "Pástor & Stambaugh (2003) Systematic Liquidity Risk Engine" in output
            assert "AAPL" in output
            assert "SPY" in output
            assert "Liquidity Beta" in output
            assert "Audit Digest (HMAC)" in output


def test_cli_liquidity_beta_command_json() -> None:
    """Verify CLI command JSON output serialization."""
    np.random.seed(42)
    n = 70
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, n)))
    volumes = np.random.uniform(1000000, 3000000, n)
    df_obs = pd.DataFrame({"adj_close": prices, "volume": volumes})

    args = argparse.Namespace(
        ticker="AAPL",
        benchmark="SPY",
        lookback=60,
        json=True,
    )

    with patch("src.data.db.get_observations", return_value=df_obs):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            liquidity_beta_command(args)
            output = mock_stdout.getvalue()

            data = json.loads(output)
            assert data["ticker"] == "AAPL"
            assert data["benchmark_ticker"] == "SPY"
            assert "signature" in data
            assert "liquidity_beta" in data
            assert "liquidity_risk_regime" in data
