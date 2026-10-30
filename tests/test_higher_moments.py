"""Unit and integration test suite for Harvey & Siddique (2000) and Dittmar (2002) Higher Moments Engine."""

import argparse
import io
import json
import math
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.higher_moments import (
    HigherMomentsReceipt,
    classify_higher_moment_risk_regime,
    compute_standardized_comoments,
    estimate_four_moment_capm_regression,
    evaluate_higher_moments_profile,
)
from src.cli import higher_moments_command


@pytest.fixture
def sample_market_and_skewed_asset():
    """Generate 100 sessions of synthetic market and asset returns with known higher moments."""
    np.random.seed(42)
    n = 100
    # Zero-skew market returns
    rm = np.random.normal(0.0003, 0.012, n)
    dev_m = rm - np.mean(rm)
    quad_dev = dev_m**2 - np.mean(dev_m**2)

    # Negative co-skewness asset: explicitly drops when market variance is high
    ri = 0.0003 + 1.1 * rm - 40.0 * quad_dev + np.random.normal(0.0, 0.002, n)

    return pd.Series(ri), pd.Series(rm)


def test_compute_standardized_comoments_market_identity() -> None:
    """Verify mathematical identity: on benchmark itself, beta_SKS == Skew(M) and beta_CKT == Kurt(M)."""
    np.random.seed(123)
    n = 150
    # Synthetic series with known sample skewness and kurtosis
    rm = np.random.exponential(scale=0.012, size=n) - 0.012
    dev_m = rm - np.mean(rm)
    std_m = np.std(rm)

    expected_skew = float(np.mean(dev_m**3) / (std_m**3))
    expected_kurt = float(np.mean(dev_m**4) / (std_m**4))

    beta_m, beta_sks, beta_ckt = compute_standardized_comoments(rm, rm)

    assert beta_m == pytest.approx(1.0, rel=1e-5)
    assert beta_sks == pytest.approx(expected_skew, rel=1e-3)
    assert beta_ckt == pytest.approx(expected_kurt, rel=1e-3)


def test_compute_standardized_comoments_negative_coskewness(
    sample_market_and_skewed_asset: Tuple[pd.Series, pd.Series],
) -> None:
    """Verify that induced negative co-skewness yields beta_SKS < 0."""
    ri, rm = sample_market_and_skewed_asset
    beta_m, beta_sks, beta_ckt = compute_standardized_comoments(ri.values, rm.values)

    assert beta_m > 0.8
    assert beta_sks < -0.10  # Confirms negative co-skewness
    assert beta_ckt > 0.0


def test_estimate_four_moment_capm_regression_properties(
    sample_market_and_skewed_asset: Tuple[pd.Series, pd.Series],
) -> None:
    """Verify Four-Moment CAPM regression output structure, degrees of freedom, and R^2 bounds."""
    ri, rm = sample_market_and_skewed_asset
    (
        alpha_ann,
        beta_m,
        gamma_sks,
        t_stat_sks,
        p_val_sks,
        delta_ckt,
        t_stat_ckt,
        p_val_ckt,
        r_sq,
    ) = estimate_four_moment_capm_regression(ri.values, rm.values)

    assert isinstance(alpha_ann, float)
    assert beta_m > 0.5
    assert 0.0 <= r_sq <= 1.0
    assert 0.0 <= p_val_sks <= 1.0
    assert 0.0 <= p_val_ckt <= 1.0
    assert not math.isnan(t_stat_sks)
    assert not math.isnan(t_stat_ckt)


def test_classify_higher_moment_risk_regime() -> None:
    """Verify higher-moment regime sorting and expected co-skewness premium calculation."""
    # Severe crash vulnerability (negative co-skewness, fat-tail co-kurtosis)
    regime, prem = classify_higher_moment_risk_regime(-0.35, 1.45)
    assert regime == "SEVERE_CRASH_VULNERABILITY"
    assert prem > 1.0  # Commands positive risk premium

    # Asymmetric hedge (positive co-skewness)
    regime, prem = classify_higher_moment_risk_regime(0.25, 0.90)
    assert regime == "ASYMMETRIC_TURBULENCE_HEDGE"
    assert prem < 0.0

    # Neutral Gaussian profile
    regime, prem = classify_higher_moment_risk_regime(0.02, 1.05)
    assert regime == "NEUTRAL_GAUSSIAN_PROFILE"


def test_evaluate_higher_moments_profile_integration() -> None:
    """Verify complete end-to-end higher moments evaluation workflow with price series."""
    np.random.seed(999)
    n = 80
    r_bench = np.random.normal(0.0004, 0.010, n)
    r_asset = 0.0002 + 1.15 * r_bench + np.random.normal(0.0, 0.005, n)
    prices_bench = 420.0 * np.exp(np.cumsum(r_bench))
    prices_asset = 120.0 * np.exp(np.cumsum(r_asset))

    receipt = evaluate_higher_moments_profile(
        ticker="AAPL",
        asset_prices=prices_asset,
        benchmark_ticker="SPY",
        benchmark_prices=prices_bench,
        lookback_window=60,
    )

    assert isinstance(receipt, HigherMomentsReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.benchmark_ticker == "SPY"
    assert receipt.lookback_window == 60
    assert receipt.market_beta > 0.5
    assert isinstance(receipt.coskewness_beta, float)
    assert isinstance(receipt.cokurtosis_beta, float)
    assert receipt.four_moment_r_squared >= 0.0
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    np.random.seed(333)
    prices_asset = 180.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.014, 60)))
    prices_bench = 440.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.009, 60)))

    receipt = evaluate_higher_moments_profile(
        ticker="NVDA",
        asset_prices=prices_asset,
        benchmark_ticker="SPY",
        benchmark_prices=prices_bench,
    )

    assert receipt.verify_integrity() is True

    # Tamper with co-skewness beta
    receipt.coskewness_beta += 0.2
    assert receipt.verify_integrity() is False

    # Reset and tamper with co-kurtosis beta
    receipt.coskewness_beta -= 0.2
    receipt.cokurtosis_beta += 0.5
    assert receipt.verify_integrity() is False


def test_security_input_validation() -> None:
    """Verify defense-in-depth against invalid tickers, non-finite parameters, and short history."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_higher_moments_profile(ticker="INVALID!TICKER", asset_prices=[100.0] * 30)

    # Insufficient observations
    with pytest.raises(ValueError, match="Insufficient asset price observations"):
        evaluate_higher_moments_profile(ticker="AAPL", asset_prices=[100.0, 101.0])

    # Non-finite parameter
    with pytest.raises(ValueError, match="lookback_window must be finite"):
        evaluate_higher_moments_profile(
            ticker="AAPL",
            asset_prices=[100.0] * 30,
            benchmark_prices=[200.0] * 30,
            lookback_window=float("nan"),
        )


def test_cli_higher_moments_command_stdout() -> None:
    """Verify CLI command standard formatted output."""
    np.random.seed(42)
    n = 70
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, n)))
    df_obs = pd.DataFrame({"adj_close": prices})

    args = argparse.Namespace(
        ticker="AAPL",
        benchmark="SPY",
        lookback=60,
        json=False,
    )

    with patch("src.data.db.get_observations", return_value=df_obs):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            higher_moments_command(args)
            output = mock_stdout.getvalue()

            assert "Harvey-Siddique & Dittmar Higher-Moment Asset Pricing" in output
            assert "AAPL" in output
            assert "SPY" in output
            assert "Co-Skewness" in output
            assert "Co-Kurtosis" in output
            assert "Audit Digest (HMAC)" in output


def test_cli_higher_moments_command_json() -> None:
    """Verify CLI command JSON output serialization."""
    np.random.seed(42)
    n = 70
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, n)))
    df_obs = pd.DataFrame({"adj_close": prices})

    args = argparse.Namespace(
        ticker="AAPL",
        benchmark="SPY",
        lookback=60,
        json=True,
    )

    with patch("src.data.db.get_observations", return_value=df_obs):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            higher_moments_command(args)
            output = mock_stdout.getvalue()

            data = json.loads(output)
            assert data["ticker"] == "AAPL"
            assert data["benchmark_ticker"] == "SPY"
            assert "coskewness_beta" in data
            assert "cokurtosis_beta" in data
            assert "signature" in data
