"""Unit and integration test suite for Bollerslev, Tauchen & Zhou (2009) Variance Risk Premium (VRP) Engine."""

import argparse
import io
import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.variance_risk_premium import (
    VarianceRiskPremiumReceipt,
    classify_vrp_regime,
    compute_newey_west_ols,
    evaluate_variance_risk_premium_profile,
)
from src.cli import vrp_command


def test_vrp_positive_crash_insurance_premium() -> None:
    """Verify standard positive variance risk premium where IV exceeds RV."""
    # Synthetic prices with ~18% realized volatility
    np.random.seed(42)
    n = 100
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.18 / math.sqrt(252.0), n)))

    receipt = evaluate_variance_risk_premium_profile(
        ticker="SPY",
        asset_prices=prices,
        implied_volatility=0.28,  # 28% IV
        horizon_sessions=21,
        lookback_window=80,
    )

    # Realized vol should be around 18% +/- 5%
    assert 12.0 < receipt.realized_volatility_pct < 25.0
    # Implied vol is 28%
    assert receipt.implied_volatility_pct == pytest.approx(28.0, rel=1e-5)
    # Positive VRP
    assert receipt.variance_risk_premium > 0.0
    assert receipt.volatility_spread_pct_points > 0.0
    assert receipt.vrp_regime in ("ELEVATED_CRASH_INSURANCE_PREMIUM", "NORMAL_RISK_PREMIUM_EQUILIBRIUM")


def test_vrp_inverted_crash_shock() -> None:
    """Verify inverted VRP during acute volatility surge where realized exceeds implied."""
    # High turbulence returns with 50% realized volatility
    np.random.seed(123)
    n = 80
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(-0.001, 0.50 / math.sqrt(252.0), n)))

    receipt = evaluate_variance_risk_premium_profile(
        ticker="TSLA",
        asset_prices=prices,
        implied_volatility=0.25,  # 25% IV (substantially below 50% RV)
        horizon_sessions=21,
        lookback_window=60,
    )

    # Realized volatility exceeds implied volatility
    assert receipt.realized_volatility_pct > receipt.implied_volatility_pct
    assert receipt.variance_risk_premium < 0.0
    assert receipt.volatility_spread_pct_points < 0.0
    assert receipt.vrp_regime == "INVERTED_CRASH_TURBULENCE"


def test_vrp_regime_classification_tiers() -> None:
    """Verify institutional VRP regime tier boundaries."""
    assert classify_vrp_regime(4.5, 6.0) == "ELEVATED_CRASH_INSURANCE_PREMIUM"
    assert classify_vrp_regime(1.5, 2.5) == "NORMAL_RISK_PREMIUM_EQUILIBRIUM"
    assert classify_vrp_regime(0.2, 0.3) == "COMPLACENT_UNDERPRICED_RISK"
    assert classify_vrp_regime(-1.5, -2.0) == "INVERTED_CRASH_TURBULENCE"
    assert classify_vrp_regime(1.0, -0.5) == "INVERTED_CRASH_TURBULENCE"


def test_compute_newey_west_ols() -> None:
    """Verify Newey-West HAC covariance estimation on autocorrelated data."""
    np.random.seed(42)
    n = 300
    x = np.random.normal(0.0, 1.0, n)
    # Autocorrelated error process e_t = 0.3 * e_{t-1} + u_t
    u = np.random.normal(0, 0.01, n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.3 * e[t - 1] + u[t]

    # True model: y = 0.01 + 0.005 * x + e
    y = 0.01 + 0.005 * x + e

    alpha, beta, t_stat, p_val, r2_pct = compute_newey_west_ols(x, y, lags=5)

    assert beta == pytest.approx(0.005, abs=0.004)
    assert t_stat > 0.0
    assert 0.0 <= p_val <= 1.0
    assert r2_pct >= 0.0


def test_volatility_of_volatility_bounds() -> None:
    """Verify that Vol-of-Vol (VoV) correctly measures second-order variance dispersion."""
    np.random.seed(55)
    n = 120
    # Prices with varying volatility regime
    ret1 = np.random.normal(0.0005, 0.01, 60)
    ret2 = np.random.normal(-0.0005, 0.03, 60)
    rets = np.concatenate([ret1, ret2])
    prices = 100.0 * np.exp(np.cumsum(rets))

    receipt = evaluate_variance_risk_premium_profile(
        ticker="QQQ",
        asset_prices=prices,
        horizon_sessions=15,
        lookback_window=100,
    )

    assert receipt.volatility_of_volatility_pct > 1.0
    assert receipt.volatility_of_volatility_pct < 100.0


def test_vrp_receipt_tamper_integrity() -> None:
    """Verify HMAC-SHA256 signature generation and tamper rejection."""
    np.random.seed(77)
    n = 80
    prices = 150.0 + np.cumsum(np.random.normal(0.05, 1.0, n))

    receipt = evaluate_variance_risk_premium_profile(
        ticker="AAPL",
        asset_prices=prices,
        implied_volatility=0.26,
        horizon_sessions=21,
    )

    assert receipt.verify_integrity() is True
    assert len(receipt.signature) == 64

    # Tampering with variance risk premium invalidates signature
    receipt.variance_risk_premium += 1.0
    assert receipt.verify_integrity() is False


def test_vrp_cli_command_plain(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with formatted terminal report."""
    np.random.seed(88)
    n = 90
    prices = 200.0 + np.cumsum(np.random.normal(0.0, 1.5, n))

    with patch("src.data.db.get_observations") as mock_db:
        mock_df = pd.DataFrame({"adj_close": prices})
        mock_db.return_value = mock_df

        args = argparse.Namespace(
            command="vrp",
            ticker="GOOGL",
            iv=0.27,
            horizon=21,
            lookback=80,
            json=False,
        )

        vrp_command(args)
        captured = capsys.readouterr().out

        assert "Bollerslev et al. (2009) Variance Risk Premium Engine" in captured
        assert "Security / Ticker:            GOOGL" in captured
        assert "Variance Risk Premium (VRP):" in captured
        assert "Volatility Spread:" in captured
        assert "Bollerslev Forward Return Prediction" in captured
        assert "Audit Digest (HMAC):" in captured
        assert "Verified: True" in captured


def test_vrp_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with JSON output."""
    np.random.seed(88)
    n = 90
    prices = 130.0 + np.cumsum(np.random.normal(0.0, 1.2, n))

    with patch("src.data.db.get_observations") as mock_db:
        mock_df = pd.DataFrame({"adj_close": prices})
        mock_db.return_value = mock_df

        args = argparse.Namespace(
            command="vrp",
            ticker="NVDA",
            iv=0.35,
            horizon=21,
            lookback=80,
            json=True,
        )

        vrp_command(args)
        captured = capsys.readouterr().out
        data = json.loads(captured)

        assert data["ticker"] == "NVDA"
        assert "variance_risk_premium" in data
        assert "volatility_spread_pct_points" in data
        assert "signature" in data


def test_vrp_input_validation() -> None:
    """Verify input validation guards against invalid tickers or parameters."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        evaluate_variance_risk_premium_profile(ticker="BAD$$$")

    with pytest.raises(ValueError, match="Insufficient price history"):
        evaluate_variance_risk_premium_profile(ticker="AAPL", asset_prices=[100.0] * 10, horizon_sessions=21)
