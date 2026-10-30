"""Unit and integration test suite for Robert C. Merton (1974) Structural Credit Risk & Distance-to-Default Engine."""

import argparse
import io
import json
import math
from typing import List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.merton_credit import (
    MertonCreditReceipt,
    classify_merton_distress_regime,
    compute_rolling_distance_to_default,
    evaluate_merton_credit_profile,
    solve_merton_model,
)
from src.cli import merton_credit_command


def test_merton_solvent_prime_firm() -> None:
    """Verify structural solvency inversion on a high-capitalization low-debt prime corporation."""
    # Firm: E = $2.5T, D = $110B, sigma_E = 25%, r = 4.5%, T = 1.0 yr
    e_val = 2.5e12
    d_val = 1.1e11
    sig_e = 0.25
    r = 0.045
    t = 1.0

    (
        v_a,
        sigma_a,
        dd,
        rn_dd,
        pi_def,
        spread_bps,
        leverage_pct,
        regime,
    ) = solve_merton_model(
        equity_value=e_val,
        equity_volatility=sig_e,
        debt_value=d_val,
        risk_free_rate_ann=r,
        horizon_years=t,
    )

    # Asset value should exceed equity value
    assert v_a > e_val
    assert v_a == pytest.approx(e_val + d_val * math.exp(-r * t), rel=0.05)

    # Asset volatility should be strictly less than equity volatility (unlevered)
    assert 0.01 < sigma_a < sig_e
    assert sigma_a == pytest.approx(sig_e * (e_val / v_a), rel=0.05)

    # Prime firm should exhibit massive distance to default and negligible default probability
    assert dd > 5.0
    assert rn_dd > 5.0
    assert pi_def < 0.001
    assert spread_bps < 5.0
    assert leverage_pct < 10.0
    assert regime == "PRIME_SOLVENT"


def test_merton_distressed_high_leverage() -> None:
    """Verify structural default probability detection on a distressed, high-leverage firm."""
    # Distressed Firm: E = $100M, D = $500M (83% leverage), sigma_E = 80%, r = 4.5%
    e_val = 1.0e8
    d_val = 5.0e8
    sig_e = 0.80
    r = 0.045
    t = 1.0

    (
        v_a,
        sigma_a,
        dd,
        rn_dd,
        pi_def,
        spread_bps,
        leverage_pct,
        regime,
    ) = solve_merton_model(
        equity_value=e_val,
        equity_volatility=sig_e,
        debt_value=d_val,
        risk_free_rate_ann=r,
        horizon_years=t,
    )

    assert v_a > e_val
    assert leverage_pct > 75.0
    assert dd < 2.0  # Near default threshold
    assert pi_def > 5.0  # Material default probability > 5%
    assert spread_bps > 50.0  # High credit spread
    assert regime in ("HIGH_DISTRESS_RISK", "CRITICAL_DEFAULT_IMMINENT")


def test_merton_monotonicity_properties() -> None:
    """Verify fundamental option-theoretic monotonicity theorems of structural credit risk."""
    e_val = 1.0e8
    sig_e = 0.35
    r = 0.04
    t = 1.0

    # 1. Monotonicity in Debt Liabilities (holding Equity and Volatility constant)
    debts = [5.0e7, 1.0e8, 2.5e8, 5.0e8]
    dds: List[float] = []
    pi_defs: List[float] = []
    spreads: List[float] = []
    leverages: List[float] = []

    for d in debts:
        _, _, dd, _, pi_def, spread_bps, lev, _ = solve_merton_model(
            equity_value=e_val,
            equity_volatility=sig_e,
            debt_value=d,
            risk_free_rate_ann=r,
            horizon_years=t,
        )
        dds.append(dd)
        pi_defs.append(pi_def)
        spreads.append(spread_bps)
        leverages.append(lev)

    # Higher debt implies strictly lower distance-to-default
    for i in range(len(debts) - 1):
        assert dds[i] > dds[i + 1]
        assert pi_defs[i] < pi_defs[i + 1]
        assert spreads[i] <= spreads[i + 1]
        assert leverages[i] < leverages[i + 1]

    # 2. Monotonicity in Equity Volatility (holding Equity and Debt constant)
    vols = [0.15, 0.30, 0.50, 0.80]
    vol_dds: List[float] = []
    vol_pis: List[float] = []

    for v in vols:
        _, _, dd, _, pi_def, _, _, _ = solve_merton_model(
            equity_value=e_val,
            equity_volatility=v,
            debt_value=1.0e8,
            risk_free_rate_ann=r,
            horizon_years=t,
        )
        vol_dds.append(dd)
        vol_pis.append(pi_def)

    # Higher equity volatility implies strictly lower distance to default and higher default risk
    for i in range(len(vols) - 1):
        assert vol_dds[i] > vol_dds[i + 1]
        assert vol_pis[i] < vol_pis[i + 1]


def test_merton_zero_debt_edge_case() -> None:
    """Verify all-equity unconstrained solvency when debt liabilities are zero."""
    v_a, sig_a, dd, rn_dd, pi_def, spread, lev, regime = solve_merton_model(
        equity_value=5.0e8,
        equity_volatility=0.25,
        debt_value=0.0,
    )
    assert v_a == 5.0e8
    assert sig_a == 0.25
    assert dd == 999.0
    assert rn_dd == 999.0
    assert pi_def == 0.0
    assert spread == 0.0
    assert lev == 0.0
    assert regime == "PRIME_SOLVENT"


def test_merton_insolvent_firm_edge_case() -> None:
    """Verify insolvent equity state handling."""
    v_a, sig_a, dd, rn_dd, pi_def, spread, lev, regime = solve_merton_model(
        equity_value=-1.0e6,
        equity_volatility=0.50,
        debt_value=1.0e8,
    )
    assert dd == -999.0
    assert pi_def == 100.0
    assert regime == "CRITICAL_DEFAULT_IMMINENT"


def test_merton_regime_classification_tiers() -> None:
    """Verify KMV / Moody's distress tier classification thresholds."""
    assert classify_merton_distress_regime(4.5) == "PRIME_SOLVENT"
    assert classify_merton_distress_regime(3.5) == "INVESTMENT_GRADE"
    assert classify_merton_distress_regime(2.5) == "SPECULATIVE_MODERATE"
    assert classify_merton_distress_regime(1.5) == "HIGH_DISTRESS_RISK"
    assert classify_merton_distress_regime(0.5) == "CRITICAL_DEFAULT_IMMINENT"
    assert classify_merton_distress_regime(-1.0) == "CRITICAL_DEFAULT_IMMINENT"


def test_rolling_distance_to_default() -> None:
    """Verify rolling historical distance to default time series calculation."""
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
    s_prices = pd.Series(prices, index=dates)

    df_rolling = compute_rolling_distance_to_default(
        prices=s_prices,
        debt_value=5.0e9,
        shares_outstanding=1.0e8,  # Equity = $10B initially
        lookback_window=30,
    )

    assert len(df_rolling) == n - 30
    assert "distance_to_default" in df_rolling.columns
    assert "default_probability_pct" in df_rolling.columns
    assert "implied_asset_value" in df_rolling.columns
    assert "credit_spread_bps" in df_rolling.columns
    assert not df_rolling["distance_to_default"].isna().any()
    assert (df_rolling["distance_to_default"] > 0).all()


def test_merton_credit_receipt_tamper_integrity() -> None:
    """Verify HMAC-SHA256 signature verification and tamper rejection."""
    receipt = evaluate_merton_credit_profile(
        ticker="AAPL",
        debt_value=1.1e11,
        equity_value=2.5e12,
        equity_volatility=0.22,
    )

    assert receipt.verify_integrity() is True
    assert isinstance(receipt.signature, str)
    assert len(receipt.signature) == 64

    # Tampering with distance to default should immediately fail integrity check
    receipt.distance_to_default += 1.0
    assert receipt.verify_integrity() is False


def test_merton_credit_profile_with_price_series() -> None:
    """Verify complete evaluation workflow using synthetic price series."""
    np.random.seed(99)
    n = 65
    prices = 150.0 + np.cumsum(np.random.normal(0.0, 1.0, n))

    receipt = evaluate_merton_credit_profile(
        ticker="MSFT",
        debt_value=8.0e10,
        shares_outstanding=7.5e9,
        asset_prices=prices,
        lookback_window=50,
    )

    assert receipt.ticker == "MSFT"
    assert receipt.debt_value == 8.0e10
    assert receipt.equity_value > 0
    assert receipt.annualized_equity_volatility > 0
    assert receipt.implied_asset_value > receipt.equity_value
    assert receipt.verify_integrity() is True


def test_merton_credit_command_cli_plain(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with formatted stdout report."""
    args = argparse.Namespace(
        command="merton-credit",
        ticker="GOOGL",
        debt=3.0e10,
        shares=1.2e10,
        market_cap=2.0e12,
        volatility=0.24,
        risk_free=0.045,
        horizon=1.0,
        lookback=60,
        json=False,
    )

    merton_credit_command(args)
    captured = capsys.readouterr().out

    assert "Merton (1974) Structural Credit Risk & Distance-to-Default Engine" in captured
    assert "Security / Ticker:            GOOGL" in captured
    assert "Distance-to-Default (DD):" in captured
    assert "Default Probability (π):" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_merton_credit_command_cli_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with JSON output."""
    args = argparse.Namespace(
        command="merton-credit",
        ticker="NVDA",
        debt=1.5e10,
        shares=2.4e10,
        market_cap=3.0e12,
        volatility=0.35,
        risk_free=0.045,
        horizon=1.0,
        lookback=60,
        json=True,
    )

    merton_credit_command(args)
    captured = capsys.readouterr().out
    data = json.loads(captured)

    assert data["ticker"] == "NVDA"
    assert data["debt_value"] == 1.5e10
    assert data["equity_value"] == 3.0e12
    assert "distance_to_default" in data
    assert "implied_asset_volatility" in data
    assert "signature" in data


def test_merton_input_validation() -> None:
    """Verify input validation guards against invalid parameters."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        evaluate_merton_credit_profile(ticker="INVALID$$$", debt_value=1.0e8)

    with pytest.raises(ValueError, match="horizon_years must be strictly positive"):
        solve_merton_model(
            equity_value=1.0e8,
            equity_volatility=0.20,
            debt_value=5.0e7,
            horizon_years=-0.5,
        )
