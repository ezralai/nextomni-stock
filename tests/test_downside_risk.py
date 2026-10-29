"""Unit test suite for Ang, Chen & Xing (2006) Downside Beta and Realized Semivariance Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.downside_risk import (
    DownsideRiskReceipt,
    compute_downside_and_upside_beta,
    compute_realized_semivariance,
    evaluate_downside_risk_profile,
)
from src.cli import downside_beta_command


def test_realized_semivariance_additivity() -> None:
    """Verify that upside and downside semivariances strictly sum to total realized variance."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)

    rv, rs_plus, rs_minus, down_ratio = compute_realized_semivariance(returns)

    assert rv > 0.0
    assert rs_plus > 0.0
    assert rs_minus > 0.0
    assert rv == pytest.approx(rs_plus + rs_minus, rel=1e-6)
    assert 0.0 <= down_ratio <= 100.0
    assert down_ratio == pytest.approx((rs_minus / rv) * 100.0, rel=1e-6)

    # Empty returns edge case
    empty_rv, empty_up, empty_down, empty_ratio = compute_realized_semivariance([])
    assert empty_rv == 0.0
    assert empty_up == 0.0
    assert empty_down == 0.0
    assert empty_ratio == 50.0


def test_downside_and_upside_beta_crash_amplifier() -> None:
    """Verify downside beta is significantly higher than upside beta for a crash amplifier."""
    np.random.seed(101)
    n = 200
    r_m = np.random.normal(0.0, 0.015, n)
    mean_m = float(np.mean(r_m))

    # Crash amplifier: beta = 2.2 on downside, beta = 0.8 on upside
    noise = np.random.normal(0.0, 0.002, n)
    r_i = np.where(r_m < mean_m, 2.2 * r_m + noise, 0.8 * r_m + noise)

    capm_b, down_b, up_b, b_spread = compute_downside_and_upside_beta(r_i, r_m)

    assert down_b > up_b
    assert b_spread > 0.50
    assert down_b > 1.8
    assert up_b < 1.2
    assert up_b < capm_b < down_b


def test_downside_and_upside_beta_crash_hedge() -> None:
    """Verify downside beta is lower than upside beta for a defensive crash hedge."""
    np.random.seed(202)
    n = 200
    r_m = np.random.normal(0.0, 0.015, n)
    mean_m = float(np.mean(r_m))

    # Defensive hedge: beta = 0.2 on downside, beta = 1.2 on upside
    noise = np.random.normal(0.0, 0.002, n)
    r_i = np.where(r_m < mean_m, 0.2 * r_m + noise, 1.2 * r_m + noise)

    capm_b, down_b, up_b, b_spread = compute_downside_and_upside_beta(r_i, r_m)

    assert down_b < up_b
    assert b_spread < -0.50
    assert down_b < 0.6
    assert up_b > 0.9


def test_downside_and_upside_beta_validation_errors() -> None:
    """Verify validation errors on mismatched lengths and insufficient observations."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_downside_and_upside_beta([0.01] * 15, [0.01] * 20)

    with pytest.raises(ValueError, match="Insufficient observations"):
        compute_downside_and_upside_beta([0.01] * 5, [0.01] * 5)


def test_evaluate_downside_risk_profile_synthetic() -> None:
    """Test full downside risk profiling workflow with synthetic price paths."""
    np.random.seed(303)
    n = 80
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    ret_asset = np.random.normal(0.0005, 0.02, n)
    ret_bench = np.random.normal(0.0003, 0.01, n)

    prices_asset = pd.Series(100.0 * np.cumprod(1.0 + ret_asset), index=dates)
    prices_bench = pd.Series(450.0 * np.cumprod(1.0 + ret_bench), index=dates)

    receipt = evaluate_downside_risk_profile(
        ticker="AAPL",
        asset_prices=prices_asset,
        benchmark_prices=prices_bench,
        benchmark_ticker="SPY",
        window=40,
    )

    assert isinstance(receipt, DownsideRiskReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.benchmark_ticker == "SPY"
    assert receipt.lookback_window == 40
    assert receipt.crash_sensitivity_tier in {"CRASH_AMPLIFIER", "BALANCED_EXPOSURE", "CRASH_HEDGE"}
    assert receipt.verify_integrity() is True

    # Realized variance additivity and vol relationship
    # sigma_total^2 == sigma_up^2 + sigma_down^2
    var_total = (receipt.annualized_total_vol_pct / 100.0) ** 2
    var_sum = ((receipt.annualized_upside_vol_pct / 100.0) ** 2) + ((receipt.annualized_downside_vol_pct / 100.0) ** 2)
    assert var_total == pytest.approx(var_sum, rel=1e-5)


def test_receipt_integrity_and_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    receipt = DownsideRiskReceipt(
        ticker="MSFT",
        benchmark_ticker="SPY",
        sample_sessions=100,
        lookback_window=60,
        capm_beta=1.10,
        downside_beta=1.45,
        upside_beta=0.95,
        beta_asymmetry_spread=0.50,
        realized_variance=0.04,
        upside_semivariance=0.018,
        downside_semivariance=0.022,
        downside_variance_ratio_pct=55.0,
        annualized_total_vol_pct=25.0,
        annualized_upside_vol_pct=16.7,
        annualized_downside_vol_pct=18.5,
        crash_sensitivity_tier="CRASH_AMPLIFIER",
        summary="Test summary",
    )

    assert receipt.verify_integrity() is True

    # Tamper with downside beta
    receipt.downside_beta = 0.50
    assert receipt.verify_integrity() is False


def test_security_input_validation() -> None:
    """Verify ticker allowlist checks and window bounds."""
    prices = pd.Series([100.0 + i for i in range(50)])

    with pytest.raises(ValueError, match="Invalid ticker"):
        evaluate_downside_risk_profile("AAPL; DROP TABLE", prices)

    with pytest.raises(ValueError, match="Lookback window must be at least 20"):
        evaluate_downside_risk_profile("AAPL", prices, window=10)


def test_cli_downside_beta_command(capsys: pytest.CaptureFixture) -> None:
    """Verify the CLI downside-beta command in text and JSON output modes."""
    n = 80
    dates = [f"2026-01-{i+1:02d}" for i in range(n)]
    prices = [150.0 + (i * 0.5) for i in range(n)]
    df_obs = pd.DataFrame({"session_date": dates, "adj_close": prices, "close": prices})

    with patch("src.data.db.get_observations", return_value=df_obs):
        # 1. Text mode
        args = argparse.Namespace(ticker="NVDA", benchmark="SPY", window=40, json=False)
        downside_beta_command(args)
        out = capsys.readouterr().out
        assert "Ang, Chen & Xing (2006) Downside Beta & Semivariance Engine" in out
        assert "Security / Ticker:            NVDA" in out
        assert "Downside Beta (Beta^-):" in out

        # 2. JSON mode
        args_json = argparse.Namespace(ticker="NVDA", benchmark="SPY", window=40, json=True)
        downside_beta_command(args_json)
        out_json = capsys.readouterr().out
        parsed = json.loads(out_json)
        assert parsed["ticker"] == "NVDA"
        assert "downside_beta" in parsed
        assert "beta_asymmetry_spread" in parsed
        assert "signature" in parsed
