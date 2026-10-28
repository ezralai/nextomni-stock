"""Unit and integration tests for Risk-Managed Momentum & Crash-Risk Dampened Trend Engine."""

import argparse
import json
import numpy as np
import pandas as pd
import pytest

from src.cli import momentum_crash_report_command
from src.analytics.risk_managed_momentum import (
    RiskManagedMomentumReceipt,
    calculate_risk_managed_momentum,
    estimate_momentum_crash_probability,
    evaluate_momentum_risk_profile,
)


def test_calculate_risk_managed_momentum_scaling_bounds() -> None:
    """Verify Barroso & Santa-Clara inverse-volatility scaling obeys leverage caps and bounds."""
    np.random.seed(42)
    n = 100
    daily_returns = np.random.normal(0.0005, 0.015, size=n)

    raw_rets, rmm_rets, weights = calculate_risk_managed_momentum(
        daily_returns,
        lookback_mom=20,
        lookback_vol=20,
        target_vol=0.12,
        max_leverage=2.0,
    )

    assert len(raw_rets) == len(rmm_rets) == len(weights)
    assert len(raw_rets) > 0

    # Weights must obey strict upper bound [0, max_leverage]
    assert np.all(weights > 0.0)
    assert np.all(weights <= 2.0 + 1e-6)

    # RMM return identity: rmm = w * raw
    np.testing.assert_allclose(rmm_rets, weights * raw_rets, atol=1e-10)


def test_momentum_crash_mitigation_and_skewness() -> None:
    """Verify that inverse-volatility scaling dampens crash spikes and normalizes negative skewness."""
    np.random.seed(101)
    n = 150
    # Simulate calm trending market, then severe crash/reversal
    calm = np.random.normal(0.001, 0.008, size=100)
    # Volatility surge and violent reversal
    shock = np.random.normal(-0.005, 0.045, size=50)
    shock[25] = 0.12  # Rebound shock against short momentum
    returns = np.concatenate([calm, shock])

    raw_rets, rmm_rets, weights = calculate_risk_managed_momentum(
        returns,
        lookback_mom=15,
        lookback_vol=15,
        target_vol=0.10,
        max_leverage=2.0,
    )

    # Realized variance in shock period should cause weights to drop significantly below 1.0x
    shock_weights = weights[-35:]
    assert np.mean(shock_weights) < 1.0

    # Maximum single-session loss in RMM should be strictly smaller than raw momentum
    max_loss_raw = float(np.min(raw_rets))
    max_loss_rmm = float(np.min(rmm_rets))
    assert max_loss_rmm > max_loss_raw  # Less negative loss


def test_estimate_momentum_crash_probability_regimes() -> None:
    """Verify Daniel & Moskowitz crash hazard probability reflects bear market drawdowns and vol surges."""
    np.random.seed(77)
    n = 150

    # 1. Bear market state with volatility acceleration (persistent negative trend)
    bear_returns = np.random.normal(-0.008, 0.015, size=n)
    # Short-term volatility spike at the end
    bear_returns[-25:] = np.random.normal(-0.01, 0.04, size=25)

    p_crash_bear, regime_bear = estimate_momentum_crash_probability(bear_returns)
    assert p_crash_bear > 60.0
    assert regime_bear in ["CRASH_RISK_ELEVATED", "CRASH_RISK_MODERATE"]

    # 2. Calm bull market state (positive trend, low volatility)
    np.random.seed(42)
    bull_returns = np.random.normal(0.002, 0.005, size=n)
    p_crash_bull, regime_bull = estimate_momentum_crash_probability(bull_returns)
    assert p_crash_bull < 40.0
    assert regime_bull == "CRASH_RISK_BENIGN"


def test_risk_managed_momentum_receipt_signature_and_tamper() -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    receipt = RiskManagedMomentumReceipt(
        ticker="TSLA",
        sample_sessions=120,
        target_volatility_pct=12.0,
        max_leverage=2.0,
        current_crash_hazard_pct=35.5,
        crash_risk_regime="CRASH_RISK_BENIGN",
        current_leverage_weight=1.15,
        raw_momentum_ann_return_pct=14.5,
        rmm_ann_return_pct=18.2,
        raw_momentum_volatility_pct=24.0,
        rmm_volatility_pct=13.5,
        raw_momentum_sharpe=0.44,
        rmm_sharpe=1.05,
        raw_momentum_max_drawdown_pct=28.5,
        rmm_max_drawdown_pct=14.2,
        raw_momentum_skewness=-1.85,
        rmm_skewness=-0.15,
        summary="Test momentum receipt",
    )

    assert receipt.signature != ""
    assert receipt.verify_integrity() is True

    # Tamper with Sharpe ratio
    receipt.rmm_sharpe = 9.99
    assert receipt.verify_integrity() is False


def test_evaluate_momentum_risk_profile_integration() -> None:
    """Verify end-to-end momentum profile evaluation on observation DataFrame."""
    np.random.seed(66)
    dates = pd.date_range("2025-01-01", periods=80, freq="B").strftime("%Y-%m-%d")
    prices = np.cumprod(1.0 + np.random.normal(0.0005, 0.015, size=80)) * 100.0

    df_obs = pd.DataFrame({
        "session_date": dates,
        "adj_close": prices,
        "close": prices,
    })

    receipt = evaluate_momentum_risk_profile(df_obs, ticker="JPM", target_vol=0.12, max_leverage=2.0)

    assert receipt.ticker == "JPM"
    assert receipt.sample_sessions == 80
    assert receipt.target_volatility_pct == 12.0
    assert receipt.max_leverage == 2.0
    assert receipt.verify_integrity() is True


def test_cli_momentum_crash_report_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verify CLI momentum-crash-report execution for stdout and JSON formats."""
    fake_receipt = RiskManagedMomentumReceipt(
        ticker="AMD",
        sample_sessions=90,
        target_volatility_pct=12.0,
        max_leverage=2.0,
        current_crash_hazard_pct=68.5,
        crash_risk_regime="CRASH_RISK_ELEVATED",
        current_leverage_weight=0.65,
        raw_momentum_ann_return_pct=10.0,
        rmm_ann_return_pct=16.5,
        raw_momentum_volatility_pct=26.0,
        rmm_volatility_pct=12.8,
        raw_momentum_sharpe=0.23,
        rmm_sharpe=0.98,
        raw_momentum_max_drawdown_pct=32.0,
        rmm_max_drawdown_pct=16.5,
        raw_momentum_skewness=-2.15,
        rmm_skewness=0.10,
        summary="AMD Momentum Profile: Crash Hazard=68.5% (CRASH_RISK_ELEVATED), Dynamic Leverage=0.65x.",
    )

    monkeypatch.setattr(
        "src.data.db.get_observations",
        lambda ticker, db_path=None: pd.DataFrame({"session_date": ["2025-01-01"], "adj_close": [100.0]}),
    )
    monkeypatch.setattr(
        "src.analytics.risk_managed_momentum.evaluate_momentum_risk_profile",
        lambda df, ticker, target_vol, max_leverage: fake_receipt,
    )

    # 1. Tabular stdout output
    args_std = argparse.Namespace(ticker="AMD", target_vol=0.12, max_leverage=2.0, json=False)
    momentum_crash_report_command(args_std)
    out_std = capsys.readouterr().out
    assert "Risk-Managed Momentum & Crash-Risk Dampened Trend Engine" in out_std
    assert "AMD" in out_std
    assert "CRASH_RISK_ELEVATED" in out_std
    assert "0.65x" in out_std

    # 2. JSON structured output
    args_json = argparse.Namespace(ticker="AMD", target_vol=0.12, max_leverage=2.0, json=True)
    momentum_crash_report_command(args_json)
    out_json = capsys.readouterr().out
    data = json.loads(out_json)
    assert data["ticker"] == "AMD"
    assert data["crash_risk_regime"] == "CRASH_RISK_ELEVATED"
    assert data["rmm_sharpe"] == 0.98
    assert "signature" in data
