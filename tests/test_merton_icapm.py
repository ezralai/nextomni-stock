"""Unit, boundary, security, and integration tests for Robert C. Merton (1973)

Intertemporal Capital Asset Pricing Model (ICAPM) & State Variable Hedging Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.merton_icapm import (
    MertonICAPMReceipt,
    calculate_merton_icapm,
    classify_icapm_regime,
    evaluate_merton_icapm_profile,
    persist_merton_icapm_record,
    verify_merton_icapm_audit_digest,
)
from src.cli import merton_icapm_command
from src.security.validators import SecurityValidationError


def test_calculate_merton_icapm_steady_state() -> None:
    """Verify balanced market-following asset exhibits steady state regime."""
    np.random.seed(42)
    mkt = np.random.normal(0.0005, 0.01, 100)
    asset = 1.05 * mkt + np.random.normal(0.0, 0.002, 100)

    receipt = calculate_merton_icapm(
        asset_returns=asset,
        market_returns=mkt,
        latest_price=150.0,
        risk_free_rate=0.04,
        ticker="STEADY",
    )

    assert receipt.ticker == "STEADY"
    assert 0.95 <= receipt.market_beta <= 1.15
    assert receipt.r_squared > 0.80
    assert receipt.regime in ("STEADY_STATE_BALANCED", "AGGRESSIVE_OPPORTUNITY_EXPANSION")
    assert verify_merton_icapm_audit_digest(receipt) is True


def test_calculate_merton_icapm_volatility_hedge() -> None:
    """Verify asset with strong positive correlation to volatility shocks is classified as volatility hedge."""
    np.random.seed(42)
    mkt = np.random.normal(0.0, 0.01, 100)
    vol_shocks = np.random.normal(0.0, 1.0, 100)
    # Asset surges when volatility surges
    asset = 0.20 * mkt + 0.35 * vol_shocks + np.random.normal(0.0, 0.005, 100)

    receipt = calculate_merton_icapm(
        asset_returns=asset,
        market_returns=mkt,
        vol_innovations=vol_shocks,
        latest_price=80.0,
        ticker="VIXHDG",
    )

    assert receipt.ticker == "VIXHDG"
    assert receipt.volatility_beta > 0.20
    assert receipt.is_volatility_hedge is True
    assert receipt.is_volatility_hedge_defense is True
    assert receipt.regime == "VOLATILITY_HEDGE_DEFENSE"
    assert verify_merton_icapm_audit_digest(receipt) is True


def test_calculate_merton_icapm_duration_stress() -> None:
    """Verify asset with extreme rate exposure triggers duration stress regime."""
    np.random.seed(42)
    mkt = np.random.normal(0.0, 0.01, 100)
    rate_shocks = np.random.normal(0.0, 0.01, 100)
    asset = 0.50 * mkt - 0.85 * rate_shocks + np.random.normal(0.0, 0.003, 100)

    receipt = calculate_merton_icapm(
        asset_returns=asset,
        market_returns=mkt,
        rate_innovations=rate_shocks,
        latest_price=95.0,
        ticker="TLTBOND",
    )

    assert receipt.interest_rate_beta < -0.40
    assert receipt.is_duration_stress is True
    assert receipt.regime == "DURATION_INTEREST_RATE_STRESS"
    assert verify_merton_icapm_audit_digest(receipt) is True


def test_calculate_merton_icapm_aggressive_expansion() -> None:
    """Verify high market beta asset with negative vol sensitivity is classified as aggressive expansion."""
    np.random.seed(42)
    mkt = np.random.normal(0.0, 0.01, 100)
    vol_shocks = np.random.normal(0.0, 1.0, 100)
    asset = 1.45 * mkt - 0.25 * vol_shocks + np.random.normal(0.0, 0.003, 100)

    receipt = calculate_merton_icapm(
        asset_returns=asset,
        market_returns=mkt,
        vol_innovations=vol_shocks,
        latest_price=220.0,
        ticker="AGGEXP",
    )

    assert receipt.market_beta > 1.25
    assert receipt.volatility_beta < -0.10
    assert receipt.is_aggressive_expansion is True
    assert receipt.regime == "AGGRESSIVE_OPPORTUNITY_EXPANSION"
    assert verify_merton_icapm_audit_digest(receipt) is True


def test_hedging_demand_reduces_hurdle_rate() -> None:
    """Verify state-variable hedging demand (beta_vol > 0, lambda_vol < 0) discounts required hurdle rate."""
    np.random.seed(42)
    mkt = np.random.normal(0.0, 0.01, 100)
    vol_shocks = np.random.normal(0.0, 1.0, 100)

    # Asset A: purely market-correlated
    rec_plain = calculate_merton_icapm(
        asset_returns=1.0 * mkt,
        market_returns=mkt,
        vol_innovations=vol_shocks,
        lambda_market=0.06,
        lambda_vol=-0.04,
        risk_free_rate=0.04,
    )

    # Asset B: hedges volatility (+0.30 * vol_shocks)
    rec_hedge = calculate_merton_icapm(
        asset_returns=1.0 * mkt + 0.30 * vol_shocks,
        market_returns=mkt,
        vol_innovations=vol_shocks,
        lambda_market=0.06,
        lambda_vol=-0.04,
        risk_free_rate=0.04,
    )

    # Volatility hedging premium is negative (beta_vol * lambda_vol = 0.30 * -0.04 = -1.2%)
    assert rec_hedge.vol_hedging_contribution_pct < 0.0
    # Hurdle rate for hedging asset is lower than standard CAPM because investors accept lower returns for hedge
    assert rec_hedge.icapm_hurdle_rate_pct < rec_plain.icapm_hurdle_rate_pct


def test_multi_variate_ols_mathematical_properties() -> None:
    """Verify OLS multi-beta solver reconstructs known linear coefficients."""
    np.random.seed(123)
    n = 200
    mkt = np.random.normal(0.0, 0.01, n)
    rate = np.random.normal(0.0, 0.005, n)
    vol = np.random.normal(0.0, 0.5, n)

    # True parameters: beta_M = 1.20, beta_r = -0.50, beta_vol = 0.15
    y = 1.20 * mkt - 0.50 * rate + 0.15 * vol

    receipt = calculate_merton_icapm(
        asset_returns=y,
        market_returns=mkt,
        rate_innovations=rate,
        vol_innovations=vol,
    )

    assert abs(receipt.market_beta - 1.20) < 0.05
    assert abs(receipt.interest_rate_beta - (-0.50)) < 0.05
    assert abs(receipt.volatility_beta - 0.15) < 0.05
    assert receipt.r_squared > 0.98


def test_audit_digest_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature detects tampering in ICAPM receipt."""
    np.random.seed(42)
    mkt = np.random.normal(0.0, 0.01, 50)
    asset = mkt * 1.1

    receipt = calculate_merton_icapm(asset_returns=asset, market_returns=mkt, ticker="AUTH")
    assert verify_merton_icapm_audit_digest(receipt) is True

    # Tamper with hurdle rate
    tampered_dict = receipt.to_dict()
    tampered_dict["icapm_hurdle_rate_pct"] = 99.99
    tampered_receipt = MertonICAPMReceipt(**tampered_dict)
    assert verify_merton_icapm_audit_digest(tampered_receipt) is False

    # Tamper with beta
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["market_beta"] = 5.0
    tampered_receipt2 = MertonICAPMReceipt(**tampered_dict2)
    assert verify_merton_icapm_audit_digest(tampered_receipt2) is False


def test_invalid_inputs_raise_errors() -> None:
    """Verify invalid numerical inputs and short series raise appropriate errors."""
    with pytest.raises(ValueError):
        calculate_merton_icapm(asset_returns=[0.01] * 20, market_returns=[0.01] * 20, latest_price=-50.0)

    with pytest.raises(ValueError):
        calculate_merton_icapm(asset_returns=[0.01] * 5, market_returns=[0.01] * 5)

    with pytest.raises(ValueError):
        calculate_merton_icapm(
            asset_returns=[0.01] * 20,
            market_returns=[0.01] * 20,
            risk_free_rate=float("nan"),
        )


def test_security_validation_ticker() -> None:
    """Verify malicious ticker symbols are blocked by security validators."""
    with pytest.raises(SecurityValidationError):
        evaluate_merton_icapm_profile(ticker="INVAL$$ID")


def test_evaluate_merton_icapm_profile_with_prices() -> None:
    """Verify evaluate_merton_icapm_profile functions with explicit price arrays."""
    prices = [100.0 * (1.001 ** i) for i in range(50)]
    mkt_prices = [100.0 * (1.0008 ** i) for i in range(50)]

    receipt = evaluate_merton_icapm_profile(
        ticker="AAPL",
        prices=prices,
        market_prices=mkt_prices,
        risk_free_rate=0.045,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == round(prices[-1], 4)
    assert verify_merton_icapm_audit_digest(receipt) is True


def test_persist_merton_icapm_record(tmp_path: Path) -> None:
    """Verify SQLite persistence of Merton ICAPM evaluation records."""
    db_file = tmp_path / "test_merton_icapm.db"
    np.random.seed(42)
    mkt = np.random.normal(0.0, 0.01, 50)
    asset = mkt * 0.9

    receipt = calculate_merton_icapm(asset_returns=asset, market_returns=mkt, ticker="PERSIST")

    success = persist_merton_icapm_record(receipt, db_path=db_file)
    assert success is True

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, market_beta, icapm_hurdle_rate_pct, audit_digest FROM merton_icapm_records")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "PERSIST"
    assert abs(row[1] - receipt.market_beta) < 1e-4
    assert abs(row[2] - receipt.icapm_hurdle_rate_pct) < 1e-4
    assert row[3] == receipt.audit_digest


def test_cli_merton_icapm_command_and_json(capsys: pytest.CaptureFixture) -> None:
    """Verify merton-icapm CLI handler runs cleanly in human-readable and JSON modes."""
    # Test human readable
    args = argparse.Namespace(
        ticker="AAPL",
        market_premium=0.055,
        rate_premium=-0.02,
        vol_premium=-0.035,
        rf=0.0425,
        json=False,
    )
    merton_icapm_command(args)
    captured = capsys.readouterr().out
    assert "NextOmni Stock: Merton (1973) Intertemporal CAPM" in captured
    assert "Market Beta (beta_M)" in captured
    assert "ICAPM Required Hurdle Rate" in captured

    # Test JSON
    args.json = True
    merton_icapm_command(args)
    captured_json = capsys.readouterr().out
    data = json.loads(captured_json)
    assert data["ticker"] == "AAPL"
    assert "market_beta" in data
    assert "total_state_hedging_premium_pct" in data
    assert "audit_digest" in data
