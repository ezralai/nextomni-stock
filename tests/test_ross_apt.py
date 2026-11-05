"""Unit, boundary, security, and integration tests for Stephen A. Ross (1976)

Arbitrage Pricing Theory (APT) & Macroeconomic Multi-Factor Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.ross_apt import (
    RossAPTReceipt,
    calculate_ross_apt,
    classify_apt_regime,
    evaluate_ross_apt_profile,
    persist_ross_apt_record,
    verify_ross_apt_audit_digest,
)
from src.cli import ross_apt_command
from src.security.validators import SecurityValidationError


def test_calculate_ross_apt_arbitrage_buy_opportunity() -> None:
    """Verify strong positive alpha above factor requirements triggers arbitrage buy regime."""
    np.random.seed(42)
    n = 100
    # High positive mean return producing ~+35% annual return
    asset = np.random.normal(0.0014, 0.01, n)
    f_growth = np.random.normal(0.0, 0.005, n)

    receipt = calculate_ross_apt(
        asset_returns=asset,
        growth_shocks=f_growth,
        latest_price=120.0,
        risk_free_rate=0.04,
        ticker="ALPHABUY",
    )

    assert receipt.ticker == "ALPHABUY"
    assert receipt.apt_arbitrage_spread_pct > 3.0
    assert receipt.is_arbitrage_buy is True
    assert receipt.regime == "ARBITRAGE_BUY_OPPORTUNITY"
    assert verify_ross_apt_audit_digest(receipt) is True


def test_calculate_ross_apt_arbitrage_sell_overpricing() -> None:
    """Verify negative alpha below factor requirements triggers arbitrage sell overpricing regime."""
    np.random.seed(42)
    n = 100
    # Negative mean return producing ~-20% annual return
    asset = np.random.normal(-0.0008, 0.01, n)

    receipt = calculate_ross_apt(
        asset_returns=asset,
        latest_price=65.0,
        risk_free_rate=0.045,
        ticker="SELLOVER",
    )

    assert receipt.apt_arbitrage_spread_pct < -3.0
    assert receipt.is_arbitrage_sell is True
    assert receipt.regime == "ARBITRAGE_SELL_OVERPRICING"
    assert verify_ross_apt_audit_digest(receipt) is True


def test_calculate_ross_apt_macro_factor_dominated() -> None:
    """Verify tight fit to macro factors with fair pricing triggers macro dominated regime."""
    np.random.seed(42)
    n = 100
    f_growth = np.random.normal(0.0, 0.01, n)
    f_growth = f_growth - np.mean(f_growth)
    f_inf = np.random.normal(0.0, 0.005, n)
    f_inf = f_inf - np.mean(f_inf)

    # Expected annual return: Rf (4.5%) + b_growth * lambda_growth (1.10 * 4.5% = 4.95%) + b_inf * lambda_inf (-0.40 * -2.5% = 1.0%) = 10.45%
    expected_ann = 0.045 + 1.10 * 0.045 + (-0.40 * -0.025)
    mean_daily = expected_ann / 252.0
    noise = np.random.normal(0.0, 0.001, n)
    noise = noise - np.mean(noise)
    asset = mean_daily + 1.10 * f_growth - 0.40 * f_inf + noise

    receipt = calculate_ross_apt(
        asset_returns=asset,
        growth_shocks=f_growth,
        inflation_shocks=f_inf,
        latest_price=90.0,
        risk_free_rate=0.045,
        lambda_growth=0.045,
        lambda_inflation=-0.025,
        ticker="MACRODOM",
    )

    assert receipt.systematic_variance_ratio_pct > 70.0
    assert abs(receipt.apt_arbitrage_spread_pct) <= 3.0
    assert receipt.is_macro_dominated is True
    assert receipt.regime == "MACRO_FACTOR_DOMINATED"
    assert verify_ross_apt_audit_digest(receipt) is True


def test_calculate_ross_apt_idiosyncratic_idle_equilibrium() -> None:
    """Verify low systematic variance and fair pricing triggers idiosyncratic idle equilibrium."""
    np.random.seed(42)
    n = 150
    f_growth = np.random.normal(0.0, 0.01, n)
    f_growth = f_growth - np.mean(f_growth)
    f_inf = np.random.normal(0.0, 0.01, n)
    f_inf = f_inf - np.mean(f_inf)
    f_def = np.random.normal(0.0, 0.01, n)
    f_def = f_def - np.mean(f_def)
    f_term = np.random.normal(0.0, 0.01, n)
    f_term = f_term - np.mean(f_term)

    # Pure idiosyncratic noise centered exactly at risk-free rate (4.25% annual)
    mean_daily = 0.0425 / 252.0
    noise = np.random.normal(0.0, 0.02, n)
    noise = noise - np.mean(noise)
    asset = mean_daily + noise

    receipt = calculate_ross_apt(
        asset_returns=asset,
        growth_shocks=f_growth,
        inflation_shocks=f_inf,
        default_shocks=f_def,
        term_shocks=f_term,
        latest_price=105.0,
        risk_free_rate=0.0425,
        ticker="IDIOIDLE",
    )

    assert abs(receipt.apt_arbitrage_spread_pct) <= 3.0
    assert receipt.systematic_variance_ratio_pct <= 70.0
    assert receipt.is_idiosyncratic_idle is True
    assert receipt.regime == "IDIOSYNCRATIC_IDLE_EQUILIBRIUM"
    assert verify_ross_apt_audit_digest(receipt) is True


def test_multi_factor_ols_exact_recovery() -> None:
    """Verify OLS multi-factor solver reconstructs known macroeconomic factor loadings."""
    np.random.seed(123)
    n = 250
    f_growth = np.random.normal(0.0, 0.01, n)
    f_inf = np.random.normal(0.0, 0.008, n)
    f_def = np.random.normal(0.0, 0.006, n)
    f_term = np.random.normal(0.0, 0.005, n)

    # True parameters: b_growth = 0.85, b_inf = -0.40, b_def = 0.65, b_term = 0.30
    y = 0.85 * f_growth - 0.40 * f_inf + 0.65 * f_def + 0.30 * f_term

    receipt = calculate_ross_apt(
        asset_returns=y,
        growth_shocks=f_growth,
        inflation_shocks=f_inf,
        default_shocks=f_def,
        term_shocks=f_term,
    )

    assert abs(receipt.growth_factor_beta - 0.85) < 0.05
    assert abs(receipt.inflation_factor_beta - (-0.40)) < 0.05
    assert abs(receipt.default_factor_beta - 0.65) < 0.05
    assert abs(receipt.term_factor_beta - 0.30) < 0.05
    assert receipt.systematic_variance_ratio_pct > 99.0


def test_variance_decomposition_conservation() -> None:
    """Verify systematic variance ratio and idiosyncratic variance ratio sum to 100%."""
    np.random.seed(42)
    asset = np.random.normal(0.0005, 0.01, 80)

    receipt = calculate_ross_apt(asset_returns=asset, ticker="CONSERVE")
    assert abs(receipt.systematic_variance_ratio_pct + receipt.idiosyncratic_variance_ratio_pct - 100.0) < 1e-4


def test_audit_digest_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature detects tampering in Ross APT receipt."""
    np.random.seed(42)
    asset = np.random.normal(0.0005, 0.01, 50)

    receipt = calculate_ross_apt(asset_returns=asset, ticker="TAMPER")
    assert verify_ross_apt_audit_digest(receipt) is True

    # Tamper with expected return
    tampered_dict = receipt.to_dict()
    tampered_dict["apt_expected_return_pct"] = 99.99
    tampered_receipt = RossAPTReceipt(**tampered_dict)
    assert verify_ross_apt_audit_digest(tampered_receipt) is False

    # Tamper with arbitrage spread
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["apt_arbitrage_spread_pct"] = 55.55
    tampered_receipt2 = RossAPTReceipt(**tampered_dict2)
    assert verify_ross_apt_audit_digest(tampered_receipt2) is False


def test_invalid_inputs_raise_errors() -> None:
    """Verify invalid numerical inputs and short series raise appropriate errors."""
    with pytest.raises(ValueError):
        calculate_ross_apt(asset_returns=[0.01] * 20, latest_price=-10.0)

    with pytest.raises(ValueError):
        calculate_ross_apt(asset_returns=[0.01] * 5)

    with pytest.raises(ValueError):
        calculate_ross_apt(asset_returns=[0.01] * 20, risk_free_rate=float("nan"))


def test_security_validation_ticker() -> None:
    """Verify malicious ticker symbols are rejected by security validators."""
    with pytest.raises(SecurityValidationError):
        evaluate_ross_apt_profile(ticker="INVAL$$ID")


def test_evaluate_ross_apt_profile_with_prices() -> None:
    """Verify evaluate_ross_apt_profile functions with explicit price arrays."""
    prices = [100.0 * (1.001 ** i) for i in range(50)]

    receipt = evaluate_ross_apt_profile(
        ticker="AAPL",
        prices=prices,
        risk_free_rate=0.045,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == round(prices[-1], 4)
    assert verify_ross_apt_audit_digest(receipt) is True


def test_persist_ross_apt_record(tmp_path: Path) -> None:
    """Verify SQLite persistence of Ross APT evaluation records."""
    db_file = tmp_path / "test_ross_apt.db"
    np.random.seed(42)
    asset = np.random.normal(0.0005, 0.01, 50)

    receipt = calculate_ross_apt(asset_returns=asset, ticker="PERSISTDB")

    success = persist_ross_apt_record(receipt, db_path=db_file)
    assert success is True

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, growth_factor_beta, apt_expected_return_pct, audit_digest FROM ross_apt_records")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "PERSISTDB"
    assert abs(row[1] - receipt.growth_factor_beta) < 1e-4
    assert abs(row[2] - receipt.apt_expected_return_pct) < 1e-4
    assert row[3] == receipt.audit_digest


def test_cli_ross_apt_command_and_json(capsys: pytest.CaptureFixture) -> None:
    """Verify ross-apt CLI handler runs cleanly in human-readable and JSON modes."""
    # Test human readable
    args = argparse.Namespace(
        ticker="AAPL",
        growth_premium=0.045,
        inflation_premium=-0.025,
        default_premium=-0.03,
        term_premium=0.02,
        rf=0.0425,
        json=False,
    )
    ross_apt_command(args)
    captured = capsys.readouterr().out
    assert "NextOmni Stock: Stephen A. Ross (1976) Arbitrage Pricing Theory (APT)" in captured
    assert "Growth Factor Loading (b_growth)" in captured
    assert "APT Expected No-Arbitrage Return" in captured

    # Test JSON
    args.json = True
    ross_apt_command(args)
    captured_json = capsys.readouterr().out
    data = json.loads(captured_json)
    assert data["ticker"] == "AAPL"
    assert "growth_factor_beta" in data
    assert "apt_expected_return_pct" in data
    assert "audit_digest" in data
