"""Unit, boundary, security, and integration tests for Grossman & Stiglitz (1980)

Informational Equilibrium & Rent Engine.
"""

import argparse
import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from src.analytics.grossman_stiglitz_equilibrium import (
    GrossmanStiglitzReceipt,
    calculate_grossman_stiglitz_equilibrium,
    classify_informational_regime,
    evaluate_grossman_stiglitz_profile,
    persist_grossman_stiglitz_record,
    solve_equilibrium_informed_fraction,
    verify_grossman_audit_digest,
)
from src.cli import grossman_stiglitz_command
from src.security.validators import SecurityValidationError


def test_solve_equilibrium_informed_fraction_monotonicity_and_bounds() -> None:
    """Verify equilibrium informed fraction monotonicity and boundary constraints."""
    # Higher information cost c must monotonically reduce or keep equal the equilibrium informed fraction lambda*
    lam_low_cost, info_low, _, _ = solve_equilibrium_informed_fraction(
        fundamental_variance=0.0004,
        signal_noise_variance=0.0001,
        noise_trader_variance=0.0002,
        risk_aversion=2.5,
        information_cost=0.0005,  # 5 bps
    )

    lam_high_cost, info_high, _, _ = solve_equilibrium_informed_fraction(
        fundamental_variance=0.0004,
        signal_noise_variance=0.0001,
        noise_trader_variance=0.0002,
        risk_aversion=2.5,
        information_cost=0.0100,  # 100 bps
    )

    assert 0.01 <= lam_low_cost <= 0.99
    assert 0.01 <= lam_high_cost <= 0.99
    assert lam_low_cost >= lam_high_cost
    assert 0.0 <= info_low <= 1.0
    assert 0.0 <= info_high <= 1.0


def test_calculate_grossman_stiglitz_equilibrium_efficient_saturation() -> None:
    """Verify parameters yielding informationally efficient saturated regime."""
    receipt = calculate_grossman_stiglitz_equilibrium(
        ticker="AAPL",
        latest_price=175.0,
        fundamental_variance=0.0004,
        signal_noise_variance=0.00010,
        noise_trader_variance=0.000002,  # negligible noise trader shocks
        risk_aversion=1.5,
        information_cost=0.0002,  # 2 bps - very cheap info
    )

    assert receipt.ticker == "AAPL"
    assert receipt.price_informativeness >= 0.80
    assert receipt.informational_regime == "INFORMATIONALLY_EFFICIENT_SATURATED"
    assert receipt.is_efficient_saturated is True
    assert receipt.is_uninformed_herd is False
    assert verify_grossman_audit_digest(receipt) is True


def test_calculate_grossman_stiglitz_equilibrium_noisy_opportunity() -> None:
    """Verify parameters yielding noisy equilibrium opportunity regime."""
    receipt = calculate_grossman_stiglitz_equilibrium(
        ticker="MSFT",
        latest_price=400.0,
        fundamental_variance=0.0004,
        signal_noise_variance=0.00015,
        noise_trader_variance=0.00030,
        risk_aversion=2.5,
        information_cost=0.0025,  # 25 bps
        observed_lambda_override=0.35,
    )

    assert receipt.ticker == "MSFT"
    assert receipt.is_noisy_opportunity is True
    assert receipt.informational_regime == "NOISY_EQUILIBRIUM_OPPORTUNITY"
    assert receipt.gross_informational_value > 0.0
    assert verify_grossman_audit_digest(receipt) is True


def test_calculate_grossman_stiglitz_equilibrium_uninformed_herd() -> None:
    """Verify parameters yielding uninformed herd dominated regime."""
    receipt = calculate_grossman_stiglitz_equilibrium(
        ticker="HERD",
        latest_price=50.0,
        fundamental_variance=0.0002,
        signal_noise_variance=0.0005,
        noise_trader_variance=0.0002,
        risk_aversion=3.0,
        information_cost=0.050,  # 500 bps - prohibitively expensive info
        observed_lambda_override=0.05,
    )

    assert receipt.informational_regime == "UNINFORMED_HERD_DOMINATED"
    assert receipt.is_uninformed_herd is True
    assert receipt.is_efficient_saturated is False


def test_calculate_grossman_stiglitz_equilibrium_high_noise_dislocation() -> None:
    """Verify parameters yielding high noise dislocation regime."""
    receipt = calculate_grossman_stiglitz_equilibrium(
        ticker="NOISE",
        latest_price=25.0,
        fundamental_variance=0.0001,
        signal_noise_variance=0.0003,
        noise_trader_variance=0.0005,  # 5x noise to signal
        risk_aversion=2.0,
        information_cost=0.008,
        observed_lambda_override=0.10,
    )

    assert receipt.informational_regime == "HIGH_NOISE_INEFFICIENT_DISLOCATION"
    assert receipt.noise_to_signal_ratio >= 3.0


def test_tamper_signature_verification() -> None:
    """Verify cryptographic audit digest rejects tampered receipts."""
    receipt = calculate_grossman_stiglitz_equilibrium(
        ticker="NVDA",
        latest_price=120.0,
        fundamental_variance=0.0005,
        signal_noise_variance=0.0001,
        noise_trader_variance=0.0002,
    )

    assert verify_grossman_audit_digest(receipt) is True

    # Tampered receipt with altered net_informational_alpha
    tampered = GrossmanStiglitzReceipt(
        ticker=receipt.ticker,
        latest_price=receipt.latest_price,
        fundamental_variance=receipt.fundamental_variance,
        signal_noise_variance=receipt.signal_noise_variance,
        noise_trader_variance=receipt.noise_trader_variance,
        risk_aversion=receipt.risk_aversion,
        information_cost=receipt.information_cost,
        equilibrium_informed_fraction=receipt.equilibrium_informed_fraction,
        price_informativeness=receipt.price_informativeness,
        uninformed_posterior_variance=receipt.uninformed_posterior_variance,
        informed_posterior_variance=receipt.informed_posterior_variance,
        gross_informational_value=receipt.gross_informational_value,
        net_informational_alpha=receipt.net_informational_alpha + 0.05,  # Tampered!
        noise_to_signal_ratio=receipt.noise_to_signal_ratio,
        informational_regime=receipt.informational_regime,
        is_efficient_saturated=receipt.is_efficient_saturated,
        is_noisy_opportunity=receipt.is_noisy_opportunity,
        is_uninformed_herd=receipt.is_uninformed_herd,
        expected_alpha_boost_bps=receipt.expected_alpha_boost_bps,
        summary=receipt.summary,
        audit_digest=receipt.audit_digest,
        calculated_at=receipt.calculated_at,
    )

    assert verify_grossman_audit_digest(tampered) is False


def test_security_validation_ticker_and_bounds() -> None:
    """Verify security validation rejects malformed inputs and out-of-bound arguments."""
    with pytest.raises(SecurityValidationError):
        calculate_grossman_stiglitz_equilibrium(ticker="INVALID_TOOLONG_TICKER")

    with pytest.raises(SecurityValidationError):
        calculate_grossman_stiglitz_equilibrium(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_grossman_stiglitz_equilibrium(ticker="AAPL", fundamental_variance=-0.001)

    with pytest.raises(SecurityValidationError):
        calculate_grossman_stiglitz_equilibrium(ticker="AAPL", risk_aversion=0.0)

    with pytest.raises(SecurityValidationError):
        calculate_grossman_stiglitz_equilibrium(ticker="AAPL", information_cost=0.50)


def test_evaluate_grossman_stiglitz_profile_synthetic_observations() -> None:
    """Verify profile evaluation with synthetic price series."""
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.018, 100)
    prices = [float(base_price * np.exp(np.sum(returns[:i]))) for i in range(1, 101)]

    receipt = evaluate_grossman_stiglitz_profile(
        ticker="GOOGL",
        prices=prices,
        risk_aversion=2.5,
        information_cost=0.0020,
    )

    assert receipt.ticker == "GOOGL"
    assert receipt.latest_price > 0.0
    assert receipt.fundamental_variance > 0.0
    assert receipt.noise_trader_variance > 0.0
    assert 0.01 <= receipt.equilibrium_informed_fraction <= 0.99
    assert verify_grossman_audit_digest(receipt) is True


def test_evaluate_grossman_stiglitz_profile_fallback() -> None:
    """Verify graceful fallback when insufficient price history is provided."""
    receipt = evaluate_grossman_stiglitz_profile(
        ticker="TSLA",
        prices=[100.0, 101.0],  # only 2 points < 20
        risk_aversion=2.5,
        information_cost=0.0020,
    )

    assert receipt.ticker == "TSLA"
    assert receipt.latest_price == 150.0  # default fallback
    assert receipt.fundamental_variance > 0.0
    assert verify_grossman_audit_digest(receipt) is True


def test_cli_grossman_stiglitz_human_readable(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler prints human-readable telemetry table."""
    args = argparse.Namespace(
        ticker="AAPL",
        cost=0.0020,
        risk_aversion=2.50,
        fundamental_var=0.0004,
        noise_var=0.0002,
        informed_fraction=None,
        json=False,
    )

    grossman_stiglitz_command(args)
    captured = capsys.readouterr()

    assert "NextOmni Stock: Grossman & Stiglitz (1980) Informational Equilibrium Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Equilibrium Informed Fraction:" in captured.out
    assert "Price Informativeness Index:" in captured.out
    assert "Audit Digest (HMAC):" in captured.out


def test_cli_grossman_stiglitz_json_output(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs valid structured JSON."""
    args = argparse.Namespace(
        ticker="AMZN",
        cost=0.0015,
        risk_aversion=2.00,
        fundamental_var=None,
        noise_var=None,
        informed_fraction=None,
        json=True,
    )

    grossman_stiglitz_command(args)
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert data["ticker"] == "AMZN"
    assert "equilibrium_informed_fraction" in data
    assert "price_informativeness" in data
    assert "audit_digest" in data
    assert "informational_regime" in data


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify SQLite table creation and parameterized audit record persistence."""
    test_db = tmp_path / "test_nextomni.db"

    receipt = calculate_grossman_stiglitz_equilibrium(
        ticker="META",
        latest_price=500.0,
        fundamental_variance=0.0003,
        signal_noise_variance=0.0001,
        noise_trader_variance=0.0002,
    )

    persist_grossman_stiglitz_record(receipt, db_path=test_db)

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, equilibrium_informed_fraction, price_informativeness, audit_digest FROM grossman_stiglitz_records WHERE ticker = ?",
        ("META",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "META"
    assert row[1] == 500.0
    assert row[4] == receipt.audit_digest
