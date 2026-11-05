"""Unit, boundary, security, and integration tests for Campbell & Vuolteenaho (2004)

Two-Beta ("Bad Beta, Good Beta") Intertemporal Asset Pricing Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.bad_beta_good_beta import (
    TwoBetaReceipt,
    calculate_two_beta_decomposition,
    classify_two_beta_regime,
    evaluate_two_beta_profile,
    persist_two_beta_record,
    verify_two_beta_audit_digest,
)
from src.cli import bad_beta_command
from src.security.validators import SecurityValidationError


def test_calculate_two_beta_decomposition_cash_flow_vulnerable() -> None:
    """Verify high cash-flow beta triggers cash-flow vulnerable bad beta regime."""
    receipt = calculate_two_beta_decomposition(
        ticker="VALUE",
        latest_price=80.0,
        cash_flow_bad_beta=0.85,
        discount_rate_good_beta=0.45,
        risk_aversion_gamma=6.0,
    )

    assert receipt.ticker == "VALUE"
    assert receipt.total_market_beta == 1.30
    assert receipt.cash_flow_bad_beta == 0.85
    assert receipt.bad_beta_ratio > 0.50
    assert receipt.regime == "CASH_FLOW_VULNERABLE_BAD_BETA"
    assert receipt.is_cash_flow_vulnerable is True
    assert receipt.bad_beta_hurdle_surcharge_bps > 0.0
    assert receipt.two_beta_icapm_return_pct > receipt.standard_capm_return_pct
    assert verify_two_beta_audit_digest(receipt) is True


def test_calculate_two_beta_decomposition_discount_rate_growth() -> None:
    """Verify high discount-rate beta triggers growth good beta regime."""
    receipt = calculate_two_beta_decomposition(
        ticker="GROWTH",
        latest_price=150.0,
        cash_flow_bad_beta=0.25,
        discount_rate_good_beta=1.10,
        risk_aversion_gamma=6.0,
    )

    assert receipt.regime == "DISCOUNT_RATE_GROWTH_GOOD_BETA"
    assert receipt.is_discount_rate_growth is True
    assert receipt.bad_beta_ratio < 0.30
    assert receipt.bad_beta_hurdle_surcharge_bps < 0.0
    assert receipt.two_beta_icapm_return_pct < receipt.standard_capm_return_pct
    assert verify_two_beta_audit_digest(receipt) is True


def test_calculate_two_beta_decomposition_defensive_insulated() -> None:
    """Verify low overall market beta triggers defensive insulated regime."""
    receipt = calculate_two_beta_decomposition(
        ticker="DEF",
        latest_price=50.0,
        cash_flow_bad_beta=0.15,
        discount_rate_good_beta=0.35,
    )

    assert receipt.regime == "DEFENSIVE_INSULATED_PROFILE"
    assert receipt.is_defensive_insulated is True
    assert receipt.total_market_beta < 0.65
    assert verify_two_beta_audit_digest(receipt) is True


def test_calculate_two_beta_decomposition_balanced_exposure() -> None:
    """Verify average betas trigger balanced intertemporal exposure regime."""
    receipt = calculate_two_beta_decomposition(
        ticker="BAL",
        latest_price=100.0,
        cash_flow_bad_beta=0.40,
        discount_rate_good_beta=0.65,
    )

    assert receipt.regime == "BALANCED_SYSTEMATIC_EXPOSURE"
    assert receipt.is_balanced_exposure is True
    assert verify_two_beta_audit_digest(receipt) is True


def test_two_beta_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: higher bad beta increases ICAPM hurdle rate and surcharge."""
    cf_betas = [0.20, 0.40, 0.60, 0.80, 1.00]
    icapm_returns = []
    surcharges = []

    for cf in cf_betas:
        rec = calculate_two_beta_decomposition(
            ticker="MONO",
            cash_flow_bad_beta=cf,
            discount_rate_good_beta=0.60,
        )
        icapm_returns.append(rec.two_beta_icapm_return_pct)
        surcharges.append(rec.bad_beta_hurdle_surcharge_bps)

    assert all(icapm_returns[i] < icapm_returns[i + 1] for i in range(len(icapm_returns) - 1))
    assert all(surcharges[i] < surcharges[i + 1] for i in range(len(surcharges) - 1))


def test_tamper_signature_verification() -> None:
    """Verify cryptographic audit digest rejects tampered receipts."""
    receipt = calculate_two_beta_decomposition(ticker="AUDIT", cash_flow_bad_beta=0.45, discount_rate_good_beta=0.60)
    assert verify_two_beta_audit_digest(receipt) is True

    tampered = TwoBetaReceipt(
        ticker=receipt.ticker,
        latest_price=receipt.latest_price,
        total_market_beta=receipt.total_market_beta,
        cash_flow_bad_beta=receipt.cash_flow_bad_beta,
        discount_rate_good_beta=receipt.discount_rate_good_beta,
        bad_beta_ratio=receipt.bad_beta_ratio,
        relative_risk_aversion_gamma=receipt.relative_risk_aversion_gamma,
        cash_flow_risk_price_pct=receipt.cash_flow_risk_price_pct,
        discount_rate_risk_price_pct=receipt.discount_rate_risk_price_pct,
        standard_capm_return_pct=receipt.standard_capm_return_pct,
        two_beta_icapm_return_pct=receipt.two_beta_icapm_return_pct + 1.0,  # Tampered!
        bad_beta_hurdle_surcharge_bps=receipt.bad_beta_hurdle_surcharge_bps,
        regime=receipt.regime,
        is_cash_flow_vulnerable=receipt.is_cash_flow_vulnerable,
        is_discount_rate_growth=receipt.is_discount_rate_growth,
        is_defensive_insulated=receipt.is_defensive_insulated,
        is_balanced_exposure=receipt.is_balanced_exposure,
        summary=receipt.summary,
        audit_digest=receipt.audit_digest,
        calculated_at=receipt.calculated_at,
    )

    assert verify_two_beta_audit_digest(tampered) is False


def test_security_validation_ticker_and_bounds() -> None:
    """Verify security validation rejects invalid tickers and parameter bounds."""
    with pytest.raises(SecurityValidationError):
        calculate_two_beta_decomposition(ticker="TOOLONG_TICKER")

    with pytest.raises(SecurityValidationError):
        calculate_two_beta_decomposition(ticker="AAPL", latest_price=-5.0)

    with pytest.raises(SecurityValidationError):
        calculate_two_beta_decomposition(ticker="AAPL", cash_flow_bad_beta=5.0)

    with pytest.raises(SecurityValidationError):
        calculate_two_beta_decomposition(ticker="AAPL", risk_aversion_gamma=0.5)

    with pytest.raises(SecurityValidationError):
        calculate_two_beta_decomposition(ticker="AAPL", market_variance=0.0)


def test_evaluate_two_beta_profile_synthetic_observations() -> None:
    """Verify profile evaluation with synthetic prices."""
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.0004, 0.016, 75)
    prices = [float(base_price * np.exp(np.sum(returns[:i]))) for i in range(1, 76)]

    receipt = evaluate_two_beta_profile(
        ticker="SYNTH",
        prices=prices,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.cash_flow_bad_beta > 0.0
    assert receipt.discount_rate_good_beta > 0.0
    assert receipt.total_market_beta > 0.0
    assert verify_two_beta_audit_digest(receipt) is True


def test_evaluate_two_beta_profile_overrides() -> None:
    """Verify profile evaluation honors explicit overrides."""
    receipt = evaluate_two_beta_profile(
        ticker="OVRD",
        cash_flow_beta_override=0.75,
        discount_rate_beta_override=0.50,
    )

    assert receipt.ticker == "OVRD"
    assert receipt.cash_flow_bad_beta == 0.75
    assert receipt.discount_rate_good_beta == 0.50
    assert receipt.is_cash_flow_vulnerable is True


def test_cli_bad_beta_human_readable(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs human-readable table."""
    args = argparse.Namespace(
        ticker="AAPL",
        beta_cf=0.45,
        beta_dr=0.65,
        gamma=6.0,
        json=False,
    )

    bad_beta_command(args)
    captured = capsys.readouterr()

    assert "NextOmni Stock: Campbell-Vuolteenaho (2004) Two-Beta Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Cash-Flow Beta ('Bad Beta'):" in captured.out
    assert "Discount-Rate Beta ('Good Beta'):" in captured.out
    assert "Audit Digest (HMAC):" in captured.out


def test_cli_bad_beta_json_output(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs valid JSON."""
    args = argparse.Namespace(
        ticker="MSFT",
        beta_cf=0.30,
        beta_dr=0.80,
        gamma=5.5,
        json=True,
    )

    bad_beta_command(args)
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert data["ticker"] == "MSFT"
    assert "cash_flow_bad_beta" in data
    assert "discount_rate_good_beta" in data
    assert "two_beta_icapm_return_pct" in data
    assert "audit_digest" in data


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify SQLite persistence and parameterized retrieval."""
    test_db = tmp_path / "test_two_beta.db"

    receipt = calculate_two_beta_decomposition(
        ticker="GOOGL",
        latest_price=160.0,
        cash_flow_bad_beta=0.35,
        discount_rate_good_beta=0.70,
    )

    persist_two_beta_record(receipt, db_path=test_db)

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, cash_flow_bad_beta, discount_rate_good_beta, audit_digest FROM bad_beta_good_beta_records WHERE ticker = ?",
        ("GOOGL",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "GOOGL"
    assert row[1] == 160.0
    assert row[2] == 0.35
    assert row[3] == 0.70
    assert row[4] == receipt.audit_digest

