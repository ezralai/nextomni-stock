"""Unit, boundary, security, and integration tests for Easley-Hvidkjaer-O'Hara (2002, 2004)

Information Risk & Cost of Capital Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.information_risk_capital import (
    InformationRiskReceipt,
    calculate_information_risk_premium,
    classify_information_risk_regime,
    evaluate_information_risk_profile,
    persist_information_risk_record,
    verify_information_risk_audit_digest,
)
from src.cli import information_risk_command
from src.security.validators import SecurityValidationError


def test_calculate_information_risk_premium_severe_surcharge() -> None:
    """Verify high PIN triggers severe adverse selection surcharge regime."""
    receipt = calculate_information_risk_premium(
        ticker="ASSET1",
        latest_price=100.0,
        pin=0.35,  # 35% PIN vs 19.1% baseline
        market_equity_beta=1.20,
        risk_free_rate=0.0425,
        equity_risk_premium=0.0550,
        lambda_info=0.2500,
    )

    assert receipt.ticker == "ASSET1"
    assert receipt.pin == 0.35
    assert receipt.cost_of_capital_surcharge_pct > 2.25
    assert receipt.is_severe_surcharge is True
    assert receipt.regime == "SEVERE_ADVERSE_SELECTION_SURCHARGE"
    assert receipt.info_risk_beta > 1.50
    assert receipt.total_required_hurdle_pct > receipt.capm_required_return_pct
    assert verify_information_risk_audit_digest(receipt) is True


def test_calculate_information_risk_premium_elevated_risk() -> None:
    """Verify moderately elevated PIN produces elevated information risk regime."""
    receipt = calculate_information_risk_premium(
        ticker="ASSET2",
        latest_price=75.0,
        pin=0.24,  # 24% PIN
        market_equity_beta=1.00,
    )

    assert receipt.is_elevated_risk is True
    assert receipt.regime == "ELEVATED_INFORMATION_RISK"
    assert 0.70 <= receipt.cost_of_capital_surcharge_pct < 2.25
    assert verify_information_risk_audit_digest(receipt) is True


def test_calculate_information_risk_premium_benign_transparent() -> None:
    """Verify baseline benchmark PIN produces benign transparent liquidity regime."""
    receipt = calculate_information_risk_premium(
        ticker="ASSET3",
        latest_price=150.0,
        pin=0.1910,  # exactly market benchmark
        market_equity_beta=1.00,
    )

    assert receipt.is_benign_transparent is True
    assert receipt.regime == "BENIGN_TRANSPARENT_LIQUIDITY"
    assert abs(receipt.cost_of_capital_surcharge_pct) < 0.10
    assert abs(receipt.info_risk_beta) < 0.10
    assert verify_information_risk_audit_digest(receipt) is True


def test_calculate_information_risk_premium_minimal_discount() -> None:
    """Verify very low PIN produces minimal disclosure discount regime."""
    receipt = calculate_information_risk_premium(
        ticker="ASSET4",
        latest_price=200.0,
        pin=0.10,  # 10% PIN - highly public disclosure
        market_equity_beta=0.80,
    )

    assert receipt.is_minimal_discount is True
    assert receipt.regime == "MINIMAL_DISCLOSURE_DISCOUNT"
    assert receipt.cost_of_capital_surcharge_pct < -1.25
    assert receipt.total_required_hurdle_pct < receipt.capm_required_return_pct
    assert verify_information_risk_audit_digest(receipt) is True


def test_information_risk_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: higher PIN increases info beta, surcharge, and hurdle."""
    pins = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    surcharges = []
    hurdles = []
    betas = []

    for p in pins:
        r = calculate_information_risk_premium(ticker="MONO", pin=p)
        surcharges.append(r.cost_of_capital_surcharge_pct)
        hurdles.append(r.total_required_hurdle_pct)
        betas.append(r.info_risk_beta)

    # All must be strictly increasing
    assert all(surcharges[i] < surcharges[i + 1] for i in range(len(surcharges) - 1))
    assert all(hurdles[i] < hurdles[i + 1] for i in range(len(hurdles) - 1))
    assert all(betas[i] < betas[i + 1] for i in range(len(betas) - 1))


def test_tamper_signature_verification() -> None:
    """Verify cryptographic audit digest detects tampered records."""
    receipt = calculate_information_risk_premium(ticker="SEC1", pin=0.22)
    assert verify_information_risk_audit_digest(receipt) is True

    tampered = InformationRiskReceipt(
        ticker=receipt.ticker,
        latest_price=receipt.latest_price,
        pin=receipt.pin,
        market_equity_beta=receipt.market_equity_beta,
        info_risk_beta=receipt.info_risk_beta,
        baseline_market_pin=receipt.baseline_market_pin,
        cost_of_capital_surcharge_pct=receipt.cost_of_capital_surcharge_pct + 1.0,  # Tampered
        capm_required_return_pct=receipt.capm_required_return_pct,
        total_required_hurdle_pct=receipt.total_required_hurdle_pct,
        cost_of_capital_surcharge_bps=receipt.cost_of_capital_surcharge_bps,
        lambda_info_factor_premium=receipt.lambda_info_factor_premium,
        regime=receipt.regime,
        is_severe_surcharge=receipt.is_severe_surcharge,
        is_elevated_risk=receipt.is_elevated_risk,
        is_benign_transparent=receipt.is_benign_transparent,
        is_minimal_discount=receipt.is_minimal_discount,
        summary=receipt.summary,
        audit_digest=receipt.audit_digest,
        calculated_at=receipt.calculated_at,
    )

    assert verify_information_risk_audit_digest(tampered) is False


def test_security_validation_ticker_and_bounds() -> None:
    """Verify strict validation rejects malicious or out-of-bounds inputs."""
    with pytest.raises(SecurityValidationError):
        calculate_information_risk_premium(ticker="BAD;DROP")

    with pytest.raises(SecurityValidationError):
        calculate_information_risk_premium(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_information_risk_premium(ticker="AAPL", pin=0.99)  # max 0.85

    with pytest.raises(SecurityValidationError):
        calculate_information_risk_premium(ticker="AAPL", risk_free_rate=-0.01)

    with pytest.raises(SecurityValidationError):
        calculate_information_risk_premium(ticker="AAPL", lambda_info=0.0)


def test_evaluate_information_risk_profile_synthetic_observations() -> None:
    """Verify profile evaluation with synthetic price series."""
    np.random.seed(123)
    base_price = 100.0
    returns = np.random.normal(0.0003, 0.015, 80)
    prices = [float(base_price * np.exp(np.sum(returns[:i]))) for i in range(1, 81)]
    volumes = [float(np.random.uniform(500_000, 2_000_000)) for _ in range(80)]

    receipt = evaluate_information_risk_profile(
        ticker="SYNTH",
        prices=prices,
        volumes=volumes,
    )

    assert receipt.ticker == "SYNTH"
    assert 0.05 <= receipt.pin <= 0.55
    assert receipt.latest_price > 0.0
    assert verify_information_risk_audit_digest(receipt) is True


def test_evaluate_information_risk_profile_overrides() -> None:
    """Verify explicit parameter overrides take precedence."""
    receipt = evaluate_information_risk_profile(
        ticker="OVRD",
        pin_override=0.32,
        market_beta_override=1.45,
    )

    assert receipt.ticker == "OVRD"
    assert receipt.pin == 0.32
    assert receipt.market_equity_beta == 1.45
    assert receipt.is_severe_surcharge is True


def test_cli_information_risk_human_readable(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI prints human-readable telemetry output."""
    args = argparse.Namespace(
        ticker="AAPL",
        pin=0.22,
        beta=1.10,
        rf=0.0425,
        erp=0.0550,
        lambda_info=0.2500,
        json=False,
    )

    information_risk_command(args)
    captured = capsys.readouterr()

    assert "NextOmni Stock: Easley-Hvidkjaer-O'Hara (2002, 2004) Information Risk Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Probability of Informed Trading:" in captured.out
    assert "Information Risk Surcharge:" in captured.out
    assert "Total Required Hurdle Rate:" in captured.out
    assert "Audit Digest (HMAC):" in captured.out


def test_cli_information_risk_json_output(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI outputs structured JSON."""
    args = argparse.Namespace(
        ticker="GOOGL",
        pin=0.18,
        beta=1.05,
        rf=0.0425,
        erp=0.0550,
        lambda_info=0.2500,
        json=True,
    )

    information_risk_command(args)
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert data["ticker"] == "GOOGL"
    assert "pin" in data
    assert "cost_of_capital_surcharge_pct" in data
    assert "total_required_hurdle_pct" in data
    assert "audit_digest" in data


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify SQLite persistence and parameterized retrieval."""
    test_db = tmp_path / "test_info_risk.db"

    receipt = calculate_information_risk_premium(
        ticker="ORCL",
        latest_price=130.0,
        pin=0.25,
    )

    persist_information_risk_record(receipt, db_path=test_db)

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, pin, cost_of_capital_surcharge_pct, audit_digest FROM information_risk_records WHERE ticker = ?",
        ("ORCL",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "ORCL"
    assert row[1] == 130.0
    assert row[2] == 0.25
    assert row[4] == receipt.audit_digest

