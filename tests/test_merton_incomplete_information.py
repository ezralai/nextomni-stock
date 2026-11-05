"""Unit, boundary, security, and integration tests for Robert C. Merton (1987)

Incomplete Information & Investor Recognition Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.merton_incomplete_information import (
    MertonIncompleteInfoReceipt,
    calculate_merton_incomplete_information,
    classify_merton_regime,
    evaluate_merton_incomplete_info_profile,
    persist_merton_incomplete_info_record,
    verify_merton_audit_digest,
)
from src.cli import merton_recognition_command
from src.security.validators import SecurityValidationError


def test_calculate_merton_incomplete_information_neglected_firm() -> None:
    """Verify low recognition ratio triggers neglected firm high surcharge regime."""
    receipt = calculate_merton_incomplete_information(
        ticker="NEGLCT",
        latest_price=50.0,
        investor_recognition_ratio_q=0.15,
        idiosyncratic_variance=0.0600,
        market_equity_beta=1.10,
        shadow_cost_lambda=0.25,
    )

    assert receipt.ticker == "NEGLCT"
    assert receipt.investor_recognition_ratio_q == 0.15
    assert receipt.merton_shadow_surcharge_pct >= 2.0
    assert receipt.is_neglected_firm is True
    assert receipt.regime == "NEGLECTED_FIRM_HIGH_SURCHARGE"
    assert receipt.total_required_hurdle_pct > receipt.capm_required_return_pct
    assert verify_merton_audit_digest(receipt) is True


def test_calculate_merton_incomplete_information_elevated() -> None:
    """Verify moderate recognition ratio triggers elevated incomplete information regime."""
    receipt = calculate_merton_incomplete_information(
        ticker="MIDCAP",
        latest_price=80.0,
        investor_recognition_ratio_q=0.40,
        idiosyncratic_variance=0.0400,
    )

    assert receipt.is_elevated_incomplete is True
    assert receipt.regime == "ELEVATED_INCOMPLETE_INFORMATION"
    assert verify_merton_audit_digest(receipt) is True


def test_calculate_merton_incomplete_information_broad_recognition() -> None:
    """Verify strong recognition ratio triggers broad recognition equilibrium regime."""
    receipt = calculate_merton_incomplete_information(
        ticker="BROAD",
        latest_price=120.0,
        investor_recognition_ratio_q=0.70,
        idiosyncratic_variance=0.0300,
    )

    assert receipt.is_broad_recognition is True
    assert receipt.regime == "BROAD_RECOGNITION_EQUILIBRIUM"
    assert verify_merton_audit_digest(receipt) is True


def test_calculate_merton_incomplete_information_universal_saturated() -> None:
    """Verify mega-cap high recognition ratio triggers universal saturated coverage regime."""
    receipt = calculate_merton_incomplete_information(
        ticker="MEGAC",
        latest_price=250.0,
        investor_recognition_ratio_q=0.95,
        idiosyncratic_variance=0.0200,
    )

    assert receipt.is_universal_saturated is True
    assert receipt.regime == "UNIVERSAL_COVERAGE_SATURATED"
    assert receipt.merton_shadow_surcharge_pct < 1.0
    assert verify_merton_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: lower recognition q strictly increases surcharge and hurdle."""
    q_values = [0.10, 0.25, 0.45, 0.65, 0.85]
    surcharges = []
    hurdles = []

    for q in q_values:
        rec = calculate_merton_incomplete_information(
            ticker="MONO",
            investor_recognition_ratio_q=q,
            idiosyncratic_variance=0.0400,
        )
        surcharges.append(rec.merton_shadow_surcharge_pct)
        hurdles.append(rec.total_required_hurdle_pct)

    # Lower q means higher surcharge -> surcharges must be strictly decreasing as q increases
    assert all(surcharges[i] > surcharges[i + 1] for i in range(len(surcharges) - 1))
    assert all(hurdles[i] > hurdles[i + 1] for i in range(len(hurdles) - 1))


def test_tamper_signature_verification() -> None:
    """Verify cryptographic audit digest rejects tampered receipts."""
    receipt = calculate_merton_incomplete_information(ticker="AUDIT", investor_recognition_ratio_q=0.50)
    assert verify_merton_audit_digest(receipt) is True

    tampered = MertonIncompleteInfoReceipt(
        ticker=receipt.ticker,
        latest_price=receipt.latest_price,
        investor_recognition_ratio_q=receipt.investor_recognition_ratio_q,
        idiosyncratic_variance=receipt.idiosyncratic_variance,
        market_equity_beta=receipt.market_equity_beta,
        shadow_cost_lambda=receipt.shadow_cost_lambda,
        capm_required_return_pct=receipt.capm_required_return_pct,
        merton_shadow_surcharge_pct=receipt.merton_shadow_surcharge_pct + 1.0,  # Tampered!
        merton_shadow_surcharge_bps=receipt.merton_shadow_surcharge_bps,
        total_required_hurdle_pct=receipt.total_required_hurdle_pct,
        regime=receipt.regime,
        is_neglected_firm=receipt.is_neglected_firm,
        is_elevated_incomplete=receipt.is_elevated_incomplete,
        is_broad_recognition=receipt.is_broad_recognition,
        is_universal_saturated=receipt.is_universal_saturated,
        summary=receipt.summary,
        audit_digest=receipt.audit_digest,
        calculated_at=receipt.calculated_at,
    )

    assert verify_merton_audit_digest(tampered) is False


def test_security_validation_ticker_and_bounds() -> None:
    """Verify security validation rejects invalid tickers and out-of-bounds parameters."""
    with pytest.raises(SecurityValidationError):
        calculate_merton_incomplete_information(ticker="INVALID_TOOLONG")

    with pytest.raises(SecurityValidationError):
        calculate_merton_incomplete_information(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_merton_incomplete_information(ticker="AAPL", investor_recognition_ratio_q=1.50)

    with pytest.raises(SecurityValidationError):
        calculate_merton_incomplete_information(ticker="AAPL", idiosyncratic_variance=0.0)

    with pytest.raises(SecurityValidationError):
        calculate_merton_incomplete_information(ticker="AAPL", shadow_cost_lambda=0.0)


def test_evaluate_merton_incomplete_info_profile_synthetic_observations() -> None:
    """Verify profile evaluation with synthetic price and volume data."""
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.0004, 0.018, 80)
    prices = [float(base_price * np.exp(np.sum(returns[:i]))) for i in range(1, 81)]
    volumes = [float(np.random.uniform(500_000, 2_000_000)) for _ in range(80)]

    receipt = evaluate_merton_incomplete_info_profile(
        ticker="SYNTH",
        prices=prices,
        volumes=volumes,
    )

    assert receipt.ticker == "SYNTH"
    assert 0.05 <= receipt.investor_recognition_ratio_q <= 1.00
    assert receipt.idiosyncratic_variance > 0.0
    assert receipt.total_required_hurdle_pct > 0.0
    assert verify_merton_audit_digest(receipt) is True


def test_evaluate_merton_incomplete_info_profile_overrides() -> None:
    """Verify profile evaluation honors explicit overrides."""
    receipt = evaluate_merton_incomplete_info_profile(
        ticker="OVRD",
        recognition_q_override=0.20,
        shadow_lambda_override=0.35,
    )

    assert receipt.ticker == "OVRD"
    assert receipt.investor_recognition_ratio_q == 0.20
    assert receipt.shadow_cost_lambda == 0.35
    assert receipt.is_neglected_firm is True


def test_cli_merton_recognition_human_readable(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs human-readable table."""
    args = argparse.Namespace(
        ticker="AAPL",
        recognition_q=0.60,
        shadow_lambda=0.25,
        json=False,
    )

    merton_recognition_command(args)
    captured = capsys.readouterr()

    assert "NextOmni Stock: Merton (1987) Incomplete Information Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Investor Recognition Ratio (q):" in captured.out
    assert "Merton Shadow Surcharge:" in captured.out
    assert "Audit Digest (HMAC):" in captured.out


def test_cli_merton_recognition_json_output(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs valid JSON."""
    args = argparse.Namespace(
        ticker="MSFT",
        recognition_q=0.85,
        shadow_lambda=0.20,
        json=True,
    )

    merton_recognition_command(args)
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert data["ticker"] == "MSFT"
    assert "investor_recognition_ratio_q" in data
    assert "merton_shadow_surcharge_pct" in data
    assert "total_required_hurdle_pct" in data
    assert "audit_digest" in data


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify SQLite persistence and parameterized retrieval."""
    test_db = tmp_path / "test_merton.db"

    receipt = calculate_merton_incomplete_information(
        ticker="AMZN",
        latest_price=180.0,
        investor_recognition_ratio_q=0.65,
    )

    persist_merton_incomplete_info_record(receipt, db_path=test_db)

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, investor_recognition_ratio_q, merton_shadow_surcharge_pct, audit_digest FROM merton_incomplete_info_records WHERE ticker = ?",
        ("AMZN",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "AMZN"
    assert row[1] == 180.0
    assert row[2] == 0.65
    assert row[4] == receipt.audit_digest
