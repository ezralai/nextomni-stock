"""Tests for Acharya et al. (2017) Systemic Expected Shortfall (SES) Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.systemic_expected_shortfall import (
    SystemicExpectedShortfallReceipt,
    calculate_systemic_expected_shortfall,
    classify_systemic_risk_regime,
    evaluate_ses_profile,
)
from src.cli import ses_command
from src.security.validators import SecurityValidationError


def test_calculate_systemic_expected_shortfall_critical_threat() -> None:
    """Verify critical systemic externality threat when MES >= 4.5, Lev >= 15.0, and SES >= 6.0."""
    receipt = calculate_systemic_expected_shortfall(
        ticker="C",
        latest_price=55.0,
        market_tail_threshold_pct=2.0,
        marginal_expected_shortfall_pct=5.50,
        institution_leverage=22.0,
        baseline_leverage=10.0,
    )
    assert receipt.ticker == "C"
    assert receipt.marginal_expected_shortfall_pct >= 4.5
    assert receipt.institution_leverage >= 15.0
    assert receipt.systemic_expected_shortfall_pct >= 6.0
    assert receipt.systemic_capital_surcharge_pct > 1.0
    assert receipt.regime == "CRITICAL_SYSTEMIC_EXTERNALITY_THREAT"
    assert receipt.is_critical_systemic_threat is True
    assert receipt.is_defensive_systemic_resilience is False
    assert receipt.expected_systemic_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_systemic_expected_shortfall_defensive_resilience() -> None:
    """Verify defensive systemic resilience when MES <= 1.2 and Lev <= 6.0."""
    receipt = calculate_systemic_expected_shortfall(
        ticker="JNJ",
        latest_price=160.0,
        market_tail_threshold_pct=2.0,
        marginal_expected_shortfall_pct=0.90,
        institution_leverage=3.5,
        baseline_leverage=5.0,
    )
    assert receipt.marginal_expected_shortfall_pct <= 1.2
    assert receipt.institution_leverage <= 6.0
    assert receipt.regime == "DEFENSIVE_SYSTEMIC_RESILIENCE"
    assert receipt.is_defensive_systemic_resilience is True
    assert receipt.is_critical_systemic_threat is False
    assert receipt.expected_systemic_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_systemic_expected_shortfall_elevated_tail() -> None:
    """Verify elevated marginal tail sensitivity when MES >= 3.0."""
    receipt = calculate_systemic_expected_shortfall(
        ticker="AMD",
        latest_price=150.0,
        market_tail_threshold_pct=2.0,
        marginal_expected_shortfall_pct=3.60,
        institution_leverage=8.0,
        baseline_leverage=8.0,
    )
    assert receipt.marginal_expected_shortfall_pct >= 3.0
    assert receipt.regime == "ELEVATED_MARGINAL_TAIL_SENSITIVITY"
    assert receipt.is_elevated_tail_sensitivity is True
    assert receipt.is_critical_systemic_threat is False
    assert receipt.verify_integrity() is True


def test_calculate_systemic_expected_shortfall_neutral() -> None:
    """Verify normal systemic capital equilibrium when values are within balanced boundaries."""
    receipt = calculate_systemic_expected_shortfall(
        ticker="AAPL",
        latest_price=220.0,
        market_tail_threshold_pct=2.0,
        marginal_expected_shortfall_pct=2.20,
        institution_leverage=7.0,
        baseline_leverage=7.0,
    )
    assert receipt.regime == "NORMAL_SYSTEMIC_CAPITAL_EQUILIBRIUM"
    assert receipt.is_critical_systemic_threat is False
    assert receipt.is_defensive_systemic_resilience is False
    assert receipt.verify_integrity() is True


def test_classify_systemic_risk_regime_direct() -> None:
    """Directly test classify_systemic_risk_regime boundary logic."""
    assert classify_systemic_risk_regime(5.0, 18.0, 7.5) == "CRITICAL_SYSTEMIC_EXTERNALITY_THREAT"
    assert classify_systemic_risk_regime(0.8, 4.0, 0.5) == "DEFENSIVE_SYSTEMIC_RESILIENCE"
    assert classify_systemic_risk_regime(3.5, 8.0, 3.0) == "ELEVATED_MARGINAL_TAIL_SENSITIVITY"
    assert classify_systemic_risk_regime(2.0, 8.0, 2.0) == "NORMAL_SYSTEMIC_CAPITAL_EQUILIBRIUM"


def test_tamper_signature_verification() -> None:
    """Verify HMAC-SHA256 signature detects any modified field."""
    receipt = calculate_systemic_expected_shortfall(
        ticker="GS",
        latest_price=420.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with numerical field
    tampered_dict = receipt.to_dict()
    tampered_dict["systemic_expected_shortfall_pct"] = 99.9
    tampered_receipt = SystemicExpectedShortfallReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with signature
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["signature"] = "badc0ffee11111111"
    tampered_receipt2 = SystemicExpectedShortfallReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_security_validation_ticker_and_bounds() -> None:
    """Ensure invalid tickers and out-of-bound arguments raise errors."""
    for mal in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_systemic_expected_shortfall(ticker=mal, latest_price=100.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_systemic_expected_shortfall(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(ValueError, match="market_tail_threshold_pct must be in"):
        calculate_systemic_expected_shortfall(ticker="AAPL", latest_price=100.0, market_tail_threshold_pct=25.0)

    with pytest.raises(ValueError, match="marginal_expected_shortfall_pct must be in"):
        calculate_systemic_expected_shortfall(ticker="AAPL", latest_price=100.0, marginal_expected_shortfall_pct=50.0)

    with pytest.raises(ValueError, match="institution_leverage must be in"):
        calculate_systemic_expected_shortfall(ticker="AAPL", latest_price=100.0, institution_leverage=90.0)


def test_evaluate_ses_profile_with_observations(tmp_path: Path) -> None:
    """Verify end-to-end evaluation using synthetic database observations."""
    test_db = tmp_path / "test_ses.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="AAPL",
        exchange="XNAS",
        company_name="Apple Inc.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(35):
        p = 150.0 + i * 0.8
        records.append({
            "ticker": "AAPL",
            "exchange": "XNAS",
            "session_date": f"2026-01-{i+1:02d}",
            "open": p,
            "high": p + 1.0,
            "low": p - 1.0,
            "close": p,
            "adj_close": p,
            "volume": 1_200_000,
            "data_status": "END_OF_DAY",
            "provider": "MOCK",
        })
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    receipt = evaluate_ses_profile(ticker="AAPL", db_path=test_db)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(177.2, abs=1e-2)
    assert receipt.marginal_expected_shortfall_pct > 0.0
    assert receipt.verify_integrity() is True


def test_evaluate_ses_profile_overrides(tmp_path: Path) -> None:
    """Verify parameter overrides take precedence over inferred defaults."""
    test_db = tmp_path / "test_ses_override.db"
    db.init_db(test_db)

    receipt = evaluate_ses_profile(
        ticker="AAPL",
        db_path=test_db,
        marginal_expected_shortfall_pct=5.20,
        institution_leverage=20.0,
        baseline_leverage=10.0,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.marginal_expected_shortfall_pct == 5.20
    assert receipt.institution_leverage == 20.0
    assert receipt.regime == "CRITICAL_SYSTEMIC_EXTERNALITY_THREAT"
    assert receipt.is_critical_systemic_threat is True


def test_cli_ses_json_output(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI ses command output in JSON mode."""
    test_db = tmp_path / "test_cli_ses.db"
    db.init_db(test_db)

    args = argparse.Namespace(
        ticker="MSFT",
        market_tail_threshold=2.5,
        marginal_shortfall=4.0,
        leverage=12.0,
        baseline_leverage=10.0,
        json=True,
    )
    ses_command(args)

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["ticker"] == "MSFT"
    assert data["marginal_expected_shortfall_pct"] == 4.0
    assert "signature" in data
    assert len(data["signature"]) == 64


def test_cli_ses_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI ses command standard formatted output."""
    args = argparse.Namespace(
        ticker="GOOGL",
        market_tail_threshold=2.0,
        marginal_shortfall=2.8,
        leverage=6.0,
        baseline_leverage=6.0,
        json=False,
    )
    ses_command(args)

    out, _ = capsys.readouterr()
    assert "NextOmni Stock: Acharya et al. (2017) Systemic Expected Shortfall (SES) Engine" in out
    assert "Security / Ticker:            GOOGL" in out
    assert "Verified: True" in out


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify parameterized database queries and clean execution without SQL injection."""
    test_db = tmp_path / "test_ses_db.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="NVDA",
        exchange="XNAS",
        company_name="Nvidia Corp.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(25):
        records.append({
            "ticker": "NVDA",
            "exchange": "XNAS",
            "session_date": f"2026-01-{i+1:02d}",
            "open": 120.0,
            "high": 125.0,
            "low": 118.0,
            "close": 122.0,
            "adj_close": 122.0,
            "volume": 2_000_000,
            "data_status": "END_OF_DAY",
            "provider": "MOCK",
        })
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    obs = db.get_observations("NVDA", db_path=test_db)
    assert len(obs) == 25

