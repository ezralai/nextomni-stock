"""Tests for Adrian & Brunnermeier (2016) CoVaR & Systemic Spillover Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.covar_systemic_spillover import (
    CoVaRSpilloverReceipt,
    calculate_covar_spillover,
    classify_covar_regime,
    evaluate_covar_profile,
)
from src.cli import covar_command
from src.security.validators import SecurityValidationError


def test_calculate_covar_spillover_critical_threat() -> None:
    """Verify high systemic spillover threat when Delta-CoVaR >= 3.50 and tail_beta >= 1.20."""
    receipt = calculate_covar_spillover(
        ticker="JPM",
        latest_price=190.0,
        tail_quantile=0.05,
        standalone_var_pct=6.50,
        system_unconditional_var_pct=2.50,
        tail_beta=1.45,
    )
    assert receipt.ticker == "JPM"
    assert receipt.delta_covar_pct >= 3.50
    assert receipt.tail_beta >= 1.20
    assert receipt.regime == "HIGH_SYSTEMIC_SPILLOVER_CONTAGION"
    assert receipt.is_critical_spillover_threat is True
    assert receipt.is_systemic_distress_absorber is False
    assert receipt.expected_covar_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_covar_spillover_distress_absorber() -> None:
    """Verify systemic distress absorber when Delta-CoVaR <= 0.80 and tail_beta <= 0.45."""
    receipt = calculate_covar_spillover(
        ticker="PG",
        latest_price=160.0,
        tail_quantile=0.05,
        standalone_var_pct=2.00,
        system_unconditional_var_pct=2.40,
        tail_beta=0.35,
    )
    assert receipt.delta_covar_pct <= 0.80
    assert receipt.tail_beta <= 0.45
    assert receipt.regime == "SYSTEMIC_DISTRESS_ABSORBER"
    assert receipt.is_systemic_distress_absorber is True
    assert receipt.is_critical_spillover_threat is False
    assert receipt.expected_covar_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_covar_spillover_elevated() -> None:
    """Verify elevated tail spillover when Delta-CoVaR >= 1.80."""
    receipt = calculate_covar_spillover(
        ticker="GOOGL",
        latest_price=175.0,
        tail_quantile=0.05,
        standalone_var_pct=4.20,
        system_unconditional_var_pct=2.40,
        tail_beta=1.10,
    )
    assert receipt.delta_covar_pct >= 1.80
    assert receipt.regime == "ELEVATED_TAIL_SPILLOVER"
    assert receipt.is_elevated_spillover is True
    assert receipt.is_critical_spillover_threat is False
    assert receipt.verify_integrity() is True


def test_calculate_covar_spillover_neutral() -> None:
    """Verify normal tail CoVaR equilibrium when metrics are within standard boundaries."""
    receipt = calculate_covar_spillover(
        ticker="AAPL",
        latest_price=220.0,
        tail_quantile=0.05,
        standalone_var_pct=2.80,
        system_unconditional_var_pct=2.40,
        tail_beta=0.75,
    )
    assert receipt.regime == "NORMAL_TAIL_COVAR_EQUILIBRIUM"
    assert receipt.is_critical_spillover_threat is False
    assert receipt.is_systemic_distress_absorber is False
    assert receipt.verify_integrity() is True


def test_classify_covar_regime_direct() -> None:
    """Directly test classify_covar_regime boundary logic."""
    assert classify_covar_regime(4.0, 1.30, 3.0) == "HIGH_SYSTEMIC_SPILLOVER_CONTAGION"
    assert classify_covar_regime(0.60, 0.40, 0.80) == "SYSTEMIC_DISTRESS_ABSORBER"
    assert classify_covar_regime(2.20, 0.90, 2.0) == "ELEVATED_TAIL_SPILLOVER"
    assert classify_covar_regime(1.20, 0.80, 1.5) == "NORMAL_TAIL_COVAR_EQUILIBRIUM"


def test_tamper_signature_verification() -> None:
    """Verify HMAC-SHA256 signature detects any modified field."""
    receipt = calculate_covar_spillover(
        ticker="MSFT",
        latest_price=420.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with numerical field
    tampered_dict = receipt.to_dict()
    tampered_dict["delta_covar_pct"] = 14.5
    tampered_receipt = CoVaRSpilloverReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with signature
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["signature"] = "deadbeef00000000"
    tampered_receipt2 = CoVaRSpilloverReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_security_validation_ticker_and_bounds() -> None:
    """Ensure invalid tickers and out-of-bound arguments raise errors."""
    for mal in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_covar_spillover(ticker=mal, latest_price=100.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_covar_spillover(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(ValueError, match="tail_quantile must be in"):
        calculate_covar_spillover(ticker="AAPL", latest_price=100.0, tail_quantile=0.50)

    with pytest.raises(ValueError, match="standalone_var_pct must be in"):
        calculate_covar_spillover(ticker="AAPL", latest_price=100.0, standalone_var_pct=50.0)

    with pytest.raises(ValueError, match="tail_beta must be in"):
        calculate_covar_spillover(ticker="AAPL", latest_price=100.0, tail_beta=5.0)


def test_evaluate_covar_profile_with_observations(tmp_path: Path) -> None:
    """Verify end-to-end evaluation using synthetic database observations."""
    test_db = tmp_path / "test_covar.db"
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

    receipt = evaluate_covar_profile(ticker="AAPL", db_path=test_db)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(177.2, abs=1e-2)
    assert receipt.standalone_var_pct > 0.0
    assert receipt.verify_integrity() is True


def test_evaluate_covar_profile_overrides(tmp_path: Path) -> None:
    """Verify parameter overrides take precedence over inferred defaults."""
    test_db = tmp_path / "test_covar_override.db"
    db.init_db(test_db)

    receipt = evaluate_covar_profile(
        ticker="AAPL",
        db_path=test_db,
        standalone_var_pct=7.0,
        system_unconditional_var_pct=2.5,
        tail_beta=1.5,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.standalone_var_pct == 7.0
    assert receipt.tail_beta == 1.5
    assert receipt.regime == "HIGH_SYSTEMIC_SPILLOVER_CONTAGION"
    assert receipt.is_critical_spillover_threat is True


def test_cli_covar_json_output(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI covar command output in JSON mode."""
    test_db = tmp_path / "test_cli_covar.db"
    db.init_db(test_db)

    args = argparse.Namespace(
        ticker="MSFT",
        quantile=0.05,
        standalone_var=4.0,
        system_var=2.2,
        tail_beta=1.1,
        json=True,
    )
    covar_command(args)

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["ticker"] == "MSFT"
    assert data["standalone_var_pct"] == 4.0
    assert "signature" in data
    assert len(data["signature"]) == 64


def test_cli_covar_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI covar command standard formatted output."""
    args = argparse.Namespace(
        ticker="GOOGL",
        quantile=0.05,
        standalone_var=3.2,
        system_var=2.4,
        tail_beta=0.9,
        json=False,
    )
    covar_command(args)

    out, _ = capsys.readouterr()
    assert "NextOmni Stock: Adrian & Brunnermeier (2016) CoVaR Spillover Engine" in out
    assert "Security / Ticker:            GOOGL" in out
    assert "Verified: True" in out


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify parameterized database queries and clean execution without SQL injection."""
    test_db = tmp_path / "test_covar_db.db"
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

