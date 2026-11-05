"""Tests for Campbell, Hilscher & Szilagyi (2008) Financial Distress Puzzle Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.campbell_distress_puzzle import (
    CampbellDistressReceipt,
    calculate_campbell_distress,
    classify_distress_regime,
    evaluate_campbell_distress_profile,
)
from src.cli import chs_distress_command
from src.security.validators import SecurityValidationError


def test_calculate_campbell_distress_overpriced_trap() -> None:
    """Verify distress anomaly overpriced trap when failure probability >= 2.50%."""
    receipt = calculate_campbell_distress(
        ticker="ZOMB",
        latest_price=4.50,
        net_income_to_market_assets=-0.25,
        total_liabilities_to_market_assets=0.85,
        cash_to_market_assets=0.03,
        trailing_excess_return=-0.65,
        idiosyncratic_volatility_sigma=0.75,
        relative_firm_size=-10.0,
        market_to_book_ratio=1.20,
    )
    assert receipt.ticker == "ZOMB"
    assert receipt.failure_probability_12m_pct >= 2.50
    assert receipt.distress_anomaly_alpha_drag_pct < 0.0
    assert receipt.regime == "DISTRESS_ANOMALY_OVERPRICED_TRAP"
    assert receipt.is_distress_anomaly_trap is True
    assert receipt.is_solvent_financial_fortress is False
    assert receipt.expected_distress_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_campbell_distress_solvent_fortress() -> None:
    """Verify solvent financial fortress when failure probability <= 0.10% and cash is robust."""
    receipt = calculate_campbell_distress(
        ticker="AAPL",
        latest_price=220.0,
        net_income_to_market_assets=0.10,
        total_liabilities_to_market_assets=0.20,
        cash_to_market_assets=0.25,
        trailing_excess_return=0.15,
        idiosyncratic_volatility_sigma=0.18,
        relative_firm_size=-3.0,
        market_to_book_ratio=5.0,
    )
    assert receipt.failure_probability_12m_pct <= 0.10
    assert receipt.cash_to_market_assets >= 0.15
    assert receipt.regime == "SOLVENT_FINANCIAL_FORTRESS"
    assert receipt.is_solvent_financial_fortress is True
    assert receipt.is_distress_anomaly_trap is False
    assert receipt.expected_distress_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_campbell_distress_elevated_watch() -> None:
    """Verify elevated financial distress watch when TLMTA >= 0.70."""
    receipt = calculate_campbell_distress(
        ticker="LEVR",
        latest_price=45.0,
        net_income_to_market_assets=0.01,
        total_liabilities_to_market_assets=0.75,
        cash_to_market_assets=0.05,
        trailing_excess_return=-0.10,
        idiosyncratic_volatility_sigma=0.35,
        relative_firm_size=-6.0,
        market_to_book_ratio=1.5,
    )
    assert receipt.total_liabilities_to_market_assets >= 0.70
    assert receipt.regime == "ELEVATED_FINANCIAL_DISTRESS_WATCH"
    assert receipt.is_elevated_distress_watch is True
    assert receipt.is_distress_anomaly_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_campbell_distress_neutral() -> None:
    """Verify normal solvency equilibrium when metrics are within healthy boundaries."""
    receipt = calculate_campbell_distress(
        ticker="MSFT",
        latest_price=410.0,
        net_income_to_market_assets=0.04,
        total_liabilities_to_market_assets=0.45,
        cash_to_market_assets=0.12,
        trailing_excess_return=0.05,
        idiosyncratic_volatility_sigma=0.22,
        relative_firm_size=-4.5,
        market_to_book_ratio=4.0,
    )
    assert receipt.regime == "NORMAL_SOLVENCY_EQUILIBRIUM"
    assert receipt.is_distress_anomaly_trap is False
    assert receipt.is_solvent_financial_fortress is False
    assert receipt.verify_integrity() is True


def test_classify_distress_regime_direct() -> None:
    """Directly test classify_distress_regime boundary logic."""
    assert classify_distress_regime(3.5, 0.80, 0.05) == "DISTRESS_ANOMALY_OVERPRICED_TRAP"
    assert classify_distress_regime(0.05, 0.20, 0.30) == "SOLVENT_FINANCIAL_FORTRESS"
    assert classify_distress_regime(1.2, 0.75, 0.05) == "ELEVATED_FINANCIAL_DISTRESS_WATCH"
    assert classify_distress_regime(0.3, 0.40, 0.10) == "NORMAL_SOLVENCY_EQUILIBRIUM"


def test_tamper_signature_verification() -> None:
    """Verify HMAC-SHA256 signature detects any modified field."""
    receipt = calculate_campbell_distress(
        ticker="NVDA",
        latest_price=120.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with numerical field
    tampered_dict = receipt.to_dict()
    tampered_dict["failure_probability_12m_pct"] = 49.9
    tampered_receipt = CampbellDistressReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with signature
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["signature"] = "deadbeef33333333"
    tampered_receipt2 = CampbellDistressReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_security_validation_ticker_and_bounds() -> None:
    """Ensure invalid tickers and out-of-bound arguments raise errors."""
    for mal in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_campbell_distress(ticker=mal, latest_price=100.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_campbell_distress(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(ValueError, match="net_income_to_market_assets must be in"):
        calculate_campbell_distress(ticker="AAPL", latest_price=100.0, net_income_to_market_assets=-1.5)

    with pytest.raises(ValueError, match="total_liabilities_to_market_assets must be in"):
        calculate_campbell_distress(ticker="AAPL", latest_price=100.0, total_liabilities_to_market_assets=1.50)

    with pytest.raises(ValueError, match="idiosyncratic_volatility_sigma must be in"):
        calculate_campbell_distress(ticker="AAPL", latest_price=100.0, idiosyncratic_volatility_sigma=2.50)


def test_evaluate_campbell_distress_profile_with_observations(tmp_path: Path) -> None:
    """Verify end-to-end evaluation using synthetic database observations."""
    test_db = tmp_path / "test_chs.db"
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

    receipt = evaluate_campbell_distress_profile(ticker="AAPL", db_path=test_db)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(177.2, abs=1e-2)
    assert receipt.failure_probability_12m_pct > 0.0
    assert receipt.verify_integrity() is True


def test_evaluate_campbell_distress_profile_overrides(tmp_path: Path) -> None:
    """Verify parameter overrides take precedence over inferred defaults."""
    test_db = tmp_path / "test_chs_override.db"
    db.init_db(test_db)

    receipt = evaluate_campbell_distress_profile(
        ticker="AAPL",
        db_path=test_db,
        net_income_to_market_assets=-0.30,
        total_liabilities_to_market_assets=0.88,
        cash_to_market_assets=0.02,
        trailing_excess_return=-0.70,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.net_income_to_market_assets == -0.30
    assert receipt.total_liabilities_to_market_assets == 0.88
    assert receipt.regime == "DISTRESS_ANOMALY_OVERPRICED_TRAP"
    assert receipt.is_distress_anomaly_trap is True


def test_cli_chs_distress_json_output(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI chs-distress command output in JSON mode."""
    test_db = tmp_path / "test_cli_chs.db"
    db.init_db(test_db)

    args = argparse.Namespace(
        ticker="MSFT",
        nimta=0.05,
        tlmta=0.30,
        cashmta=0.20,
        excess_return=0.10,
        sigma=0.20,
        relative_size=-4.0,
        market_to_book=4.5,
        json=True,
    )
    chs_distress_command(args)

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["ticker"] == "MSFT"
    assert data["net_income_to_market_assets"] == 0.05
    assert "signature" in data
    assert len(data["signature"]) == 64


def test_cli_chs_distress_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI chs-distress command standard formatted output."""
    args = argparse.Namespace(
        ticker="GOOGL",
        nimta=0.06,
        tlmta=0.25,
        cashmta=0.22,
        excess_return=0.08,
        sigma=0.22,
        relative_size=-4.2,
        market_to_book=3.8,
        json=False,
    )
    chs_distress_command(args)

    out, _ = capsys.readouterr()
    assert "NextOmni Stock: Campbell-Hilscher-Szilagyi (2008) Distress Puzzle Engine" in out
    assert "Security / Ticker:            GOOGL" in out
    assert "Verified: True" in out


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify parameterized database queries and clean execution without SQL injection."""
    test_db = tmp_path / "test_chs_db.db"
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

