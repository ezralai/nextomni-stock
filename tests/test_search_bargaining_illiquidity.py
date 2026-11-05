"""Tests for Duffie-Gârleanu-Pedersen (2005) OTC Search & Bargaining Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.search_bargaining_illiquidity import (
    SearchBargainingReceipt,
    calculate_search_bargaining,
    classify_search_regime,
    evaluate_search_bargaining_profile,
)
from src.cli import search_bargaining_command
from src.security.validators import SecurityValidationError


def test_calculate_search_bargaining_severe_freeze() -> None:
    """Verify severe OTC search freeze when lambda <= 3.0 and search discount >= 3.5%."""
    receipt = calculate_search_bargaining(
        ticker="DISTR",
        latest_price=45.0,
        search_intensity_lambda=1.5,
        investor_bargaining_power=0.25,
        dealer_inventory_cost=0.005,
        opportunity_cost_rate=0.05,
        asset_cash_flow_rate=0.03,
    )
    assert receipt.ticker == "DISTR"
    assert receipt.search_intensity_lambda <= 3.0
    assert receipt.search_illiquidity_discount_pct >= 3.50
    assert receipt.regime == "SEVERE_OTC_SEARCH_FREEZE"
    assert receipt.is_severe_search_freeze is True
    assert receipt.is_centralized_high_liquidity is False
    assert receipt.walrasian_frictionless_price > receipt.latest_price
    assert receipt.verify_integrity() is True


def test_calculate_search_bargaining_centralized() -> None:
    """Verify high-liquidity centralized matching when lambda >= 20.0 and discount <= 1.0%."""
    receipt = calculate_search_bargaining(
        ticker="AAPL",
        latest_price=220.0,
        search_intensity_lambda=35.0,
        investor_bargaining_power=0.75,
        dealer_inventory_cost=0.001,
        opportunity_cost_rate=0.05,
        asset_cash_flow_rate=0.05,
    )
    assert receipt.search_intensity_lambda >= 20.0
    assert receipt.search_illiquidity_discount_pct <= 1.00
    assert receipt.regime == "HIGH_LIQUIDITY_CENTRALIZED_MATCHING"
    assert receipt.is_centralized_high_liquidity is True
    assert receipt.is_severe_search_freeze is False
    assert receipt.verify_integrity() is True


def test_calculate_search_bargaining_elevated() -> None:
    """Verify elevated search & bargaining friction when discount >= 2.0%."""
    receipt = calculate_search_bargaining(
        ticker="BLOCK",
        latest_price=85.0,
        search_intensity_lambda=8.0,
        investor_bargaining_power=0.45,
        dealer_inventory_cost=0.0025,
        opportunity_cost_rate=0.05,
        asset_cash_flow_rate=0.04,
    )
    assert receipt.search_illiquidity_discount_pct >= 2.00
    assert receipt.regime == "ELEVATED_SEARCH_BARGAINING_FRICTION"
    assert receipt.is_elevated_search_friction is True
    assert receipt.is_severe_search_freeze is False
    assert receipt.verify_integrity() is True


def test_calculate_search_bargaining_neutral() -> None:
    """Verify balanced OTC intermediation equilibrium under baseline values."""
    receipt = calculate_search_bargaining(
        ticker="MSFT",
        latest_price=410.0,
        search_intensity_lambda=15.0,
        investor_bargaining_power=0.60,
        dealer_inventory_cost=0.001,
        opportunity_cost_rate=0.05,
        asset_cash_flow_rate=0.05,
    )
    assert receipt.regime == "BALANCED_OTC_INTERMEDIATION_EQUILIBRIUM"
    assert receipt.is_severe_search_freeze is False
    assert receipt.is_centralized_high_liquidity is False
    assert receipt.verify_integrity() is True


def test_classify_search_regime_direct() -> None:
    """Directly test classify_search_regime boundary logic."""
    assert classify_search_regime(1.5, 6.0, 2.5) == "SEVERE_OTC_SEARCH_FREEZE"
    assert classify_search_regime(30.0, 0.40, 0.20) == "HIGH_LIQUIDITY_CENTRALIZED_MATCHING"
    assert classify_search_regime(10.0, 2.20, 1.40) == "ELEVATED_SEARCH_BARGAINING_FRICTION"
    assert classify_search_regime(15.0, 1.00, 0.50) == "BALANCED_OTC_INTERMEDIATION_EQUILIBRIUM"


def test_tamper_signature_verification() -> None:
    """Verify HMAC-SHA256 signature detects any modified field."""
    receipt = calculate_search_bargaining(
        ticker="NVDA",
        latest_price=120.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with numerical field
    tampered_dict = receipt.to_dict()
    tampered_dict["search_illiquidity_discount_pct"] = 19.9
    tampered_receipt = SearchBargainingReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with signature
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["signature"] = "deadbeef22222222"
    tampered_receipt2 = SearchBargainingReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_security_validation_ticker_and_bounds() -> None:
    """Ensure invalid tickers and out-of-bound arguments raise errors."""
    for mal in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_search_bargaining(ticker=mal, latest_price=100.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_search_bargaining(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(ValueError, match="search_intensity_lambda must be in"):
        calculate_search_bargaining(ticker="AAPL", latest_price=100.0, search_intensity_lambda=90.0)

    with pytest.raises(ValueError, match="investor_bargaining_power must be in"):
        calculate_search_bargaining(ticker="AAPL", latest_price=100.0, investor_bargaining_power=0.99)

    with pytest.raises(ValueError, match="dealer_inventory_cost must be in"):
        calculate_search_bargaining(ticker="AAPL", latest_price=100.0, dealer_inventory_cost=0.50)


def test_evaluate_search_bargaining_profile_with_observations(tmp_path: Path) -> None:
    """Verify end-to-end evaluation using synthetic database observations."""
    test_db = tmp_path / "test_dgp.db"
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

    receipt = evaluate_search_bargaining_profile(ticker="AAPL", db_path=test_db)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(177.2, abs=1e-2)
    assert receipt.search_intensity_lambda > 0.0
    assert receipt.verify_integrity() is True


def test_evaluate_search_bargaining_profile_overrides(tmp_path: Path) -> None:
    """Verify parameter overrides take precedence over inferred defaults."""
    test_db = tmp_path / "test_dgp_override.db"
    db.init_db(test_db)

    receipt = evaluate_search_bargaining_profile(
        ticker="AAPL",
        db_path=test_db,
        search_intensity_lambda=1.8,
        dealer_inventory_cost=0.005,
        asset_cash_flow_rate=0.02,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.search_intensity_lambda == 1.8
    assert receipt.dealer_inventory_cost == 0.005
    assert receipt.regime == "SEVERE_OTC_SEARCH_FREEZE"
    assert receipt.is_severe_search_freeze is True


def test_cli_search_bargaining_json_output(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI search-bargaining command output in JSON mode."""
    test_db = tmp_path / "test_cli_dgp.db"
    db.init_db(test_db)

    args = argparse.Namespace(
        ticker="MSFT",
        search_intensity=14.0,
        bargaining_power=0.60,
        inventory_cost=0.002,
        discount_rate=0.05,
        cash_flow_rate=0.04,
        json=True,
    )
    search_bargaining_command(args)

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["ticker"] == "MSFT"
    assert data["search_intensity_lambda"] == 14.0
    assert "signature" in data
    assert len(data["signature"]) == 64


def test_cli_search_bargaining_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI search-bargaining command standard formatted output."""
    args = argparse.Namespace(
        ticker="GOOGL",
        search_intensity=20.0,
        bargaining_power=0.55,
        inventory_cost=0.001,
        discount_rate=0.05,
        cash_flow_rate=0.04,
        json=False,
    )
    search_bargaining_command(args)

    out, _ = capsys.readouterr()
    assert "NextOmni Stock: Duffie-Gârleanu-Pedersen (2005) OTC Search & Bargaining Engine" in out
    assert "Security / Ticker:            GOOGL" in out
    assert "Verified: True" in out


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify parameterized database queries and clean execution without SQL injection."""
    test_db = tmp_path / "test_dgp_db.db"
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
