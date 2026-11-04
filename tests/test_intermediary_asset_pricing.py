"""Tests for Zhiguo He, Bryan Kelly & Asaf Manela (2017) Intermediary Asset Pricing Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.intermediary_asset_pricing import (
    IntermediaryAssetPricingReceipt,
    calculate_intermediary_pricing,
    classify_intermediary_regime,
    evaluate_intermediary_pricing_profile,
)
from src.cli import intermediary_command
from src.security.validators import SecurityValidationError


def test_calculate_intermediary_pricing_high_risk_leader() -> None:
    """Verify high intermediary risk capital premium leader regime when beta >= 1.35 and capacity <= 0.85."""
    receipt = calculate_intermediary_pricing(
        ticker="BAC",
        latest_price=35.0,
        dealer_capital_ratio=0.07,
        baseline_capital_ratio=0.10,
        dealer_leverage_ratio=20.0,
        baseline_leverage_ratio=15.0,
        intermediary_capital_beta=1.60,
    )
    assert receipt.ticker == "BAC"
    assert receipt.balance_sheet_capacity_index <= 0.85
    assert receipt.intermediary_capital_beta >= 1.35
    assert receipt.regime == "HIGH_INTERMEDIARY_RISK_CAPITAL_PREMIUM"
    assert receipt.is_high_intermediary_risk_leader is True
    assert receipt.is_intermediary_distress_safe_haven is False
    assert receipt.intermediary_risk_premium_pct > 0.0
    assert receipt.expected_intermediary_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_intermediary_pricing_safe_haven() -> None:
    """Verify intermediary distress safe haven regime when beta <= -0.20."""
    receipt = calculate_intermediary_pricing(
        ticker="GLD",
        latest_price=180.0,
        dealer_capital_ratio=0.06,
        baseline_capital_ratio=0.10,
        dealer_leverage_ratio=22.0,
        baseline_leverage_ratio=15.0,
        intermediary_capital_beta=-0.45,
    )
    assert receipt.intermediary_capital_beta <= -0.20
    assert receipt.regime == "INTERMEDIARY_DISTRESS_SAFE_HAVEN"
    assert receipt.is_intermediary_distress_safe_haven is True
    assert receipt.is_high_intermediary_risk_leader is False
    assert receipt.expected_intermediary_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_intermediary_pricing_leverage_constrained() -> None:
    """Verify leverage-constrained dealer overhang regime when leverage >= 25.0 and capacity < 0.60."""
    receipt = calculate_intermediary_pricing(
        ticker="LEHM",
        latest_price=20.0,
        dealer_capital_ratio=0.03,
        baseline_capital_ratio=0.10,
        dealer_leverage_ratio=32.0,
        baseline_leverage_ratio=15.0,
        intermediary_capital_beta=0.80,
    )
    assert receipt.dealer_leverage_ratio >= 25.0
    assert receipt.balance_sheet_capacity_index < 0.60
    assert receipt.regime == "LEVERAGE_CONSTRAINED_DEALER_OVERHANG"
    assert receipt.is_high_intermediary_risk_leader is False
    assert receipt.is_intermediary_distress_safe_haven is False
    assert receipt.intermediary_pricing_impulse < 0.0
    assert receipt.expected_intermediary_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_intermediary_pricing_neutral() -> None:
    """Verify unconstrained dealer equilibrium when values reflect healthy capitalization."""
    receipt = calculate_intermediary_pricing(
        ticker="AAPL",
        latest_price=225.0,
        dealer_capital_ratio=0.10,
        baseline_capital_ratio=0.10,
        dealer_leverage_ratio=15.0,
        baseline_leverage_ratio=15.0,
        intermediary_capital_beta=0.90,
    )
    assert receipt.balance_sheet_capacity_index == pytest.approx(1.0, rel=1e-2)
    assert receipt.regime == "UNCONSTRAINED_DEALER_EQUILIBRIUM"
    assert receipt.is_high_intermediary_risk_leader is False
    assert receipt.is_intermediary_distress_safe_haven is False
    assert receipt.verify_integrity() is True


def test_classify_intermediary_regime_direct() -> None:
    """Directly test classify_intermediary_regime boundary logic."""
    assert classify_intermediary_regime(1.45, 0.75, 18.0, 0.50) == "HIGH_INTERMEDIARY_RISK_CAPITAL_PREMIUM"
    assert classify_intermediary_regime(-0.35, 0.70, 20.0, 0.40) == "INTERMEDIARY_DISTRESS_SAFE_HAVEN"
    assert classify_intermediary_regime(0.60, 0.50, 28.0, -1.30) == "LEVERAGE_CONSTRAINED_DEALER_OVERHANG"
    assert classify_intermediary_regime(0.90, 1.10, 15.0, 0.20) == "UNCONSTRAINED_DEALER_EQUILIBRIUM"


def test_tamper_signature_verification() -> None:
    """Verify HMAC signature detects any metadata modification."""
    receipt = calculate_intermediary_pricing(
        ticker="MSFT",
        latest_price=410.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with numerical field
    tampered_dict = receipt.to_dict()
    tampered_dict["dealer_capital_ratio"] = 0.35
    tampered_receipt = IntermediaryAssetPricingReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with signature
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["signature"] = "deadbeef12345678"
    tampered_receipt2 = IntermediaryAssetPricingReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_security_validation_ticker_and_bounds() -> None:
    """Ensure invalid ticker and out-of-range inputs raise errors."""
    for mal in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_intermediary_pricing(ticker=mal, latest_price=100.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_intermediary_pricing(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(ValueError, match="dealer_capital_ratio must be in"):
        calculate_intermediary_pricing(ticker="AAPL", latest_price=100.0, dealer_capital_ratio=0.99)

    with pytest.raises(ValueError, match="dealer_leverage_ratio must be in"):
        calculate_intermediary_pricing(ticker="AAPL", latest_price=100.0, dealer_leverage_ratio=100.0)


def test_evaluate_intermediary_pricing_profile_with_observations(tmp_path: Path) -> None:
    """Verify end-to-end evaluation using synthetic database observations."""
    test_db = tmp_path / "test_iap.db"
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
        p = 100.0 + i * 0.5
        records.append({
            "ticker": "AAPL",
            "exchange": "XNAS",
            "session_date": f"2026-01-{i+1:02d}",
            "open": p,
            "high": p + 1.0,
            "low": p - 1.0,
            "close": p,
            "adj_close": p,
            "volume": 1_000_000,
            "data_status": "END_OF_DAY",
            "provider": "MOCK",
        })
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    receipt = evaluate_intermediary_pricing_profile(ticker="AAPL", db_path=test_db)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(117.0, abs=1e-2)
    assert receipt.verify_integrity() is True


def test_evaluate_intermediary_pricing_profile_overrides(tmp_path: Path) -> None:
    """Verify parameter overrides take precedence over inferred defaults."""
    test_db = tmp_path / "test_iap_override.db"
    db.init_db(test_db)

    receipt = evaluate_intermediary_pricing_profile(
        ticker="AAPL",
        db_path=test_db,
        dealer_capital_ratio=0.05,
        baseline_capital_ratio=0.10,
        dealer_leverage_ratio=28.0,
        intermediary_capital_beta=-0.35,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.dealer_capital_ratio == 0.05
    assert receipt.intermediary_capital_beta == -0.35
    assert receipt.regime == "INTERMEDIARY_DISTRESS_SAFE_HAVEN"
    assert receipt.is_intermediary_distress_safe_haven is True


def test_cli_intermediary_json_output(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI intermediary command output in JSON mode."""
    test_db = tmp_path / "test_cli_iap.db"
    db.init_db(test_db)

    args = argparse.Namespace(
        ticker="MSFT",
        dealer_capital_ratio=0.12,
        baseline_capital_ratio=0.10,
        dealer_leverage=16.0,
        baseline_leverage=15.0,
        capital_beta=1.45,
        json=True,
    )
    intermediary_command(args)

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["ticker"] == "MSFT"
    assert data["dealer_capital_ratio"] == 0.12
    assert "signature" in data
    assert len(data["signature"]) == 64


def test_cli_intermediary_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI intermediary command standard formatted output."""
    args = argparse.Namespace(
        ticker="GOOGL",
        dealer_capital_ratio=0.08,
        baseline_capital_ratio=0.10,
        dealer_leverage=18.0,
        baseline_leverage=15.0,
        capital_beta=1.40,
        json=False,
    )
    intermediary_command(args)

    out, _ = capsys.readouterr()
    assert "NextOmni Stock: He-Kelly-Manela (2017) Intermediary Pricing Engine" in out
    assert "Security / Ticker:            GOOGL" in out
    assert "Verified: True" in out


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify parameterized database queries and clean execution without SQL injection."""
    test_db = tmp_path / "test_iap_db.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="TSLA",
        exchange="XNAS",
        company_name="Tesla Inc.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(25):
        records.append({
            "ticker": "TSLA",
            "exchange": "XNAS",
            "session_date": f"2026-01-{i+1:02d}",
            "open": 200.0,
            "high": 205.0,
            "low": 195.0,
            "close": 202.0,
            "adj_close": 202.0,
            "volume": 500_000,
            "data_status": "END_OF_DAY",
            "provider": "MOCK",
        })
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    obs = db.get_observations("TSLA", db_path=test_db)
    assert len(obs) == 25
