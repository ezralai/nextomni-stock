"""Tests for Nicolae Gârleanu & Lasse Heje Pedersen (2011) Margin-Based Asset Pricing Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.margin_asset_pricing import (
    MarginAssetPricingReceipt,
    calculate_margin_asset_pricing,
    classify_margin_pricing_regime,
    evaluate_margin_pricing_profile,
)
from src.cli import margin_pricing_command
from src.security.validators import SecurityValidationError


def test_calculate_margin_asset_pricing_basis_discount() -> None:
    """Verify margin-constrained basis discount regime when psi >= 0.08 and m_i <= 0.35."""
    receipt = calculate_margin_asset_pricing(
        ticker="SPY",
        latest_price=510.0,
        margin_requirement=0.15,
        baseline_margin_requirement=0.15,
        margin_shadow_cost_psi=0.10,
        market_beta=1.00,
        market_risk_premium_pct=6.5,
        risk_free_rate_pct=4.0,
    )
    assert receipt.ticker == "SPY"
    assert receipt.margin_requirement <= 0.35
    assert receipt.margin_shadow_cost_psi >= 0.08
    assert receipt.basis_spread_discount_pct >= 2.0
    assert receipt.regime == "MARGIN_CONSTRAINED_BASIS_DISCOUNT"
    assert receipt.is_margin_constrained_discount is True
    assert receipt.is_margin_overpricing_trap is False
    assert receipt.expected_margin_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_margin_asset_pricing_high_beta_trap() -> None:
    """Verify high-beta margin overpricing trap when beta >= 1.40 and m_i >= 0.40."""
    receipt = calculate_margin_asset_pricing(
        ticker="MEME",
        latest_price=45.0,
        margin_requirement=0.60,
        baseline_margin_requirement=0.25,
        margin_shadow_cost_psi=0.08,
        market_beta=1.85,
        market_risk_premium_pct=6.5,
        risk_free_rate_pct=4.0,
    )
    assert receipt.market_beta >= 1.40
    assert receipt.margin_requirement >= 0.40
    assert receipt.regime == "HIGH_BETA_MARGIN_OVERPRICING_TRAP"
    assert receipt.is_margin_overpricing_trap is True
    assert receipt.is_margin_constrained_discount is False
    assert receipt.expected_margin_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_margin_asset_pricing_bab_opportunity() -> None:
    """Verify betting against beta alpha opportunity when beta <= 0.75 and psi >= 0.05."""
    receipt = calculate_margin_asset_pricing(
        ticker="KO",
        latest_price=62.0,
        margin_requirement=0.20,
        baseline_margin_requirement=0.20,
        margin_shadow_cost_psi=0.07,
        market_beta=0.60,
        market_risk_premium_pct=6.5,
        risk_free_rate_pct=4.0,
    )
    assert receipt.market_beta <= 0.75
    assert receipt.margin_shadow_cost_psi >= 0.05
    assert receipt.regime == "BETTING_AGAINST_BETA_ALPHA_OPPORTUNITY"
    assert receipt.is_bab_alpha_opportunity is True
    assert receipt.bab_leverage_compression_score > 0.0
    assert receipt.expected_margin_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_margin_asset_pricing_neutral() -> None:
    """Verify margin equilibrium neutral when values are within balanced boundaries."""
    receipt = calculate_margin_asset_pricing(
        ticker="AAPL",
        latest_price=220.0,
        margin_requirement=0.20,
        baseline_margin_requirement=0.20,
        margin_shadow_cost_psi=0.02,
        market_beta=1.05,
        market_risk_premium_pct=6.0,
        risk_free_rate_pct=4.0,
    )
    assert receipt.regime == "MARGIN_EQUILIBRIUM_NEUTRAL"
    assert receipt.is_margin_constrained_discount is False
    assert receipt.is_margin_overpricing_trap is False
    assert receipt.is_bab_alpha_opportunity is False
    assert receipt.verify_integrity() is True


def test_classify_margin_pricing_regime_direct() -> None:
    """Directly test classify_margin_pricing_regime boundary logic."""
    assert classify_margin_pricing_regime(0.20, 0.10, 1.00, 3.5) == "MARGIN_CONSTRAINED_BASIS_DISCOUNT"
    assert classify_margin_pricing_regime(0.50, 0.08, 1.60, 1.2) == "HIGH_BETA_MARGIN_OVERPRICING_TRAP"
    assert classify_margin_pricing_regime(0.25, 0.07, 0.65, 1.5) == "BETTING_AGAINST_BETA_ALPHA_OPPORTUNITY"
    assert classify_margin_pricing_regime(0.20, 0.02, 1.05, 1.0) == "MARGIN_EQUILIBRIUM_NEUTRAL"


def test_tamper_signature_verification() -> None:
    """Verify HMAC-SHA256 signature detects any modified field."""
    receipt = calculate_margin_asset_pricing(
        ticker="MSFT",
        latest_price=420.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with numerical field
    tampered_dict = receipt.to_dict()
    tampered_dict["margin_requirement"] = 0.85
    tampered_receipt = MarginAssetPricingReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with signature
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["signature"] = "badc0ffee00000000"
    tampered_receipt2 = MarginAssetPricingReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_security_validation_ticker_and_bounds() -> None:
    """Ensure invalid tickers and out-of-bound arguments raise errors."""
    for mal in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_margin_asset_pricing(ticker=mal, latest_price=100.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_margin_asset_pricing(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(ValueError, match="margin_requirement must be in"):
        calculate_margin_asset_pricing(ticker="AAPL", latest_price=100.0, margin_requirement=0.99)

    with pytest.raises(ValueError, match="margin_shadow_cost_psi must be in"):
        calculate_margin_asset_pricing(ticker="AAPL", latest_price=100.0, margin_shadow_cost_psi=0.80)


def test_evaluate_margin_pricing_profile_with_observations(tmp_path: Path) -> None:
    """Verify end-to-end evaluation using synthetic database observations."""
    test_db = tmp_path / "test_map.db"
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

    receipt = evaluate_margin_pricing_profile(ticker="AAPL", db_path=test_db)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(177.2, abs=1e-2)
    assert receipt.margin_requirement >= 0.05
    assert receipt.verify_integrity() is True


def test_evaluate_margin_pricing_profile_overrides(tmp_path: Path) -> None:
    """Verify parameter overrides take precedence over inferred defaults."""
    test_db = tmp_path / "test_map_override.db"
    db.init_db(test_db)

    receipt = evaluate_margin_pricing_profile(
        ticker="AAPL",
        db_path=test_db,
        margin_requirement=0.12,
        baseline_margin_requirement=0.15,
        margin_shadow_cost_psi=0.12,
        market_beta=0.95,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.margin_requirement == 0.12
    assert receipt.margin_shadow_cost_psi == 0.12
    assert receipt.regime == "MARGIN_CONSTRAINED_BASIS_DISCOUNT"
    assert receipt.is_margin_constrained_discount is True


def test_cli_margin_pricing_json_output(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI margin-pricing command output in JSON mode."""
    test_db = tmp_path / "test_cli_map.db"
    db.init_db(test_db)

    args = argparse.Namespace(
        ticker="MSFT",
        margin_requirement=0.20,
        baseline_margin=0.20,
        shadow_cost=0.08,
        market_beta=1.10,
        market_risk_premium=6.5,
        risk_free_rate=4.0,
        json=True,
    )
    margin_pricing_command(args)

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["ticker"] == "MSFT"
    assert data["margin_requirement"] == 0.20
    assert "signature" in data
    assert len(data["signature"]) == 64


def test_cli_margin_pricing_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI margin-pricing command standard formatted output."""
    args = argparse.Namespace(
        ticker="GOOGL",
        margin_requirement=0.22,
        baseline_margin=0.20,
        shadow_cost=0.05,
        market_beta=1.15,
        market_risk_premium=6.0,
        risk_free_rate=4.0,
        json=False,
    )
    margin_pricing_command(args)

    out, _ = capsys.readouterr()
    assert "NextOmni Stock: Gârleanu & Pedersen (2011) Margin-Based Asset Pricing Engine" in out
    assert "Security / Ticker:            GOOGL" in out
    assert "Verified: True" in out


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify parameterized database queries and clean execution without SQL injection."""
    test_db = tmp_path / "test_map_db.db"
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

