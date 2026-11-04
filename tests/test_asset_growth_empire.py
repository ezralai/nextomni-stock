"""Unit and integration tests for Cooper et al. (2008) & Titman et al. (2004) Asset Growth & Empire-Building Engine."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.asset_growth_empire import (
    AssetGrowthEmpireReceipt,
    calculate_asset_growth_empire,
    classify_empire_regime,
    evaluate_asset_growth_profile,
)
from src.cli import empire_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_asset_growth_disciplined_allocator() -> None:
    """Test disciplined capital shrinker / restrained allocator delivering positive alpha."""
    receipt = calculate_asset_growth_empire(
        ticker="DISC",
        latest_price=125.0,
        current_total_assets=9_500.0,
        prior_total_assets=10_000.0,  # -5% asset shrinkage (buybacks / divestitures)
        current_capex_to_sales_pct=4.5,
        historical_3yr_capex_to_sales_pct=6.0,  # ACI = -0.25
        fcf_to_assets_pct=8.0,
        debt_to_assets_pct=35.0,
    )

    assert receipt.ticker == "DISC"
    assert abs(receipt.asset_growth_pct - (-5.0)) < 1e-4
    assert receipt.abnormal_capital_investment < 0.0
    assert receipt.empire_building_score <= -0.40
    assert receipt.is_disciplined_allocator is True
    assert receipt.is_empire_builder_trap is False
    assert receipt.regime == "DISCIPLINED_CAPITAL_ALLOCATOR"
    assert receipt.expected_allocation_alpha_pct > 2.0
    assert receipt.verify_integrity() is True


def test_calculate_asset_growth_sustainable_organic() -> None:
    """Test balanced organic reinvestment aligned with historical baseline."""
    receipt = calculate_asset_growth_empire(
        ticker="ORGN",
        latest_price=100.0,
        current_total_assets=11_000.0,
        prior_total_assets=10_000.0,  # +10% asset growth
        current_capex_to_sales_pct=6.2,
        historical_3yr_capex_to_sales_pct=6.0,
        fcf_to_assets_pct=5.0,
        debt_to_assets_pct=30.0,
    )

    assert receipt.ticker == "ORGN"
    assert -0.40 < receipt.empire_building_score < 0.65
    assert receipt.regime == "SUSTAINABLE_ORGANIC_GROWTH"
    assert receipt.is_disciplined_allocator is False
    assert receipt.is_empire_builder_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_asset_growth_elevated_watch() -> None:
    """Test elevated expansion requiring ROIC monitoring (0.65 <= EBOS < 1.25)."""
    receipt = calculate_asset_growth_empire(
        ticker="WTCH",
        latest_price=90.0,
        current_total_assets=12_200.0,
        prior_total_assets=10_000.0,  # +22% asset growth
        current_capex_to_sales_pct=7.5,
        historical_3yr_capex_to_sales_pct=6.0,  # ACI = +0.25
        fcf_to_assets_pct=6.0,
        debt_to_assets_pct=25.0,
    )

    assert 0.65 <= receipt.empire_building_score < 1.25
    assert receipt.regime == "ELEVATED_EXPANSION_WATCH"
    assert receipt.verify_integrity() is True


def test_calculate_asset_growth_empire_building_trap() -> None:
    """Test aggressive asset bloat + abnormal capex surge predicting severe negative return drag."""
    receipt = calculate_asset_growth_empire(
        ticker="BLOT",
        latest_price=60.0,
        current_total_assets=15_000.0,
        prior_total_assets=10_000.0,  # +50% asset bloat
        current_capex_to_sales_pct=12.0,
        historical_3yr_capex_to_sales_pct=6.0,  # ACI = +1.00 (100% surge)
        fcf_to_assets_pct=12.0,
        debt_to_assets_pct=10.0,
    )

    assert receipt.asset_growth_pct == 50.0
    assert abs(receipt.abnormal_capital_investment - 1.0) < 1e-4
    assert receipt.empire_building_score >= 1.25
    assert receipt.is_empire_builder_trap is True
    assert receipt.is_disciplined_allocator is False
    assert receipt.regime == "EMPIRE_BUILDING_OVERINVESTOR"
    assert receipt.expected_allocation_alpha_pct <= -6.0
    assert receipt.verify_integrity() is True


def test_jensen_agency_discretion_amplification() -> None:
    """Verify Jensen (1986) Free Cash Flow & Low Debt agency index amplifies overinvestment score."""
    # Low agency risk: low FCF (2%), high debt discipline (60%)
    rec_low_agency = calculate_asset_growth_empire(
        ticker="AGCY",
        latest_price=100.0,
        current_total_assets=13_000.0,
        prior_total_assets=10_000.0,
        current_capex_to_sales_pct=9.0,
        historical_3yr_capex_to_sales_pct=6.0,
        fcf_to_assets_pct=2.0,
        debt_to_assets_pct=60.0,
    )

    # High agency risk: high FCF (18%), low debt (5%)
    rec_high_agency = calculate_asset_growth_empire(
        ticker="AGCY",
        latest_price=100.0,
        current_total_assets=13_000.0,
        prior_total_assets=10_000.0,
        current_capex_to_sales_pct=9.0,
        historical_3yr_capex_to_sales_pct=6.0,
        fcf_to_assets_pct=18.0,
        debt_to_assets_pct=5.0,
    )

    assert rec_high_agency.jensen_agency_discretion_index > rec_low_agency.jensen_agency_discretion_index
    assert rec_high_agency.empire_building_score > rec_low_agency.empire_building_score


def test_titman_wei_xie_aci_math() -> None:
    """Verify exact Titman, Wei & Xie (2004) Abnormal Capital Investment formula."""
    rec = calculate_asset_growth_empire(
        ticker="ACIM",
        latest_price=100.0,
        current_total_assets=10_500.0,
        prior_total_assets=10_000.0,
        current_capex_to_sales_pct=9.0,
        historical_3yr_capex_to_sales_pct=6.0,
    )
    # (9.0 / 6.0) - 1.0 = 0.50
    assert abs(rec.abnormal_capital_investment - 0.50) < 1e-4


def test_classify_empire_regime_boundaries() -> None:
    """Test regime classification boundary thresholds."""
    assert classify_empire_regime(1.50, 35.0, 0.60) == "EMPIRE_BUILDING_OVERINVESTOR"
    assert classify_empire_regime(0.85, 20.0, 0.25) == "ELEVATED_EXPANSION_WATCH"
    assert classify_empire_regime(-0.60, -2.0, -0.20) == "DISCIPLINED_CAPITAL_ALLOCATOR"
    assert classify_empire_regime(0.10, 8.0, 0.05) == "SUSTAINABLE_ORGANIC_GROWTH"


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 signature verification and tamper detection."""
    receipt = calculate_asset_growth_empire("HMAC", 100.0, 10_500.0, 10_000.0)
    assert receipt.verify_integrity() is True

    receipt.asset_growth_pct = 99.9
    assert receipt.verify_integrity() is False


def test_evaluate_asset_growth_profile_database(tmp_path: Path) -> None:
    """Test end-to-end evaluation using seeded SQLite database observations."""
    test_db = tmp_path / "test_stock.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="EMPR",
        exchange="XNYS",
        company_name="Empire Test Corp.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(25):
        p = 110.0 + i * 0.5
        records.append(
            {
                "ticker": "EMPR",
                "exchange": "XNYS",
                "session_date": f"2026-01-{i+1:02d}",
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "adj_close": p,
                "volume": 400_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    receipt = evaluate_asset_growth_profile(
        ticker="EMPR",
        db_path=test_db,
        current_total_assets=9_800.0,
        prior_total_assets=10_000.0,
    )

    assert receipt.ticker == "EMPR"
    assert abs(receipt.latest_price - (110.0 + 24 * 0.5)) < 1e-2
    assert receipt.asset_growth_pct == -2.0
    assert receipt.verify_integrity() is True


def test_evaluate_asset_growth_profile_fallback() -> None:
    """Test deterministic fallback when ticker is not in database."""
    receipt = evaluate_asset_growth_profile(
        ticker="XYZ",
        current_total_assets=12_000.0,
        prior_total_assets=10_000.0,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price == 100.0
    assert receipt.asset_growth_pct == 20.0
    assert receipt.verify_integrity() is True


def test_cli_empire_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI empire command output formatting and JSON mode."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        current_assets=9_600.0,
        prior_assets=10_000.0,
        current_capex=5.0,
        historical_capex=6.0,
        fcf_to_assets=8.0,
        debt_to_assets=30.0,
        json=False,
    )
    empire_command(args_text)
    captured = capsys.readouterr()
    assert "Cooper et al. (2008) & Titman et al. (2004) Asset Growth Engine" in captured.out
    assert "AAPL" in captured.out
    assert "YoY Total Asset Growth (AG)" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        current_assets=9_600.0,
        prior_assets=10_000.0,
        current_capex=5.0,
        historical_capex=6.0,
        fcf_to_assets=8.0,
        debt_to_assets=30.0,
        json=True,
    )
    empire_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "AAPL"
    assert data["asset_growth_pct"] == -4.0
    assert "signature" in data


def test_input_validation_empire() -> None:
    """Test strict security input validation on ticker, assets, capex, and finite numbers."""
    with pytest.raises(SecurityValidationError):
        calculate_asset_growth_empire("BAD$TICKER", 100.0, 10_000.0, 10_000.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_asset_growth_empire("AAPL", 0.0, 10_000.0, 10_000.0)

    with pytest.raises(ValueError, match="Total assets must be strictly positive"):
        calculate_asset_growth_empire("AAPL", 100.0, -500.0, 10_000.0)

    with pytest.raises(ValueError, match="CapEx-to-Sales ratios must be non-negative"):
        calculate_asset_growth_empire("AAPL", 100.0, 10_000.0, 10_000.0, current_capex_to_sales_pct=-2.0)

    with pytest.raises(ValueError, match="must be finite"):
        calculate_asset_growth_empire("AAPL", 100.0, float("nan"), 10_000.0)
