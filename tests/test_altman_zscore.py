"""Unit and integration tests for Edward I. Altman (1968, 2000) Z-Score and James A. Ohlson (1980) O-Score Engine."""

import argparse
import json
import math
from pathlib import Path
from typing import List

import pytest

from src.analytics.altman_zscore import (
    AltmanZScoreReceipt,
    calculate_altman_zscore,
    classify_solvency_regime,
    evaluate_solvency_profile,
)
from src.cli import altman_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_altman_zscore_safe_fortress() -> None:
    """Test pristine balance sheet in Safe Zone with high solvency alpha."""
    # Assets: 10,000M, Liab: 3,000M, WC: 2,500M, RE: 4,000M, EBIT: 1,800M, Sales: 11,000M, Mcap: 35,000M
    receipt = calculate_altman_zscore(
        ticker="SAFE",
        latest_price=150.0,
        total_assets=10_000.0,
        total_liabilities=3_000.0,
        working_capital=2_500.0,
        retained_earnings=4_000.0,
        ebit=1_800.0,
        sales=11_000.0,
        market_cap=35_000.0,
        current_assets=4_000.0,
        current_liabilities=1_500.0,
        net_income=1_400.0,
        funds_from_operations=1_900.0,
        prior_net_income=1_200.0,
    )

    assert receipt.ticker == "SAFE"
    assert receipt.altman_z_score > 2.99
    assert receipt.altman_z_double_prime > 2.60
    assert receipt.is_safe_zone is True
    assert receipt.is_grey_zone is False
    assert receipt.is_distress_zone is False
    assert receipt.is_negative_equity is False
    assert receipt.consecutive_losses is False
    assert receipt.default_probability_pct < 10.0
    assert receipt.regime == "SOLVENT_FORTRESS_SAFE"
    assert receipt.expected_solvency_alpha_pct >= 2.5
    assert receipt.verify_integrity() is True


def test_calculate_altman_zscore_grey_zone() -> None:
    """Test moderate balance sheet falling within Altman Grey Zone (1.81 <= Z <= 2.99)."""
    # Assets: 10,000M, Liab: 5,000M, WC: 1,500M, RE: 2,000M, EBIT: 800M, Sales: 7,500M, Mcap: 8,000M
    receipt = calculate_altman_zscore(
        ticker="GREY",
        latest_price=50.0,
        total_assets=10_000.0,
        total_liabilities=5_000.0,
        working_capital=1_500.0,
        retained_earnings=2_000.0,
        ebit=800.0,
        sales=7_500.0,
        market_cap=8_000.0,
        current_assets=3_000.0,
        current_liabilities=1_500.0,
        net_income=500.0,
        funds_from_operations=750.0,
    )

    assert receipt.ticker == "GREY"
    assert 1.81 <= receipt.altman_z_score <= 2.99
    assert receipt.is_safe_zone is False
    assert receipt.is_grey_zone is True
    assert receipt.is_distress_zone is False
    assert receipt.verify_integrity() is True


def test_calculate_altman_zscore_distress_zone() -> None:
    """Test distressed balance sheet with heavy debt, negative earnings, and acute bankruptcy hazard."""
    # Assets: 10,000M, Liab: 9,000M, WC: -500M, RE: -2,000M, EBIT: -600M, Sales: 4,000M, Mcap: 1,000M
    receipt = calculate_altman_zscore(
        ticker="DIST",
        latest_price=5.0,
        total_assets=10_000.0,
        total_liabilities=9_000.0,
        working_capital=-500.0,
        retained_earnings=-2_000.0,
        ebit=-600.0,
        sales=4_000.0,
        market_cap=1_000.0,
        current_assets=1_000.0,
        current_liabilities=1_500.0,
        net_income=-800.0,
        funds_from_operations=-300.0,
        prior_net_income=-400.0,
    )

    assert receipt.ticker == "DIST"
    assert receipt.altman_z_score < 1.81
    assert receipt.altman_z_double_prime < 1.10
    assert receipt.is_distress_zone is True
    assert receipt.consecutive_losses is True
    assert receipt.default_probability_pct > 25.0
    assert receipt.regime == "ACUTE_CREDIT_DISTRESS"
    assert receipt.expected_solvency_alpha_pct <= -5.0
    assert receipt.verify_integrity() is True


def test_calculate_altman_zscore_negative_equity() -> None:
    """Test insolvent firm where Total Liabilities exceed Total Assets."""
    receipt = calculate_altman_zscore(
        ticker="INSL",
        latest_price=2.0,
        total_assets=5_000.0,
        total_liabilities=7_000.0,  # Negative equity: TL > TA
        working_capital=-1_000.0,
        retained_earnings=-3_000.0,
        ebit=-400.0,
        sales=3_000.0,
        market_cap=500.0,
        current_assets=1_000.0,
        current_liabilities=2_000.0,
        net_income=-500.0,
        funds_from_operations=-100.0,
    )

    assert receipt.is_negative_equity is True
    assert receipt.regime == "ACUTE_CREDIT_DISTRESS"
    assert receipt.verify_integrity() is True


def test_non_manufacturing_z_double_prime() -> None:
    """Test 4-variable non-manufacturing Z''-Score without asset turnover bias."""
    receipt = calculate_altman_zscore(
        ticker="SERV",
        latest_price=80.0,
        total_assets=8_000.0,
        total_liabilities=2_000.0,
        working_capital=2_000.0,
        retained_earnings=3_000.0,
        ebit=1_000.0,
        sales=1_000.0,  # Low asset turnover due to software/service model
        market_cap=20_000.0,
        current_assets=3_000.0,
        current_liabilities=1_000.0,
        net_income=800.0,
        funds_from_operations=1_100.0,
    )

    # In services, Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    assert receipt.altman_z_double_prime > 5.0
    assert receipt.altman_z_double_prime > 2.60  # Safe for non-mfg
    assert receipt.verify_integrity() is True


def test_ohlson_default_probability_bounds() -> None:
    """Verify Ohlson conditional logit probability is strictly bounded in [0, 100]%."""
    for assets, liab, ebit in [
        (10_000.0, 1_000.0, 2_000.0),
        (5_000.0, 4_500.0, 100.0),
        (2_000.0, 5_000.0, -800.0),
    ]:
        receipt = calculate_altman_zscore(
            ticker="BNDS",
            latest_price=50.0,
            total_assets=assets,
            total_liabilities=liab,
            working_capital=assets * 0.2,
            retained_earnings=assets * 0.3,
            ebit=ebit,
            sales=assets * 0.8,
            market_cap=10_000.0,
            current_assets=assets * 0.4,
            current_liabilities=assets * 0.2,
            net_income=ebit * 0.7,
            funds_from_operations=ebit * 0.9,
        )
        assert 0.0 <= receipt.default_probability_pct <= 100.0


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 signature verification and tamper detection."""
    receipt = calculate_altman_zscore(
        ticker="HMAC",
        latest_price=100.0,
        total_assets=10_000.0,
        total_liabilities=4_000.0,
        working_capital=2_000.0,
        retained_earnings=2_500.0,
        ebit=1_000.0,
        sales=8_000.0,
        market_cap=15_000.0,
        current_assets=3_500.0,
        current_liabilities=1_500.0,
        net_income=750.0,
        funds_from_operations=1_100.0,
    )

    assert receipt.verify_integrity() is True

    # Tampering with score must fail verification
    receipt.altman_z_score = 9.99
    assert receipt.verify_integrity() is False


def test_classify_solvency_regime() -> None:
    """Test boundary conditions for all solvency regimes."""
    assert classify_solvency_regime(3.5, 2.0, False) == "SOLVENT_FORTRESS_SAFE"
    assert classify_solvency_regime(2.5, 8.0, False) == "STABLE_MODERATE_SOLVENCY"
    assert classify_solvency_regime(2.0, 20.0, False) == "VULNERABLE_GREY_ZONE"
    assert classify_solvency_regime(1.5, 40.0, False) == "ACUTE_CREDIT_DISTRESS"
    assert classify_solvency_regime(3.5, 2.0, True) == "ACUTE_CREDIT_DISTRESS"


def test_evaluate_solvency_profile_database(tmp_path: Path) -> None:
    """Test end-to-end evaluation using seeded SQLite database observations."""
    test_db = tmp_path / "test_stock.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="SOLV",
        exchange="XNYS",
        company_name="Solvency Test Inc.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    # Seed 30 synthetic price observations
    base_price = 120.0
    records = []
    for i in range(30):
        records.append(
            {
                "ticker": "SOLV",
                "exchange": "XNYS",
                "session_date": f"2026-01-{i+1:02d}",
                "open": base_price + i * 0.5,
                "high": base_price + i * 0.5 + 1.0,
                "low": base_price + i * 0.5 - 1.0,
                "close": base_price + i * 0.5,
                "adj_close": base_price + i * 0.5,
                "volume": 500_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    import pandas as pd
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    receipt = evaluate_solvency_profile(
        ticker="SOLV",
        db_path=test_db,
    )

    assert receipt.ticker == "SOLV"
    assert abs(receipt.latest_price - (base_price + 29 * 0.5)) < 1e-4
    assert receipt.altman_z_score > 0
    assert receipt.verify_integrity() is True


def test_evaluate_solvency_profile_fallback() -> None:
    """Test deterministic fallback when ticker is not in database."""
    receipt = evaluate_solvency_profile(
        ticker="XYZ",
        total_assets=15_000.0,
        total_liabilities=5_000.0,
        ebit=2_000.0,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert receipt.altman_z_score > 0
    assert receipt.verify_integrity() is True


def test_cli_altman_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI altman command output formatting and JSON mode."""
    # Text output test
    args_text = argparse.Namespace(
        ticker="MSFT",
        total_assets=50_000.0,
        total_liabilities=15_000.0,
        working_capital=10_000.0,
        retained_earnings=20_000.0,
        ebit=12_000.0,
        sales=40_000.0,
        market_cap=200_000.0,
        current_assets=20_000.0,
        current_liabilities=10_000.0,
        net_income=10_000.0,
        funds_from_operations=14_000.0,
        json=False,
    )
    altman_command(args_text)
    captured = capsys.readouterr()
    assert "Altman (1968, 2000) Z-Score & Ohlson (1980) O-Score Solvency Engine" in captured.out
    assert "MSFT" in captured.out
    assert "Altman Manufacturing Z-Score" in captured.out

    # JSON output test
    args_json = argparse.Namespace(
        ticker="MSFT",
        total_assets=50_000.0,
        total_liabilities=15_000.0,
        working_capital=10_000.0,
        retained_earnings=20_000.0,
        ebit=12_000.0,
        sales=40_000.0,
        market_cap=200_000.0,
        current_assets=20_000.0,
        current_liabilities=10_000.0,
        net_income=10_000.0,
        funds_from_operations=14_000.0,
        json=True,
    )
    altman_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "MSFT"
    assert "altman_z_score" in data
    assert "signature" in data


def test_input_validation_altman() -> None:
    """Test strict security input validation on ticker and financial metrics."""
    with pytest.raises(SecurityValidationError):
        calculate_altman_zscore(
            ticker="BAD$TICKER!",
            latest_price=100.0,
            total_assets=1000.0,
            total_liabilities=500.0,
            working_capital=200.0,
            retained_earnings=300.0,
            ebit=100.0,
            sales=800.0,
            market_cap=1200.0,
            current_assets=400.0,
            current_liabilities=200.0,
            net_income=80.0,
            funds_from_operations=120.0,
        )

    with pytest.raises(ValueError, match="must be finite"):
        calculate_altman_zscore(
            ticker="AAPL",
            latest_price=float("nan"),
            total_assets=1000.0,
            total_liabilities=500.0,
            working_capital=200.0,
            retained_earnings=300.0,
            ebit=100.0,
            sales=800.0,
            market_cap=1200.0,
            current_assets=400.0,
            current_liabilities=200.0,
            net_income=80.0,
            funds_from_operations=120.0,
        )
