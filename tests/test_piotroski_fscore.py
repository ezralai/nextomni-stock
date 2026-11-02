"""Unit and integration test suite for Joseph D. Piotroski (2000) Fundamental F-Score Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.piotroski_fscore import (
    PiotroskiFScoreReceipt,
    calculate_piotroski_fscore,
    classify_fscore_regime,
    evaluate_piotroski_fscore_profile,
)
from src.cli import fscore_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_piotroski_fscore_perfect_compounder() -> None:
    """Test perfect 9/9 fundamental compounder passing all profitability, leverage, and efficiency tests."""
    receipt = calculate_piotroski_fscore(
        ticker="PERFECT",
        latest_price=180.0,
        net_income=1_000_000_000.0,
        operating_cash_flow=1_400_000_000.0,  # CFO > NI (accrual discipline)
        total_assets=10_000_000_000.0,
        return_on_assets_pct=10.0,  # ROA > 0
        delta_roa_pct=1.5,  # ΔROA > 0
        delta_long_term_debt=-150_000_000.0,  # Debt reduced
        delta_current_ratio=0.25,  # Liquidity expanded
        shares_dilution_pct=-1.2,  # Shares repurchased
        delta_gross_margin_pct=1.1,  # Margin expanded
        delta_asset_turnover=0.04,  # Turnover improved
        market_cap=25_000_000_000.0,
    )

    assert receipt.ticker == "PERFECT"
    assert receipt.latest_price == 180.0
    assert receipt.f_score == 9
    assert receipt.profitability_points == 4
    assert receipt.leverage_liquidity_points == 3
    assert receipt.operating_efficiency_points == 2
    assert receipt.is_strong_compounder is True
    assert receipt.is_value_trap is False
    assert receipt.regime == "STRONG_FUNDAMENTAL_COMPOUNDER"
    assert receipt.expected_fundamental_alpha_pct == +7.20
    assert receipt.verify_integrity() is True


def test_calculate_piotroski_fscore_distressed_value_trap() -> None:
    """Test acute fundamental distress value trap failing all 9 signals (F = 0)."""
    receipt = calculate_piotroski_fscore(
        ticker="DISTRESS",
        latest_price=6.0,
        net_income=-150_000_000.0,  # Negative net income
        operating_cash_flow=-300_000_000.0,  # Negative CFO and CFO < NI
        total_assets=5_000_000_000.0,
        return_on_assets_pct=-3.0,  # Negative ROA
        delta_roa_pct=-1.8,  # Deteriorating ROA
        delta_long_term_debt=450_000_000.0,  # Debt increased
        delta_current_ratio=-0.35,  # Liquidity deteriorated
        shares_dilution_pct=6.5,  # Highly dilutive equity issuance
        delta_gross_margin_pct=-2.2,  # Margin collapsed
        delta_asset_turnover=-0.08,  # Turnover decayed
        market_cap=400_000_000.0,
    )

    assert receipt.f_score == 0
    assert receipt.profitability_points == 0
    assert receipt.leverage_liquidity_points == 0
    assert receipt.operating_efficiency_points == 0
    assert receipt.is_value_trap is True
    assert receipt.is_strong_compounder is False
    assert receipt.regime == "ACUTE_FUNDAMENTAL_DISTRESS"
    assert receipt.expected_fundamental_alpha_pct == -7.20
    assert receipt.verify_integrity() is True


def test_calculate_piotroski_fscore_sound_profile() -> None:
    """Test sound financial profile with 6/9 passing signals."""
    receipt = calculate_piotroski_fscore(
        ticker="SOUND",
        latest_price=55.0,
        net_income=500_000_000.0,
        operating_cash_flow=600_000_000.0,
        total_assets=6_000_000_000.0,
        return_on_assets_pct=8.3,
        delta_roa_pct=0.5,
        delta_long_term_debt=50_000_000.0,  # Debt up (fail)
        delta_current_ratio=0.10,
        shares_dilution_pct=0.8,  # Dilution (fail)
        delta_gross_margin_pct=-0.2,  # Margin down (fail)
        delta_asset_turnover=0.02,
    )

    assert receipt.f_score == 6
    assert receipt.regime == "SOUND_FINANCIAL_PROFILE"
    assert receipt.is_strong_compounder is False
    assert receipt.is_value_trap is False
    assert receipt.expected_fundamental_alpha_pct == +2.40
    assert receipt.verify_integrity() is True


def test_calculate_piotroski_fscore_vulnerable_trap() -> None:
    """Test vulnerable value trap with 4/9 passing signals."""
    receipt = calculate_piotroski_fscore(
        ticker="VULN",
        latest_price=30.0,
        net_income=100_000_000.0,
        operating_cash_flow=50_000_000.0,  # CFO < NI (fail)
        total_assets=4_000_000_000.0,
        return_on_assets_pct=2.5,
        delta_roa_pct=-0.4,  # dROA < 0 (fail)
        delta_long_term_debt=100_000_000.0,  # Debt up (fail)
        delta_current_ratio=-0.05,  # Liq down (fail)
        shares_dilution_pct=-0.1,  # No dilution (pass)
        delta_gross_margin_pct=-0.5,  # Margin down (fail)
        delta_asset_turnover=0.01,  # Turnover up (pass)
    )

    assert receipt.f_score == 4
    assert receipt.regime == "VULNERABLE_VALUE_TRAP"
    assert receipt.is_value_trap is False
    assert receipt.verify_integrity() is True


def test_piotroski_monotonicity() -> None:
    """Verify that passing incremental signals strictly increases F-score and expected alpha."""
    base_price = 100.0
    rec_base = calculate_piotroski_fscore(
        ticker="T1",
        latest_price=base_price,
        net_income=500_000_000.0,
        operating_cash_flow=600_000_000.0,
        delta_long_term_debt=100_000_000.0,  # Debt increased (0 pt)
    )
    rec_improved = calculate_piotroski_fscore(
        ticker="T1",
        latest_price=base_price,
        net_income=500_000_000.0,
        operating_cash_flow=600_000_000.0,
        delta_long_term_debt=-100_000_000.0,  # Debt reduced (1 pt)
    )

    assert rec_improved.f_score == rec_base.f_score + 1
    assert rec_improved.expected_fundamental_alpha_pct > rec_base.expected_fundamental_alpha_pct
    assert abs(rec_improved.expected_fundamental_alpha_pct - rec_base.expected_fundamental_alpha_pct - 1.6) < 1e-4


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    receipt = calculate_piotroski_fscore(
        ticker="AUDIT",
        latest_price=100.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with f_score
    receipt.f_score += 1
    assert receipt.verify_integrity() is False

    # Tamper with return_on_assets_pct
    receipt.f_score -= 1
    assert receipt.verify_integrity() is True
    receipt.return_on_assets_pct += 2.0
    assert receipt.verify_integrity() is False


def test_classify_fscore_regime_boundaries() -> None:
    """Test regime boundary classification."""
    assert classify_fscore_regime(9) == "STRONG_FUNDAMENTAL_COMPOUNDER"
    assert classify_fscore_regime(8) == "STRONG_FUNDAMENTAL_COMPOUNDER"
    assert classify_fscore_regime(7) == "SOUND_FINANCIAL_PROFILE"
    assert classify_fscore_regime(5) == "SOUND_FINANCIAL_PROFILE"
    assert classify_fscore_regime(4) == "VULNERABLE_VALUE_TRAP"
    assert classify_fscore_regime(3) == "VULNERABLE_VALUE_TRAP"
    assert classify_fscore_regime(2) == "ACUTE_FUNDAMENTAL_DISTRESS"
    assert classify_fscore_regime(0) == "ACUTE_FUNDAMENTAL_DISTRESS"


def test_evaluate_piotroski_fscore_profile_database(tmp_path: Path) -> None:
    """Test profile evaluation pulling observations from SQLite database."""
    test_db = tmp_path / "test_piot.db"
    conn = db.get_connection(test_db)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE stock_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            observation_time TEXT,
            close_price REAL,
            volume REAL
        )
        """
    )
    prices = [100.0 + i * 0.5 for i in range(50)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("PIOT", f"2026-01-{i+1:02d}", p, 2_000_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_piotroski_fscore_profile(
        ticker="PIOT",
        return_on_assets_pct=14.0,
        delta_roa_pct=1.0,
        db_path=test_db,
    )

    assert receipt.ticker == "PIOT"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.f_score >= 7
    assert receipt.verify_integrity() is True


def test_evaluate_piotroski_fscore_profile_fallback() -> None:
    """Test profile evaluation fallback when ticker is not in database."""
    receipt = evaluate_piotroski_fscore_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert 0 <= receipt.f_score <= 9
    assert receipt.verify_integrity() is True


def test_cli_fscore_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI nextomni fscore command text and json output."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        net_income=1_500_000_000.0,
        operating_cash_flow=1_900_000_000.0,
        cash_flow=None,
        total_assets=12_000_000_000.0,
        roa=12.5,
        delta_roa=1.2,
        delta_debt=-100_000_000.0,
        delta_current_ratio=0.15,
        shares_dilution=-0.8,
        delta_gross_margin=0.9,
        delta_asset_turnover=0.03,
        json=False,
    )
    fscore_command(args_text)
    captured_text = capsys.readouterr().out
    assert "Joseph D. Piotroski (2000) Fundamental F-Score Engine" in captured_text
    assert "AAPL" in captured_text

    args_json = argparse.Namespace(
        ticker="AAPL",
        net_income=1_500_000_000.0,
        operating_cash_flow=1_900_000_000.0,
        cash_flow=None,
        total_assets=12_000_000_000.0,
        roa=12.5,
        delta_roa=1.2,
        delta_debt=-100_000_000.0,
        delta_current_ratio=0.15,
        shares_dilution=-0.8,
        delta_gross_margin=0.9,
        delta_asset_turnover=0.03,
        json=True,
    )
    fscore_command(args_json)
    captured_json = capsys.readouterr().out
    parsed = json.loads(captured_json)
    assert parsed["ticker"] == "AAPL"
    assert parsed["f_score"] == 9
    assert "signature" in parsed


def test_input_validation_piotroski_fscore() -> None:
    """Test defensive security validation on non-positive or invalid input parameters."""
    with pytest.raises(ValueError):
        calculate_piotroski_fscore(ticker="TEST", latest_price=-10.0)

    with pytest.raises(ValueError):
        calculate_piotroski_fscore(ticker="TEST", latest_price=50.0, total_assets=0.0)

    with pytest.raises(ValueError):
        calculate_piotroski_fscore(ticker="TEST", latest_price=50.0, market_cap=-500.0)

    with pytest.raises(ValueError):
        calculate_piotroski_fscore(ticker="TEST", latest_price=float("nan"))

    with pytest.raises(ValueError):
        calculate_piotroski_fscore(ticker="TEST", latest_price=50.0, return_on_assets_pct=float("inf"))

    with pytest.raises(SecurityValidationError):
        evaluate_piotroski_fscore_profile(ticker="INVALID TICKER$$$")
