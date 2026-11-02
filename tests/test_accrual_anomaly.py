"""Unit and integration test suite for Richard G. Sloan (1996) Accrual Anomaly Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.accrual_anomaly import (
    AccrualAnomalyReceipt,
    calculate_accrual_anomaly,
    classify_accrual_regime,
    evaluate_accrual_anomaly_profile,
)
from src.cli import accruals_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_accrual_anomaly_cash_compounder() -> None:
    """Test high cash conversion compounder where operating cash flow strongly exceeds net income."""
    receipt = calculate_accrual_anomaly(
        ticker="CASHFLOW",
        latest_price=120.0,
        net_income=1_000_000_000.0,  # $1.0B
        operating_cash_flow=1_600_000_000.0,  # $1.6B (+60% cash conversion)
        average_total_assets=10_000_000_000.0,  # $10B
    )

    assert receipt.ticker == "CASHFLOW"
    assert receipt.latest_price == 120.0
    assert receipt.total_accruals == -600_000_000.0
    assert receipt.accrual_ratio_cfa == -0.06
    assert receipt.accrual_ratio_pct == -6.0
    assert receipt.cash_flow_to_net_income == 1.6
    assert receipt.earnings_quality_index == 65.0
    assert receipt.is_cash_quality_compounder is True
    assert receipt.is_accrual_trap is False
    assert receipt.regime == "HIGH_CASH_CONVERSION_QUALITY"
    assert receipt.expected_accrual_alpha_pct == +2.7
    assert receipt.verify_integrity() is True


def test_calculate_accrual_anomaly_accrual_trap() -> None:
    """Test accrual inflation trap where net income is largely comprised of paper accounting accruals."""
    receipt = calculate_accrual_anomaly(
        ticker="TRAP",
        latest_price=22.0,
        net_income=1_200_000_000.0,  # $1.2B headline net income
        operating_cash_flow=200_000_000.0,  # Only $200M actual cash flow!
        average_total_assets=10_000_000_000.0,  # $10B
    )

    assert receipt.total_accruals == +1_000_000_000.0
    assert receipt.accrual_ratio_cfa == 0.10
    assert receipt.accrual_ratio_pct == 10.0
    assert receipt.cash_flow_to_net_income < 0.20
    assert receipt.earnings_quality_index == 25.0
    assert receipt.is_accrual_trap is True
    assert receipt.is_cash_quality_compounder is False
    assert receipt.regime == "ACCRUAL_MANIPULATION_TRAP"
    assert receipt.expected_accrual_alpha_pct == -4.5
    assert receipt.verify_integrity() is True


def test_calculate_accrual_anomaly_sound_persistence() -> None:
    """Test balanced corporate earnings with sound persistence and modest accruals."""
    receipt = calculate_accrual_anomaly(
        ticker="SOUND",
        latest_price=80.0,
        net_income=1_000_000_000.0,
        operating_cash_flow=1_100_000_000.0,
        average_total_assets=10_000_000_000.0,
    )

    assert receipt.accrual_ratio_cfa == -0.01
    assert receipt.earnings_quality_index == 52.5
    assert receipt.regime == "SOUND_EARNINGS_PERSISTENCE"
    assert receipt.is_cash_quality_compounder is False
    assert receipt.is_accrual_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_accrual_anomaly_moderate_distortion() -> None:
    """Test firm with elevated positive accruals warranting scrutiny."""
    receipt = calculate_accrual_anomaly(
        ticker="DISTORT",
        latest_price=45.0,
        net_income=1_000_000_000.0,
        operating_cash_flow=500_000_000.0,
        average_total_assets=10_000_000_000.0,
    )

    assert receipt.accrual_ratio_cfa == 0.05
    assert receipt.earnings_quality_index == 37.5
    assert receipt.regime == "MODERATE_ACCRUAL_DISTORTION"
    assert receipt.verify_integrity() is True


def test_accrual_monotonicity() -> None:
    """Verify Sloan (1996) monotonicity: higher cash flow increases EQI and alpha, higher paper profit decreases it."""
    base_assets = 10_000_000_000.0
    base_price = 100.0
    base_net_income = 1_000_000_000.0

    # 1. Monotonicity with respect to Operating Cash Flow
    rec_low_cash = calculate_accrual_anomaly(
        ticker="T1", latest_price=base_price, net_income=base_net_income,
        operating_cash_flow=800_000_000.0, average_total_assets=base_assets
    )
    rec_high_cash = calculate_accrual_anomaly(
        ticker="T1", latest_price=base_price, net_income=base_net_income,
        operating_cash_flow=1_500_000_000.0, average_total_assets=base_assets
    )
    assert rec_high_cash.earnings_quality_index > rec_low_cash.earnings_quality_index
    assert rec_high_cash.expected_accrual_alpha_pct > rec_low_cash.expected_accrual_alpha_pct
    assert rec_high_cash.accrual_ratio_cfa < rec_low_cash.accrual_ratio_cfa

    # 2. Monotonicity with respect to Headline Net Income (Accrual inflation)
    rec_honest = calculate_accrual_anomaly(
        ticker="T2", latest_price=base_price, net_income=1_000_000_000.0,
        operating_cash_flow=1_000_000_000.0, average_total_assets=base_assets
    )
    rec_inflated = calculate_accrual_anomaly(
        ticker="T2", latest_price=base_price, net_income=1_800_000_000.0,
        operating_cash_flow=1_000_000_000.0, average_total_assets=base_assets
    )
    assert rec_honest.earnings_quality_index > rec_inflated.earnings_quality_index
    assert rec_honest.expected_accrual_alpha_pct > rec_inflated.expected_accrual_alpha_pct


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    receipt = calculate_accrual_anomaly(
        ticker="AUDIT",
        latest_price=100.0,
        net_income=500_000_000.0,
        operating_cash_flow=650_000_000.0,
        average_total_assets=5_000_000_000.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with earnings quality index
    receipt.earnings_quality_index += 5.0
    assert receipt.verify_integrity() is False

    # Tamper with net income
    receipt.earnings_quality_index -= 5.0
    assert receipt.verify_integrity() is True
    receipt.net_income += 10_000_000.0
    assert receipt.verify_integrity() is False


def test_classify_accrual_regime_boundaries() -> None:
    """Test regime boundary classification."""
    assert classify_accrual_regime(-0.06, 65.0) == "HIGH_CASH_CONVERSION_QUALITY"
    assert classify_accrual_regime(-0.04, 60.0) == "HIGH_CASH_CONVERSION_QUALITY"
    assert classify_accrual_regime(0.10, 25.0) == "ACCRUAL_MANIPULATION_TRAP"
    assert classify_accrual_regime(0.08, 30.0) == "ACCRUAL_MANIPULATION_TRAP"
    assert classify_accrual_regime(0.01, 47.5) == "SOUND_EARNINGS_PERSISTENCE"
    assert classify_accrual_regime(0.05, 37.5) == "MODERATE_ACCRUAL_DISTORTION"


def test_evaluate_accrual_anomaly_profile_database(tmp_path: Path) -> None:
    """Test profile evaluation pulling observations from SQLite database."""
    test_db = tmp_path / "test_accrual.db"
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
    prices = [100.0 * (1.0 + 0.012 * math.sin(i / 5.0)) for i in range(50)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("ACCR", f"2026-01-{i+1:02d}", p, 2_000_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_accrual_anomaly_profile(
        ticker="ACCR",
        net_income=1_000_000_000.0,
        operating_cash_flow=1_400_000_000.0,
        average_total_assets=8_000_000_000.0,
        db_path=test_db,
    )

    assert receipt.ticker == "ACCR"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.total_accruals == -400_000_000.0
    assert receipt.verify_integrity() is True


def test_evaluate_accrual_anomaly_profile_fallback() -> None:
    """Test profile evaluation fallback when ticker is not in database."""
    receipt = evaluate_accrual_anomaly_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert receipt.earnings_quality_index > 0
    assert receipt.verify_integrity() is True


def test_cli_accruals_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI nextomni accruals command text and json output."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        net_income=1_000_000_000.0,
        operating_cash_flow=1_500_000_000.0,
        cash_flow=None,
        total_assets=10_000_000_000.0,
        json=False,
    )
    accruals_command(args_text)
    captured_text = capsys.readouterr().out
    assert "Richard G. Sloan (1996) Accrual Anomaly Engine" in captured_text
    assert "AAPL" in captured_text

    args_json = argparse.Namespace(
        ticker="AAPL",
        net_income=1_000_000_000.0,
        operating_cash_flow=1_500_000_000.0,
        cash_flow=None,
        total_assets=10_000_000_000.0,
        json=True,
    )
    accruals_command(args_json)
    captured_json = capsys.readouterr().out
    parsed = json.loads(captured_json)
    assert parsed["ticker"] == "AAPL"
    assert "earnings_quality_index" in parsed
    assert "signature" in parsed


def test_input_validation_accrual_anomaly() -> None:
    """Test defensive security validation on non-positive or invalid input parameters."""
    with pytest.raises(ValueError):
        calculate_accrual_anomaly(ticker="TEST", latest_price=-10.0)

    with pytest.raises(ValueError):
        calculate_accrual_anomaly(ticker="TEST", latest_price=50.0, average_total_assets=-1000.0)

    with pytest.raises(ValueError):
        calculate_accrual_anomaly(ticker="TEST", latest_price=50.0, average_total_assets=0.0)

    with pytest.raises(ValueError):
        calculate_accrual_anomaly(ticker="TEST", latest_price=float("nan"))

    with pytest.raises(ValueError):
        calculate_accrual_anomaly(ticker="TEST", latest_price=50.0, net_income=float("inf"))

    with pytest.raises(SecurityValidationError):
        evaluate_accrual_anomaly_profile(ticker="INVALID TICKER$$$")
