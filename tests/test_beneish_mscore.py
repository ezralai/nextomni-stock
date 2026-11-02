"""Unit and integration test suite for Messod D. Beneish (1999) Forensic Accounting M-Score Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.beneish_mscore import (
    BeneishMScoreReceipt,
    calculate_beneish_mscore,
    classify_beneish_regime,
    evaluate_beneish_mscore_profile,
)
from src.cli import beneish_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_beneish_mscore_clean_financials() -> None:
    """Test clean, unmanipulated corporate financial statement profile."""
    receipt = calculate_beneish_mscore(
        ticker="PRISTINE",
        latest_price=150.0,
        dsri=0.92,
        gmi=0.95,
        aqi=0.90,
        sgi=1.02,
        depi=1.00,
        sgai=1.05,
        lvgi=0.95,
        tata=-0.04,  # Negative accruals (cash flows exceed net income)
        market_cap=50_000_000_000.0,
    )

    assert receipt.ticker == "PRISTINE"
    assert receipt.latest_price == 150.0
    assert receipt.m_score <= -2.22
    assert receipt.manipulation_probability_pct <= 1.5
    assert receipt.is_clean_profile is True
    assert receipt.is_manipulator_risk is False
    assert receipt.regime == "CLEAN_UNMANIPULATED_FINANCIALS"
    assert receipt.expected_forensic_alpha_pct == +2.5
    assert receipt.verify_integrity() is True


def test_calculate_beneish_mscore_manipulator_enron_profile() -> None:
    """Test acute accounting fraud footprint mimicking Enron's historical pre-bankruptcy profile."""
    receipt = calculate_beneish_mscore(
        ticker="ENRON",
        latest_price=12.0,
        dsri=1.65,  # Aggressive uncollected billing
        gmi=1.45,  # Deteriorating gross margins
        aqi=1.40,  # Capitalizing expenses into intangibles
        sgi=1.50,  # Decelerating growth pressure
        depi=1.15,  # Lengthening asset depreciable lives
        sgai=0.95,
        lvgi=1.25,  # Rising debt leverage
        tata=0.14,  # High Sloan paper accruals
        market_cap=2_000_000_000.0,
    )

    assert receipt.m_score > -1.78
    assert receipt.manipulation_probability_pct >= 25.0
    assert receipt.is_manipulator_risk is True
    assert receipt.is_clean_profile is False
    assert receipt.regime == "HIGH_MANIPULATION_RISK"
    assert receipt.expected_forensic_alpha_pct == -14.5
    assert receipt.verify_integrity() is True


def test_calculate_beneish_mscore_moderate_concern() -> None:
    """Test borderline accounting profile warranting forensic scrutiny."""
    receipt = calculate_beneish_mscore(
        ticker="BORDER",
        latest_price=60.0,
        dsri=1.10,
        gmi=1.08,
        aqi=1.06,
        sgi=1.09,
        depi=1.00,
        sgai=1.00,
        lvgi=1.00,
        tata=0.02,
    )

    assert -2.22 < receipt.m_score <= -1.78
    assert receipt.regime == "MODERATE_ACCOUNTING_CONCERN"
    assert receipt.is_manipulator_risk is False
    assert receipt.is_clean_profile is False
    assert receipt.expected_forensic_alpha_pct == -1.0
    assert receipt.verify_integrity() is True


def test_beneish_index_monotonicity() -> None:
    """Verify Beneish model monotonicity across DSRI, GMI, AQI, and TATA."""
    base_price = 100.0

    # 1. Monotonicity with respect to DSRI (revenue acceleration)
    low_dsri = calculate_beneish_mscore(ticker="T1", latest_price=base_price, dsri=0.90)
    high_dsri = calculate_beneish_mscore(ticker="T1", latest_price=base_price, dsri=1.50)
    assert high_dsri.m_score > low_dsri.m_score
    assert high_dsri.manipulation_probability_pct > low_dsri.manipulation_probability_pct

    # 2. Monotonicity with respect to TATA (accruals)
    low_tata = calculate_beneish_mscore(ticker="T2", latest_price=base_price, tata=-0.05)
    high_tata = calculate_beneish_mscore(ticker="T2", latest_price=base_price, tata=0.10)
    assert high_tata.m_score > low_tata.m_score

    # 3. Monotonicity with respect to AQI (expense capitalization)
    low_aqi = calculate_beneish_mscore(ticker="T3", latest_price=base_price, aqi=0.85)
    high_aqi = calculate_beneish_mscore(ticker="T3", latest_price=base_price, aqi=1.40)
    assert high_aqi.m_score > low_aqi.m_score


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    receipt = calculate_beneish_mscore(
        ticker="AUDIT",
        latest_price=100.0,
        dsri=1.1,
        gmi=1.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with m_score
    receipt.m_score += 0.5
    assert receipt.verify_integrity() is False

    # Tamper with dsri
    receipt.m_score -= 0.5
    assert receipt.verify_integrity() is True
    receipt.dsri += 0.2
    assert receipt.verify_integrity() is False


def test_classify_beneish_regime_boundaries() -> None:
    """Test regime boundary classification."""
    assert classify_beneish_regime(-1.50) == "HIGH_MANIPULATION_RISK"
    assert classify_beneish_regime(-1.77) == "HIGH_MANIPULATION_RISK"
    assert classify_beneish_regime(-1.78) == "MODERATE_ACCOUNTING_CONCERN"
    assert classify_beneish_regime(-2.00) == "MODERATE_ACCOUNTING_CONCERN"
    assert classify_beneish_regime(-2.22) == "CLEAN_UNMANIPULATED_FINANCIALS"
    assert classify_beneish_regime(-2.50) == "CLEAN_UNMANIPULATED_FINANCIALS"


def test_evaluate_beneish_mscore_profile_database(tmp_path: Path) -> None:
    """Test profile evaluation pulling observations from SQLite database."""
    test_db = tmp_path / "test_beneish.db"
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
    prices = [100.0 * (1.0 + 0.01 * math.sin(i / 4.0)) for i in range(50)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("ENRN", f"2026-01-{i+1:02d}", p, 2_500_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_beneish_mscore_profile(
        ticker="ENRN",
        dsri=1.25,
        gmi=1.15,
        aqi=1.10,
        sgi=1.12,
        db_path=test_db,
    )

    assert receipt.ticker == "ENRN"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.dsri == 1.25
    assert receipt.gmi == 1.15
    assert receipt.verify_integrity() is True


def test_evaluate_beneish_mscore_profile_fallback() -> None:
    """Test profile evaluation fallback when ticker is not in database."""
    receipt = evaluate_beneish_mscore_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert receipt.m_score < 0
    assert receipt.verify_integrity() is True


def test_cli_beneish_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI nextomni beneish command text and json output."""
    args_text = argparse.Namespace(
        ticker="ENRN",
        dsri=1.5,
        gmi=1.3,
        aqi=1.2,
        sgi=1.3,
        depi=1.1,
        sgai=1.0,
        lvgi=1.1,
        tata=0.08,
        json=False,
    )
    beneish_command(args_text)
    captured_text = capsys.readouterr().out
    assert "Messod D. Beneish (1999) Forensic Accounting M-Score" in captured_text
    assert "ENRN" in captured_text

    args_json = argparse.Namespace(
        ticker="ENRN",
        dsri=1.5,
        gmi=1.3,
        aqi=1.2,
        sgi=1.3,
        depi=1.1,
        sgai=1.0,
        lvgi=1.1,
        tata=0.08,
        json=True,
    )
    beneish_command(args_json)
    captured_json = capsys.readouterr().out
    parsed = json.loads(captured_json)
    assert parsed["ticker"] == "ENRN"
    assert "m_score" in parsed
    assert "signature" in parsed


def test_input_validation_beneish_mscore() -> None:
    """Test defensive security validation on non-positive or negative input parameters."""
    with pytest.raises(ValueError):
        calculate_beneish_mscore(ticker="TEST", latest_price=-5.0)

    with pytest.raises(ValueError):
        calculate_beneish_mscore(ticker="TEST", latest_price=50.0, market_cap=-1000.0)

    with pytest.raises(ValueError):
        calculate_beneish_mscore(ticker="TEST", latest_price=50.0, dsri=-0.5)

    with pytest.raises(ValueError):
        calculate_beneish_mscore(ticker="TEST", latest_price=float("nan"))

    with pytest.raises(ValueError):
        calculate_beneish_mscore(ticker="TEST", latest_price=50.0, gmi=float("inf"))

    with pytest.raises(SecurityValidationError):
        evaluate_beneish_mscore_profile(ticker="INVALID TICKER$$$")
