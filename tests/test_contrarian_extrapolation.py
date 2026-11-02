"""Unit and integration tests for Lakonishok, Shleifer & Vishny (1994) Contrarian Engine."""

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.analytics.contrarian_extrapolation import (
    ContrarianExtrapolationReceipt,
    calculate_contrarian_extrapolation,
    classify_contrarian_regime,
    evaluate_contrarian_extrapolation_profile,
)
from src.cli import contrarian_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_contrarian_extrapolation_value_winner() -> None:
    """Test high conviction contrarian value stock with depressed multiples and asymmetric upside."""
    receipt = calculate_contrarian_extrapolation(
        ticker="VALU",
        latest_price=45.0,
        sales_growth_3yr_pct=3.0,
        earnings_growth_3yr_pct=4.0,
        trailing_momentum_pct=-10.0,
        price_to_earnings=11.0,
        price_to_cash_flow=8.0,
        price_to_book=1.2,
    )

    assert receipt.ticker == "VALU"
    assert receipt.latest_price == 45.0
    assert receipt.naive_extrapolation_index <= 0.45
    assert receipt.valuation_multiple_composite <= 0.35
    assert receipt.is_contrarian_value is True
    assert receipt.is_glamour_trap is False
    assert receipt.regime == "HIGH_CONVICTION_CONTRARIAN_VALUE"
    assert receipt.expected_contrarian_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_contrarian_extrapolation_glamour_trap() -> None:
    """Test vulnerable glamour stock with extreme past growth extrapolated into bubble multiples."""
    receipt = calculate_contrarian_extrapolation(
        ticker="GLAM",
        latest_price=220.0,
        sales_growth_3yr_pct=35.0,
        earnings_growth_3yr_pct=45.0,
        trailing_momentum_pct=60.0,
        price_to_earnings=65.0,
        price_to_cash_flow=45.0,
        price_to_book=12.0,
    )

    assert receipt.ticker == "GLAM"
    assert receipt.naive_extrapolation_index >= 0.70
    assert receipt.valuation_multiple_composite >= 0.75
    assert receipt.is_glamour_trap is True
    assert receipt.is_contrarian_value is False
    assert receipt.regime == "VULNERABLE_GLAMOUR_TRAP"
    assert receipt.expected_contrarian_alpha_pct < 0.0  # Sizable multiple compression drag
    assert receipt.verify_integrity() is True


def test_calculate_contrarian_extrapolation_rational_growth() -> None:
    """Test rational growth compounder with strong growth but disciplined multiple."""
    receipt = calculate_contrarian_extrapolation(
        ticker="GROW",
        latest_price=120.0,
        sales_growth_3yr_pct=22.0,
        earnings_growth_3yr_pct=25.0,
        trailing_momentum_pct=20.0,
        price_to_earnings=26.0,
        price_to_cash_flow=20.0,
        price_to_book=4.5,
    )

    assert receipt.ticker == "GROW"
    assert receipt.naive_extrapolation_index >= 0.60
    assert 0.45 <= receipt.valuation_multiple_composite < 0.65
    assert receipt.regime == "RATIONAL_GROWTH_COMPOUNDER"
    assert receipt.verify_integrity() is True


def test_calculate_contrarian_extrapolation_stable_out_of_favor() -> None:
    """Test stable out-of-favor stock with low valuation and stabilizing fundamentals."""
    receipt = calculate_contrarian_extrapolation(
        ticker="STAB",
        latest_price=80.0,
        sales_growth_3yr_pct=14.0,
        earnings_growth_3yr_pct=15.0,
        trailing_momentum_pct=8.0,
        price_to_earnings=14.0,
        price_to_cash_flow=10.0,
        price_to_book=1.8,
    )

    assert receipt.ticker == "STAB"
    assert receipt.valuation_multiple_composite <= 0.45
    assert receipt.regime in ("STABLE_OUT_OF_FAVOR", "HIGH_CONVICTION_CONTRARIAN_VALUE", "NEUTRAL_MARKET_EQUILIBRIUM")
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    receipt = calculate_contrarian_extrapolation(
        ticker="AUDIT",
        latest_price=100.0,
        sales_growth_3yr_pct=10.0,
        earnings_growth_3yr_pct=12.0,
        trailing_momentum_pct=15.0,
    )

    assert receipt.verify_integrity() is True

    # Tamper with naive_extrapolation_index
    receipt.naive_extrapolation_index += 0.05
    assert receipt.verify_integrity() is False

    # Restore and tamper with expected_contrarian_alpha_pct
    receipt.naive_extrapolation_index -= 0.05
    assert receipt.verify_integrity() is True
    receipt.expected_contrarian_alpha_pct += 1.5
    assert receipt.verify_integrity() is False


def test_classify_contrarian_regime_boundaries() -> None:
    """Test boundary conditions of regime classification."""
    assert classify_contrarian_regime(0.40, 0.30) == "HIGH_CONVICTION_CONTRARIAN_VALUE"
    assert classify_contrarian_regime(0.70, 0.80) == "VULNERABLE_GLAMOUR_TRAP"
    assert classify_contrarian_regime(0.65, 0.50) == "RATIONAL_GROWTH_COMPOUNDER"
    assert classify_contrarian_regime(0.55, 0.40) == "STABLE_OUT_OF_FAVOR"
    assert classify_contrarian_regime(0.52, 0.52) == "NEUTRAL_MARKET_EQUILIBRIUM"


def test_loss_making_negative_multiples_handling() -> None:
    """Test robust handling of negative P/E and P/CF without mathematical failure."""
    receipt = calculate_contrarian_extrapolation(
        ticker="LOSS",
        latest_price=25.0,
        sales_growth_3yr_pct=-15.0,
        earnings_growth_3yr_pct=-30.0,
        trailing_momentum_pct=-35.0,
        price_to_earnings=-12.0,
        price_to_cash_flow=-8.0,
        price_to_book=-1.5,
    )

    assert receipt.ticker == "LOSS"
    assert receipt.price_to_earnings == -12.0
    assert 0.0 < receipt.valuation_multiple_composite <= 1.0
    assert receipt.verify_integrity() is True


def test_evaluate_contrarian_extrapolation_profile_database(tmp_path: Path) -> None:
    """Test evaluation pulling real observations from a SQLite database."""
    test_db = tmp_path / "test_lsv.db"
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
    prices = [80.0 + i * 0.3 for i in range(40)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("LSVT", f"2026-03-{i+1:02d}", p, 1_200_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_contrarian_extrapolation_profile(
        ticker="LSVT",
        db_path=test_db,
    )

    assert receipt.ticker == "LSVT"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.trailing_momentum_pct > 0
    assert receipt.verify_integrity() is True


def test_evaluate_contrarian_extrapolation_profile_fallback() -> None:
    """Test evaluation deterministic fallback when ticker is not in database."""
    receipt = evaluate_contrarian_extrapolation_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert 0.0 <= receipt.naive_extrapolation_index <= 1.0
    assert 0.0 <= receipt.valuation_multiple_composite <= 1.0
    assert receipt.verify_integrity() is True


def test_cli_contrarian_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI contrarian command text and json output."""
    args_text = argparse.Namespace(
        ticker="BRK",
        sales_growth=5.0,
        earnings_growth=8.0,
        pe=14.0,
        pcf=10.0,
        pb=1.3,
        json=False,
    )
    contrarian_command(args_text)
    captured = capsys.readouterr()
    assert "Lakonishok, Shleifer & Vishny (1994) Contrarian Investment Engine" in captured.out
    assert "BRK" in captured.out
    assert "Naive Extrapolation (NEI)" in captured.out

    args_json = argparse.Namespace(
        ticker="BRK",
        sales_growth=5.0,
        earnings_growth=8.0,
        pe=14.0,
        pcf=10.0,
        pb=1.3,
        json=True,
    )
    contrarian_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "BRK"
    assert "naive_extrapolation_index" in data
    assert "valuation_multiple_composite" in data
    assert "signature" in data


def test_input_validation_contrarian() -> None:
    """Test defensive security validation on tickers and input parameters."""
    with pytest.raises(SecurityValidationError):
        calculate_contrarian_extrapolation(
            ticker="INVALID_TICKER_SYMBOL",
            latest_price=100.0,
            sales_growth_3yr_pct=5.0,
            earnings_growth_3yr_pct=5.0,
            trailing_momentum_pct=5.0,
        )

    with pytest.raises(ValueError, match="Latest price must be strictly positive"):
        calculate_contrarian_extrapolation(
            ticker="AAPL",
            latest_price=-5.0,
            sales_growth_3yr_pct=5.0,
            earnings_growth_3yr_pct=5.0,
            trailing_momentum_pct=5.0,
        )

    with pytest.raises(ValueError, match="sales_growth_3yr_pct must be finite"):
        calculate_contrarian_extrapolation(
            ticker="AAPL",
            latest_price=100.0,
            sales_growth_3yr_pct=float("nan"),
            earnings_growth_3yr_pct=5.0,
            trailing_momentum_pct=5.0,
        )
