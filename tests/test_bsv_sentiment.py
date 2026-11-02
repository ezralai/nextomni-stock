"""Unit and integration tests for Barberis, Shleifer & Vishny (1998) Dual-Heuristic Sentiment Engine."""

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.analytics.bsv_sentiment import (
    BSVSentimentReceipt,
    calculate_bsv_sentiment,
    classify_bsv_regime,
    evaluate_bsv_sentiment_profile,
)
from src.cli import bsv_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_bsv_sentiment_conservatism_drift() -> None:
    """Test conservatism underreaction drift where an isolated surprise is underreacted to."""
    prices = [100.0 * (1.0 + 0.002 * i) for i in range(25)]
    receipt = calculate_bsv_sentiment(
        ticker="CONS",
        prices=prices,
        latest_surprise_pct=6.5,
        consecutive_streak=1,  # Isolated single surprise
    )

    assert receipt.ticker == "CONS"
    assert receipt.consecutive_streak_quarters == 1
    assert receipt.trending_state_probability <= 0.45
    assert receipt.is_conservatism_drift is True
    assert receipt.is_representativeness_peak is False
    assert receipt.regime == "CONSERVATISM_UNDERREACTION_DRIFT"
    assert receipt.expected_bsv_alpha_pct > 0.0  # Positive PEAD continuation drift
    assert receipt.verify_integrity() is True


def test_calculate_bsv_sentiment_representativeness_peak() -> None:
    """Test representativeness overreaction peak where a 4-quarter streak creates speculative exhaustion."""
    prices = [100.0 * (1.0 + 0.008 * i) for i in range(40)]
    receipt = calculate_bsv_sentiment(
        ticker="REPR",
        prices=prices,
        latest_surprise_pct=8.0,
        consecutive_streak=4,  # 4 consecutive positive quarters
    )

    assert receipt.ticker == "REPR"
    assert receipt.consecutive_streak_quarters == 4
    assert receipt.trending_state_probability >= 0.65
    assert receipt.is_representativeness_peak is True
    assert receipt.is_conservatism_drift is False
    assert receipt.regime == "REPRESENTATIVENESS_OVERREACTION_PEAK"
    assert receipt.expected_bsv_alpha_pct < 0.0  # Mean-reversion reversal drag
    assert receipt.verify_integrity() is True


def test_calculate_bsv_sentiment_transitional_updating() -> None:
    """Test transitional belief updating regime with moderate streak length."""
    prices = [100.0 * (1.0 + 0.003 * i) for i in range(30)]
    receipt = calculate_bsv_sentiment(
        ticker="TRAN",
        prices=prices,
        latest_surprise_pct=4.0,
        consecutive_streak=2,
    )

    assert receipt.ticker == "TRAN"
    assert 0.40 <= receipt.trending_state_probability <= 0.62
    assert receipt.regime in ("TRANSITIONAL_BELIEF_UPDATING", "CONSERVATISM_UNDERREACTION_DRIFT")
    assert receipt.verify_integrity() is True


def test_calculate_bsv_sentiment_equilibrium_neutral() -> None:
    """Test neutral equilibrium regime when earnings surprise is negligible."""
    prices = [100.0 + 0.2 * math.sin(i) for i in range(25)]
    receipt = calculate_bsv_sentiment(
        ticker="FLAT",
        prices=prices,
        latest_surprise_pct=0.5,
        consecutive_streak=0,
    )

    assert receipt.ticker == "FLAT"
    assert receipt.regime == "EQUILIBRIUM_NEUTRAL_DRIFT"
    assert receipt.is_conservatism_drift is False
    assert receipt.is_representativeness_peak is False
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    prices = [100.0 + i * 0.2 for i in range(25)]
    receipt = calculate_bsv_sentiment(
        ticker="AUDIT",
        prices=prices,
        latest_surprise_pct=5.0,
        consecutive_streak=2,
    )

    assert receipt.verify_integrity() is True

    # Tamper with trending_state_probability
    receipt.trending_state_probability += 0.05
    assert receipt.verify_integrity() is False

    # Restore and tamper with expected_bsv_alpha_pct
    receipt.trending_state_probability -= 0.05
    assert receipt.verify_integrity() is True
    receipt.expected_bsv_alpha_pct += 1.8
    assert receipt.verify_integrity() is False


def test_classify_bsv_regime_boundaries() -> None:
    """Test boundary conditions across BSV sentiment regimes."""
    # High pi_t + streak >= 3 -> Peak
    assert classify_bsv_regime(0.70, 4, 6.0) == "REPRESENTATIVENESS_OVERREACTION_PEAK"
    # Low pi_t + significant surprise -> Underreaction drift
    assert classify_bsv_regime(0.35, 1, 5.0) == "CONSERVATISM_UNDERREACTION_DRIFT"
    # Intermediate pi_t -> Transitional
    assert classify_bsv_regime(0.50, 2, 4.0) == "TRANSITIONAL_BELIEF_UPDATING"
    # Negligible surprise -> Equilibrium Neutral
    assert classify_bsv_regime(0.30, 1, 0.5) == "EQUILIBRIUM_NEUTRAL_DRIFT"


def test_earnings_surprises_array_streak_parsing() -> None:
    """Test parsing a full array of historical quarterly surprises to determine streak length and direction."""
    prices = [100.0 + i * 0.3 for i in range(30)]
    # History: negative, then 3 consecutive positive quarters
    surprises = [-2.0, 3.5, 4.2, 5.8]

    receipt = calculate_bsv_sentiment(
        ticker="STRK",
        prices=prices,
        earnings_surprises=surprises,
    )

    assert receipt.ticker == "STRK"
    assert receipt.consecutive_streak_quarters == 3
    assert receipt.latest_earnings_surprise_pct == 5.8
    assert receipt.verify_integrity() is True


def test_evaluate_bsv_sentiment_profile_database(tmp_path: Path) -> None:
    """Test evaluation pulling real observations from a SQLite database."""
    test_db = tmp_path / "test_bsv.db"
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
    prices = [70.0 + i * 0.4 for i in range(35)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("BSVT", f"2026-05-{i+1:02d}", p, 1_400_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_bsv_sentiment_profile(
        ticker="BSVT",
        latest_surprise_pct=5.5,
        consecutive_streak=2,
        db_path=test_db,
    )

    assert receipt.ticker == "BSVT"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.trailing_momentum_pct > 0
    assert receipt.verify_integrity() is True


def test_evaluate_bsv_sentiment_profile_fallback() -> None:
    """Test evaluation deterministic fallback when ticker is not in database."""
    receipt = evaluate_bsv_sentiment_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert 0.05 <= receipt.trending_state_probability <= 0.95
    assert 0 <= receipt.consecutive_streak_quarters <= 12
    assert receipt.verify_integrity() is True


def test_cli_bsv_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI bsv command text and json output."""
    args_text = argparse.Namespace(
        ticker="MSFT",
        earnings_surprise=6.0,
        streak=2,
        json=False,
    )
    bsv_command(args_text)
    captured = capsys.readouterr()
    assert "Barberis, Shleifer & Vishny (1998) Dual-Heuristic Sentiment" in captured.out
    assert "MSFT" in captured.out
    assert "Trending State Prob" in captured.out

    args_json = argparse.Namespace(
        ticker="MSFT",
        earnings_surprise=6.0,
        streak=2,
        json=True,
    )
    bsv_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "MSFT"
    assert "trending_state_probability" in data
    assert "conservatism_drift_index" in data
    assert "signature" in data


def test_input_validation_bsv() -> None:
    """Test defensive security validation on tickers and input observations."""
    with pytest.raises(SecurityValidationError):
        calculate_bsv_sentiment(
            ticker="INVALID_TICKER_SYMBOL",
            prices=[100.0] * 25,
        )

    with pytest.raises(ValueError, match="requires at least 20 price observations"):
        calculate_bsv_sentiment(
            ticker="AAPL",
            prices=[100.0] * 12,
        )

    with pytest.raises(ValueError, match="strictly positive, finite numbers"):
        calculate_bsv_sentiment(
            ticker="AAPL",
            prices=[100.0, float("nan")] + [100.0] * 20,
        )

    with pytest.raises(ValueError, match="strictly positive, finite numbers"):
        calculate_bsv_sentiment(
            ticker="AAPL",
            prices=[-5.0] + [100.0] * 20,
        )

    with pytest.raises(ValueError, match="must be finite"):
        calculate_bsv_sentiment(
            ticker="AAPL",
            prices=[100.0] * 25,
            latest_surprise_pct=float("inf"),
        )
