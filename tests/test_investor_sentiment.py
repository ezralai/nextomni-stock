"""Unit and integration test suite for Malcolm Baker & Jeffrey Wurgler (2006, 2007) Investor Sentiment Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.investor_sentiment import (
    InvestorSentimentReceipt,
    calculate_investor_sentiment,
    classify_sentiment_regime,
    evaluate_investor_sentiment_profile,
)
from src.cli import sentiment_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_investor_sentiment_speculative_bubble() -> None:
    """Test speculative bubble asset: hard to value, hard to arbitrage, euphoric sentiment."""
    receipt = calculate_investor_sentiment(
        ticker="MEME",
        latest_price=45.0,
        market_sentiment_index=1.5,  # Euphoric retail wave
        annualized_volatility_pct=65.0,  # High volatility
        net_profit_margin_pct=-15.0,  # Unprofitable
        market_cap=500_000_000.0,  # $500M Small cap
        max_daily_return_5d_pct=9.5,  # High lottery appeal
    )

    assert receipt.ticker == "MEME"
    assert receipt.latest_price == 45.0
    assert receipt.market_sentiment_index == 1.5
    assert receipt.speculative_index >= 65.0
    assert receipt.sentiment_beta > 0.8
    assert receipt.is_speculative_bubble is True
    assert receipt.is_contrarian_bargain is False
    assert receipt.regime == "SPECULATIVE_BUBBLE_OVERPRICED"
    # Overpriced in euphoria -> negative subsequent alpha drag
    assert receipt.expected_sentiment_alpha_pct < -5.0
    assert receipt.verify_integrity() is True


def test_calculate_investor_sentiment_contrarian_bargain() -> None:
    """Test speculative turnaround asset during depressed market sentiment panic."""
    receipt = calculate_investor_sentiment(
        ticker="TURN",
        latest_price=12.0,
        market_sentiment_index=-1.8,  # Depressed sentiment panic
        annualized_volatility_pct=70.0,
        net_profit_margin_pct=-10.0,
        market_cap=300_000_000.0,
        max_daily_return_5d_pct=8.5,
    )

    assert receipt.speculative_index >= 65.0
    assert receipt.sentiment_beta > 0.8
    assert receipt.is_contrarian_bargain is True
    assert receipt.is_speculative_bubble is False
    assert receipt.regime == "CONTRARIAN_BARGAIN_OPPORTUNITY"
    # Undervalued in panic -> strong positive rebound alpha
    assert receipt.expected_sentiment_alpha_pct > +5.0
    assert receipt.verify_integrity() is True


def test_calculate_investor_sentiment_bond_like_haven() -> None:
    """Test high-profitability, large-cap, low-volatility compounder insulated from sentiment."""
    receipt = calculate_investor_sentiment(
        ticker="JNJ",
        latest_price=180.0,
        market_sentiment_index=1.5,  # Even in euphoria, safe stocks remain anchored
        annualized_volatility_pct=14.0,  # Low volatility
        net_profit_margin_pct=28.0,  # Robust profitability
        market_cap=200_000_000_000.0,  # $200B Mega cap
        max_daily_return_5d_pct=1.8,  # Low lottery spikes
    )

    assert receipt.speculative_index <= 35.0
    assert receipt.sentiment_beta < 0.0
    assert receipt.is_speculative_bubble is False
    assert receipt.is_contrarian_bargain is False
    assert receipt.regime == "BOND_LIKE_STABLE_HAVEN"
    assert receipt.verify_integrity() is True


def test_calculate_investor_sentiment_moderate() -> None:
    """Test average equity profile falling in moderate sentiment exposure regime."""
    receipt = calculate_investor_sentiment(
        ticker="BAL",
        latest_price=85.0,
        market_sentiment_index=0.0,
        annualized_volatility_pct=25.0,
        net_profit_margin_pct=12.0,
        market_cap=15_000_000_000.0,
        max_daily_return_5d_pct=3.0,
    )

    assert 35.0 < receipt.speculative_index < 65.0
    assert receipt.regime == "MODERATE_SENTIMENT_EXPOSURE"
    assert receipt.is_speculative_bubble is False
    assert receipt.is_contrarian_bargain is False
    assert receipt.verify_integrity() is True


def test_speculative_index_monotonicity() -> None:
    """Verify Baker & Wurgler sensitivity monotonicity across volatility, profitability, size, and lottery."""
    # 1. Volatility sensitivity
    low_vol = calculate_investor_sentiment(
        ticker="T1", latest_price=100.0, annualized_volatility_pct=15.0
    )
    high_vol = calculate_investor_sentiment(
        ticker="T1", latest_price=100.0, annualized_volatility_pct=55.0
    )
    assert high_vol.speculative_index > low_vol.speculative_index
    assert high_vol.sentiment_beta > low_vol.sentiment_beta

    # 2. Profit margin sensitivity
    high_margin = calculate_investor_sentiment(
        ticker="T2", latest_price=100.0, net_profit_margin_pct=25.0
    )
    unprofitable = calculate_investor_sentiment(
        ticker="T2", latest_price=100.0, net_profit_margin_pct=-5.0
    )
    assert unprofitable.speculative_index > high_margin.speculative_index

    # 3. Market cap sensitivity
    mega_cap = calculate_investor_sentiment(
        ticker="T3", latest_price=100.0, market_cap=100_000_000_000.0
    )
    micro_cap = calculate_investor_sentiment(
        ticker="T3", latest_price=100.0, market_cap=50_000_000.0
    )
    assert micro_cap.speculative_index > mega_cap.speculative_index

    # 4. Lottery MAX_5 sensitivity
    low_lottery = calculate_investor_sentiment(
        ticker="T4", latest_price=100.0, max_daily_return_5d_pct=2.0
    )
    high_lottery = calculate_investor_sentiment(
        ticker="T4", latest_price=100.0, max_daily_return_5d_pct=8.0
    )
    assert high_lottery.speculative_index > low_lottery.speculative_index


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    receipt = calculate_investor_sentiment(
        ticker="AUDIT",
        latest_price=100.0,
        market_sentiment_index=1.0,
        annualized_volatility_pct=30.0,
        net_profit_margin_pct=10.0,
        market_cap=5_000_000_000.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with speculative index
    receipt.speculative_index += 5.0
    assert receipt.verify_integrity() is False

    # Tamper with regime
    receipt.speculative_index -= 5.0
    assert receipt.verify_integrity() is True
    receipt.regime = "TAMPERED_REGIME"
    assert receipt.verify_integrity() is False


def test_classify_sentiment_regime_boundaries() -> None:
    """Test regime boundary logic."""
    assert classify_sentiment_regime(70.0, 1.0) == "SPECULATIVE_BUBBLE_OVERPRICED"
    assert classify_sentiment_regime(65.0, 0.75) == "SPECULATIVE_BUBBLE_OVERPRICED"
    assert classify_sentiment_regime(70.0, -1.0) == "CONTRARIAN_BARGAIN_OPPORTUNITY"
    assert classify_sentiment_regime(65.0, -0.75) == "CONTRARIAN_BARGAIN_OPPORTUNITY"
    assert classify_sentiment_regime(30.0, 1.5) == "BOND_LIKE_STABLE_HAVEN"
    assert classify_sentiment_regime(35.0, -1.5) == "BOND_LIKE_STABLE_HAVEN"
    assert classify_sentiment_regime(50.0, 0.5) == "MODERATE_SENTIMENT_EXPOSURE"


def test_evaluate_investor_sentiment_profile_database(tmp_path: Path) -> None:
    """Test profile evaluation pulling price observations from SQLite database."""
    test_db = tmp_path / "test_sentiment.db"
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
    # Generate 50 realistic synthetic prices
    prices = [100.0 * (1.0 + 0.01 * math.sin(i / 5.0)) for i in range(50)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("SENT", f"2026-01-{i+1:02d}", p, 1_000_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_investor_sentiment_profile(
        ticker="SENT",
        market_sentiment_index=0.8,
        net_profit_margin_pct=15.0,
        market_cap=25_000_000_000.0,
        db_path=test_db,
    )

    assert receipt.ticker == "SENT"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.verify_integrity() is True


def test_evaluate_investor_sentiment_profile_fallback() -> None:
    """Test profile evaluation fallback when ticker is not in database."""
    receipt = evaluate_investor_sentiment_profile(
        ticker="XYZ",
        market_sentiment_index=0.5,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert receipt.speculative_index > 0
    assert receipt.verify_integrity() is True


def test_cli_sentiment_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI nextomni sentiment command text and json output."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        market_sentiment=1.2,
        net_margin=22.0,
        market_cap=3_000_000_000_000.0,
        json=False,
    )
    sentiment_command(args_text)
    captured_text = capsys.readouterr().out
    assert "Baker & Wurgler (2006) Investor Sentiment Engine" in captured_text
    assert "AAPL" in captured_text

    args_json = argparse.Namespace(
        ticker="AAPL",
        market_sentiment=1.2,
        net_margin=22.0,
        market_cap=3_000_000_000_000.0,
        json=True,
    )
    sentiment_command(args_json)
    captured_json = capsys.readouterr().out
    parsed = json.loads(captured_json)
    assert parsed["ticker"] == "AAPL"
    assert "speculative_index" in parsed
    assert "signature" in parsed


def test_input_validation_sentiment() -> None:
    """Test defensive validation for non-finite and illegal inputs."""
    with pytest.raises(ValueError):
        calculate_investor_sentiment(ticker="TEST", latest_price=-10.0)

    with pytest.raises(ValueError):
        calculate_investor_sentiment(ticker="TEST", latest_price=100.0, market_cap=-500.0)

    with pytest.raises(ValueError):
        calculate_investor_sentiment(ticker="TEST", latest_price=float("nan"))

    with pytest.raises(ValueError):
        calculate_investor_sentiment(ticker="TEST", latest_price=100.0, annualized_volatility_pct=float("inf"))

    with pytest.raises(SecurityValidationError):
        evaluate_investor_sentiment_profile(ticker="INVALID TICKER$$$")
