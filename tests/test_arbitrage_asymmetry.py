"""Unit and integration test suite for Robert F. Stambaugh, Jianfeng Yu & Yu Yuan (2012, 2015) Arbitrage Asymmetry Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.arbitrage_asymmetry import (
    ArbitrageAsymmetryReceipt,
    calculate_arbitrage_asymmetry,
    classify_arbitrage_regime,
    evaluate_arbitrage_asymmetry_profile,
)
from src.cli import arbitrage_asymmetry_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_arbitrage_asymmetry_severe_overpriced() -> None:
    """Test asset with severe short-sale impediments during euphoric sentiment wave."""
    receipt = calculate_arbitrage_asymmetry(
        ticker="BUBBLE",
        latest_price=55.0,
        market_sentiment_index=2.0,  # Euphoric market sentiment
        idiosyncratic_volatility_pct=60.0,  # High arbitrage risk
        borrow_fee_annual_pct=18.0,  # Prohibitive special borrow fee
        daily_turnover_pct=6.5,  # Retail churning
        short_interest_pct_float=14.0,  # High short interest
        days_to_cover=4.5,
        market_cap=2_000_000_000.0,
    )

    assert receipt.ticker == "BUBBLE"
    assert receipt.latest_price == 55.0
    assert receipt.market_sentiment_index == 2.0
    assert receipt.short_sale_impediment_index >= 70.0
    assert receipt.asymmetric_overpricing_hazard >= 45.0
    assert receipt.is_severe_overpriced is True
    assert receipt.regime == "SEVERE_SHORT_SALE_RESTRICTED_OVERPRICING"
    assert receipt.expected_asymmetry_alpha_pct < -5.0
    assert receipt.verify_integrity() is True


def test_calculate_arbitrage_asymmetry_squeeze_powder_keg() -> None:
    """Test short-squeeze powder keg asset with extreme short interest and days to cover."""
    receipt = calculate_arbitrage_asymmetry(
        ticker="SQUEEZE",
        latest_price=25.0,
        market_sentiment_index=0.5,
        idiosyncratic_volatility_pct=50.0,
        borrow_fee_annual_pct=22.0,  # High borrow fee
        daily_turnover_pct=4.0,
        short_interest_pct_float=25.0,  # Extreme short interest (>20%)
        days_to_cover=7.5,  # High days to cover
        market_cap=1_500_000_000.0,
    )

    assert receipt.short_interest_pct_float == 25.0
    assert receipt.short_squeeze_vulnerability >= 65.0
    assert receipt.is_squeeze_risk is True
    assert receipt.regime == "SHORT_SQUEEZE_POWDER_KEG"
    assert receipt.verify_integrity() is True


def test_calculate_arbitrage_asymmetry_unconstrained_symmetric() -> None:
    """Test institutional liquid compounder with unconstrained, symmetric arbitrage."""
    receipt = calculate_arbitrage_asymmetry(
        ticker="LIQUID",
        latest_price=220.0,
        market_sentiment_index=1.0,
        idiosyncratic_volatility_pct=14.0,  # Low arbitrage risk
        borrow_fee_annual_pct=0.35,  # General collateral borrow
        daily_turnover_pct=0.8,
        short_interest_pct_float=1.5,  # Low short interest
        days_to_cover=1.2,
        market_cap=150_000_000_000.0,
    )

    assert receipt.short_sale_impediment_index < 30.0
    assert receipt.asymmetric_overpricing_hazard < 15.0
    assert receipt.is_severe_overpriced is False
    assert receipt.is_squeeze_risk is False
    assert receipt.regime == "SYMMETRIC_UNCONSTRAINED_ARBITRAGE"
    assert receipt.verify_integrity() is True


def test_asymmetric_overpricing_sentiment_interaction() -> None:
    """Verify Stambaugh-Yu-Yuan core theorem: short impediments cause overpricing ONLY when sentiment is positive."""
    # Under identical high short impediments
    rec_euphoria = calculate_arbitrage_asymmetry(
        ticker="TEST",
        latest_price=50.0,
        market_sentiment_index=2.0,  # Positive sentiment
        idiosyncratic_volatility_pct=50.0,
        borrow_fee_annual_pct=12.0,
        short_interest_pct_float=12.0,
    )
    rec_panic = calculate_arbitrage_asymmetry(
        ticker="TEST",
        latest_price=50.0,
        market_sentiment_index=-2.0,  # Negative sentiment
        idiosyncratic_volatility_pct=50.0,
        borrow_fee_annual_pct=12.0,
        short_interest_pct_float=12.0,
    )

    # Short impediments are identical
    assert abs(rec_euphoria.short_sale_impediment_index - rec_panic.short_sale_impediment_index) < 1e-4

    # Overpricing hazard is high in euphoria, but zero in panic (unconstrained long arbitrage)
    assert rec_euphoria.asymmetric_overpricing_hazard > 35.0
    assert rec_panic.asymmetric_overpricing_hazard == 0.0
    assert rec_panic.expected_asymmetry_alpha_pct == 0.0


def test_impediment_monotonicity() -> None:
    """Test monotonic response to idiosyncratic volatility, borrow fees, short interest, and days to cover."""
    # 1. Idiosyncratic volatility sensitivity
    low_ivol = calculate_arbitrage_asymmetry(
        ticker="M1", latest_price=100.0, idiosyncratic_volatility_pct=15.0
    )
    high_ivol = calculate_arbitrage_asymmetry(
        ticker="M1", latest_price=100.0, idiosyncratic_volatility_pct=55.0
    )
    assert high_ivol.short_sale_impediment_index > low_ivol.short_sale_impediment_index

    # 2. Borrow fee sensitivity
    low_fee = calculate_arbitrage_asymmetry(
        ticker="M2", latest_price=100.0, borrow_fee_annual_pct=0.5
    )
    high_fee = calculate_arbitrage_asymmetry(
        ticker="M2", latest_price=100.0, borrow_fee_annual_pct=15.0
    )
    assert high_fee.short_sale_impediment_index > low_fee.short_sale_impediment_index
    assert high_fee.short_squeeze_vulnerability > low_fee.short_squeeze_vulnerability

    # 3. Short interest sensitivity
    low_si = calculate_arbitrage_asymmetry(
        ticker="M3", latest_price=100.0, short_interest_pct_float=2.0
    )
    high_si = calculate_arbitrage_asymmetry(
        ticker="M3", latest_price=100.0, short_interest_pct_float=18.0
    )
    assert high_si.short_sale_impediment_index > low_si.short_sale_impediment_index
    assert high_si.short_squeeze_vulnerability > low_si.short_squeeze_vulnerability


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    receipt = calculate_arbitrage_asymmetry(
        ticker="AUDIT",
        latest_price=100.0,
        market_sentiment_index=1.0,
        idiosyncratic_volatility_pct=35.0,
        borrow_fee_annual_pct=2.5,
    )
    assert receipt.verify_integrity() is True

    # Tamper with impediment index
    receipt.short_sale_impediment_index += 5.0
    assert receipt.verify_integrity() is False

    # Tamper with hazard
    receipt.short_sale_impediment_index -= 5.0
    assert receipt.verify_integrity() is True
    receipt.asymmetric_overpricing_hazard += 2.0
    assert receipt.verify_integrity() is False


def test_classify_arbitrage_regime_boundaries() -> None:
    """Test regime classification boundary branches."""
    assert classify_arbitrage_regime(70.0, 50.0, 70.0, 18.0) == "SHORT_SQUEEZE_POWDER_KEG"
    assert classify_arbitrage_regime(70.0, 50.0, 50.0, 10.0) == "SEVERE_SHORT_SALE_RESTRICTED_OVERPRICING"
    assert classify_arbitrage_regime(45.0, 30.0, 40.0, 6.0) == "MODERATE_ARBITRAGE_FRICTION"
    assert classify_arbitrage_regime(20.0, 10.0, 20.0, 3.0) == "SYMMETRIC_UNCONSTRAINED_ARBITRAGE"


def test_evaluate_arbitrage_asymmetry_profile_database(tmp_path: Path) -> None:
    """Test profile evaluation pulling price and volume observations from SQLite database."""
    test_db = tmp_path / "test_asym.db"
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
    prices = [100.0 * (1.0 + 0.015 * math.sin(i / 4.0)) for i in range(50)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("ASYM", f"2026-01-{i+1:02d}", p, 2_000_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_arbitrage_asymmetry_profile(
        ticker="ASYM",
        market_sentiment_index=1.2,
        borrow_fee_annual_pct=4.5,
        short_interest_pct_float=8.0,
        days_to_cover=3.5,
        db_path=test_db,
    )

    assert receipt.ticker == "ASYM"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.borrow_fee_annual_pct == 4.5
    assert receipt.short_interest_pct_float == 8.0
    assert receipt.verify_integrity() is True


def test_evaluate_arbitrage_asymmetry_profile_fallback() -> None:
    """Test profile evaluation fallback when ticker is not in database."""
    receipt = evaluate_arbitrage_asymmetry_profile(
        ticker="XYZ",
        market_sentiment_index=0.5,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert receipt.short_sale_impediment_index > 0
    assert receipt.verify_integrity() is True


def test_cli_arbitrage_asymmetry_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI nextomni arbitrage-asymmetry command text and json output."""
    args_text = argparse.Namespace(
        ticker="TSLA",
        market_sentiment=1.5,
        borrow_fee=6.5,
        short_interest=10.0,
        days_to_cover=4.0,
        json=False,
    )
    arbitrage_asymmetry_command(args_text)
    captured_text = capsys.readouterr().out
    assert "Stambaugh, Yu & Yuan (2012) Arbitrage Asymmetry Engine" in captured_text
    assert "TSLA" in captured_text

    args_json = argparse.Namespace(
        ticker="TSLA",
        market_sentiment=1.5,
        borrow_fee=6.5,
        short_interest=10.0,
        days_to_cover=4.0,
        json=True,
    )
    arbitrage_asymmetry_command(args_json)
    captured_json = capsys.readouterr().out
    parsed = json.loads(captured_json)
    assert parsed["ticker"] == "TSLA"
    assert "short_sale_impediment_index" in parsed
    assert "signature" in parsed


def test_input_validation_arbitrage_asymmetry() -> None:
    """Test defensive security validation on non-positive or negative input parameters."""
    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=-10.0)

    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=50.0, market_cap=-100.0)

    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=50.0, borrow_fee_annual_pct=-1.0)

    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=50.0, short_interest_pct_float=-5.0)

    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=50.0, days_to_cover=-2.0)

    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=50.0, daily_turnover_pct=-1.0)

    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=50.0, idiosyncratic_volatility_pct=-10.0)

    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=float("nan"))

    with pytest.raises(ValueError):
        calculate_arbitrage_asymmetry(ticker="TEST", latest_price=50.0, borrow_fee_annual_pct=float("inf"))

    with pytest.raises(SecurityValidationError):
        evaluate_arbitrage_asymmetry_profile(ticker="INVALID TICKER$$$")
