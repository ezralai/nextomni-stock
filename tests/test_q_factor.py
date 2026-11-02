"""Unit and integration test suite for Kewei Hou, Chen Xue & Lu Zhang (2015, 2020) q-Factor Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.q_factor import (
    QFactorReceipt,
    calculate_q_factor,
    classify_q_factor_regime,
    evaluate_q_factor_profile,
)
from src.cli import q_factor_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_q_factor_disciplined_compounder() -> None:
    """Test high-ROE, low-investment disciplined corporate compounder."""
    receipt = calculate_q_factor(
        ticker="COMPOUND",
        latest_price=160.0,
        return_on_equity_pct=26.0,  # High profitability
        investment_to_assets_pct=4.5,  # Conservative investment
        tobin_q_proxy=2.4,
        cash_flow_to_assets_pct=18.0,
        market_cap=50_000_000_000.0,
    )

    assert receipt.ticker == "COMPOUND"
    assert receipt.latest_price == 160.0
    assert receipt.return_on_equity_pct == 26.0
    assert receipt.profitability_score >= 75.0
    assert receipt.investment_score <= 35.0
    assert receipt.q_alpha_score >= 70.0
    assert receipt.is_disciplined_compounder is True
    assert receipt.is_overinvestor_trap is False
    assert receipt.regime == "DISCIPLINED_HIGH_ROE_COMPOUNDER"
    assert receipt.expected_q_alpha_pct > +3.0
    assert receipt.verify_integrity() is True


def test_calculate_q_factor_overinvestor_trap() -> None:
    """Test low-ROE, aggressive asset expansion over-investment trap."""
    receipt = calculate_q_factor(
        ticker="OVEREXPAND",
        latest_price=15.0,
        return_on_equity_pct=-8.0,  # Negative profitability
        investment_to_assets_pct=22.0,  # Aggressive asset growth
        tobin_q_proxy=0.85,
        cash_flow_to_assets_pct=-4.0,
        market_cap=800_000_000.0,
    )

    assert receipt.profitability_score <= 20.0
    assert receipt.investment_score >= 75.0
    assert receipt.q_alpha_score <= 30.0
    assert receipt.is_overinvestor_trap is True
    assert receipt.is_disciplined_compounder is False
    assert receipt.regime == "CAPITAL_DESTRUCTIVE_OVERINVESTOR"
    assert receipt.expected_q_alpha_pct < -3.0
    assert receipt.verify_integrity() is True


def test_calculate_q_factor_cash_cow() -> None:
    """Test mature cash cow franchise with solid profitability and minimal reinvestment."""
    receipt = calculate_q_factor(
        ticker="CASHCOW",
        latest_price=75.0,
        return_on_equity_pct=16.0,
        investment_to_assets_pct=1.5,
        tobin_q_proxy=1.4,
    )

    assert receipt.profitability_score >= 50.0
    assert receipt.regime == "CASH_COW_MATURE_HARVESTER"
    assert receipt.verify_integrity() is True


def test_calculate_q_factor_growth_capital() -> None:
    """Test high-profitability growth firm aggressively deploying capital."""
    receipt = calculate_q_factor(
        ticker="GROWTH",
        latest_price=95.0,
        return_on_equity_pct=25.0,
        investment_to_assets_pct=18.0,
        tobin_q_proxy=3.2,
    )

    assert receipt.profitability_score >= 70.0
    assert receipt.regime == "GROWTH_CAPITAL_ALLOCATOR"
    assert receipt.verify_integrity() is True


def test_q_factor_monotonicity() -> None:
    """Verify Hou-Xue-Zhang monotonicity: higher ROE increases alpha, higher investment penalizes alpha."""
    base_price = 100.0

    # 1. Monotonicity with respect to ROE (profitability)
    low_roe = calculate_q_factor(
        ticker="T1", latest_price=base_price, return_on_equity_pct=5.0, investment_to_assets_pct=6.0
    )
    high_roe = calculate_q_factor(
        ticker="T1", latest_price=base_price, return_on_equity_pct=25.0, investment_to_assets_pct=6.0
    )
    assert high_roe.profitability_score > low_roe.profitability_score
    assert high_roe.q_alpha_score > low_roe.q_alpha_score
    assert high_roe.expected_q_alpha_pct > low_roe.expected_q_alpha_pct

    # 2. Monotonicity with respect to Investment / Asset Growth (corporate over-investment penalty)
    conservative_inv = calculate_q_factor(
        ticker="T2", latest_price=base_price, return_on_equity_pct=15.0, investment_to_assets_pct=3.0
    )
    aggressive_inv = calculate_q_factor(
        ticker="T2", latest_price=base_price, return_on_equity_pct=15.0, investment_to_assets_pct=25.0
    )
    assert aggressive_inv.investment_score > conservative_inv.investment_score
    # Conservative investment produces higher q-alpha score
    assert conservative_inv.q_alpha_score > aggressive_inv.q_alpha_score
    assert conservative_inv.expected_q_alpha_pct > aggressive_inv.expected_q_alpha_pct


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    receipt = calculate_q_factor(
        ticker="AUDIT",
        latest_price=100.0,
        return_on_equity_pct=18.0,
        investment_to_assets_pct=5.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with ROE
    receipt.return_on_equity_pct += 5.0
    assert receipt.verify_integrity() is False

    # Tamper with q-alpha score
    receipt.return_on_equity_pct -= 5.0
    assert receipt.verify_integrity() is True
    receipt.q_alpha_score += 1.0
    assert receipt.verify_integrity() is False


def test_classify_q_factor_regime_boundaries() -> None:
    """Test regime boundary classification."""
    assert classify_q_factor_regime(70.0, 30.0, 20.0, 5.0) == "DISCIPLINED_HIGH_ROE_COMPOUNDER"
    assert classify_q_factor_regime(25.0, 75.0, -5.0, 18.0) == "CAPITAL_DESTRUCTIVE_OVERINVESTOR"
    assert classify_q_factor_regime(50.0, 20.0, 14.0, 2.0) == "CASH_COW_MATURE_HARVESTER"
    assert classify_q_factor_regime(70.0, 70.0, 22.0, 18.0) == "GROWTH_CAPITAL_ALLOCATOR"
    assert classify_q_factor_regime(40.0, 50.0, 10.0, 10.0) == "MODERATE_INVESTMENT_PROFILE"


def test_evaluate_q_factor_profile_database(tmp_path: Path) -> None:
    """Test profile evaluation pulling observations from SQLite database."""
    test_db = tmp_path / "test_qfactor.db"
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
    prices = [100.0 * (1.0 + 0.01 * math.sin(i / 6.0)) for i in range(50)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("QSTK", f"2026-01-{i+1:02d}", p, 1_500_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_q_factor_profile(
        ticker="QSTK",
        return_on_equity_pct=20.0,
        investment_to_assets_pct=5.0,
        tobin_q_proxy=1.9,
        db_path=test_db,
    )

    assert receipt.ticker == "QSTK"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.return_on_equity_pct == 20.0
    assert receipt.investment_to_assets_pct == 5.0
    assert receipt.verify_integrity() is True


def test_evaluate_q_factor_profile_fallback() -> None:
    """Test profile evaluation fallback when ticker is not in database."""
    receipt = evaluate_q_factor_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert receipt.profitability_score > 0
    assert receipt.verify_integrity() is True


def test_cli_q_factor_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI nextomni q-factor command text and json output."""
    args_text = argparse.Namespace(
        ticker="MSFT",
        roe=32.0,
        investment_rate=5.0,
        asset_growth=None,
        tobin_q=3.5,
        json=False,
    )
    q_factor_command(args_text)
    captured_text = capsys.readouterr().out
    assert "Hou, Xue & Zhang (2015) q-Factor Investment Engine" in captured_text
    assert "MSFT" in captured_text

    args_json = argparse.Namespace(
        ticker="MSFT",
        roe=32.0,
        investment_rate=5.0,
        asset_growth=None,
        tobin_q=3.5,
        json=True,
    )
    q_factor_command(args_json)
    captured_json = capsys.readouterr().out
    parsed = json.loads(captured_json)
    assert parsed["ticker"] == "MSFT"
    assert "q_alpha_score" in parsed
    assert "signature" in parsed


def test_input_validation_q_factor() -> None:
    """Test defensive security validation on non-positive or invalid input parameters."""
    with pytest.raises(ValueError):
        calculate_q_factor(ticker="TEST", latest_price=-5.0)

    with pytest.raises(ValueError):
        calculate_q_factor(ticker="TEST", latest_price=50.0, market_cap=-1000.0)

    with pytest.raises(ValueError):
        calculate_q_factor(ticker="TEST", latest_price=50.0, tobin_q_proxy=0.0)

    with pytest.raises(ValueError):
        calculate_q_factor(ticker="TEST", latest_price=float("nan"))

    with pytest.raises(ValueError):
        calculate_q_factor(ticker="TEST", latest_price=50.0, return_on_equity_pct=float("inf"))

    with pytest.raises(SecurityValidationError):
        evaluate_q_factor_profile(ticker="INVALID TICKER$$$")
