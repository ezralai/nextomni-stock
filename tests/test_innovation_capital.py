"""Unit and integration tests for Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Engine."""

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.analytics.innovation_capital import (
    InnovationCapitalReceipt,
    calculate_innovation_capital,
    classify_innovation_regime,
    evaluate_innovation_capital_profile,
)
from src.cli import innovation_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_innovation_capital_high_conviction() -> None:
    """Test high conviction innovation alpha with substantial R&D intensity and knowledge accumulation."""
    receipt = calculate_innovation_capital(
        ticker="INNO",
        latest_price=100.0,
        current_rd_expense=450_000_000.0,
        annual_revenues=2_500_000_000.0,
        total_assets=3_500_000_000.0,
        market_cap=5_000_000_000.0,
    )

    assert receipt.ticker == "INNO"
    assert receipt.rd_to_market_pct == 9.0
    assert receipt.rd_to_sales_pct == 18.0
    assert receipt.innovation_misvaluation_index >= 0.58
    assert receipt.is_high_innovation_alpha is True
    assert receipt.is_innovation_laggard is False
    assert receipt.regime == "HIGH_CONVICTION_INNOVATION_ALPHA"
    assert receipt.expected_innovation_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_innovation_capital_disciplined_compounder() -> None:
    """Test disciplined R&D compounder with moderate research reinvestment."""
    receipt = calculate_innovation_capital(
        ticker="COMP",
        latest_price=80.0,
        current_rd_expense=120_000_000.0,
        annual_revenues=3_000_000_000.0,
        total_assets=4_000_000_000.0,
        market_cap=6_000_000_000.0,
    )

    assert receipt.ticker == "COMP"
    assert receipt.rd_to_market_pct == 2.0
    assert 0.25 <= receipt.innovation_misvaluation_index < 0.58
    assert receipt.regime == "DISCIPLINED_RND_COMPOUNDER"
    assert receipt.verify_integrity() is True


def test_calculate_innovation_capital_laggard() -> None:
    """Test innovation-starved laggard with minimal R&D facing technological obsolescence."""
    receipt = calculate_innovation_capital(
        ticker="LAGG",
        latest_price=40.0,
        current_rd_expense=5_000_000.0,
        annual_revenues=2_000_000_000.0,
        total_assets=3_000_000_000.0,
        market_cap=4_000_000_000.0,
    )

    assert receipt.ticker == "LAGG"
    assert receipt.rd_to_market_pct < 0.50
    assert receipt.innovation_misvaluation_index <= 0.15
    assert receipt.is_innovation_laggard is True
    assert receipt.is_high_innovation_alpha is False
    assert receipt.regime == "INNOVATION_STARVED_LAGGARD"
    assert receipt.expected_innovation_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_innovation_capital_zero_rnd() -> None:
    """Test asset-light / traditional company with zero reported R&D expenditures."""
    receipt = calculate_innovation_capital(
        ticker="ZERO",
        latest_price=50.0,
        current_rd_expense=0.0,
        annual_revenues=1_000_000_000.0,
        total_assets=2_000_000_000.0,
        market_cap=3_000_000_000.0,
    )

    assert receipt.ticker == "ZERO"
    assert receipt.current_rd_expense == 0.0
    assert receipt.capitalized_knowledge_stock == 0.0
    assert receipt.innovation_misvaluation_index == 0.0
    assert receipt.regime == "CAPITAL_LIGHT_NON_RND"
    assert receipt.expected_innovation_alpha_pct == 0.0
    assert receipt.verify_integrity() is True


def test_perpetual_inventory_knowledge_accumulation() -> None:
    """Test perpetual inventory accumulation with 20% annual economic depreciation weights."""
    current_rd = 100_000_000.0
    # Prior years: t-1=90M, t-2=80M, t-3=70M, t-4=60M
    prior_rds = [90_000_000.0, 80_000_000.0, 70_000_000.0, 60_000_000.0]

    # Expected K_t = 100M + 0.8*90M + 0.6*80M + 0.4*70M + 0.2*60M
    # = 100M + 72M + 48M + 28M + 12M = 260M
    expected_k = 260_000_000.0

    receipt = calculate_innovation_capital(
        ticker="PERP",
        latest_price=100.0,
        current_rd_expense=current_rd,
        prior_rd_expenses=prior_rds,
    )

    assert abs(receipt.capitalized_knowledge_stock - expected_k) < 1e-4
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    receipt = calculate_innovation_capital(
        ticker="AUDIT",
        latest_price=100.0,
        current_rd_expense=200_000_000.0,
    )

    assert receipt.verify_integrity() is True

    # Tamper with capitalized_knowledge_stock
    receipt.capitalized_knowledge_stock += 10_000_000.0
    assert receipt.verify_integrity() is False

    # Restore and tamper with expected_innovation_alpha_pct
    receipt.capitalized_knowledge_stock -= 10_000_000.0
    assert receipt.verify_integrity() is True
    receipt.expected_innovation_alpha_pct += 1.5
    assert receipt.verify_integrity() is False


def test_classify_innovation_regime_boundaries() -> None:
    """Test boundary conditions across innovation regimes."""
    # Zero R&D -> Non R&D
    assert classify_innovation_regime(0.0, 0.0, 0.0) == "CAPITAL_LIGHT_NON_RND"
    # High IMI + High RD/M -> High Conviction Alpha
    assert classify_innovation_regime(0.65, 4.5, 300_000_000.0) == "HIGH_CONVICTION_INNOVATION_ALPHA"
    # Moderate IMI -> Disciplined Compounder
    assert classify_innovation_regime(0.40, 2.0, 100_000_000.0) == "DISCIPLINED_RND_COMPOUNDER"
    # Low IMI -> Innovation-Starved Laggard
    assert classify_innovation_regime(0.10, 0.3, 5_000_000.0) == "INNOVATION_STARVED_LAGGARD"


def test_evaluate_innovation_capital_profile_database(tmp_path: Path) -> None:
    """Test evaluation pulling real observations from a SQLite database."""
    test_db = tmp_path / "test_inno.db"
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
    prices = [120.0 + i * 0.5 for i in range(30)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("INNT", f"2026-06-{i+1:02d}", p, 2_000_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_innovation_capital_profile(
        ticker="INNT",
        current_rd_expense=300_000_000.0,
        db_path=test_db,
    )

    assert receipt.ticker == "INNT"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.current_rd_expense == 300_000_000.0
    assert receipt.verify_integrity() is True


def test_evaluate_innovation_capital_profile_fallback() -> None:
    """Test evaluation deterministic fallback when ticker is not in database."""
    receipt = evaluate_innovation_capital_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert receipt.current_rd_expense > 0
    assert 0.0 <= receipt.innovation_misvaluation_index <= 1.0
    assert receipt.verify_integrity() is True


def test_cli_innovation_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI innovation command text and json output."""
    args_text = argparse.Namespace(
        ticker="BMY",
        rd_expense=4_000_000_000.0,
        revenues=45_000_000_000.0,
        total_assets=90_000_000_000.0,
        json=False,
    )
    innovation_command(args_text)
    captured = capsys.readouterr()
    assert "Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Engine" in captured.out
    assert "BMY" in captured.out
    assert "R&D-to-Market Ratio (RD/M)" in captured.out

    args_json = argparse.Namespace(
        ticker="BMY",
        rd_expense=4_000_000_000.0,
        revenues=45_000_000_000.0,
        total_assets=90_000_000_000.0,
        json=True,
    )
    innovation_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "BMY"
    assert "current_rd_expense" in data
    assert "capitalized_knowledge_stock" in data
    assert "signature" in data


def test_input_validation_innovation() -> None:
    """Test defensive security validation on tickers and input parameters."""
    with pytest.raises(SecurityValidationError):
        calculate_innovation_capital(
            ticker="INVALID_TICKER_SYMBOL",
            latest_price=100.0,
            current_rd_expense=50_000_000.0,
        )

    with pytest.raises(ValueError, match="Latest price must be strictly positive"):
        calculate_innovation_capital(
            ticker="AAPL",
            latest_price=-10.0,
            current_rd_expense=50_000_000.0,
        )

    with pytest.raises(ValueError, match="R&D expense cannot be negative"):
        calculate_innovation_capital(
            ticker="AAPL",
            latest_price=100.0,
            current_rd_expense=-5_000_000.0,
        )

    with pytest.raises(ValueError, match="Historical R&D cannot be negative"):
        calculate_innovation_capital(
            ticker="AAPL",
            latest_price=100.0,
            current_rd_expense=50_000_000.0,
            prior_rd_expenses=[40_000_000.0, -10_000_000.0],
        )
