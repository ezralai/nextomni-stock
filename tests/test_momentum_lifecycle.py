"""Unit and integration tests for Lee & Swaminathan (2000) Price Momentum and Trading Volume Life-Cycle Engine."""

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.analytics.momentum_lifecycle import (
    MomentumLifeCycleReceipt,
    calculate_momentum_lifecycle,
    classify_lifecycle_stage,
    evaluate_momentum_lifecycle_profile,
)
from src.cli import lifecycle_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_momentum_lifecycle_stage1_neglected_winner() -> None:
    """Test Stage 1 Neglected Winner: high past return with low trading volume, delivering prolonged alpha."""
    # Strong upward trend (+25%) with low daily volume (0.8% turnover)
    prices = [100.0 * (1.0 + 0.007 * i) for i in range(35)]
    shares = 60_000_000.0
    # 0.8% turnover -> 480,000 shares/day
    volumes = [480_000.0 for _ in range(35)]

    receipt = calculate_momentum_lifecycle(
        ticker="WIN1",
        prices=prices,
        volumes=volumes,
        shares_outstanding=shares,
    )

    assert receipt.ticker == "WIN1"
    assert receipt.trailing_momentum_pct >= 12.0
    assert receipt.daily_turnover_ratio_pct <= 1.4
    assert receipt.is_neglected_winner is True
    assert receipt.is_crowded_winner is False
    assert receipt.regime == "STAGE_1_EARLY_NEGLECTED_WINNER"
    assert receipt.expected_mlc_alpha_pct >= 8.0  # Superior momentum persistence
    assert receipt.verify_integrity() is True


def test_calculate_momentum_lifecycle_stage2_crowded_winner() -> None:
    """Test Stage 2 Crowded Winner: high past return with heavy speculative volume, facing exhaustion reversal."""
    # Strong upward trend (+30%) with surging volume (3.0% turnover)
    prices = [100.0 * (1.0 + 0.008 * i) for i in range(35)]
    shares = 60_000_000.0
    # 3.0% turnover -> 1,800,000 shares/day
    volumes = [1_800_000.0 for _ in range(35)]

    receipt = calculate_momentum_lifecycle(
        ticker="WIN2",
        prices=prices,
        volumes=volumes,
        shares_outstanding=shares,
    )

    assert receipt.ticker == "WIN2"
    assert receipt.trailing_momentum_pct >= 12.0
    assert receipt.daily_turnover_ratio_pct >= 2.2
    assert receipt.is_crowded_winner is True
    assert receipt.is_neglected_winner is False
    assert receipt.regime == "STAGE_2_LATE_CROWDED_WINNER"
    assert receipt.expected_mlc_alpha_pct < 0.0  # Momentum crash / exhaustion drag
    assert receipt.verify_integrity() is True


def test_calculate_momentum_lifecycle_stage3_capitulation_loser() -> None:
    """Test Stage 3 Capitulation Loser: deep negative return with climactic volume, triggering sharp rebound."""
    # Deep downward trend (-25%) with heavy panic turnover (2.8% turnover)
    prices = [100.0 * (1.0 - 0.008 * i) for i in range(35)]
    shares = 60_000_000.0
    volumes = [1_680_000.0 for _ in range(35)]

    receipt = calculate_momentum_lifecycle(
        ticker="LOSE3",
        prices=prices,
        volumes=volumes,
        shares_outstanding=shares,
    )

    assert receipt.ticker == "LOSE3"
    assert receipt.trailing_momentum_pct <= -12.0
    assert receipt.daily_turnover_ratio_pct >= 2.2
    assert receipt.is_capitulation_loser is True
    assert receipt.is_neglected_loser is False
    assert receipt.regime == "STAGE_3_CAPITULATION_LOSER"
    assert receipt.expected_mlc_alpha_pct > 0.0  # Rebound mean-reversion alpha
    assert receipt.verify_integrity() is True


def test_calculate_momentum_lifecycle_stage4_neglected_zombie() -> None:
    """Test Stage 4 Neglected Zombie: negative return with negligible volume, continuing persistent downward drift."""
    # Deep downward trend (-22%) with dry volume (0.7% turnover)
    prices = [100.0 * (1.0 - 0.007 * i) for i in range(35)]
    shares = 60_000_000.0
    volumes = [420_000.0 for _ in range(35)]

    receipt = calculate_momentum_lifecycle(
        ticker="LOSE4",
        prices=prices,
        volumes=volumes,
        shares_outstanding=shares,
    )

    assert receipt.ticker == "LOSE4"
    assert receipt.trailing_momentum_pct <= -12.0
    assert receipt.daily_turnover_ratio_pct <= 1.4
    assert receipt.is_neglected_loser is True
    assert receipt.is_capitulation_loser is False
    assert receipt.regime == "STAGE_4_NEGLECTED_ZOMBIE_LOSER"
    assert receipt.expected_mlc_alpha_pct < 0.0  # Continued zombie underperformance
    assert receipt.verify_integrity() is True


def test_calculate_momentum_lifecycle_transitional_equilibrium() -> None:
    """Test transitional equilibrium with moderate momentum and turnover."""
    prices = [100.0 * (1.0 + 0.001 * i) for i in range(30)]
    shares = 60_000_000.0
    volumes = [1_000_000.0 for _ in range(30)]

    receipt = calculate_momentum_lifecycle(
        ticker="TRAN",
        prices=prices,
        volumes=volumes,
        shares_outstanding=shares,
    )

    assert receipt.ticker == "TRAN"
    assert receipt.regime == "TRANSITIONAL_EQUILIBRIUM"
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    prices = [100.0 + i * 0.3 for i in range(25)]
    receipt = calculate_momentum_lifecycle(
        ticker="AUDIT",
        prices=prices,
    )

    assert receipt.verify_integrity() is True

    # Tamper with trailing_momentum_pct
    receipt.trailing_momentum_pct += 2.0
    assert receipt.verify_integrity() is False

    # Restore and tamper with expected_mlc_alpha_pct
    receipt.trailing_momentum_pct -= 2.0
    assert receipt.verify_integrity() is True
    receipt.expected_mlc_alpha_pct += 1.5
    assert receipt.verify_integrity() is False


def test_classify_lifecycle_stage_boundaries() -> None:
    """Test boundary conditions across Lee-Swaminathan Momentum Life Cycle stages."""
    # Winner + Low Volume -> Stage 1
    assert classify_lifecycle_stage(15.0, 1.2) == "STAGE_1_EARLY_NEGLECTED_WINNER"
    # Winner + High Volume -> Stage 2
    assert classify_lifecycle_stage(15.0, 2.5) == "STAGE_2_LATE_CROWDED_WINNER"
    # Loser + High Volume -> Stage 3
    assert classify_lifecycle_stage(-15.0, 2.5) == "STAGE_3_CAPITULATION_LOSER"
    # Loser + Low Volume -> Stage 4
    assert classify_lifecycle_stage(-15.0, 1.2) == "STAGE_4_NEGLECTED_ZOMBIE_LOSER"
    # Intermediate -> Transitional
    assert classify_lifecycle_stage(5.0, 1.8) == "TRANSITIONAL_EQUILIBRIUM"


def test_evaluate_momentum_lifecycle_profile_database(tmp_path: Path) -> None:
    """Test evaluation pulling real observations from a SQLite database."""
    test_db = tmp_path / "test_mlc.db"
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
    prices = [80.0 + i * 0.6 for i in range(35)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("MLCT", f"2026-07-{i+1:02d}", p, 600_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_momentum_lifecycle_profile(
        ticker="MLCT",
        db_path=test_db,
    )

    assert receipt.ticker == "MLCT"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.trailing_momentum_pct > 0
    assert receipt.verify_integrity() is True


def test_evaluate_momentum_lifecycle_profile_fallback() -> None:
    """Test evaluation deterministic fallback when ticker is not in database."""
    receipt = evaluate_momentum_lifecycle_profile(
        ticker="XYZ",
        turnover_ratio_override=1.1,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert abs(receipt.daily_turnover_ratio_pct - 1.1) < 0.05
    assert receipt.verify_integrity() is True


def test_cli_lifecycle_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI lifecycle command text and json output."""
    args_text = argparse.Namespace(
        ticker="TSLA",
        turnover=1.0,
        json=False,
    )
    lifecycle_command(args_text)
    captured = capsys.readouterr()
    assert "Lee & Swaminathan (2000) Momentum Life-Cycle Engine" in captured.out
    assert "TSLA" in captured.out
    assert "Trailing 12-M Momentum" in captured.out

    args_json = argparse.Namespace(
        ticker="TSLA",
        turnover=1.0,
        json=True,
    )
    lifecycle_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "TSLA"
    assert "trailing_momentum_pct" in data
    assert "volume_momentum_interaction_score" in data
    assert "signature" in data


def test_input_validation_lifecycle() -> None:
    """Test defensive security validation on tickers and input observations."""
    with pytest.raises(SecurityValidationError):
        calculate_momentum_lifecycle(
            ticker="INVALID_TICKER_SYMBOL",
            prices=[100.0] * 25,
        )

    with pytest.raises(ValueError, match="requires at least 20 price observations"):
        calculate_momentum_lifecycle(
            ticker="AAPL",
            prices=[100.0] * 12,
        )

    with pytest.raises(ValueError, match="strictly positive, finite numbers"):
        calculate_momentum_lifecycle(
            ticker="AAPL",
            prices=[100.0, float("nan")] + [100.0] * 20,
        )

    with pytest.raises(ValueError, match="strictly positive, finite numbers"):
        calculate_momentum_lifecycle(
            ticker="AAPL",
            prices=[-5.0] + [100.0] * 20,
        )
