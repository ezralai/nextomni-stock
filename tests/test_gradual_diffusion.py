"""Unit and integration tests for Hong & Stein (1999) Gradual Information Diffusion Engine."""

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.analytics.gradual_diffusion import (
    GradualDiffusionReceipt,
    calculate_gradual_information_diffusion,
    classify_diffusion_regime,
    evaluate_gradual_diffusion_profile,
)
from src.cli import diffusion_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_gradual_diffusion_early_underreaction() -> None:
    """Test early underreaction phase where information is diffusing and momentum continues."""
    # Gradual, steady upward drift over 30 days (+8% cumulative)
    prices = [100.0 * (1.0 + 0.0025 * i) for i in range(30)]
    volumes = [1_000_000.0 for _ in range(30)]

    receipt = calculate_gradual_information_diffusion(
        ticker="UNDR",
        prices=prices,
        volumes=volumes,
        market_cap=5_000_000_000.0,
        analyst_coverage=10,
    )

    assert receipt.ticker == "UNDR"
    assert receipt.latest_price > 100.0
    assert receipt.cumulative_momentum_pct > 5.0
    assert receipt.trend_maturity_index < 0.45
    assert receipt.is_early_underreaction is True
    assert receipt.is_crowded_overreaction is False
    assert receipt.regime == "EARLY_UNDERREACTION_EXPANSION"
    assert receipt.expected_hong_stein_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_gradual_diffusion_crowded_overreaction_peak() -> None:
    """Test crowded overreaction peak where trend has gone parabolic with surging volume."""
    # Parabolic upward run-up (+75%) with accelerating recent return
    base = 100.0
    prices = [base * (1.0 + 0.005 * i + 0.0003 * (i**2)) for i in range(50)]
    # Volume expands by 3x near the end
    volumes = [1_000_000.0 if i < 35 else 3_000_000.0 for i in range(50)]

    receipt = calculate_gradual_information_diffusion(
        ticker="CROWD",
        prices=prices,
        volumes=volumes,
        market_cap=50_000_000_000.0,
        analyst_coverage=35,
    )

    assert receipt.ticker == "CROWD"
    assert receipt.cumulative_momentum_pct > 25.0
    assert receipt.trend_maturity_index >= 0.65
    assert receipt.is_crowded_overreaction is True
    assert receipt.regime == "CROWDED_OVERREACTION_PEAK"
    assert receipt.momentum_crowding_risk_pct > 50.0
    assert receipt.expected_hong_stein_alpha_pct < 0.0  # High reversal/crash drag
    assert receipt.verify_integrity() is True


def test_calculate_gradual_diffusion_mature_convergence() -> None:
    """Test mature momentum convergence regime with balanced diffusion."""
    prices = [100.0 * (1.0 + 0.002 * i) for i in range(40)]
    receipt = calculate_gradual_information_diffusion(
        ticker="MATR",
        prices=prices,
        market_cap=10_000_000_000.0,
    )

    assert receipt.ticker == "MATR"
    assert receipt.regime in ("MATURE_MOMENTUM_CONVERGENCE", "EARLY_UNDERREACTION_EXPANSION")
    assert receipt.verify_integrity() is True


def test_calculate_gradual_diffusion_stagnant_equilibrium() -> None:
    """Test stagnant equilibrium regime with near-zero price momentum."""
    # Random oscillation around 100.0 with < 1.0% net drift
    prices = [100.0 + 0.5 * math.sin(i / 2.0) for i in range(40)]

    receipt = calculate_gradual_information_diffusion(
        ticker="FLAT",
        prices=prices,
    )

    assert receipt.ticker == "FLAT"
    assert abs(receipt.cumulative_momentum_pct) < 4.0
    assert receipt.regime == "STAGNANT_INFORMATION_EQUILIBRIUM"
    assert receipt.is_early_underreaction is False
    assert receipt.is_crowded_overreaction is False
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    prices = [100.0 + i * 0.2 for i in range(25)]
    receipt = calculate_gradual_information_diffusion(
        ticker="AUDIT",
        prices=prices,
    )

    assert receipt.verify_integrity() is True

    # Tamper with trend_maturity_index
    receipt.trend_maturity_index += 0.1
    assert receipt.verify_integrity() is False

    # Restore and tamper with expected_hong_stein_alpha_pct
    receipt.trend_maturity_index -= 0.1
    assert receipt.verify_integrity() is True
    receipt.expected_hong_stein_alpha_pct += 2.0
    assert receipt.verify_integrity() is False


def test_classify_diffusion_regime_boundaries() -> None:
    """Test regime classification across various boundary conditions."""
    # High momentum, high maturity -> CROWDED_OVERREACTION_PEAK
    assert classify_diffusion_regime(0.6, 0.75, 18.0) == "CROWDED_OVERREACTION_PEAK"
    # High momentum, low maturity -> EARLY_UNDERREACTION_EXPANSION
    assert classify_diffusion_regime(0.4, 0.30, 8.0) == "EARLY_UNDERREACTION_EXPANSION"
    # Moderate momentum, intermediate maturity -> MATURE_MOMENTUM_CONVERGENCE
    assert classify_diffusion_regime(0.5, 0.55, 6.0) == "MATURE_MOMENTUM_CONVERGENCE"
    # Low momentum -> STAGNANT_INFORMATION_EQUILIBRIUM
    assert classify_diffusion_regime(0.5, 0.20, 1.5) == "STAGNANT_INFORMATION_EQUILIBRIUM"


def test_analyst_coverage_scaling() -> None:
    """Test that higher analyst coverage accelerates information diffusion velocity (IDV)."""
    prices = [100.0 + i * 0.3 for i in range(30)]

    rec_low = calculate_gradual_information_diffusion(
        ticker="COV",
        prices=prices,
        market_cap=5_000_000_000.0,
        analyst_coverage=2,
    )
    rec_high = calculate_gradual_information_diffusion(
        ticker="COV",
        prices=prices,
        market_cap=5_000_000_000.0,
        analyst_coverage=35,
    )

    assert rec_high.information_diffusion_velocity > rec_low.information_diffusion_velocity


def test_evaluate_gradual_diffusion_profile_database(tmp_path: Path) -> None:
    """Test evaluation pulling real observations from a SQLite database."""
    test_db = tmp_path / "test_diffusion.db"
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
    prices = [50.0 + i * 0.4 for i in range(45)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("DIFF", f"2026-02-{i+1:02d}", p, 1_500_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_gradual_diffusion_profile(
        ticker="DIFF",
        db_path=test_db,
    )

    assert receipt.ticker == "DIFF"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.cumulative_momentum_pct > 0
    assert receipt.verify_integrity() is True


def test_evaluate_gradual_diffusion_profile_fallback() -> None:
    """Test evaluation deterministic fallback when ticker is not in database."""
    receipt = evaluate_gradual_diffusion_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert 0.05 <= receipt.information_diffusion_velocity <= 0.95
    assert 0.0 <= receipt.trend_maturity_index <= 1.0
    assert receipt.verify_integrity() is True


def test_cli_diffusion_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI diffusion command text and json output."""
    args_text = argparse.Namespace(
        ticker="NVDA",
        analyst_coverage=20,
        json=False,
    )
    diffusion_command(args_text)
    captured = capsys.readouterr()
    assert "Hong & Stein (1999) Gradual Information Diffusion Engine" in captured.out
    assert "NVDA" in captured.out
    assert "Information Diffusion (IDV)" in captured.out

    args_json = argparse.Namespace(
        ticker="NVDA",
        analyst_coverage=20,
        json=True,
    )
    diffusion_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "NVDA"
    assert "information_diffusion_velocity" in data
    assert "signature" in data


def test_input_validation_diffusion() -> None:
    """Test defensive security validation on tickers and input observations."""
    with pytest.raises(SecurityValidationError):
        calculate_gradual_information_diffusion(
            ticker="INVALID_TICKER_SYMBOL",
            prices=[100.0] * 25,
        )

    with pytest.raises(ValueError, match="requires at least 20 price observations"):
        calculate_gradual_information_diffusion(
            ticker="AAPL",
            prices=[100.0] * 10,
        )

    with pytest.raises(ValueError, match="strictly positive, finite numbers"):
        calculate_gradual_information_diffusion(
            ticker="AAPL",
            prices=[100.0, float("nan")] + [100.0] * 20,
        )

    with pytest.raises(ValueError, match="strictly positive, finite numbers"):
        calculate_gradual_information_diffusion(
            ticker="AAPL",
            prices=[-10.0] + [100.0] * 20,
        )
