"""Unit and integration tests for Daniel, Hirshleifer & Subrahmanyam (1998) DHS Behavioral Engine."""

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.analytics.dhs_overconfidence import (
    DHSOverconfidenceReceipt,
    calculate_dhs_overconfidence,
    classify_dhs_regime,
    evaluate_dhs_overconfidence_profile,
)
from src.cli import dhs_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_dhs_overconfidence_self_attribution_acceleration() -> None:
    """Test self-attribution acceleration where confirmatory public news amplifies momentum."""
    # Strong upward trend with elevated volatility (+18% momentum)
    prices = [100.0 * (1.0 + 0.006 * i + 0.02 * math.sin(i)) for i in range(30)]
    volumes = [2_500_000.0 for _ in range(30)]

    receipt = calculate_dhs_overconfidence(
        ticker="ACCEL",
        prices=prices,
        volumes=volumes,
        earnings_surprise_pct=5.5,  # Confirms the upward price trend
        analyst_dispersion_pct=18.0,
        market_cap=15_000_000_000.0,
    )

    assert receipt.ticker == "ACCEL"
    assert receipt.trailing_momentum_pct > 6.0
    assert receipt.overconfidence_intensity >= 0.55
    assert receipt.biased_self_attribution_index >= 0.15
    assert receipt.is_attribution_accelerating is True
    assert receipt.is_contradiction_fragile is False
    assert receipt.regime == "SELF_ATTRIBUTION_ACCELERATION"
    assert receipt.verify_integrity() is True


def test_calculate_dhs_overconfidence_contradiction_fragility() -> None:
    """Test contradiction fragility where public news contradicts an overconfident trend."""
    prices = [100.0 * (1.0 + 0.005 * i + 0.025 * math.sin(i)) for i in range(35)]
    volumes = [2_500_000.0 for _ in range(35)]

    receipt = calculate_dhs_overconfidence(
        ticker="FRAG",
        prices=prices,
        volumes=volumes,
        earnings_surprise_pct=-6.0,  # Negative surprise contradicts upward trend
        analyst_dispersion_pct=20.0,
    )

    assert receipt.ticker == "FRAG"
    assert receipt.trailing_momentum_pct > 6.0
    assert receipt.overconfidence_intensity >= 0.55
    assert receipt.biased_self_attribution_index <= -0.15
    assert receipt.is_contradiction_fragile is True
    assert receipt.is_attribution_accelerating is False
    assert receipt.regime == "ACCUMULATED_CONTRADICTION_FRAGILITY"
    assert receipt.reversal_crash_hazard_pct > 30.0
    assert receipt.verify_integrity() is True


def test_calculate_dhs_overconfidence_corrective_reversal() -> None:
    """Test long-horizon corrective reversal phase after parabolic overreaction."""
    # Extended parabolic surge (+45%) with extreme volatility and turnover
    prices = [100.0 * (1.0 + 0.012 * i + 0.03 * math.sin(i)) for i in range(40)]
    volumes = [5_000_000.0 for _ in range(40)]

    receipt = calculate_dhs_overconfidence(
        ticker="REV",
        prices=prices,
        volumes=volumes,
        analyst_dispersion_pct=28.0,
        earnings_surprise_pct=-4.0,
    )

    assert receipt.ticker == "REV"
    assert receipt.trailing_momentum_pct >= 22.0
    assert receipt.overconfidence_intensity >= 0.70
    assert receipt.regime == "LONG_HORIZON_CORRECTIVE_REVERSAL"
    assert receipt.reversal_crash_hazard_pct > 40.0
    assert receipt.verify_integrity() is True


def test_calculate_dhs_overconfidence_inception() -> None:
    """Test initial overconfidence inception phase with modest volatility and developing trend."""
    prices = [100.0 * (1.0 + 0.002 * i) for i in range(30)]
    volumes = [800_000.0 for _ in range(30)]

    receipt = calculate_dhs_overconfidence(
        ticker="INCP",
        prices=prices,
        volumes=volumes,
        earnings_surprise_pct=1.0,
        analyst_dispersion_pct=6.0,
    )

    assert receipt.ticker == "INCP"
    assert receipt.overconfidence_intensity < 0.55
    assert receipt.regime in ("INITIAL_OVERCONFIDENCE_INCEPTION", "NEUTRAL_PSYCHOLOGICAL_EQUILIBRIUM")
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 audit receipt integrity and tamper detection."""
    prices = [100.0 + i * 0.3 for i in range(25)]
    receipt = calculate_dhs_overconfidence(
        ticker="AUDIT",
        prices=prices,
    )

    assert receipt.verify_integrity() is True

    # Tamper with overconfidence_intensity
    receipt.overconfidence_intensity += 0.05
    assert receipt.verify_integrity() is False

    # Restore and tamper with expected_dhs_alpha_pct
    receipt.overconfidence_intensity -= 0.05
    assert receipt.verify_integrity() is True
    receipt.expected_dhs_alpha_pct += 2.0
    assert receipt.verify_integrity() is False


def test_classify_dhs_regime_boundaries() -> None:
    """Test boundary conditions across psychological life-cycle regimes."""
    # Extended trend + extreme OCI -> Corrective Reversal
    assert classify_dhs_regime(0.75, 0.10, 25.0) == "LONG_HORIZON_CORRECTIVE_REVERSAL"
    # High OCI + positive confirmation -> Acceleration
    assert classify_dhs_regime(0.60, 0.30, 10.0) == "SELF_ATTRIBUTION_ACCELERATION"
    # High OCI + negative contradiction -> Fragility
    assert classify_dhs_regime(0.60, -0.30, 10.0) == "ACCUMULATED_CONTRADICTION_FRAGILITY"
    # Moderate OCI + developing trend -> Inception
    assert classify_dhs_regime(0.40, 0.05, 5.0) == "INITIAL_OVERCONFIDENCE_INCEPTION"
    # Low momentum -> Neutral Equilibrium
    assert classify_dhs_regime(0.40, 0.0, 1.5) == "NEUTRAL_PSYCHOLOGICAL_EQUILIBRIUM"


def test_confirmatory_earnings_surprise_amplification() -> None:
    """Test that positive confirmatory earnings surprise increases self-attribution index and alpha."""
    prices = [100.0 + i * 0.4 for i in range(30)]

    rec_neg = calculate_dhs_overconfidence(
        ticker="AMPL",
        prices=prices,
        earnings_surprise_pct=-5.0,
    )
    rec_pos = calculate_dhs_overconfidence(
        ticker="AMPL",
        prices=prices,
        earnings_surprise_pct=5.0,
    )

    assert rec_pos.biased_self_attribution_index > rec_neg.biased_self_attribution_index
    assert rec_pos.expected_dhs_alpha_pct > rec_neg.expected_dhs_alpha_pct


def test_evaluate_dhs_overconfidence_profile_database(tmp_path: Path) -> None:
    """Test evaluation pulling real observations from a SQLite database."""
    test_db = tmp_path / "test_dhs.db"
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
    prices = [60.0 + i * 0.5 for i in range(40)]
    for i, p in enumerate(prices):
        cursor.execute(
            """
            INSERT INTO stock_observations (ticker, observation_time, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            ("DHST", f"2026-04-{i+1:02d}", p, 1_800_000),
        )
    conn.commit()
    conn.close()

    receipt = evaluate_dhs_overconfidence_profile(
        ticker="DHST",
        earnings_surprise_pct=4.0,
        db_path=test_db,
    )

    assert receipt.ticker == "DHST"
    assert abs(receipt.latest_price - prices[-1]) < 1e-4
    assert receipt.trailing_momentum_pct > 0
    assert receipt.verify_integrity() is True


def test_evaluate_dhs_overconfidence_profile_fallback() -> None:
    """Test evaluation deterministic fallback when ticker is not in database."""
    receipt = evaluate_dhs_overconfidence_profile(
        ticker="XYZ",
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert 0.05 <= receipt.overconfidence_intensity <= 0.95
    assert -1.0 <= receipt.biased_self_attribution_index <= 1.0
    assert receipt.verify_integrity() is True


def test_cli_dhs_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI dhs command text and json output."""
    args_text = argparse.Namespace(
        ticker="GOOG",
        earnings_surprise=5.0,
        dispersion=10.0,
        json=False,
    )
    dhs_command(args_text)
    captured = capsys.readouterr()
    assert "Daniel, Hirshleifer & Subrahmanyam (1998) DHS Behavioral Engine" in captured.out
    assert "GOOG" in captured.out
    assert "Overconfidence Intensity" in captured.out

    args_json = argparse.Namespace(
        ticker="GOOG",
        earnings_surprise=5.0,
        dispersion=10.0,
        json=True,
    )
    dhs_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "GOOG"
    assert "overconfidence_intensity" in data
    assert "biased_self_attribution_index" in data
    assert "signature" in data


def test_input_validation_dhs() -> None:
    """Test defensive security validation on tickers and input observations."""
    with pytest.raises(SecurityValidationError):
        calculate_dhs_overconfidence(
            ticker="INVALID_TICKER_SYMBOL",
            prices=[100.0] * 25,
        )

    with pytest.raises(ValueError, match="requires at least 20 price observations"):
        calculate_dhs_overconfidence(
            ticker="AAPL",
            prices=[100.0] * 12,
        )

    with pytest.raises(ValueError, match="strictly positive, finite numbers"):
        calculate_dhs_overconfidence(
            ticker="AAPL",
            prices=[100.0, float("nan")] + [100.0] * 20,
        )

    with pytest.raises(ValueError, match="strictly positive, finite numbers"):
        calculate_dhs_overconfidence(
            ticker="AAPL",
            prices=[-5.0] + [100.0] * 20,
        )
