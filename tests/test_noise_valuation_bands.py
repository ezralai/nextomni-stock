"""Unit, boundary, security, and integration tests for Black (1986)

Noise Valuation Bounds & Mean-Reversion Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.noise_valuation_bands import (
    NoiseValuationReceipt,
    calculate_noise_valuation_bands,
    classify_noise_regime,
    evaluate_noise_valuation_profile,
    persist_noise_valuation_record,
    verify_noise_valuation_audit_digest,
)
from src.cli import noise_bands_command
from src.security.validators import SecurityValidationError


def test_calculate_noise_valuation_bands_extreme_dislocation_over() -> None:
    """Verify market price exceeding 2.0x fundamental value triggers extreme dislocation."""
    receipt = calculate_noise_valuation_bands(
        ticker="BUBBLE",
        latest_price=220.0,
        fundamental_value=100.0,
        noise_to_information_ratio=1.50,
        mean_reversion_speed_kappa=0.35,
    )

    assert receipt.ticker == "BUBBLE"
    assert receipt.is_within_factor2_bounds is False
    assert receipt.is_extreme_dislocation is True
    assert receipt.regime == "EXTREME_NOISE_DISLOCATION"
    assert receipt.noise_deviation_pct == 120.0
    assert receipt.expected_reversion_alpha_pct < 0.0
    assert verify_noise_valuation_audit_digest(receipt) is True


def test_calculate_noise_valuation_bands_extreme_dislocation_under() -> None:
    """Verify market price below 0.50x fundamental value triggers extreme dislocation."""
    receipt = calculate_noise_valuation_bands(
        ticker="CRASH",
        latest_price=40.0,
        fundamental_value=100.0,
    )

    assert receipt.is_within_factor2_bounds is False
    assert receipt.is_extreme_dislocation is True
    assert receipt.regime == "EXTREME_NOISE_DISLOCATION"
    assert receipt.noise_deviation_pct == -60.0
    assert receipt.expected_reversion_alpha_pct > 0.0
    assert verify_noise_valuation_audit_digest(receipt) is True


def test_calculate_noise_valuation_bands_overvalued() -> None:
    """Verify price above inner corridor but inside factor-of-two envelope triggers overvaluation."""
    receipt = calculate_noise_valuation_bands(
        ticker="OVER",
        latest_price=150.0,
        fundamental_value=100.0,
    )

    assert receipt.is_within_factor2_bounds is True
    assert receipt.is_overvalued is True
    assert receipt.regime == "NOISE_DRIVEN_OVERVALUATION"
    assert receipt.noise_deviation_pct == 50.0
    assert receipt.expected_reversion_alpha_pct < 0.0
    assert verify_noise_valuation_audit_digest(receipt) is True


def test_calculate_noise_valuation_bands_undervalued() -> None:
    """Verify price below inner corridor triggers deep value undervaluation."""
    receipt = calculate_noise_valuation_bands(
        ticker="UNDER",
        latest_price=70.0,
        fundamental_value=100.0,
    )

    assert receipt.is_within_factor2_bounds is True
    assert receipt.is_undervalued is True
    assert receipt.regime == "NOISE_DRIVEN_UNDERVALUATION"
    assert receipt.noise_deviation_pct == -30.0
    assert receipt.expected_reversion_alpha_pct > 0.0
    assert verify_noise_valuation_audit_digest(receipt) is True


def test_calculate_noise_valuation_bands_equilibrium_corridor() -> None:
    """Verify price within 80%-125% of P* triggers equilibrium corridor."""
    receipt = calculate_noise_valuation_bands(
        ticker="EQUIL",
        latest_price=105.0,
        fundamental_value=100.0,
    )

    assert receipt.is_within_factor2_bounds is True
    assert receipt.is_equilibrium_band is True
    assert receipt.regime == "FUNDAMENTAL_EQUILIBRIUM_BAND"
    assert abs(receipt.noise_deviation_pct) <= 25.0
    assert verify_noise_valuation_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify higher price strictly increases deviation and decreases reversion alpha."""
    prices = [60.0, 80.0, 100.0, 120.0, 140.0, 180.0]
    devs = []
    alphas = []

    for p in prices:
        rec = calculate_noise_valuation_bands(ticker="MONO", latest_price=p, fundamental_value=100.0)
        devs.append(rec.noise_deviation_pct)
        alphas.append(rec.expected_reversion_alpha_pct)

    assert all(devs[i] < devs[i + 1] for i in range(len(devs) - 1))
    assert all(alphas[i] > alphas[i + 1] for i in range(len(alphas) - 1))


def test_tamper_signature_verification() -> None:
    """Verify cryptographic audit digest detects tampered records."""
    receipt = calculate_noise_valuation_bands(ticker="AUDIT", latest_price=110.0, fundamental_value=100.0)
    assert verify_noise_valuation_audit_digest(receipt) is True

    tampered = NoiseValuationReceipt(
        ticker=receipt.ticker,
        latest_price=receipt.latest_price,
        fundamental_value=receipt.fundamental_value,
        lower_factor2_bound=receipt.lower_factor2_bound,
        upper_factor2_bound=receipt.upper_factor2_bound,
        inner_lower_bound=receipt.inner_lower_bound,
        inner_upper_bound=receipt.inner_upper_bound,
        noise_deviation_pct=receipt.noise_deviation_pct + 10.0,  # Tampered!
        noise_to_information_ratio=receipt.noise_to_information_ratio,
        information_share_pct=receipt.information_share_pct,
        mean_reversion_speed_kappa=receipt.mean_reversion_speed_kappa,
        expected_reversion_alpha_pct=receipt.expected_reversion_alpha_pct,
        is_within_factor2_bounds=receipt.is_within_factor2_bounds,
        regime=receipt.regime,
        is_extreme_dislocation=receipt.is_extreme_dislocation,
        is_overvalued=receipt.is_overvalued,
        is_undervalued=receipt.is_undervalued,
        is_equilibrium_band=receipt.is_equilibrium_band,
        summary=receipt.summary,
        audit_digest=receipt.audit_digest,
        calculated_at=receipt.calculated_at,
    )

    assert verify_noise_valuation_audit_digest(tampered) is False


def test_security_validation_ticker_and_bounds() -> None:
    """Verify security validation rejects invalid tickers and parameter bounds."""
    with pytest.raises(SecurityValidationError):
        calculate_noise_valuation_bands(ticker="INVALID_TOOLONG")

    with pytest.raises(SecurityValidationError):
        calculate_noise_valuation_bands(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_noise_valuation_bands(ticker="AAPL", fundamental_value=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_noise_valuation_bands(ticker="AAPL", noise_to_information_ratio=0.0)

    with pytest.raises(SecurityValidationError):
        calculate_noise_valuation_bands(ticker="AAPL", mean_reversion_speed_kappa=0.0)


def test_evaluate_noise_valuation_profile_synthetic_observations() -> None:
    """Verify profile evaluation with synthetic prices."""
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.0003, 0.015, 60)
    prices = [float(base_price * np.exp(np.sum(returns[:i]))) for i in range(1, 61)]

    receipt = evaluate_noise_valuation_profile(
        ticker="SYNTH",
        prices=prices,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.fundamental_value > 0.0
    assert receipt.noise_to_information_ratio > 0.0
    assert verify_noise_valuation_audit_digest(receipt) is True


def test_evaluate_noise_valuation_profile_overrides() -> None:
    """Verify profile evaluation honors explicit overrides."""
    receipt = evaluate_noise_valuation_profile(
        ticker="OVRD",
        fundamental_value_override=120.0,
        noise_ratio_override=0.85,
    )

    assert receipt.ticker == "OVRD"
    assert receipt.fundamental_value == 120.0
    assert receipt.noise_to_information_ratio == 0.85


def test_cli_noise_bands_human_readable(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs human-readable table."""
    args = argparse.Namespace(
        ticker="AAPL",
        fundamental_value=150.0,
        noise_ratio=0.50,
        kappa=0.35,
        json=False,
    )

    noise_bands_command(args)
    captured = capsys.readouterr()

    assert "NextOmni Stock: Black (1986) Noise Valuation Bounds Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Black (1986) Factor-of-Two Efficiency Envelopes:" in captured.out
    assert "Expected Mean-Reversion Alpha:" in captured.out
    assert "Audit Digest (HMAC):" in captured.out


def test_cli_noise_bands_json_output(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs valid JSON."""
    args = argparse.Namespace(
        ticker="GOOGL",
        fundamental_value=175.0,
        noise_ratio=0.40,
        kappa=0.35,
        json=True,
    )

    noise_bands_command(args)
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert data["ticker"] == "GOOGL"
    assert "fundamental_value" in data
    assert "lower_factor2_bound" in data
    assert "expected_reversion_alpha_pct" in data
    assert "audit_digest" in data


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify SQLite persistence and parameterized retrieval."""
    test_db = tmp_path / "test_noise.db"

    receipt = calculate_noise_valuation_bands(
        ticker="NVDA",
        latest_price=135.0,
        fundamental_value=100.0,
    )

    persist_noise_valuation_record(receipt, db_path=test_db)

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, fundamental_value, noise_deviation_pct, audit_digest FROM noise_valuation_records WHERE ticker = ?",
        ("NVDA",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "NVDA"
    assert row[1] == 135.0
    assert row[2] == 100.0
    assert row[4] == receipt.audit_digest

