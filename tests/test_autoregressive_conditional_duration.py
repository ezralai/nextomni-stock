"""Unit tests for Robert F. Engle & Jeffrey R. Russell (1998) ACD Duration Engine."""

import sqlite3
import numpy as np
import pytest

from src.analytics.autoregressive_conditional_duration import (
    ACDReceipt,
    calculate_acd_model,
    evaluate_acd_from_prices,
    persist_acd_record,
    verify_hmac_signature,
)


def test_acd_synthetic_dgp_recovery():
    """Verify MLE parameter recovery under synthetic Engle-Russell ACD(1, 1) process."""
    rng = np.random.default_rng(42)
    N = 600
    true_omega = 0.20
    true_alpha = 0.12
    true_beta = 0.75
    # Target persistence: 0.87, unconditional duration: 0.20 / (1 - 0.87) ~ 1.54

    x = np.empty(N)
    psi = np.empty(N)
    psi[0] = true_omega / (1.0 - true_alpha - true_beta)
    eps = rng.exponential(scale=1.0, size=N)
    x[0] = psi[0] * eps[0]

    for i in range(1, N):
        psi[i] = true_omega + true_alpha * x[i - 1] + true_beta * psi[i - 1]
        x[i] = psi[i] * eps[i]

    receipt = calculate_acd_model(durations=x, ticker="ACD_DGP")

    assert isinstance(receipt, ACDReceipt)
    assert receipt.ticker == "ACD_DGP"
    assert receipt.sample_size == N
    assert receipt.mean_duration > 0.0

    # Parameter estimates within reasonable bounds
    assert 0.05 <= receipt.omega <= 0.60
    assert 0.04 <= receipt.alpha <= 0.25
    assert 0.55 <= receipt.beta <= 0.90
    assert 0.70 <= receipt.persistence < 1.00
    assert receipt.current_hazard_rate > 0.0
    assert receipt.half_life_shocks > 1.0

    # Cryptographic HMAC verification
    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.mean_duration:.4f}|{receipt.omega:.6f}|"
        f"{receipt.alpha:.6f}|{receipt.beta:.6f}|{receipt.persistence:.4f}|"
        f"{receipt.unconditional_expected_duration:.4f}|{receipt.current_conditional_duration:.4f}|"
        f"{receipt.current_hazard_rate:.4f}|{receipt.trading_speed_ratio:.4f}|"
        f"{receipt.half_life_shocks:.2f}|{receipt.log_likelihood:.2f}|{receipt.regime}|"
        f"{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True


def test_acd_clustering_and_speed_ratio():
    """Verify that a sudden burst of short trade arrivals triggers accelerated clustering regime."""
    rng = np.random.default_rng(101)
    N = 250
    # Normal baseline durations around 5.0 seconds
    x = rng.exponential(scale=5.0, size=N)
    # Sudden burst of rapid high-frequency orders at the tail
    x[-25:] = rng.exponential(scale=0.5, size=25)

    receipt = calculate_acd_model(durations=x, ticker="BURST_ACD")

    # Current conditional duration should be significantly shorter than mean
    assert receipt.current_conditional_duration < receipt.mean_duration
    assert receipt.trading_speed_ratio > 1.20
    assert receipt.regime in ("ACCELERATED_TRADING_CLUSTERING", "HIGH_DURATION_PERSISTENCE")


def test_acd_hmac_tamper_detection():
    """Verify cryptographic HMAC-SHA256 signature detects tampering."""
    durations = [2.0 + 0.5 * (i % 5) for i in range(40)]
    receipt = calculate_acd_model(durations, ticker="AUTHENTIC_ACD")

    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.mean_duration:.4f}|{receipt.omega:.6f}|"
        f"{receipt.alpha:.6f}|{receipt.beta:.6f}|{receipt.persistence:.4f}|"
        f"{receipt.unconditional_expected_duration:.4f}|{receipt.current_conditional_duration:.4f}|"
        f"{receipt.current_hazard_rate:.4f}|{receipt.trading_speed_ratio:.4f}|"
        f"{receipt.half_life_shocks:.2f}|{receipt.log_likelihood:.2f}|{receipt.regime}|"
        f"{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered payload fails
    tampered = payload.replace("AUTHENTIC_ACD", "FORGED_ACD")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_acd_sqlite_persistence(tmp_path):
    """Verify SQLite database persistence and schema creation."""
    db_file = str(tmp_path / "test_acd.db")
    prices = [100.0 + 0.2 * i for i in range(35)]

    receipt = evaluate_acd_from_prices(prices, ticker="PERSIST_ACD")
    assert persist_acd_record(receipt, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, mean_duration, persistence, regime, hmac_signature FROM acd_duration_receipts WHERE ticker = ?",
            ("PERSIST_ACD",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "PERSIST_ACD"
        assert abs(row[1] - receipt.mean_duration) < 1e-4
        assert row[3] == receipt.regime
        assert row[4] == receipt.hmac_signature

