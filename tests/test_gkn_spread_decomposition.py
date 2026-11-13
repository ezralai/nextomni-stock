"""Unit tests for Thomas J. George, Gautam Kaul & M. Nimalendran (1991) GKN Spread Decomposition."""

import sqlite3
import numpy as np
import pytest

from src.analytics.gkn_spread_decomposition import (
    GKNReceipt,
    calculate_gkn_spread,
    persist_gkn_record,
    verify_hmac_signature,
)


def test_gkn_synthetic_dgp_recovery():
    """Verify GKN recovers order processing component s0 from observed transaction and quote midpoints."""
    rng = np.random.default_rng(42)
    T = 500
    true_s0 = 0.0020  # 20 bps

    # Midpoint process
    u = rng.normal(0.0, 0.001, size=T)
    m = 100.0 * np.exp(np.cumsum(u))

    # Transitory trade bounces
    q = rng.choice([-1.0, 1.0], size=T)
    p = m * (1.0 + 0.5 * true_s0 * q)

    receipt = calculate_gkn_spread(
        transaction_prices=p,
        quote_midpoints=m,
        ticker="GKN_TEST",
    )

    assert isinstance(receipt, GKNReceipt)
    assert receipt.ticker == "GKN_TEST"
    assert receipt.sample_size == T
    assert receipt.derivation_mode == "OBSERVED_QUOTE_MIDPOINTS"
    # Order processing spread should accurately track ~20 bps (15 - 25 bps)
    assert 14.0 <= receipt.order_processing_spread_bps <= 26.0
    assert receipt.total_effective_spread_bps >= receipt.order_processing_spread_bps
    assert 0.0 <= receipt.order_processing_ratio <= 1.0

    # Cryptographic HMAC verification
    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.total_effective_spread_bps:.2f}|"
        f"{receipt.order_processing_spread_bps:.2f}|{receipt.adverse_selection_spread_bps:.2f}|"
        f"{receipt.order_processing_ratio:.4f}|{receipt.adverse_selection_ratio:.4f}|"
        f"{receipt.roll_spread_bps:.2f}|{receipt.expected_return_autocovariance:.8f}|"
        f"{receipt.roll_bias_bps:.2f}|{receipt.regime}|{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True


def test_gkn_expected_return_bias_purging():
    """Verify GKN purges time-varying expected return autocorrelation that contaminates Roll."""
    rng = np.random.default_rng(101)
    T = 600
    true_s0 = 0.0030  # 30 bps

    # Strongly persistent macroeconomic expected return process
    mu = np.zeros(T)
    for t in range(1, T):
        mu[t] = 0.85 * mu[t - 1] + rng.normal(0.0, 0.002)

    # Base fundamental price contaminated by time-varying drift
    innov = rng.normal(0.0, 0.001, size=T)
    m = 100.0 * np.exp(np.cumsum(mu + innov))

    # Trades
    q = rng.choice([-1.0, 1.0], size=T)
    p = m * (1.0 + 0.5 * true_s0 * q)

    receipt = calculate_gkn_spread(p, quote_midpoints=m, ticker="MACRO_DRIFT")

    # Positive expected return autocorrelation pushes Roll spread to 0.0 or underestimation
    assert receipt.expected_return_autocovariance > 0.0
    # GKN cleanly recovers positive order processing spread despite macro drift
    assert receipt.order_processing_spread_bps > 15.0
    # Roll bias should be positive (Roll underestimated spread)
    assert receipt.roll_bias_bps >= 0.0


def test_gkn_hmac_tamper_detection():
    """Verify cryptographic HMAC-SHA256 signature detects modifications."""
    prices = [100.0 + 0.1 * ((-1) ** i) for i in range(50)]
    receipt = calculate_gkn_spread(prices, ticker="AUTHENTIC")

    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.total_effective_spread_bps:.2f}|"
        f"{receipt.order_processing_spread_bps:.2f}|{receipt.adverse_selection_spread_bps:.2f}|"
        f"{receipt.order_processing_ratio:.4f}|{receipt.adverse_selection_ratio:.4f}|"
        f"{receipt.roll_spread_bps:.2f}|{receipt.expected_return_autocovariance:.8f}|"
        f"{receipt.roll_bias_bps:.2f}|{receipt.regime}|{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered payload fails
    tampered = payload.replace("AUTHENTIC", "FORGED")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_gkn_sqlite_persistence(tmp_path):
    """Verify SQLite database persistence and schema creation."""
    db_file = str(tmp_path / "test_gkn.db")
    prices = [100.0 + 0.05 * ((-1) ** i) for i in range(40)]

    receipt = calculate_gkn_spread(prices, ticker="PERSIST_TEST")
    assert persist_gkn_record(receipt, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, total_effective_spread_bps, order_processing_spread_bps, regime, hmac_signature FROM gkn_spread_receipts WHERE ticker = ?",
            ("PERSIST_TEST",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "PERSIST_TEST"
        assert abs(row[1] - receipt.total_effective_spread_bps) < 1e-4
        assert row[3] == receipt.regime
        assert row[4] == receipt.hmac_signature

