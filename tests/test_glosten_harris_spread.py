"""Unit tests for Lawrence R. Glosten & Lawrence E. Harris (1988) Spread Decomposition."""

import sqlite3
import numpy as np
import pytest

from src.analytics.glosten_harris_spread import (
    GlostenHarrisReceipt,
    calculate_glosten_harris_spread,
    evaluate_glosten_harris_from_prices,
    persist_glosten_harris_record,
    verify_hmac_signature,
)


def test_glosten_harris_synthetic_dgp_recovery():
    """Verify OLS parameter recovery of fixed & variable spread components under true DGP."""
    rng = np.random.default_rng(42)
    T = 600
    true_c0 = 0.06
    true_c1 = 0.02
    true_z0 = 0.04
    true_z1 = 0.03

    # Generate trade indicators and trade volumes
    q = rng.choice([-1.0, 1.0], size=T)
    v = rng.uniform(0.5, 3.0, size=T)

    # Regressors
    dq = q[1:] - q[:-1]
    dqv = (q[1:] * v[1:]) - (q[:-1] * v[:-1])
    q_curr = q[1:]
    qv_curr = q[1:] * v[1:]

    # Price change: Delta P_t = c0 * dq + c1 * dqv + z0 * q + z1 * qv + noise
    noise = rng.normal(0.0, 0.005, size=T - 1)
    dp = true_c0 * dq + true_c1 * dqv + true_z0 * q_curr + true_z1 * qv_curr + noise

    receipt = calculate_glosten_harris_spread(
        price_changes=dp,
        trade_indicators=q,
        trade_volumes=v,
        ticker="GH_TEST",
        reference_price=100.0,
        benchmark_volume=1.5,
    )

    assert isinstance(receipt, GlostenHarrisReceipt)
    assert receipt.ticker == "GH_TEST"
    assert receipt.sample_size == T - 1
    assert receipt.benchmark_volume == 1.5

    # Parameter estimates within reasonable empirical bounds
    assert 0.03 <= receipt.c0_fixed_transitory <= 0.09
    assert 0.008 <= receipt.c1_variable_transitory <= 0.04
    assert 0.02 <= receipt.z0_fixed_adverse <= 0.07
    assert 0.015 <= receipt.z1_variable_adverse <= 0.05

    # Spread metrics
    assert receipt.effective_spread_dollar > 0.0
    assert receipt.effective_spread_bps > 0.0
    assert 0.0 < receipt.adverse_selection_ratio < 1.0
    assert 0.0 < receipt.order_processing_ratio < 1.0
    assert receipt.r_squared > 0.70

    # Cryptographic HMAC verification
    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.benchmark_volume:.2f}|"
        f"{receipt.c0_fixed_transitory:.6f}|{receipt.c1_variable_transitory:.6f}|"
        f"{receipt.z0_fixed_adverse:.6f}|{receipt.z1_variable_adverse:.6f}|"
        f"{receipt.effective_spread_dollar:.4f}|{receipt.effective_spread_bps:.2f}|"
        f"{receipt.adverse_selection_ratio:.4f}|{receipt.order_processing_ratio:.4f}|"
        f"{receipt.r_squared:.4f}|{receipt.regime}|{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True


def test_glosten_harris_volume_monotonicity():
    """Verify that effective spread increases monotonically with trade size V."""
    rng = np.random.default_rng(888)
    T = 300
    q = rng.choice([-1.0, 1.0], size=T)
    v = rng.uniform(1.0, 5.0, size=T)

    dq = q[1:] - q[:-1]
    dqv = (q[1:] * v[1:]) - (q[:-1] * v[:-1])
    dp = 0.05 * dq + 0.02 * dqv + 0.03 * q[1:] + 0.02 * (q[1:] * v[1:]) + rng.normal(0, 0.01, size=T - 1)

    rec_small = calculate_glosten_harris_spread(dp, q, v, ticker="MONOTONE", benchmark_volume=1.0)
    rec_large = calculate_glosten_harris_spread(dp, q, v, ticker="MONOTONE", benchmark_volume=4.0)

    # Larger order size incurs higher spread and adverse selection
    assert rec_large.effective_spread_dollar > rec_small.effective_spread_dollar
    assert rec_large.effective_spread_bps > rec_small.effective_spread_bps
    assert rec_large.adverse_spread_dollar > rec_small.adverse_spread_dollar


def test_glosten_harris_hmac_tamper_detection():
    """Verify cryptographic HMAC-SHA256 tampering rejection."""
    prices = [100.0 + 0.05 * ((-1) ** i) for i in range(40)]
    receipt = evaluate_glosten_harris_from_prices(prices, ticker="TAMPER_TEST")

    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.benchmark_volume:.2f}|"
        f"{receipt.c0_fixed_transitory:.6f}|{receipt.c1_variable_transitory:.6f}|"
        f"{receipt.z0_fixed_adverse:.6f}|{receipt.z1_variable_adverse:.6f}|"
        f"{receipt.effective_spread_dollar:.4f}|{receipt.effective_spread_bps:.2f}|"
        f"{receipt.adverse_selection_ratio:.4f}|{receipt.order_processing_ratio:.4f}|"
        f"{receipt.r_squared:.4f}|{receipt.regime}|{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered payload fails
    tampered = payload.replace("TAMPER_TEST", "FORGED_TEST")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_glosten_harris_sqlite_persistence(tmp_path):
    """Verify SQLite database persistence and schema creation."""
    db_file = str(tmp_path / "test_gh.db")
    prices = [100.0 + 0.1 * ((-1) ** i) for i in range(35)]

    receipt = evaluate_glosten_harris_from_prices(prices, ticker="PERSIST_GH")
    assert persist_glosten_harris_record(receipt, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, effective_spread_dollar, adverse_selection_ratio, regime, hmac_signature FROM glosten_harris_receipts WHERE ticker = ?",
            ("PERSIST_GH",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "PERSIST_GH"
        assert abs(row[1] - receipt.effective_spread_dollar) < 1e-4
        assert row[3] == receipt.regime
        assert row[4] == receipt.hmac_signature

