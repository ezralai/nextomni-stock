"""Unit tests for Joel Hasbrouck (2009) Bayesian Gibbs Sampler Effective Spread Engine."""

import os
import sqlite3
import numpy as np
import pytest

from src.analytics.hasbrouck_gibbs_spread import (
    HasbrouckGibbsReceipt,
    calculate_roll_spread,
    persist_hasbrouck_gibbs_record,
    run_hasbrouck_gibbs,
    verify_hmac_signature,
)


def test_hasbrouck_gibbs_synthetic_roll_recovery():
    """Verify Gibbs sampler accurately estimates half-spread c under true Roll DGP."""
    rng = np.random.default_rng(12345)
    T = 200
    true_c = 0.25
    sigma_u = 0.10

    # True Roll DGP: m_t = m_{t-1} + u_t; p_t = m_t + c * q_t
    u = rng.normal(0.0, sigma_u, size=T)
    m = 100.0 + np.cumsum(u)
    q = rng.choice([-1.0, 1.0], size=T)
    prices = m + true_c * q

    receipt = run_hasbrouck_gibbs(
        prices=prices,
        iterations=300,
        burn_in=80,
        seed=42,
        ticker="TEST_TICKER",
    )

    assert isinstance(receipt, HasbrouckGibbsReceipt)
    assert receipt.ticker == "TEST_TICKER"
    assert receipt.sample_size == T
    assert receipt.effective_spread_dollar > 0.0
    assert 0.12 <= receipt.posterior_c_mean <= 0.40
    assert receipt.posterior_sigma_u > 0.0
    assert receipt.regime in (
        "HIGH_LIQUIDITY_TIGHT_SPREAD",
        "MODERATE_EFFECTIVE_SPREAD",
        "ILLIQUID_WIDE_SPREAD",
    )

    # Verify HMAC signature
    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.iterations}|{receipt.burn_in}|"
        f"{receipt.effective_spread_dollar:.6f}|{receipt.effective_spread_bps:.4f}|"
        f"{receipt.posterior_c_mean:.6f}|{receipt.posterior_c_std:.6f}|"
        f"{receipt.posterior_sigma_u:.6f}|{receipt.roll_spread_dollar:.6f}|"
        f"{receipt.positive_autocovariance_flag}|{receipt.regime}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True


def test_hasbrouck_gibbs_positive_autocovariance_breakdown():
    """Verify Gibbs sampler resolves Roll model breakdown when Cov(dp_t, dp_{t-1}) > 0."""
    # Construct a strongly persistent momentum price path with positive autocorrelation
    rng = np.random.default_rng(999)
    T = 150
    # Positive AR(1) return process
    r = np.zeros(T)
    for i in range(1, T):
        r[i] = 0.7 * r[i - 1] + rng.normal(0.01, 0.02)
    prices = 100.0 * np.exp(np.cumsum(r))

    roll_spread, pos_cov = calculate_roll_spread(prices)
    assert pos_cov is True
    assert roll_spread == 0.0

    # Run Hasbrouck Gibbs
    receipt = run_hasbrouck_gibbs(
        prices=prices,
        iterations=200,
        burn_in=50,
        seed=123,
        ticker="MOMENTUM_STOCK",
    )

    assert receipt.positive_autocovariance_flag is True
    assert receipt.roll_spread_dollar == 0.0
    # Hasbrouck Gibbs strictly bounds c >= 0 and yields a valid positive effective spread
    assert receipt.effective_spread_dollar > 0.0
    assert receipt.posterior_c_mean > 0.0
    assert receipt.effective_spread_bps > 0.0


def test_hasbrouck_gibbs_hmac_tamper_detection():
    """Verify cryptographic HMAC-SHA256 tampering rejection."""
    prices = [100.0 + 0.1 * i + (-1.0 if i % 2 == 0 else 1.0) * 0.2 for i in range(40)]
    receipt = run_hasbrouck_gibbs(prices, iterations=100, burn_in=20, seed=77, ticker="SECURE")

    # True payload passes
    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.iterations}|{receipt.burn_in}|"
        f"{receipt.effective_spread_dollar:.6f}|{receipt.effective_spread_bps:.4f}|"
        f"{receipt.posterior_c_mean:.6f}|{receipt.posterior_c_std:.6f}|"
        f"{receipt.posterior_sigma_u:.6f}|{receipt.roll_spread_dollar:.6f}|"
        f"{receipt.positive_autocovariance_flag}|{receipt.regime}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered payload fails
    tampered_payload = payload.replace("SECURE", "TAMPERED")
    assert verify_hmac_signature(tampered_payload, receipt.hmac_signature) is False


def test_hasbrouck_gibbs_sqlite_persistence(tmp_path):
    """Verify SQLite persistence and schema creation."""
    db_file = str(tmp_path / "test_hasbrouck.db")
    prices = [50.0 + 0.05 * (-1 if i % 2 == 0 else 1) for i in range(30)]

    receipt = run_hasbrouck_gibbs(prices, iterations=100, burn_in=25, seed=42, ticker="STORE")
    assert persist_hasbrouck_gibbs_record(receipt, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, effective_spread_dollar, regime, hmac_signature FROM hasbrouck_gibbs_receipts WHERE ticker = ?", ("STORE",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "STORE"
        assert abs(row[1] - receipt.effective_spread_dollar) < 1e-6
        assert row[2] == receipt.regime
        assert row[3] == receipt.hmac_signature

