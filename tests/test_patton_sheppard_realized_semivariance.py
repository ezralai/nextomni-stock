"""Unit tests for Patton & Sheppard (2015) Realized Semivariance & SHAR Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.patton_sheppard_realized_semivariance import (
    PattonSheppardReceipt,
    calculate_patton_sheppard_shar,
    calculate_signed_semivariances,
    evaluate_semivariance_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def test_signed_semivariance_exact_partition():
    """Verify that Realized Semivariance partitions Total Quadratic Variation exactly: RV = RS+ + RS-."""
    np.random.seed(42)
    intraday_rets = np.random.normal(0.0001, 0.003, 100)

    rv, rs_pos, rs_neg, signed_ratio = calculate_signed_semivariances(intraday_rets)

    assert rv > 0.0
    assert rs_pos > 0.0
    assert rs_neg > 0.0
    assert np.isclose(rv, rs_pos + rs_neg, rtol=1e-6)
    assert -1.0 <= signed_ratio <= 1.0


def test_patton_sheppard_bad_volatility_persistence_dominance():
    """Verify empirical stylized fact that bad volatility (RS-) persistence exceeds good volatility (RS+)."""
    np.random.seed(42)
    n = 150
    base_rv = (0.16 / np.sqrt(252)) ** 2

    rv_series = np.zeros(n)
    rs_plus = np.zeros(n)
    rs_minus = np.zeros(n)

    rv_series[0] = base_rv
    rs_plus[0] = 0.5 * base_rv
    rs_minus[0] = 0.5 * base_rv

    for t in range(1, n):
        # Simulate intraday sampling where negative shocks feed higher subsequent volatility
        intraday = np.random.normal(0, np.sqrt(base_rv / 10), 10)
        pos = intraday[intraday > 0]
        neg = intraday[intraday < 0]
        p_var = float(np.sum(pos ** 2)) if len(pos) > 0 else 1e-8
        n_var = float(np.sum(neg ** 2)) if len(neg) > 0 else 1e-8

        # Bad volatility (n_var) has 3x greater predictive coefficient than good volatility (p_var)
        rv_next = 0.1 * base_rv + 0.55 * n_var + 0.15 * p_var + 0.3 * rv_series[t - 1]
        rv_series[t] = rv_next
        rs_plus[t] = p_var
        rs_minus[t] = n_var

    receipt = calculate_patton_sheppard_shar(
        daily_rv=rv_series,
        daily_rs_plus=rs_plus,
        daily_rs_minus=rs_minus,
        ticker="SPY",
        use_log=True,
    )

    assert isinstance(receipt, PattonSheppardReceipt)
    assert receipt.sample_sessions == n
    assert receipt.beta_negative_semivar > receipt.beta_positive_semivar
    assert receipt.asymmetry_ratio > 1.0
    assert receipt.bad_volatility_premium_bps > 0.0


def test_patton_sheppard_log_vs_linear_spec():
    """Test both logarithmic and linear SHAR specifications on simulated equity prices."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.014, 120)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    rec_log = evaluate_semivariance_from_prices(prices, ticker="SPY", use_log=True)
    rec_lin = evaluate_semivariance_from_prices(prices, ticker="SPY", use_log=False)

    assert rec_log.ticker == "SPY"
    assert rec_lin.ticker == "SPY"
    assert rec_log.sample_sessions == len(prices) - 1
    assert rec_lin.sample_sessions == len(prices) - 1
    assert 0.0 < rec_log.forecast_1d_ann_vol < 2.0
    assert 0.0 < rec_lin.forecast_1d_ann_vol < 2.0
    assert rec_log.regime in (
        "DOWNSIDE_VOLATILITY_DOMINATED",
        "UPSIDE_VOLATILITY_DOMINATED",
        "SYMMETRIC_VARIANCE_PERSISTENCE",
        "ELEVATED_TAIL_ASYMMETRY",
    )


def test_patton_sheppard_persistence_and_hmac():
    """Verify SQLite audit record persistence and HMAC-SHA256 signature verification."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0002, 0.011, 75)
    prices = 75.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_semivariance_from_prices(prices, ticker="QQQ", use_log=True)

    raw_payload = (
        f"{receipt.ticker}|{receipt.sample_sessions}|{receipt.beta_positive_semivar:.4f}|"
        f"{receipt.beta_negative_semivar:.4f}|{receipt.asymmetry_ratio:.4f}|"
        f"{receipt.r_squared:.4f}|{receipt.downside_variance_share:.4f}|"
        f"{receipt.regime}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(raw_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(raw_payload + "_corrupted", receipt.hmac_signature) is False

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        success = persist_as_record(receipt, db_path=temp_db)
        assert success is True
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
