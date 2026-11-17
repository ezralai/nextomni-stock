"""Unit tests for Gatheral, Jaisson & Rosenbaum (2018) Rough Volatility Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.gatheral_rough_volatility import (
    GatheralRoughVolReceipt,
    calculate_gatheral_rough_volatility,
    estimate_hurst_exponent_rough_vol,
    evaluate_rough_vol_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def test_structure_function_scaling_law():
    """Verify that structure function m(q, Delta) yields monotonic exponents zeta_2 >= zeta_1."""
    np.random.seed(42)
    # Simulate a path with subdiffusive roughness (Riemann-Liouville fractional kernel)
    n = 300
    white_noise = np.random.normal(0, 0.1, n)
    # Power-law kernel (k+1)^(H - 0.5) for H = 0.15
    kernel = np.array([(k + 1.0) ** (-0.35) for k in range(40)])
    log_vol = np.convolve(white_noise, kernel, mode="same")

    h, zeta_1, zeta_2, r2 = estimate_hurst_exponent_rough_vol(
        log_vol,
        lags=[1, 2, 3, 4, 5, 7, 10],
    )

    assert 0.0 < h < 0.5  # Subdiffusive rough regime confirmed
    assert zeta_1 > 0.0
    assert zeta_2 >= zeta_1  # Monotonic scaling with moment order q
    assert r2 > 0.80  # Strong power-law fit on log-log scale


def test_rough_fractional_hurst_exponent_estimation():
    """Verify Hurst parameter detection distinguishes rough paths from smooth/persistent paths."""
    np.random.seed(42)
    n = 150

    # 1. Ultra-rough series: high negative autocorrelation
    white_noise = np.random.normal(0, 0.05, n)
    rough_vol = np.exp(white_noise)
    rec_rough = calculate_gatheral_rough_volatility(rough_vol, ticker="SPY")

    assert isinstance(rec_rough, GatheralRoughVolReceipt)
    assert rec_rough.hurst_exponent_h < 0.35
    assert rec_rough.regime in ("ULTRA_ROUGH_SUBDIFFUSIVE", "MODERATELY_ROUGH")
    assert rec_rough.holder_regularity_alpha < 0.35

    # 2. Smooth / persistent random walk: H ~ 0.5
    rw = np.cumsum(np.random.normal(0, 0.02, n))
    smooth_vol = np.exp(rw - np.min(rw) + 0.1)
    rec_smooth = calculate_gatheral_rough_volatility(smooth_vol, ticker="SPY")

    assert rec_smooth.hurst_exponent_h > rec_rough.hurst_exponent_h


def test_rough_vol_from_price_series_and_regimes():
    """Test estimation from simulated equity prices and regime classification."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0004, 0.012, 120)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_rough_vol_from_prices(prices, ticker="QQQ")

    assert receipt.ticker == "QQQ"
    assert receipt.sample_sessions > 30
    assert 0.01 <= receipt.hurst_exponent_h <= 0.99
    assert receipt.structure_function_r2 > 0.0
    assert receipt.volatility_of_volatility_nu > 0.0
    assert receipt.mean_annualized_volatility > 0.0
    assert receipt.regime in (
        "ULTRA_ROUGH_SUBDIFFUSIVE",
        "MODERATELY_ROUGH",
        "NEAR_BROWNIAN_DIFFUSIVE",
        "PERSISTENT_LONG_MEMORY",
    )


def test_gatheral_rough_vol_persistence_and_hmac():
    """Verify SQLite audit record persistence and cryptographic HMAC-SHA256 validation."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.015, 80)
    prices = 50.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_rough_vol_from_prices(prices, ticker="IWM")

    # HMAC validation
    raw_payload = (
        f"{receipt.ticker}|{receipt.sample_sessions}|{receipt.hurst_exponent_h:.4f}|"
        f"{receipt.zeta_q1:.4f}|{receipt.zeta_q2:.4f}|{receipt.holder_regularity_alpha:.4f}|"
        f"{receipt.volatility_of_volatility_nu:.4f}|{receipt.regime}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(raw_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(raw_payload + "_altered", receipt.hmac_signature) is False

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        success = persist_as_record(receipt, db_path=temp_db)
        assert success is True
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
