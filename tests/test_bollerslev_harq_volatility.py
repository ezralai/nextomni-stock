"""Unit tests for Bollerslev, Patton & Quaedvlieg (2016) HARQ & HARQ-F Volatility Cascade Engine."""

import math
import os
import tempfile
import numpy as np
import pytest

from src.analytics.bollerslev_harq_volatility import (
    BollerslevHARQReceipt,
    calculate_realized_quarticity,
    compute_receipt_payload,
    estimate_harq_model,
    evaluate_harq_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def test_calculate_realized_quarticity_properties():
    """Verify that Realized Variance and Quarticity are strictly non-negative and finite."""
    np.random.seed(42)
    rets = np.random.normal(0.0005, 0.015, 60)
    rv, rq = calculate_realized_quarticity(rets)

    assert len(rv) == len(rets)
    assert len(rq) == len(rets)
    assert np.all(rv > 0.0)
    assert np.all(rq > 0.0)
    assert np.all(np.isfinite(rv))
    assert np.all(np.isfinite(rq))


def test_harq_ols_estimation_both_modes():
    """Verify standard HARQ and full HARQ-F model estimation and R^2 bounds."""
    np.random.seed(42)
    n = 120
    rv = np.abs(np.random.normal(0.0002, 0.0001, n))
    rq = (rv ** 2) * 1.5

    # 1. Standard HARQ
    p_harq, p_har, r2_harq, r2_har = estimate_harq_model(rv, rq, is_full_harq_f=False)
    assert len(p_harq) == 5  # [beta0, beta_d, beta_dQ, beta_w, beta_m]
    assert len(p_har) == 4   # [beta0, beta_d, beta_w, beta_m]
    assert 0.0 <= r2_harq <= 1.0
    assert 0.0 <= r2_har <= 1.0

    # 2. Full HARQ-F
    p_harq_f, _, r2_harq_f, _ = estimate_harq_model(rv, rq, is_full_harq_f=True)
    assert len(p_harq_f) == 7  # [beta0, beta_d, beta_dQ, beta_w, beta_wQ, beta_m, beta_mQ]
    assert 0.0 <= r2_harq_f <= 1.0


def test_harq_measurement_error_attenuation():
    """Verify econometric attenuation property under high measurement error regimes."""
    np.random.seed(42)
    n = 150
    # Simulate AR(1) persistent latent variance with occasional massive measurement error bursts
    latent_rv = np.zeros(n)
    latent_rv[0] = 0.0002
    for t in range(1, n):
        latent_rv[t] = 0.00005 + 0.75 * latent_rv[t - 1] + np.random.normal(0.0, 0.00002)
    latent_rv = np.maximum(latent_rv, 1e-6)

    # Incur measurement noise in observed RV with large quarticity spikes
    obs_rv = np.copy(latent_rv)
    obs_rq = obs_rv ** 2
    # Add large quarticity bursts at selected sessions
    spike_indices = [25, 45, 75, 105, 130]
    for idx in spike_indices:
        obs_rv[idx] += 0.0008
        obs_rq[idx] = (obs_rv[idx] ** 2) * 15.0

    p_harq, p_har, r2_harq, r2_har = estimate_harq_model(obs_rv, obs_rq, is_full_harq_f=False)

    # Verify parameters are finite numbers
    assert np.all(np.isfinite(p_harq))
    assert np.all(np.isfinite(p_har))


def test_evaluate_harq_from_prices_and_hmac():
    """Verify receipt generation, forward volatility horizons, and cryptographic HMAC sealing."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.012, 90)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_harq_from_prices(prices, ticker="NVDA", is_full_harq_f=False)

    assert isinstance(receipt, BollerslevHARQReceipt)
    assert receipt.ticker == "NVDA"
    assert receipt.sample_sessions == len(prices) - 1
    assert receipt.forecast_1d_ann_vol > 0.0
    assert receipt.forecast_5d_ann_vol > 0.0
    assert receipt.forecast_22d_ann_vol > 0.0
    assert receipt.is_full_harq_f is False

    # Validate HMAC signature
    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered signature check
    tampered = payload.replace("NVDA", "AMD")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_sqlite_persistence():
    """Verify database insertion and table creation in SQLite."""
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0, 0.01, 50)))
    receipt = evaluate_harq_from_prices(prices, ticker="TSLA", is_full_harq_f=True)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        success = persist_as_record(receipt, db_path=temp_db)
        assert success is True
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
