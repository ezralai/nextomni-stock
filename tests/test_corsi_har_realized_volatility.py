"""Unit tests for Fulvio Corsi (2009) Heterogeneous Autoregressive Realized Volatility (HAR-RV) Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.corsi_har_realized_volatility import (
    calculate_corsi_har_rv,
    compute_receipt_payload,
    evaluate_corsi_har_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def _generate_synthetic_har_rv(n=150, seed=42):
    rng = np.random.default_rng(seed)
    # Simulate a cascade process
    rv = np.zeros(n)
    rv[:22] = rng.uniform(0.01, 0.02, size=22)
    for t in range(22, n):
        w = np.mean(rv[t - 5 : t])
        m = np.mean(rv[t - 22 : t])
        # Cascade equation: 0.002 + 0.35 * d + 0.35 * w + 0.25 * m + eps
        rv[t] = max(0.005, 0.002 + 0.35 * rv[t - 1] + 0.35 * w + 0.25 * m + float(rng.normal(0, 0.002)))
    return rv


def test_corsi_har_cascade_components_and_persistence():
    rv_series = _generate_synthetic_har_rv(n=180, seed=101)
    receipt = calculate_corsi_har_rv(rv_series, ticker="SPY", use_log=True)

    assert receipt.ticker == "SPY"
    assert receipt.sample_sessions == 180
    assert receipt.r_squared > 0.10
    assert receipt.persistence > 0.0
    assert receipt.half_life_sessions > 0.0
    assert receipt.forecast_1d_ann_vol > 0.0
    assert receipt.forecast_5d_ann_vol > 0.0
    assert receipt.forecast_22d_ann_vol > 0.0


def test_corsi_har_horizon_dominance_regimes():
    rng = np.random.default_rng(202)
    # 1. Pure daily shock series (daily noise trader dominated)
    n = 100
    rv_daily_heavy = np.zeros(n)
    rv_daily_heavy[:22] = 0.015
    for t in range(22, n):
        # 80% daily persistence
        rv_daily_heavy[t] = max(0.005, 0.002 + 0.75 * rv_daily_heavy[t - 1] + float(rng.normal(0, 0.003)))

    rec_daily = calculate_corsi_har_rv(rv_daily_heavy, ticker="QQQ", use_log=True)
    assert rec_daily.beta_daily > 0.30
    assert rec_daily.regime in ("DAILY_NOISE_TRADER_DOMINATED", "BALANCED_MULTI_HORIZON_CASCADE")


def test_corsi_har_log_vs_linear_spec():
    rv_series = _generate_synthetic_har_rv(n=120, seed=303)

    # Log-HAR
    rec_log = calculate_corsi_har_rv(rv_series, ticker="AAPL", use_log=True)
    assert rec_log.forecast_1d_ann_vol > 0.0

    # Linear HAR
    rec_lin = calculate_corsi_har_rv(rv_series, ticker="AAPL", use_log=False)
    assert rec_lin.forecast_1d_ann_vol > 0.0
    assert rec_lin.r_squared >= 0.0


def test_corsi_har_persistence_and_hmac():
    rng = np.random.default_rng(42)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, size=80)))

    receipt = evaluate_corsi_har_from_prices(prices, ticker="MSFT", use_log=True)
    assert receipt.ticker == "MSFT"
    assert len(receipt.hmac_signature) == 64

    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "_tampered", receipt.hmac_signature) is False

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_corsi_har.db")
        success = persist_as_record(receipt, test_db)
        assert success is True

