"""Unit tests for Bouchaud, Gefen, Potters & Wyart (2004) Transient Impact Propagator Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.bouchaud_propagator_impact import (
    calculate_bouchaud_propagator,
    compute_receipt_payload,
    evaluate_bouchaud_propagator_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def _generate_synthetic_transient_impact(n=300, gamma_true=0.4, beta_true=0.3, seed=42):
    rng = np.random.default_rng(seed)
    # Generate autoregressive signs to mimic long memory
    signs = np.zeros(n)
    curr = 1.0
    for t in range(n):
        if rng.random() < 0.70:  # Persistence
            signs[t] = curr
        else:
            curr = -curr
            signs[t] = curr

    # Generate price with power-law kernel G(l) = 0.05 / (1 + l)^beta
    prices = np.zeros(n)
    prices[0] = 100.0
    for t in range(1, n):
        # Accumulate transient impact from past trades
        impact = 0.0
        lookback = min(15, t)
        for s in range(lookback):
            ell = s
            g_ell = 0.05 / ((1.0 + ell) ** beta_true)
            impact += g_ell * signs[t - 1 - s]
        noise = float(rng.normal(0, 0.02))
        prices[t] = 100.0 + impact + noise
    return prices, signs


def test_bouchaud_power_law_decay_and_half_life():
    prices, signs = _generate_synthetic_transient_impact(n=300, seed=101)
    receipt = calculate_bouchaud_propagator(
        prices=prices,
        signed_trades=signs,
        ticker="SPY",
        max_lags=12,
    )

    assert receipt.ticker == "SPY"
    assert receipt.sample_trades == 300
    assert 0.02 <= receipt.beta_propagator_decay <= 1.50
    assert 0.05 <= receipt.gamma_order_flow <= 0.95
    assert receipt.resilience_half_life > 0.0
    assert 0.0 < receipt.permanent_to_transient_ratio <= 1.0
    assert len(receipt.response_lags) == 12
    assert len(receipt.propagator_values) == 12


def test_bouchaud_efficiency_balance_condition():
    prices, signs = _generate_synthetic_transient_impact(n=250, seed=202)
    receipt = calculate_bouchaud_propagator(
        prices=prices,
        signed_trades=signs,
        ticker="QQQ",
        max_lags=10,
    )

    expected_balance = round((1.0 - receipt.gamma_order_flow) / 2.0, 4)
    assert abs(receipt.theoretical_beta_balance - expected_balance) < 1e-4
    assert abs(receipt.efficiency_gap - abs(receipt.beta_propagator_decay - expected_balance)) < 1e-3
    assert receipt.regime in (
        "EFFICIENT_DIFFUSIVE_EQUILIBRIUM",
        "SUB_DIFFUSIVE_OVER_MEAN_REVERSION",
        "SUPER_DIFFUSIVE_MOMENTUM_RUNS",
    )


def test_bouchaud_tick_test_inference_and_boundary():
    # Boundary: insufficient prices (< 30)
    short_prices = [100.0, 100.1, 100.05]
    bad_rec = evaluate_bouchaud_propagator_from_prices(short_prices, ticker="AAPL")
    assert bad_rec.derivation_mode == "FAILED"
    assert "ERROR" in bad_rec.regime

    # Valid tick price series
    rng = np.random.default_rng(42)
    prices = 150.0 + np.cumsum(rng.normal(0, 0.05, size=150))
    good_rec = evaluate_bouchaud_propagator_from_prices(prices, ticker="AAPL", max_lags=10)
    assert good_rec.ticker == "AAPL"
    assert good_rec.sample_trades == 149
    assert good_rec.derivation_mode == "TICK_TEST_INFERRED_TRADES"


def test_bouchaud_persistence_and_hmac():
    prices, signs = _generate_synthetic_transient_impact(n=200, seed=303)
    receipt = calculate_bouchaud_propagator(prices, signs, ticker="MSFT", max_lags=8)

    assert receipt.ticker == "MSFT"
    assert len(receipt.hmac_signature) == 64

    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "_tampered", receipt.hmac_signature) is False

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_bouchaud.db")
        success = persist_as_record(receipt, test_db)
        assert success is True

