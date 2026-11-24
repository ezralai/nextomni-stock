"""Unit tests for Barndorff-Nielsen, Hansen, Lunde & Shephard (2008, 2011) Realized Kernel Engine."""

import math
import os
import tempfile
import numpy as np
import pytest

from src.analytics.realized_kernel import (
    RealizedKernelReceipt,
    calculate_optimal_bandwidth,
    calculate_realized_kernel,
    compute_autocovariance,
    compute_receipt_payload,
    estimate_integrated_quarticity,
    evaluate_realized_kernel_from_prices,
    parzen_kernel_weight,
    persist_as_record,
    verify_hmac_signature,
)


def test_parzen_kernel_weight_properties():
    """Verify mathematical properties of the modified Parzen kernel weight function."""
    # Peak at origin
    assert math.isclose(parzen_kernel_weight(0.0), 1.0)

    # Halfway point: 1 - 6*(0.25) + 6*(0.125) = 0.25
    assert math.isclose(parzen_kernel_weight(0.5), 0.25)

    # Cutoff at 1.0
    assert math.isclose(parzen_kernel_weight(1.0), 0.0)

    # Zero outside [0, 1]
    assert parzen_kernel_weight(1.5) == 0.0
    assert parzen_kernel_weight(2.0) == 0.0

    # Strict symmetry k(-x) = k(x)
    for x in [0.1, 0.25, 0.5, 0.75, 1.2]:
        assert math.isclose(parzen_kernel_weight(x), parzen_kernel_weight(-x))


def test_autocovariance_computation():
    """Verify sample autocovariance matches known theoretical values."""
    rets = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    # lag 0 is sum of squares
    gamma_0 = compute_autocovariance(rets, 0)
    assert math.isclose(gamma_0, float(np.sum(rets ** 2)))

    # lag 1
    gamma_1 = compute_autocovariance(rets, 1)
    expected_1 = float(np.sum(rets[1:] * rets[:-1]))
    assert math.isclose(gamma_1, expected_1)

    # lag beyond length
    assert compute_autocovariance(rets, 10) == 0.0


def test_integrated_quarticity_estimation():
    """Verify integrated quarticity estimation produces strictly positive, finite values."""
    np.random.seed(42)
    rets = np.random.normal(0.0, 0.01, 100)
    iq = estimate_integrated_quarticity(rets)
    assert iq > 0.0
    assert math.isfinite(iq)


def test_optimal_bandwidth_scaling():
    """Verify Barndorff-Nielsen optimal bandwidth H* scales appropriately with sample size."""
    np.random.seed(42)
    rets_small = np.random.normal(0.0, 0.01, 50)
    rets_large = np.random.normal(0.0, 0.01, 500)

    h_small, omega_small, xi_small, iq_small = calculate_optimal_bandwidth(rets_small)
    h_large, omega_large, xi_large, iq_large = calculate_optimal_bandwidth(rets_large)

    assert h_small >= 1
    assert h_large >= 1
    # H* grows as n^(3/5), so larger sample should have larger or equal bandwidth
    assert h_large >= h_small
    assert omega_small > 0.0
    assert omega_large > 0.0


def test_realized_kernel_microstructure_noise_correction():
    """Verify that Realized Kernel removes bid-ask bounce inflation from noisy returns."""
    np.random.seed(42)
    n = 200
    # True latent Brownian diffusion returns
    latent_rets = np.random.normal(0.0, 0.005, n)

    # Add i.i.d. microstructure bid-ask bounce noise u_t
    u = np.random.normal(0.0, 0.003, n + 1)
    # Observed returns: r_t = latent_rets_t + (u_t - u_{t-1})
    noisy_rets = latent_rets + np.diff(u)

    # In the presence of bid-ask bounce (u_t - u_{t-1}), adjacent returns have negative covariance
    # and raw RV is artificially inflated by 2 * var(u)
    rv_raw, rk_var, omega_sq, xi, iq, opt_h, bias_pct = calculate_realized_kernel(noisy_rets)

    assert rv_raw > 0.0
    assert rk_var > 0.0
    # Realized Kernel should filter negative autocovariance and correct upward noise bias
    assert rk_var < rv_raw
    assert bias_pct > 0.0
    assert opt_h >= 1


def test_realized_kernel_receipt_and_hmac_integrity():
    """Verify RealizedKernelReceipt generation, fields, and HMAC-SHA256 signature sealing."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.012, 80)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_realized_kernel_from_prices(prices, ticker="AAPL")

    assert isinstance(receipt, RealizedKernelReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.sample_sessions == len(prices) - 1
    assert receipt.raw_realized_variance > 0.0
    assert receipt.realized_kernel_variance > 0.0
    assert receipt.annualized_kernel_vol > 0.0
    assert receipt.kernel_function == "MODIFIED_PARZEN"

    # Validate HMAC signature
    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered signature check
    tampered_payload = payload.replace("AAPL", "GOOG")
    assert verify_hmac_signature(tampered_payload, receipt.hmac_signature) is False


def test_sqlite_persistence():
    """Verify database insertion and table creation in SQLite."""
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0, 0.01, 40)))
    receipt = evaluate_realized_kernel_from_prices(prices, ticker="MSFT")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        success = persist_as_record(receipt, db_path=temp_db)
        assert success is True
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
