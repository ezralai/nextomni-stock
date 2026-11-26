"""Unit tests for Christensen, Oomen & Podolskij (2014) Pre-averaging Volatility & Jump Test Engine."""

import math
import os
import tempfile
import numpy as np
import pytest

from src.analytics.christensen_preaveraging_jumps import (
    ChristensenPreaveragingReceipt,
    calculate_christensen_preaveraging,
    compute_preaveraged_increments,
    compute_receipt_payload,
    evaluate_preaveraging_from_prices,
    persist_as_record,
    tent_weight,
    verify_hmac_signature,
)


def test_tent_weight_properties():
    """Verify mathematical properties of the tent weight function g(x) = min(x, 1 - x)."""
    assert tent_weight(0.0) == 0.0
    assert tent_weight(1.0) == 0.0
    assert tent_weight(0.5) == 0.5
    assert tent_weight(0.25) == 0.25
    assert tent_weight(0.75) == 0.25

    # Symmetry g(x) == g(1 - x)
    for x in [0.1, 0.2, 0.35, 0.5, 0.8]:
        assert math.isclose(tent_weight(x), tent_weight(1.0 - x))


def test_compute_preaveraged_increments_shape():
    """Verify pre-averaged increment dimensionality and filtering."""
    rets = np.random.normal(0.0, 0.01, 100)
    k_n = 6
    y_bar = compute_preaveraged_increments(rets, k_n)
    # Length is n - k_n + 2
    assert len(y_bar) == 100 - 6 + 2
    assert np.all(np.isfinite(y_bar))


def test_christensen_preaveraging_pure_diffusion():
    """Verify that pure Gaussian diffusion yields low jump ratio and no jump detection."""
    np.random.seed(42)
    normal_rets = np.random.normal(0.0, 0.005, 120)
    prv, pbv, pj, raw_rv, omega_sq, j_ratio, z_score, is_jump, k_n = calculate_christensen_preaveraging(
        normal_rets,
        theta=0.8,
    )

    assert prv > 0.0
    assert pbv > 0.0
    assert raw_rv > 0.0
    assert k_n >= 2
    assert pbv <= prv
    # On continuous Gaussian path, jump ratio is small and no jump is detected
    assert j_ratio < 0.20
    assert is_jump is False


def test_christensen_preaveraging_with_jump():
    """Verify that a large discontinuous jump is detected with statistically significant Z-score."""
    np.random.seed(42)
    rets = np.random.normal(0.0, 0.003, 150)
    # Add large discrete jump shock
    rets[75] = 0.065

    prv, pbv, pj, raw_rv, omega_sq, j_ratio, z_score, is_jump, k_n = calculate_christensen_preaveraging(
        rets,
        theta=0.8,
    )

    assert prv > 0.0
    assert pj > 0.0
    assert j_ratio > 0.10
    # Z-statistic should be positive and elevated due to jump
    assert z_score > 1.5


def test_evaluate_preaveraging_from_prices_and_hmac():
    """Verify receipt generation, fields, and cryptographic HMAC-SHA256 signature sealing."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.012, 90)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_preaveraging_from_prices(prices, ticker="AMZN", theta=0.8)

    assert isinstance(receipt, ChristensenPreaveragingReceipt)
    assert receipt.ticker == "AMZN"
    assert receipt.sample_sessions == len(prices) - 1
    assert receipt.preaveraged_realized_var > 0.0
    assert receipt.preaveraged_continuous_var > 0.0
    assert receipt.annualized_total_vol > 0.0
    assert receipt.tuning_theta == 0.8
    assert receipt.derivation_mode == "TENT_WEIGHTED_PREAVERAGING"

    # Validate HMAC signature
    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered signature check
    tampered = payload.replace("AMZN", "GOOG")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_sqlite_persistence():
    """Verify database insertion and table creation in SQLite."""
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0, 0.01, 50)))
    receipt = evaluate_preaveraging_from_prices(prices, ticker="META", theta=0.8)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        success = persist_as_record(receipt, db_path=temp_db)
        assert success is True
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
