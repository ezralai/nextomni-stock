"""Unit tests for Aït-Sahalia & Jacod (2009, 2012) High-Frequency Jump Test Engine."""

import math
import os
import tempfile
import numpy as np
import pytest

from src.analytics.aitsahalia_jacod_jumps import (
    AitSahaliaJacodReceipt,
    calculate_aitsahalia_jacod_test,
    calculate_power_variation,
    compute_coarse_returns,
    compute_receipt_payload,
    evaluate_aitsahalia_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def test_power_variation_and_coarse_returns():
    """Verify power variation calculation and coarse frequency log return sampling."""
    prices = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0])
    coarse = compute_coarse_returns(prices, k=2)
    # k=2 prices: [100.0, 101.0, 102.0], returns: [ln(101/100), ln(102/101)]
    assert len(coarse) == 2
    assert math.isclose(coarse[0], math.log(101.0 / 100.0))

    rets = np.array([0.01, -0.02, 0.015])
    # p=2 power var is sum of squares
    pv_2 = calculate_power_variation(rets, p=2.0)
    assert math.isclose(pv_2, float(np.sum(rets ** 2)))

    # p=4 power var
    pv_4 = calculate_power_variation(rets, p=4.0)
    assert math.isclose(pv_4, float(np.sum(rets ** 4)))


def test_continuous_diffusion_ratio_asymptotics():
    """Verify that continuous Brownian motion yields S(4, 2) close to theoretical limit 2.0."""
    np.random.seed(42)
    n = 200
    daily_rets = np.random.normal(0.0002, 0.005, n)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    s_ratio, target_null, target_alt, t_stat, p_val, is_jump, b_fine, b_coarse = calculate_aitsahalia_jacod_test(
        prices=prices,
        p=4.0,
        k=2,
    )

    assert target_null == 2.0
    assert target_alt == 1.0
    # Under continuous diffusion, S should be substantially above 1.0 toward 2.0
    assert s_ratio > 1.35
    assert is_jump is False
    assert p_val > 0.05


def test_discontinuous_jump_ratio_asymptotics():
    """Verify that discontinuous Poisson jumps drive S(4, 2) down toward 1.0, rejecting continuity."""
    np.random.seed(42)
    n = 200
    daily_rets = np.random.normal(0.0001, 0.002, n)
    # Inject large discrete jump shock
    daily_rets[100] = 0.065
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    s_ratio, target_null, target_alt, t_stat, p_val, is_jump, b_fine, b_coarse = calculate_aitsahalia_jacod_test(
        prices=prices,
        p=4.0,
        k=2,
    )

    assert target_null == 2.0
    assert target_alt == 1.0
    # Jumps push S toward 1.0
    assert s_ratio < 1.35
    assert is_jump is True
    assert t_stat < -1.0
    assert p_val < 0.15


def test_evaluate_from_prices_and_hmac():
    """Verify receipt generation, fields, and cryptographic HMAC-SHA256 signature sealing."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.012, 100)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_aitsahalia_from_prices(prices, ticker="AAPL", p=4.0, k=2)

    assert isinstance(receipt, AitSahaliaJacodReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.sample_sessions == len(prices) - 1
    assert receipt.power_p == 4.0
    assert receipt.subsample_k == 2
    assert receipt.continuous_null_target == 2.0
    assert receipt.jump_alternative_target == 1.0
    assert receipt.derivation_mode == "POWER_VARIATION_FREQUENCY_RATIO"

    # Validate HMAC signature
    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered signature check
    tampered = payload.replace("AAPL", "MSFT")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_sqlite_persistence():
    """Verify database insertion and table creation in SQLite."""
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0, 0.01, 50)))
    receipt = evaluate_aitsahalia_from_prices(prices, ticker="JPM")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        success = persist_as_record(receipt, db_path=temp_db)
        assert success is True
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
