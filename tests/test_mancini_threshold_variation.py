"""Unit tests for Mancini (2009, 2011) Threshold Realized Variation Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.mancini_threshold_variation import (
    ManciniThresholdReceipt,
    calculate_mancini_threshold_variation,
    compute_receipt_payload,
    evaluate_mancini_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def test_truncated_variation_exact_partition():
    """Verify that Mancini truncation partitions Total Realized Volatility exactly: RV = TRV + TJ."""
    np.random.seed(42)
    # Simulate mixed returns: mostly Gaussian diffusion with 2 jump spikes
    rets = np.random.normal(0, 0.002, 100)
    rets[25] = 0.035   # 3.5% jump shock
    rets[75] = -0.040  # -4.0% jump shock

    rv, trv, tj, tbv, bv, jump_count, consec_jumps, cutoff_bps = calculate_mancini_threshold_variation(rets)

    assert rv > 0.0
    assert trv > 0.0
    assert tj > 0.0
    assert jump_count >= 2
    assert np.isclose(rv, trv + tj, rtol=1e-5)
    assert cutoff_bps > 0.0


def test_consecutive_jumps_robustness():
    """Verify that Threshold Bipower Variation (TBV) is immune to consecutive jump contamination unlike BV."""
    np.random.seed(42)
    m = 120
    # Clean continuous Gaussian returns
    sigma_true = 0.002
    clean_rets = np.random.normal(0, sigma_true, m)
    true_cont_var = float(np.sum(clean_rets ** 2))

    # Add two consecutive jump shocks at t=40 and t=41
    jump_rets = np.copy(clean_rets)
    jump_rets[40] = 0.030
    jump_rets[41] = 0.035

    rv, trv, tj, tbv, bv, jump_count, consec_jumps, cutoff_bps = calculate_mancini_threshold_variation(jump_rets)

    assert jump_count >= 2
    assert consec_jumps >= 1
    # Classical Bipower Variation product |r_40| * |r_41| causes BV to be heavily inflated by the consecutive jumps
    assert bv > tbv
    # TBV should be close to true continuous variance, whereas BV is contaminated
    assert np.isclose(tbv, true_cont_var, rtol=0.35)
    assert bv > true_cont_var * 1.5


def test_mancini_price_series_evaluation():
    """Test price series evaluation, jump arrival intensity, and regime identification."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0003, 0.012, 100)
    # Inject a single macro announcement shock
    daily_rets[50] = 0.065
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_mancini_from_prices(prices, ticker="SPY")

    assert isinstance(receipt, ManciniThresholdReceipt)
    assert receipt.ticker == "SPY"
    assert receipt.sample_sessions == len(prices) - 1
    assert receipt.total_realized_variance > 0.0
    assert receipt.truncated_continuous_var > 0.0
    assert receipt.jump_count >= 1
    assert receipt.jump_arrival_intensity_ann > 0.0
    assert receipt.regime in (
        "PURE_DIFFUSION_NO_JUMPS",
        "ISOLATED_POISSON_JUMPS",
        "CLUSTERED_JUMP_ACTIVITY",
        "SEVERE_DISCONTINUITY_SHOCK",
    )


def test_mancini_persistence_and_hmac():
    """Verify SQLite audit record persistence and cryptographic HMAC-SHA256 validation."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0002, 0.015, 60)
    prices = 75.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_mancini_from_prices(prices, ticker="QQQ")

    # HMAC validation
    raw_payload = compute_receipt_payload(receipt)
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

