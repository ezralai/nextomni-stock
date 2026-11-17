"""Unit tests for Andersen, Bollerslev & Diebold (2007) HAR-RV-J Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.andersen_har_rv_jump import (
    AndersenHARJReceipt,
    calculate_andersen_har_rv_j,
    decompose_single_session_jumps,
    evaluate_har_j_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def test_decompose_single_session_jumps():
    """Verify single-session decomposition of RV into Bipower Variation, Continuous diffusion, and Jumps."""
    np.random.seed(42)
    # 1. Continuous Gaussian returns (no jump)
    normal_rets = np.random.normal(0.0, 0.002, 78)  # ~5-minute intervals for 6.5 hour session
    rv_norm, bv_norm, c_norm, j_norm, z_norm = decompose_single_session_jumps(normal_rets)

    assert rv_norm > 0.0
    assert bv_norm > 0.0
    assert c_norm > 0.0
    # On pure gaussian diffusion, jump variance should be 0 or small
    assert j_norm == 0.0 or j_norm / rv_norm < 0.2
    assert np.isclose(rv_norm, c_norm + j_norm, rtol=1e-5)

    # 2. Add an explicit large jump shock
    jump_rets = np.copy(normal_rets)
    jump_rets[35] = 0.045  # massive 4.5% jump in a single 5-min bar
    rv_jump, bv_jump, c_jump, j_jump, z_jump = decompose_single_session_jumps(jump_rets)

    assert rv_jump > rv_norm * 2.0
    assert j_jump > 0.0
    assert z_jump > 2.0  # Significant jump test statistic
    assert np.isclose(rv_jump, c_jump + j_jump, rtol=1e-5)


def test_andersen_har_j_continuous_persistence_dominance():
    """Verify empirical stylized fact that continuous diffusion persistence exceeds jump persistence."""
    np.random.seed(42)
    n = 100
    base_rv = (0.16 / np.sqrt(252)) ** 2
    # Persistent continuous component
    continuous_series = np.zeros(n)
    continuous_series[0] = base_rv
    for t in range(1, n):
        continuous_series[t] = 0.05 * base_rv + 0.85 * continuous_series[t - 1] + np.random.normal(0.0, 0.1 * base_rv)
    continuous_series = np.maximum(continuous_series, 1e-8)

    # Transient, unpersistent jumps
    jumps_series = np.zeros(n)
    jump_indices = [15, 38, 62, 85]
    for idx in jump_indices:
        jumps_series[idx] = 3.0 * base_rv

    rv_series = continuous_series + jumps_series

    receipt = calculate_andersen_har_rv_j(
        daily_rv=rv_series,
        daily_bv=continuous_series,
        daily_jumps=jumps_series,
        ticker="SPY",
        use_log=True,
    )

    assert isinstance(receipt, AndersenHARJReceipt)
    assert receipt.sample_sessions == n
    assert receipt.continuous_persistence > receipt.beta_jump
    assert receipt.jump_ratio_mean > 0.0
    assert receipt.forecast_1d_ann_vol > 0.0
    assert receipt.forecast_5d_ann_vol > 0.0
    assert receipt.forecast_22d_ann_vol > 0.0


def test_andersen_har_j_log_vs_linear_specification():
    """Test both logarithmic and linear HAR-RV-J formulations on simulated asset prices."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0005, 0.015, 120)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    # Log specification
    rec_log = evaluate_har_j_from_prices(prices, ticker="QQQ", use_log=True)
    # Linear specification
    rec_lin = evaluate_har_j_from_prices(prices, ticker="QQQ", use_log=False)

    assert rec_log.ticker == "QQQ"
    assert rec_lin.ticker == "QQQ"
    assert rec_log.sample_sessions == len(prices) - 1
    assert rec_lin.sample_sessions == len(prices) - 1
    assert 0.0 < rec_log.forecast_1d_ann_vol < 2.0
    assert 0.0 < rec_lin.forecast_1d_ann_vol < 2.0


def test_andersen_har_j_persistence_and_hmac():
    """Verify SQLite audit record persistence and HMAC-SHA256 signature sealing."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0002, 0.012, 80)
    prices = 50.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_har_j_from_prices(prices, ticker="IWM", use_log=True)

    # Validate HMAC signature
    raw_payload = (
        f"{receipt.ticker}|{receipt.sample_sessions}|{receipt.beta_continuous_daily:.4f}|"
        f"{receipt.beta_continuous_weekly:.4f}|{receipt.beta_continuous_monthly:.4f}|"
        f"{receipt.beta_jump:.4f}|{receipt.r_squared:.4f}|{receipt.jump_ratio_mean:.4f}|"
        f"{receipt.regime}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(raw_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(raw_payload + "tampered", receipt.hmac_signature) is False

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        success = persist_as_record(receipt, db_path=temp_db)
        assert success is True
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)

