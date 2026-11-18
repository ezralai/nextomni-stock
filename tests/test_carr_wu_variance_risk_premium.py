"""Unit tests for Carr & Wu (2009) Variance Risk Premium Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.carr_wu_variance_risk_premium import (
    CarrWuVRPReceipt,
    calculate_carr_wu_vrp,
    compute_receipt_payload,
    evaluate_vrp_from_market_data,
    persist_as_record,
    verify_hmac_signature,
)


def test_carr_wu_vrp_positive_baseline():
    """Verify standard positive Variance Risk Premium dynamics (IV > RV)."""
    # 20% implied vol vs 15% realized vol
    receipt = calculate_carr_wu_vrp(
        implied_vol=0.20,
        realized_vol_forecast=0.15,
        ticker="SPY",
        horizon_days=22,
    )

    assert isinstance(receipt, CarrWuVRPReceipt)
    assert receipt.ticker == "SPY"
    assert receipt.implied_variance == pytest.approx(0.04, rel=1e-4)
    assert receipt.realized_variance == pytest.approx(0.0225, rel=1e-4)
    assert receipt.variance_risk_premium > 0.0
    assert receipt.volatility_spread_pts == pytest.approx(5.0, rel=1e-3)
    assert receipt.short_variance_swap_expected_pnl_pct > 0.0
    assert receipt.expected_equity_excess_return_bps > 0.0
    assert receipt.regime in ("NORMAL_POSITIVE_RISK_PREMIUM", "ELEVATED_TAIL_FEAR")


def test_carr_wu_vrp_inverted_crisis_regime():
    """Verify inverted VRP behavior during severe market shocks (RV > IV)."""
    # 22% implied vol vs 35% trailing realized crash vol
    receipt = calculate_carr_wu_vrp(
        implied_vol=0.22,
        realized_vol_forecast=0.35,
        ticker="SPY",
    )

    assert receipt.variance_risk_premium < 0.0
    assert receipt.volatility_spread_pts < 0.0
    assert receipt.log_vrp < 0.0
    assert receipt.short_variance_swap_expected_pnl_pct < 0.0
    assert receipt.regime == "INVERTED_CRISIS_EXPANSION"


def test_vrp_from_price_series_evaluation():
    """Test evaluation of VRP from price series with and without explicit implied vol."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0004, 0.012, 100)
    prices = 100.0 * np.exp(np.cumsum(daily_rets))

    # 1. With explicit market implied vol
    rec_explicit = evaluate_vrp_from_market_data(prices, implied_vol=0.22, ticker="SPY")
    assert rec_explicit.implied_vol_ann == 0.22
    assert rec_explicit.derivation_mode == "MARKET_OBSERVED_IMPLIED_VOL"

    # 2. With proxy surface markup
    rec_proxy = evaluate_vrp_from_market_data(prices, ticker="QQQ")
    assert rec_proxy.derivation_mode == "SURFACE_PROXY_IMPLIED_VOL"
    assert rec_proxy.variance_risk_premium > 0.0
    assert rec_proxy.ticker == "QQQ"


def test_carr_wu_vrp_persistence_and_hmac():
    """Verify SQLite audit record persistence and cryptographic HMAC-SHA256 validation."""
    np.random.seed(42)
    daily_rets = np.random.normal(0.0002, 0.014, 60)
    prices = 80.0 * np.exp(np.cumsum(daily_rets))

    receipt = evaluate_vrp_from_market_data(prices, ticker="IWM")

    # HMAC validation
    raw_payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(raw_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(raw_payload + "_tampered", receipt.hmac_signature) is False

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        temp_db = tf.name

    try:
        success = persist_as_record(receipt, db_path=temp_db)
        assert success is True
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)
