"""Unit tests for Robert Almgren, Chee Thum, Emmanuel Hauptmann & Hong Li (2005) Metaorder Impact Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.almgren_thum_metaorder import (
    calculate_almgren_thum_impact,
    compute_receipt_payload,
    evaluate_almgren_thum_from_order,
    persist_as_record,
    verify_hmac_signature,
)


def test_almgren_thum_sublinear_velocity_scaling():
    # Base: Q = 100,000 shares, ADV = 1,000,000 shares (10% ADV), horizon = 1.0 day
    rec_base = calculate_almgren_thum_impact(
        price=100.0,
        order_shares=100000.0,
        adv_shares=1000000.0,
        daily_volatility=0.02,
        horizon_days=1.0,
        alpha_exp=0.60,
        ticker="SPY",
    )

    # Fast: Same order executed in 0.5 days (2x trading velocity)
    rec_fast = calculate_almgren_thum_impact(
        price=100.0,
        order_shares=100000.0,
        adv_shares=1000000.0,
        daily_volatility=0.02,
        horizon_days=0.5,
        alpha_exp=0.60,
        ticker="SPY",
    )

    # Permanent impact must remain identical regardless of execution horizon T
    assert abs(rec_fast.permanent_impact_bps - rec_base.permanent_impact_bps) < 1e-4

    # Temporary impact must scale as 2^(0.60) = 1.5157x
    ratio_temp = rec_fast.temporary_impact_bps / rec_base.temporary_impact_bps
    expected_ratio = 2.0 ** 0.60
    assert abs(ratio_temp - expected_ratio) < 0.02

    # Expected shortfall must be higher for the faster execution
    assert rec_fast.expected_shortfall_bps > rec_base.expected_shortfall_bps
    # Timing variance must be lower for faster execution (less exposure to market risk)
    assert rec_fast.variance_shortfall_dollar < rec_base.variance_shortfall_dollar


def test_almgren_thum_linear_permanent_impact():
    # Metaorder 1: 50,000 shares (5% ADV)
    rec1 = calculate_almgren_thum_impact(
        price=50.0,
        order_shares=50000.0,
        adv_shares=1000000.0,
        daily_volatility=0.015,
        horizon_days=1.0,
        ticker="QQQ",
    )

    # Metaorder 2: 100,000 shares (10% ADV)
    rec2 = calculate_almgren_thum_impact(
        price=50.0,
        order_shares=100000.0,
        adv_shares=1000000.0,
        daily_volatility=0.015,
        horizon_days=1.0,
        ticker="QQQ",
    )

    # Permanent impact must scale linearly with order size: rec2 == 2 * rec1
    ratio_perm = rec2.permanent_impact_bps / rec1.permanent_impact_bps
    assert abs(ratio_perm - 2.0) < 0.01


def test_almgren_thum_risk_aversion_horizon_contraction():
    # Low risk aversion: patient trader prioritizes minimizing slippage
    rec_patient = calculate_almgren_thum_impact(
        price=100.0,
        order_shares=200000.0,
        adv_shares=1000000.0,
        daily_volatility=0.025,
        risk_aversion=1e-7,
        ticker="AAPL",
    )

    # High risk aversion: urgent trader prioritizes minimizing volatility exposure
    rec_urgent = calculate_almgren_thum_impact(
        price=100.0,
        order_shares=200000.0,
        adv_shares=1000000.0,
        daily_volatility=0.025,
        risk_aversion=1e-4,
        ticker="AAPL",
    )

    # Optimal horizon must contract when risk aversion increases
    assert rec_urgent.optimal_horizon_days < rec_patient.optimal_horizon_days
    # Recommended participation rate (% ADV) must be higher for urgent trader
    assert rec_urgent.optimal_participation_pct > rec_patient.optimal_participation_pct


def test_almgren_thum_persistence_and_hmac():
    rng = np.random.default_rng(42)
    prices = 150.0 + np.cumsum(rng.normal(0, 0.5, size=40))
    volumes = rng.uniform(800000, 1200000, size=40)

    receipt = evaluate_almgren_thum_from_order(
        prices=prices,
        volumes=volumes,
        order_shares=50000.0,
        ticker="MSFT",
        horizon_days=0.5,
    )

    assert receipt.ticker == "MSFT"
    assert len(receipt.hmac_signature) == 64

    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "_tampered", receipt.hmac_signature) is False

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_almgren_thum.db")
        success = persist_as_record(receipt, test_db)
        assert success is True
