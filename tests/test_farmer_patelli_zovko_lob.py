"""Unit tests for Farmer, Patelli & Zovko (2005) Zero-Intelligence LOB Profile Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.farmer_patelli_zovko_lob import (
    calculate_fpz_liquidity_profile,
    compute_receipt_payload,
    evaluate_fpz_from_order_book,
    persist_as_record,
    verify_hmac_signature,
)


def test_fpz_power_law_depth_monotonicity():
    # 5-level standard ladder
    bids = [200.0, 300.0, 400.0, 450.0, 500.0]
    asks = [220.0, 310.0, 390.0, 440.0, 510.0]
    receipt = calculate_fpz_liquidity_profile(
        mid_price=500.0,
        bid_depths=bids,
        ask_depths=asks,
        tick_size=0.01,
        ticker="SPY",
    )

    assert receipt.ticker == "SPY"
    assert receipt.levels_analyzed == 5
    assert 0.10 <= receipt.depth_exponent_delta <= 1.80
    assert 0.80 <= receipt.placement_exponent_mu <= 2.20
    assert receipt.cumulative_depth_total == sum(0.5 * (b + a) for b, a in zip(bids, asks))

    # Verify cumulative depth is strictly increasing
    assert all(
        receipt.cumulative_depths[i] < receipt.cumulative_depths[i + 1]
        for i in range(len(receipt.cumulative_depths) - 1)
    )


def test_fpz_depth_concentration_regimes():
    # 1. Concentrated touch book: 90% of liquidity at best quotes
    bids_touch = [1000.0, 10.0, 10.0, 10.0, 10.0]
    asks_touch = [1000.0, 10.0, 10.0, 10.0, 10.0]
    rec_touch = calculate_fpz_liquidity_profile(500.0, bids_touch, asks_touch, ticker="QQQ")

    assert rec_touch.touch_depth_ratio > 0.80
    assert rec_touch.depth_exponent_delta < 0.40
    assert rec_touch.regime == "TOUCH_HEAVY_CONCENTRATED_BOOK"

    # 2. Deep support book: volume balloons deeper in the book
    bids_deep = [10.0, 100.0, 500.0, 2000.0, 8000.0]
    asks_deep = [10.0, 100.0, 500.0, 2000.0, 8000.0]
    rec_deep = calculate_fpz_liquidity_profile(500.0, bids_deep, asks_deep, ticker="QQQ")

    assert rec_deep.depth_exponent_delta > 0.85
    assert rec_deep.regime == "DEEP_ACCUMULATED_SUPPORT_BOOK"


def test_fpz_zero_intelligence_spread_scaling():
    bids = [300.0, 400.0, 500.0, 600.0, 700.0]
    asks = [300.0, 400.0, 500.0, 600.0, 700.0]

    # Baseline market clearing
    rec_base = calculate_fpz_liquidity_profile(
        mid_price=100.0,
        bid_depths=bids,
        ask_depths=asks,
        limit_arrival_rate_alpha=200.0,
        market_order_rate_mu=100.0,
    )

    # Higher aggressive market order flow mu -> wider theoretical spread
    rec_heavy_flow = calculate_fpz_liquidity_profile(
        mid_price=100.0,
        bid_depths=bids,
        ask_depths=asks,
        limit_arrival_rate_alpha=200.0,
        market_order_rate_mu=400.0,
    )

    assert rec_heavy_flow.zi_equilibrium_spread_bps > rec_base.zi_equilibrium_spread_bps


def test_fpz_persistence_and_hmac():
    receipt = evaluate_fpz_from_order_book(
        mid_price=450.0,
        bid_depth_touch=350.0,
        ask_depth_touch=280.0,
        actual_spread=0.04,
        ticker="AAPL",
    )

    assert receipt.ticker == "AAPL"
    assert len(receipt.hmac_signature) == 64

    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "_tampered", receipt.hmac_signature) is False

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_fpz.db")
        success = persist_as_record(receipt, test_db)
        assert success is True

