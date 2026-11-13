"""Unit tests for Rama Cont, Sasha Stoikov & Rishi Talreja (2010) Markovian LOB Queue Engine."""

import os
import tempfile
import pytest

from src.analytics.cont_stoikov_lob_queue import (
    calculate_cont_stoikov_lob,
    compute_receipt_payload,
    evaluate_cont_stoikov_from_quotes,
    persist_as_record,
    verify_hmac_signature,
)


def test_cont_stoikov_order_imbalance_monotonicity():
    # Symmetrical queues: x = 100, y = 100
    res_balanced = calculate_cont_stoikov_lob(
        best_bid=500.0,
        best_ask=500.05,
        bid_depth=100.0,
        ask_depth=100.0,
        ticker="SPY",
    )
    assert abs(res_balanced.prob_up_tick - 0.5) < 0.05
    assert res_balanced.order_imbalance_ratio == 0.0
    assert res_balanced.regime == "BALANCED_ORDER_BOOK"

    # Excess bid depth (heavy bid queue protects bid, ask depleted first): x = 300, y = 100
    res_buy = calculate_cont_stoikov_lob(
        best_bid=500.0,
        best_ask=500.05,
        bid_depth=300.0,
        ask_depth=100.0,
        ticker="SPY",
    )
    assert res_buy.prob_up_tick > 0.60
    assert res_buy.prob_down_tick < 0.40
    assert res_buy.order_imbalance_ratio == 0.5
    assert res_buy.regime == "STRONG_BUY_PRESSURE"

    # Excess ask depth (heavy ask queue protects ask, bid depleted first): x = 100, y = 300
    res_sell = calculate_cont_stoikov_lob(
        best_bid=500.0,
        best_ask=500.05,
        bid_depth=100.0,
        ask_depth=300.0,
        ticker="SPY",
    )
    assert res_sell.prob_up_tick < 0.40
    assert res_sell.prob_down_tick > 0.60
    assert res_sell.order_imbalance_ratio == -0.5
    assert res_sell.regime == "STRONG_SELL_PRESSURE"


def test_cont_stoikov_priority_fill_decay():
    # As queue priority depth increases, fill probability must monotonically decrease
    res_p1 = calculate_cont_stoikov_lob(500.0, 500.05, 200.0, 200.0, order_priority_k=1)
    res_p5 = calculate_cont_stoikov_lob(500.0, 500.05, 200.0, 200.0, order_priority_k=5)
    res_p20 = calculate_cont_stoikov_lob(500.0, 500.05, 200.0, 200.0, order_priority_k=20)

    assert res_p1.bid_fill_prob > res_p5.bid_fill_prob > res_p20.bid_fill_prob
    assert 0.0 <= res_p20.bid_fill_prob <= 1.0


def test_cont_stoikov_boundary_conditions():
    # Depleted ask queue -> immediate upward tick
    res_empty_ask = calculate_cont_stoikov_lob(500.0, 500.05, 150.0, 0.0)
    assert res_empty_ask.prob_up_tick == 1.0
    assert res_empty_ask.prob_down_tick == 0.0
    assert res_empty_ask.expected_duration_sec == 0.0

    # Depleted bid queue -> immediate downward tick
    res_empty_bid = calculate_cont_stoikov_lob(500.0, 500.05, 0.0, 150.0)
    assert res_empty_bid.prob_up_tick == 0.0
    assert res_empty_bid.prob_down_tick == 1.0
    assert res_empty_bid.expected_duration_sec == 0.0

    # Inverted quotes -> empty receipt
    res_invalid = calculate_cont_stoikov_lob(500.10, 500.05, 100.0, 100.0)
    assert res_invalid.derivation_mode == "FAILED"
    assert "ERROR" in res_invalid.regime


def test_cont_stoikov_persistence_and_hmac():
    receipt = evaluate_cont_stoikov_from_quotes(
        best_bid=450.0,
        best_ask=450.08,
        bid_depth=250.0,
        ask_depth=120.0,
        ticker="QQQ",
    )
    assert receipt.ticker == "QQQ"
    assert len(receipt.hmac_signature) == 64

    # Verification payload via canonical helper
    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "_tampered", receipt.hmac_signature) is False

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_lob.db")
        success = persist_as_record(receipt, test_db)
        assert success is True
