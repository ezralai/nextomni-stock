"""Unit tests for Bruno Biais, Pierre Hillion & Chester Spatt (1995) LOB Order Flow Engine."""

import os
import tempfile
import pytest

from src.analytics.biais_hillion_spatt_lob import (
    calculate_bhs_transition_matrix,
    compute_receipt_payload,
    evaluate_bhs_from_event_stream,
    persist_as_record,
    verify_hmac_signature,
)


def test_bhs_stochastic_properties_and_stationary_vector():
    # Benchmark simulation
    receipt = evaluate_bhs_from_event_stream(ticker="SPY")
    assert receipt.ticker == "SPY"
    assert receipt.sample_events > 100
    assert len(receipt.transition_matrix) == 6

    # Verify every row sums to 1.0
    for row in receipt.transition_matrix:
        assert len(row) == 6
        assert abs(sum(row) - 1.0) < 1e-3

    # Verify stationary vector sums to 1.0
    assert len(receipt.stationary_distribution) == 6
    assert abs(sum(receipt.stationary_distribution) - 1.0) < 1e-3
    assert 0.0 < receipt.shannon_entropy_bits <= 2.585


def test_bhs_diagonal_effect_and_clustering():
    # Construct sequence with extreme clustering: 10 MBs, then 10 MSs, then 10 LBs, etc.
    clustered_events = (
        ["MB"] * 15 +
        ["MS"] * 15 +
        ["LB"] * 15 +
        ["LS"] * 15 +
        ["CB"] * 15 +
        ["CS"] * 15
    )
    receipt = calculate_bhs_transition_matrix(clustered_events, ticker="QQQ", laplace_alpha=0.1)

    assert receipt.diagonal_persistence_ratio > 2.5
    assert receipt.clustering_intensity > 0.4
    assert receipt.regime == "HIGH_ORDER_CLUSTERING"


def test_bhs_liquidity_replenishment():
    # Construct sequence where market hits are always followed by limit quote additions
    # MB -> LB, MS -> LS
    replenish_events = ["MB", "LB", "MS", "LS"] * 25
    receipt = calculate_bhs_transition_matrix(replenish_events, ticker="AAPL", laplace_alpha=0.1)

    assert receipt.liquidity_replenishment_ratio > 1.25
    assert receipt.regime == "RAPID_LIQUIDITY_REPLENISHMENT"


def test_bhs_persistence_and_hmac():
    receipt = evaluate_bhs_from_event_stream(ticker="MSFT")
    assert receipt.ticker == "MSFT"
    assert len(receipt.hmac_signature) == 64

    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "_tampered", receipt.hmac_signature) is False

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_bhs.db")
        success = persist_as_record(receipt, test_db)
        assert success is True

