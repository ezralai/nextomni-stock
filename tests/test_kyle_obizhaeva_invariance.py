"""Unit tests for Albert S. Kyle & Anna A. Obizhaeva (2016) Market Microstructure Invariance Engine."""

import os
import tempfile
import numpy as np
import pytest

from src.analytics.kyle_obizhaeva_invariance import (
    calculate_kyle_obizhaeva_invariance,
    compute_receipt_payload,
    evaluate_kyle_obizhaeva_from_market_data,
    persist_as_record,
    verify_hmac_signature,
)


def test_kyle_obizhaeva_scaling_laws():
    # Case 1: Standard stock
    v1 = 1e8      # $100M daily volume
    sig1 = 0.02   # 2% volatility
    rec1 = calculate_kyle_obizhaeva_invariance(price=100.0, daily_volume_dollar=v1, daily_volatility=sig1, ticker="SPY")

    # Case 2: 8x business risk W (e.g. 8x volume, same volatility)
    v2 = 8e8
    rec2 = calculate_kyle_obizhaeva_invariance(price=100.0, daily_volume_dollar=v2, daily_volatility=sig1, ticker="SPY")

    # Scaling law: N* scales as W^(2/3). If W is 8x larger, N* should be 8^(2/3) = 4x larger!
    ratio_n = rec2.invariant_trades_n / rec1.invariant_trades_n
    assert abs(ratio_n - 4.0) < 0.05

    # Scaling law for spread: S* scales as V^(-1/3). If V is 8x larger, S* should be 8^(-1/3) = 0.5x (halved)!
    ratio_s = rec2.invariant_spread_bps / rec1.invariant_spread_bps
    assert abs(ratio_s - 0.5) < 0.02


def test_kyle_obizhaeva_market_depth_and_trade_size():
    p = 50.0
    v = 5e7
    sig = 0.015
    rec = calculate_kyle_obizhaeva_invariance(price=p, daily_volume_dollar=v, daily_volatility=sig, ticker="QQQ")

    assert rec.invariant_trades_n > 0
    assert rec.invariant_trade_size_dollar > 0
    # Q* = V / N*
    assert abs(rec.invariant_trade_size_dollar - (v / rec.invariant_trades_n)) < 1.0
    assert rec.invariant_depth_dollar > 0
    assert rec.invariant_spread_bps > 0


def test_kyle_obizhaeva_friction_and_slicing_regimes():
    p = 100.0
    v = 1e8
    sig = 0.02
    base_rec = calculate_kyle_obizhaeva_invariance(price=p, daily_volume_dollar=v, daily_volatility=sig)
    base_spread = base_rec.invariant_spread_bps
    base_n = base_rec.invariant_trades_n

    # High actual spread -> high friction
    rec_fric = calculate_kyle_obizhaeva_invariance(
        price=p, daily_volume_dollar=v, daily_volatility=sig, actual_spread_bps=base_spread * 2.0
    )
    assert rec_fric.friction_ratio > 1.8
    assert rec_fric.regime == "HIGH_FRICTION_SPREAD_WIDENING"

    # High actual trade count -> intense algorithmic slicing
    rec_slice = calculate_kyle_obizhaeva_invariance(
        price=p, daily_volume_dollar=v, daily_volatility=sig, actual_trades=int(base_n * 3.5)
    )
    assert rec_slice.slicing_intensity > 3.0
    assert rec_slice.regime == "INTENSE_ALGORITHMIC_SLICING"


def test_kyle_obizhaeva_persistence_and_hmac():
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, size=50))
    volumes = rng.uniform(500000, 1500000, size=50)

    receipt = evaluate_kyle_obizhaeva_from_market_data(
        prices=prices,
        volumes=volumes,
        ticker="AAPL",
        actual_spread_bps=2.5,
    )
    assert receipt.ticker == "AAPL"
    assert len(receipt.hmac_signature) == 64

    payload = compute_receipt_payload(receipt)
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "_tampered", receipt.hmac_signature) is False

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_kyle_obizhaeva.db")
        success = persist_as_record(receipt, test_db)
        assert success is True

