"""Unit tests for David A. Lesmond, Joseph P. Ogden & Charles A. Trzcinka (1999) LOT Engine."""

import sqlite3
import numpy as np
import pytest

from src.analytics.lesmond_ogden_trzcinka_spread import (
    LOTReceipt,
    calculate_lot_spread,
    evaluate_lot_from_prices,
    persist_lot_record,
    verify_hmac_signature,
)


def test_lot_synthetic_dgp_recovery():
    """Verify MLE recovery of true buy/sell thresholds under synthetic LOT LDV model."""
    rng = np.random.default_rng(42)
    T = 600
    true_beta = 1.15
    true_sigma = 0.012
    true_a1 = -0.0060  # -60 bps sell threshold
    true_a2 = 0.0070   # +70 bps buy threshold
    # Total round trip: 130 bps

    r_m = rng.normal(0.0004, 0.010, size=T)
    e = rng.normal(0.0, true_sigma, size=T)
    r_star = true_beta * r_m + e

    # Apply LOT double-hurdle censoring
    r_obs = np.zeros(T)
    for t in range(T):
        if r_star[t] < true_a1:
            r_obs[t] = r_star[t] - true_a1
        elif r_star[t] > true_a2:
            r_obs[t] = r_star[t] - true_a2
        else:
            r_obs[t] = 0.0

    receipt = calculate_lot_spread(
        stock_returns=r_obs,
        market_returns=r_m,
        ticker="LOT_DGP",
    )

    assert isinstance(receipt, LOTReceipt)
    assert receipt.ticker == "LOT_DGP"
    assert receipt.sample_size == T
    assert receipt.zero_return_count > 0
    assert 0.10 <= receipt.zero_return_frequency <= 0.60

    # Verify recovered thresholds
    assert -0.015 <= receipt.alpha1_sell_threshold <= -0.001
    assert 0.001 <= receipt.alpha2_buy_threshold <= 0.018
    # Round trip spread should be within reasonable proximity to 130 bps (70 to 220 bps)
    assert 60.0 <= receipt.round_trip_spread_bps <= 250.0
    assert 0.8 <= receipt.market_beta <= 1.6
    assert 0.005 <= receipt.idiosyncratic_sigma <= 0.025

    # Cryptographic HMAC verification
    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.zero_return_count}|{receipt.zero_return_frequency:.4f}|"
        f"{receipt.alpha1_sell_threshold:.6f}|{receipt.alpha2_buy_threshold:.6f}|"
        f"{receipt.round_trip_spread_pct:.6f}|{receipt.round_trip_spread_bps:.2f}|"
        f"{receipt.buy_cost_bps:.2f}|{receipt.sell_cost_bps:.2f}|{receipt.market_beta:.4f}|"
        f"{receipt.idiosyncratic_sigma:.6f}|{receipt.log_likelihood:.2f}|{receipt.regime}|"
        f"{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True


def test_lot_zero_return_frequency_monotonicity():
    """Verify that wider transaction cost bands increase zero return count and spread."""
    rng = np.random.default_rng(999)
    T = 400
    r_m = rng.normal(0.0002, 0.008, size=T)
    e = rng.normal(0.0, 0.010, size=T)
    r_star = 1.0 * r_m + e

    # Narrow band (tight liquidity)
    r_tight = np.zeros(T)
    for t in range(T):
        if r_star[t] < -0.002:
            r_tight[t] = r_star[t] - (-0.002)
        elif r_star[t] > 0.002:
            r_tight[t] = r_star[t] - 0.002
        else:
            r_tight[t] = 0.0

    # Wide band (illiquid asset)
    r_wide = np.zeros(T)
    for t in range(T):
        if r_star[t] < -0.012:
            r_wide[t] = r_star[t] - (-0.012)
        elif r_star[t] > 0.012:
            r_wide[t] = r_star[t] - 0.012
        else:
            r_wide[t] = 0.0

    rec_tight = calculate_lot_spread(r_tight, r_m, ticker="TIGHT")
    rec_wide = calculate_lot_spread(r_wide, r_m, ticker="WIDE")

    # Wide band should have strictly more zero return days and higher estimated spread
    assert rec_wide.zero_return_count > rec_tight.zero_return_count
    assert rec_wide.zero_return_frequency > rec_tight.zero_return_frequency
    assert rec_wide.round_trip_spread_bps > rec_tight.round_trip_spread_bps


def test_lot_hmac_tamper_detection():
    """Verify cryptographic HMAC-SHA256 signature detects tampering."""
    prices = [100.0 * (1.001 ** i) if i % 4 != 0 else 100.0 * (1.001 ** (i - 1)) for i in range(50)]
    receipt = evaluate_lot_from_prices(prices, ticker="AUTHENTIC_LOT")

    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.zero_return_count}|{receipt.zero_return_frequency:.4f}|"
        f"{receipt.alpha1_sell_threshold:.6f}|{receipt.alpha2_buy_threshold:.6f}|"
        f"{receipt.round_trip_spread_pct:.6f}|{receipt.round_trip_spread_bps:.2f}|"
        f"{receipt.buy_cost_bps:.2f}|{receipt.sell_cost_bps:.2f}|{receipt.market_beta:.4f}|"
        f"{receipt.idiosyncratic_sigma:.6f}|{receipt.log_likelihood:.2f}|{receipt.regime}|"
        f"{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered payload fails
    tampered = payload.replace("AUTHENTIC_LOT", "TAMPERED_LOT")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_lot_sqlite_persistence(tmp_path):
    """Verify SQLite database persistence and schema creation."""
    db_file = str(tmp_path / "test_lot.db")
    prices = [100.0, 100.0, 100.5, 100.5, 99.8, 99.8, 100.2, 100.2, 100.4, 100.4] * 5

    receipt = evaluate_lot_from_prices(prices, ticker="LOT_STORE")
    assert persist_lot_record(receipt, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, round_trip_spread_bps, zero_return_count, regime, hmac_signature FROM lot_spread_receipts WHERE ticker = ?",
            ("LOT_STORE",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "LOT_STORE"
        assert abs(row[1] - receipt.round_trip_spread_bps) < 1e-4
        assert row[2] == receipt.zero_return_count
        assert row[3] == receipt.regime
        assert row[4] == receipt.hmac_signature

