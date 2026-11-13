"""Unit tests for Marco Avellaneda & Sasha Stoikov (2008) Market Making Engine."""

import sqlite3
import numpy as np
import pytest

from src.analytics.avellaneda_stoikov_market_making import (
    AvellanedaStoikovReceipt,
    calculate_avellaneda_stoikov,
    evaluate_avellaneda_stoikov_from_prices,
    persist_as_record,
    verify_hmac_signature,
)


def test_avellaneda_stoikov_inventory_skew():
    """Verify reservation price and asymmetric quote skew under long vs short inventory."""
    mid_price = 100.0
    vol = 0.015
    gamma = 0.10

    # 1. Neutral inventory (q = 0)
    rec_neutral = calculate_avellaneda_stoikov(
        mid_price=mid_price,
        inventory=0,
        volatility=vol,
        risk_aversion=gamma,
        ticker="AS_NEUTRAL",
    )
    assert rec_neutral.reservation_price == mid_price
    assert rec_neutral.inventory_skew_dollar == 0.0
    assert abs(rec_neutral.bid_depth_delta - rec_neutral.ask_depth_delta) < 1e-6
    assert abs(rec_neutral.bid_fill_probability - rec_neutral.ask_fill_probability) < 1e-4
    assert rec_neutral.regime == "BALANCED_MARKET_MAKING"

    # 2. Long inventory (q = +5) -> should skew quotes downward
    rec_long = calculate_avellaneda_stoikov(
        mid_price=mid_price,
        inventory=5,
        volatility=vol,
        risk_aversion=gamma,
        ticker="AS_LONG",
    )
    assert rec_long.reservation_price < mid_price
    assert rec_long.inventory_skew_dollar < 0.0
    assert rec_long.optimal_ask_price < rec_neutral.optimal_ask_price
    assert rec_long.optimal_bid_price < rec_neutral.optimal_bid_price
    # Lower ask price increases probability of selling to shed inventory
    assert rec_long.ask_fill_probability > rec_long.bid_fill_probability
    assert rec_long.regime in ("LONG_INVENTORY_SKEW_DOWN", "EXTREME_INVENTORY_RISK")

    # 3. Short inventory (q = -5) -> should skew quotes upward
    rec_short = calculate_avellaneda_stoikov(
        mid_price=mid_price,
        inventory=-5,
        volatility=vol,
        risk_aversion=gamma,
        ticker="AS_SHORT",
    )
    assert rec_short.reservation_price > mid_price
    assert rec_short.inventory_skew_dollar > 0.0
    assert rec_short.optimal_bid_price > rec_neutral.optimal_bid_price
    assert rec_short.optimal_ask_price > rec_neutral.optimal_ask_price
    # Higher bid price increases probability of buying to cover short
    assert rec_short.bid_fill_probability > rec_short.ask_fill_probability
    assert rec_short.regime in ("SHORT_INVENTORY_SKEW_UP", "EXTREME_INVENTORY_RISK")


def test_avellaneda_stoikov_spread_sensitivity():
    """Verify optimal spread widens with volatility and risk aversion."""
    mid_price = 150.0

    rec_low_risk = calculate_avellaneda_stoikov(
        mid_price=mid_price,
        inventory=0,
        volatility=0.01,
        risk_aversion=0.02,
        kappa=2.0,
        ticker="LOW_RISK",
    )

    rec_high_risk = calculate_avellaneda_stoikov(
        mid_price=mid_price,
        inventory=0,
        volatility=0.04,
        risk_aversion=0.20,
        kappa=1.0,
        ticker="HIGH_RISK",
    )

    assert rec_high_risk.optimal_spread_dollar > 0.0
    assert rec_low_risk.optimal_spread_dollar > 0.0
    assert rec_high_risk.optimal_spread_bps > 0.0


def test_avellaneda_stoikov_hmac_tamper_detection():
    """Verify cryptographic HMAC-SHA256 signature detects tampering."""
    receipt = calculate_avellaneda_stoikov(mid_price=100.0, inventory=2, ticker="AUTHENTIC_AS")

    payload = (
        f"{receipt.ticker}|{receipt.mid_price:.4f}|{receipt.inventory_q}|{receipt.volatility_sigma:.6f}|"
        f"{receipt.risk_aversion_gamma:.4f}|{receipt.order_intensity_kappa:.4f}|{receipt.horizon_tau:.4f}|"
        f"{receipt.reservation_price:.4f}|{receipt.inventory_skew_dollar:.4f}|{receipt.optimal_bid_price:.4f}|"
        f"{receipt.optimal_ask_price:.4f}|{receipt.optimal_spread_dollar:.4f}|{receipt.optimal_spread_bps:.2f}|"
        f"{receipt.bid_fill_probability:.4f}|{receipt.ask_fill_probability:.4f}|{receipt.inventory_risk_penalty:.4f}|"
        f"{receipt.regime}|{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered payload fails
    tampered = payload.replace("AUTHENTIC_AS", "FORGED_AS")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_avellaneda_stoikov_sqlite_persistence(tmp_path):
    """Verify SQLite database persistence and schema creation."""
    db_file = str(tmp_path / "test_as.db")
    prices = [100.0 + 0.1 * i for i in range(25)]

    receipt = evaluate_avellaneda_stoikov_from_prices(prices, inventory=3, ticker="PERSIST_AS")
    assert persist_as_record(receipt, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, mid_price, reservation_price, regime, hmac_signature FROM avellaneda_stoikov_receipts WHERE ticker = ?",
            ("PERSIST_AS",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "PERSIST_AS"
        assert abs(row[1] - receipt.mid_price) < 1e-4
        assert abs(row[2] - receipt.reservation_price) < 1e-4
        assert row[3] == receipt.regime
        assert row[4] == receipt.hmac_signature

