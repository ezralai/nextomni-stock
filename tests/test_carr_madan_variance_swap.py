"""Tests for Carr & Madan (1999) Variance Swap Fair Strike Pricing Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.carr_madan_variance_swap import (
    _build_hmac_payload,
    calculate_carr_madan_variance_swap,
    evaluate_variance_swap_profile,
    persist_variance_swap_record,
    verify_hmac_signature,
)


def test_flat_volatility_convergence_to_black_scholes() -> None:
    """Under zero skew, Carr-Madan replication converges to flat Black-Scholes vol."""
    atm_vol = 0.20
    rec = calculate_carr_madan_variance_swap(
        spot_price=100.0,
        time_to_maturity=0.50,
        risk_free_rate=0.03,
        atm_implied_vol=atm_vol,
        vol_skew_slope=0.0,
        realized_vol=0.18,
        n_strikes=300,
        ticker="SPY_FLAT",
    )
    # Fair strike volatility closely matches the input flat IV
    assert rec.fair_vol_strike == pytest.approx(atm_vol, rel=0.02)
    assert rec.fair_variance_strike == pytest.approx(atm_vol ** 2, rel=0.04)
    assert rec.vol_risk_premium == pytest.approx(rec.fair_vol_strike - 0.18, abs=1e-6)


def test_negative_skew_boosts_put_integral_and_vrp() -> None:
    """Negative equity skew increases OTM put contribution and lifts fair strike above ATM vol."""
    rec_skew = calculate_carr_madan_variance_swap(
        spot_price=100.0,
        time_to_maturity=0.25,
        risk_free_rate=0.04,
        atm_implied_vol=0.20,
        vol_skew_slope=-0.15,
        realized_vol=0.16,
        n_strikes=250,
        ticker="SPY_SKEW",
    )
    assert rec_skew.fair_vol_strike > rec_skew.atm_implied_vol
    assert rec_skew.put_call_variance_ratio > 1.0
    assert rec_skew.vol_risk_premium > 0.04
    assert rec_skew.regime == "HIGH_VOL_RISK_PREMIUM"

    # Crisis environment where realized vol exceeds implied fair strike
    rec_crisis = calculate_carr_madan_variance_swap(
        spot_price=100.0,
        time_to_maturity=0.25,
        risk_free_rate=0.04,
        atm_implied_vol=0.25,
        vol_skew_slope=-0.15,
        realized_vol=0.35,
        n_strikes=250,
        ticker="CRISIS_VOL",
    )
    assert rec_crisis.vol_risk_premium < 0.0
    assert rec_crisis.regime == "INVERTED_CRISIS_SPREAD"


def test_carr_madan_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite persistence."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_carr_madan_variance_swap(ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="spot_price"):
        calculate_carr_madan_variance_swap(spot_price=-100.0, ticker="SPY")

    with pytest.raises(ValueError, match="atm_implied_vol"):
        calculate_carr_madan_variance_swap(atm_implied_vol=-0.2, ticker="SPY")

    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0007)))
    receipt = evaluate_variance_swap_profile(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        spot_price=receipt.spot_price,
        time_to_maturity=receipt.time_to_maturity,
        fair_variance_strike=receipt.fair_variance_strike,
        fair_vol_strike=receipt.fair_vol_strike,
        vol_risk_premium=receipt.vol_risk_premium,
        put_call_variance_ratio=receipt.put_call_variance_ratio,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "var_swap_test.db"
    assert persist_variance_swap_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, n_strikes, regime, hmac_signature FROM carr_madan_variance_swap_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 200
    assert rows[0][3] == receipt.hmac_signature


def test_cli_variance_swap_json_and_aliases() -> None:
    """Verify CLI variance-swap and aliases emit valid JSON receipt."""
    for alias in ("variance-swap", "carr-madan", "cm-1999", "var-swap"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--spot",
                "100.0",
                "--vol",
                "0.20",
                "--skew",
                "-0.15",
                "--realized-vol",
                "0.16",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["fair_vol_strike"] > 0.0
        assert len(data["hmac_signature"]) == 64

