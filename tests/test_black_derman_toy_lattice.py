"""Tests for Black, Derman & Toy (1990) Lognormal Short-Rate Binomial Lattice."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.black_derman_toy_lattice import (
    _build_hmac_payload,
    build_synthetic_yield_curve,
    calibrate_bdt_lattice,
    evaluate_bdt_lattice_profile,
    persist_bdt_lattice_record,
    verify_hmac_signature,
)


def test_bdt_arbitrage_free_zero_bond_calibration_and_positivity() -> None:
    """Lattice matches market zero-coupon discount factors and keeps all short rates > 0."""
    curve = build_synthetic_yield_curve(n_steps=8, base_rate=0.04, slope_bps=120.0)
    receipt = calibrate_bdt_lattice(
        spot_rates=curve,
        volatilities=0.18,
        dt=1.0,
        coupon_rate=0.05,
        call_price=100.0,
        ticker="UST10Y",
    )
    assert receipt.n_steps == 8
    assert receipt.max_calibration_error < 1e-10
    assert 0.0 < receipt.terminal_rate_low < receipt.terminal_rate_median < receipt.terminal_rate_high
    assert receipt.short_rate_initial == pytest.approx(curve[0], abs=1e-9)
    assert receipt.regime == "NORMAL_STEEP"


def test_bdt_callable_bond_monotonicity_and_inverted_regime() -> None:
    """Embedded call option is non-negative and inverted curve maps to INVERTED_TIGHTENING."""
    curve = build_synthetic_yield_curve(n_steps=6, base_rate=0.055, slope_bps=-100.0)
    receipt = calibrate_bdt_lattice(
        spot_rates=curve,
        volatilities=0.20,
        dt=0.5,
        coupon_rate=0.07,
        call_price=100.0,
        ticker="UST2S10S",
    )
    assert receipt.max_calibration_error < 1e-10
    assert receipt.straight_bond_price >= receipt.callable_bond_price
    assert receipt.embedded_call_value == pytest.approx(
        receipt.straight_bond_price - receipt.callable_bond_price, abs=1e-9
    )
    assert receipt.embedded_call_value > 0.0
    assert receipt.regime == "INVERTED_TIGHTENING"


def test_bdt_validation_guards_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate error guards, HMAC-SHA256 audit verification, and SQLite persistence."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calibrate_bdt_lattice([0.04, 0.045], ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_bdt_lattice([-0.01, 0.03], ticker="SPY")

    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.001)))
    receipt = evaluate_bdt_lattice_profile(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        n_steps=receipt.n_steps,
        dt=receipt.dt,
        short_rate_initial=receipt.short_rate_initial,
        terminal_rate_median=receipt.terminal_rate_median,
        max_calibration_error=receipt.max_calibration_error,
        straight_bond_price=receipt.straight_bond_price,
        callable_bond_price=receipt.callable_bond_price,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "bdt_test.db"
    assert persist_bdt_lattice_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, n_steps, regime, hmac_signature FROM bdt_lattice_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 8
    assert rows[0][3] == receipt.hmac_signature


def test_cli_bdt_lattice_json_and_aliases() -> None:
    """Verify CLI bdt-lattice and aliases emit valid JSON receipt."""
    for alias in ("bdt-lattice", "black-derman-toy", "bdt-1990", "rate-lattice"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "UST",
                "--steps",
                "6",
                "--base-rate",
                "0.04",
                "--slope-bps",
                "100",
                "--vol",
                "0.16",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "UST"
        assert data["n_steps"] == 6
        assert data["max_calibration_error"] < 1e-10
        assert data["regime"] == "NORMAL_STEEP"
        assert len(data["hmac_signature"]) == 64

