"""Tests for David Easley, Nicholas M. Kiefer, Maureen O'Hara & Joseph B. Paperman (1996) EKOP PIN Engine."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.ekop_pin import (
    _build_hmac_payload,
    _ekop_daily_log_likelihood,
    calculate_ekop_pin,
    calibrate_ekop_pin,
    evaluate_ekop_pin_from_trades,
    persist_ekop_record,
    verify_hmac_signature,
)


def test_analytical_pin_formula_and_zero_information_limit() -> None:
    """Exact PIN equation matches theoretical formulation and vanishes when alpha = 0."""
    # Base test case: alpha=0.30, mu=100.0, epsilon=50.0 => PIN = 30 / (30 + 100) = 30/130 ~ 0.23077
    rec = calculate_ekop_pin(
        alpha=0.30,
        delta=0.50,
        mu=100.0,
        epsilon=50.0,
        sample_size=50,
        ticker="SPY_PIN",
    )
    expected_pin = (0.30 * 100.0) / (0.30 * 100.0 + 2.0 * 50.0)
    assert rec.pin == pytest.approx(expected_pin, rel=1e-6)
    assert rec.informed_order_share + rec.uninformed_order_share == pytest.approx(1.0, abs=1e-8)
    assert rec.daily_order_turnover == pytest.approx(130.0, rel=1e-6)
    assert rec.regime == "BALANCED_INFORMATION_EQUILIBRIUM"

    # Zero information limit: alpha = 0.0 => PIN = 0.0
    rec_zero = calculate_ekop_pin(
        alpha=0.0,
        delta=0.50,
        mu=100.0,
        epsilon=50.0,
        sample_size=50,
        ticker="SPY_ZERO",
    )
    assert rec_zero.pin == 0.0
    assert rec_zero.regime == "NOISE_TRADER_DOMINATED_LIQUIDITY"


def test_log_sum_exp_stability_under_large_trade_volumes() -> None:
    """Log-sum-exp factorization evaluates extreme high-volume order counts without factorial overflow."""
    # 10,000 buys and 10,000 sells would cause 10000! overflow in naive implementations
    buys_large = np.array([10000.0, 12000.0, 8500.0])
    sells_large = np.array([9800.0, 11500.0, 8900.0])

    ll = _ekop_daily_log_likelihood(
        buys_large,
        sells_large,
        alpha=0.35,
        delta=0.45,
        mu=2500.0,
        epsilon=9000.0,
    )
    assert math.isfinite(ll)
    assert ll < 0.0


def test_mle_calibration_recovery() -> None:
    """Maximum likelihood optimization recovers structural parameters from synthetic Poisson trade data."""
    np.random.seed(42)
    n = 60
    # Simulate days: 30% event days, 50% bad news
    # Uninformed: eps = 60 buys, 60 sells
    # Informed: mu = 80
    buys = []
    sells = []
    for _ in range(n):
        event = np.random.rand() < 0.35
        bad = np.random.rand() < 0.50 if event else False
        b = np.random.poisson(60.0 + (80.0 if (event and not bad) else 0.0))
        s = np.random.poisson(60.0 + (80.0 if (event and bad) else 0.0))
        buys.append(b)
        sells.append(s)

    rec = calibrate_ekop_pin(buys, sells, ticker="SPY_MLE")
    assert 0.05 <= rec.pin <= 0.45
    assert rec.sample_size == n
    assert math.isfinite(rec.log_likelihood)
    assert math.isfinite(rec.aic)


def test_ekop_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate parameter boundaries, HMAC audit sealing, and SQLite logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_ekop_pin(ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="Alpha"):
        calculate_ekop_pin(alpha=1.5, ticker="SPY")

    with pytest.raises(ValueError, match="Epsilon"):
        calculate_ekop_pin(epsilon=-10.0, ticker="SPY")

    # Evaluate profile from simulated price sequence
    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0006)))
    receipt = evaluate_ekop_pin_from_trades(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        sample_size=receipt.sample_size,
        alpha=receipt.alpha,
        delta=receipt.delta,
        mu=receipt.mu,
        epsilon=receipt.epsilon,
        pin=receipt.pin,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "ekop_test.db"
    assert persist_ekop_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, sample_size, pin, regime, hmac_signature FROM ekop_pin_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == receipt.sample_size
    assert rows[0][4] == receipt.hmac_signature


def test_cli_ekop_pin_json_and_aliases() -> None:
    """Verify CLI ekop-pin and aliases emit valid JSON receipt."""
    for alias in ("ekop-pin", "ekop", "pin-model", "informed-trading"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--alpha",
                "0.30",
                "--delta",
                "0.40",
                "--mu",
                "100.0",
                "--epsilon",
                "60.0",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["pin"] > 0.0
        assert len(data["hmac_signature"]) == 64

