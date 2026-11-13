"""Tests for Steven G. Kou (2002) Double Exponential Jump-Diffusion Option Pricing Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.kou_double_exponential_jump import (
    _build_hmac_payload,
    calculate_kou_option_price,
    evaluate_kou_jump_profile,
    persist_kou_record,
    verify_hmac_signature,
)


def test_black_scholes_convergence_when_jump_intensity_zero() -> None:
    """When jump intensity lambda = 0, Kou jump-diffusion reduces to standard Black-Scholes."""
    rec = calculate_kou_option_price(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=0.50,
        risk_free_rate=0.05,
        diffusion_vol=0.20,
        jump_intensity=0.0,
        prob_up_jump=0.40,
        eta1_up=10.0,
        eta2_down=5.0,
        ticker="SPY_BS",
    )
    # Call price converges within numerical quadrature error to BS benchmark call
    assert rec.kou_call_price == pytest.approx(rec.bs_benchmark_call, rel=0.005)
    assert abs(rec.jump_premium) < 0.05
    assert rec.regime == "DIFFUSION_DOMINANT"
    # Put-call parity holds: C - P = S - K * exp(-r * T)
    forward_discounted = 100.0 - 100.0 * np.exp(-0.05 * 0.50)
    assert (rec.kou_call_price - rec.kou_put_price) == pytest.approx(forward_discounted, abs=0.05)


def test_asymmetric_downward_jumps_induce_crash_premium_and_skew() -> None:
    """High jump intensity with downward jump bias (1-p > p, eta2 < eta1) elevates put prices."""
    rec = calculate_kou_option_price(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=0.25,
        risk_free_rate=0.03,
        diffusion_vol=0.15,
        jump_intensity=1.5,
        prob_up_jump=0.25,  # 75% probability of negative jumps
        eta1_up=15.0,        # upward jumps are small (mean +6.7%)
        eta2_down=4.0,       # downward jumps are severe (mean -25%)
        ticker="SPY_CRASH",
    )
    # Jumps add option premium relative to pure diffusion
    assert rec.kou_call_price > rec.bs_benchmark_call
    assert rec.jump_premium > 0.0
    assert rec.atm_skew < 0.0
    assert rec.regime == "HIGH_ASYMMETRIC_CRASH_RISK"


def test_kou_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate parameter boundaries, HMAC audit sealing, and SQLite persistence."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_kou_option_price(ticker="INVALID TICKER$$")

    with pytest.raises(ValueError, match="eta1_up"):
        calculate_kou_option_price(eta1_up=0.8, ticker="SPY")

    with pytest.raises(ValueError, match="diffusion_vol"):
        calculate_kou_option_price(diffusion_vol=-0.1, ticker="SPY")

    with pytest.raises(ValueError, match="prob_up_jump"):
        calculate_kou_option_price(prob_up_jump=1.5, ticker="SPY")

    # Evaluate profile from simulated price series
    prices = 100.0 * np.exp(np.cumsum(np.full(100, 0.0005)))
    receipt = evaluate_kou_jump_profile(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        spot_price=receipt.spot_price,
        strike_price=receipt.strike_price,
        time_to_maturity=receipt.time_to_maturity,
        kou_call_price=receipt.kou_call_price,
        kou_put_price=receipt.kou_put_price,
        bs_benchmark_call=receipt.bs_benchmark_call,
        jump_premium=receipt.jump_premium,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "kou_test.db"
    assert persist_kou_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, jump_intensity, regime, hmac_signature FROM kou_jump_diffusion_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == pytest.approx(receipt.jump_intensity)
    assert rows[0][3] == receipt.hmac_signature


def test_cli_kou_jump_json_and_aliases() -> None:
    """Verify CLI kou-jump and aliases emit valid JSON receipt."""
    for alias in ("kou-jump", "kou-2002", "double-exponential", "kou-model"):
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
                "--strike",
                "100.0",
                "--vol",
                "0.18",
                "--lambda-jump",
                "0.8",
                "--p-up",
                "0.35",
                "--eta1",
                "10.0",
                "--eta2",
                "5.0",
                "--ttm",
                "0.50",
                "--rf",
                "0.04",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["kou_call_price"] > 0.0
        assert data["kou_put_price"] > 0.0
        assert len(data["hmac_signature"]) == 64
