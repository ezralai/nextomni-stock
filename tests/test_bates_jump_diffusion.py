"""Tests for David S. Bates (1996) Stochastic Volatility Jump-Diffusion (SVJ) Option Pricing Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.bates_jump_diffusion import (
    _build_hmac_payload,
    calculate_bates_option_price,
    evaluate_bates_profile_from_prices,
    persist_bates_record,
    verify_hmac_signature,
)


def test_heston_convergence_when_jump_intensity_zero() -> None:
    """When jump intensity lambda = 0, Bates model converges exactly to pure Heston model."""
    rec = calculate_bates_option_price(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=0.50,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.30,
        rho=-0.50,
        lambda_jump=0.0,
        mu_jump=0.0,
        sigma_jump=0.10,
        ticker="SPY_HESTON",
    )
    # Call price converges within numerical quadrature error to pure Heston benchmark
    assert rec.bates_call_price == pytest.approx(rec.heston_benchmark_call, abs=1e-5)
    assert abs(rec.jump_contribution) < 1e-5
    assert rec.regime == "DIFFUSION_DOMINANT_HESTON_REGIME"
    assert rec.feller_satisfied is True

    # Put-call parity: C - P = S0 * exp(-q*T) - K * exp(-r*T)
    forward_discounted = 100.0 - 100.0 * np.exp(-0.04 * 0.50)
    assert (rec.bates_call_price - rec.bates_put_price) == pytest.approx(forward_discounted, abs=0.02)


def test_asymmetric_crash_jumps_and_feller_violation() -> None:
    """Crash jumps elevate jump contribution; violating Feller condition flags regime."""
    rec_jump = calculate_bates_option_price(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=0.50,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        v0=0.04,
        kappa=2.5,
        theta=0.04,
        sigma_v=0.25,
        rho=-0.50,
        lambda_jump=0.80,
        mu_jump=-0.08,
        sigma_jump=0.15,
        ticker="SPY_CRASH",
    )
    assert abs(rec_jump.jump_contribution) > 0.05
    assert rec_jump.regime == "ASYMMETRIC_CRASH_JUMP_DOMINANT"

    # Feller violation: 2 * kappa * theta <= sigma_v^2
    rec_feller = calculate_bates_option_price(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=0.50,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        v0=0.04,
        kappa=0.5,
        theta=0.04,
        sigma_v=0.60,  # 2 * 0.5 * 0.04 = 0.04 < 0.36
        rho=-0.50,
        lambda_jump=0.10,
        mu_jump=-0.02,
        sigma_jump=0.10,
        ticker="SPY_FELLER",
    )
    assert rec_feller.feller_satisfied is False
    assert rec_feller.feller_ratio < 1.0
    assert rec_feller.regime == "FELLER_VIOLATION_SUB_CRITICAL"


def test_bates_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate parameter boundaries, HMAC audit sealing, and SQLite logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_bates_option_price(ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="Spot and strike"):
        calculate_bates_option_price(spot_price=-100.0, ticker="SPY")

    with pytest.raises(ValueError, match="Initial variance"):
        calculate_bates_option_price(v0=-0.04, ticker="SPY")

    with pytest.raises(ValueError, match="Correlation rho"):
        calculate_bates_option_price(rho=-1.5, ticker="SPY")

    # Evaluate profile from simulated price sequence
    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0006)))
    receipt = evaluate_bates_profile_from_prices(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        spot_price=receipt.spot_price,
        strike_price=receipt.strike_price,
        time_to_maturity=receipt.time_to_maturity,
        bates_call_price=receipt.bates_call_price,
        bates_put_price=receipt.bates_put_price,
        heston_benchmark_call=receipt.heston_benchmark_call,
        jump_contribution=receipt.jump_contribution,
        feller_ratio=receipt.feller_ratio,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "bates_test.db"
    assert persist_bates_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, spot_price, bates_call_price, regime, hmac_signature FROM bates_jump_diffusion_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == pytest.approx(receipt.spot_price)
    assert rows[0][4] == receipt.hmac_signature


def test_cli_bates_opt_json_and_aliases() -> None:
    """Verify CLI bates-opt and aliases emit valid JSON receipt."""
    for alias in ("bates-opt", "bates", "bates-1996", "svj-model"):
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
                "--v0",
                "0.04",
                "--kappa",
                "2.0",
                "--theta",
                "0.04",
                "--sigma-v",
                "0.30",
                "--rho",
                "-0.50",
                "--lambda-jump",
                "0.50",
                "--mu-jump",
                "-0.05",
                "--sigma-jump",
                "0.15",
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
        assert data["bates_call_price"] > 0.0
        assert len(data["hmac_signature"]) == 64

