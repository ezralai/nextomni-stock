"""Tests for Patrick S. Hagan et al. (2002) SABR Stochastic Volatility Smile Calibration Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.sabr_stochastic_volatility import (
    _build_hmac_payload,
    calculate_sabr_profile,
    evaluate_sabr_from_history,
    hagan_sabr_atm_volatility,
    hagan_sabr_implied_volatility,
    persist_sabr_record,
    verify_hmac_signature,
)


def test_atm_limit_convergence_and_near_atm_smoothness() -> None:
    """Implied volatility at K = F converges exactly to analytical ATM limit."""
    f = 100.0
    t = 0.50
    alpha = 0.20
    beta = 0.50
    rho = -0.35
    nu = 0.40

    atm_analytical = hagan_sabr_atm_volatility(f, t, alpha, beta, rho, nu)
    atm_implied = hagan_sabr_implied_volatility(f, f, t, alpha, beta, rho, nu)

    # Exact ATM equality
    assert atm_implied == pytest.approx(atm_analytical, abs=1e-10)

    # Near ATM smoothness: K = F +/- 1e-4 matches within 1e-4
    near_below = hagan_sabr_implied_volatility(f, f - 0.0001, t, alpha, beta, rho, nu)
    near_above = hagan_sabr_implied_volatility(f, f + 0.0001, t, alpha, beta, rho, nu)
    assert near_below == pytest.approx(atm_analytical, abs=1e-4)
    assert near_above == pytest.approx(atm_analytical, abs=1e-4)


def test_sabr_skew_directionality_and_smile_convexity() -> None:
    """Negative rho produces equity-style downward skew; positive nu generates smile convexity."""
    f = 100.0
    t = 0.50

    # Negative skew (crash fear)
    rec_skew = calculate_sabr_profile(
        forward_price=f,
        time_to_maturity=t,
        alpha=0.25,
        beta=0.70,
        rho=-0.50,
        nu=0.45,
        ticker="SPY_SKEW",
    )
    assert rec_skew.otm_put_vol > rec_skew.atm_vol > rec_skew.otm_call_vol
    assert rec_skew.skew_slope < 0.0
    assert rec_skew.regime == "STEEP_NEGATIVE_SKEW_CRASH_FEAR"

    # Flat Black-Scholes limit: beta=1, rho=0, nu=0 => constant IV = alpha
    rec_flat = calculate_sabr_profile(
        forward_price=f,
        time_to_maturity=t,
        alpha=0.20,
        beta=1.0,
        rho=0.0,
        nu=0.0,
        ticker="SPY_FLAT",
    )
    assert rec_flat.atm_vol == pytest.approx(0.20, rel=1e-4)
    assert rec_flat.otm_put_vol == pytest.approx(0.20, rel=1e-4)
    assert rec_flat.otm_call_vol == pytest.approx(0.20, rel=1e-4)
    assert abs(rec_flat.skew_slope) < 1e-4
    assert rec_flat.regime == "FLAT_SABR_SURFACE"


def test_sabr_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate parameter boundaries, HMAC audit sealing, and SQLite logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_sabr_profile(ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="Alpha"):
        calculate_sabr_profile(alpha=-0.2, ticker="SPY")

    with pytest.raises(ValueError, match="Rho"):
        calculate_sabr_profile(rho=-1.5, ticker="SPY")

    with pytest.raises(ValueError, match="Beta"):
        calculate_sabr_profile(beta=1.5, ticker="SPY")

    # Evaluate profile from simulated price sequence
    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0006)))
    receipt = evaluate_sabr_from_history(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        forward_price=receipt.forward_price,
        time_to_maturity=receipt.time_to_maturity,
        alpha=receipt.alpha,
        beta=receipt.beta,
        rho=receipt.rho,
        nu=receipt.nu,
        atm_vol=receipt.atm_vol,
        skew_slope=receipt.skew_slope,
        smile_convexity=receipt.smile_convexity,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "sabr_test.db"
    assert persist_sabr_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, forward_price, regime, hmac_signature FROM sabr_volatility_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == pytest.approx(receipt.forward_price)
    assert rows[0][3] == receipt.hmac_signature


def test_cli_sabr_vol_json_and_aliases() -> None:
    """Verify CLI sabr-vol and aliases emit valid JSON receipt."""
    for alias in ("sabr-vol", "sabr", "sabr-2002", "hagan-sabr"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--forward",
                "100.0",
                "--alpha",
                "0.22",
                "--beta",
                "0.60",
                "--rho",
                "-0.35",
                "--nu",
                "0.40",
                "--ttm",
                "0.50",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["atm_vol"] > 0.0
        assert len(data["hmac_signature"]) == 64

