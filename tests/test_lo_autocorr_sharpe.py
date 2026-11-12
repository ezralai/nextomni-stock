"""Tests for Andrew Lo (2002) Autocorrelation-Adjusted Sharpe Ratio Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.lo_autocorr_sharpe import (
    _build_hmac_payload,
    build_synthetic_ar1_returns,
    calculate_lo_autocorr_sharpe,
    evaluate_lo_sharpe_profile,
    persist_lo_sharpe_record,
    verify_hmac_signature,
)


def test_lo_positive_and_negative_autocorrelation_direction() -> None:
    """Positive AR(1) deflates Sharpe vs naive sqrt(q); negative AR(1) boosts Sharpe."""
    pos_r = build_synthetic_ar1_returns(n_obs=400, rho1=0.35, mean_ret=0.004, seed=42)
    rec_pos = calculate_lo_autocorr_sharpe(pos_r, q_horizon=12, ticker="HEDGE")
    assert rec_pos.lag1_autocorr > 0.15
    assert rec_pos.sharpe_lo_q < rec_pos.sharpe_naive_q
    assert rec_pos.inflation_pct > 10.0
    assert rec_pos.regime == "SMOOTHED_SERIAL_INFLATION"

    neg_r = build_synthetic_ar1_returns(n_obs=400, rho1=-0.35, mean_ret=0.004, seed=42)
    rec_neg = calculate_lo_autocorr_sharpe(neg_r, q_horizon=12, ticker="STATARB")
    assert rec_neg.lag1_autocorr < -0.15
    assert rec_neg.sharpe_lo_q > rec_neg.sharpe_naive_q
    assert rec_neg.inflation_pct < -10.0
    assert rec_neg.regime == "MEAN_REVERTING_UNDERSTATED"


def test_lo_iid_consistency_and_ci_bounds() -> None:
    """Near-zero autocorrelation gives eta ~= sqrt(q) and valid 95% CI."""
    iid_r = build_synthetic_ar1_returns(n_obs=600, rho1=0.0, mean_ret=0.0012, seed=7)
    rec = calculate_lo_autocorr_sharpe(iid_r, q_horizon=12, ticker="SPY")
    assert rec.ci95_low < rec.sharpe_lo_q < rec.ci95_high
    assert abs(rec.eta_scaling_factor - np.sqrt(12.0)) < 0.6


def test_lo_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite audit persistence."""
    r = build_synthetic_ar1_returns(n_obs=100, rho1=0.2)
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_lo_autocorr_sharpe(r, ticker="BAD TICKER")

    with pytest.raises(ValueError, match="q_horizon"):
        calculate_lo_autocorr_sharpe(r, q_horizon=1, ticker="SPY")

    prices = 100.0 * np.exp(np.cumsum(r))
    receipt = evaluate_lo_sharpe_profile(prices, ticker="SPY", q_horizon=21)
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        n_obs=receipt.n_obs,
        q_horizon=receipt.q_horizon,
        lag1_autocorr=receipt.lag1_autocorr,
        sharpe_naive_q=receipt.sharpe_naive_q,
        sharpe_lo_q=receipt.sharpe_lo_q,
        se_lo_q=receipt.se_lo_q,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "lo_sharpe_test.db"
    assert persist_lo_sharpe_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, q_horizon, regime, hmac_signature FROM lo_autocorr_sharpe_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 21
    assert rows[0][3] == receipt.hmac_signature


def test_cli_lo_sharpe_json_and_aliases() -> None:
    """Verify CLI lo-sharpe and aliases emit valid JSON receipt."""
    for alias in ("lo-sharpe", "lo-2002", "autocorr-sharpe", "lo-adjusted-sr"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--q",
                "12",
                "--rho1",
                "0.35",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["q_horizon"] == 12
        assert data["sharpe_lo_q"] < data["sharpe_naive_q"]
        assert data["regime"] == "SMOOTHED_SERIAL_INFLATION"
        assert len(data["hmac_signature"]) == 64
