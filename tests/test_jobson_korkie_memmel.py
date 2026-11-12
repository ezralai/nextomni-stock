"""Tests for Jobson & Korkie (1981) & Memmel (2003) Paired Sharpe Ratio Equality Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.jobson_korkie_memmel import (
    _build_hmac_payload,
    build_synthetic_paired_returns,
    calculate_jobson_korkie_memmel,
    evaluate_sharpe_comparison_profile,
    persist_sharpe_comparison_record,
    verify_hmac_signature,
)


def test_equal_sharpes_fail_to_reject_null() -> None:
    """Equal Sharpe ratios yield z_stat = 0, p_value = 1, and INDISTINGUISHABLE regime."""
    ra, rb = build_synthetic_paired_returns(
        n_obs=252,
        sharpe_a_ann=0.90,
        sharpe_b_ann=0.90,
        correlation=0.80,
        seed=184,
    )
    rec = calculate_jobson_korkie_memmel(ra, rb, ticker_a="ALPHA", ticker_b="SPY")
    assert rec.sharpe_a_ann == pytest.approx(0.90, abs=1e-6)
    assert rec.sharpe_b_ann == pytest.approx(0.90, abs=1e-6)
    assert rec.z_stat == pytest.approx(0.0, abs=1e-9)
    assert rec.p_value == pytest.approx(1.0, abs=1e-9)
    assert rec.regime == "INDISTINGUISHABLE_SHARPE_FAIL_REJECT"


def test_correlation_power_amplification_outperformance_and_underperformance() -> None:
    """High correlation shrinks Memmel SE below naive SE, amplifying z-stat power."""
    ra_hi, rb_hi = build_synthetic_paired_returns(
        n_obs=756,
        sharpe_a_ann=1.45,
        sharpe_b_ann=0.70,
        correlation=0.90,
        seed=7,
    )
    rec_hi = calculate_jobson_korkie_memmel(ra_hi, rb_hi, ticker_a="STRAT", ticker_b="SPY")
    assert rec_hi.memmel_se_daily < rec_hi.naive_uncorr_se_daily * 0.5
    assert rec_hi.z_stat > 2.5
    assert rec_hi.p_value < 0.05
    assert rec_hi.ci_95_low_ann > 0.0
    assert rec_hi.regime == "STATISTICAL_OUTPERFORMANCE_A_BEATS_B"

    # Flip comparison -> significant underperformance
    rec_rev = calculate_jobson_korkie_memmel(rb_hi, ra_hi, ticker_a="SPY", ticker_b="STRAT")
    assert rec_rev.z_stat == pytest.approx(-rec_hi.z_stat, abs=1e-9)
    assert rec_rev.regime == "STATISTICAL_UNDERPERFORMANCE_B_BEATS_A"


def test_identical_series_and_empirical_profile() -> None:
    """Identical return streams handle zero Memmel variance cleanly without ZeroDivisionError."""
    _, rb = build_synthetic_paired_returns(n_obs=120, sharpe_b_ann=0.80)
    rec_same = calculate_jobson_korkie_memmel(rb, rb, ticker_a="SPY", ticker_b="SPY")
    assert rec_same.correlation == pytest.approx(1.0, abs=1e-9)
    assert rec_same.memmel_se_daily == pytest.approx(0.0, abs=1e-12)
    assert rec_same.z_stat == pytest.approx(0.0, abs=1e-12)
    assert rec_same.p_value == pytest.approx(1.0, abs=1e-12)
    assert rec_same.regime == "INDISTINGUISHABLE_SHARPE_FAIL_REJECT"

    t = np.arange(80, dtype=float)
    prices = 100.0 * np.exp(0.0006 * t + 0.01 * np.sin(t * 0.25))
    emp = evaluate_sharpe_comparison_profile(prices, ticker="AAPL")
    assert emp.ticker_b == "AAPL"
    assert emp.n_obs == 79

    emp_short = evaluate_sharpe_comparison_profile([100.0, 101.0], ticker="MSFT")
    assert emp_short.ticker_b == "MSFT"


def test_validation_guards_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Reject bad tickers/length mismatch/flat returns, verify HMAC, and persist to SQLite."""
    ra, rb = build_synthetic_paired_returns(n_obs=120, sharpe_a_ann=1.30, sharpe_b_ann=0.70, correlation=0.85)
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_jobson_korkie_memmel(ra, rb, ticker_a="../BAD", ticker_b="SPY")
    with pytest.raises(ValueError, match="identical length"):
        calculate_jobson_korkie_memmel(ra[:-1], rb)
    with pytest.raises(ValueError, match="Degenerate zero-volatility"):
        calculate_jobson_korkie_memmel(np.full(30, 0.001), rb[:30])

    rec = calculate_jobson_korkie_memmel(ra, rb, ticker_a="QQQ", ticker_b="SPY")
    payload = _build_hmac_payload(
        ticker_a=rec.ticker_a,
        ticker_b=rec.ticker_b,
        n_obs=rec.n_obs,
        sharpe_a_ann=rec.sharpe_a_ann,
        sharpe_b_ann=rec.sharpe_b_ann,
        correlation=rec.correlation,
        z_stat=rec.z_stat,
        p_value=rec.p_value,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("QQQ", "IWM"), rec.hmac_signature) is False

    db_file = str(tmp_path / "jkm_test.db")
    assert persist_sharpe_comparison_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker_a, ticker_b, regime FROM jkm_sharpe_equality_audits").fetchall()
    assert rows == [("QQQ", "SPY", rec.regime)]


def test_cli_sharpe_test_json_and_aliases() -> None:
    """Verify CLI sharpe-test and aliases emit valid JSON receipt."""
    for alias in ("sharpe-test", "jobson-korkie", "memmel", "jkm-2003"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker-a",
                "STRAT",
                "--ticker-b",
                "SPY",
                "--sr-a",
                "1.50",
                "--sr-b",
                "0.70",
                "--rho",
                "0.90",
                "--n-obs",
                "756",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker_a"] == "STRAT"
        assert data["ticker_b"] == "SPY"
        assert data["z_stat"] > 2.5
        assert data["regime"] == "STATISTICAL_OUTPERFORMANCE_A_BEATS_B"
