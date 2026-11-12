"""Tests for Hurst (1951) & Peters (1994) Rescaled Range (R/S) Fractal Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.hurst_fractal_persistence import (
    _build_hmac_payload,
    build_synthetic_fractional_prices,
    calculate_hurst_exponent,
    persist_hurst_record,
    verify_hmac_signature,
)


def test_brownian_random_walk_hurst_and_fractal_dimension() -> None:
    """Pure Brownian fBm with H=0.50 yields corrected H ~ 0.50 and D ~ 1.50."""
    prices = build_synthetic_fractional_prices(hurst_target=0.50, n_obs=600, seed=7)
    rec = calculate_hurst_exponent(prices, min_window=8, ticker="SPY")
    assert 0.44 <= rec.hurst_corrected <= 0.56
    assert 1.44 <= rec.fractal_dimension <= 1.56
    assert rec.scaling_r2 > 0.98
    assert rec.regime == "BROWNIAN_RANDOM_WALK"


def test_persistent_fractal_trend_and_anti_persistent_mean_reversion() -> None:
    """Verify clear separation across H=0.78 (persistent) vs H=0.22 (anti-persistent)."""
    prices_trend = build_synthetic_fractional_prices(hurst_target=0.78, n_obs=600, seed=12)
    rec_trend = calculate_hurst_exponent(prices_trend, min_window=8, ticker="QQQ")
    assert rec_trend.hurst_corrected > 0.60
    assert rec_trend.fractal_dimension < 1.40
    assert rec_trend.implied_correlation > 0.14
    assert rec_trend.regime == "PERSISTENT_FRACTAL_TREND"

    prices_mr = build_synthetic_fractional_prices(hurst_target=0.22, n_obs=600, seed=12)
    rec_mr = calculate_hurst_exponent(prices_mr, min_window=8, ticker="IWM")
    assert rec_mr.hurst_corrected < 0.38
    assert rec_mr.fractal_dimension > 1.62
    assert rec_mr.implied_correlation < -0.15
    assert rec_mr.regime == "ANTI_PERSISTENT_MEAN_REVERSION"


def test_validation_guards_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Guard invalid tickers/flat/short/negative prices, verify HMAC, and test SQLite."""
    prices = build_synthetic_fractional_prices(hurst_target=0.70, n_obs=260, seed=9)
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_hurst_exponent(prices, ticker="../BAD")
    with pytest.raises(ValueError, match="Degenerate zero-variance"):
        calculate_hurst_exponent(np.full(80, 100.0), ticker="SPY")
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_hurst_exponent([100.0, -2.0] * 30, ticker="SPY")
    with pytest.raises(ValueError, match="Need at least"):
        calculate_hurst_exponent(prices[:15], min_window=8, ticker="SPY")

    rec = calculate_hurst_exponent(prices, ticker="AAPL")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_returns=rec.n_returns,
        hurst_classical=rec.hurst_classical,
        hurst_corrected=rec.hurst_corrected,
        fractal_dimension=rec.fractal_dimension,
        scaling_r2=rec.scaling_r2,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("AAPL", "MSFT"), rec.hmac_signature) is False

    db_file = str(tmp_path / "hurst_test.db")
    assert persist_hurst_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, n_returns, regime FROM hurst_fractal_audits").fetchall()
    assert rows == [("AAPL", rec.n_returns, rec.regime)]


def test_cli_hurst_exponent_json_and_aliases() -> None:
    """Verify CLI hurst-exponent and aliases emit valid JSON receipt."""
    for alias in ("hurst-exponent", "fractal-market", "rs-1951"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--hurst-target",
                "0.78",
                "--n-obs",
                "600",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["hurst_corrected"] > 0.60
        assert data["regime"] == "PERSISTENT_FRACTAL_TREND"
