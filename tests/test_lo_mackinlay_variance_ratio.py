"""Tests for Lo & MacKinlay (1988) Heteroskedasticity-Robust Variance Ratio Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.lo_mackinlay_variance_ratio import (
    _build_hmac_payload,
    build_synthetic_vr_prices,
    calculate_lo_mackinlay_variance_ratio,
    evaluate_multi_horizon_variance_ratio,
    persist_variance_ratio_record,
    verify_hmac_signature,
)


def test_random_walk_martingale_neutrality_and_vr2_identity() -> None:
    """Pure i.i.d. martingale path yields VR(q) ~ 1.0 and fails to reject Random Walk."""
    prices = build_synthetic_vr_prices(rho=0.0, n_obs=320, seed=101)
    rec = calculate_lo_mackinlay_variance_ratio(prices, lag_q=5, ticker="SPY")
    assert 0.80 <= rec.variance_ratio <= 1.20
    assert abs(rec.z_stat_heteroskedastic) < 1.96
    assert rec.p_value_heteroskedastic > 0.05
    assert rec.regime == "RANDOM_WALK_MARTINGALE_CONSISTENT"

    # At q = 2, check phi_hom == 1 / N exact identity
    rec2 = calculate_lo_mackinlay_variance_ratio(prices, lag_q=2, ticker="SPY")
    assert rec2.phi_homoskedastic == pytest.approx(1.0 / rec2.n_returns, abs=1e-12)


def test_positive_autocorrelation_momentum_rejects_random_walk() -> None:
    """Strong positive AR(1) serial correlation yields VR(q) > 1 and positive z* >= 1.96."""
    prices = build_synthetic_vr_prices(rho=0.45, n_obs=260, seed=7)
    rec = calculate_lo_mackinlay_variance_ratio(prices, lag_q=5, ticker="QQQ")
    assert rec.variance_ratio > 1.35
    assert rec.excess_variance_ratio > 0.35
    assert rec.z_stat_heteroskedastic >= 1.96
    assert rec.p_value_heteroskedastic < 0.05
    assert rec.regime == "MOMENTUM_PERSISTENCE_REJECT_RW"


def test_negative_autocorrelation_mean_reversion_rejects_random_walk() -> None:
    """Strong negative AR(1) serial correlation yields VR(q) < 1 and negative z* <= -1.96."""
    prices = build_synthetic_vr_prices(rho=-0.45, n_obs=260, seed=11)
    rec = calculate_lo_mackinlay_variance_ratio(prices, lag_q=5, ticker="IWM")
    assert rec.variance_ratio < 0.70
    assert rec.excess_variance_ratio < -0.30
    assert rec.z_stat_heteroskedastic <= -1.96
    assert rec.p_value_heteroskedastic < 0.05
    assert rec.regime == "MEAN_REVERSION_REJECT_RW"


def test_heteroskedastic_clustering_widens_phi_star_and_dampens_z() -> None:
    """Clustered ARCH-like volatility inflates phi*(q) above homoskedastic phi(q)."""
    prices_het = build_synthetic_vr_prices(rho=0.0, n_obs=300, seed=42, heteroskedastic=True)
    rec = calculate_lo_mackinlay_variance_ratio(prices_het, lag_q=5, ticker="NVDA")
    assert rec.phi_heteroskedastic > rec.phi_homoskedastic
    assert abs(rec.z_stat_heteroskedastic) < abs(rec.z_stat_homoskedastic)


def test_multi_horizon_evaluation_filters_infeasible_long_lags() -> None:
    """Multi-horizon helper evaluates feasible lags cleanly."""
    prices = build_synthetic_vr_prices(rho=0.0, n_obs=55, seed=3)
    receipts = evaluate_multi_horizon_variance_ratio(prices, lags=(2, 5, 10, 21), ticker="AAPL")
    assert [r.lag_q for r in receipts] == [2, 5, 10, 21]

    prices_short = build_synthetic_vr_prices(rho=0.0, n_obs=24, seed=3)
    receipts_short = evaluate_multi_horizon_variance_ratio(prices_short, lags=(2, 5, 21), ticker="AAPL")
    assert [r.lag_q for r in receipts_short] == [2, 5]


def test_input_validation_hmac_tamper_and_sqlite_persistence(tmp_path) -> None:
    """Guard invalid inputs/flat prices, verify HMAC integrity, and persist to SQLite."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_lo_mackinlay_variance_ratio([100.0] * 30, lag_q=2, ticker="../BAD")
    with pytest.raises(ValueError, match="Degenerate zero-variance"):
        calculate_lo_mackinlay_variance_ratio(np.full(30, 100.0), lag_q=2, ticker="SPY")
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_lo_mackinlay_variance_ratio([100.0, -1.0] * 15, lag_q=2, ticker="SPY")

    prices = build_synthetic_vr_prices(rho=0.40, n_obs=150, seed=19)
    rec = calculate_lo_mackinlay_variance_ratio(prices, lag_q=5, ticker="MSFT")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_returns=rec.n_returns,
        lag_q=rec.lag_q,
        variance_ratio=rec.variance_ratio,
        z_stat_homoskedastic=rec.z_stat_homoskedastic,
        z_stat_heteroskedastic=rec.z_stat_heteroskedastic,
        p_value_heteroskedastic=rec.p_value_heteroskedastic,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("MSFT", "AAPL"), rec.hmac_signature) is False

    db_file = str(tmp_path / "vr_test.db")
    assert persist_variance_ratio_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, lag_q, regime FROM lo_mackinlay_vr_audits").fetchall()
    assert rows == [("MSFT", 5, rec.regime)]


def test_cli_variance_ratio_json_and_aliases() -> None:
    """Verify CLI variance-ratio and aliases emit valid JSON receipt."""
    for alias in ("variance-ratio", "lo-mackinlay", "lm-1988"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--lag",
                "5",
                "--rho",
                "0.45",
                "--n-obs",
                "260",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["lag_q"] == 5
        assert data["variance_ratio"] > 1.35
        assert data["regime"] == "MOMENTUM_PERSISTENCE_REJECT_RW"

