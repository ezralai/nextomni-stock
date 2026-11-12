"""Tests for Gibbons, Ross & Shanken (1989) Multivariate GRS Efficiency F-Test."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.gibbons_ross_shanken import (
    _build_hmac_payload,
    build_synthetic_grs_returns,
    calculate_grs_test,
    evaluate_grs_profile,
    persist_grs_record,
    verify_hmac_signature,
)


def test_grs_zero_alpha_accepts_mean_variance_efficiency() -> None:
    """Exact zero-alpha residuals relative to factors yield W = 0 and p = 1.0."""
    assets, factors = build_synthetic_grs_returns(
        n_obs=180,
        n_assets=4,
        n_factors=1,
        alpha_ann_level=0.0,
        seed=183,
    )
    rec = calculate_grs_test(assets, factors, ticker="SPY")
    assert rec.grs_f_stat == pytest.approx(0.0, abs=1e-9)
    assert rec.grs_p_value == pytest.approx(1.0, abs=1e-9)
    assert rec.sharpe_spanning_gap_sq == pytest.approx(0.0, abs=1e-10)
    assert rec.combined_sharpe_ann == pytest.approx(rec.factor_sharpe_ann, abs=1e-6)
    assert rec.regime == "MEAN_VARIANCE_EFFICIENT_FAIL_REJECT"


def test_grs_joint_alpha_rejects_factor_efficiency_and_sharpe_identity() -> None:
    """Large joint alpha triggers F-rejection and verifies exact theta_p^2 - theta_K^2 identity."""
    assets, factors = build_synthetic_grs_returns(
        n_obs=500,
        n_assets=4,
        n_factors=2,
        alpha_ann_level=0.25,
        idio_vol_ann=0.05,
        seed=42,
    )
    rec = calculate_grs_test(assets, factors, ticker="QQQ")
    assert rec.grs_f_stat > 5.0
    assert rec.grs_p_value < 0.01
    assert rec.df_num == 4
    assert rec.df_den == 500 - 4 - 2
    assert rec.theta_p_sq > rec.theta_k_sq
    assert (rec.theta_p_sq - rec.theta_k_sq) == pytest.approx(rec.sharpe_spanning_gap_sq, abs=1e-10)
    assert rec.combined_sharpe_ann > rec.factor_sharpe_ann
    assert rec.regime == "FACTOR_INEFFICIENCY_REJECT_GRS"


def test_grs_marginal_alert_regime_and_empirical_profile() -> None:
    """Moderate spanning gap yields marginal or clean finite output across 1D factor input."""
    t_len = 120
    t_idx = np.arange(t_len, dtype=float)
    f_1d = 0.0005 + 0.01 * np.sin(t_idx * 0.25)
    # Construct two assets with small non-zero residual orthogonal to factors
    e1 = 0.004 * np.cos(t_idx * 0.4)
    e2 = 0.004 * np.sin(t_idx * 0.7)
    r_mat = np.column_stack([0.8 * f_1d + e1, 1.1 * f_1d + e2])
    rec = calculate_grs_test(r_mat, f_1d, ticker="IWM")
    assert rec.n_factors == 1
    assert rec.n_assets == 2
    assert rec.grs_f_stat >= 0.0

    # Empirical profile with price series and short-series fallback
    prices = 100.0 * np.exp(0.0004 * t_idx + 0.01 * np.sin(t_idx * 0.3))
    emp = evaluate_grs_profile(prices, ticker="AAPL")
    assert emp.ticker == "AAPL"
    assert emp.n_assets == 3

    emp_short = evaluate_grs_profile([100.0, 101.0], ticker="MSFT")
    assert emp_short.regime == "MEAN_VARIANCE_EFFICIENT_FAIL_REJECT"


def test_validation_guards_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Guard invalid shapes/tickers/collinear factors, verify HMAC, and test SQLite."""
    assets, factors = build_synthetic_grs_returns(n_obs=80, n_assets=3, n_factors=1, alpha_ann_level=0.08)

    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_grs_test(assets, factors, ticker="../BAD")
    with pytest.raises(ValueError, match="Insufficient observations"):
        calculate_grs_test(assets[:4], factors[:4], ticker="SPY")
    with pytest.raises(ValueError, match="rank-deficient"):
        collinear = np.column_stack([factors[:, 0], 2.0 * factors[:, 0]])
        calculate_grs_test(assets, collinear, ticker="SPY")

    rec = calculate_grs_test(assets, factors, ticker="NVDA")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_obs=rec.n_obs,
        n_assets=rec.n_assets,
        n_factors=rec.n_factors,
        grs_f_stat=rec.grs_f_stat,
        grs_p_value=rec.grs_p_value,
        sharpe_spanning_gap_sq=rec.sharpe_spanning_gap_sq,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("NVDA", "META"), rec.hmac_signature) is False

    db_file = str(tmp_path / "grs_test.db")
    assert persist_grs_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, n_assets, regime FROM grs_multivariate_audits").fetchall()
    assert rows == [("NVDA", 3, rec.regime)]


def test_cli_grs_test_json_and_aliases() -> None:
    """Verify CLI grs-test and aliases emit valid JSON receipt."""
    for alias in ("grs-test", "gibbons-ross-shanken", "grs-1989"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--n-obs",
                "500",
                "--n-assets",
                "4",
                "--n-factors",
                "1",
                "--alpha-ann",
                "0.25",
                "--idio-vol",
                "0.05",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["n_assets"] == 4
        assert data["grs_f_stat"] > 3.0
        assert data["regime"] == "FACTOR_INEFFICIENCY_REJECT_GRS"
