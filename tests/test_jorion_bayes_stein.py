"""Tests for Philippe Jorion (1986) Bayes-Stein Shrinkage & Predictive Covariance Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.jorion_bayes_stein import (
    _build_hmac_payload,
    build_synthetic_asset_moments,
    calculate_jorion_bayes_stein,
    evaluate_bayes_stein_profile,
    persist_jorion_bayes_stein_record,
    verify_hmac_signature,
)


def test_jorion_shrinkage_formula_and_gmv_target() -> None:
    """Verify w_hat = (N+2)/((N+2) + T*d^2) and exact GMV weights sum to 1."""
    means, cov = build_synthetic_asset_moments(n_assets=4, mean_dispersion=0.06, base_vol=0.20)
    rec = calculate_jorion_bayes_stein(means, cov, sample_size_t=60, ticker="PORT")

    expected_w = 6.0 / (6.0 + 60.0 * rec.mahalanobis_d2)
    assert sum(rec.gmv_weights) == pytest.approx(1.0, rel=1e-12)
    assert rec.shrinkage_intensity_w == pytest.approx(expected_w, rel=1e-12)
    assert rec.mean_dispersion_bs == pytest.approx(
        (1.0 - rec.shrinkage_intensity_w) * rec.mean_dispersion_sample, rel=1e-12
    )


def test_jorion_zero_dispersion_collapses_to_full_gmv_shrinkage() -> None:
    """When sample means equal GMV mean (d^2 == 0), w_hat == 1.0 cleanly."""
    means = np.array([0.08, 0.08, 0.08])
    cov = np.diag([0.04, 0.05, 0.06])
    rec = calculate_jorion_bayes_stein(means, cov, sample_size_t=50)

    assert rec.mahalanobis_d2 == pytest.approx(0.0, abs=1e-12)
    assert rec.shrinkage_intensity_w == pytest.approx(1.0, rel=1e-12)
    assert rec.regime == "HEAVY_PRIOR_GMV_DOMINANCE"
    assert rec.cov_inflation_ratio > 1.0


def test_jorion_asymptotic_large_t_vanishing_shrinkage() -> None:
    """As sample window T -> infinity, shrinkage weight w_hat -> 0 and sample trust rises."""
    means, cov = build_synthetic_asset_moments(n_assets=4, mean_dispersion=0.10, base_vol=0.18)
    rec_short = calculate_jorion_bayes_stein(means, cov, sample_size_t=30)
    rec_long = calculate_jorion_bayes_stein(means, cov, sample_size_t=5000)

    assert rec_long.shrinkage_intensity_w < rec_short.shrinkage_intensity_w
    assert rec_long.regime == "HIGH_SIGNAL_SAMPLE_TRUST"
    assert rec_long.cov_inflation_ratio < rec_short.cov_inflation_ratio


def test_jorion_tangency_leverage_stabilization() -> None:
    """Bayes-Stein shrinkage reduces extreme Markowitz tangency gross leverage."""
    means, cov = build_synthetic_asset_moments(
        n_assets=5, mean_dispersion=0.12, base_vol=0.20, correlation=0.50
    )
    rec = calculate_jorion_bayes_stein(means, cov, sample_size_t=48, risk_free_rate_r=0.04)

    assert rec.bs_gross_leverage < rec.mle_gross_leverage
    assert rec.leverage_reduction_pct > 0.0
    assert sum(rec.bs_tangency_weights) == pytest.approx(1.0, rel=1e-10)


def test_jorion_short_window_estimation_risk_regime() -> None:
    """Flag SHORT_WINDOW_ESTIMATION_RISK when T is dangerously close to N+3."""
    means, cov = build_synthetic_asset_moments(n_assets=6, mean_dispersion=0.05)
    rec = calculate_jorion_bayes_stein(means, cov, sample_size_t=10)
    assert rec.regime == "SHORT_WINDOW_ESTIMATION_RISK"


def test_jorion_input_validation_guards() -> None:
    """Validate ticker regex, sample T > N+2, matrix shape, and PSD checks."""
    means, cov = build_synthetic_asset_moments(n_assets=3)
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_jorion_bayes_stein(means, cov, 30, ticker="BAD;SQL")
    with pytest.raises(ValueError, match="Sample size T"):
        calculate_jorion_bayes_stein(means, cov, 5)
    with pytest.raises(ValueError, match="Covariance matrix shape"):
        calculate_jorion_bayes_stein(means, np.eye(2), 30)
    with pytest.raises(ValueError, match="positive semidefinite"):
        bad_cov = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        calculate_jorion_bayes_stein(means, bad_cov, 30)


def test_jorion_hmac_integrity_and_tamper_detection() -> None:
    """Verify HMAC-SHA256 seal verification and tamper sensitivity."""
    means, cov = build_synthetic_asset_moments(n_assets=4)
    rec = calculate_jorion_bayes_stein(means, cov, sample_size_t=60, ticker="VTI")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_assets=rec.n_assets,
        sample_size_t=rec.sample_size_t,
        gmv_mu0=rec.gmv_grand_mean_mu0,
        mahal_d2=rec.mahalanobis_d2,
        shrinkage_w=rec.shrinkage_intensity_w,
        bs_gross_leverage=rec.bs_gross_leverage,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("VTI", "QQQ"), rec.hmac_signature) is False


def test_jorion_empirical_profile_calibration() -> None:
    """Calibrate multi-sleeve moments from trailing historical prices."""
    rng = np.random.default_rng(175)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, size=60)))
    rec = evaluate_bayes_stein_profile(prices, ticker="SPY", n_assets=4, sample_size_t=60)
    assert rec.ticker == "SPY"
    assert 0.0 < rec.shrinkage_intensity_w <= 1.0
    assert rec.cov_inflation_ratio > 1.0


def test_jorion_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite audit persistence into jorion_bayes_stein_audits."""
    db_file = tmp_path / "jbs_audit.db"
    means, cov = build_synthetic_asset_moments(n_assets=4)
    rec = calculate_jorion_bayes_stein(means, cov, sample_size_t=60, ticker="AGG")
    assert persist_jorion_bayes_stein_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, n_assets, shrinkage_intensity_w, bs_gross_leverage, regime "
            "FROM jorion_bayes_stein_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "AGG"
    assert rows[0][2] == pytest.approx(rec.shrinkage_intensity_w, rel=1e-9)


def test_jorion_cli_integration_json() -> None:
    """Execute CLI subcommand `bayes-stein --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "bayes-stein",
            "--ticker",
            "PORT",
            "--n-assets",
            "4",
            "--sample-t",
            "60",
            "--disp",
            "0.08",
            "--vol",
            "0.20",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "PORT"
    assert 0.0 < payload["shrinkage_intensity_w"] < 1.0
    assert payload["bs_gross_leverage"] <= payload["mle_gross_leverage"] + 1e-9

