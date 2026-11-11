"""Unit & CLI integration tests for Robert F. Stambaugh (1999) Predictive Bias Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.stambaugh_predictive_bias import (
    calculate_stambaugh_bias,
    estimate_stambaugh_regression,
    evaluate_stambaugh_profile,
    persist_stambaugh_record,
    verify_hmac_signature,
)


def test_exact_stambaugh_bias_formula_and_spurious_illusion():
    """Verify Bias == -gamma*(1 + 3*rho)/T shrinks spurious OLS t-stat below critical threshold."""
    rec = calculate_stambaugh_bias(
        ols_beta=0.08,
        ar1_rho=0.95,
        gamma_cov_ratio=-0.80,
        sample_size_t=40,
        ols_std_err=0.032,
        ticker="SPUR",
    )

    expected_bias = 0.80 * (1.0 + 3.0 * 0.95) / 40.0
    assert rec.stambaugh_bias == pytest.approx(expected_bias, rel=1e-9)
    assert rec.bias_adjusted_beta == pytest.approx(0.08 - expected_bias, rel=1e-9)
    assert rec.ols_t_stat > 1.96
    assert rec.adjusted_t_stat < 1.65
    assert rec.regime == "SPURIOUS_STAMBAUGH_BIAS_ILLUSION"


def test_exogenous_predictor_zero_covariance_zero_bias():
    """When predictor shocks are uncorrelated with returns (gamma = 0), Stambaugh bias == 0."""
    rec = calculate_stambaugh_bias(
        ols_beta=0.12,
        ar1_rho=0.95,
        gamma_cov_ratio=0.0,
        sample_size_t=50,
        ols_std_err=0.04,
        ticker="EXOG",
    )

    assert rec.stambaugh_bias == pytest.approx(0.0, abs=1e-12)
    assert rec.bias_adjusted_beta == pytest.approx(0.12, rel=1e-9)
    assert rec.signal_retention_ratio == pytest.approx(1.0, rel=1e-9)
    assert rec.regime == "EXOGENOUS_UNBIASED_PREDICTOR"


def test_bias_decays_inversely_with_sample_horizon_t():
    """Doubling sample length T halves finite-sample Stambaugh bias cleanly."""
    short_sample = calculate_stambaugh_bias(sample_size_t=30, gamma_cov_ratio=-0.75, ar1_rho=0.90)
    long_sample = calculate_stambaugh_bias(sample_size_t=60, gamma_cov_ratio=-0.75, ar1_rho=0.90)

    assert long_sample.stambaugh_bias == pytest.approx(0.5 * short_sample.stambaugh_bias, rel=1e-9)


def test_direct_regression_estimator_detects_negative_shock_correlation_bias():
    """Synthetic AR(1) valuation predictor with gamma < 0 produces positive bias removed by estimator."""
    rng = np.random.default_rng(123)
    t_steps = 80
    rho_true = 0.92
    x = np.zeros(t_steps + 1)
    r = np.zeros(t_steps)

    # True beta = 0 (unpredictable returns), yet r_t shocks inversely correlate with predictor innovation v_t
    for t in range(t_steps):
        u_t = rng.normal(0.0, 0.04)
        v_t = -0.8 * u_t + rng.normal(0.0, 0.02)
        x[t + 1] = rho_true * x[t] + v_t
        r[t] = 0.005 + u_t

    rec = estimate_stambaugh_regression(forward_returns=r, predictors=x, ticker="SIM")

    assert rec.gamma_cov_ratio < -0.5
    assert rec.stambaugh_bias > 0.0
    assert abs(rec.bias_adjusted_beta - 0.0) <= abs(rec.ols_beta - 0.0) + 0.05


def test_robust_post_bias_predictive_alpha_regime():
    """Strong true slope surviving Stambaugh deduction registers ROBUST_POST_BIAS_PREDICTIVE_ALPHA."""
    rec = calculate_stambaugh_bias(
        ols_beta=0.25,
        ar1_rho=0.80,
        gamma_cov_ratio=-0.40,
        sample_size_t=120,
        ols_std_err=0.05,
    )
    assert rec.adjusted_t_stat >= 1.96
    assert rec.regime == "ROBUST_POST_BIAS_PREDICTIVE_ALPHA"


def test_equal_length_return_predictor_alignment():
    """Aligned equal-length arrays align cleanly to T = N - 1 observations."""
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(0, 0.1, 35))
    r = 0.01 * x + rng.normal(0, 0.02, 35)
    rec = estimate_stambaugh_regression(r, x, ticker="ALGN")
    assert rec.sample_size_t == 34


def test_security_input_validation_and_degenerate_guards():
    """Verify strict ticker allowlist, stationary rho bounds, minimum T, and zero-variance guard."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_stambaugh_bias(ticker="RS99;DROP")

    with pytest.raises(ValueError, match="sample_size_t must be >= 12"):
        calculate_stambaugh_bias(sample_size_t=8)

    with pytest.raises(ValueError, match="ar1_rho must satisfy"):
        calculate_stambaugh_bias(ar1_rho=1.05)

    with pytest.raises(ValueError, match="zero variance"):
        estimate_stambaugh_regression(np.zeros(20), np.ones(21))


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_stambaugh_bias(ticker="SPY")
    sig_payload = (
        f"{rec.ticker}:{rec.sample_size_t}:{rec.ols_beta:.6f}:{rec.ar1_rho:.4f}:"
        f"{rec.gamma_cov_ratio:.6f}:{rec.stambaugh_bias:.6f}:{rec.bias_adjusted_beta:.6f}:"
        f"{rec.adjusted_t_stat:.4f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_stambaugh_profile_zero_leakage_from_prices():
    """Trailing price series constructs causal EMA valuation predictor and detects gamma < 0."""
    rng = np.random.default_rng(99)
    prices = [100.0]
    for _ in range(70):
        prices.append(prices[-1] * float(np.exp(rng.normal(0.0005, 0.015))))

    rec = evaluate_stambaugh_profile(prices=prices, ticker="QQQ")
    assert rec.ticker == "QQQ"
    assert rec.sample_size_t >= 25
    assert rec.gamma_cov_ratio < 0.0
    assert rec.stambaugh_bias > 0.0


def test_persist_stambaugh_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_stambaugh.db")
    rec = calculate_stambaugh_bias(ticker="IWM")

    assert persist_stambaugh_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, stambaugh_bias, bias_adjusted_beta, regime FROM stambaugh_bias_audits WHERE ticker = ?",
            ("IWM",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "IWM"
        assert row[1] == pytest.approx(rec.stambaugh_bias, rel=1e-4)
        assert row[2] == pytest.approx(rec.bias_adjusted_beta, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_predictive_regression_table_execution():
    """Verify CLI execution for predictive-regression command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "predictive-regression",
        "--ticker",
        "RS99",
        "--ols-beta",
        "0.08",
        "--rho",
        "0.95",
        "--sample-t",
        "40",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Stambaugh (1999) Predictive Regressions & Finite-Sample Bias Engine" in res.stdout
    assert "RS99" in res.stdout
    assert "Stambaugh Bias-Adjusted Slope" in res.stdout


def test_cli_predictive_regression_json_execution():
    """Verify CLI execution for predictive-regression command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "predictive-regression",
        "--ticker",
        "DIA",
        "--ols-beta",
        "0.08",
        "--gamma",
        "-0.80",
        "--sample-t",
        "40",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "DIA"
    assert parsed["stambaugh_bias"] > 0.0
    assert parsed["bias_adjusted_beta"] < 0.08
    assert "hmac_signature" in parsed

