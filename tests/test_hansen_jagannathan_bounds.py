"""Unit and integration tests for Hansen & Jagannathan (1991, 1997) SDF Bounds & Distance Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.hansen_jagannathan_bounds import (
    HansenJagannathanReceipt,
    calculate_hansen_jagannathan_bounds,
    evaluate_hansen_jagannathan_profile,
    persist_hansen_jagannathan_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_minimum_variance_projection_sdf_exactness():
    """Verify minimum-variance projection SDF m* attains zero HJ-distance and meets the HJ bound."""
    rng = np.random.default_rng(123)
    t = 120
    rf = 1.02
    r1 = 1.05 + rng.normal(0.0, 0.12, size=t)
    r2 = 1.07 + rng.normal(0.0, 0.18, size=t)
    # Include risk-free asset so 1/R_f is priced in span(R)
    r_mat = np.column_stack([np.full(t, rf), r1, r2])

    receipt = calculate_hansen_jagannathan_bounds(
        gross_returns=r_mat,
        candidate_sdf=None,
        risk_free_gross=rf,
        annualization_factor=1.0,
        ticker="SDF",
    )

    assert isinstance(receipt, HansenJagannathanReceipt)
    assert receipt.ticker == "SDF"
    assert receipt.hj_distance < 1e-6
    assert receipt.mean_abs_pricing_error < 1e-6
    assert receipt.hj_bound_satisfied is True
    assert receipt.candidate_sdf_std == pytest.approx(receipt.hj_volatility_bound, rel=1e-4)
    assert receipt.regime == "EXACT_NO_ARBITRAGE_PRICING_KERNEL"


def test_equity_premium_puzzle_hj_bound_violation():
    """Verify that an overly smooth SDF fails the Hansen-Jagannathan volatility bound."""
    rng = np.random.default_rng(7)
    t = 100
    rf = 1.01
    # High Sharpe test assets
    r_high = 1.08 + rng.normal(0.0, 0.08, size=t)
    r_mat = np.column_stack([r_high, 1.06 + rng.normal(0.0, 0.07, size=t)])

    # Smooth low-volatility consumption SDF (gamma=1)
    cons_growth = rng.normal(0.02, 0.005, size=t)
    smooth_sdf = 0.99 * ((1.0 + cons_growth) ** -1.0)

    receipt = calculate_hansen_jagannathan_bounds(
        gross_returns=r_mat,
        candidate_sdf=smooth_sdf,
        risk_free_gross=rf,
        annualization_factor=1.0,
        consumption_vol_period=0.005,
        ticker="PUZZLE",
    )

    assert receipt.candidate_sdf_std < receipt.hj_volatility_bound
    assert receipt.hj_bound_satisfied is False
    assert receipt.hj_distance > 0.05
    assert receipt.regime == "EQUITY_PREMIUM_PUZZLE_HJ_BOUND_VIOLATION"


def test_orthogonal_noise_invariance_of_hj_distance():
    """Verify HJ-distance depends only on pricing errors E[mR]-1, remaining invariant to noise orthogonal to R."""
    rng = np.random.default_rng(99)
    t = 80
    r_mat = np.column_stack([1.05 + rng.normal(0.0, 0.10, size=t), 1.03 + rng.normal(0.0, 0.08, size=t)])
    g2 = (r_mat.T @ r_mat) / float(t)
    m_star = r_mat @ np.linalg.solve(g2, np.ones(2))

    # Construct vector orthogonal to R columns: R^T eps = 0
    raw_v = rng.normal(0.0, 0.25, size=t)
    eps_orth = raw_v - r_mat @ np.linalg.lstsq(r_mat, raw_v, rcond=None)[0]
    m_perturbed = m_star + eps_orth

    rec_star = calculate_hansen_jagannathan_bounds(r_mat, candidate_sdf=m_star, annualization_factor=1.0, ticker="ORTH1")
    rec_pert = calculate_hansen_jagannathan_bounds(r_mat, candidate_sdf=m_perturbed, annualization_factor=1.0, ticker="ORTH2")

    assert rec_star.hj_distance < 1e-8
    assert rec_pert.hj_distance < 1e-8
    assert rec_pert.candidate_sdf_std > rec_star.candidate_sdf_std


def test_monotonicity_sharpe_ratio_bound():
    """Verify increasing asset excess return mean raises max Sharpe and tightens HJ lower bound."""
    rng = np.random.default_rng(11)
    t = 100
    rf = 1.01
    noise = rng.normal(0.0, 0.10, size=(t, 2))
    r_low_sr = np.column_stack([1.02 + noise[:, 0], 1.015 + noise[:, 1]])
    r_high_sr = np.column_stack([1.12 + noise[:, 0], 1.09 + noise[:, 1]])

    rec_low = calculate_hansen_jagannathan_bounds(r_low_sr, risk_free_gross=rf, annualization_factor=1.0)
    rec_high = calculate_hansen_jagannathan_bounds(r_high_sr, risk_free_gross=rf, annualization_factor=1.0)

    assert rec_high.max_sharpe_period > rec_low.max_sharpe_period
    assert rec_high.implied_min_risk_aversion > rec_low.implied_min_risk_aversion


def test_singular_collinear_assets_ridge_stability():
    """Verify collinear basis assets do not cause singular matrix crashes."""
    rng = np.random.default_rng(5)
    t = 50
    base = 1.04 + rng.normal(0.0, 0.05, size=t)
    # Exact duplicate / collinear asset column
    r_mat = np.column_stack([base, base * 1.0])

    receipt = calculate_hansen_jagannathan_bounds(r_mat, annualization_factor=1.0, ticker="SING")
    assert np.isfinite(receipt.max_sharpe_period)
    assert np.isfinite(receipt.hj_distance)


def test_security_input_validation():
    """Verify strict ticker regex allowlist and dimension/finiteness validation."""
    r_valid = np.ones((20, 2)) * 1.01

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_hansen_jagannathan_bounds(r_valid, ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_hansen_jagannathan_bounds(r_valid, ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="at least 5 time observations"):
        calculate_hansen_jagannathan_bounds(np.ones((3, 2)))

    with pytest.raises(ValueError, match="must match gross_returns rows"):
        calculate_hansen_jagannathan_bounds(r_valid, candidate_sdf=np.ones(10))


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    rng = np.random.default_rng(1)
    r_mat = 1.02 + rng.normal(0.0, 0.05, size=(40, 2))
    receipt = calculate_hansen_jagannathan_bounds(r_mat, ticker="SPY")

    sig_payload = (
        f"{receipt.ticker}:{receipt.num_observations}:{receipt.num_assets}:"
        f"{receipt.max_sharpe_period:.6f}:{receipt.candidate_sdf_mean:.6f}:"
        f"{receipt.candidate_sdf_std:.6f}:{receipt.hj_volatility_bound:.6f}:"
        f"{receipt.hj_distance:.6f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_hansen_jagannathan_profile_with_prices():
    """Verify empirical calibration from historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 1.8 * np.sin(i / 2.5) for i in range(60)]

    receipt = evaluate_hansen_jagannathan_profile(
        prices=prices,
        ticker="QQQ",
        risk_aversion=10.0,
        discount_beta=0.98,
    )

    assert receipt.ticker == "QQQ"
    assert receipt.num_observations == len(prices) - 1
    assert receipt.num_assets == 3
    assert receipt.max_sharpe_annualized > 0.0


def test_persist_hansen_jagannathan_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_hj.db")
    rng = np.random.default_rng(10)
    r_mat = 1.02 + rng.normal(0.0, 0.03, size=(30, 2))
    receipt = calculate_hansen_jagannathan_bounds(r_mat, ticker="TSLA")

    success = persist_hansen_jagannathan_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, max_sharpe_annualized, hj_distance, regime FROM hansen_jagannathan_audits WHERE ticker = ?",
            ("TSLA",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "TSLA"
        assert row[1] == pytest.approx(receipt.max_sharpe_annualized, rel=1e-4)
        assert row[2] == pytest.approx(receipt.hj_distance, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_hansen_jagannathan_table_execution():
    """Verify CLI invocation for hansen-jagannathan command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "hansen-jagannathan",
        "--ticker",
        "SDF",
        "--risk-aversion",
        "8.0",
        "--beta",
        "0.98",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Hansen & Jagannathan (1991, 1997) SDF Volatility Bounds" in res_table.stdout
    assert "SDF" in res_table.stdout
    assert "HJ Volatility Lower Bound" in res_table.stdout


def test_cli_hansen_jagannathan_json_execution():
    """Verify CLI invocation for hansen-jagannathan command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "hansen-jagannathan",
        "--ticker",
        "SPY",
        "--risk-aversion",
        "12.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "SPY"
    assert "hj_volatility_bound" in parsed
    assert "hj_distance" in parsed
    assert "hmac_signature" in parsed


def test_crra_high_risk_aversion_increases_sdf_dispersion():
    """Verify raising relative risk aversion gamma strictly increases candidate SDF dispersion ratio."""
    prices = [100.0 * (1.001 ** i) * (1.0 + 0.015 * np.sin(i / 2.0)) for i in range(50)]
    r_low = evaluate_hansen_jagannathan_profile(prices, risk_aversion=2.0, ticker="LOWG")
    r_high = evaluate_hansen_jagannathan_profile(prices, risk_aversion=30.0, ticker="HIGHG")

    assert r_high.hj_dispersion_ratio > r_low.hj_dispersion_ratio
    assert r_high.candidate_sdf_std > r_low.candidate_sdf_std

