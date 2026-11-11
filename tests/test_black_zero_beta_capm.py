"""Unit & CLI integration tests for Fischer Black (1972) Zero-Beta CAPM Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.black_zero_beta_capm import (
    calculate_black_zero_beta_capm,
    evaluate_black_zero_beta_profile,
    persist_black_zero_beta_record,
    solve_frontier_zero_beta_portfolio,
    verify_hmac_signature,
)


def test_exact_markowitz_frontier_zero_covariance_orthogonality():
    """On a multi-asset Markowitz frontier, solve_frontier_zero_beta_portfolio yields Cov(R_z, R_m)==0."""
    mu = np.array([0.06, 0.10, 0.14])
    cov = np.array(
        [
            [0.020, 0.005, 0.003],
            [0.005, 0.045, 0.012],
            [0.003, 0.012, 0.080],
        ]
    )

    res = solve_frontier_zero_beta_portfolio(expected_returns=mu, cov_matrix=cov)
    w_z = np.asarray(res["zero_beta_weights"])
    w_m = np.asarray(res["market_weights"])

    assert np.sum(w_z) == pytest.approx(1.0, rel=1e-9)
    assert np.sum(w_m) == pytest.approx(1.0, rel=1e-9)
    assert res["zero_beta_covariance"] == pytest.approx(0.0, abs=1e-12)
    assert res["zero_beta_expected_return"] < res["gmv_expected_return"] < res["market_expected_return"]


def test_sml_flattening_alpha_wedge_identity():
    """Black (1972) SML flattening produces exact alpha wedge (R_z - r_f) * (1 - beta)."""
    defensive = calculate_black_zero_beta_capm(
        asset_beta=0.60,
        risk_free_rate_rf=0.03,
        zero_beta_rate_rz=0.06,
        market_return_rm=0.10,
        ticker="JNJ",
    )
    assert defensive.sml_alpha_wedge_pct == pytest.approx((0.03 * 0.40) * 100.0, rel=1e-9)
    assert defensive.black_expected_return > defensive.sharpe_lintner_return
    assert defensive.regime == "LOW_BETA_DEFENSIVE_POSITIVE_BLACK_ALPHA"

    aggressive = calculate_black_zero_beta_capm(
        asset_beta=1.50,
        risk_free_rate_rf=0.03,
        zero_beta_rate_rz=0.06,
        market_return_rm=0.10,
        ticker="NVDA",
    )
    assert aggressive.sml_alpha_wedge_pct == pytest.approx((-0.03 * 0.50) * 100.0, rel=1e-9)
    assert aggressive.black_expected_return < aggressive.sharpe_lintner_return
    assert aggressive.regime == "HIGH_BETA_BORROWING_CONSTRAINED_OVERPRICED"


def test_unit_beta_market_asset_invariant_to_zero_beta_spread():
    """At beta = 1.0, Black Zero-Beta CAPM and Sharpe-Lintner CAPM agree on E[R_m]."""
    rec = calculate_black_zero_beta_capm(
        asset_beta=1.00,
        risk_free_rate_rf=0.03,
        zero_beta_rate_rz=0.06,
        market_return_rm=0.10,
    )
    assert rec.black_expected_return == pytest.approx(0.10, rel=1e-9)
    assert rec.sharpe_lintner_return == pytest.approx(0.10, rel=1e-9)
    assert rec.sml_alpha_wedge_pct == pytest.approx(0.0, abs=1e-12)
    assert rec.regime == "NEUTRAL_BETA_FLATTENED_SML_EQUILIBRIUM"


def test_unconstrained_sharpe_lintner_recovery_when_rz_equals_rf():
    """When E[R_z] == r_f, Black (1972) collapses cleanly to Sharpe-Lintner CAPM."""
    rec = calculate_black_zero_beta_capm(
        asset_beta=0.70,
        risk_free_rate_rf=0.04,
        zero_beta_rate_rz=0.04,
        market_return_rm=0.10,
    )
    assert rec.black_expected_return == pytest.approx(rec.sharpe_lintner_return, rel=1e-9)
    assert rec.sml_flattening_pct == pytest.approx(0.0, abs=1e-12)
    assert rec.regime == "UNCONSTRAINED_SHARPE_LINTNER_PARITY"


def test_custom_market_weights_zero_beta_covariance_zero():
    """Verify exact Cov(w_z, w_m) == 0 when arbitrary valid risky market weights are supplied."""
    mu = np.array([0.07, 0.11, 0.15])
    cov = np.diag([0.02, 0.04, 0.09])
    custom_w_m = np.array([0.15, 0.35, 0.50])

    res = solve_frontier_zero_beta_portfolio(mu, cov, market_weights=custom_w_m)
    w_z = np.asarray(res["zero_beta_weights"])
    assert float(w_z @ cov @ custom_w_m) == pytest.approx(0.0, abs=1e-12)


def test_frontier_solver_rejects_degenerate_returns_and_gmv_weights():
    """Verify frontier solver guards against identical returns and GMV singularity."""
    cov = np.diag([0.02, 0.04])
    with pytest.raises(ValueError, match="degenerate frontier"):
        solve_frontier_zero_beta_portfolio([0.08, 0.08], cov)

    with pytest.raises(ValueError, match="positive definite"):
        solve_frontier_zero_beta_portfolio([0.06, 0.10], np.array([[0.04, 0.04], [0.04, 0.04]]))


def test_security_input_validation():
    """Verify strict ticker allowlist, R_m > R_z >= r_f hierarchy, and beta bounds."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_black_zero_beta_capm(ticker="BZ72;DROP")

    with pytest.raises(ValueError, match="Restricted borrowing requires"):
        calculate_black_zero_beta_capm(risk_free_rate_rf=0.05, zero_beta_rate_rz=0.03)

    with pytest.raises(ValueError, match="Positive equity risk premium requires"):
        calculate_black_zero_beta_capm(zero_beta_rate_rz=0.08, market_return_rm=0.07)

    with pytest.raises(ValueError, match="asset_beta must be finite"):
        calculate_black_zero_beta_capm(asset_beta=10.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_black_zero_beta_capm(ticker="PG")
    sig_payload = (
        f"{rec.ticker}:{rec.asset_beta:.4f}:{rec.risk_free_rate_rf:.6f}:"
        f"{rec.zero_beta_rate_rz:.6f}:{rec.market_return_rm:.6f}:"
        f"{rec.black_expected_return:.6f}:{rec.sharpe_lintner_return:.6f}:"
        f"{rec.sml_alpha_wedge_pct:.4f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_black_zero_beta_profile_from_prices():
    """Verify trailing volatility-to-beta calibration from asset price series."""
    prices = [100.0 + 0.25 * i + 1.5 * np.sin(i / 2.5) for i in range(40)]
    rec = evaluate_black_zero_beta_profile(prices=prices, ticker="KO")

    assert rec.ticker == "KO"
    assert 0.35 <= rec.asset_beta <= 2.20
    assert rec.market_return_rm > rec.zero_beta_rate_rz


def test_persist_black_zero_beta_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_zero_beta.db")
    rec = calculate_black_zero_beta_capm(ticker="PEP")

    assert persist_black_zero_beta_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, black_expected_return, sml_alpha_wedge_pct, regime FROM black_zero_beta_audits WHERE ticker = ?",
            ("PEP",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "PEP"
        assert row[1] == pytest.approx(rec.black_expected_return, rel=1e-4)
        assert row[2] == pytest.approx(rec.sml_alpha_wedge_pct, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_zero_beta_table_execution():
    """Verify CLI execution for zero-beta command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "zero-beta",
        "--ticker",
        "BZ72",
        "--beta",
        "0.65",
        "--rz",
        "0.055",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Fischer Black (1972) Zero-Beta CAPM & SML Flattening Engine" in res.stdout
    assert "BZ72" in res.stdout
    assert "SML Flattening" in res.stdout


def test_cli_zero_beta_json_execution():
    """Verify CLI execution for zero-beta command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "zero-beta",
        "--ticker",
        "WMT",
        "--beta",
        "0.60",
        "--rz",
        "0.06",
        "--rf",
        "0.03",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "WMT"
    assert parsed["sml_alpha_wedge_pct"] > 0.0
    assert "hmac_signature" in parsed

