"""Tests for Choueifaty & Coignard (2008) Maximum Diversification Portfolio Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.choueifaty_diversification import (
    _build_hmac_payload,
    build_synthetic_correlation_cov,
    calculate_maximum_diversification,
    evaluate_diversification_profile,
    persist_diversification_record,
    verify_hmac_signature,
)


def test_uncorrelated_diagonal_achieves_sqrt_n_diversification() -> None:
    """In N uncorrelated assets, MDP achieves exact DR = sqrt(N) and N_eff = N."""
    cov = build_synthetic_correlation_cov(n_assets=4, base_corr=0.0, vols=[0.12, 0.16, 0.20, 0.24])
    rec = calculate_maximum_diversification(cov, ticker="SPY")
    assert rec.dr_mdp == pytest.approx(2.0, abs=1e-5)
    assert rec.effective_independent_bets == pytest.approx(4.0, abs=1e-4)
    assert rec.dr_mdp >= rec.dr_equal_weight - 1e-8
    assert rec.active_correlation_spread < 1e-5
    assert sum(rec.weights) == pytest.approx(1.0, abs=1e-9)
    assert rec.regime == "HIGH_DIVERSIFICATION_LEVERAGE"


def test_equal_correlation_optimality_and_mdp_dominance() -> None:
    """MDP dominates Equal-Weight and Minimum-Variance on Diversification Ratio."""
    # Asymmetric correlated covariance matrix
    vols = np.array([0.12, 0.15, 0.22, 0.28])
    corr = np.array([
        [1.00, 0.70, 0.15, 0.10],
        [0.70, 1.00, 0.20, 0.15],
        [0.15, 0.20, 1.00, 0.65],
        [0.10, 0.15, 0.65, 1.00],
    ])
    cov = np.outer(vols, vols) * corr
    rec = calculate_maximum_diversification(cov, ticker="QQQ")
    assert rec.dr_mdp >= rec.dr_equal_weight - 1e-8
    assert rec.dr_mdp >= rec.dr_min_variance - 1e-8
    assert rec.active_correlation_spread < 1e-4
    assert rec.diversification_benefit_pct > 0.20


def test_concentrated_high_correlation_and_empirical_profile() -> None:
    """High uniform correlation yields concentrated regime near DR = 1.0."""
    cov = build_synthetic_correlation_cov(n_assets=4, base_corr=0.90)
    rec = calculate_maximum_diversification(cov, ticker="IWM")
    assert rec.dr_mdp < 1.20
    assert rec.regime == "CONCENTRATED_SINGLE_FACTOR_RISK"

    t = np.arange(80, dtype=float)
    prices = 100.0 * np.exp(0.0005 * t + 0.01 * np.sin(t * 0.3))
    emp = evaluate_diversification_profile(prices, ticker="AAPL")
    assert emp.ticker == "AAPL"
    assert emp.n_assets == 5

    emp_short = evaluate_diversification_profile([100.0, 101.0], ticker="MSFT")
    assert emp_short.n_assets == 5


def test_validation_guards_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Guard invalid tickers/non-PD/asymmetric matrices, verify HMAC, and test SQLite."""
    cov = build_synthetic_correlation_cov(n_assets=4, base_corr=0.25)
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_maximum_diversification(cov, ticker="../BAD")
    with pytest.raises(ValueError, match="symmetric"):
        asym = cov.copy()
        asym[0, 1] += 0.05
        calculate_maximum_diversification(asym, ticker="SPY")
    with pytest.raises(ValueError, match="strictly positive definite"):
        singular = np.ones((3, 3), dtype=float)
        calculate_maximum_diversification(singular, ticker="SPY")

    rec = calculate_maximum_diversification(cov, ticker="NVDA")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_assets=rec.n_assets,
        dr_mdp=rec.dr_mdp,
        dr_equal_weight=rec.dr_equal_weight,
        effective_independent_bets=rec.effective_independent_bets,
        mdp_volatility=rec.mdp_volatility,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("NVDA", "META"), rec.hmac_signature) is False

    db_file = str(tmp_path / "dr_test.db")
    assert persist_diversification_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, n_assets, regime FROM choueifaty_dr_audits").fetchall()
    assert rows == [("NVDA", 4, rec.regime)]


def test_cli_max_diversification_json_and_aliases() -> None:
    """Verify CLI max-diversification and aliases emit valid JSON receipt."""
    for alias in ("max-diversification", "choueifaty-dr", "cc-2008"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--n-assets",
                "4",
                "--corr",
                "0.0",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["dr_mdp"] == pytest.approx(2.0, abs=1e-4)
        assert data["regime"] == "HIGH_DIVERSIFICATION_LEVERAGE"

