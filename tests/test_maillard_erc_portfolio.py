"""Tests for Maillard, Roncalli & Teiletche (2010) & Spinu (2013) ERC Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.maillard_erc_portfolio import (
    _build_hmac_payload,
    build_synthetic_erc_cov,
    calculate_erc_portfolio,
    evaluate_erc_profile,
    persist_erc_record,
    verify_hmac_signature,
)


def test_equal_risk_parity_convergence_and_volatility_ordering() -> None:
    """Verify exact Euler risk parity RC_i = 1/N and sigma_ERC <= sigma_EW."""
    cov = build_synthetic_erc_cov(vols=[0.08, 0.14, 0.22, 0.34], base_corr=0.30)
    rec = calculate_erc_portfolio(cov, ticker="SPY")
    assert rec.max_risk_budget_error < 1e-9
    for share in rec.risk_shares:
        assert share == pytest.approx(0.25, abs=1e-9)
    assert rec.effective_risk_bets == pytest.approx(4.0, abs=1e-6)
    assert rec.erc_volatility <= rec.equal_weight_volatility + 1e-12
    assert sum(rec.weights) == pytest.approx(1.0, abs=1e-12)
    assert rec.regime == "EXACT_EQUAL_RISK_PARITY"


def test_uncorrelated_inverse_vol_equivalence_and_custom_budgets() -> None:
    """Zero-correlation ERC matches 1/sigma_i weights; custom budgets match targets."""
    vols = np.array([0.10, 0.20, 0.40])
    cov_diag = build_synthetic_erc_cov(vols=vols, base_corr=0.0)
    rec_diag = calculate_erc_portfolio(cov_diag, ticker="QQQ")
    expected_w = (1.0 / vols) / np.sum(1.0 / vols)
    for w, ew in zip(rec_diag.weights, expected_w):
        assert w == pytest.approx(float(ew), abs=1e-7)

    budgets = [0.50, 0.30, 0.20]
    cov_corr = build_synthetic_erc_cov(vols=vols, base_corr=0.35)
    rec_custom = calculate_erc_portfolio(cov_corr, risk_budgets=budgets, ticker="IWM")
    assert rec_custom.max_risk_budget_error < 1e-9
    for rs, b in zip(rec_custom.risk_shares, budgets):
        assert rs == pytest.approx(b, abs=1e-9)
    assert rec_custom.regime == "CUSTOM_RISK_BUDGET_CONVERGED"


def test_empirical_profile_and_validation_guards_hmac_sqlite(tmp_path) -> None:
    """Guard invalid inputs/budgets/non-PD, verify HMAC, and test SQLite persistence."""
    emp = evaluate_erc_profile([100.0 + (i % 6) * 0.5 for i in range(40)], ticker="AAPL")
    assert emp.n_assets == 4
    assert emp.max_risk_budget_error < 1e-9

    emp_short = evaluate_erc_profile([100.0, 101.0], ticker="MSFT")
    assert emp_short.regime == "EXACT_EQUAL_RISK_PARITY"

    cov = build_synthetic_erc_cov(vols=[0.12, 0.20, 0.30], base_corr=0.25)
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_erc_portfolio(cov, ticker="../BAD")
    with pytest.raises(ValueError, match="strictly positive floats"):
        calculate_erc_portfolio(cov, risk_budgets=[0.5, 0.5, 0.0], ticker="SPY")
    with pytest.raises(ValueError, match="positive definite"):
        calculate_erc_portfolio(np.ones((3, 3), dtype=float), ticker="SPY")

    rec = calculate_erc_portfolio(cov, ticker="NVDA")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_assets=rec.n_assets,
        erc_volatility=rec.erc_volatility,
        equal_weight_volatility=rec.equal_weight_volatility,
        max_risk_budget_error=rec.max_risk_budget_error,
        effective_risk_bets=rec.effective_risk_bets,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("NVDA", "META"), rec.hmac_signature) is False

    db_file = str(tmp_path / "erc_test.db")
    assert persist_erc_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, n_assets, regime FROM maillard_erc_audits").fetchall()
    assert rows == [("NVDA", 3, rec.regime)]


def test_cli_erc_portfolio_json_and_aliases() -> None:
    """Verify CLI erc-portfolio and aliases emit valid JSON receipt."""
    for alias in ("erc-portfolio", "risk-budget", "mrt-2010"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--corr",
                "0.30",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["max_risk_budget_error"] < 1e-8
        assert data["regime"] == "EXACT_EQUAL_RISK_PARITY"

