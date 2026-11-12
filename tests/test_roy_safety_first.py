"""Tests for Roy (1952), Kataoka (1963) & Telser (1955) Safety-First Portfolio Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.roy_safety_first import (
    _build_hmac_payload,
    build_synthetic_safety_first_inputs,
    evaluate_safety_first_profile,
    persist_safety_first_record,
    solve_safety_first_portfolio,
    verify_hmac_signature,
)


def test_roy_kataoka_telser_feasible_optimality() -> None:
    """When threshold is feasible, Telser mean >= Kataoka mean and Roy SFR > 0."""
    mu, sigma = build_synthetic_safety_first_inputs(n_assets=4)
    rec = solve_safety_first_portfolio(
        expected_returns=mu,
        cov_matrix=sigma,
        threshold_return=-0.10,
        alpha_shortfall=0.05,
        ticker="ENDOWMENT",
    )
    assert rec.telser_feasible is True
    assert rec.roy_sfr > 0.75
    assert rec.kataoka_floor >= -0.10
    assert rec.telser_mean >= rec.kataoka_mean - 1e-6
    assert rec.roy_shortfall_prob <= rec.chebyshev_bound + 1e-9
    assert rec.regime == "TELSER_FEASIBLE_SAFE"


def test_telser_infeasible_threshold_is_detected_cleanly() -> None:
    """Excessively high threshold flags telser_feasible=False and INFEASIBLE_THRESHOLD_BREACH."""
    mu, sigma = build_synthetic_safety_first_inputs(n_assets=4)
    rec = solve_safety_first_portfolio(
        expected_returns=mu,
        cov_matrix=sigma,
        threshold_return=0.25,
        alpha_shortfall=0.05,
        ticker="ENDOWMENT",
    )
    assert rec.telser_feasible is False
    assert rec.regime == "INFEASIBLE_THRESHOLD_BREACH"


def test_safety_first_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite persistence."""
    mu, sigma = build_synthetic_safety_first_inputs()
    with pytest.raises(ValueError, match="Invalid ticker"):
        solve_safety_first_portfolio(mu, sigma, ticker="BAD TICKER")

    with pytest.raises(ValueError, match="alpha_shortfall"):
        solve_safety_first_portfolio(mu, sigma, alpha_shortfall=0.6, ticker="SPY")

    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0006)))
    receipt = evaluate_safety_first_profile(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        n_assets=receipt.n_assets,
        threshold_return=receipt.threshold_return,
        alpha_shortfall=receipt.alpha_shortfall,
        roy_sfr=receipt.roy_sfr,
        kataoka_floor=receipt.kataoka_floor,
        telser_feasible=receipt.telser_feasible,
        telser_mean=receipt.telser_mean,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "roy_test.db"
    assert persist_safety_first_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, n_assets, regime, hmac_signature FROM roy_safety_first_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 2
    assert rows[0][3] == receipt.hmac_signature


def test_cli_safety_first_json_and_aliases() -> None:
    """Verify CLI safety-first and aliases emit valid JSON receipt."""
    for alias in ("safety-first", "roy-1952", "kataoka", "telser-1955"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--threshold",
                "-0.10",
                "--alpha",
                "0.05",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["telser_feasible"] is True
        assert data["regime"] == "TELSER_FEASIBLE_SAFE"
        assert len(data["hmac_signature"]) == 64

