"""Tests for Gârleanu & Pedersen (2013) Dynamic Trading Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.garleanu_pedersen_trading import (
    _build_hmac_payload,
    build_synthetic_gp_inputs,
    evaluate_gp_profile,
    persist_gp_record,
    solve_garleanu_pedersen,
    verify_hmac_signature,
)


def test_gp_aim_weights_slow_alpha_more_than_fast_alpha() -> None:
    """Under positive transaction costs, slow alpha is attenuated less than fast alpha."""
    sigma, b_mat, phi, f_t, x_prev = build_synthetic_gp_inputs(
        n_assets=3,
        fast_rho=0.45,
        slow_rho=0.03,
    )
    receipt = solve_garleanu_pedersen(
        sigma=sigma,
        b_loadings=b_mat,
        phi_decay=phi,
        f_t=f_t,
        x_prev=x_prev,
        risk_aversion=2.5,
        cost_bps=25.0,
        ticker="SPY",
    )
    assert receipt.riccati_residual < 1e-11
    assert 0.0 < receipt.mean_trade_rate < 1.0
    assert receipt.slow_alpha_weight_ratio > receipt.fast_alpha_weight_ratio
    assert receipt.turnover_reduction_pct > 0.0
    assert receipt.optimal_trade_norm < receipt.myopic_norm


def test_gp_zero_cost_recovers_myopic_markowitz_and_higher_cost_slows_trading() -> None:
    """Zero cost recovers trade_rate=1.0; higher cost reduces optimal trading speed."""
    sigma, b_mat, phi, f_t, x_prev = build_synthetic_gp_inputs()
    zero_rec = solve_garleanu_pedersen(
        sigma=sigma,
        b_loadings=b_mat,
        phi_decay=phi,
        f_t=f_t,
        x_prev=x_prev,
        cost_bps=0.0,
        ticker="QQQ",
    )
    low_rec = solve_garleanu_pedersen(
        sigma=sigma,
        b_loadings=b_mat,
        phi_decay=phi,
        f_t=f_t,
        x_prev=x_prev,
        cost_bps=5.0,
        ticker="QQQ",
    )
    high_rec = solve_garleanu_pedersen(
        sigma=sigma,
        b_loadings=b_mat,
        phi_decay=phi,
        f_t=f_t,
        x_prev=x_prev,
        cost_bps=60.0,
        ticker="QQQ",
    )
    assert zero_rec.mean_trade_rate == pytest.approx(1.0, abs=1e-9)
    assert zero_rec.aim_norm == pytest.approx(zero_rec.myopic_norm, abs=1e-9)
    assert low_rec.mean_trade_rate > high_rec.mean_trade_rate


def test_gp_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite persistence."""
    sigma, b_mat, phi, f_t, x_prev = build_synthetic_gp_inputs()
    with pytest.raises(ValueError, match="Invalid ticker"):
        solve_garleanu_pedersen(sigma, b_mat, phi, f_t, x_prev, ticker="BAD TICKER")

    with pytest.raises(ValueError, match="risk_aversion"):
        solve_garleanu_pedersen(sigma, b_mat, phi, f_t, x_prev, risk_aversion=-1.0, ticker="SPY")

    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.001)))
    receipt = evaluate_gp_profile(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        n_assets=receipt.n_assets,
        n_factors=receipt.n_factors,
        risk_aversion=receipt.risk_aversion,
        cost_bps=receipt.cost_bps,
        mean_trade_rate=receipt.mean_trade_rate,
        aim_norm=receipt.aim_norm,
        optimal_trade_norm=receipt.optimal_trade_norm,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "gp_test.db"
    assert persist_gp_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, n_assets, regime, hmac_signature FROM gp_dynamic_trading_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 2
    assert rows[0][3] == receipt.hmac_signature


def test_cli_gp_trading_json_and_aliases() -> None:
    """Verify CLI gp-trading and aliases emit valid JSON receipt."""
    for alias in ("gp-trading", "gp-dynamic", "gp-2013", "dynamic-aim"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--cost-bps",
                "20",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["riccati_residual"] < 1e-11
        assert 0.0 < data["mean_trade_rate"] < 1.0
        assert len(data["hmac_signature"]) == 64
