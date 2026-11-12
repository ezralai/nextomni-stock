"""Tests for Engle (2002) Dynamic Conditional Correlation (DCC-GARCH(1,1)) Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.dcc_garch import (
    _build_hmac_payload,
    build_synthetic_dcc_returns,
    evaluate_dcc_profile,
    fit_dcc_garch,
    persist_dcc_record,
    verify_hmac_signature,
)


def test_dcc_garch_crisis_spike_and_stationarity_constraints() -> None:
    """DCC-GARCH detects crisis correlation spike and satisfies a+b < 1."""
    rets = build_synthetic_dcc_returns(n_obs=220, regime="crisis_spike", seed=11)
    receipt = fit_dcc_garch(rets, ticker="SPY_TLT")
    assert receipt.n_obs == 220
    assert receipt.n_assets == 2
    assert 0.0 <= receipt.a_dcc < 1.0
    assert 0.0 <= receipt.b_dcc < 1.0
    assert receipt.dcc_persistence < 1.0
    assert -1.0 <= receipt.corr_min <= receipt.latest_corr_mean <= receipt.corr_max <= 1.0
    assert receipt.dcc_loglik >= receipt.ccc_loglik
    assert receipt.regime == "CRISIS_CORRELATION_SPIKE"


def test_dcc_garch_low_corr_regime() -> None:
    """Low correlation asset pair maps to LOW_DIVERSIFIED_CORRELATION."""
    rets = build_synthetic_dcc_returns(n_obs=180, regime="low_corr", seed=23)
    receipt = fit_dcc_garch(rets, ticker="GLD_SPY")
    assert receipt.dcc_persistence < 1.0
    assert receipt.latest_corr_mean < 0.35
    assert receipt.regime == "LOW_DIVERSIFIED_CORRELATION"


def test_dcc_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite persistence."""
    rets = build_synthetic_dcc_returns(n_obs=100, regime="calm")
    with pytest.raises(ValueError, match="Invalid ticker"):
        fit_dcc_garch(rets, ticker="BAD TICKER")

    with pytest.raises(ValueError, match="at least 40"):
        fit_dcc_garch(rets[:20], ticker="SPY")

    prices = 100.0 * np.exp(np.cumsum(np.random.default_rng(5).normal(0.0005, 0.012, 80)))
    receipt = evaluate_dcc_profile(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        n_obs=receipt.n_obs,
        n_assets=receipt.n_assets,
        a_dcc=receipt.a_dcc,
        b_dcc=receipt.b_dcc,
        unconditional_corr_mean=receipt.unconditional_corr_mean,
        latest_corr_mean=receipt.latest_corr_mean,
        dcc_loglik=receipt.dcc_loglik,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "dcc_test.db"
    assert persist_dcc_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, n_obs, regime, hmac_signature FROM dcc_garch_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 79
    assert rows[0][3] == receipt.hmac_signature


def test_cli_dcc_garch_json_and_aliases() -> None:
    """Verify CLI dcc-garch and aliases emit valid JSON receipt."""
    for alias in ("dcc-garch", "dcc-2002", "dynamic-correlation", "engle-dcc"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--regime",
                "crisis_spike",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["dcc_persistence"] < 1.0
        assert data["regime"] == "CRISIS_CORRELATION_SPIKE"
        assert len(data["hmac_signature"]) == 64

