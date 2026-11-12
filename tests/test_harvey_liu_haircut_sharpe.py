"""Tests for Harvey & Liu (2015) Multiple Testing & Haircut Sharpe Ratio Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.harvey_liu_haircut_sharpe import (
    _build_hmac_payload,
    calculate_haircut_sharpe,
    evaluate_haircut_sharpe_profile,
    persist_haircut_sharpe_record,
    verify_hmac_signature,
)


def test_haircut_sharpe_monotonicity_and_single_trial_identity() -> None:
    """Single trial has zero haircut; M > 1 yields SR_bhy <= SR_bonf <= SR_obs."""
    single = calculate_haircut_sharpe(sharpe_observed=1.5, n_obs=1260, n_trials=1, ticker="SPY")
    assert single.haircut_pct_bhy == pytest.approx(0.0, abs=1e-9)
    assert single.sharpe_bhy == pytest.approx(single.sharpe_observed, abs=1e-6)

    multi = calculate_haircut_sharpe(sharpe_observed=2.0, n_obs=1260, n_trials=200, rank=1, ticker="SPY")
    assert 0.0 < multi.sharpe_bhy <= multi.sharpe_bonferroni <= multi.sharpe_holm <= multi.sharpe_bh <= multi.sharpe_observed
    assert 0.0 < multi.haircut_pct_bhy < 100.0
    assert multi.regime in ("ROBUST_MULTIPLE_TEST_ALPHA", "MARGINAL_BHY_SURVIVOR")


def test_data_mined_noise_collapses_under_many_trials() -> None:
    """Marginal Sharpe under hundreds of trials is haircut down to INSIGNIFICANT_NOISE."""
    rec = calculate_haircut_sharpe(sharpe_observed=0.85, n_obs=504, n_trials=500, rank=1, ticker="QQQ")
    assert rec.p_value_bhy >= 0.10
    assert rec.sharpe_bhy == 0.0
    assert rec.haircut_pct_bhy == pytest.approx(100.0, abs=1e-6)
    assert rec.regime == "INSIGNIFICANT_NOISE"


def test_validation_guards_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite audit logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_haircut_sharpe(1.2, ticker="BAD TICKER")

    with pytest.raises(ValueError, match="n_trials"):
        calculate_haircut_sharpe(1.2, n_trials=0, ticker="SPY")

    prices = 100.0 * np.exp(np.cumsum(np.full(80, 0.0015)))
    receipt = evaluate_haircut_sharpe_profile(prices, ticker="SPY", n_trials=50)
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        n_obs=receipt.n_obs,
        n_trials=receipt.n_trials,
        sharpe_observed=receipt.sharpe_observed,
        sharpe_bh=receipt.sharpe_bh,
        sharpe_bhy=receipt.sharpe_bhy,
        haircut_pct_bhy=receipt.haircut_pct_bhy,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "hlz_test.db"
    assert persist_haircut_sharpe_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, n_trials, regime, hmac_signature FROM harvey_liu_haircut_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 50
    assert rows[0][3] == receipt.hmac_signature


def test_cli_haircut_sharpe_json_and_aliases() -> None:
    """Verify CLI haircut-sharpe and aliases emit valid JSON receipt."""
    for alias in ("haircut-sharpe", "harvey-liu", "hlz-2016", "bh-fdr-sharpe"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--sharpe",
                "1.85",
                "--n-obs",
                "1260",
                "--n-trials",
                "100",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["n_trials"] == 100
        assert data["sharpe_bhy"] < data["sharpe_observed"]
        assert len(data["hmac_signature"]) == 64
