"""Tests for Kupiec (1995) & Christoffersen (1998) Conditional Coverage VaR Backtest."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.kupiec_christoffersen_var import (
    _build_hmac_payload,
    build_synthetic_var_series,
    calculate_var_backtest,
    evaluate_var_backtest_profile,
    persist_var_backtest_record,
    verify_hmac_signature,
)


def test_calibrated_independent_var_passes_both_tests() -> None:
    """Evenly spaced exceptions at target alpha pass Kupiec LR_uc, LR_ind, and LR_cc."""
    rets, var_fc = build_synthetic_var_series(n_obs=250, alpha=0.05, mode="calibrated")
    rec = calculate_var_backtest(rets, var_fc, alpha=0.05, ticker="SPY")
    assert rec.lr_cc == pytest.approx(rec.lr_uc + rec.lr_ind, abs=1e-12)
    assert rec.p_value_uc > 0.05
    assert rec.p_value_ind > 0.05
    assert rec.p_value_cc > 0.05
    assert rec.basel_zone == "GREEN_ZONE"
    assert rec.regime == "VAR_MODEL_CALIBRATED_PASS"


def test_excessive_exceptions_fail_kupiec_and_clustered_fail_christoffersen() -> None:
    """Excess failures fail Kupiec; clustered exceptions at alpha fail Christoffersen LR_ind."""
    rets_bad, var_bad = build_synthetic_var_series(n_obs=250, alpha=0.05, mode="excessive")
    rec_bad = calculate_var_backtest(rets_bad, var_bad, alpha=0.05, ticker="QQQ")
    assert rec_bad.p_value_uc <= 0.05
    assert rec_bad.basel_zone == "RED_ZONE"
    assert rec_bad.regime in ("UNCONDITIONAL_COVERAGE_REJECT_KUPIEC", "CONDITIONAL_COVERAGE_REJECT_BOTH")

    # Exact 5% exception rate clustered consecutively -> passes Kupiec LR_uc, fails Independence!
    rets_clust, var_clust = build_synthetic_var_series(n_obs=250, alpha=0.05, mode="clustered")
    rec_clust = calculate_var_backtest(rets_clust, var_clust, alpha=0.05, ticker="IWM")
    assert rec_clust.p_value_uc > 0.50
    assert rec_clust.pi_11_after_breach > 0.80
    assert rec_clust.p_value_ind < 0.001
    assert rec_clust.p_value_cc < 0.001
    assert rec_clust.regime == "EXCEPTION_CLUSTERING_FAIL_INDEP"


def test_zero_exception_boundary_and_empirical_profile() -> None:
    """Zero exceptions evaluate cleanly without NaN log(0); empirical rolling profile works."""
    rets = np.full(60, 0.001)
    vars_ = np.full(60, 0.025)
    rec_zero = calculate_var_backtest(rets, vars_, alpha=0.05, ticker="TLT")
    assert rec_zero.n_exceptions == 0
    assert rec_zero.lr_ind == pytest.approx(0.0, abs=1e-12)
    assert rec_zero.lr_uc > 0.0

    prices = [100.0 * np.exp(0.0005 * i + 0.01 * np.sin(i * 0.4)) for i in range(90)]
    emp = evaluate_var_backtest_profile(prices, ticker="AAPL", alpha=0.05)
    assert emp.ticker == "AAPL"
    assert emp.n_obs == 69


def test_validation_guards_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Guard bad tickers/alphas/negative VaR, verify HMAC seal, and test SQLite."""
    rets, var_fc = build_synthetic_var_series(n_obs=100, alpha=0.05, mode="calibrated")
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_var_backtest(rets, var_fc, ticker="../BAD")
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_var_backtest(rets, -var_fc, ticker="SPY")
    with pytest.raises(ValueError, match="alpha"):
        calculate_var_backtest(rets, var_fc, alpha=0.60, ticker="SPY")

    rec = calculate_var_backtest(rets, var_fc, alpha=0.05, ticker="NVDA")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_obs=rec.n_obs,
        alpha_target=rec.alpha_target,
        n_exceptions=rec.n_exceptions,
        lr_uc=rec.lr_uc,
        lr_ind=rec.lr_ind,
        lr_cc=rec.lr_cc,
        p_value_cc=rec.p_value_cc,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("NVDA", "META"), rec.hmac_signature) is False

    db_file = str(tmp_path / "var_bt.db")
    assert persist_var_backtest_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, n_exceptions, regime FROM kupiec_christoffersen_audits").fetchall()
    assert rows == [("NVDA", rec.n_exceptions, rec.regime)]


def test_cli_var_backtest_json_and_aliases() -> None:
    """Verify CLI var-backtest and aliases emit valid JSON receipt."""
    for alias in ("var-backtest", "kupiec-pof", "christoffersen-cc"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--alpha",
                "0.05",
                "--mode",
                "clustered",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["regime"] == "EXCEPTION_CLUSTERING_FAIL_INDEP"
        assert data["lr_cc"] == pytest.approx(data["lr_uc"] + data["lr_ind"], abs=1e-9)

