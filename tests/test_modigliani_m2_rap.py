"""Tests for Modigliani & Modigliani (1997) M^2 Risk-Adjusted Performance Engine."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.modigliani_m2_rap import (
    _build_hmac_payload,
    calculate_modigliani_m2,
    evaluate_modigliani_m2_profile,
    persist_modigliani_m2_record,
    verify_hmac_signature,
)


def test_modigliani_m2_volatility_matching_identity() -> None:
    """Verify M^2 = L* * r_p + (1 - L*) * r_f == r_f + SR_p * sigma_m."""
    rec = calculate_modigliani_m2(
        portfolio_return_rp=0.14,
        portfolio_vol_p=0.20,
        benchmark_return_rm=0.10,
        benchmark_vol_m=0.16,
        risk_free_rate_r=0.04,
        portfolio_beta=1.0,
        ticker="QQQ",
    )
    expected_m2 = 0.04 + rec.portfolio_sharpe * 0.16
    assert rec.leverage_ratio == pytest.approx(0.80, rel=1e-12)
    assert rec.cash_or_margin_weight == pytest.approx(0.20, rel=1e-12)
    assert rec.m2_return == pytest.approx(expected_m2, rel=1e-12)
    assert rec.m2_alpha_bps == pytest.approx((expected_m2 - 0.10) * 10000.0, rel=1e-12)


def test_modigliani_m2_levered_low_vol_outperformer() -> None:
    """Low-vol high-Sharpe portfolio gets L* > 1 margin boost into levered outperformer regime."""
    rec = calculate_modigliani_m2(
        portfolio_return_rp=0.10,
        portfolio_vol_p=0.08,
        benchmark_return_rm=0.10,
        benchmark_vol_m=0.16,
        risk_free_rate_r=0.04,
        portfolio_beta=0.40,
    )
    assert rec.leverage_ratio == pytest.approx(2.0, rel=1e-12)
    assert rec.cash_or_margin_weight == pytest.approx(-1.0, rel=1e-12)
    assert rec.m2_return == pytest.approx(0.16, rel=1e-12)
    assert rec.m2_alpha_spread == pytest.approx(0.06, rel=1e-12)
    assert rec.regime == "LEVERED_SHARPE_OUTPERFORMER"


def test_treynor_black_squared_sharpe_and_appraisal_identity() -> None:
    """Verify total variance decomposition and Treynor-Black SR_opt^2 = SR_m^2 + AR^2."""
    rec = calculate_modigliani_m2(
        portfolio_return_rp=0.16,
        portfolio_vol_p=0.25,
        benchmark_return_rm=0.10,
        benchmark_vol_m=0.15,
        risk_free_rate_r=0.04,
        portfolio_beta=1.20,
    )
    expected_idio = math.sqrt(0.25**2 - (1.20 * 0.15) ** 2)
    expected_alpha = (0.16 - 0.04) - 1.20 * (0.10 - 0.04)
    assert rec.idiosyncratic_vol == pytest.approx(expected_idio, rel=1e-12)
    assert rec.jensen_alpha == pytest.approx(expected_alpha, rel=1e-12)
    assert rec.appraisal_ratio == pytest.approx(expected_alpha / expected_idio, rel=1e-12)
    assert rec.treynor_black_sharpe**2 == pytest.approx(
        rec.benchmark_sharpe**2 + rec.appraisal_ratio**2, rel=1e-12
    )


def test_modigliani_m2_regime_classification_spectrum() -> None:
    """Verify all four risk-adjusted performance regimes."""
    lev = calculate_modigliani_m2(0.11, 0.10, benchmark_return_rm=0.10, benchmark_vol_m=0.16, portfolio_beta=0.5)
    unlev = calculate_modigliani_m2(0.16, 0.22, benchmark_return_rm=0.10, benchmark_vol_m=0.16, portfolio_beta=1.1)
    laggard = calculate_modigliani_m2(0.07, 0.24, benchmark_return_rm=0.10, benchmark_vol_m=0.16, portfolio_beta=1.1)
    neg = calculate_modigliani_m2(0.01, 0.18, benchmark_return_rm=0.10, benchmark_vol_m=0.16, risk_free_rate_r=0.04, portfolio_beta=0.9)

    assert lev.regime == "LEVERED_SHARPE_OUTPERFORMER"
    assert unlev.regime == "UNLEVERED_QUALITY_COMPOUNDER"
    assert laggard.regime == "HIGH_VOL_DILUTED_LAGGARD"
    assert neg.regime == "NEGATIVE_SHARPE_UNDERPERFORMER"


def test_modigliani_m2_zero_idiosyncratic_risk_boundary() -> None:
    """When vol_p == |beta| * vol_m, idiosyncratic vol vanishes and appraisal stays finite."""
    rec = calculate_modigliani_m2(
        portfolio_return_rp=0.10,
        portfolio_vol_p=0.16,
        benchmark_return_rm=0.10,
        benchmark_vol_m=0.16,
        portfolio_beta=1.0,
    )
    assert rec.idiosyncratic_vol == pytest.approx(0.0, abs=1e-12)
    assert rec.appraisal_ratio == pytest.approx(0.0, abs=1e-12)
    assert rec.treynor_black_sharpe == pytest.approx(rec.benchmark_sharpe, rel=1e-12)


def test_modigliani_m2_input_validation_guards() -> None:
    """Enforce ticker allowlist, volatility bounds, beta bounds, and variance feasibility."""
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_modigliani_m2(0.10, 0.16, ticker="SPY; DROP")
    with pytest.raises(ValueError, match="Annualized volatilities"):
        calculate_modigliani_m2(0.10, 0.0)
    with pytest.raises(ValueError, match="residual variance"):
        calculate_modigliani_m2(0.10, 0.10, benchmark_vol_m=0.16, portfolio_beta=1.0)
    with pytest.raises(ValueError, match="Portfolio beta"):
        calculate_modigliani_m2(0.10, 0.80, portfolio_beta=6.0)


def test_modigliani_m2_hmac_integrity_and_tamper_detection() -> None:
    """Validate HMAC-SHA256 seal integrity and tamper sensitivity."""
    rec = calculate_modigliani_m2(0.14, 0.20, ticker="IWM")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        rp=rec.portfolio_return_rp,
        vol_p=rec.portfolio_vol_p,
        rm=rec.benchmark_return_rm,
        vol_m=rec.benchmark_vol_m,
        rf=rec.risk_free_rate_r,
        beta=rec.portfolio_beta,
        m2_return=rec.m2_return,
        m2_alpha_bps=rec.m2_alpha_bps,
        appraisal_ratio=rec.appraisal_ratio,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("IWM", "DIA"), rec.hmac_signature) is False


def test_modigliani_m2_empirical_profile_calibration() -> None:
    """Calibrate trailing annualized return/vol profile from price closes."""
    rng = np.random.default_rng(176)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=60)))
    rec = evaluate_modigliani_m2_profile(prices, ticker="SPY")
    assert rec.ticker == "SPY"
    assert rec.leverage_ratio > 0.0


def test_modigliani_m2_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite audit persistence into modigliani_m2_audits."""
    db_file = tmp_path / "m2_audit.db"
    rec = calculate_modigliani_m2(0.14, 0.20, ticker="VOO")
    assert persist_modigliani_m2_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, leverage_ratio, m2_return, m2_alpha_bps, regime "
            "FROM modigliani_m2_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "VOO"
    assert rows[0][2] == pytest.approx(rec.m2_return, rel=1e-9)


def test_modigliani_m2_cli_integration_json() -> None:
    """Execute CLI subcommand `m2-modigliani --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "m2-modigliani",
            "--ticker",
            "QQQ",
            "--rp",
            "0.14",
            "--vol-p",
            "0.20",
            "--rm",
            "0.10",
            "--vol-m",
            "0.16",
            "--rf",
            "0.04",
            "--beta",
            "1.0",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "QQQ"
    assert payload["m2_return"] == pytest.approx(0.12, rel=1e-9)
    assert payload["m2_alpha_bps"] == pytest.approx(200.0, rel=1e-9)

