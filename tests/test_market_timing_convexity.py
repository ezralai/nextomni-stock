"""Tests for Treynor-Mazuy (1966) & Henriksson-Merton (1981) Market-Timing Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.market_timing_convexity import (
    _build_hmac_payload,
    build_synthetic_timing_returns,
    calculate_market_timing,
    evaluate_market_timing_profile,
    persist_market_timing_record,
    verify_hmac_signature,
)


def test_hm_exact_zero_noise_bull_bear_beta_recovery() -> None:
    """Zero-noise HM payoff recovers beta_down, gamma_hm, beta_up, and positive TM curvature."""
    rp, rm = build_synthetic_timing_returns(
        beta_down=0.60,
        gamma_hm=0.80,
        alpha_ann=0.0,
        n_obs=160,
        noise_vol_ann=0.0,
        seed=180,
    )
    rec = calculate_market_timing(rp, rm, risk_free_annual=0.0, ticker="QQQ")
    assert rec.hm_beta_down == pytest.approx(0.60, abs=1e-10)
    assert rec.hm_gamma == pytest.approx(0.80, abs=1e-10)
    assert rec.hm_beta_up == pytest.approx(1.40, abs=1e-10)
    assert rec.tm_gamma > 0.0
    assert rec.hm_r2 == pytest.approx(1.0, abs=1e-10)
    assert rec.regime == "CONVEX_MARKET_TIMER"


def test_tm_and_hm_concave_adverse_timer_regime() -> None:
    """Negative timing gamma captures concave adverse market timing (buys high, sells low)."""
    rp, rm = build_synthetic_timing_returns(
        beta_down=1.20,
        gamma_hm=-0.65,
        n_obs=160,
        noise_vol_ann=0.02,
        seed=77,
    )
    rec = calculate_market_timing(rp, rm, risk_free_annual=0.0, ticker="FOMO")
    assert rec.hm_gamma < -0.05
    assert rec.tm_gamma < 0.0
    assert rec.hm_beta_up < rec.hm_beta_down
    assert rec.regime == "CONCAVE_ADVERSE_TIMER"


def test_linear_passive_exposure_zero_timing_limit() -> None:
    """Pure passive index replication yields symmetric bull/bear beta and linear regime."""
    rp, rm = build_synthetic_timing_returns(
        beta_down=1.00,
        gamma_hm=0.00,
        n_obs=160,
        noise_vol_ann=0.015,
        seed=12,
    )
    rec = calculate_market_timing(rp, rm, risk_free_annual=0.0, ticker="SPY")
    assert abs(rec.hm_gamma) < 0.15
    assert abs(rec.hm_gamma_tstat) < 1.96
    assert rec.regime == "LINEAR_PASSIVE_EXPOSURE"


def test_option_like_downside_shield_regime() -> None:
    """Moderate asymmetric protection without extreme t-stat maps to downside shield regime."""
    rp, rm = build_synthetic_timing_returns(
        beta_down=0.55,
        gamma_hm=0.25,
        n_obs=160,
        noise_vol_ann=0.08,
        seed=45,
    )
    rec = calculate_market_timing(rp, rm, risk_free_annual=0.0, ticker="SHLD")
    assert rec.hm_beta_down < 0.85 * rec.hm_beta_up
    assert rec.regime in ("OPTION_LIKE_DOWNSIDE_SHIELD", "CONVEX_MARKET_TIMER")


def test_market_timing_input_validation_and_singularity_guards() -> None:
    """Validate ticker regex, sample length, variance, and upside/downside balance."""
    rp, rm = build_synthetic_timing_returns(n_obs=40)
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_market_timing(rp, rm, ticker="SPY; DROP")
    with pytest.raises(ValueError, match="At least 20 paired"):
        calculate_market_timing(rp[:10], rm[:10])
    with pytest.raises(ValueError, match="upside and downside observations"):
        pos_only = np.abs(rm) + 0.005
        calculate_market_timing(pos_only, pos_only, risk_free_annual=0.0)
    with pytest.raises(ValueError, match="variance is too small"):
        calculate_market_timing(rp, np.full(40, 0.001))


def test_market_timing_hmac_integrity_and_tamper_detection() -> None:
    """Validate HMAC-SHA256 seal integrity and tamper sensitivity."""
    rp, rm = build_synthetic_timing_returns(beta_down=0.70, gamma_hm=0.50, n_obs=100)
    rec = calculate_market_timing(rp, rm, ticker="IWM")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_obs=rec.n_obs,
        tm_gamma=rec.tm_gamma,
        hm_beta_down=rec.hm_beta_down,
        hm_beta_up=rec.hm_beta_up,
        hm_gamma=rec.hm_gamma,
        hm_gamma_tstat=rec.hm_gamma_tstat,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("IWM", "DIA"), rec.hmac_signature) is False


def test_market_timing_empirical_profile_calibration() -> None:
    """Calibrate directional bull/bear timing profile from trailing close prices."""
    rng = np.random.default_rng(180)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.013, size=80)))
    rec = evaluate_market_timing_profile(prices, ticker="SPY")
    assert rec.ticker == "SPY"
    assert rec.bull_bear_beta_spread == pytest.approx(rec.hm_beta_up - rec.hm_beta_down, rel=1e-12)


def test_market_timing_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite persistence into market_timing_convexity_audits."""
    db_file = tmp_path / "mt_audit.db"
    rp, rm = build_synthetic_timing_returns(beta_down=0.70, gamma_hm=0.50, n_obs=100)
    rec = calculate_market_timing(rp, rm, ticker="XLK")
    assert persist_market_timing_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, tm_gamma, hm_beta_down, hm_beta_up, regime "
            "FROM market_timing_convexity_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "XLK"
    assert rows[0][3] == pytest.approx(rec.hm_beta_up, rel=1e-9)


def test_market_timing_cli_integration_json() -> None:
    """Execute CLI subcommand `market-timing --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "market-timing",
            "--ticker",
            "SPY",
            "--beta-down",
            "0.65",
            "--gamma-hm",
            "0.75",
            "--n-obs",
            "160",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "SPY"
    assert payload["hm_beta_up"] > payload["hm_beta_down"]
    assert payload["tm_gamma"] > 0.0

