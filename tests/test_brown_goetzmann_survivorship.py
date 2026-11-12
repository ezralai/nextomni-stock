"""Tests for Brown, Goetzmann, Ibbotson & Ross (1992) Survivorship Bias Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.brown_goetzmann_survivorship import (
    _build_hmac_payload,
    calculate_survivorship_bias,
    evaluate_survivorship_profile,
    persist_brown_goetzmann_record,
    verify_hmac_signature,
)


def test_bgir_inverse_mills_mean_uplift_and_vol_attenuation() -> None:
    """Survivors simultaneously report higher mean and lower volatility (variance attenuation)."""
    rec = calculate_survivorship_bias(
        true_universe_mean=0.08,
        true_universe_vol=0.20,
        liquidation_cutoff_c=-0.10,
        horizon_years=5.0,
        risk_free_rate_r=0.04,
        ticker="HFRI",
    )
    assert rec.truncation_z_alpha == pytest.approx(-0.90, rel=1e-12)
    assert rec.observed_survivor_mean > rec.true_universe_mean
    assert rec.survivorship_return_bias == pytest.approx(
        rec.true_universe_vol * rec.inverse_mills_ratio, rel=1e-12
    )
    assert 0.0 < rec.variance_attenuation_factor < 1.0
    assert rec.observed_survivor_vol < rec.true_universe_vol
    assert rec.apparent_survivor_sharpe > rec.true_universe_sharpe


def test_bgir_multiyear_compounded_attrition_rate() -> None:
    """Verify cumulative attrition increases monotonically across longer horizons."""
    rec_1y = calculate_survivorship_bias(0.06, 0.20, liquidation_cutoff_c=-0.05, horizon_years=1.0)
    rec_10y = calculate_survivorship_bias(0.06, 0.20, liquidation_cutoff_c=-0.05, horizon_years=10.0)
    assert rec_10y.multiyear_survival_prob == pytest.approx(
        rec_1y.annual_survival_prob ** 10, rel=1e-12
    )
    assert rec_10y.cumulative_attrition_rate > rec_1y.cumulative_attrition_rate


def test_bgir_monotone_bias_in_harsher_cutoff() -> None:
    """Raising liquidation threshold strictly cuts survival rate and inflates return bias."""
    mild = calculate_survivorship_bias(0.08, 0.20, liquidation_cutoff_c=-0.40)
    harsh = calculate_survivorship_bias(0.08, 0.20, liquidation_cutoff_c=0.05)
    assert harsh.annual_survival_prob < mild.annual_survival_prob
    assert harsh.survivorship_bias_bps > mild.survivorship_bias_bps
    assert harsh.observed_survivor_vol < mild.observed_survivor_vol


def test_bgir_regime_classification_spectrum() -> None:
    """Verify truncation distortion regimes from untruncated baseline to severe mirage."""
    untrunc = calculate_survivorship_bias(0.10, 0.15, liquidation_cutoff_c=-0.65)
    mild = calculate_survivorship_bias(0.08, 0.18, liquidation_cutoff_c=-0.30)
    mod = calculate_survivorship_bias(0.08, 0.20, liquidation_cutoff_c=-0.21)
    sev = calculate_survivorship_bias(0.05, 0.22, liquidation_cutoff_c=-0.02)

    assert untrunc.regime == "UNTRUNCATED_UNIVERSE_BASELINE"
    assert mild.regime == "MILD_ATTRITION_DRIFT"
    assert mod.regime == "MODERATE_DELISTING_BIAS"
    assert sev.regime == "SEVERE_TRUNCATION_MIRAGE"


def test_bgir_input_validation_guards() -> None:
    """Enforce ticker allowlist, volatility positivity, horizon bounds, and tail z-score guard."""
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_survivorship_bias(0.08, 0.20, ticker="HFRI; DROP")
    with pytest.raises(ValueError, match="True universe volatility"):
        calculate_survivorship_bias(0.08, 0.0)
    with pytest.raises(ValueError, match="Attrition horizon"):
        calculate_survivorship_bias(0.08, 0.20, horizon_years=0.5)
    with pytest.raises(ValueError, match="Standardized truncation z-score"):
        calculate_survivorship_bias(0.08, 0.05, liquidation_cutoff_c=1.0)


def test_bgir_hmac_integrity_and_tamper_detection() -> None:
    """Validate HMAC-SHA256 seal integrity and tamper sensitivity."""
    rec = calculate_survivorship_bias(0.08, 0.20, ticker="QQQ")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        mu_true=rec.true_universe_mean,
        vol_true=rec.true_universe_vol,
        cutoff_c=rec.liquidation_cutoff_c,
        horizon=rec.horizon_years,
        surv_prob=rec.annual_survival_prob,
        bias_bps=rec.survivorship_bias_bps,
        obs_vol=rec.observed_survivor_vol,
        sharpe_delta=rec.sharpe_bias_delta,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("QQQ", "SPY"), rec.hmac_signature) is False


def test_bgir_empirical_profile_calibration() -> None:
    """Calibrate trailing session mean and volatility into BGIR truncation model."""
    rng = np.random.default_rng(178)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.014, size=60)))
    rec = evaluate_survivorship_profile(prices, ticker="SPY")
    assert rec.ticker == "SPY"
    assert 0.0 < rec.annual_survival_prob < 1.0


def test_bgir_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite persistence into brown_goetzmann_survivorship_audits."""
    db_file = tmp_path / "bgir_audit.db"
    rec = calculate_survivorship_bias(0.08, 0.20, ticker="IWM")
    assert persist_brown_goetzmann_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, annual_survival_prob, survivorship_bias_bps, sharpe_bias_delta, regime "
            "FROM brown_goetzmann_survivorship_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "IWM"
    assert rows[0][2] == pytest.approx(rec.survivorship_bias_bps, rel=1e-9)


def test_bgir_cli_integration_json() -> None:
    """Execute CLI subcommand `survivorship-bias --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "survivorship-bias",
            "--ticker",
            "HFRI",
            "--mu",
            "0.08",
            "--vol",
            "0.20",
            "--cutoff",
            "-0.10",
            "--horizon",
            "5.0",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "HFRI"
    assert payload["observed_survivor_mean"] > payload["true_universe_mean"]
    assert payload["observed_survivor_vol"] < payload["true_universe_vol"]
