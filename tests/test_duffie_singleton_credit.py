"""Tests for Duffie & Singleton (1999) Reduced-Form Credit Spread & Hazard Engine."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.duffie_singleton_credit import (
    _build_hmac_payload,
    calculate_duffie_singleton_credit,
    evaluate_duffie_singleton_profile,
    persist_duffie_singleton_record,
    verify_hmac_signature,
)


def test_duffie_singleton_rmv_discount_and_credit_triangle() -> None:
    """Verify R = r + h*(1 - omega) and v_RMV = N*exp(-R*T)."""
    rec = calculate_duffie_singleton_credit(
        hazard_rate_h=0.04,
        recovery_rate_omega=0.40,
        risk_free_rate_r=0.05,
        maturity_years=5.0,
        notional=100.0,
        ticker="HYG",
    )
    assert rec.loss_given_default_lgd == pytest.approx(0.60, rel=1e-12)
    assert rec.credit_spread_s == pytest.approx(0.024, rel=1e-12)
    assert rec.credit_spread_bps == pytest.approx(240.0, rel=1e-12)
    assert rec.adjusted_short_rate == pytest.approx(0.074, rel=1e-12)
    assert rec.rmv_bond_price == pytest.approx(100.0 * math.exp(-0.074 * 5.0), rel=1e-12)
    assert rec.implied_hazard_check == pytest.approx(0.04, rel=1e-12)


def test_duffie_singleton_rmv_vs_rt_convexity_bound() -> None:
    """Under positive recovery omega in (0, 1), v_RMV(T) < v_RT(T) strictly."""
    rec = calculate_duffie_singleton_credit(
        hazard_rate_h=0.05,
        recovery_rate_omega=0.45,
        risk_free_rate_r=0.04,
        maturity_years=10.0,
    )
    assert rec.rmv_bond_price < rec.rt_bond_price < rec.risk_free_bond_price
    assert rec.rmv_rt_convexity_gap > 0.0
    assert rec.cva_discount == pytest.approx(rec.risk_free_bond_price - rec.rmv_bond_price)


def test_duffie_singleton_zero_recovery_equivalence() -> None:
    """At zero recovery (omega == 0.0), RMV and RT bond valuations coincide identically."""
    rec = calculate_duffie_singleton_credit(
        hazard_rate_h=0.06,
        recovery_rate_omega=0.0,
        risk_free_rate_r=0.04,
        maturity_years=5.0,
    )
    assert rec.rmv_bond_price == pytest.approx(rec.rt_bond_price, rel=1e-12)
    assert rec.rmv_rt_convexity_gap == pytest.approx(0.0, abs=1e-12)


def test_duffie_singleton_survival_and_default_probability_sum() -> None:
    """Verify survival probability S(T) = exp(-h*T) and S(T) + PD(T) == 1."""
    rec = calculate_duffie_singleton_credit(hazard_rate_h=0.03, maturity_years=7.0)
    assert rec.survival_probability == pytest.approx(math.exp(-0.21), rel=1e-12)
    assert rec.survival_probability + rec.default_probability == pytest.approx(1.0, rel=1e-12)


def test_duffie_singleton_regime_classification_spectrum() -> None:
    """Verify credit quality tiers from Investment Grade to Distressed Jump-to-Default."""
    ig = calculate_duffie_singleton_credit(hazard_rate_h=0.01, recovery_rate_omega=0.40)
    xo = calculate_duffie_singleton_credit(hazard_rate_h=0.025, recovery_rate_omega=0.40)
    hy = calculate_duffie_singleton_credit(hazard_rate_h=0.055, recovery_rate_omega=0.40)
    dist = calculate_duffie_singleton_credit(hazard_rate_h=0.10, recovery_rate_omega=0.40)
    assert ig.regime == "INVESTMENT_GRADE_LOW_HAZARD"
    assert xo.regime == "CROSSOVER_SPREAD_WATCH"
    assert hy.regime == "HIGH_YIELD_INTENSITY_BUILDUP"
    assert dist.regime == "DISTRESSED_JUMP_TO_DEFAULT"


def test_duffie_singleton_input_validation_guards() -> None:
    """Enforce strict allowlists on ticker, hazard rate, recovery, maturity, and notional."""
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_duffie_singleton_credit(0.02, ticker="HYG; DROP")
    with pytest.raises(ValueError, match="Hazard intensity"):
        calculate_duffie_singleton_credit(0.0)
    with pytest.raises(ValueError, match="Recovery rate"):
        calculate_duffie_singleton_credit(0.02, recovery_rate_omega=1.0)
    with pytest.raises(ValueError, match="Maturity"):
        calculate_duffie_singleton_credit(0.02, maturity_years=-1.0)
    with pytest.raises(ValueError, match="Notional"):
        calculate_duffie_singleton_credit(0.02, notional=0.0)


def test_duffie_singleton_hmac_integrity_and_tamper_detection() -> None:
    """Verify HMAC-SHA256 seal integrity and tamper sensitivity."""
    rec = calculate_duffie_singleton_credit(0.03, ticker="LQD")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        hazard_h=rec.hazard_rate_h,
        recovery_omega=rec.recovery_rate_omega,
        rate_r=rec.risk_free_rate_r,
        maturity=rec.maturity_years,
        spread_s=rec.credit_spread_s,
        rmv_price=rec.rmv_bond_price,
        survival_prob=rec.survival_probability,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("LQD", "JNK"), rec.hmac_signature) is False


def test_duffie_singleton_empirical_profile_calibration() -> None:
    """Calibrate implied default hazard intensity from trailing asset volatility."""
    rng = np.random.default_rng(173)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, size=60)))
    rec = evaluate_duffie_singleton_profile(prices, ticker="HYG", maturity_years=5.0)
    assert rec.ticker == "HYG"
    assert rec.hazard_rate_h > 0.002
    assert rec.rmv_bond_price < rec.risk_free_bond_price


def test_duffie_singleton_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite audit persistence into duffie_singleton_credit_audits."""
    db_file = tmp_path / "ds_audit.db"
    rec = calculate_duffie_singleton_credit(0.035, ticker="JNK")
    assert persist_duffie_singleton_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, hazard_rate_h, credit_spread_bps, rmv_bond_price, regime "
            "FROM duffie_singleton_credit_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "JNK"
    assert rows[0][3] == pytest.approx(rec.rmv_bond_price, rel=1e-9)


def test_duffie_singleton_cli_integration_json() -> None:
    """Execute CLI subcommand `reduced-credit --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "reduced-credit",
            "--ticker",
            "HYG",
            "--hazard",
            "0.04",
            "--recovery",
            "0.40",
            "--rate",
            "0.05",
            "--maturity",
            "5.0",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "HYG"
    assert payload["credit_spread_bps"] == pytest.approx(240.0, rel=1e-9)
    assert payload["rmv_bond_price"] < payload["rt_bond_price"]

