"""Tests for Macaulay (1938) & Malkiel (1962) Bond Duration & Convexity Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.malkiel_bond_duration import (
    _build_hmac_payload,
    calculate_malkiel_bond_duration,
    evaluate_malkiel_duration_profile,
    persist_malkiel_duration_record,
    verify_hmac_signature,
)


def test_macaulay_zero_coupon_duration_equals_maturity() -> None:
    """Zero-coupon bond Macaulay duration must equal exact maturity T."""
    rec = calculate_malkiel_bond_duration(
        coupon_rate=0.0,
        yield_to_maturity=0.05,
        maturity_years=10.0,
        compounding_freq=2,
        ticker="ZROZ",
    )
    assert rec.macaulay_duration == pytest.approx(10.0, rel=1e-12)
    assert rec.modified_duration == pytest.approx(10.0 / 1.025, rel=1e-12)
    assert rec.par_status == "DISCOUNT"


def test_par_bond_valuation_and_duration_coupon_monotonicity() -> None:
    """When coupon == YTM, price == par; higher coupon lowers Macaulay duration (Malkiel)."""
    par_bond = calculate_malkiel_bond_duration(
        coupon_rate=0.05,
        yield_to_maturity=0.05,
        maturity_years=10.0,
        compounding_freq=2,
    )
    high_cpn = calculate_malkiel_bond_duration(
        coupon_rate=0.09,
        yield_to_maturity=0.05,
        maturity_years=10.0,
        compounding_freq=2,
    )
    assert par_bond.bond_price == pytest.approx(100.0, rel=1e-12)
    assert par_bond.par_status == "AT_PAR"
    assert high_cpn.par_status == "PREMIUM"
    assert high_cpn.macaulay_duration < par_bond.macaulay_duration


def test_malkiel_convexity_asymmetry_and_taylor_accuracy() -> None:
    """Positive convexity causes upside gain on -dy to exceed downside loss on +dy."""
    rec = calculate_malkiel_bond_duration(
        coupon_rate=0.04,
        yield_to_maturity=0.05,
        maturity_years=20.0,
        compounding_freq=2,
        shock_bps=100.0,
    )
    assert rec.convexity > 0.0
    assert rec.exact_gain_on_cut > rec.exact_loss_on_hike
    assert rec.malkiel_convexity_asymmetry > 0.0
    assert abs(rec.taylor_on_yield_cut - rec.repriced_on_yield_cut) < 0.25
    assert abs(rec.taylor_on_yield_hike - rec.repriced_on_yield_hike) < 0.25


def test_malkiel_regime_classification_spectrum() -> None:
    """Verify duration regimes from short defensive and high-coupon carry to extended bond."""
    ext = calculate_malkiel_bond_duration(0.03, 0.04, maturity_years=20.0)
    core = calculate_malkiel_bond_duration(0.045, 0.045, maturity_years=7.0)
    carry = calculate_malkiel_bond_duration(0.08, 0.05, maturity_years=4.0)
    short = calculate_malkiel_bond_duration(0.02, 0.04, maturity_years=2.0)

    assert ext.regime == "EXTENDED_DURATION_CONVEX_BOND"
    assert core.regime == "CORE_INVESTMENT_DURATION"
    assert carry.regime == "HIGH_COUPON_SHORT_CARRY"
    assert short.regime == "SHORT_DURATION_DEFENSIVE"


def test_malkiel_input_validation_guards() -> None:
    """Enforce ticker regex, allowed frequencies, integer period alignment, and shock bounds."""
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_malkiel_bond_duration(0.04, 0.05, ticker="TLT; DROP")
    with pytest.raises(ValueError, match="Compounding frequency"):
        calculate_malkiel_bond_duration(0.04, 0.05, compounding_freq=3)
    with pytest.raises(ValueError, match="integer number of periods"):
        calculate_malkiel_bond_duration(0.04, 0.05, maturity_years=5.3, compounding_freq=2)
    with pytest.raises(ValueError, match="Parallel yield shock"):
        calculate_malkiel_bond_duration(0.04, 0.05, shock_bps=0.0)


def test_malkiel_hmac_integrity_and_tamper_detection() -> None:
    """Validate HMAC-SHA256 seal integrity and tamper sensitivity."""
    rec = calculate_malkiel_bond_duration(0.045, 0.05, maturity_years=10.0, ticker="IEF")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        par=rec.par_value,
        coupon=rec.coupon_rate,
        ytm=rec.yield_to_maturity,
        maturity=rec.maturity_years,
        freq=rec.compounding_freq,
        price=rec.bond_price,
        mac_dur=rec.macaulay_duration,
        mod_dur=rec.modified_duration,
        convexity=rec.convexity,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("IEF", "SHY"), rec.hmac_signature) is False


def test_malkiel_empirical_profile_calibration() -> None:
    """Calibrate effective yield and parallel shock stress from trailing close prices."""
    rng = np.random.default_rng(177)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.011, size=60)))
    rec = evaluate_malkiel_duration_profile(prices, ticker="TLT", maturity_years=10.0)
    assert rec.ticker == "TLT"
    assert rec.malkiel_convexity_asymmetry > 0.0


def test_malkiel_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite audit persistence into malkiel_bond_duration_audits."""
    db_file = tmp_path / "mbd_audit.db"
    rec = calculate_malkiel_bond_duration(0.04, 0.05, maturity_years=10.0, ticker="AGG")
    assert persist_malkiel_duration_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, bond_price, macaulay_duration, convexity, regime "
            "FROM malkiel_bond_duration_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "AGG"
    assert rows[0][2] == pytest.approx(rec.macaulay_duration, rel=1e-9)


def test_malkiel_cli_integration_json() -> None:
    """Execute CLI subcommand `bond-duration --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "bond-duration",
            "--ticker",
            "TLT",
            "--coupon",
            "0.04",
            "--ytm",
            "0.05",
            "--maturity",
            "10.0",
            "--freq",
            "2",
            "--shock-bps",
            "100.0",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "TLT"
    assert payload["convexity"] > 0.0
    assert payload["malkiel_convexity_asymmetry"] > 0.0

