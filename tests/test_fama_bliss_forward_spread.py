"""Tests for Fama & Bliss (1987) Forward-Spot Term Spread & Bond Excess Return Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.fama_bliss_forward_spread import (
    _build_hmac_payload,
    calculate_fama_bliss_forward_spread,
    estimate_fama_bliss_ols,
    evaluate_fama_bliss_profile,
    persist_fama_bliss_record,
    verify_hmac_signature,
)


def test_fama_bliss_forward_rate_and_spread_identity() -> None:
    """Verify f_t^(n) = n*y_t^(n) - (n-1)*y_t^(n-1) and spread s = f - y1."""
    rec = calculate_fama_bliss_forward_spread(
        spot_y1=0.04,
        prev_maturity_y_nm1=0.05,
        long_yn=0.055,
        maturity_n=5,
        next_y_nm1=0.052,
        ticker="TLT",
    )
    expected_fwd = 5.0 * 0.055 - 4.0 * 0.05
    expected_spread = expected_fwd - 0.04
    assert rec.implied_forward_fn == pytest.approx(expected_fwd, rel=1e-12)
    assert rec.forward_spot_spread == pytest.approx(expected_spread, rel=1e-12)
    assert rec.forward_spread_bps == pytest.approx(expected_spread * 10000.0, rel=1e-12)


def test_fama_bliss_exact_accounting_complementarity_identity() -> None:
    """Verify rx + (n-1)*(y_{t+1}^(n-1) - y_t^(n-1)) == f_t^(n) - y_t^(1) for any n."""
    for n, y1, y_nm1, yn, y_next in [
        (2, 0.040, 0.040, 0.048, 0.045),
        (5, 0.035, 0.042, 0.046, 0.038),
        (10, 0.030, 0.044, 0.047, 0.051),
    ]:
        rec = calculate_fama_bliss_forward_spread(
            spot_y1=y1,
            prev_maturity_y_nm1=y_nm1,
            long_yn=yn,
            maturity_n=n,
            next_y_nm1=y_next,
        )
        assert rec.excess_bond_return_rx + rec.scaled_yield_change == pytest.approx(
            rec.forward_spot_spread, abs=1e-12
        )
        assert rec.identity_residual == pytest.approx(0.0, abs=1e-12)


def test_fama_bliss_random_walk_realized_carry_equals_spread() -> None:
    """Under unchanged yields (next_y_nm1 == y_nm1), 1y excess return equals full spread."""
    rec = calculate_fama_bliss_forward_spread(
        spot_y1=0.040,
        long_yn=0.052,
        maturity_n=2,
    )
    assert rec.scaled_yield_change == pytest.approx(0.0, abs=1e-12)
    assert rec.excess_bond_return_rx == pytest.approx(rec.forward_spot_spread, rel=1e-12)
    assert rec.regime == "EXTENDED_TERM_PREMIUM_CARRY"


def test_fama_bliss_expectations_hypothesis_breakeven_shift() -> None:
    """When spot yield rises by exact breakeven spread/(n-1), excess return is zero."""
    rec_base = calculate_fama_bliss_forward_spread(spot_y1=0.04, long_yn=0.05, maturity_n=2)
    breakeven_next = 0.04 + rec_base.eh_breakeven_yield_shift
    rec_zero = calculate_fama_bliss_forward_spread(
        spot_y1=0.04,
        long_yn=0.05,
        maturity_n=2,
        next_y_nm1=breakeven_next,
    )
    assert rec_zero.excess_bond_return_rx == pytest.approx(0.0, abs=1e-12)


def test_fama_bliss_ols_complementarity_theorem() -> None:
    """Verify paired OLS regressions obey exact beta + c == 1.0 and alpha + a == 0.0."""
    rng = np.random.default_rng(172)
    y1 = 0.04 + 0.015 * np.sin(np.linspace(0, 6.0, 40)) + rng.normal(0.0, 0.003, 40)
    y2 = y1 + 0.008 + rng.normal(0.0, 0.004, 40)
    y1_next = np.roll(y1, -1)
    res = estimate_fama_bliss_ols(
        spot_y1_series=y1[:-1],
        long_yn_series=y2[:-1],
        next_y_nm1_series=y1_next[:-1],
        maturity_n=2,
    )
    assert res.slope_sum_check == pytest.approx(1.0, abs=1e-12)
    assert res.intercept_sum_check == pytest.approx(0.0, abs=1e-12)
    assert res.n_obs == 39


def test_fama_bliss_regime_classification_spectrum() -> None:
    """Verify inverted, neutral, balanced, and extended carry regimes."""
    inv = calculate_fama_bliss_forward_spread(spot_y1=0.055, long_yn=0.045, maturity_n=2)
    neu = calculate_fama_bliss_forward_spread(spot_y1=0.040, long_yn=0.041, maturity_n=2)
    bal = calculate_fama_bliss_forward_spread(spot_y1=0.040, long_yn=0.044, maturity_n=2)
    ext = calculate_fama_bliss_forward_spread(spot_y1=0.035, long_yn=0.048, maturity_n=2)
    assert inv.regime == "INVERTED_CARRY_COMPRESSION"
    assert neu.regime == "EXPECTATIONS_NEUTRAL_CURVE"
    assert bal.regime == "BALANCED_FORWARD_SLOPE"
    assert ext.regime == "EXTENDED_TERM_PREMIUM_CARRY"


def test_fama_bliss_input_validation_and_guards() -> None:
    """Validate ticker regex, maturity range, yield bounds, and zero-variance OLS guard."""
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_fama_bliss_forward_spread(0.04, 0.05, ticker="TLT; DROP")
    with pytest.raises(ValueError, match="Maturity n"):
        calculate_fama_bliss_forward_spread(0.04, 0.05, maturity_n=1)
    with pytest.raises(ValueError, match="Spot yields"):
        calculate_fama_bliss_forward_spread(0.55, 0.05)
    with pytest.raises(ValueError, match="When maturity n == 2"):
        calculate_fama_bliss_forward_spread(0.04, 0.05, maturity_n=2, prev_maturity_y_nm1=0.045)
    with pytest.raises(ValueError, match="insufficient variance"):
        estimate_fama_bliss_ols([0.04] * 10, [0.05] * 10, [0.04] * 10, maturity_n=2)


def test_fama_bliss_hmac_integrity_and_tamper_detection() -> None:
    """Verify HMAC-SHA256 seal verification and tamper sensitivity."""
    rec = calculate_fama_bliss_forward_spread(0.042, 0.051, maturity_n=5, ticker="IEF")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        maturity_n=rec.maturity_n,
        spot_y1=rec.spot_y1,
        long_yn=rec.long_yn,
        implied_forward_fn=rec.implied_forward_fn,
        forward_spot_spread=rec.forward_spot_spread,
        excess_bond_return_rx=rec.excess_bond_return_rx,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("IEF", "SHY"), rec.hmac_signature) is False


def test_fama_bliss_empirical_profile_calibration() -> None:
    """Calibrate forward term slope proxy from trailing equity/bond prices."""
    rng = np.random.default_rng(172)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, size=60)))
    rec = evaluate_fama_bliss_profile(prices, ticker="TLT", spot_y1=0.04, maturity_n=5)
    assert rec.ticker == "TLT"
    assert rec.forward_spot_spread > 0.0


def test_fama_bliss_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite audit persistence into fama_bliss_audits."""
    db_file = tmp_path / "fb_audit.db"
    rec = calculate_fama_bliss_forward_spread(0.04, 0.05, maturity_n=5, ticker="SHY")
    assert persist_fama_bliss_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, maturity_n, forward_spot_spread, excess_bond_return_rx, regime "
            "FROM fama_bliss_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SHY"
    assert rows[0][2] == pytest.approx(rec.forward_spot_spread, rel=1e-9)


def test_fama_bliss_cli_integration_json() -> None:
    """Execute CLI subcommand `fama-bliss --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "fama-bliss",
            "--ticker",
            "TLT",
            "--y1",
            "0.040",
            "--yn",
            "0.052",
            "--maturity",
            "2",
            "--next-y1",
            "0.043",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "TLT"
    assert payload["excess_bond_return_rx"] + payload["scaled_yield_change"] == pytest.approx(
        payload["forward_spot_spread"], abs=1e-10
    )

