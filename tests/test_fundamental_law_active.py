"""Tests for Grinold (1989) & Clarke, de Silva & Thorley (2002) Fundamental Law."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.fundamental_law_active import (
    _build_hmac_payload,
    calculate_fundamental_law,
    compute_transfer_coefficient,
    evaluate_fundamental_law_profile,
    persist_fundamental_law_record,
    verify_hmac_signature,
)


def test_fundamental_law_ir_and_effective_breadth_identity() -> None:
    """Verify IR_constrained = TC * IC * sqrt(BR) == IC * sqrt(TC^2 * BR)."""
    rec = calculate_fundamental_law(
        information_coefficient_ic=0.06,
        breadth_br=400.0,
        transfer_coefficient_tc=0.50,
        tracking_error_te=0.04,
        ticker="SPY",
    )
    assert rec.unconstrained_ir == pytest.approx(1.20, rel=1e-12)
    assert rec.constrained_ir == pytest.approx(0.60, rel=1e-12)
    assert rec.effective_breadth == pytest.approx(100.0, rel=1e-12)
    assert rec.breadth_loss_pct == pytest.approx(0.75, rel=1e-12)
    assert rec.information_coefficient_ic * math.sqrt(rec.effective_breadth) == pytest.approx(
        rec.constrained_ir, rel=1e-12
    )


def test_fundamental_law_alpha_and_constraint_drag() -> None:
    """Verify alpha_active = IR_constrained * TE and drag identity."""
    rec = calculate_fundamental_law(
        information_coefficient_ic=0.08,
        breadth_br=256.0,
        transfer_coefficient_tc=0.75,
        tracking_error_te=0.05,
    )
    assert rec.unconstrained_alpha == pytest.approx(0.08 * 16.0 * 0.05, rel=1e-12)
    assert rec.constrained_alpha == pytest.approx(0.75 * 0.08 * 16.0 * 0.05, rel=1e-12)
    assert rec.constraint_alpha_drag == pytest.approx(
        rec.unconstrained_alpha - rec.constrained_alpha, rel=1e-12
    )


def test_transfer_coefficient_vector_and_covariance_alignment() -> None:
    """Verify Clarke-de Silva-Thorley transfer correlation matches vector alignment."""
    w_unconstrained = np.array([0.05, -0.04, 0.03, -0.04])
    w_identical = np.array([0.025, -0.02, 0.015, -0.02])
    w_long_only = np.array([0.05, 0.00, 0.03, 0.00])

    assert compute_transfer_coefficient(w_unconstrained, w_identical) == pytest.approx(
        1.0, rel=1e-12
    )
    tc_lo = compute_transfer_coefficient(w_unconstrained, w_long_only)
    assert 0.0 < tc_lo < 1.0

    cov = np.diag([0.04, 0.09, 0.04, 0.09])
    tc_cov = compute_transfer_coefficient(w_unconstrained, w_long_only, covariance=cov)
    assert 0.0 < tc_cov < 1.0


def test_fundamental_law_regime_classification_spectrum() -> None:
    """Verify all four active management skill/constraint regimes."""
    elite = calculate_fundamental_law(0.08, breadth_br=250.0, transfer_coefficient_tc=0.80)
    core = calculate_fundamental_law(0.04, breadth_br=200.0, transfer_coefficient_tc=0.65)
    starved = calculate_fundamental_law(0.08, breadth_br=250.0, transfer_coefficient_tc=0.35)
    churner = calculate_fundamental_law(-0.02, breadth_br=250.0, transfer_coefficient_tc=0.80)

    assert elite.regime == "ELITE_TRANSFER_ALPHA_ENGINE"
    assert core.regime == "BREADTH_SCALED_CORE_STRATEGY"
    assert starved.regime == "CONSTRAINT_STARVED_LONG_ONLY"
    assert churner.regime == "ZERO_SKILL_NOISE_CHURNER"


def test_fundamental_law_input_validation_guards() -> None:
    """Block invalid tickers, out-of-range IC/TC/BR/TE, and degenerate weight arrays."""
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_fundamental_law(0.05, ticker="SPY; DROP")
    with pytest.raises(ValueError, match="Information Coefficient"):
        calculate_fundamental_law(1.20)
    with pytest.raises(ValueError, match="Decision breadth"):
        calculate_fundamental_law(0.05, breadth_br=0.5)
    with pytest.raises(ValueError, match="Transfer Coefficient"):
        calculate_fundamental_law(0.05, transfer_coefficient_tc=1.05)
    with pytest.raises(ValueError, match="Tracking error"):
        calculate_fundamental_law(0.05, tracking_error_te=0.0)
    with pytest.raises(ValueError, match="Active weight variance"):
        compute_transfer_coefficient([0.0, 0.0], [0.1, -0.1])


def test_fundamental_law_hmac_integrity_and_tamper_detection() -> None:
    """Validate HMAC-SHA256 seal integrity and tamper sensitivity."""
    rec = calculate_fundamental_law(0.07, breadth_br=300.0, transfer_coefficient_tc=0.75, ticker="QQQ")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        ic=rec.information_coefficient_ic,
        breadth=rec.breadth_br,
        tc=rec.transfer_coefficient_tc,
        te=rec.tracking_error_te,
        constrained_ir=rec.constrained_ir,
        effective_breadth=rec.effective_breadth,
        constrained_alpha=rec.constrained_alpha,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("QQQ", "IWM"), rec.hmac_signature) is False


def test_fundamental_law_empirical_profile_calibration() -> None:
    """Calibrate trailing signal persistence and tracking error proxy from prices."""
    rng = np.random.default_rng(174)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.013, size=60)))
    rec = evaluate_fundamental_law_profile(prices, ticker="SPY", breadth_br=250.0)
    assert rec.ticker == "SPY"
    assert 0.01 <= rec.tracking_error_te <= 0.25


def test_fundamental_law_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite audit persistence into fundamental_law_audits."""
    db_file = tmp_path / "fl_audit.db"
    rec = calculate_fundamental_law(0.06, breadth_br=200.0, ticker="IWM")
    assert persist_fundamental_law_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, constrained_ir, effective_breadth, constrained_alpha, regime "
            "FROM fundamental_law_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "IWM"
    assert rows[0][1] == pytest.approx(rec.constrained_ir, rel=1e-9)


def test_fundamental_law_cli_integration_json() -> None:
    """Execute CLI subcommand `fundamental-law --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "fundamental-law",
            "--ticker",
            "SPY",
            "--ic",
            "0.06",
            "--breadth",
            "400",
            "--tc",
            "0.50",
            "--te",
            "0.04",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "SPY"
    assert payload["unconstrained_ir"] == pytest.approx(1.20, rel=1e-9)
    assert payload["constrained_ir"] == pytest.approx(0.60, rel=1e-9)
    assert payload["effective_breadth"] == pytest.approx(100.0, rel=1e-9)

