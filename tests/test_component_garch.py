"""Tests for Robert F. Engle & Gary G.J. Lee (1999) Component GARCH (C-GARCH) Two-Scale Volatility Engine."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.component_garch import (
    _build_hmac_payload,
    calculate_component_garch,
    calculate_component_garch_trajectory,
    evaluate_component_garch_profile,
    persist_component_garch_record,
    verify_hmac_signature,
)


def test_two_scale_half_life_hierarchy() -> None:
    """Component GARCH correctly decomposes variance into permanent and transitory scales."""
    np.random.seed(42)
    # Simulated daily returns with typical variance
    returns = np.random.normal(0.0004, 0.012, size=150)

    rec = calculate_component_garch(
        returns,
        omega=0.0001,
        rho=0.98,
        phi=0.04,
        alpha=0.06,
        beta=0.70,
        ticker="SPY_CGARCH",
    )
    # Permanent half life = ln(0.5)/ln(0.98) ~ 34.3 sessions
    # Transitory half life = ln(0.5)/ln(0.76) ~ 2.5 sessions
    assert rec.permanent_half_life > rec.transitory_half_life
    assert rec.permanent_half_life == pytest.approx(34.3, rel=0.05)
    assert rec.transitory_half_life == pytest.approx(2.52, rel=0.05)
    assert rec.latest_permanent_vol > 0.0
    assert rec.latest_total_vol > 0.0
    assert math.isfinite(rec.log_likelihood)
    assert math.isfinite(rec.aic)


def test_transitory_shock_spike_and_hierarchy_guard() -> None:
    """An acute return shock spikes transitory variance; parameter hierarchy violation is guarded."""
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.010, size=100)
    # Inject an acute crash shock on the penultimate session driving final variance
    returns[-2] = -0.065

    rec = calculate_component_garch(
        returns,
        omega=0.0001,
        rho=0.97,
        phi=0.05,
        alpha=0.10,
        beta=0.65,
        ticker="SPY_SHOCK",
    )
    # Total volatility spikes well above the permanent baseline
    assert rec.latest_total_vol > rec.latest_permanent_vol
    assert rec.regime == "ACUTE_TRANSITORY_SHOCK_SPIKE"

    # Hierarchy violation: alpha + beta >= rho
    with pytest.raises(ValueError, match="Hierarchy violation"):
        calculate_component_garch(
            returns,
            omega=0.0001,
            rho=0.80,
            phi=0.05,
            alpha=0.15,
            beta=0.75,  # 0.15 + 0.75 = 0.90 > 0.80
            ticker="SPY_BAD",
        )


def test_cgarch_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate parameter boundaries, HMAC audit sealing, and SQLite logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_component_garch([0.01] * 30, ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="Omega"):
        calculate_component_garch([0.01] * 30, omega=-0.001, ticker="SPY")

    with pytest.raises(ValueError, match="Rho"):
        calculate_component_garch([0.01] * 30, rho=1.2, ticker="SPY")

    # Evaluate profile from simulated price series
    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0006)))
    receipt = evaluate_component_garch_profile(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        sample_size=receipt.sample_size,
        omega=receipt.omega,
        rho=receipt.rho,
        alpha=receipt.alpha,
        beta=receipt.beta,
        latest_total_vol=receipt.latest_total_vol,
        latest_permanent_vol=receipt.latest_permanent_vol,
        permanent_variance_ratio=receipt.permanent_variance_ratio,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "cgarch_test.db"
    assert persist_component_garch_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, sample_size, latest_total_vol, regime, hmac_signature FROM component_garch_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == receipt.sample_size
    assert rows[0][4] == receipt.hmac_signature


def test_cli_cgarch_json_and_aliases() -> None:
    """Verify CLI cgarch and aliases emit valid JSON receipt."""
    for alias in ("cgarch", "c-garch", "component-garch", "engle-lee"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--omega",
                "0.0001",
                "--rho",
                "0.98",
                "--phi",
                "0.04",
                "--alpha",
                "0.06",
                "--beta",
                "0.70",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["latest_total_vol"] > 0.0
        assert len(data["hmac_signature"]) == 64
