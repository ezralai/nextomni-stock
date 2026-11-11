"""Tests for Merton & Perold (1993) Risk Capital & Institutional Allocation Engine."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.merton_perold_risk_capital import (
    _build_hmac_payload,
    calculate_merton_perold_risk_capital,
    evaluate_merton_perold_profile,
    persist_merton_perold_record,
    verify_hmac_signature,
)


def test_merton_perold_standalone_put_guarantee_identity() -> None:
    """Verify exact closed-form guarantee formula V * [2*N(0.5*sigma*sqrt(T)) - 1]."""
    rec = calculate_merton_perold_risk_capital(
        unit1_asset_v1=100.0,
        unit1_volatility=0.30,
        unit2_asset_v2=100.0,
        unit2_volatility=0.30,
        cross_unit_correlation=0.50,
        horizon_years=1.0,
        ticker="JPM",
    )
    expected_rc1 = 100.0 * math.erf(0.5 * 0.30 / math.sqrt(2.0))
    assert rec.standalone_rc_1 == pytest.approx(expected_rc1, rel=1e-12)
    assert rec.standalone_rc_2 == pytest.approx(expected_rc1, rel=1e-12)


def test_merton_perold_subadditivity_and_marginal_underallocation_theorem() -> None:
    """For rho < 1, RC_P < RC_1 + RC_2 and unallocated gap equals diversification benefit."""
    rec = calculate_merton_perold_risk_capital(
        unit1_asset_v1=120.0,
        unit1_volatility=0.35,
        unit2_asset_v2=80.0,
        unit2_volatility=0.25,
        cross_unit_correlation=0.20,
        horizon_years=1.0,
    )
    assert rec.portfolio_rc < rec.sum_standalone_rc
    assert rec.marginal_rc_1 < rec.standalone_rc_1
    assert rec.marginal_rc_2 < rec.standalone_rc_2
    assert rec.diversification_benefit > 0.0
    assert rec.unallocated_capital_gap == pytest.approx(rec.diversification_benefit, abs=1e-10)
    assert rec.marginal_rc_1 + rec.marginal_rc_2 < rec.portfolio_rc


def test_merton_perold_euler_homogeneous_full_allocation() -> None:
    """Euler homogeneous marginal contributions must sum exactly to 100% of firm RC_P."""
    rec = calculate_merton_perold_risk_capital(
        unit1_asset_v1=150.0,
        unit1_volatility=0.40,
        unit2_asset_v2=90.0,
        unit2_volatility=0.22,
        cross_unit_correlation=0.15,
        horizon_years=2.0,
    )
    assert rec.euler_rc_1 + rec.euler_rc_2 == pytest.approx(rec.portfolio_rc, rel=1e-12)


def test_merton_perold_perfect_correlation_limit_and_concentrated_regime() -> None:
    """At rho == 1.0, diversification benefit collapses to zero under homogeneous vols."""
    rec = calculate_merton_perold_risk_capital(
        unit1_asset_v1=100.0,
        unit1_volatility=0.25,
        unit2_asset_v2=100.0,
        unit2_volatility=0.25,
        cross_unit_correlation=1.0,
        horizon_years=1.0,
    )
    assert rec.diversification_benefit == pytest.approx(0.0, abs=1e-10)
    assert rec.euler_rc_1 == pytest.approx(rec.standalone_rc_1, rel=1e-10)
    assert rec.regime == "CONCENTRATED_CORRELATION_DRAG"


def test_merton_perold_degree_one_homogeneity_scaling() -> None:
    """Scaling V1 and V2 by lambda > 0 must scale all capital figures by exact lambda."""
    base = calculate_merton_perold_risk_capital(
        unit1_asset_v1=100.0,
        unit1_volatility=0.28,
        unit2_asset_v2=50.0,
        unit2_volatility=0.20,
        cross_unit_correlation=0.30,
        horizon_years=1.5,
    )
    scaled = calculate_merton_perold_risk_capital(
        unit1_asset_v1=350.0,
        unit1_volatility=0.28,
        unit2_asset_v2=175.0,
        unit2_volatility=0.20,
        cross_unit_correlation=0.30,
        horizon_years=1.5,
    )
    assert scaled.portfolio_volatility == pytest.approx(base.portfolio_volatility, rel=1e-12)
    assert scaled.portfolio_rc == pytest.approx(3.5 * base.portfolio_rc, rel=1e-12)
    assert scaled.euler_rc_1 == pytest.approx(3.5 * base.euler_rc_1, rel=1e-12)


def test_merton_perold_diversification_surplus_raroc_boost() -> None:
    """Negative/low correlation expands RAROC above standalone RAROC into accumulator regime."""
    rec = calculate_merton_perold_risk_capital(
        unit1_asset_v1=100.0,
        unit1_volatility=0.30,
        unit2_asset_v2=100.0,
        unit2_volatility=0.30,
        cross_unit_correlation=-0.30,
        horizon_years=1.0,
        expected_excess_return=0.08,
    )
    assert rec.euler_raroc > rec.standalone_raroc
    assert rec.diversification_benefit_pct >= 0.20
    assert rec.regime == "DIVERSIFICATION_SURPLUS_ACCUMULATOR"


def test_merton_perold_insolvency_tail_trap_regime() -> None:
    """Extreme volatility over long horizon drives guarantee capital >= 40% of gross assets."""
    rec = calculate_merton_perold_risk_capital(
        unit1_asset_v1=100.0,
        unit1_volatility=0.90,
        unit2_asset_v2=100.0,
        unit2_volatility=0.90,
        cross_unit_correlation=0.60,
        horizon_years=4.0,
    )
    assert rec.capital_intensity_ratio >= 0.40
    assert rec.regime == "INSOLVENCY_TAIL_CAPITAL_TRAP"


def test_merton_perold_input_validation_guards() -> None:
    """Strict allowlist and bounds validation blocks malformed tickers and out-of-range inputs."""
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_merton_perold_risk_capital(100.0, 0.25, ticker="JPM; DROP")
    with pytest.raises(ValueError, match="Correlation|correlation"):
        calculate_merton_perold_risk_capital(100.0, 0.25, cross_unit_correlation=1.20)
    with pytest.raises(ValueError, match="Volatilities"):
        calculate_merton_perold_risk_capital(100.0, -0.10)
    with pytest.raises(ValueError, match="Unit asset values"):
        calculate_merton_perold_risk_capital(0.0, 0.25)


def test_merton_perold_hmac_integrity_and_tamper_detection() -> None:
    """Validates HMAC-SHA256 seal verification and tamper sensitivity."""
    rec = calculate_merton_perold_risk_capital(100.0, 0.28, ticker="GS")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        v1=rec.unit1_asset_v1,
        vol1=rec.unit1_volatility,
        v2=rec.unit2_asset_v2,
        vol2=rec.unit2_volatility,
        corr=rec.cross_unit_correlation,
        horizon=rec.horizon_years,
        standalone_rc_1=rec.standalone_rc_1,
        portfolio_rc=rec.portfolio_rc,
        euler_rc_1=rec.euler_rc_1,
        euler_raroc=rec.euler_raroc,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("GS", "MS"), rec.hmac_signature) is False


def test_merton_perold_empirical_profile_calibration() -> None:
    """Calibrate trailing session volatility from simulated price history."""
    rng = np.random.default_rng(171)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, size=65)))
    rec = evaluate_merton_perold_profile(prices, ticker="AAPL", horizon_years=1.0)
    assert rec.ticker == "AAPL"
    assert 0.08 <= rec.unit1_volatility <= 1.50
    assert rec.portfolio_rc > 0.0


def test_merton_perold_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite audit persistence into merton_perold_risk_capital_audits."""
    db_file = tmp_path / "mp_audit.db"
    rec = calculate_merton_perold_risk_capital(100.0, 0.25, ticker="MS")
    assert persist_merton_perold_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, standalone_rc_1, portfolio_rc, euler_rc_1, regime "
            "FROM merton_perold_risk_capital_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "MS"
    assert rows[0][2] == pytest.approx(rec.portfolio_rc, rel=1e-9)


def test_merton_perold_cli_integration_json() -> None:
    """Execute CLI subcommand `risk-capital --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "risk-capital",
            "--ticker",
            "JPM",
            "--v1",
            "100.0",
            "--vol1",
            "0.30",
            "--v2",
            "100.0",
            "--vol2",
            "0.22",
            "--corr",
            "0.15",
            "--horizon",
            "1.0",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "JPM"
    assert payload["euler_rc_1"] + payload["euler_rc_2"] == pytest.approx(
        payload["portfolio_rc"], rel=1e-9
    )
    assert payload["diversification_benefit"] > 0.0

