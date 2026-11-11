"""Unit & CLI integration tests for Myers & Rajan (1998) Paradox of Liquidity Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.myers_rajan_liquidity_paradox import (
    calculate_myers_rajan_paradox,
    evaluate_myers_rajan_profile,
    persist_myers_rajan_record,
    verify_hmac_signature,
)


def test_humped_non_monotonic_debt_capacity_around_optimal_alpha():
    """Debt capacity rises with liquidity below alpha* and falls above alpha* (Paradox of Liquidity)."""
    peak_check = calculate_myers_rajan_paradox(
        invested_assets_c1=100.0,
        continuation_value_d2=120.0,
        asset_liquidity_alpha=0.50,
        transformation_friction_lambda=0.25,
        ticker="PEAK",
    )
    alpha_star = peak_check.optimal_liquidity_alpha_star
    assert alpha_star == pytest.approx(120.0 / 175.0, rel=1e-9)

    at_star = calculate_myers_rajan_paradox(
        invested_assets_c1=100.0,
        continuation_value_d2=120.0,
        asset_liquidity_alpha=alpha_star,
        transformation_friction_lambda=0.25,
    )
    assert at_star.liquidation_debt_capacity == pytest.approx(at_star.transformation_debt_ceiling, rel=1e-9)
    assert at_star.paradox_zone_active is False

    low_liq = calculate_myers_rajan_paradox(
        invested_assets_c1=100.0,
        continuation_value_d2=120.0,
        asset_liquidity_alpha=0.35,
        transformation_friction_lambda=0.25,
    )
    paradox_liq = calculate_myers_rajan_paradox(
        invested_assets_c1=100.0,
        continuation_value_d2=120.0,
        asset_liquidity_alpha=0.92,
        transformation_friction_lambda=0.25,
    )

    assert low_liq.binding_debt_capacity < at_star.binding_debt_capacity
    assert paradox_liq.binding_debt_capacity < at_star.binding_debt_capacity
    assert low_liq.paradox_zone_active is False
    assert paradox_liq.paradox_zone_active is True
    assert paradox_liq.paradox_capacity_loss > 0.0


def test_higher_transformation_friction_relaxes_paradox_constraint():
    """Greater transformation cost lambda curbs insider diversion and boosts borrowing capacity."""
    easy_transform = calculate_myers_rajan_paradox(
        asset_liquidity_alpha=0.85,
        transformation_friction_lambda=0.10,
        ticker="EASY",
    )
    hard_transform = calculate_myers_rajan_paradox(
        asset_liquidity_alpha=0.85,
        transformation_friction_lambda=0.60,
        ticker="HARD",
    )

    assert hard_transform.transformation_debt_ceiling > easy_transform.transformation_debt_ceiling
    assert hard_transform.binding_debt_capacity > easy_transform.binding_debt_capacity
    assert hard_transform.optimal_liquidity_alpha_star > easy_transform.optimal_liquidity_alpha_star


def test_insider_rent_share_shrinks_pledgeable_debt_capacity():
    """Higher insider rent theta shrinks going-concern pledgeability and lowers alpha*."""
    low_rent = calculate_myers_rajan_paradox(asset_liquidity_alpha=0.75, insider_rent_share_theta=0.05)
    high_rent = calculate_myers_rajan_paradox(asset_liquidity_alpha=0.75, insider_rent_share_theta=0.40)

    assert high_rent.transformation_debt_ceiling < low_rent.transformation_debt_ceiling
    assert high_rent.optimal_liquidity_alpha_star < low_rent.optimal_liquidity_alpha_star
    assert high_rent.binding_debt_capacity <= low_rent.binding_debt_capacity


def test_bank_monitoring_requirement_scales_with_paradox_loss():
    """Severe Paradox Zone firms require larger short-term/callable bank monitoring shares."""
    mild = calculate_myers_rajan_paradox(asset_liquidity_alpha=0.72, transformation_friction_lambda=0.25)
    severe = calculate_myers_rajan_paradox(asset_liquidity_alpha=0.95, transformation_friction_lambda=0.15)

    assert severe.bank_monitoring_requirement_pct > mild.bank_monitoring_requirement_pct
    assert severe.regime == "SEVERE_LIQUIDITY_PARADOX_SHORT_DEBT_REQUIRED"


def test_collateral_limited_regime_for_illiquid_assets():
    """Low-liquidity assets stay collateral-limited with zero paradox capacity loss."""
    rec = calculate_myers_rajan_paradox(asset_liquidity_alpha=0.25, continuation_value_d2=140.0)

    assert rec.binding_debt_capacity == pytest.approx(rec.liquidation_debt_capacity, rel=1e-9)
    assert rec.paradox_capacity_loss == 0.0
    assert rec.bank_monitoring_requirement_pct == 0.0
    assert rec.regime == "COLLATERAL_LIMITED_ILLIQUID_ZONE"


def test_optimal_alpha_clamped_at_one_when_d2_abundant():
    """When continuation value exceeds (2 - lambda)*c1, alpha* caps cleanly at 1.0."""
    rec = calculate_myers_rajan_paradox(
        invested_assets_c1=100.0,
        continuation_value_d2=250.0,
        asset_liquidity_alpha=0.90,
        transformation_friction_lambda=0.30,
    )
    assert rec.optimal_liquidity_alpha_star == pytest.approx(1.0, rel=1e-9)
    assert rec.paradox_zone_active is False


def test_security_input_validation():
    """Verify strict ticker allowlist and unit-interval parameter validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_myers_rajan_paradox(ticker="BAD;SQL")

    with pytest.raises(ValueError, match="asset_liquidity_alpha must satisfy"):
        calculate_myers_rajan_paradox(asset_liquidity_alpha=0.0)

    with pytest.raises(ValueError, match="transformation_friction_lambda must satisfy"):
        calculate_myers_rajan_paradox(transformation_friction_lambda=1.5)

    with pytest.raises(ValueError, match="insider_rent_share_theta must satisfy"):
        calculate_myers_rajan_paradox(insider_rent_share_theta=1.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_myers_rajan_paradox(ticker="JPM")
    sig_payload = (
        f"{rec.ticker}:{rec.invested_assets_c1:.4f}:{rec.continuation_value_d2:.4f}:"
        f"{rec.asset_liquidity_alpha:.4f}:{rec.transformation_friction_lambda:.4f}:"
        f"{rec.insider_rent_share_theta:.4f}:{rec.binding_debt_capacity:.4f}:"
        f"{rec.optimal_liquidity_alpha_star:.4f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_myers_rajan_profile_from_prices():
    """Verify empirical calibration from market price series."""
    prices = [100.0 + 0.35 * i + 1.5 * np.sin(i / 3.0) for i in range(40)]
    rec = evaluate_myers_rajan_profile(prices=prices, ticker="GS")

    assert rec.ticker == "GS"
    assert rec.invested_assets_c1 == pytest.approx(prices[-1], rel=1e-6)
    assert rec.binding_debt_capacity > 0.0


def test_persist_myers_rajan_record_sqlite(tmp_path):
    """Verify parameterized SQLite audit persistence for Myers & Rajan evaluations."""
    db_file = str(tmp_path / "test_myers_rajan.db")
    rec = calculate_myers_rajan_paradox(ticker="BAC")

    assert persist_myers_rajan_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, binding_debt_capacity, paradox_zone_active, regime FROM myers_rajan_audits WHERE ticker = ?",
            ("BAC",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "BAC"
        assert row[1] == pytest.approx(rec.binding_debt_capacity, rel=1e-4)
        assert bool(row[2]) == rec.paradox_zone_active
        assert row[3] == rec.regime


def test_cli_liquidity_paradox_table_execution():
    """Verify CLI execution for liquidity-paradox command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "liquidity-paradox",
        "--ticker",
        "MR98",
        "--alpha",
        "0.85",
        "--lambda-trans",
        "0.25",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Myers & Rajan (1998) Paradox of Liquidity Engine" in res.stdout
    assert "MR98" in res.stdout
    assert "Binding Debt Capacity" in res.stdout


def test_cli_liquidity_paradox_json_execution():
    """Verify CLI execution for liquidity-paradox command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "liquidity-paradox",
        "--ticker",
        "C",
        "--alpha",
        "0.90",
        "--lambda-trans",
        "0.20",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "C"
    assert parsed["paradox_zone_active"] is True
    assert parsed["paradox_capacity_loss"] > 0.0
    assert "hmac_signature" in parsed

