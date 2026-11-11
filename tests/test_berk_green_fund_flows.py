"""Unit & CLI integration tests for Berk & Green (2004) Rational Fund Flows Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.berk_green_fund_flows import (
    calculate_berk_green_equilibrium,
    evaluate_berk_green_profile,
    persist_berk_green_record,
    verify_hmac_signature,
)


def test_zero_net_alpha_clearing_at_equilibrium_aum_q_star():
    """Equilibrium AUM q* clears expected future net alpha identically to zero."""
    rec = calculate_berk_green_equilibrium(
        prior_skill_phi0=0.03,
        prior_skill_vol=0.04,
        realized_alpha_r=0.07,
        return_noise_vol=0.04,
        management_fee_f=0.01,
        cost_convexity_a=0.005,
        current_aum_q=3.0,
        ticker="EQBM",
    )

    assert rec.signal_weight == pytest.approx(0.5, rel=1e-9)
    assert rec.posterior_skill_phi_hat == pytest.approx(0.05, rel=1e-9)
    assert rec.equilibrium_aum_q_star == pytest.approx(8.0, rel=1e-9)
    assert rec.equilibrium_net_alpha == pytest.approx(0.0, abs=1e-12)
    assert rec.expected_flow_delta_q == pytest.approx(5.0, rel=1e-9)
    assert rec.current_net_alpha > 0.0
    assert rec.regime == "UNDER_CAPACITY_POSITIVE_ALPHA_INFLOWS"


def test_outperformance_surprise_triggers_rational_inflows():
    """Positive realized performance surprise elevates posterior skill and raises rational inflows."""
    flat = calculate_berk_green_equilibrium(realized_alpha_r=0.02, current_aum_q=2.0)
    beat = calculate_berk_green_equilibrium(realized_alpha_r=0.10, current_aum_q=2.0)

    assert beat.posterior_skill_phi_hat > flat.posterior_skill_phi_hat
    assert beat.equilibrium_aum_q_star > flat.equilibrium_aum_q_star
    assert beat.expected_flow_delta_q > flat.expected_flow_delta_q


def test_over_capacity_fund_suffers_negative_alpha_and_outflows():
    """Funds exceeding equilibrium capacity q* earn negative net alpha and face outflows."""
    rec = calculate_berk_green_equilibrium(
        prior_skill_phi0=0.03,
        realized_alpha_r=0.03,
        management_fee_f=0.01,
        cost_convexity_a=0.005,  # q* = (0.03 - 0.01)/0.005 = 4.0
        current_aum_q=7.0,
        ticker="OVER",
    )

    assert rec.equilibrium_aum_q_star == pytest.approx(4.0, rel=1e-9)
    assert rec.expected_flow_delta_q < 0.0
    assert rec.current_net_alpha < 0.0
    assert rec.regime == "OVER_CAPACITY_NEGATIVE_ALPHA_OUTFLOWS"


def test_monopoly_manager_fee_maximizes_economic_rent():
    """Fee f* = 0.5 * phi_hat maximizes equilibrium manager fee revenue f * q*."""
    phi = 0.04
    opt = calculate_berk_green_equilibrium(
        prior_skill_phi0=phi,
        realized_alpha_r=phi,
        management_fee_f=0.02,
        cost_convexity_a=0.004,
    )
    low_fee = calculate_berk_green_equilibrium(
        prior_skill_phi0=phi,
        realized_alpha_r=phi,
        management_fee_f=0.01,
        cost_convexity_a=0.004,
    )
    high_fee = calculate_berk_green_equilibrium(
        prior_skill_phi0=phi,
        realized_alpha_r=phi,
        management_fee_f=0.03,
        cost_convexity_a=0.004,
    )

    assert opt.optimal_monopoly_fee_f_star == pytest.approx(0.02, rel=1e-9)
    assert opt.equilibrium_fee_revenue == pytest.approx(opt.max_manager_fee_rent, rel=1e-9)
    assert opt.equilibrium_fee_revenue > low_fee.equilibrium_fee_revenue
    assert opt.equilibrium_fee_revenue > high_fee.equilibrium_fee_revenue


def test_sub_fee_skill_forces_exit_to_index_zero_aum():
    """When posterior skill drops below management fee, rational AUM drops to zero."""
    rec = calculate_berk_green_equilibrium(
        prior_skill_phi0=0.01,
        realized_alpha_r=-0.02,
        management_fee_f=0.015,
        current_aum_q=2.0,
    )

    assert rec.equilibrium_aum_q_star == 0.0
    assert rec.expected_flow_delta_q == pytest.approx(-2.0, rel=1e-9)
    assert rec.regime == "SUB_FEE_SKILL_EXIT_TO_INDEX"


def test_lower_scale_convexity_expands_capacity_and_fee_rent():
    """Liquid mega-cap strategies (low cost_a) sustain larger equilibrium AUM and rents."""
    illiquid = calculate_berk_green_equilibrium(cost_convexity_a=0.010)
    liquid = calculate_berk_green_equilibrium(cost_convexity_a=0.002)

    assert liquid.equilibrium_aum_q_star > illiquid.equilibrium_aum_q_star
    assert liquid.max_manager_fee_rent > illiquid.max_manager_fee_rent


def test_competitive_zero_alpha_equilibrium_regime():
    """Fund already sitting at equilibrium AUM q == q* sits in COMPETITIVE_ZERO_ALPHA_EQUILIBRIUM."""
    rec = calculate_berk_green_equilibrium(
        prior_skill_phi0=0.04,
        realized_alpha_r=0.04,
        management_fee_f=0.01,
        cost_convexity_a=0.005,
        current_aum_q=6.0,
    )
    assert rec.current_net_alpha == pytest.approx(0.0, abs=1e-12)
    assert rec.regime == "COMPETITIVE_ZERO_ALPHA_EQUILIBRIUM"


def test_security_input_validation():
    """Verify strict ticker allowlist and positive convexity/volatility bounds."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_berk_green_equilibrium(ticker="BG04;DROP")

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_berk_green_equilibrium(cost_convexity_a=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_berk_green_equilibrium(prior_skill_vol=-0.02)

    with pytest.raises(ValueError, match="non-negative"):
        calculate_berk_green_equilibrium(management_fee_f=-0.01)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_berk_green_equilibrium(ticker="ARKK")
    sig_payload = (
        f"{rec.ticker}:{rec.prior_skill_phi0:.6f}:{rec.realized_alpha_r:.6f}:"
        f"{rec.management_fee_f:.6f}:{rec.cost_convexity_a:.6f}:{rec.current_aum_q:.4f}:"
        f"{rec.posterior_skill_phi_hat:.6f}:{rec.equilibrium_aum_q_star:.4f}:"
        f"{rec.current_net_alpha:.6f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_berk_green_profile_from_prices():
    """Verify trailing active return & dispersion calibration from price history."""
    prices = [100.0 + 0.35 * i + 1.5 * np.sin(i / 3.0) for i in range(40)]
    rec = evaluate_berk_green_profile(prices=prices, ticker="BLK")

    assert rec.ticker == "BLK"
    assert rec.equilibrium_aum_q_star >= 0.0


def test_persist_berk_green_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_berk_green.db")
    rec = calculate_berk_green_equilibrium(ticker="TROW")

    assert persist_berk_green_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, equilibrium_aum_q_star, current_net_alpha, regime FROM berk_green_audits WHERE ticker = ?",
            ("TROW",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "TROW"
        assert row[1] == pytest.approx(rec.equilibrium_aum_q_star, rel=1e-4)
        assert row[2] == pytest.approx(rec.current_net_alpha, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_fund_flows_table_and_json_execution():
    """Verify CLI execution for fund-flows command in both table and --json modes."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "fund-flows",
        "--ticker",
        "BG04",
        "--realized",
        "0.07",
        "--fee",
        "0.01",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Berk & Green (2004) Rational Fund Flows & Alpha Capacity Engine" in res_table.stdout
    assert "BG04" in res_table.stdout
    assert "Equilibrium AUM q*" in res_table.stdout

    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "fund-flows",
        "--ticker",
        "IVZ",
        "--realized",
        "0.08",
        "--aum",
        "2.5",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "IVZ"
    assert parsed["equilibrium_aum_q_star"] > 2.5
    assert parsed["expected_flow_delta_q"] > 0.0
    assert "hmac_signature" in parsed

