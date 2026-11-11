"""Unit and integration tests for Campbell & Cochrane (1999) External Habit Formation Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.campbell_cochrane_habit import (
    CampbellCochraneReceipt,
    calculate_campbell_cochrane_habit,
    evaluate_campbell_cochrane_profile,
    persist_campbell_cochrane_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_steady_state_habit_equity_premium_and_rfr_puzzles():
    """Verify baseline Campbell-Cochrane calibration yields ~0.51 Sharpe and ~2.4% real r_f with gamma=2."""
    receipt = calculate_campbell_cochrane_habit(
        utility_curvature_gamma=2.0,
        habit_persistence_phi=0.87,
        consumption_growth_mean=0.0189,
        consumption_growth_vol=0.0150,
        subjective_beta=0.89,
        current_surplus_ratio=None,  # Evaluated at steady state S_bar
        ticker="HABIT",
    )

    assert isinstance(receipt, CampbellCochraneReceipt)
    assert receipt.ticker == "HABIT"
    assert receipt.steady_state_surplus_s_bar == pytest.approx(0.05883, rel=1e-3)
    assert receipt.conditional_max_sharpe == pytest.approx(np.sqrt(2.0 * 0.13), rel=1e-5)
    assert receipt.local_risk_aversion_rra == pytest.approx(2.0 / receipt.steady_state_surplus_s_bar, rel=1e-5)
    assert 0.01 <= receipt.implied_risk_free_rate <= 0.04
    assert receipt.regime == "STEADY_STATE_HABIT_EQUILIBRIUM"


def test_recession_surplus_compression_spikes_rra_and_sharpe():
    """Verify deep recession (S_t << S_bar) sharply amplifies RRA and counter-cyclical equity premium."""
    steady = calculate_campbell_cochrane_habit()
    recession = calculate_campbell_cochrane_habit(
        current_surplus_ratio=0.025,  # Consumption dangerously close to habit X_t
        ticker="CRASH",
    )

    assert recession.local_risk_aversion_rra > steady.local_risk_aversion_rra * 2.0
    assert recession.conditional_max_sharpe > steady.conditional_max_sharpe
    assert recession.expected_equity_premium > steady.expected_equity_premium
    assert recession.regime == "SURPLUS_CRISIS_SPIKE_HIGH_RISK_AVERSION"


def test_boom_s_max_boundary_sensitivity_vanishes():
    """Verify at or above upper surplus boundary S_max, habit sensitivity lambda(s_t) = 0 identically."""
    base = calculate_campbell_cochrane_habit()
    at_boom = calculate_campbell_cochrane_habit(current_surplus_ratio=base.max_surplus_ratio_s_max * 1.05, ticker="BOOM")

    assert at_boom.sensitivity_lambda == 0.0
    assert at_boom.conditional_max_sharpe == pytest.approx(2.0 * 0.0150, rel=1e-6)
    assert at_boom.regime == "HABIT_COMPLACENCY_LOW_VOLATILITY_BOOM"


def test_constant_risk_free_rate_across_business_cycle():
    """Verify C-C Eq. (12) risk-free rate remains strictly invariant across surplus states S_t."""
    r_recession = calculate_campbell_cochrane_habit(current_surplus_ratio=0.02)
    r_steady = calculate_campbell_cochrane_habit(current_surplus_ratio=None)
    r_boom = calculate_campbell_cochrane_habit(current_surplus_ratio=0.09)

    assert r_recession.implied_risk_free_rate == pytest.approx(r_steady.implied_risk_free_rate, rel=1e-8)
    assert r_boom.implied_risk_free_rate == pytest.approx(r_steady.implied_risk_free_rate, rel=1e-8)


def test_monotonicity_countercyclical_equity_premium():
    """Verify falling surplus consumption strictly elevates RRA, sensitivity lambda, and expected premium."""
    r_high = calculate_campbell_cochrane_habit(current_surplus_ratio=0.08)
    r_mid = calculate_campbell_cochrane_habit(current_surplus_ratio=0.05)
    r_low = calculate_campbell_cochrane_habit(current_surplus_ratio=0.025)

    assert r_low.local_risk_aversion_rra > r_mid.local_risk_aversion_rra > r_high.local_risk_aversion_rra
    assert r_low.sensitivity_lambda > r_mid.sensitivity_lambda > r_high.sensitivity_lambda
    assert r_low.expected_equity_premium > r_mid.expected_equity_premium > r_high.expected_equity_premium


def test_habit_persistence_half_life_formula():
    """Verify exact AR(1) habit decay half-life ln(2)/(-ln(phi))."""
    receipt = calculate_campbell_cochrane_habit(habit_persistence_phi=0.87)
    assert receipt.habit_half_life_years == pytest.approx(np.log(2.0) / -np.log(0.87), rel=1e-6)


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positivity validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_campbell_cochrane_habit(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_campbell_cochrane_habit(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="utility_curvature_gamma must be finite and positive"):
        calculate_campbell_cochrane_habit(utility_curvature_gamma=-2.0)

    with pytest.raises(ValueError, match="consumption_growth_vol must be finite and positive"):
        calculate_campbell_cochrane_habit(consumption_growth_vol=0.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_campbell_cochrane_habit(ticker="SPY")

    sig_payload = (
        f"{receipt.ticker}:{receipt.utility_curvature_gamma:.4f}:{receipt.habit_persistence_phi:.4f}:"
        f"{receipt.consumption_growth_vol:.4f}:{receipt.steady_state_surplus_s_bar:.6f}:"
        f"{receipt.current_surplus_ratio:.6f}:{receipt.sensitivity_lambda:.6f}:"
        f"{receipt.local_risk_aversion_rra:.4f}:{receipt.conditional_max_sharpe:.6f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_campbell_cochrane_profile_with_drawdown_prices():
    """Verify empirical calibration reflects drawdown by lowering surplus consumption and raising Sharpe."""
    prices_up = [100.0 + 0.5 * i for i in range(40)]
    prices_crash = [120.0 - 0.8 * i for i in range(40)]

    rec_up = evaluate_campbell_cochrane_profile(prices_up, ticker="BOOM")
    rec_crash = evaluate_campbell_cochrane_profile(prices_crash, ticker="BUST")

    assert rec_crash.current_surplus_ratio < rec_up.current_surplus_ratio
    assert rec_crash.local_risk_aversion_rra > rec_up.local_risk_aversion_rra


def test_persist_campbell_cochrane_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_cc_habit.db")
    receipt = calculate_campbell_cochrane_habit(ticker="QQQ")

    success = persist_campbell_cochrane_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, local_risk_aversion_rra, conditional_max_sharpe, regime FROM campbell_cochrane_audits WHERE ticker = ?",
            ("QQQ",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "QQQ"
        assert row[1] == pytest.approx(receipt.local_risk_aversion_rra, rel=1e-4)
        assert row[2] == pytest.approx(receipt.conditional_max_sharpe, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_habit_formation_table_execution():
    """Verify CLI invocation for habit-formation command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "habit-formation",
        "--ticker",
        "HABIT",
        "--gamma",
        "2.0",
        "--phi",
        "0.87",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Campbell & Cochrane (1999) External Habit Formation Engine" in res_table.stdout
    assert "HABIT" in res_table.stdout
    assert "Local Relative Risk Aversion" in res_table.stdout


def test_cli_habit_formation_json_execution():
    """Verify CLI invocation for habit-formation command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "habit-formation",
        "--ticker",
        "SPY",
        "--surplus-ratio",
        "0.03",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "SPY"
    assert "conditional_max_sharpe" in parsed
    assert "local_risk_aversion_rra" in parsed
    assert "hmac_signature" in parsed

