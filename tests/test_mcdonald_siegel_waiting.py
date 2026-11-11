"""Unit & CLI integration tests for McDonald & Siegel (1986) Value of Waiting Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.mcdonald_siegel_waiting import (
    calculate_mcdonald_siegel_waiting,
    evaluate_mcdonald_siegel_profile,
    persist_mcdonald_siegel_record,
    verify_hmac_signature,
)


def test_value_matching_and_smooth_pasting_at_optimal_trigger():
    """Verify exact Value-Matching F(V*) = V* - I and Smooth-Pasting F'(V*) = 1 continuity."""
    base = calculate_mcdonald_siegel_waiting(
        project_value_v=120.0,
        investment_cost_i=100.0,
        risk_free_rate_r=0.05,
        dividend_shortfall_delta=0.04,
        cashflow_volatility_sigma=0.25,
        ticker="SMTH",
    )
    v_star = base.optimal_trigger_v_star
    assert base.hysteresis_hurdle_multiple > 1.0
    assert v_star > 100.0

    just_below = calculate_mcdonald_siegel_waiting(
        project_value_v=v_star * (1.0 - 1e-6),
        investment_cost_i=100.0,
        risk_free_rate_r=0.05,
        dividend_shortfall_delta=0.04,
        cashflow_volatility_sigma=0.25,
    )
    at_trigger = calculate_mcdonald_siegel_waiting(
        project_value_v=v_star,
        investment_cost_i=100.0,
        risk_free_rate_r=0.05,
        dividend_shortfall_delta=0.04,
        cashflow_volatility_sigma=0.25,
    )

    assert just_below.invest_immediately is False
    assert at_trigger.invest_immediately is True
    assert just_below.real_option_value_f == pytest.approx(v_star - 100.0, rel=1e-5)
    assert at_trigger.real_option_value_f == pytest.approx(v_star - 100.0, rel=1e-9)
    assert just_below.waiting_option_delta == pytest.approx(1.0, abs=1e-4)


def test_naive_npv_trap_positive_npv_still_waits():
    """Even when V > I (positive static NPV), waiting below V* yields strictly positive waiting premium."""
    rec = calculate_mcdonald_siegel_waiting(
        project_value_v=125.0,
        investment_cost_i=100.0,
        cashflow_volatility_sigma=0.30,
        ticker="TRAP",
    )

    assert rec.naive_static_npv == pytest.approx(25.0, rel=1e-9)
    assert rec.invest_immediately is False
    assert rec.real_option_value_f > rec.naive_static_npv
    assert rec.pure_value_of_waiting > 5.0
    assert rec.regime == "NAIVE_NPV_TRAP_WAIT_FOR_HYSTERESIS_HURDLE"


def test_volatility_increases_hysteresis_hurdle_and_option_value():
    """Higher cash-flow volatility raises hysteresis hurdle multiple and real option value."""
    low_vol = calculate_mcdonald_siegel_waiting(cashflow_volatility_sigma=0.15, project_value_v=110.0, ticker="LOW")
    high_vol = calculate_mcdonald_siegel_waiting(cashflow_volatility_sigma=0.45, project_value_v=110.0, ticker="HIGH")

    assert high_vol.characteristic_root_beta1 < low_vol.characteristic_root_beta1
    assert high_vol.hysteresis_hurdle_multiple > low_vol.hysteresis_hurdle_multiple
    assert high_vol.optimal_trigger_v_star > low_vol.optimal_trigger_v_star
    assert high_vol.real_option_value_f > low_vol.real_option_value_f


def test_deterministic_zero_volatility_limit():
    """As sigma -> 0+, beta_1 -> r/(r-delta) for r>delta and V* -> I for delta>=r."""
    det_equal = calculate_mcdonald_siegel_waiting(
        risk_free_rate_r=0.05,
        dividend_shortfall_delta=0.05,
        cashflow_volatility_sigma=0.0,
        project_value_v=110.0,
        investment_cost_i=100.0,
    )
    assert det_equal.optimal_trigger_v_star == pytest.approx(100.0, rel=1e-4)
    assert det_equal.invest_immediately is True

    det_growth = calculate_mcdonald_siegel_waiting(
        risk_free_rate_r=0.05,
        dividend_shortfall_delta=0.025,
        cashflow_volatility_sigma=0.0,
        project_value_v=150.0,
        investment_cost_i=100.0,
    )
    assert det_growth.hysteresis_hurdle_multiple == pytest.approx(2.0, rel=1e-6)
    assert det_growth.optimal_trigger_v_star == pytest.approx(200.0, rel=1e-6)


def test_dividend_shortfall_delta_lowers_hurdle_multiple():
    """Higher waiting opportunity cost delta shrinks the deferral hurdle V*."""
    patient = calculate_mcdonald_siegel_waiting(dividend_shortfall_delta=0.02, cashflow_volatility_sigma=0.25)
    impatient = calculate_mcdonald_siegel_waiting(dividend_shortfall_delta=0.08, cashflow_volatility_sigma=0.25)

    assert impatient.optimal_trigger_v_star < patient.optimal_trigger_v_star


def test_exercise_regime_above_trigger_zero_waiting_premium():
    """When V >= V*, pure value of waiting vanishes and optimal action is invest immediately."""
    rec = calculate_mcdonald_siegel_waiting(
        project_value_v=400.0,
        investment_cost_i=100.0,
        cashflow_volatility_sigma=0.25,
    )
    assert rec.invest_immediately is True
    assert rec.pure_value_of_waiting == 0.0
    assert rec.real_option_value_f == pytest.approx(300.0, rel=1e-9)
    assert rec.regime == "OPTIMAL_CAPEX_TRIGGER_REACHED_INVEST_NOW"


def test_out_of_money_and_high_uncertainty_regimes():
    """Verify OTM and high-uncertainty deep deferral classifications."""
    high_unc = calculate_mcdonald_siegel_waiting(
        project_value_v=90.0,
        investment_cost_i=100.0,
        dividend_shortfall_delta=0.03,
        cashflow_volatility_sigma=0.40,
    )
    assert high_unc.hysteresis_hurdle_multiple >= 2.25
    assert high_unc.regime == "HIGH_UNCERTAINTY_DEEP_OPTION_DEFERRAL"

    otm = calculate_mcdonald_siegel_waiting(
        project_value_v=85.0,
        investment_cost_i=100.0,
        dividend_shortfall_delta=0.06,
        cashflow_volatility_sigma=0.18,
    )
    assert otm.regime == "OUT_OF_THE_MONEY_PROJECT_OPTION"


def test_characteristic_quadratic_root_satisfaction():
    """Confirm beta_1 satisfies 0.5*sigma^2*b*(b-1) + (r-delta)*b - r == 0 to 1e-10."""
    r, delta, sig = 0.055, 0.035, 0.32
    rec = calculate_mcdonald_siegel_waiting(
        risk_free_rate_r=r,
        dividend_shortfall_delta=delta,
        cashflow_volatility_sigma=sig,
    )
    b = rec.characteristic_root_beta1
    lhs = 0.5 * sig * sig * b * (b - 1.0) + (r - delta) * b - r
    assert lhs == pytest.approx(0.0, abs=1e-10)
    assert b > 1.0


def test_security_input_validation():
    """Verify strict ticker allowlist and positivity guards for V, I, r, delta, sigma."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_mcdonald_siegel_waiting(ticker="MS86;DROP")

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_mcdonald_siegel_waiting(dividend_shortfall_delta=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_mcdonald_siegel_waiting(project_value_v=-100.0)

    with pytest.raises(ValueError, match="non-negative"):
        calculate_mcdonald_siegel_waiting(cashflow_volatility_sigma=-0.2)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_mcdonald_siegel_waiting(ticker="XOM")
    sig_payload = (
        f"{rec.ticker}:{rec.project_value_v:.4f}:{rec.investment_cost_i:.4f}:"
        f"{rec.risk_free_rate_r:.6f}:{rec.dividend_shortfall_delta:.6f}:"
        f"{rec.cashflow_volatility_sigma:.4f}:{rec.optimal_trigger_v_star:.4f}:"
        f"{rec.real_option_value_f:.6f}:{rec.pure_value_of_waiting:.6f}:"
        f"{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_mcdonald_siegel_profile_from_prices():
    """Verify empirical volatility calibration and trigger estimation from market closes."""
    prices = [100.0 + 0.3 * i + 2.0 * np.sin(i / 2.5) for i in range(40)]
    rec = evaluate_mcdonald_siegel_profile(prices=prices, ticker="CVX")

    assert rec.ticker == "CVX"
    assert rec.project_value_v == pytest.approx(prices[-1], rel=1e-6)
    assert rec.optimal_trigger_v_star > rec.investment_cost_i


def test_persist_mcdonald_siegel_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_mcdonald_siegel.db")
    rec = calculate_mcdonald_siegel_waiting(ticker="CAT")

    assert persist_mcdonald_siegel_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, optimal_trigger_v_star, pure_value_of_waiting, regime FROM mcdonald_siegel_audits WHERE ticker = ?",
            ("CAT",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "CAT"
        assert row[1] == pytest.approx(rec.optimal_trigger_v_star, rel=1e-4)
        assert row[2] == pytest.approx(rec.pure_value_of_waiting, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_waiting_to_invest_table_execution():
    """Verify CLI execution for waiting-to-invest command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "waiting-to-invest",
        "--ticker",
        "MS86",
        "--v",
        "130",
        "--cost",
        "100",
        "--vol",
        "0.28",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "McDonald & Siegel (1986) Value of Waiting & Irreversible Investment Engine" in res.stdout
    assert "MS86" in res.stdout
    assert "Optimal Hurdle Trigger" in res.stdout


def test_cli_waiting_to_invest_json_execution():
    """Verify CLI execution for waiting-to-invest command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "waiting-to-invest",
        "--ticker",
        "COP",
        "--v",
        "140",
        "--cost",
        "100",
        "--vol",
        "0.30",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "COP"
    assert parsed["hysteresis_hurdle_multiple"] > 1.0
    assert parsed["real_option_value_f"] >= max(0.0, parsed["naive_static_npv"])
    assert "hmac_signature" in parsed

