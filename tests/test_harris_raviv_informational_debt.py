"""Unit & CLI integration tests for Harris & Raviv (1990) Informational Role of Debt Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.harris_raviv_informational_debt import (
    calculate_harris_raviv_debt,
    evaluate_harris_raviv_profile,
    persist_harris_raviv_record,
    verify_hmac_signature,
)


def test_optimal_debt_maximizes_firm_value_above_unlevered_and_excess_debt():
    """Interior D* = L - k strictly dominates both zero-debt equity and excess leverage."""
    opt = calculate_harris_raviv_debt(
        y_min=40.0,
        y_max=160.0,
        liquidation_value_l=95.0,
        investigation_cost_k=8.0,
        ticker="OPT",
    )
    assert opt.optimal_debt_d_star == pytest.approx(87.0, rel=1e-9)
    assert opt.promised_debt_d == pytest.approx(87.0, rel=1e-9)
    assert opt.firm_value_v_d > opt.unlevered_value
    assert opt.net_discipline_value > 0.0
    assert opt.over_leverage_waste == pytest.approx(0.0, abs=1e-9)
    assert opt.regime == "OPTIMAL_INFORMATIONAL_DEBT_DISCIPLINE"

    zero_debt = calculate_harris_raviv_debt(
        y_min=40.0,
        y_max=160.0,
        liquidation_value_l=95.0,
        investigation_cost_k=8.0,
        promised_debt_d=40.0,
    )
    assert zero_debt.firm_value_v_d == pytest.approx(zero_debt.unlevered_value, rel=1e-9)
    assert zero_debt.regime == "UNDER_LEVERAGED_CONTINUATION_BIAS"

    excess_debt = calculate_harris_raviv_debt(
        y_min=40.0,
        y_max=160.0,
        liquidation_value_l=95.0,
        investigation_cost_k=8.0,
        promised_debt_d=140.0,
    )
    assert excess_debt.firm_value_v_d < opt.firm_value_v_d
    assert excess_debt.over_leverage_waste > 0.25
    assert excess_debt.regime == "EXCESS_DEBT_OVER_INVESTIGATION_WASTE"


def test_higher_investigation_cost_k_reduces_optimal_debt_and_value():
    """Higher default audit cost k shrinks optimal leverage D* and net discipline gain."""
    low_k = calculate_harris_raviv_debt(investigation_cost_k=4.0, liquidation_value_l=100.0)
    high_k = calculate_harris_raviv_debt(investigation_cost_k=25.0, liquidation_value_l=100.0)

    assert high_k.optimal_debt_d_star < low_k.optimal_debt_d_star
    assert high_k.optimal_firm_value_v_star < low_k.optimal_firm_value_v_star
    assert high_k.default_audit_prob < low_k.default_audit_prob


def test_higher_tangibility_liquidation_value_raises_optimal_debt():
    """Higher asset liquidation value L increases D*, default-audit frequency, and firm value."""
    intangible = calculate_harris_raviv_debt(liquidation_value_l=65.0, investigation_cost_k=8.0)
    tangible = calculate_harris_raviv_debt(liquidation_value_l=120.0, investigation_cost_k=8.0)

    assert tangible.optimal_debt_d_star > intangible.optimal_debt_d_star
    assert tangible.optimal_firm_value_v_star > intangible.optimal_firm_value_v_star
    assert tangible.default_audit_prob > intangible.default_audit_prob


def test_prohibitive_investigation_cost_collapses_to_all_equity():
    """When k >= L - y_min, auditing never pays off and D* collapses to y_min (zero default)."""
    rec = calculate_harris_raviv_debt(
        y_min=40.0,
        y_max=160.0,
        liquidation_value_l=70.0,
        investigation_cost_k=35.0,  # > 70 - 40 = 30
    )

    assert rec.optimal_debt_d_star == pytest.approx(40.0, rel=1e-9)
    assert rec.default_audit_prob == pytest.approx(0.0, abs=1e-9)
    assert rec.regime == "ALL_EQUITY_ZERO_DEFAULT_PROHIBITIVE_AUDIT_COST"


def test_exact_net_discipline_closed_form_integral():
    """Verify V(D*) - E[Y] == (L - k - y_min)^2 / (2*(y_max - y_min)) analytically."""
    y0, y1, l_val, k_val = 40.0, 160.0, 100.0, 12.0
    rec = calculate_harris_raviv_debt(
        y_min=y0, y_max=y1, liquidation_value_l=l_val, investigation_cost_k=k_val
    )
    expected_gain = ((l_val - k_val - y0) ** 2) / (2.0 * (y1 - y0))
    assert rec.net_discipline_value == pytest.approx(expected_gain, rel=1e-9)


def test_liquidation_prob_bounded_by_default_audit_prob():
    """Liquidation occurs only upon investigating default (Y < D), so P(liq) <= P(default)."""
    rec = calculate_harris_raviv_debt(promised_debt_d=130.0, liquidation_value_l=95.0)
    assert rec.liquidation_prob <= rec.default_audit_prob
    assert 0.0 <= rec.liquidation_prob <= 1.0


def test_security_input_validation():
    """Verify strict ticker allowlist, cash-flow support y_min < y_max, and non-negativity guards."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_harris_raviv_debt(ticker="HR90;DROP")

    with pytest.raises(ValueError, match="Requires 0 <= y_min < y_max"):
        calculate_harris_raviv_debt(y_min=100.0, y_max=90.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_harris_raviv_debt(investigation_cost_k=-5.0)

    with pytest.raises(ValueError, match="promised_debt_d must be non-negative"):
        calculate_harris_raviv_debt(promised_debt_d=-10.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_harris_raviv_debt(ticker="T")
    sig_payload = (
        f"{rec.ticker}:{rec.y_min:.4f}:{rec.y_max:.4f}:{rec.liquidation_value_l:.4f}:"
        f"{rec.investigation_cost_k:.4f}:{rec.promised_debt_d:.4f}:"
        f"{rec.firm_value_v_d:.6f}:{rec.optimal_debt_d_star:.4f}:"
        f"{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_harris_raviv_profile_from_prices():
    """Verify empirical support [y_min, y_max] and liquidation calibration from prices."""
    prices = [100.0 + 0.3 * i + 2.0 * np.sin(i / 2.5) for i in range(40)]
    rec = evaluate_harris_raviv_profile(prices=prices, ticker="VZ")

    assert rec.ticker == "VZ"
    assert rec.y_min < rec.y_max
    assert rec.firm_value_v_d >= rec.unlevered_value - 1e-9


def test_persist_harris_raviv_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_hr90.db")
    rec = calculate_harris_raviv_debt(ticker="F")

    assert persist_harris_raviv_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, optimal_debt_d_star, firm_value_v_d, regime FROM harris_raviv_audits WHERE ticker = ?",
            ("F",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "F"
        assert row[1] == pytest.approx(rec.optimal_debt_d_star, rel=1e-4)
        assert row[2] == pytest.approx(rec.firm_value_v_d, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_informational_debt_table_execution():
    """Verify CLI execution for informational-debt command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "informational-debt",
        "--ticker",
        "HR90",
        "--liq-val",
        "95",
        "--audit-cost",
        "8",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Harris & Raviv (1990) Capital Structure & Informational Debt Engine" in res.stdout
    assert "HR90" in res.stdout
    assert "Optimal Debt Face D*" in res.stdout


def test_cli_informational_debt_json_execution():
    """Verify CLI execution for informational-debt command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "informational-debt",
        "--ticker",
        "GM",
        "--liq-val",
        "100",
        "--audit-cost",
        "10",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "GM"
    assert parsed["optimal_debt_d_star"] == pytest.approx(90.0)
    assert parsed["net_discipline_value"] > 0.0
    assert "hmac_signature" in parsed

