"""Unit & CLI integration tests for Modigliani & Miller (1958, 1961, 1963) Corporate Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.modigliani_miller_corporate import (
    calculate_modigliani_miller_corporate,
    evaluate_modigliani_miller_profile,
    persist_modigliani_miller_record,
    verify_hmac_signature,
)


def test_mm_1958_tax_free_proposition_one_and_wacc_invariance():
    """With tau_c = 0, firm value V_L and WACC are invariant across debt leverage levels."""
    unlevered = calculate_modigliani_miller_corporate(
        ebit=20.0,
        debt_d=0.0,
        unlevered_r0=0.10,
        cost_of_debt_rd=0.05,
        corporate_tax_rate_tc=0.0,
        incremental_roic=0.10,
        ticker="MM58A",
    )
    levered = calculate_modigliani_miller_corporate(
        ebit=20.0,
        debt_d=80.0,
        unlevered_r0=0.10,
        cost_of_debt_rd=0.05,
        corporate_tax_rate_tc=0.0,
        incremental_roic=0.10,
        ticker="MM58B",
    )

    assert unlevered.levered_value_vl == pytest.approx(200.0, rel=1e-9)
    assert levered.levered_value_vl == pytest.approx(unlevered.levered_value_vl, rel=1e-9)
    assert levered.wacc == pytest.approx(0.10, rel=1e-9)
    assert levered.levered_equity_cost_re > unlevered.levered_equity_cost_re
    assert unlevered.regime == "TAX_FREE_MM_PROPOSITION_INVARIANCE"


def test_mm_1963_corporate_tax_shield_and_exact_wacc_identity():
    """With corporate taxes tau_c > 0, V_L = V_U + tau_c*D and WACC == r_0*(1 - tau_c*D/V_L)."""
    rec = calculate_modigliani_miller_corporate(
        ebit=25.0,
        debt_d=60.0,
        unlevered_r0=0.10,
        cost_of_debt_rd=0.05,
        corporate_tax_rate_tc=0.25,
        ticker="MM63",
    )

    expected_vu = (25.0 * 0.75) / 0.10
    expected_vl = expected_vu + 0.25 * 60.0
    expected_wacc = 0.10 * (1.0 - 0.25 * (60.0 / expected_vl))

    assert rec.unlevered_value_vu == pytest.approx(expected_vu, rel=1e-9)
    assert rec.tax_shield_pv == pytest.approx(15.0, rel=1e-9)
    assert rec.levered_value_vl == pytest.approx(expected_vl, rel=1e-9)
    assert rec.wacc == pytest.approx(expected_wacc, rel=1e-9)
    assert (rec.nopat / rec.wacc) == pytest.approx(rec.levered_value_vl, rel=1e-9)


def test_mm_1961_dividend_and_retention_irrelevance_at_fair_roic():
    """When incremental ROIC == r_0, reinvestment rate k and payout ratio yield zero PVGO."""
    low_ret = calculate_modigliani_miller_corporate(
        ebit=20.0,
        unlevered_r0=0.10,
        reinvestment_rate_k=0.10,
        incremental_roic=0.10,
        payout_ratio=0.90,
        ticker="DIV90",
    )
    high_ret = calculate_modigliani_miller_corporate(
        ebit=20.0,
        unlevered_r0=0.10,
        reinvestment_rate_k=0.80,
        incremental_roic=0.10,
        payout_ratio=0.20,
        ticker="DIV20",
    )

    assert low_ret.pvgo == pytest.approx(0.0, abs=1e-9)
    assert high_ret.pvgo == pytest.approx(0.0, abs=1e-9)
    assert low_ret.mm_growth_equity_value == pytest.approx(high_ret.mm_growth_equity_value, rel=1e-9)
    assert low_ret.dividend_irrelevance_wedge == 0.0


def test_positive_and_negative_roic_spread_drive_pvgo():
    """Value creation arises strictly from ROIC > r_0; ROIC < r_0 destroys equity value."""
    compounder = calculate_modigliani_miller_corporate(
        unlevered_r0=0.10,
        reinvestment_rate_k=0.60,
        incremental_roic=0.18,
        ticker="COMP",
    )
    destroyer = calculate_modigliani_miller_corporate(
        unlevered_r0=0.10,
        reinvestment_rate_k=0.60,
        incremental_roic=0.05,
        ticker="DEST",
    )

    assert compounder.pvgo > 0.0
    assert compounder.pvgo_share_pct >= 25.0
    assert compounder.regime == "HIGH_PVGO_COMPOUNDING_FRANCHISE"
    assert destroyer.pvgo < 0.0
    assert destroyer.regime == "NEGATIVE_SPREAD_VALUE_DESTRUCTIVE_GROWTH"


def test_mm_proposition_two_linear_cost_of_equity():
    """Doubling Debt/Equity ratio doubles leverage risk premium (r_E - r_0)."""
    rec1 = calculate_modigliani_miller_corporate(ebit=20.0, debt_d=30.0, corporate_tax_rate_tc=0.20, unlevered_r0=0.10, cost_of_debt_rd=0.05)
    rec2 = calculate_modigliani_miller_corporate(ebit=20.0, debt_d=60.0, corporate_tax_rate_tc=0.20, unlevered_r0=0.10, cost_of_debt_rd=0.05)

    prem1 = rec1.levered_equity_cost_re - 0.10
    prem2 = rec2.levered_equity_cost_re - 0.10
    assert (prem1 / rec1.debt_to_equity_ratio) == pytest.approx(0.05 * 0.80, rel=1e-9)
    assert (prem2 / rec2.debt_to_equity_ratio) == pytest.approx(0.05 * 0.80, rel=1e-9)


def test_security_input_validation_and_insolvency_guard():
    """Verify strict ticker allowlist, rate bounds, and insolvent capital structure guard."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_modigliani_miller_corporate(ticker="MM61$BAD")

    with pytest.raises(ValueError, match="ebit must be strictly positive"):
        calculate_modigliani_miller_corporate(ebit=-10.0)

    with pytest.raises(ValueError, match="corporate_tax_rate_tc must satisfy"):
        calculate_modigliani_miller_corporate(corporate_tax_rate_tc=1.0)

    with pytest.raises(ValueError, match="insolvent capital structure"):
        calculate_modigliani_miller_corporate(ebit=10.0, unlevered_r0=0.10, debt_d=250.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_modigliani_miller_corporate(ticker="AAPL")
    sig_payload = (
        f"{rec.ticker}:{rec.ebit:.4f}:{rec.debt_d:.4f}:{rec.unlevered_r0:.6f}:"
        f"{rec.corporate_tax_rate_tc:.4f}:{rec.reinvestment_rate_k:.4f}:"
        f"{rec.incremental_roic:.6f}:{rec.levered_value_vl:.6f}:{rec.wacc:.6f}:"
        f"{rec.pvgo:.6f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_modigliani_miller_profile_from_prices():
    """Verify empirical discount rate calibration from market price series."""
    prices = [110.0 + 0.4 * i + 1.5 * np.sin(i / 2.5) for i in range(40)]
    rec = evaluate_modigliani_miller_profile(prices=prices, ticker="MSFT")

    assert rec.ticker == "MSFT"
    assert rec.levered_value_vl > rec.debt_d
    assert 0.0 < rec.wacc < 0.25


def test_persist_modigliani_miller_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_mm.db")
    rec = calculate_modigliani_miller_corporate(ticker="BRK")

    assert persist_modigliani_miller_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, levered_value_vl, wacc, pvgo, regime FROM modigliani_miller_audits WHERE ticker = ?",
            ("BRK",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "BRK"
        assert row[1] == pytest.approx(rec.levered_value_vl, rel=1e-4)
        assert row[2] == pytest.approx(rec.wacc, rel=1e-4)
        assert row[3] == pytest.approx(rec.pvgo, rel=1e-4)
        assert row[4] == rec.regime


def test_cli_modigliani_miller_table_execution():
    """Verify CLI execution for modigliani-miller command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "modigliani-miller",
        "--ticker",
        "MM61",
        "--ebit",
        "25",
        "--debt",
        "60",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Modigliani & Miller (1958, 1961, 1963) Capital Structure & Dividend Engine" in res.stdout
    assert "MM61" in res.stdout
    assert "After-Tax WACC" in res.stdout


def test_cli_modigliani_miller_json_execution():
    """Verify CLI execution for modigliani-miller command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "modigliani-miller",
        "--ticker",
        "KO",
        "--roic",
        "0.16",
        "--tax-rate",
        "0.21",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "KO"
    assert parsed["incremental_roic"] == pytest.approx(0.16)
    assert parsed["pvgo"] > 0.0
    assert "hmac_signature" in parsed


def test_leverage_tax_shield_monotonically_reduces_wacc():
    """Under positive corporate taxes, increasing safe leverage monotonically lowers WACC."""
    low_lev = calculate_modigliani_miller_corporate(ebit=30.0, debt_d=20.0, corporate_tax_rate_tc=0.25)
    high_lev = calculate_modigliani_miller_corporate(ebit=30.0, debt_d=90.0, corporate_tax_rate_tc=0.25)

    assert high_lev.wacc < low_lev.wacc
    assert high_lev.levered_value_vl > low_lev.levered_value_vl

