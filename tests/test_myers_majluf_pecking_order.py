"""Unit and integration tests for Stewart C. Myers & Nicholas S. Majluf (1984) Pecking Order Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.myers_majluf_pecking_order import (
    MyersMajlufReceipt,
    calculate_myers_majluf_equilibrium,
    evaluate_myers_majluf_profile,
    persist_myers_majluf_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_calculate_myers_majluf_baseline():
    """Verify standard Myers-Majluf equity issuance and dilution transfer."""
    receipt = calculate_myers_majluf_equilibrium(
        assets_in_place_true=120.0,
        assets_market_price=100.0,
        project_outlay=30.0,
        project_payoff=36.0,
        cash_slack=10.0,
        information_asymmetry_spread=0.15,
        ticker="CORP",
    )

    assert isinstance(receipt, MyersMajlufReceipt)
    assert receipt.ticker == "CORP"
    assert receipt.assets_in_place_true == 120.0
    assert receipt.assets_market_price == 100.0
    assert receipt.project_npv == 6.0
    assert receipt.external_financing_needed == 20.0
    assert receipt.financing_source == "EXTERNAL_EQUITY"
    assert receipt.undervaluation_wedge == 20.0
    assert receipt.dilution_transfer > 0.0
    assert receipt.net_old_shareholder_gain == pytest.approx(receipt.project_npv - receipt.dilution_transfer, rel=1e-5)
    assert receipt.underinvestment_triggered is False


def test_pecking_order_internal_cash_insulation():
    """Verify that when internal cash slack C >= I, firm completely avoids equity dilution."""
    receipt = calculate_myers_majluf_equilibrium(
        assets_in_place_true=150.0,  # Heavily undervalued
        assets_market_price=100.0,
        project_outlay=25.0,
        project_payoff=32.0,
        cash_slack=30.0,  # Cash slack C (30) > Outlay I (25)
        information_asymmetry_spread=0.25,
        ticker="CASH",
    )

    assert receipt.financing_source == "INTERNAL_CASH"
    assert receipt.external_financing_needed == 0.0
    assert receipt.dilution_transfer == 0.0
    assert receipt.net_old_shareholder_gain == receipt.project_npv == 7.0
    assert receipt.underinvestment_triggered is False
    assert receipt.equity_lemons_discount_pct == 0.0
    assert receipt.regime == "PECKING_ORDER_INTERNAL_FINANCED"


def test_underinvestment_trap_condition():
    """Verify Myers-Majluf underinvestment trap where a positive-NPV project is rejected due to dilution."""
    receipt = calculate_myers_majluf_equilibrium(
        assets_in_place_true=160.0,  # Severe undervaluation: A - P = 60
        assets_market_price=100.0,
        project_outlay=40.0,
        project_payoff=43.0,         # Modest positive NPV = +3.0
        cash_slack=0.0,              # No cash: must issue 40 of equity
        information_asymmetry_spread=0.20,
        ticker="TRAP",
    )

    assert receipt.project_npv == 3.0
    # Dilution transfer = [40 / (100 + 3)] * 60 = 23.30 > NPV (3.0)!
    assert receipt.dilution_transfer > receipt.project_npv
    assert receipt.net_old_shareholder_gain < 0.0
    assert receipt.underinvestment_triggered is True
    assert receipt.regime == "UNDERINVESTMENT_TRAP"


def test_adverse_selection_lemons_discount_regime():
    """Verify adverse selection lemons discount regime when announcement markdown is substantial."""
    receipt = calculate_myers_majluf_equilibrium(
        assets_in_place_true=105.0,
        assets_market_price=100.0,
        project_outlay=50.0,
        project_payoff=65.0,
        cash_slack=0.0,
        information_asymmetry_spread=0.30,  # High information asymmetry
        ticker="LEMON",
    )

    assert receipt.equity_lemons_discount_pct >= 0.04
    assert receipt.regime == "ADVERSE_SELECTION_LEMONS_DISCOUNT"


def test_mathematical_monotonicity_cash_slack():
    """Verify that increasing cash slack strictly decreases dilution transfer and external financing."""
    r_low_cash = calculate_myers_majluf_equilibrium(cash_slack=5.0)
    r_med_cash = calculate_myers_majluf_equilibrium(cash_slack=15.0)
    r_high_cash = calculate_myers_majluf_equilibrium(cash_slack=25.0)

    assert r_low_cash.external_financing_needed > r_med_cash.external_financing_needed > r_high_cash.external_financing_needed
    assert r_low_cash.dilution_transfer > r_med_cash.dilution_transfer > r_high_cash.dilution_transfer
    assert r_low_cash.net_old_shareholder_gain < r_med_cash.net_old_shareholder_gain < r_high_cash.net_old_shareholder_gain


def test_mathematical_monotonicity_info_asymmetry():
    """Verify that increasing information asymmetry spread strictly expands the lemons discount."""
    r1 = calculate_myers_majluf_equilibrium(information_asymmetry_spread=0.10)
    r2 = calculate_myers_majluf_equilibrium(information_asymmetry_spread=0.25)

    assert r1.equity_lemons_discount_pct < r2.equity_lemons_discount_pct
    assert r1.equity_lemons_discount_dollar < r2.equity_lemons_discount_dollar


def test_fairly_valued_or_overvalued_zero_dilution():
    """Verify that when assets in place are overvalued (A <= P), dilution transfer is zero."""
    receipt = calculate_myers_majluf_equilibrium(
        assets_in_place_true=90.0,   # True value A (90) <= Market price P (100)
        assets_market_price=100.0,
        project_outlay=20.0,
        project_payoff=26.0,
        cash_slack=5.0,
        ticker="OVER",
    )

    assert receipt.undervaluation_wedge == 0.0
    assert receipt.dilution_transfer == 0.0
    assert receipt.net_old_shareholder_gain == receipt.project_npv == 6.0
    assert receipt.underinvestment_triggered is False


def test_security_input_validation():
    """Verify strict ticker regex allowlist validation and exception handling."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_myers_majluf_equilibrium(ticker="INVALID$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_myers_majluf_equilibrium(ticker="TOOLONGTICKERNAME")


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_myers_majluf_equilibrium(ticker="NVDA")

    sig_payload = (
        f"{receipt.ticker}:{receipt.assets_in_place_true:.4f}:{receipt.assets_market_price:.4f}:"
        f"{receipt.project_outlay:.4f}:{receipt.project_payoff:.4f}:{receipt.cash_slack:.4f}:"
        f"{receipt.dilution_transfer:.4f}:{receipt.net_old_shareholder_gain:.4f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_myers_majluf_profile_with_prices():
    """Verify empirical profile evaluation with historical price series."""
    np.random.seed(42)
    prices = [150.0 + 0.5 * i + 3.0 * np.sin(i / 3.0) for i in range(40)]

    receipt = evaluate_myers_majluf_profile(
        prices=prices,
        ticker="TSLA",
        cash_slack=20.0,
        capex_outlay=30.0,
        info_spread=0.18,
    )

    assert receipt.ticker == "TSLA"
    assert receipt.assets_in_place_true >= receipt.assets_market_price
    assert receipt.project_npv > 0.0
    assert receipt.financing_source in ("INTERNAL_CASH", "EXTERNAL_EQUITY")


def test_persist_myers_majluf_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_mm.db")
    receipt = calculate_myers_majluf_equilibrium(ticker="INTC")

    success = persist_myers_majluf_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, dilution_transfer, net_old_shareholder_gain, regime FROM myers_majluf_audits WHERE ticker = ?", ("INTC",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "INTC"
        assert row[1] == pytest.approx(receipt.dilution_transfer, rel=1e-4)
        assert row[2] == pytest.approx(receipt.net_old_shareholder_gain, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_myers_majluf_execution():
    """Verify CLI invocation for myers-majluf command in standard and JSON modes."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "myers-majluf",
        "--ticker",
        "CORP",
        "--assets",
        "120.0",
        "--outlay",
        "30.0",
        "--payoff",
        "35.0",
        "--cash-slack",
        "10.0",
        "--info-spread",
        "0.15",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Myers & Majluf (1984) Pecking Order & Dilution Engine" in res_table.stdout
    assert "CORP" in res_table.stdout
    assert "Dilution Transfer to New Shs" in res_table.stdout

    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "myers-majluf",
        "--ticker",
        "CORP",
        "--assets",
        "120.0",
        "--outlay",
        "30.0",
        "--payoff",
        "35.0",
        "--cash-slack",
        "10.0",
        "--info-spread",
        "0.15",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "CORP"
    assert "dilution_transfer" in parsed
    assert "underinvestment_triggered" in parsed
    assert "hmac_signature" in parsed
