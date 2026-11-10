"""Unit and integration tests for Aghion & Bolton (1992) / Hart & Moore (1994) Control Rights & Strategic Default Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.aghion_bolton_control_rights import (
    AghionBoltonReceipt,
    calculate_aghion_bolton_equilibrium,
    evaluate_aghion_bolton_profile,
    persist_aghion_bolton_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_solvent_equity_control_no_strategic_default():
    """Verify solvent operation where creditor power is strong enough to prevent strategic default."""
    receipt = calculate_aghion_bolton_equilibrium(
        cash_flow=120.0,
        debt_face_value=70.0,
        liquidation_value=60.0,
        private_benefit=15.0,
        creditor_bargaining_power=0.80,  # D* = min(70, 0.8*120 + 0.2*60) = min(70, 108) = 70 -> no haircut!
        ticker="GOV",
    )

    assert isinstance(receipt, AghionBoltonReceipt)
    assert receipt.ticker == "GOV"
    assert receipt.contractual_control_holder == "ENTREPRENEUR_EQUITY"
    assert receipt.strategic_default_triggered is False
    assert receipt.liquidation_triggered is False
    assert receipt.renegotiated_debt_service == 70.0
    assert receipt.debt_haircut_pct == 0.0
    assert receipt.creditor_payoff == 70.0
    assert receipt.entrepreneur_payoff == 65.0  # (120 - 70) + 15
    assert receipt.regime == "EQUITY_CONTROL_SOLVENT_OPERATION"


def test_strategic_default_under_inalienable_human_capital():
    """Verify Hart-Moore strategic default when solvent borrower forces debt renegotiation."""
    receipt = calculate_aghion_bolton_equilibrium(
        cash_flow=100.0,
        debt_face_value=90.0,   # Solvent (y >= D)
        liquidation_value=40.0, # Low collateral liquidation value
        private_benefit=20.0,
        creditor_bargaining_power=0.40,  # D* = 0.40*100 + 0.60*40 = 64.0 < 90.0!
        ticker="STRAT",
    )

    assert receipt.contractual_control_holder == "ENTREPRENEUR_EQUITY"
    assert receipt.strategic_default_triggered is True
    assert receipt.liquidation_triggered is False
    assert receipt.renegotiated_debt_service == pytest.approx(64.0, rel=1e-4)
    assert receipt.debt_haircut_pct == pytest.approx((90.0 - 64.0) / 90.0, rel=1e-4)
    assert receipt.creditor_payoff == pytest.approx(64.0, rel=1e-4)
    assert receipt.entrepreneur_payoff == pytest.approx((100.0 - 64.0) + 20.0, rel=1e-4)
    assert receipt.regime == "STRATEGIC_DEFAULT_INALIENABLE_HUMAN_CAPITAL"


def test_liquidity_default_creditor_control_workout():
    """Verify Aghion-Bolton control shift to creditors during liquidity default and value-preserving workout."""
    receipt = calculate_aghion_bolton_equilibrium(
        cash_flow=60.0,         # y < D (Liquidity default)
        debt_face_value=100.0,
        liquidation_value=50.0, # L < y + B (60 + 25 = 85), so continuation preferred over liquidation
        private_benefit=25.0,
        creditor_bargaining_power=0.50,  # D* = min(100, 0.5*60 + 0.5*50) = 55.0
        ticker="WORK",
    )

    assert receipt.contractual_control_holder == "CREDITOR_DEBT"
    assert receipt.strategic_default_triggered is False
    assert receipt.liquidation_triggered is False
    assert receipt.renegotiated_debt_service == pytest.approx(55.0, rel=1e-4)
    assert receipt.debt_haircut_pct == pytest.approx(0.45, rel=1e-4)
    assert receipt.regime == "CREDITOR_CONTROL_SEVERE_RENEGOTIATION"


def test_efficient_creditor_liquidation():
    """Verify that creditors liquidate when liquidation value exceeds total continuation surplus (L > y + B)."""
    receipt = calculate_aghion_bolton_equilibrium(
        cash_flow=40.0,
        debt_face_value=90.0,
        liquidation_value=75.0, # L (75) > y + B (40 + 10 = 50)
        private_benefit=10.0,
        creditor_bargaining_power=0.50,
        ticker="LIQ",
    )

    assert receipt.contractual_control_holder == "CREDITOR_DEBT"
    assert receipt.liquidation_triggered is True
    assert receipt.creditor_payoff == 75.0
    assert receipt.entrepreneur_payoff == 0.0
    assert receipt.total_surplus_realized == 75.0 == receipt.first_best_surplus
    assert receipt.regime == "EFFICIENT_CREDITOR_LIQUIDATION"


def test_monotonicity_creditor_bargaining_power():
    """Verify that increasing creditor bargaining power strictly raises debt recovery and lowers haircut."""
    r_weak = calculate_aghion_bolton_equilibrium(cash_flow=100.0, debt_face_value=95.0, liquidation_value=40.0, creditor_bargaining_power=0.20)
    r_mid = calculate_aghion_bolton_equilibrium(cash_flow=100.0, debt_face_value=95.0, liquidation_value=40.0, creditor_bargaining_power=0.50)
    r_strong = calculate_aghion_bolton_equilibrium(cash_flow=100.0, debt_face_value=95.0, liquidation_value=40.0, creditor_bargaining_power=0.80)

    assert r_weak.creditor_payoff < r_mid.creditor_payoff < r_strong.creditor_payoff
    assert r_weak.debt_haircut_pct > r_mid.debt_haircut_pct > r_strong.debt_haircut_pct


def test_monotonicity_liquidation_collateral():
    """Verify that stronger collateral liquidation floor strictly strengthens debt capacity and recovery."""
    r_low_col = calculate_aghion_bolton_equilibrium(cash_flow=100.0, debt_face_value=95.0, liquidation_value=20.0, creditor_bargaining_power=0.40)
    r_high_col = calculate_aghion_bolton_equilibrium(cash_flow=100.0, debt_face_value=95.0, liquidation_value=70.0, creditor_bargaining_power=0.40)

    assert r_high_col.creditor_payoff > r_low_col.creditor_payoff
    assert r_high_col.debt_haircut_pct < r_low_col.debt_haircut_pct


def test_surplus_conservation_identity():
    """Verify total surplus realized equals creditor payoff plus entrepreneur payoff."""
    receipt = calculate_aghion_bolton_equilibrium(ticker="SURP")
    assert receipt.total_surplus_realized == pytest.approx(receipt.creditor_payoff + receipt.entrepreneur_payoff, rel=1e-6)
    assert receipt.efficiency_loss >= 0.0


def test_security_input_validation():
    """Verify strict ticker regex allowlist validation and exception handling."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_aghion_bolton_equilibrium(ticker="INVALID$$$TICKER")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_aghion_bolton_equilibrium(ticker="TOOLONGTICKERNAME")


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_aghion_bolton_equilibrium(ticker="AAPL")

    sig_payload = (
        f"{receipt.ticker}:{receipt.realized_cash_flow:.4f}:{receipt.debt_face_value:.4f}:"
        f"{receipt.liquidation_value:.4f}:{receipt.private_benefit:.4f}:{receipt.creditor_bargaining_power:.4f}:"
        f"{receipt.renegotiated_debt_service:.4f}:{receipt.creditor_payoff:.4f}:"
        f"{receipt.entrepreneur_payoff:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_aghion_bolton_profile_with_prices():
    """Verify empirical profile evaluation with historical price series."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 2.0 * np.cos(i / 3.0) for i in range(40)]

    receipt = evaluate_aghion_bolton_profile(
        prices=prices,
        ticker="MSFT",
        debt_ratio=0.60,
        liquidation_ratio=0.35,
        creditor_power=0.45,
    )

    assert receipt.ticker == "MSFT"
    assert receipt.realized_cash_flow > 0.0
    assert receipt.total_surplus_realized > 0.0


def test_persist_aghion_bolton_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_ab.db")
    receipt = calculate_aghion_bolton_equilibrium(ticker="JPM")

    success = persist_aghion_bolton_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, creditor_payoff, entrepreneur_payoff, regime FROM aghion_bolton_audits WHERE ticker = ?",
            ("JPM",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "JPM"
        assert row[1] == pytest.approx(receipt.creditor_payoff, rel=1e-4)
        assert row[2] == pytest.approx(receipt.entrepreneur_payoff, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_aghion_bolton_execution():
    """Verify CLI invocation for aghion-bolton command in standard and JSON modes."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "aghion-bolton",
        "--ticker",
        "GOV",
        "--cash-flow",
        "100.0",
        "--debt",
        "90.0",
        "--liquidation",
        "40.0",
        "--creditor-power",
        "0.40",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Aghion & Bolton (1992) / Hart & Moore (1994) Control Rights Engine" in res_table.stdout
    assert "GOV" in res_table.stdout
    assert "Strategic Default" in res_table.stdout

    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "aghion-bolton",
        "--ticker",
        "GOV",
        "--cash-flow",
        "100.0",
        "--debt",
        "90.0",
        "--liquidation",
        "40.0",
        "--creditor-power",
        "0.40",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "GOV"
    assert parsed["strategic_default_triggered"] is True
    assert "renegotiated_debt_service" in parsed
    assert "hmac_signature" in parsed

