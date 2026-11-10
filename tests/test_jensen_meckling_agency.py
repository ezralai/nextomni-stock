"""Unit and integration tests for Michael C. Jensen & William H. Meckling (1976) Agency Costs Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.jensen_meckling_agency import (
    JensenMecklingReceipt,
    calculate_jensen_meckling_equilibrium,
    evaluate_jensen_meckling_profile,
    persist_jensen_meckling_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_aligned_owner_manager_all_equity_first_best():
    """Verify 100% insider ownership (alpha=1.0) and zero debt eliminates all agency conflicts."""
    receipt = calculate_jensen_meckling_equilibrium(
        firm_potential_value=100.0,
        insider_ownership_alpha=1.0,
        debt_face_value=0.0,
        safe_volatility=0.20,
        risky_volatility=0.60,
        risky_npv_discount_pct=0.08,
        ticker="ALIGN",
    )

    assert isinstance(receipt, JensenMecklingReceipt)
    assert receipt.ticker == "ALIGN"
    assert receipt.equity_agency_loss == pytest.approx(0.0, abs=1e-8)
    assert receipt.actual_perk_consumption == pytest.approx(receipt.optimal_perk_consumption, rel=1e-6)
    assert receipt.risk_shifting_triggered is False
    assert receipt.asset_substitution_loss == 0.0
    assert receipt.total_agency_cost == pytest.approx(0.0, abs=1e-8)
    assert receipt.regime == "ALIGNED_OWNER_MANAGER_FIRST_BEST"


def test_asset_substitution_risk_shifting_expropriation():
    """Verify levered equity risk-shifting swaps high-NPV safe project for lower-NPV risky gamble."""
    receipt = calculate_jensen_meckling_equilibrium(
        firm_potential_value=100.0,
        insider_ownership_alpha=0.80,
        debt_face_value=85.0,       # High leverage -> strong call convexity
        safe_volatility=0.18,
        risky_volatility=0.65,      # High variance gamble
        risky_npv_discount_pct=0.06,# Destroys 6% of total firm value
        ticker="SHIFT",
    )

    assert receipt.risk_shifting_triggered is True
    assert receipt.risky_equity_value > receipt.safe_equity_value
    assert receipt.equity_risk_shifting_gain > 0.0
    assert receipt.creditor_wealth_expropriation > 0.0
    assert receipt.asset_substitution_loss > 0.0
    assert receipt.risky_debt_value < receipt.safe_debt_value


def test_outside_equity_perk_dilution_monotonicity():
    """Verify falling insider ownership alpha strictly increases perks and deadweight equity agency loss."""
    r_high_alpha = calculate_jensen_meckling_equilibrium(insider_ownership_alpha=0.90, debt_face_value=0.0)
    r_med_alpha = calculate_jensen_meckling_equilibrium(insider_ownership_alpha=0.50, debt_face_value=0.0)
    r_low_alpha = calculate_jensen_meckling_equilibrium(insider_ownership_alpha=0.25, debt_face_value=0.0)

    assert r_low_alpha.actual_perk_consumption > r_med_alpha.actual_perk_consumption > r_high_alpha.actual_perk_consumption
    assert r_low_alpha.equity_agency_loss > r_med_alpha.equity_agency_loss > r_high_alpha.equity_agency_loss


def test_debt_expropriation_monotonicity_in_leverage():
    """Verify higher debt face value expands equity risk-shifting gain from volatility expansion."""
    r_low_debt = calculate_jensen_meckling_equilibrium(debt_face_value=30.0, safe_volatility=0.18, risky_volatility=0.55, risky_npv_discount_pct=0.04)
    r_high_debt = calculate_jensen_meckling_equilibrium(debt_face_value=90.0, safe_volatility=0.18, risky_volatility=0.55, risky_npv_discount_pct=0.04)

    assert r_high_debt.equity_risk_shifting_gain >= r_low_debt.equity_risk_shifting_gain
    assert r_high_debt.creditor_wealth_expropriation >= r_low_debt.creditor_wealth_expropriation


def test_conservative_leverage_avoids_risk_shifting():
    """Verify low debt plus high risky NPV penalty prevents inefficient asset substitution."""
    receipt = calculate_jensen_meckling_equilibrium(
        firm_potential_value=100.0,
        insider_ownership_alpha=0.95,
        debt_face_value=15.0,
        safe_volatility=0.20,
        risky_volatility=0.35,
        risky_npv_discount_pct=0.15,
        ticker="SAFE",
    )

    assert receipt.risk_shifting_triggered is False
    assert receipt.asset_substitution_loss == 0.0
    assert receipt.creditor_wealth_expropriation == 0.0


def test_value_conservation_debt_plus_equity():
    """Verify total firm value splits strictly between debt and equity in both safe and risky projects."""
    receipt = calculate_jensen_meckling_equilibrium(ticker="VAL")

    assert receipt.safe_debt_value + receipt.safe_equity_value > 0.0
    assert receipt.risky_debt_value + receipt.risky_equity_value <= receipt.safe_debt_value + receipt.safe_equity_value + 1e-6


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positive finite value checks."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_jensen_meckling_equilibrium(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_jensen_meckling_equilibrium(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="must be finite and positive"):
        calculate_jensen_meckling_equilibrium(firm_potential_value=-50.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_jensen_meckling_equilibrium(ticker="META")

    sig_payload = (
        f"{receipt.ticker}:{receipt.firm_potential_value:.4f}:{receipt.insider_ownership_alpha:.4f}:"
        f"{receipt.debt_face_value:.4f}:{receipt.safe_volatility:.4f}:{receipt.risky_volatility:.4f}:"
        f"{receipt.equity_agency_loss:.4f}:{receipt.asset_substitution_loss:.4f}:"
        f"{receipt.total_agency_cost:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_jensen_meckling_profile_with_prices():
    """Verify empirical calibration from historical price series."""
    np.random.seed(42)
    prices = [100.0 + 0.6 * i + 2.5 * np.cos(i / 3.0) for i in range(45)]

    receipt = evaluate_jensen_meckling_profile(
        prices=prices,
        ticker="NFLX",
        insider_alpha=0.35,
        risky_volatility=0.60,
        debt_to_asset_ratio=0.70,
    )

    assert receipt.ticker == "NFLX"
    assert receipt.firm_potential_value > float(prices[-1])
    assert receipt.total_agency_cost > 0.0


def test_persist_jensen_meckling_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_jm.db")
    receipt = calculate_jensen_meckling_equilibrium(ticker="ORCL")

    success = persist_jensen_meckling_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, equity_agency_loss, total_agency_cost, regime FROM jensen_meckling_audits WHERE ticker = ?",
            ("ORCL",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "ORCL"
        assert row[1] == pytest.approx(receipt.equity_agency_loss, rel=1e-4)
        assert row[2] == pytest.approx(receipt.total_agency_cost, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_jensen_meckling_table_execution():
    """Verify CLI invocation for jensen-meckling command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "jensen-meckling",
        "--ticker",
        "FIRM",
        "--firm-value",
        "100.0",
        "--alpha",
        "0.40",
        "--debt",
        "80.0",
        "--risky-vol",
        "0.60",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Jensen & Meckling (1976) Agency Costs & Asset Substitution Engine" in res_table.stdout
    assert "FIRM" in res_table.stdout
    assert "Asset Substitution Loss" in res_table.stdout


def test_cli_jensen_meckling_json_execution():
    """Verify CLI invocation for jensen-meckling command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "jensen-meckling",
        "--ticker",
        "FIRM",
        "--firm-value",
        "100.0",
        "--debt",
        "80.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "FIRM"
    assert "equity_agency_loss" in parsed
    assert "risk_shifting_triggered" in parsed
    assert "hmac_signature" in parsed

