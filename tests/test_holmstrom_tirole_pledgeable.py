"""Unit and integration tests for Holmstrom & Tirole (1997) Pledgeable Income & Credit Rationing Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.holmstrom_tirole_pledgeable import (
    HolmstromTiroleReceipt,
    calculate_holmstrom_tirole_equilibrium,
    evaluate_holmstrom_tirole_profile,
    persist_holmstrom_tirole_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_direct_market_financing_tier():
    """Verify high-net-worth firm (A >= A_bar_direct) obtains unconstrained direct market financing."""
    receipt = calculate_holmstrom_tirole_equilibrium(
        investment_outlay=100.0,
        corporate_net_worth=40.0,  # Above A_bar_direct (~31.43)
        success_payoff=160.0,
        prob_diligence=0.80,
        prob_shirk=0.40,
        private_benefit_unmonitored=28.0,
        private_benefit_monitored=12.0,
        monitoring_cost=8.0,
        required_return_gamma=1.05,
        ticker="DIRECT",
    )

    assert isinstance(receipt, HolmstromTiroleReceipt)
    assert receipt.ticker == "DIRECT"
    assert receipt.direct_pledgeable_income == pytest.approx(72.0, rel=1e-5)
    assert receipt.monitored_pledgeable_income == pytest.approx(96.0, rel=1e-5)
    assert receipt.expected_project_npv == pytest.approx(23.0, rel=1e-5)
    assert receipt.financing_tier == "DIRECT_MARKET"
    assert receipt.credit_rationed is False
    assert receipt.funding_gap == 0.0
    assert receipt.regime == "DIRECT_CAPITAL_MARKET_UNCONSTRAINED"


def test_intermediated_bank_monitoring_tier():
    """Verify mid-tier firm (A_bar_bank <= A < A_bar_direct) accesses capital via bank monitoring."""
    receipt = calculate_holmstrom_tirole_equilibrium(
        corporate_net_worth=20.0,  # Between ~8.57 and ~31.43
        ticker="INTERM",
    )

    assert receipt.financing_tier == "INTERMEDIATED_BANK"
    assert receipt.credit_rationed is False
    assert receipt.funding_gap == 0.0
    assert receipt.min_net_worth_bank < receipt.corporate_net_worth < receipt.min_net_worth_direct
    assert receipt.regime == "BANK_MONITORED_INTERMEDIATION"


def test_credit_crunch_moral_hazard_rationing():
    """Verify capital-starved firm (A < A_bar_bank) suffers credit rationing despite positive NPV."""
    receipt = calculate_holmstrom_tirole_equilibrium(
        corporate_net_worth=5.0,   # Below A_bar_bank (~8.57)
        ticker="CRUNCH",
    )

    assert receipt.expected_project_npv > 0.0
    assert receipt.financing_tier == "CREDIT_RATIONED"
    assert receipt.credit_rationed is True
    assert receipt.funding_gap > 3.0
    assert receipt.regime == "CREDIT_CRUNCH_MORAL_HAZARD_RATIONING"


def test_negative_npv_rejection():
    """Verify negative expected NPV projects are rejected cleanly."""
    receipt = calculate_holmstrom_tirole_equilibrium(
        investment_outlay=100.0,
        corporate_net_worth=80.0,
        success_payoff=110.0,
        prob_diligence=0.80,        # Expected gross = 88 < 105 = gamma * I
        required_return_gamma=1.05,
        ticker="NEGNPV",
    )

    assert receipt.expected_project_npv < 0.0
    assert receipt.financing_tier == "NEGATIVE_NPV_REJECTED"
    assert receipt.credit_rationed is True
    assert receipt.regime == "NEGATIVE_NPV_PROJECT_UNFUNDABLE"


def test_monotonicity_private_benefit_shrinks_pledgeable_income():
    """Verify escalating shirking benefit B strictly shrinks pledgeable income and widens net-worth floor."""
    r_low_b = calculate_holmstrom_tirole_equilibrium(private_benefit_unmonitored=15.0)
    r_high_b = calculate_holmstrom_tirole_equilibrium(private_benefit_unmonitored=36.0)

    assert r_high_b.direct_pledgeable_income < r_low_b.direct_pledgeable_income
    assert r_high_b.min_net_worth_direct > r_low_b.min_net_worth_direct
    assert r_high_b.pledgeable_share_of_payoff < r_low_b.pledgeable_share_of_payoff


def test_pledgeable_income_never_exceeds_expected_payoff():
    """Verify P_direct <= p_H * R and zero private benefit yields 100% pledgeability."""
    r_zero_b = calculate_holmstrom_tirole_equilibrium(private_benefit_unmonitored=0.0, private_benefit_monitored=0.0)
    assert r_zero_b.pledgeable_share_of_payoff == pytest.approx(1.0, rel=1e-6)
    assert r_zero_b.min_net_worth_direct == 0.0


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positivity validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_holmstrom_tirole_equilibrium(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_holmstrom_tirole_equilibrium(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="investment_outlay must be finite and positive"):
        calculate_holmstrom_tirole_equilibrium(investment_outlay=-50.0)

    with pytest.raises(ValueError, match="success_payoff must be finite and positive"):
        calculate_holmstrom_tirole_equilibrium(success_payoff=0.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_holmstrom_tirole_equilibrium(ticker="JPM")

    sig_payload = (
        f"{receipt.ticker}:{receipt.investment_outlay:.4f}:{receipt.corporate_net_worth:.4f}:"
        f"{receipt.success_payoff:.4f}:{receipt.prob_diligence:.4f}:{receipt.prob_shirk:.4f}:"
        f"{receipt.private_benefit_unmonitored:.4f}:{receipt.direct_pledgeable_income:.4f}:"
        f"{receipt.monitored_pledgeable_income:.4f}:{receipt.financing_tier}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_holmstrom_tirole_profile_with_prices():
    """Verify empirical calibration from historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 2.0 * np.sin(i / 2.5) for i in range(40)]

    receipt = evaluate_holmstrom_tirole_profile(
        prices=prices,
        ticker="BAC",
        net_worth_ratio=0.25,
        private_benefit_pct=0.28,
    )

    assert receipt.ticker == "BAC"
    assert receipt.investment_outlay == pytest.approx(float(prices[-1]), rel=1e-4)
    assert receipt.direct_pledgeable_income > 0.0


def test_persist_holmstrom_tirole_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_ht.db")
    receipt = calculate_holmstrom_tirole_equilibrium(ticker="C")

    success = persist_holmstrom_tirole_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, direct_pledgeable_income, financing_tier, regime FROM holmstrom_tirole_audits WHERE ticker = ?",
            ("C",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "C"
        assert row[1] == pytest.approx(receipt.direct_pledgeable_income, rel=1e-4)
        assert row[2] == receipt.financing_tier
        assert row[3] == receipt.regime


def test_cli_holmstrom_tirole_table_execution():
    """Verify CLI invocation for holmstrom-tirole command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "holmstrom-tirole",
        "--ticker",
        "BANK",
        "--outlay",
        "100.0",
        "--net-worth",
        "20.0",
        "--payoff",
        "160.0",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Holmstrom & Tirole (1997) Pledgeable Income & Credit Rationing Engine" in res_table.stdout
    assert "BANK" in res_table.stdout
    assert "Pledgeable Income" in res_table.stdout


def test_cli_holmstrom_tirole_json_execution():
    """Verify CLI invocation for holmstrom-tirole command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "holmstrom-tirole",
        "--ticker",
        "GS",
        "--net-worth",
        "5.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "GS"
    assert parsed["credit_rationed"] is True
    assert "direct_pledgeable_income" in parsed
    assert "hmac_signature" in parsed

