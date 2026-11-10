"""Unit and integration tests for Douglas W. Diamond (1984) Delegated Monitoring Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.diamond_delegated_monitoring import (
    DiamondMonitoringReceipt,
    calculate_diamond_delegated_monitoring,
    evaluate_diamond_monitoring_profile,
    persist_diamond_monitoring_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_diversified_bank_dominates_direct_lending():
    """Verify diversified bank (N=50, low rho) achieves K + d_N < min(mK, D_direct)."""
    receipt = calculate_diamond_delegated_monitoring(
        num_borrowers_n=50,
        depositors_per_firm_m=10,
        monitoring_cost_k=2.0,
        project_mean_payoff=125.0,
        project_volatility=25.0,
        loan_correlation_rho=0.0,
        required_depositor_return=104.0,
        ticker="BANK",
    )

    assert isinstance(receipt, DiamondMonitoringReceipt)
    assert receipt.ticker == "BANK"
    assert receipt.direct_duplicative_cost == 20.0
    assert receipt.delegated_total_cost < receipt.best_direct_cost
    assert receipt.net_intermediation_surplus > 0.0
    assert receipt.delegated_intermediation_viable is True
    assert receipt.diversification_efficiency_pct > 0.80


def test_single_borrower_bank_theorem_unviable():
    """Verify Diamond Theorem 1: single-loan bank (N=1) provides zero diversification and is unviable."""
    receipt = calculate_diamond_delegated_monitoring(
        num_borrowers_n=1,
        depositors_per_firm_m=10,
        monitoring_cost_k=2.0,
        project_mean_payoff=125.0,
        project_volatility=25.0,
        loan_correlation_rho=0.0,
        required_depositor_return=104.0,
        ticker="ONE",
    )

    assert receipt.portfolio_volatility == pytest.approx(receipt.project_volatility, rel=1e-6)
    assert receipt.diversification_efficiency_pct == pytest.approx(0.0, abs=1e-8)
    assert receipt.delegated_total_cost > receipt.direct_unmonitored_cost
    assert receipt.delegated_intermediation_viable is False
    assert receipt.regime == "SINGLE_LOAN_NO_DIVERSIFICATION_UNVIABLE"


def test_asymptotic_zero_delegation_cost_large_n():
    """Verify large independent bank portfolio collapses delegation cost d_N near zero."""
    receipt = calculate_diamond_delegated_monitoring(
        num_borrowers_n=500,
        loan_correlation_rho=0.0,
        ticker="MEGA",
    )

    assert receipt.delegation_cost_d_n <= 0.15 * receipt.monitoring_cost_k
    assert receipt.delegated_intermediation_viable is True
    assert receipt.regime == "ASYMPTOTIC_ZERO_DELEGATION_BANK_DOMINANCE"


def test_systemic_correlation_breaks_delegated_monitoring():
    """Verify surging systemic loan correlation rho destroys diversification and spikes d_N."""
    r_indep = calculate_diamond_delegated_monitoring(
        num_borrowers_n=100,
        project_mean_payoff=120.0,
        project_volatility=30.0,
        loan_correlation_rho=0.0,
        monitoring_cost_k=3.0,
        depositors_per_firm_m=20,
    )
    r_corr = calculate_diamond_delegated_monitoring(
        num_borrowers_n=100,
        project_mean_payoff=120.0,
        project_volatility=30.0,
        loan_correlation_rho=0.85,
        monitoring_cost_k=3.0,
        depositors_per_firm_m=20,
    )

    assert r_corr.portfolio_volatility > r_indep.portfolio_volatility
    assert r_corr.delegation_cost_d_n > r_indep.delegation_cost_d_n
    assert r_corr.net_intermediation_surplus < r_indep.net_intermediation_surplus


def test_monotonicity_in_scale_n():
    """Verify increasing independent borrower count strictly shrinks portfolio vol and d_N."""
    r_small = calculate_diamond_delegated_monitoring(num_borrowers_n=5, loan_correlation_rho=0.0)
    r_mid = calculate_diamond_delegated_monitoring(num_borrowers_n=25, loan_correlation_rho=0.0)
    r_large = calculate_diamond_delegated_monitoring(num_borrowers_n=125, loan_correlation_rho=0.0)

    assert r_small.portfolio_volatility > r_mid.portfolio_volatility > r_large.portfolio_volatility
    assert r_small.delegated_total_cost > r_mid.delegated_total_cost > r_large.delegated_total_cost
    assert r_large.net_intermediation_surplus > r_mid.net_intermediation_surplus > r_small.net_intermediation_surplus


def test_duplicative_monitoring_savings_bounds():
    """Verify optimal deposit promise exceeds safe return and total bank cost >= K."""
    receipt = calculate_diamond_delegated_monitoring(ticker="SAVE")
    assert receipt.optimal_deposit_face_value >= receipt.required_depositor_return
    assert receipt.delegated_total_cost >= receipt.monitoring_cost_k


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positivity validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_diamond_delegated_monitoring(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_diamond_delegated_monitoring(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="must be finite positive numbers"):
        calculate_diamond_delegated_monitoring(project_mean_payoff=-10.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_diamond_delegated_monitoring(ticker="JPM")

    sig_payload = (
        f"{receipt.ticker}:{receipt.num_borrowers_n}:{receipt.depositors_per_firm_m}:"
        f"{receipt.monitoring_cost_k:.4f}:{receipt.project_mean_payoff:.4f}:"
        f"{receipt.project_volatility:.4f}:{receipt.loan_correlation_rho:.4f}:"
        f"{receipt.delegation_cost_d_n:.4f}:{receipt.delegated_total_cost:.4f}:"
        f"{receipt.net_intermediation_surplus:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_diamond_monitoring_profile_with_prices():
    """Verify empirical calibration from historical price series."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 2.0 * np.sin(i / 2.5) for i in range(45)]

    receipt = evaluate_diamond_monitoring_profile(
        prices=prices,
        ticker="BAC",
        num_borrowers_n=60,
        loan_correlation_rho=0.05,
    )

    assert receipt.ticker == "BAC"
    assert receipt.delegated_intermediation_viable is True
    assert receipt.net_intermediation_surplus > 0.0


def test_persist_diamond_monitoring_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_dm.db")
    receipt = calculate_diamond_delegated_monitoring(ticker="WFC")

    success = persist_diamond_monitoring_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, delegation_cost_d_n, net_intermediation_surplus, regime FROM diamond_monitoring_audits WHERE ticker = ?",
            ("WFC",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "WFC"
        assert row[1] == pytest.approx(receipt.delegation_cost_d_n, rel=1e-4)
        assert row[2] == pytest.approx(receipt.net_intermediation_surplus, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_delegated_monitoring_table_execution():
    """Verify CLI invocation for delegated-monitoring command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "delegated-monitoring",
        "--ticker",
        "BANK",
        "--borrowers",
        "50",
        "--correlation",
        "0.05",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Diamond (1984) Financial Intermediation & Delegated Monitoring Engine" in res_table.stdout
    assert "BANK" in res_table.stdout
    assert "Delegation Cost d_N" in res_table.stdout


def test_cli_delegated_monitoring_json_execution():
    """Verify CLI invocation for delegated-monitoring command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "delegated-monitoring",
        "--ticker",
        "C",
        "--borrowers",
        "1",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "C"
    assert parsed["delegated_intermediation_viable"] is False
    assert "delegation_cost_d_n" in parsed
    assert "hmac_signature" in parsed

