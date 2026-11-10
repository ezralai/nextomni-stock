"""Unit and integration tests for Robert E. Lucas Jr. (1978) Pure Exchange Tree Equilibrium Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.lucas_tree_equilibrium import (
    LucasTreeReceipt,
    calculate_lucas_tree_equilibrium,
    evaluate_lucas_tree_profile,
    persist_lucas_tree_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_calculate_lucas_tree_steady_state():
    """Verify standard Lucas tree steady state equilibrium with positive pricing kernel and equity premium."""
    receipt = calculate_lucas_tree_equilibrium(
        current_dividend=2.0,
        subjective_discount_factor=0.96,
        relative_risk_aversion=2.5,
        dividend_growth_mean=0.02,
        dividend_growth_vol=0.04,
        ticker="SPY",
    )

    assert isinstance(receipt, LucasTreeReceipt)
    assert receipt.ticker == "SPY"
    assert receipt.current_dividend == 2.0
    assert receipt.transversality_satisfied is True
    assert 0.80 < receipt.transversality_ratio < 1.0
    assert receipt.price_dividend_ratio > 0.0
    assert receipt.equilibrium_price == pytest.approx(receipt.price_dividend_ratio * 2.0, rel=1e-5)
    assert receipt.risk_free_gross_rate > 1.0
    assert receipt.expected_equity_gross_return > receipt.risk_free_gross_rate
    assert receipt.equity_premium > 0.0
    assert receipt.hansen_jagannathan_bound > 0.0


def test_calculate_lucas_tree_equity_premium_puzzle():
    """Verify classic Mehra & Prescott (1985) / Lucas (1978) Equity Premium Puzzle regime."""
    # Modest risk aversion gamma=2.0 and low consumption vol sigma=0.018 produce negligible theoretical premium (<1.0%)
    receipt = calculate_lucas_tree_equilibrium(
        current_dividend=3.0,
        subjective_discount_factor=0.96,
        relative_risk_aversion=2.0,
        dividend_growth_mean=0.018,
        dividend_growth_vol=0.018,
        ticker="AGG",
    )

    assert receipt.transversality_satisfied is True
    assert receipt.equity_premium < 0.010  # less than 1% premium
    assert receipt.regime == "EQUITY_PREMIUM_PUZZLE"


def test_calculate_lucas_tree_transversality_violation_bubble():
    """Verify transversality violation when dividend growth exceeds discounting (divergent infinite valuation)."""
    # Extremely high dividend growth drift mu=0.10 with high patience beta=0.999
    receipt = calculate_lucas_tree_equilibrium(
        current_dividend=1.0,
        subjective_discount_factor=0.999,
        relative_risk_aversion=0.5,  # 1 - gamma = 0.5 > 0, drift drives kappa > 1.0
        dividend_growth_mean=0.15,
        dividend_growth_vol=0.05,
        ticker="BUBBLE",
    )

    assert receipt.transversality_satisfied is False
    assert receipt.transversality_ratio >= 1.0
    assert receipt.price_dividend_ratio == 1000.0  # Clamped to upper bound
    assert receipt.regime == "TRANSVERSALITY_VIOLATION_BUBBLE"


def test_calculate_lucas_tree_high_risk_aversion_compression():
    """Verify high risk aversion regime (gamma >= 8.0)."""
    receipt = calculate_lucas_tree_equilibrium(
        current_dividend=2.0,
        subjective_discount_factor=0.95,
        relative_risk_aversion=10.0,
        dividend_growth_mean=0.01,
        dividend_growth_vol=0.03,
        ticker="DEF",
    )

    assert receipt.relative_risk_aversion == 10.0
    assert receipt.regime == "HIGH_RISK_AVERSION_COMPRESSION"
    assert receipt.hansen_jagannathan_bound > 0.30


def test_hansen_jagannathan_bound_properties():
    """Verify Hansen-Jagannathan maximal Sharpe ratio increases monotonically with gamma and sigma."""
    r1 = calculate_lucas_tree_equilibrium(relative_risk_aversion=2.0, dividend_growth_vol=0.03)
    r2 = calculate_lucas_tree_equilibrium(relative_risk_aversion=5.0, dividend_growth_vol=0.03)
    r3 = calculate_lucas_tree_equilibrium(relative_risk_aversion=5.0, dividend_growth_vol=0.06)

    assert r1.hansen_jagannathan_bound < r2.hansen_jagannathan_bound
    assert r2.hansen_jagannathan_bound < r3.hansen_jagannathan_bound


def test_mathematical_monotonicity_discount_factor_beta():
    """Verify patience monotonicity: higher beta increases price-dividend ratio V."""
    r_impatient = calculate_lucas_tree_equilibrium(subjective_discount_factor=0.92)
    r_patient = calculate_lucas_tree_equilibrium(subjective_discount_factor=0.97)

    assert r_patient.price_dividend_ratio > r_impatient.price_dividend_ratio
    assert r_patient.equilibrium_price > r_impatient.equilibrium_price


def test_mathematical_monotonicity_risk_free_rate_beta():
    """Verify interest rate monotonicity: higher patience beta lowers equilibrium gross risk-free rate R_f."""
    r_impatient = calculate_lucas_tree_equilibrium(subjective_discount_factor=0.92)
    r_patient = calculate_lucas_tree_equilibrium(subjective_discount_factor=0.97)

    assert r_patient.risk_free_gross_rate < r_impatient.risk_free_gross_rate


def test_security_input_validation():
    """Verify strict ticker regex allowlist validation and exception handling."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_lucas_tree_equilibrium(ticker="INVALID$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_lucas_tree_equilibrium(ticker="TOOLONGTICKERNAME")


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_lucas_tree_equilibrium(ticker="AAPL")

    sig_payload = (
        f"{receipt.ticker}:{receipt.subjective_discount_factor:.6f}:"
        f"{receipt.relative_risk_aversion:.6f}:{receipt.dividend_growth_mean:.6f}:"
        f"{receipt.dividend_growth_vol:.6f}:{receipt.price_dividend_ratio:.4f}:"
        f"{receipt.equity_premium:.6f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    # Tampering test
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_lucas_tree_profile_with_prices():
    """Verify empirical profile evaluation with historical price and dividend arrays."""
    np.random.seed(42)
    prices = [100.0 * (1.0 + 0.001 * i + 0.01 * np.random.randn()) for i in range(40)]
    dividends = [2.0 * (1.0 + 0.0005 * i + 0.005 * np.random.randn()) for i in range(40)]

    receipt = evaluate_lucas_tree_profile(
        prices=prices,
        dividends=dividends,
        ticker="MSFT",
        beta=0.96,
        gamma=2.5,
    )

    assert receipt.ticker == "MSFT"
    assert receipt.current_dividend > 0.0
    assert receipt.price_dividend_ratio > 0.0
    assert receipt.expected_equity_gross_return > 0.0


def test_persist_lucas_tree_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized query with integrity checks."""
    db_file = str(tmp_path / "test_lucas.db")
    receipt = calculate_lucas_tree_equilibrium(ticker="NVDA")

    success = persist_lucas_tree_record(receipt, db_path=db_file)
    assert success is True

    # Inspect persisted table
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, price_dividend_ratio, equity_premium, regime FROM lucas_tree_audits WHERE ticker = ?", ("NVDA",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "NVDA"
        assert row[1] == pytest.approx(receipt.price_dividend_ratio, rel=1e-4)
        assert row[2] == pytest.approx(receipt.equity_premium, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_lucas_tree_execution():
    """Verify CLI invocation for lucas-tree command in standard and JSON modes."""
    # Test human-readable table
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "lucas-tree",
        "--ticker",
        "AAPL",
        "--beta",
        "0.96",
        "--gamma",
        "2.5",
        "--growth-mean",
        "0.02",
        "--growth-vol",
        "0.04",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Robert E. Lucas Jr. (1978) Pure Exchange Tree Equilibrium" in res_table.stdout
    assert "Price-Dividend Ratio (P/D)" in res_table.stdout
    assert "AAPL" in res_table.stdout

    # Test JSON output
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "lucas-tree",
        "--ticker",
        "AAPL",
        "--beta",
        "0.96",
        "--gamma",
        "2.5",
        "--growth-mean",
        "0.02",
        "--growth-vol",
        "0.04",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "AAPL"
    assert "price_dividend_ratio" in parsed
    assert "hansen_jagannathan_bound" in parsed
    assert "hmac_signature" in parsed
