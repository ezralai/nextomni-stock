"""Unit and integration tests for Andrei Shleifer & Robert W. Vishny (1997) Limits of Arbitrage Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.shleifer_vishny_arbitrage import (
    ShleiferVishnyReceipt,
    calculate_shleifer_vishny_equilibrium,
    evaluate_shleifer_vishny_profile,
    persist_shleifer_vishny_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_calculate_shleifer_vishny_baseline():
    """Verify standard Shleifer-Vishny limits of arbitrage equilibrium."""
    receipt = calculate_shleifer_vishny_equilibrium(
        fundamental_value=100.0,
        initial_price=90.0,
        adverse_shock_price=80.0,
        flow_sensitivity=1.20,
        adverse_shock_prob=0.30,
        initial_capital=1000.0,
        ticker="ARB",
    )

    assert isinstance(receipt, ShleiferVishnyReceipt)
    assert receipt.ticker == "ARB"
    assert receipt.fundamental_value == 100.0
    assert receipt.initial_price == 90.0
    assert receipt.initial_mispricing_dollar == 10.0
    assert receipt.initial_mispricing_pct == pytest.approx(0.10, rel=1e-5)
    assert receipt.optimal_initial_position > 0.0
    assert receipt.cash_buffer > 0.0
    assert receipt.capital_deployed + receipt.cash_buffer == pytest.approx(1000.0, rel=1e-5)
    assert receipt.interim_loss_pct < 0.0
    assert receipt.redemption_amount > 0.0


def test_frictionless_proprietary_arbitrage_zero_flow_sensitivity():
    """Verify that when flow sensitivity a=0, arbitrageur fully deploys capital without redemption risk."""
    receipt = calculate_shleifer_vishny_equilibrium(
        fundamental_value=100.0,
        initial_price=90.0,
        adverse_shock_price=75.0,
        flow_sensitivity=0.0,  # Zero outside redemptions
        initial_capital=900.0,
        ticker="PROP",
    )

    assert receipt.flow_sensitivity == 0.0
    # 100% of capital deployed: 900 / 90 = 10 units
    assert receipt.optimal_initial_position == pytest.approx(10.0, rel=1e-5)
    assert receipt.cash_buffer == pytest.approx(0.0, abs=1e-5)
    assert receipt.redemption_amount == 0.0
    assert receipt.forced_liquidation_pct == 0.0
    assert receipt.regime == "FRICTIONLESS_PROPRIETARY_ARBITRAGE"


def test_performance_based_fire_sale_regime():
    """Verify performance-based fire-sale liquidation regime under severe shock and high flow sensitivity."""
    receipt = calculate_shleifer_vishny_equilibrium(
        fundamental_value=100.0,
        initial_price=90.0,
        adverse_shock_price=75.0,
        flow_sensitivity=1.80,
        adverse_shock_prob=0.30,
        initial_capital=1000.0,
        ticker="HEDGE",
    )

    assert receipt.forced_liquidation_pct >= 0.15
    assert receipt.regime == "PERFORMANCE_BASED_FIRE_SALE"


def test_capital_constrained_arbitrage_regime():
    """Verify capital-constrained arbitrage regime under persistent mispricing."""
    receipt = calculate_shleifer_vishny_equilibrium(
        fundamental_value=100.0,
        initial_price=85.0,
        adverse_shock_price=82.0,
        flow_sensitivity=0.50,
        adverse_shock_prob=0.10,
        initial_capital=1000.0,
        ticker="FUND",
    )

    assert receipt.residual_mispricing_ratio >= 0.10
    assert receipt.regime == "CAPITAL_CONSTRAINED_ARBITRAGE"


def test_mathematical_monotonicity_flow_sensitivity():
    """Verify that increasing flow sensitivity a curtails optimal position size (precautionary restraint)."""
    r_patient = calculate_shleifer_vishny_equilibrium(flow_sensitivity=0.20)
    r_med = calculate_shleifer_vishny_equilibrium(flow_sensitivity=1.00)
    r_skittish = calculate_shleifer_vishny_equilibrium(flow_sensitivity=2.50)

    assert r_patient.optimal_initial_position > r_med.optimal_initial_position
    assert r_med.optimal_initial_position > r_skittish.optimal_initial_position
    assert r_patient.cash_buffer < r_skittish.cash_buffer


def test_mathematical_monotonicity_adverse_shock_depth():
    """Verify that deeper interim price shocks decrease initial trade size and increase precautionary cash."""
    r_mild = calculate_shleifer_vishny_equilibrium(adverse_shock_price=88.0)
    r_severe = calculate_shleifer_vishny_equilibrium(adverse_shock_price=75.0)

    assert r_mild.optimal_initial_position > r_severe.optimal_initial_position
    assert r_mild.cash_buffer < r_severe.cash_buffer


def test_mathematical_monotonicity_capital_scale():
    """Verify that initial trade size scales linearly with initial capital."""
    r1 = calculate_shleifer_vishny_equilibrium(initial_capital=1000.0)
    r2 = calculate_shleifer_vishny_equilibrium(initial_capital=2000.0)

    assert r2.optimal_initial_position == pytest.approx(r1.optimal_initial_position * 2.0, rel=1e-5)
    assert r2.cash_buffer == pytest.approx(r1.cash_buffer * 2.0, rel=1e-5)


def test_security_input_validation():
    """Verify strict ticker regex allowlist validation and exception handling."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_shleifer_vishny_equilibrium(ticker="INVALID$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_shleifer_vishny_equilibrium(ticker="TOOLONGTICKERNAME")


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_shleifer_vishny_equilibrium(ticker="SPY")

    sig_payload = (
        f"{receipt.ticker}:{receipt.fundamental_value:.4f}:{receipt.initial_price:.4f}:"
        f"{receipt.adverse_shock_price:.4f}:{receipt.flow_sensitivity:.4f}:"
        f"{receipt.adverse_shock_prob:.4f}:{receipt.optimal_initial_position:.4f}:"
        f"{receipt.forced_liquidation_pct:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_shleifer_vishny_profile_with_prices():
    """Verify empirical profile evaluation with historical price series."""
    np.random.seed(42)
    prices = [100.0 + 0.1 * i + 2.0 * np.sin(i / 4.0) for i in range(35)]

    receipt = evaluate_shleifer_vishny_profile(
        prices=prices,
        ticker="QQQ",
        flow_sensitivity=1.20,
        initial_capital=1500.0,
    )

    assert receipt.ticker == "QQQ"
    assert receipt.fundamental_value >= receipt.initial_price
    assert receipt.optimal_initial_position > 0.0


def test_persist_shleifer_vishny_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_sv.db")
    receipt = calculate_shleifer_vishny_equilibrium(ticker="LTCM")

    success = persist_shleifer_vishny_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, optimal_initial_position, forced_liquidation_pct, regime FROM shleifer_vishny_audits WHERE ticker = ?", ("LTCM",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "LTCM"
        assert row[1] == pytest.approx(receipt.optimal_initial_position, rel=1e-4)
        assert row[2] == pytest.approx(receipt.forced_liquidation_pct, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_shleifer_vishny_execution():
    """Verify CLI invocation for shleifer-vishny command in standard and JSON modes."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "shleifer-vishny",
        "--ticker",
        "ARB",
        "--fundamental-val",
        "100.0",
        "--shock-severity",
        "0.15",
        "--flow-sensitivity",
        "1.50",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Shleifer & Vishny (1997) Limits of Arbitrage & PBA Engine" in res_table.stdout
    assert "ARB" in res_table.stdout
    assert "Forced Fire-Sale Liquidation" in res_table.stdout

    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "shleifer-vishny",
        "--ticker",
        "ARB",
        "--fundamental-val",
        "100.0",
        "--shock-severity",
        "0.15",
        "--flow-sensitivity",
        "1.50",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "ARB"
    assert "optimal_initial_position" in parsed
    assert "forced_liquidation_pct" in parsed
    assert "hmac_signature" in parsed
