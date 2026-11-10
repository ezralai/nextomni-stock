"""Unit and integration tests for Stephen A. Ross (1977) Debt Incentive Signaling Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.ross_debt_signaling import (
    RossSignalingReceipt,
    calculate_ross_debt_signaling,
    evaluate_ross_signaling_profile,
    persist_ross_signaling_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_ross_incentive_compatibility_foc_and_optimal_debt():
    """Verify exact Ross (1977) FOC gamma_0 * dV_0/dF = gamma_1 * L / [(1 + r) * t*(F)]."""
    receipt = calculate_ross_debt_signaling(
        floor_quality_t_min=100.0,
        true_quality_t=160.0,
        debt_face_value=65.0,
        bankruptcy_penalty_l=60.0,
        gamma_0_weight=0.50,
        gamma_1_weight=0.50,
        risk_free_rate=0.04,
        ticker="ROSS",
    )

    assert isinstance(receipt, RossSignalingReceipt)
    assert receipt.ticker == "ROSS"
    lhs = receipt.gamma_0_weight * receipt.marginal_value_dv_df
    rhs = (receipt.gamma_1_weight * receipt.bankruptcy_penalty_l) / ((1.0 + receipt.risk_free_rate) * receipt.signaled_quality_t_star)
    assert lhs == pytest.approx(rhs, rel=1e-6)
    assert receipt.signaled_market_value > receipt.floor_market_value
    assert receipt.optimal_separating_debt_f_star == pytest.approx(65.0, rel=1e-6)
    assert receipt.mimicry_deterrence_gap == pytest.approx(0.0, abs=1e-8)
    assert receipt.regime == "TRUTHFUL_SEPARATING_DEBT_SIGNAL"


def test_false_mimicry_is_strictly_penalized():
    """Verify lower-quality firm falsely issuing excessive debt incurs strictly negative utility gap."""
    receipt = calculate_ross_debt_signaling(
        floor_quality_t_min=100.0,
        true_quality_t=110.0,   # Low quality (optimal debt F* ~ 8.75)
        debt_face_value=90.0,   # Excessive mimicry debt!
        bankruptcy_penalty_l=60.0,
        ticker="MIMIC",
    )

    assert receipt.signaled_quality_t_star > receipt.true_quality_t * 1.10
    assert receipt.mimicry_deterrence_gap > 0.0
    assert receipt.true_default_probability > receipt.signaled_default_probability
    assert receipt.regime == "OVER_SIGNALED_COSTLY_MIMICRY_PENALIZED" or receipt.regime == "EXCESS_LEVERAGE_BANKRUPTCY_TRAP"


def test_zero_debt_collapses_to_floor():
    """Verify zero debt face value signals baseline lemon floor valuation with zero default odds."""
    receipt = calculate_ross_debt_signaling(debt_face_value=0.0, ticker="ZERO")

    assert receipt.signaled_quality_t_star == receipt.floor_quality_t_min
    assert receipt.signaled_market_value == receipt.floor_market_value
    assert receipt.signaling_valuation_lift_pct == 0.0
    assert receipt.true_default_probability == 0.0
    assert receipt.regime == "UNLEVERED_LEMON_FLOOR_BASELINE"


def test_excess_leverage_bankruptcy_trap():
    """Verify heavy debt ratio exceeding 45% of quality cash flow triggers bankruptcy trap regime."""
    receipt = calculate_ross_debt_signaling(
        floor_quality_t_min=80.0,
        true_quality_t=100.0,
        debt_face_value=60.0,  # 60 / 100 = 60% >= 45%
        bankruptcy_penalty_l=50.0,
        ticker="TRAP",
    )

    assert receipt.true_default_probability >= 0.45
    assert receipt.regime == "EXCESS_LEVERAGE_BANKRUPTCY_TRAP"


def test_monotonicity_leverage_raises_market_valuation():
    """Verify issuing larger debt raises signaled market valuation (positive debt-value correlation)."""
    r_low = calculate_ross_debt_signaling(debt_face_value=20.0)
    r_mid = calculate_ross_debt_signaling(debt_face_value=50.0)
    r_high = calculate_ross_debt_signaling(debt_face_value=90.0)

    assert r_high.signaled_market_value > r_mid.signaled_market_value > r_low.signaled_market_value
    assert r_high.signaling_valuation_lift_pct > r_mid.signaling_valuation_lift_pct > r_low.signaling_valuation_lift_pct


def test_monotonicity_penalty_lowers_required_leverage():
    """Verify stiffer bankruptcy penalty reduces equilibrium separating debt F*(t_true) needed."""
    r_mild = calculate_ross_debt_signaling(bankruptcy_penalty_l=30.0)
    r_stiff = calculate_ross_debt_signaling(bankruptcy_penalty_l=90.0)

    assert r_stiff.optimal_separating_debt_f_star < r_mild.optimal_separating_debt_f_star


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positive finite quality floor check."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_ross_debt_signaling(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_ross_debt_signaling(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="floor_quality_t_min must be finite and positive"):
        calculate_ross_debt_signaling(floor_quality_t_min=-10.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_ross_debt_signaling(ticker="IBM")

    sig_payload = (
        f"{receipt.ticker}:{receipt.floor_quality_t_min:.4f}:{receipt.true_quality_t:.4f}:"
        f"{receipt.debt_face_value:.4f}:{receipt.bankruptcy_penalty_l:.4f}:"
        f"{receipt.signaled_market_value:.4f}:{receipt.optimal_separating_debt_f_star:.4f}:"
        f"{receipt.true_default_probability:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_ross_signaling_profile_with_prices():
    """Verify empirical calibration from historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 2.0 * np.sin(i / 2.5) for i in range(40)]

    receipt = evaluate_ross_signaling_profile(
        prices=prices,
        ticker="AAPL",
        debt_to_price_ratio=0.65,
        bankruptcy_penalty_ratio=0.60,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.signaled_market_value > receipt.floor_market_value
    assert receipt.debt_face_value == pytest.approx(float(prices[-1]) * 0.65, rel=1e-4)


def test_persist_ross_signaling_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_ross_sig.db")
    receipt = calculate_ross_debt_signaling(ticker="CAT")

    success = persist_ross_signaling_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, signaled_market_value, optimal_separating_debt_f_star, regime FROM ross_signaling_audits WHERE ticker = ?",
            ("CAT",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "CAT"
        assert row[1] == pytest.approx(receipt.signaled_market_value, rel=1e-4)
        assert row[2] == pytest.approx(receipt.optimal_separating_debt_f_star, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_debt_signaling_table_execution():
    """Verify CLI invocation for debt-signaling command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "debt-signaling",
        "--ticker",
        "ROSS",
        "--floor-t",
        "100.0",
        "--debt",
        "65.0",
        "--penalty",
        "60.0",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Ross (1977) Incentive Debt-Signaling Engine" in res_table.stdout
    assert "ROSS" in res_table.stdout
    assert "Signaled Market Valuation" in res_table.stdout


def test_cli_debt_signaling_json_execution():
    """Verify CLI invocation for debt-signaling command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "debt-signaling",
        "--ticker",
        "BA",
        "--debt",
        "65.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "BA"
    assert "signaled_market_value" in parsed
    assert "optimal_separating_debt_f_star" in parsed
    assert "hmac_signature" in parsed

