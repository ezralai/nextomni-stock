"""Unit and integration tests for Harrison & Kreps (1978) Speculative Disagreement & Resale Option Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.harrison_kreps_resale import (
    HarrisonKrepsReceipt,
    calculate_harrison_kreps_equilibrium,
    evaluate_harrison_kreps_profile,
    persist_harrison_kreps_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_calculate_harrison_kreps_equilibrium_baseline():
    """Verify standard 2-state Harrison-Kreps equilibrium with positive resale option premia."""
    receipt = calculate_harrison_kreps_equilibrium(
        dividend_states=[1.0, 3.0],
        discount_factor=0.95,
        belief_divergence=0.20,
        ticker="TSLA",
    )

    assert isinstance(receipt, HarrisonKrepsReceipt)
    assert receipt.ticker == "TSLA"
    assert receipt.states_count == 2
    assert len(receipt.equilibrium_prices) == 2
    assert len(receipt.resale_option_premia) == 2

    # Harrison-Kreps Theorem: P*(s) >= V_max(s) for all s
    for s in range(2):
        assert receipt.equilibrium_prices[s] >= receipt.max_fundamental_val[s]
        assert receipt.resale_option_premia[s] >= 0.0

    assert receipt.mean_resale_premium > 0.0
    assert 0.0 < receipt.mean_bubble_ratio < 1.0


def test_mathematical_property_zero_divergence_homogeneous_beliefs():
    """Verify that when beliefs are homogeneous (delta=0), the resale option premium collapses to 0."""
    receipt = calculate_harrison_kreps_equilibrium(
        dividend_states=[1.0, 3.0],
        discount_factor=0.95,
        belief_divergence=0.0,  # Zero divergence: Pi_A == Pi_B
        ticker="SPY",
    )

    # In homogeneous belief markets, no speculative resale option exists
    assert receipt.mean_resale_premium == pytest.approx(0.0, abs=1e-5)
    assert receipt.mean_bubble_ratio == pytest.approx(0.0, abs=1e-5)
    for s in range(2):
        assert receipt.equilibrium_prices[s] == pytest.approx(receipt.max_fundamental_val[s], rel=1e-4)
    assert receipt.regime == "BELIEF_CONVERGENCE_ANCHORED"


def test_mathematical_monotonicity_belief_divergence():
    """Verify that increasing subjective belief divergence strictly increases the resale option premium."""
    r_low = calculate_harrison_kreps_equilibrium(belief_divergence=0.05)
    r_med = calculate_harrison_kreps_equilibrium(belief_divergence=0.15)
    r_high = calculate_harrison_kreps_equilibrium(belief_divergence=0.30)

    assert r_low.mean_resale_premium < r_med.mean_resale_premium
    assert r_med.mean_resale_premium < r_high.mean_resale_premium
    assert r_low.mean_bubble_ratio < r_high.mean_bubble_ratio


def test_speculative_bubble_cascade_regime():
    """Verify extreme speculative bubble cascade regime under wide belief disagreement."""
    receipt = calculate_harrison_kreps_equilibrium(
        dividend_states=[1.0, 4.0],
        discount_factor=0.98,
        belief_divergence=0.40,
        ticker="GME",
    )

    assert receipt.mean_bubble_ratio >= 0.15
    assert receipt.regime in ("SPECULATIVE_BUBBLE_CASCADE", "MODERATE_RESALE_PREMIUM")


def test_ownership_alternation_property():
    """Verify that in canonical Harrison-Kreps equilibrium, active owners alternate by state."""
    receipt = calculate_harrison_kreps_equilibrium(
        dividend_states=[1.0, 3.0],
        discount_factor=0.95,
        belief_divergence=0.20,
    )

    # State 0 (Low): Agent A is more optimistic about jumping to High state 1
    # State 1 (High): Agent B is more optimistic about staying in High state 1
    assert receipt.active_owner_states[0] == "Agent_A"
    assert receipt.active_owner_states[1] == "Agent_B"


def test_custom_transition_matrices():
    """Verify fixed-point convergence with custom asymmetric transition matrices."""
    pi_A = [
        [0.3, 0.7],
        [0.4, 0.6],
    ]
    pi_B = [
        [0.8, 0.2],
        [0.2, 0.8],
    ]
    receipt = calculate_harrison_kreps_equilibrium(
        dividend_states=[2.0, 5.0],
        transition_prob_A=pi_A,
        transition_prob_B=pi_B,
        discount_factor=0.92,
        ticker="CUST",
    )

    assert receipt.ticker == "CUST"
    assert receipt.equilibrium_prices[0] > 0
    assert receipt.equilibrium_prices[1] > receipt.equilibrium_prices[0]
    assert receipt.mean_resale_premium >= 0.0


def test_mathematical_monotonicity_discount_factor():
    """Verify that higher patience beta increases equilibrium prices in all states."""
    r_impatient = calculate_harrison_kreps_equilibrium(discount_factor=0.88)
    r_patient = calculate_harrison_kreps_equilibrium(discount_factor=0.96)

    for s in range(2):
        assert r_patient.equilibrium_prices[s] > r_impatient.equilibrium_prices[s]


def test_security_input_validation():
    """Verify strict ticker regex allowlist validation and exception handling."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_harrison_kreps_equilibrium(ticker="INVALID$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_harrison_kreps_equilibrium(ticker="TOOLONGTICKERNAME")


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_harrison_kreps_equilibrium(ticker="NVDA")

    sig_payload = (
        f"{receipt.ticker}:{receipt.discount_factor:.6f}:{receipt.belief_divergence:.6f}:"
        f"{receipt.mean_resale_premium:.6f}:{receipt.mean_bubble_ratio:.6f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_harrison_kreps_profile_with_prices():
    """Verify empirical profile evaluation with historical price arrays."""
    np.random.seed(42)
    prices = [50.0 + i * 0.5 + 2.0 * np.sin(i / 3.0) for i in range(30)]

    receipt = evaluate_harrison_kreps_profile(
        prices=prices,
        ticker="AAPL",
        discount_factor=0.95,
        belief_spread_override=0.18,
    )

    assert receipt.ticker == "AAPL"
    assert len(receipt.dividend_states) == 2
    assert receipt.equilibrium_prices[0] > 0.0
    assert receipt.mean_resale_premium >= 0.0


def test_persist_harrison_kreps_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries with JSON payloads."""
    db_file = str(tmp_path / "test_hk.db")
    receipt = calculate_harrison_kreps_equilibrium(ticker="COIN")

    success = persist_harrison_kreps_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, mean_resale_premium, mean_bubble_ratio, regime FROM harrison_kreps_audits WHERE ticker = ?", ("COIN",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "COIN"
        assert row[1] == pytest.approx(receipt.mean_resale_premium, rel=1e-4)
        assert row[2] == pytest.approx(receipt.mean_bubble_ratio, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_harrison_kreps_execution():
    """Verify CLI invocation for harrison-kreps command in standard and JSON modes."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "harrison-kreps",
        "--ticker",
        "TSLA",
        "--beta",
        "0.95",
        "--belief-spread",
        "0.20",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Harrison & Kreps (1978) Speculative Disagreement & Resale Option" in res_table.stdout
    assert "TSLA" in res_table.stdout
    assert "Resale Option Premium" in res_table.stdout

    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "harrison-kreps",
        "--ticker",
        "TSLA",
        "--beta",
        "0.95",
        "--belief-spread",
        "0.20",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "TSLA"
    assert "resale_option_premia" in parsed
    assert "mean_bubble_ratio" in parsed
    assert "hmac_signature" in parsed
