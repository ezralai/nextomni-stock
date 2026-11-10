"""Unit and integration tests for Leland & Pyle (1977) Insider Ownership Signaling Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.leland_pyle_signaling import (
    LelandPyleReceipt,
    calculate_leland_pyle_equilibrium,
    evaluate_leland_pyle_profile,
    persist_leland_pyle_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_incentive_compatibility_foc_identity():
    """Verify exact rational expectations FOC (1 - alpha) * dV/dalpha = K_lp * alpha."""
    receipt = calculate_leland_pyle_equilibrium(
        floor_value_mu_min=100.0,
        true_project_value=115.0,
        annual_volatility=0.40,
        relative_risk_aversion=3.0,
        insider_retention_alpha=0.55,
        ticker="SIGNAL",
    )

    assert isinstance(receipt, LelandPyleReceipt)
    assert receipt.ticker == "SIGNAL"
    lhs = (1.0 - receipt.insider_retention_alpha) * receipt.marginal_value_sensitivity
    rhs = receipt.signaling_scale_k * receipt.insider_retention_alpha
    assert lhs == pytest.approx(rhs, rel=1e-6)
    assert receipt.signaled_market_value > receipt.floor_value_mu_min


def test_zero_insider_retention_collapses_to_lemon_floor():
    """Verify alpha = 0 eliminates signaling premium and diversification cost."""
    receipt = calculate_leland_pyle_equilibrium(
        floor_value_mu_min=100.0,
        insider_retention_alpha=0.0,
        ticker="ZERO",
    )

    assert receipt.signaled_market_value == 100.0
    assert receipt.signaling_premium_dollar == 0.0
    assert receipt.diversification_risk_cost == 0.0
    assert receipt.marginal_value_sensitivity == 0.0
    assert receipt.regime == "UNSIGNALED_LEMON_FLOOR_VALUATION"


def test_exact_bisection_inversion_zero_mispricing():
    """Verify retaining quality_implied_alpha*(mu_true) reproduces mu_true identically."""
    base = calculate_leland_pyle_equilibrium(
        floor_value_mu_min=100.0,
        true_project_value=120.0,
        annual_volatility=0.40,
        relative_risk_aversion=3.0,
        ticker="INV",
    )
    at_eq = calculate_leland_pyle_equilibrium(
        floor_value_mu_min=100.0,
        true_project_value=120.0,
        annual_volatility=0.40,
        relative_risk_aversion=3.0,
        insider_retention_alpha=base.quality_implied_alpha,
        ticker="INV",
    )

    assert at_eq.signaled_market_value == pytest.approx(120.0, rel=1e-4)
    assert abs(at_eq.market_mispricing_wedge) < 0.05


def test_insufficient_skin_in_game_discount():
    """Verify under-signaling (alpha << quality_implied_alpha) triggers discount warning."""
    receipt = calculate_leland_pyle_equilibrium(
        floor_value_mu_min=100.0,
        true_project_value=135.0,
        annual_volatility=0.40,
        relative_risk_aversion=3.0,
        insider_retention_alpha=0.20,  # Far below implied retention for 135!
        ticker="UNDER",
    )

    assert receipt.market_mispricing_wedge < 0.0
    assert receipt.regime == "INSUFFICIENT_SKIN_IN_GAME_DISCOUNTED"


def test_monotonicity_insider_retention_raises_valuation_and_cost():
    """Verify increasing insider retention alpha strictly raises signaled valuation and risk cost."""
    r_low = calculate_leland_pyle_equilibrium(insider_retention_alpha=0.25)
    r_mid = calculate_leland_pyle_equilibrium(insider_retention_alpha=0.50)
    r_high = calculate_leland_pyle_equilibrium(insider_retention_alpha=0.75)

    assert r_high.signaled_market_value > r_mid.signaled_market_value > r_low.signaled_market_value
    assert r_high.diversification_risk_cost > r_mid.diversification_risk_cost > r_low.diversification_risk_cost
    assert r_high.marginal_value_sensitivity > r_mid.marginal_value_sensitivity > r_low.marginal_value_sensitivity


def test_intermediary_coalition_cost_threshold():
    """Verify high signaling risk cost flags justification for financial intermediary screening."""
    receipt = calculate_leland_pyle_equilibrium(
        floor_value_mu_min=100.0,
        true_project_value=110.0,
        annual_volatility=0.50,
        relative_risk_aversion=4.0,
        insider_retention_alpha=0.60,
        ticker="BANK",
    )

    assert receipt.diversification_risk_cost > 10.0
    assert receipt.intermediary_savings_threshold == receipt.diversification_risk_cost
    assert receipt.regime == "HIGH_SIGNALING_COST_INTERMEDIARY_JUSTIFIED"


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positivity validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_leland_pyle_equilibrium(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_leland_pyle_equilibrium(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="floor_value_mu_min must be finite and positive"):
        calculate_leland_pyle_equilibrium(floor_value_mu_min=-20.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_leland_pyle_equilibrium(ticker="PLTR")

    sig_payload = (
        f"{receipt.ticker}:{receipt.floor_value_mu_min:.4f}:{receipt.true_project_value:.4f}:"
        f"{receipt.annual_volatility:.4f}:{receipt.relative_risk_aversion:.4f}:"
        f"{receipt.insider_retention_alpha:.4f}:{receipt.signaled_market_value:.4f}:"
        f"{receipt.diversification_risk_cost:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_leland_pyle_profile_with_prices():
    """Verify empirical calibration from historical price series."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 2.0 * np.sin(i / 2.5) for i in range(45)]

    receipt = evaluate_leland_pyle_profile(
        prices=prices,
        ticker="ARM",
        insider_retention_alpha=0.65,
        relative_risk_aversion=3.0,
    )

    assert receipt.ticker == "ARM"
    assert receipt.signaled_market_value > receipt.floor_value_mu_min


def test_persist_leland_pyle_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_lp.db")
    receipt = calculate_leland_pyle_equilibrium(ticker="SNOW")

    success = persist_leland_pyle_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, signaled_market_value, diversification_risk_cost, regime FROM leland_pyle_audits WHERE ticker = ?",
            ("SNOW",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "SNOW"
        assert row[1] == pytest.approx(receipt.signaled_market_value, rel=1e-4)
        assert row[2] == pytest.approx(receipt.diversification_risk_cost, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_leland_pyle_table_execution():
    """Verify CLI invocation for leland-pyle command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "leland-pyle",
        "--ticker",
        "SIGNAL",
        "--floor-value",
        "100.0",
        "--alpha",
        "0.55",
        "--volatility",
        "0.40",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Leland & Pyle (1977) Insider Ownership Signaling Engine" in res_table.stdout
    assert "SIGNAL" in res_table.stdout
    assert "Signaled Market Valuation" in res_table.stdout


def test_cli_leland_pyle_json_execution():
    """Verify CLI invocation for leland-pyle command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "leland-pyle",
        "--ticker",
        "IPO",
        "--alpha",
        "0.60",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "IPO"
    assert "signaled_market_value" in parsed
    assert "diversification_risk_cost" in parsed
    assert "hmac_signature" in parsed

