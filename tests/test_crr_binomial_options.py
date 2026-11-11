"""Unit & CLI integration tests for Cox, Ross & Rubinstein (1979) Binomial Option Pricing Engine."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.crr_binomial_options import (
    calculate_crr_binomial_option,
    evaluate_binomial_option_profile,
    persist_binomial_option_record,
    verify_hmac_signature,
)


def test_american_put_early_exercise_premium_positive():
    """Deep ITM American put with positive interest carry must strictly beat European put."""
    receipt = calculate_crr_binomial_option(
        spot_price=80.0,
        strike_price=100.0,
        maturity_years=1.0,
        risk_free_rate=0.06,
        dividend_yield=0.0,
        volatility=0.25,
        steps=150,
        option_type="PUT",
        exercise_style="AMERICAN",
        ticker="SPY",
    )

    assert receipt.american_price > receipt.european_price
    assert receipt.early_exercise_premium > 0.25
    assert receipt.early_exercise_nodes > 0
    assert receipt.option_price == pytest.approx(receipt.american_price, rel=1e-9)
    assert -1.0 <= receipt.delta <= 0.0
    assert receipt.gamma >= 0.0


def test_american_call_zero_dividend_equals_european():
    """Merton no-early-exercise theorem: American call equals European call when q = 0 and r >= 0."""
    receipt = calculate_crr_binomial_option(
        spot_price=110.0,
        strike_price=100.0,
        maturity_years=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        volatility=0.30,
        steps=120,
        option_type="CALL",
        exercise_style="AMERICAN",
        ticker="AAPL",
    )

    assert receipt.early_exercise_premium == pytest.approx(0.0, abs=1e-9)
    assert receipt.early_exercise_nodes == 0
    assert receipt.american_price == pytest.approx(receipt.european_price, rel=1e-9)
    assert 0.0 <= receipt.delta <= 1.0


def test_american_call_high_dividend_triggers_early_exercise():
    """High continuous dividend yield q > r induces optimal early exercise for deep ITM calls."""
    receipt = calculate_crr_binomial_option(
        spot_price=135.0,
        strike_price=100.0,
        maturity_years=1.0,
        risk_free_rate=0.02,
        dividend_yield=0.09,
        volatility=0.22,
        steps=150,
        option_type="CALL",
        exercise_style="AMERICAN",
        ticker="T",
    )

    assert receipt.early_exercise_premium > 0.50
    assert receipt.early_exercise_nodes > 0
    assert receipt.american_price > receipt.european_price


def test_black_scholes_convergence_with_increasing_steps():
    """European CRR binomial prices converge cleanly to continuous Black-Scholes-Merton limit."""
    coarse = calculate_crr_binomial_option(
        spot_price=100.0,
        strike_price=100.0,
        maturity_years=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        volatility=0.25,
        steps=20,
        option_type="CALL",
        exercise_style="EUROPEAN",
        ticker="CONV",
    )
    fine = calculate_crr_binomial_option(
        spot_price=100.0,
        strike_price=100.0,
        maturity_years=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        volatility=0.25,
        steps=300,
        option_type="CALL",
        exercise_style="EUROPEAN",
        ticker="CONV",
    )

    assert fine.discretization_error_vs_bsm < coarse.discretization_error_vs_bsm
    assert fine.discretization_error_vs_bsm < 0.02


def test_european_put_call_parity_on_lattice():
    """European CRR call and put on identical grid must satisfy discrete Put-Call Parity."""
    s, k, t, r, q, sig, n = 105.0, 100.0, 0.75, 0.045, 0.015, 0.28, 120
    call_rec = calculate_crr_binomial_option(
        spot_price=s,
        strike_price=k,
        maturity_years=t,
        risk_free_rate=r,
        dividend_yield=q,
        volatility=sig,
        steps=n,
        option_type="CALL",
        exercise_style="EUROPEAN",
        ticker="PAR",
    )
    put_rec = calculate_crr_binomial_option(
        spot_price=s,
        strike_price=k,
        maturity_years=t,
        risk_free_rate=r,
        dividend_yield=q,
        volatility=sig,
        steps=n,
        option_type="PUT",
        exercise_style="EUROPEAN",
        ticker="PAR",
    )

    lhs = call_rec.european_price - put_rec.european_price
    rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_lattice_greeks_and_martingale_identities():
    """Verify u*d=1 recombining symmetry, 0 < p < 1 martingale bounds, and negative long option theta."""
    receipt = calculate_crr_binomial_option(
        spot_price=100.0,
        strike_price=100.0,
        maturity_years=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        volatility=0.25,
        steps=100,
        option_type="PUT",
        exercise_style="EUROPEAN",
        ticker="GRK",
    )

    assert receipt.up_factor_u * receipt.down_factor_d == pytest.approx(1.0, rel=1e-9)
    assert 0.0 < receipt.risk_neutral_prob_p < 1.0
    assert receipt.gamma > 0.0
    assert receipt.theta_annual < 0.0


def test_security_input_validation_and_arbitrage_guard():
    """Verify strict ticker allowlist, option/style allowlists, bounds, and arbitrage rejection."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_crr_binomial_option(ticker="DROP TABLE;")

    with pytest.raises(ValueError, match="Unsupported option_type"):
        calculate_crr_binomial_option(option_type="STRADDLE")

    with pytest.raises(ValueError, match="Unsupported exercise_style"):
        calculate_crr_binomial_option(exercise_style="BERMUDAN")

    with pytest.raises(ValueError, match="steps must be between 10 and 1000"):
        calculate_crr_binomial_option(steps=4)

    with pytest.raises(ValueError, match="No-arbitrage lattice condition violated"):
        calculate_crr_binomial_option(
            risk_free_rate=0.90,
            dividend_yield=0.0,
            volatility=0.05,
            maturity_years=1.0,
            steps=10,
        )


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature verification and tampering rejection."""
    receipt = calculate_crr_binomial_option(ticker="NVDA", option_type="PUT", exercise_style="AMERICAN")
    sig_payload = (
        f"{receipt.ticker}:{receipt.option_type}:{receipt.exercise_style}:"
        f"{receipt.spot_price:.4f}:{receipt.strike_price:.4f}:"
        f"{receipt.maturity_years:.4f}:{receipt.volatility:.4f}:{receipt.steps}:"
        f"{receipt.option_price:.6f}:{receipt.early_exercise_premium:.6f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_binomial_option_profile_from_prices():
    """Verify empirical spot and realized volatility extraction from trailing closes."""
    prices = [100.0 + 0.5 * i + 2.0 * np.sin(i / 3.0) for i in range(45)]
    receipt = evaluate_binomial_option_profile(
        prices=prices,
        ticker="MSFT",
        maturity_years=0.5,
        option_type="PUT",
        exercise_style="AMERICAN",
    )

    assert receipt.ticker == "MSFT"
    assert receipt.spot_price == pytest.approx(prices[-1], rel=1e-6)
    assert receipt.option_price > 0.0
    assert receipt.american_price >= receipt.european_price


def test_persist_binomial_option_record_sqlite(tmp_path):
    """Verify parameterized SQLite audit persistence for CRR binomial evaluations."""
    db_file = str(tmp_path / "test_crr_binomial.db")
    receipt = calculate_crr_binomial_option(ticker="QQQ", spot_price=90.0, strike_price=100.0)

    assert persist_binomial_option_record(receipt, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, option_type, exercise_style, option_price, early_exercise_premium FROM binomial_option_audits WHERE ticker = ?",
            ("QQQ",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "QQQ"
        assert row[1] == "PUT"
        assert row[2] == "AMERICAN"
        assert row[3] == pytest.approx(receipt.option_price, rel=1e-4)
        assert row[4] == pytest.approx(receipt.early_exercise_premium, rel=1e-4)


def test_cli_binomial_option_table_execution():
    """Verify CLI execution for binomial-option in formatted table output mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "binomial-option",
        "--ticker",
        "OPT",
        "--spot",
        "90",
        "--strike",
        "100",
        "--type",
        "PUT",
        "--style",
        "AMERICAN",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Cox, Ross & Rubinstein (1979) Binomial Option Pricing Engine" in res.stdout
    assert "OPT" in res.stdout
    assert "Early-Exercise Premium" in res.stdout


def test_cli_binomial_option_json_execution():
    """Verify CLI execution for binomial-option with --json output flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "binomial-option",
        "--ticker",
        "SPY",
        "--spot",
        "100",
        "--strike",
        "110",
        "--steps",
        "120",
        "--type",
        "PUT",
        "--style",
        "AMERICAN",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "SPY"
    assert parsed["option_type"] == "PUT"
    assert parsed["exercise_style"] == "AMERICAN"
    assert parsed["early_exercise_premium"] >= 0.0
    assert "hmac_signature" in parsed

