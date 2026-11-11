"""Unit & CLI integration tests for Blanchard & Watson (1982) Rational Speculative Bubbles."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.blanchard_watson_bubble import (
    calculate_blanchard_watson_bubble,
    evaluate_blanchard_watson_profile,
    infer_implied_survival_prob,
    persist_blanchard_watson_record,
    verify_hmac_signature,
)


def test_euler_martingale_expectation_identity_exact():
    """Rational bubble B_t must satisfy exact Euler condition E_t[B_{t+1}] == (1 + r) * B_t."""
    rec = calculate_blanchard_watson_bubble(
        observed_price_p=160.0,
        fundamental_value_f=100.0,
        discount_rate_r=0.06,
        survival_prob_pi=0.80,
        ticker="EULR",
    )

    assert rec.bubble_component_b == pytest.approx(60.0, rel=1e-9)
    assert rec.expected_next_bubble == pytest.approx(1.06 * 60.0, rel=1e-9)
    assert rec.survival_prob_pi * rec.next_surviving_bubble == pytest.approx(rec.expected_next_bubble, rel=1e-9)


def test_expected_total_asset_return_equals_discount_rate():
    """Verify E_t[P_{t+1}] / P_t - 1 == discount_rate_r despite explosive surviving bubble path."""
    p, f, r, pi = 200.0, 100.0, 0.05, 0.75
    rec = calculate_blanchard_watson_bubble(
        observed_price_p=p,
        fundamental_value_f=f,
        discount_rate_r=r,
        survival_prob_pi=pi,
    )

    surviving_price = (1.0 + r) * f + rec.next_surviving_bubble
    burst_price = (1.0 + r) * f
    expected_price = pi * surviving_price + (1.0 - pi) * burst_price
    expected_return = (expected_price / p) - 1.0

    assert rec.surviving_asset_return > r
    assert expected_return == pytest.approx(r, rel=1e-9)


def test_lower_survival_probability_forces_higher_explosive_bubble_growth():
    """Higher crash hazard (lower pi) requires faster explosive appreciation while bubble survives."""
    calm = calculate_blanchard_watson_bubble(survival_prob_pi=0.92, observed_price_p=150.0, fundamental_value_f=100.0)
    fragile = calculate_blanchard_watson_bubble(survival_prob_pi=0.65, observed_price_p=150.0, fundamental_value_f=100.0)

    assert fragile.required_bubble_growth > calm.required_bubble_growth
    assert fragile.surviving_asset_return > calm.surviving_asset_return
    assert fragile.expected_remaining_periods < calm.expected_remaining_periods


def test_fundamental_equilibrium_when_price_at_or_below_fundamental():
    """When observed_price <= fundamental_value, bubble B_t = 0 with zero crash risk."""
    rec = calculate_blanchard_watson_bubble(
        observed_price_p=95.0,
        fundamental_value_f=100.0,
        survival_prob_pi=0.70,
        ticker="VALU",
    )

    assert rec.bubble_component_b == 0.0
    assert rec.bubble_share_pct == 0.0
    assert rec.burst_drawdown_loss_pct == 0.0
    assert rec.crash_prob == 0.0
    assert rec.regime == "FUNDAMENTAL_EQUILIBRIUM_NO_BUBBLE"


def test_extreme_overvaluation_imminent_burst_regime():
    """Bubble >= 50% of asset price with crash probability >= 20% flags imminent burst hazard."""
    rec = calculate_blanchard_watson_bubble(
        observed_price_p=250.0,
        fundamental_value_f=100.0,
        survival_prob_pi=0.75,
        ticker="TULP",
    )

    assert rec.bubble_share_pct == pytest.approx(60.0, rel=1e-9)
    assert rec.burst_drawdown_loss_pct == pytest.approx(60.0, rel=1e-9)
    assert rec.regime == "IMMINENT_BURST_EXTREME_OVERVALUATION"


def test_explosive_and_mild_bubble_classifications():
    """Verify mild drift and critical explosive bubble regimes cleanly separate."""
    mild = calculate_blanchard_watson_bubble(
        observed_price_p=115.0, fundamental_value_f=100.0, discount_rate_r=0.05, survival_prob_pi=0.92
    )
    assert mild.regime == "MILD_RATIONAL_BUBBLE_DRIFT"

    explosive = calculate_blanchard_watson_bubble(
        observed_price_p=160.0, fundamental_value_f=100.0, discount_rate_r=0.05, survival_prob_pi=0.72
    )
    assert explosive.regime == "EXPLOSIVE_BUBBLE_CRITICAL_HAZARD"


def test_certain_survival_deterministic_bubble_limit():
    """At pi = 1.0, expected remaining periods cap safely without zero-division crash."""
    rec = calculate_blanchard_watson_bubble(
        observed_price_p=140.0,
        fundamental_value_f=100.0,
        discount_rate_r=0.05,
        survival_prob_pi=1.0,
    )
    assert rec.crash_prob == 0.0
    assert rec.required_bubble_growth == pytest.approx(0.05, rel=1e-9)
    assert rec.expected_remaining_periods == 100.0


def test_infer_implied_survival_prob_inversion():
    """Implied survival probability declines monotonically as explosive run-up g accelerates."""
    pi_mod = infer_implied_survival_prob(observed_growth_g=0.20, discount_rate_r=0.05)
    pi_fast = infer_implied_survival_prob(observed_growth_g=0.80, discount_rate_r=0.05)

    assert pi_mod == pytest.approx(1.05 / 1.20, rel=1e-9)
    assert pi_fast < pi_mod
    assert 0.15 <= pi_fast <= 0.99


def test_security_input_validation():
    """Verify strict ticker allowlist and probability/price bounds."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_blanchard_watson_bubble(ticker="BW82;DROP")

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_blanchard_watson_bubble(observed_price_p=-10.0)

    with pytest.raises(ValueError, match="discount_rate_r must be strictly positive"):
        calculate_blanchard_watson_bubble(discount_rate_r=0.0)

    with pytest.raises(ValueError, match="survival_prob_pi must satisfy"):
        calculate_blanchard_watson_bubble(survival_prob_pi=0.05)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_blanchard_watson_bubble(ticker="NVDA")
    sig_payload = (
        f"{rec.ticker}:{rec.observed_price_p:.4f}:{rec.fundamental_value_f:.4f}:"
        f"{rec.discount_rate_r:.6f}:{rec.survival_prob_pi:.4f}:"
        f"{rec.bubble_component_b:.4f}:{rec.bubble_share_pct:.2f}:"
        f"{rec.required_bubble_growth:.6f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_blanchard_watson_profile_detects_parabolic_runup():
    """Exponential run-up series generates positive detected bubble share and burst hazard."""
    parabolic_prices = [100.0 * (1.02 ** i) for i in range(45)]
    rec = evaluate_blanchard_watson_profile(prices=parabolic_prices, ticker="MEME")

    assert rec.ticker == "MEME"
    assert rec.bubble_component_b > 0.0
    assert rec.bubble_share_pct > 15.0
    assert rec.crash_prob > 0.0


def test_persist_blanchard_watson_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_blanchard_watson.db")
    rec = calculate_blanchard_watson_bubble(ticker="BTC")

    assert persist_blanchard_watson_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, bubble_component_b, crash_prob, regime FROM blanchard_watson_audits WHERE ticker = ?",
            ("BTC",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "BTC"
        assert row[1] == pytest.approx(rec.bubble_component_b, rel=1e-4)
        assert row[2] == pytest.approx(rec.crash_prob, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_rational_bubble_table_execution():
    """Verify CLI execution for rational-bubble command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "rational-bubble",
        "--ticker",
        "BW82",
        "--price",
        "150",
        "--fundamental",
        "100",
        "--pi",
        "0.85",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Blanchard & Watson (1982) Rational Speculative Bubbles Engine" in res.stdout
    assert "BW82" in res.stdout
    assert "Required Bubble Growth" in res.stdout


def test_cli_rational_bubble_json_execution():
    """Verify CLI execution for rational-bubble command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "rational-bubble",
        "--ticker",
        "TSLA",
        "--price",
        "220",
        "--fundamental",
        "100",
        "--pi",
        "0.75",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "TSLA"
    assert parsed["bubble_component_b"] == pytest.approx(120.0)
    assert parsed["crash_prob"] == pytest.approx(0.25)
    assert "hmac_signature" in parsed

