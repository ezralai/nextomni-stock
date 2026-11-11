"""Unit & CLI integration tests for Douglas W. Diamond (1991) Debt Maturity Structure Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.diamond_debt_maturity import (
    calculate_diamond_debt_maturity,
    evaluate_diamond_maturity_profile,
    persist_diamond_maturity_record,
    verify_hmac_signature,
)


def test_non_monotonic_debt_maturity_spectrum_short_long_short():
    """Diamond (1991) Proposition 1: AAA borrows Short, BBB borrows Long, High-Risk Monitored Short."""
    aaa = calculate_diamond_debt_maturity(
        prior_good_prob_f=0.95,
        investment_i=1.00,
        continuation_payoff_x=1.60,
        liquidation_value_l=0.70,
        control_rent_c=0.50,
        good_bad_news_prob_qg=0.20,
        bad_bad_news_prob_qb=0.85,
        ticker="AAA",
    )
    bbb = calculate_diamond_debt_maturity(
        prior_good_prob_f=0.72,
        investment_i=1.00,
        continuation_payoff_x=1.60,
        liquidation_value_l=0.65,
        control_rent_c=0.50,
        good_bad_news_prob_qg=0.25,
        bad_bad_news_prob_qb=0.80,
        ticker="BBB",
    )
    high_risk = calculate_diamond_debt_maturity(
        prior_good_prob_f=0.56,
        investment_i=1.00,
        continuation_payoff_x=1.60,
        liquidation_value_l=0.70,
        control_rent_c=0.50,
        good_bad_news_prob_qg=0.25,
        bad_bad_news_prob_qb=0.80,
        ticker="HY",
    )

    assert aaa.rollover_panic_triggered is False
    assert aaa.recommended_maturity == "SHORT_TERM"
    assert aaa.regime == "SHORT_TERM_HIGH_GRADE"

    assert bbb.rollover_panic_triggered is True
    assert bbb.long_term_feasible is True
    assert bbb.recommended_maturity == "LONG_TERM"
    assert bbb.regime == "LONG_TERM_ROLLOVER_SHIELD"

    assert high_risk.long_term_feasible is False
    assert high_risk.short_term_feasible is True
    assert high_risk.recommended_maturity == "SHORT_TERM"
    assert high_risk.regime == "MONITORED_SHORT_TERM_HIGH_RISK"


def test_exact_net_advantage_identity_under_rollover_panic():
    """Verify exact identity U_ST - U_LT == ((1-f)/f)*q_B*L - q_G*(X - L + C)."""
    rec = calculate_diamond_debt_maturity(
        prior_good_prob_f=0.72,
        investment_i=1.00,
        continuation_payoff_x=1.60,
        liquidation_value_l=0.65,
        control_rent_c=0.50,
        good_bad_news_prob_qg=0.25,
        bad_bad_news_prob_qb=0.80,
    )

    expected = ((0.28 / 0.72) * 0.80 * 0.65) - (0.25 * (1.60 - 0.65 + 0.50))
    assert rec.short_vs_long_net_advantage == pytest.approx(expected, rel=1e-9)
    assert rec.short_vs_long_net_advantage == pytest.approx(
        rec.upgrade_screening_gain - rec.rollover_liquidation_cost, rel=1e-9
    )


def test_higher_control_rent_or_lower_liquidation_value_pushes_toward_long_term():
    """Higher private control rent C increases rollover penalty and favors Long-Term debt."""
    low_c = calculate_diamond_debt_maturity(control_rent_c=0.05, prior_good_prob_f=0.72, good_bad_news_prob_qg=0.20)
    high_c = calculate_diamond_debt_maturity(control_rent_c=0.90, prior_good_prob_f=0.72, good_bad_news_prob_qg=0.20)

    assert high_c.rollover_liquidation_cost > low_c.rollover_liquidation_cost
    assert high_c.short_vs_long_net_advantage < low_c.short_vs_long_net_advantage
    assert high_c.recommended_maturity == "LONG_TERM"


def test_screening_dominant_short_term_when_rollover_loss_small():
    """When upgrade screening gain exceeds rollover loss, even risky borrower prefers Short-Term."""
    rec = calculate_diamond_debt_maturity(
        prior_good_prob_f=0.70,
        investment_i=1.00,
        continuation_payoff_x=1.50,
        liquidation_value_l=1.35,
        control_rent_c=0.10,
        good_bad_news_prob_qg=0.08,
        bad_bad_news_prob_qb=0.90,
    )
    assert rec.rollover_panic_triggered is True
    assert rec.short_vs_long_net_advantage > 0.0
    assert rec.regime == "SHORT_TERM_SCREENING_DOMINANT"


def test_unfundable_credit_rationing_when_both_maturities_fail():
    """Very low credit quality fails both short-term and long-term lender participation constraints."""
    rec = calculate_diamond_debt_maturity(
        prior_good_prob_f=0.25,
        investment_i=1.00,
        continuation_payoff_x=1.50,
        liquidation_value_l=0.40,
    )

    assert rec.short_term_feasible is False
    assert rec.long_term_feasible is False
    assert rec.recommended_maturity == "CREDIT_RATIONED"
    assert rec.regime == "UNFUNDABLE_BOND_CREDIT_RATIONING"


def test_bayesian_downgrade_posterior_fd_less_than_prior_f():
    """Adverse interim signal q_G < q_B strictly lowers posterior solvency rating fd < f."""
    rec = calculate_diamond_debt_maturity(prior_good_prob_f=0.75, good_bad_news_prob_qg=0.20, bad_bad_news_prob_qb=0.80)
    assert rec.downgrade_posterior_fd < rec.prior_good_prob_f


def test_security_input_validation():
    """Verify strict ticker allowlist, probability ordering, and payoff ordering 0 < L < X."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_diamond_debt_maturity(ticker="DM91;DROP")

    with pytest.raises(ValueError, match="prior_good_prob_f must satisfy"):
        calculate_diamond_debt_maturity(prior_good_prob_f=1.0)

    with pytest.raises(ValueError, match="Requires 0 < liquidation_value_l < continuation_payoff_x"):
        calculate_diamond_debt_maturity(liquidation_value_l=1.80, continuation_payoff_x=1.60)

    with pytest.raises(ValueError, match="Requires 0 <= q_g <= q_b <= 1"):
        calculate_diamond_debt_maturity(good_bad_news_prob_qg=0.85, bad_bad_news_prob_qb=0.40)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_diamond_debt_maturity(ticker="MSFT")
    sig_payload = (
        f"{rec.ticker}:{rec.prior_good_prob_f:.4f}:{rec.investment_i:.4f}:"
        f"{rec.continuation_payoff_x:.4f}:{rec.liquidation_value_l:.4f}:"
        f"{rec.control_rent_c:.4f}:{rec.downgrade_posterior_fd:.4f}:"
        f"{rec.short_vs_long_net_advantage:.6f}:{rec.recommended_maturity}:"
        f"{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_diamond_maturity_profile_from_prices():
    """Verify trailing volatility calibration of liquidation and bad-news signal risk."""
    prices = [100.0 + 0.25 * i + 2.0 * np.sin(i / 2.2) for i in range(40)]
    rec = evaluate_diamond_maturity_profile(prices=prices, ticker="ORCL")

    assert rec.ticker == "ORCL"
    assert 0.08 <= rec.good_bad_news_prob_qg <= 0.45
    assert 0.45 <= rec.liquidation_value_l <= 0.85


def test_persist_diamond_maturity_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_diamond_maturity.db")
    rec = calculate_diamond_debt_maturity(ticker="IBM")

    assert persist_diamond_maturity_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, recommended_maturity, rollover_liquidation_cost, regime FROM diamond_maturity_audits WHERE ticker = ?",
            ("IBM",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "IBM"
        assert row[1] == rec.recommended_maturity
        assert row[2] == pytest.approx(rec.rollover_liquidation_cost, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_debt_maturity_table_execution():
    """Verify CLI execution for debt-maturity command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "debt-maturity",
        "--ticker",
        "DM91",
        "--rating-f",
        "0.72",
        "--liq-l",
        "0.65",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Diamond (1991) Debt Maturity Structure & Rollover Liquidity Risk Engine" in res.stdout
    assert "DM91" in res.stdout
    assert "Recommended Maturity" in res.stdout


def test_cli_debt_maturity_json_execution():
    """Verify CLI execution for debt-maturity command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "debt-maturity",
        "--ticker",
        "F",
        "--rating-f",
        "0.95",
        "--qg",
        "0.20",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "F"
    assert parsed["recommended_maturity"] == "SHORT_TERM"
    assert parsed["rollover_panic_triggered"] is False
    assert "hmac_signature" in parsed

