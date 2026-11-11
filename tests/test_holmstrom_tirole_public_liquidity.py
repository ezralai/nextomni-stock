"""Unit & CLI integration tests for Holmström & Tirole (1998) Private & Public Supply of Liquidity."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.holmstrom_tirole_public_liquidity import (
    calculate_public_liquidity_supply,
    evaluate_public_liquidity_profile,
    persist_public_liquidity_record,
    verify_hmac_signature,
)


def test_idiosyncratic_shocks_eliminated_by_private_pooling_zero_treasury_premium():
    """When aggregate_corr_lambda = 0, intermediaries pool risks cleanly with q_L = 1.0."""
    rec = calculate_public_liquidity_supply(
        interim_shock_rho=0.95,
        pledgeable_return_rho0=0.70,
        aggregate_corr_lambda=0.0,
        ticker="IDIO",
    )

    assert rec.treasury_liquidity_price_ql == pytest.approx(1.0, rel=1e-9)
    assert rec.liquidity_shortage_premium_bps == pytest.approx(0.0, abs=1e-9)
    assert rec.aggregate_investment_scale == pytest.approx(rec.pooled_investment_scale, rel=1e-9)
    assert rec.pooled_investment_scale > rec.hoarding_investment_scale
    assert rec.public_liquidity_scale_gain == pytest.approx(0.0, abs=1e-9)
    assert rec.regime == "PRIVATE_INTERMEDIARY_POOLING_SECOND_BEST"


def test_aggregate_shocks_trigger_treasury_premium_and_scale_loss():
    """Aggregate correlated liquidity shocks push Treasury price q_L > 1 and depress scale."""
    rec = calculate_public_liquidity_supply(
        interim_shock_rho=0.95,
        pledgeable_return_rho0=0.70,
        aggregate_corr_lambda=0.80,
        ticker="CRISIS",
    )

    assert rec.treasury_liquidity_price_ql > 1.0
    assert rec.liquidity_shortage_premium_bps > 0.0
    assert rec.pooled_investment_scale > rec.aggregate_investment_scale > rec.hoarding_investment_scale
    assert rec.public_liquidity_scale_gain > 0.0
    assert rec.welfare_gain_public_supply > 0.0
    assert rec.regime == "SEVERE_AGGREGATE_LIQUIDITY_SHORTAGE_PUBLIC_TREASURY_NEEDED"


def test_self_liquid_regime_when_pledgeable_income_covers_shock():
    """When interim_shock <= rho0, firm self-funds shocks with zero Treasury liquidity premium."""
    rec = calculate_public_liquidity_supply(
        interim_shock_rho=0.60,
        pledgeable_return_rho0=0.70,
        aggregate_corr_lambda=0.90,
        ticker="SELF",
    )

    assert rec.pledgeable_liquidity_gap == 0.0
    assert rec.treasury_liquidity_price_ql == 1.0
    assert rec.pooled_investment_scale == pytest.approx(rec.hoarding_investment_scale, rel=1e-9)
    assert rec.regime == "SELF_LIQUID_PLEDGEABLE_INCOME_SUFFICIENT"


def test_ex_post_liquidation_efficient_when_shock_exceeds_total_return():
    """When interim reinvestment need exceeds total return rho1, rescue is inefficient."""
    rec = calculate_public_liquidity_supply(
        interim_shock_rho=1.60,
        total_return_rho1=1.40,
        pledgeable_return_rho0=0.70,
        ticker="ABND",
    )

    assert rec.continuation_efficient is False
    assert rec.treasury_liquidity_price_ql == 1.0
    assert rec.regime == "EX_POST_LIQUIDATION_EFFICIENT_SHOCK_EXCEEDS_RHO1"


def test_higher_shock_severity_widens_treasury_liquidity_premium():
    """Larger pledgeable gaps widen Treasury liquidity premiums and contingent public gains."""
    mild = calculate_public_liquidity_supply(interim_shock_rho=0.80, aggregate_corr_lambda=0.60)
    severe = calculate_public_liquidity_supply(interim_shock_rho=1.15, aggregate_corr_lambda=0.60)

    assert severe.treasury_liquidity_price_ql > mild.treasury_liquidity_price_ql
    assert severe.liquidity_shortage_premium_bps > mild.liquidity_shortage_premium_bps
    assert severe.public_liquidity_scale_gain > mild.public_liquidity_scale_gain


def test_moderate_aggregate_shortage_classification():
    """Verify moderate aggregate liquidity shortage regime boundary."""
    rec = calculate_public_liquidity_supply(
        interim_shock_rho=0.85,
        pledgeable_return_rho0=0.70,
        total_return_rho1=1.45,
        shock_prob_theta=0.30,
        aggregate_corr_lambda=0.40,
    )
    assert rec.regime == "MODERATE_AGGREGATE_LIQUIDITY_PREMIUM"


def test_security_input_validation():
    """Verify strict ticker allowlist, rho1 > rho0 ordering, and probability bounds."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_public_liquidity_supply(ticker="HT98;DROP")

    with pytest.raises(ValueError, match="strictly exceed pledgeable_return_rho0"):
        calculate_public_liquidity_supply(pledgeable_return_rho0=1.20, total_return_rho1=1.10)

    with pytest.raises(ValueError, match="aggregate_corr_lambda"):
        calculate_public_liquidity_supply(aggregate_corr_lambda=1.25)

    with pytest.raises(ValueError, match="shock_prob_theta"):
        calculate_public_liquidity_supply(shock_prob_theta=0.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_public_liquidity_supply(ticker="JPM")
    sig_payload = (
        f"{rec.ticker}:{rec.entrepreneur_equity_a:.4f}:{rec.pledgeable_return_rho0:.4f}:"
        f"{rec.total_return_rho1:.4f}:{rec.interim_shock_rho:.4f}:{rec.aggregate_corr_lambda:.4f}:"
        f"{rec.aggregate_investment_scale:.4f}:{rec.treasury_liquidity_price_ql:.6f}:"
        f"{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_public_liquidity_profile_from_prices():
    """Verify empirical liquidity shock calibration from market volatility and drawdowns."""
    prices = [100.0 + 0.25 * i + 2.5 * np.sin(i / 2.2) for i in range(40)]
    rec = evaluate_public_liquidity_profile(prices=prices, ticker="GS")

    assert rec.ticker == "GS"
    assert 0.65 <= rec.interim_shock_rho <= 1.35
    assert rec.aggregate_investment_scale > 0.0


def test_persist_public_liquidity_record_sqlite(tmp_path):
    """Verify parameterized SQLite audit persistence for HT (1998) liquidity records."""
    db_file = str(tmp_path / "test_ht98_liquidity.db")
    rec = calculate_public_liquidity_supply(ticker="C")

    assert persist_public_liquidity_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, treasury_liquidity_price_ql, public_liquidity_scale_gain, regime FROM ht1998_public_liquidity_audits WHERE ticker = ?",
            ("C",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "C"
        assert row[1] == pytest.approx(rec.treasury_liquidity_price_ql, rel=1e-4)
        assert row[2] == pytest.approx(rec.public_liquidity_scale_gain, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_public_liquidity_table_execution():
    """Verify CLI execution for public-liquidity command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "public-liquidity",
        "--ticker",
        "HT98",
        "--shock",
        "0.95",
        "--lambda-agg",
        "0.75",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Holmstrom & Tirole (1998) Private & Public Supply of Liquidity Engine" in res.stdout
    assert "HT98" in res.stdout
    assert "Treasury Liquidity q_L" in res.stdout


def test_cli_public_liquidity_json_execution():
    """Verify CLI execution for public-liquidity command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "public-liquidity",
        "--ticker",
        "BAC",
        "--shock",
        "1.00",
        "--lambda-agg",
        "0.80",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "BAC"
    assert parsed["treasury_liquidity_price_ql"] > 1.0
    assert parsed["public_liquidity_scale_gain"] > 0.0
    assert "hmac_signature" in parsed

