"""Unit & CLI integration tests for Amihud & Mendelson (1986) Clientele Asset Pricing Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.amihud_mendelson_clientele import (
    calculate_amihud_mendelson_clientele,
    evaluate_amihud_mendelson_profile,
    persist_amihud_mendelson_record,
    verify_hmac_signature,
)


def test_piecewise_linear_concave_spread_return_schedule():
    """Gross expected return R(S) is strictly increasing and discrete-concave across spread tiers."""
    s1 = calculate_amihud_mendelson_clientele(proportional_spread_s=0.004, ticker="LIQ")
    s2 = calculate_amihud_mendelson_clientele(proportional_spread_s=0.015, ticker="MID")
    s3 = calculate_amihud_mendelson_clientele(proportional_spread_s=0.045, ticker="ILLQ")

    assert s1.gross_expected_return < s2.gross_expected_return < s3.gross_expected_return
    slope_12 = (s2.gross_expected_return - s1.gross_expected_return) / (0.015 - 0.004)
    slope_23 = (s3.gross_expected_return - s2.gross_expected_return) / (0.045 - 0.015)

    assert slope_23 < slope_12
    assert s1.equilibrium_clientele_tier == "TIER1_SHORT_HORIZON_LIQUID_TRADER"
    assert s2.equilibrium_clientele_tier == "TIER2_INSTITUTIONAL_MEDIUM_HORIZON"
    assert s3.equilibrium_clientele_tier == "TIER3_PATIENT_ENDOWMENT_ILLIQUID_VALUE"


def test_patient_investor_captures_positive_net_illiquidity_rent():
    """Long-horizon investor (low turnover mu) captures positive net alpha in wide-spread assets."""
    patient = calculate_amihud_mendelson_clientele(
        proportional_spread_s=0.035,
        investor_turnover_mu=0.40,
        ticker="ENDW",
    )
    assert patient.clientele_net_alpha_bps > 50.0
    assert patient.investor_net_return > patient.base_liquid_return_r0
    assert patient.regime == "PATIENT_CAPITAL_ILLIQUIDITY_RENT_CAPTURE"


def test_short_horizon_trader_loses_money_in_wide_spread_asset():
    """High-turnover trader holding illiquid wide-spread asset suffers severe clientele mismatch."""
    hft_mismatch = calculate_amihud_mendelson_clientele(
        proportional_spread_s=0.035,
        investor_turnover_mu=4.0,
        ticker="HFT",
    )
    assert hft_mismatch.investor_net_return < hft_mismatch.base_liquid_return_r0
    assert hft_mismatch.clientele_net_alpha_bps < 0.0
    assert hft_mismatch.regime == "CLIENTELE_MISMATCH_OVERTRADING_DRAG"


def test_zero_spread_recovers_base_liquid_return():
    """Zero bid-ask spread returns exact base liquid cost of equity r0 with zero illiquidity prem."""
    rec = calculate_amihud_mendelson_clientele(proportional_spread_s=0.0, base_liquid_return_r0=0.07)
    assert rec.gross_expected_return == pytest.approx(0.07, rel=1e-9)
    assert rec.illiquidity_premium_bps == 0.0
    assert rec.regime == "ZERO_SPREAD_FRICTIONLESS_BENCHMARK"


def test_spread_compression_creates_positive_valuation_rerating():
    """Moving from wider illiquid spread toward tight spreads lowers cost of capital and boosts price."""
    med = calculate_amihud_mendelson_clientele(proportional_spread_s=0.015)
    wide = calculate_amihud_mendelson_clientele(proportional_spread_s=0.050)

    assert wide.spread_compression_upside_pct > med.spread_compression_upside_pct > 0.0


def test_matched_clientele_horizon_equilibrium():
    """An investor trading exactly at marginal cohort speed in Tier 1 sits at matched equilibrium."""
    rec = calculate_amihud_mendelson_clientele(
        proportional_spread_s=0.004,
        investor_turnover_mu=4.0,
    )
    assert rec.clientele_net_alpha_bps == pytest.approx(0.0, abs=1e-9)
    assert rec.regime == "MATCHED_HORIZON_CLIENTELE_EQUILIBRIUM"


def test_holding_period_reciprocal_identity():
    """Verify holding_years == 1.0 / turnover_mu for both investor and marginal clientele."""
    rec = calculate_amihud_mendelson_clientele(proportional_spread_s=0.030, investor_turnover_mu=0.5)
    assert rec.investor_holding_years == pytest.approx(2.0, rel=1e-9)
    assert rec.optimal_clientele_holding_years == pytest.approx(2.5, rel=1e-9)


def test_wide_spread_moderate_turnover_equilibrium_regime():
    """Investor matching Tier-3 wide-spread marginal turnover stays within matched/endowment bounds."""
    rec = calculate_amihud_mendelson_clientele(
        proportional_spread_s=0.030,
        investor_turnover_mu=1.50,
    )
    assert rec.regime == "WIDE_SPREAD_PATIENT_ENDOWMENT_EQUILIBRIUM"


def test_security_input_validation():
    """Verify strict ticker allowlist and valid bounds on spread, base return, and turnover."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_amihud_mendelson_clientele(ticker="AM86;DROP")

    with pytest.raises(ValueError, match="proportional_spread_s must satisfy"):
        calculate_amihud_mendelson_clientele(proportional_spread_s=0.40)

    with pytest.raises(ValueError, match="base_liquid_return_r0 must be strictly positive"):
        calculate_amihud_mendelson_clientele(base_liquid_return_r0=0.0)

    with pytest.raises(ValueError, match="investor_turnover_mu must be strictly positive"):
        calculate_amihud_mendelson_clientele(investor_turnover_mu=-1.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_amihud_mendelson_clientele(ticker="IWM")
    sig_payload = (
        f"{rec.ticker}:{rec.proportional_spread_s:.6f}:{rec.base_liquid_return_r0:.6f}:"
        f"{rec.investor_turnover_mu:.4f}:{rec.gross_expected_return:.6f}:"
        f"{rec.illiquidity_premium_bps:.2f}:{rec.investor_net_return:.6f}:"
        f"{rec.equilibrium_clientele_tier}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_amihud_mendelson_profile_from_prices():
    """Verify effective spread extraction and clientele pricing from market closes."""
    prices = [100.0 + 0.2 * i + 1.8 * np.sin(i * 1.3) for i in range(45)]
    rec = evaluate_amihud_mendelson_profile(prices=prices, ticker="MDY")

    assert rec.ticker == "MDY"
    assert 0.001 <= rec.proportional_spread_s <= 0.080
    assert rec.gross_expected_return > rec.base_liquid_return_r0


def test_persist_amihud_mendelson_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_amihud_mendelson.db")
    rec = calculate_amihud_mendelson_clientele(ticker="XBI")

    assert persist_amihud_mendelson_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, illiquidity_premium_bps, equilibrium_clientele_tier, regime FROM amihud_mendelson_audits WHERE ticker = ?",
            ("XBI",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "XBI"
        assert row[1] == pytest.approx(rec.illiquidity_premium_bps, rel=1e-4)
        assert row[2] == rec.equilibrium_clientele_tier
        assert row[3] == rec.regime


def test_cli_amihud_mendelson_table_execution():
    """Verify CLI execution for amihud-mendelson command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "amihud-mendelson",
        "--ticker",
        "AM86",
        "--spread",
        "0.015",
        "--turnover",
        "1.0",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Amihud & Mendelson (1986) Bid-Ask Spread & Clientele Engine" in res.stdout
    assert "AM86" in res.stdout
    assert "Illiquidity Return Premium" in res.stdout


def test_cli_amihud_mendelson_json_execution():
    """Verify CLI execution for amihud-mendelson command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "amihud-mendelson",
        "--ticker",
        "SLV",
        "--spread",
        "0.035",
        "--turnover",
        "0.40",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "SLV"
    assert parsed["clientele_net_alpha_bps"] > 50.0
    assert parsed["equilibrium_clientele_tier"] == "TIER3_PATIENT_ENDOWMENT_ILLIQUID_VALUE"
    assert "hmac_signature" in parsed

