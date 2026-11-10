"""Unit and integration tests for Glosten & Milgrom (1985) Sequential Trade Spread Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.glosten_milgrom_spread import (
    GlostenMilgromReceipt,
    calculate_glosten_milgrom_spread,
    evaluate_glosten_milgrom_profile,
    persist_glosten_milgrom_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_symmetric_prior_closed_form_spread_identity():
    """Verify exact Glosten-Milgrom spread theorem S = pi * (V_H - V_L) at symmetric prior theta=0.5."""
    receipt = calculate_glosten_milgrom_spread(
        value_low=80.0,
        value_high=120.0,       # dV = 40.0
        informed_fraction_pi=0.30,
        prior_prob_high=0.50,
        ticker="BOOK",
    )

    assert isinstance(receipt, GlostenMilgromReceipt)
    assert receipt.ticker == "BOOK"
    assert receipt.initial_midprice == 100.0
    # S = 0.30 * 40.0 = 12.0 exactly
    assert receipt.initial_spread == pytest.approx(12.0, rel=1e-6)
    assert receipt.initial_bid == pytest.approx(94.0, rel=1e-6)
    assert receipt.initial_ask == pytest.approx(106.0, rel=1e-6)


def test_zero_informed_traders_zero_spread():
    """Verify competitive specialist sets Bid = Ask = midprice (zero spread) when pi = 0."""
    receipt = calculate_glosten_milgrom_spread(
        value_low=90.0,
        value_high=110.0,
        informed_fraction_pi=0.0,
        prior_prob_high=0.60,
        ticker="NOINF",
    )

    assert receipt.initial_spread == 0.0
    assert receipt.initial_bid == receipt.initial_ask == receipt.initial_midprice
    assert receipt.regime == "ZERO_INFO_ASYMMETRY_ZERO_SPREAD"


def test_extreme_adverse_selection_market_breakdown():
    """Verify 100% informed traders (pi = 1.0) causes Ask=V_H, Bid=V_L (market breakdown)."""
    receipt = calculate_glosten_milgrom_spread(
        value_low=85.0,
        value_high=115.0,
        informed_fraction_pi=1.0,
        prior_prob_high=0.50,
        ticker="BREAK",
    )

    assert receipt.initial_bid == pytest.approx(85.0, rel=1e-6)
    assert receipt.initial_ask == pytest.approx(115.0, rel=1e-6)
    assert receipt.initial_spread == pytest.approx(30.0, rel=1e-6)
    assert receipt.regime == "EXTREME_ADVERSE_SELECTION_MARKET_BREAKDOWN"


def test_bayesian_sequential_price_discovery_convergence():
    """Verify consecutive buy orders update belief theta toward 1 and narrow posterior spreads."""
    receipt = calculate_glosten_milgrom_spread(
        value_low=90.0,
        value_high=110.0,
        informed_fraction_pi=0.40,
        prior_prob_high=0.50,
        order_flow=[1, 1, 1, 1, 1],  # Five consecutive informed/noise buys
        ticker="LEARN",
    )

    assert receipt.order_flow_steps == 5
    assert receipt.posterior_prob_high > 0.90
    assert receipt.final_midprice > receipt.initial_midprice
    assert receipt.final_spread < receipt.initial_spread
    assert receipt.spread_narrowing_pct >= 0.25
    assert receipt.price_discovery_gain > 0.0
    assert receipt.regime == "BAYESIAN_PRICE_DISCOVERY_CONVERGENCE"


def test_monotonicity_informed_fraction_widens_spread():
    """Verify escalating informed trader share pi strictly widens initial bid-ask spread."""
    r_low = calculate_glosten_milgrom_spread(informed_fraction_pi=0.10)
    r_med = calculate_glosten_milgrom_spread(informed_fraction_pi=0.35)
    r_high = calculate_glosten_milgrom_spread(informed_fraction_pi=0.70)

    assert r_high.initial_spread > r_med.initial_spread > r_low.initial_spread
    assert r_high.initial_spread_bps > r_med.initial_spread_bps > r_low.initial_spread_bps


def test_regret_free_no_arbitrage_quote_ordering():
    """Verify Bid <= Midprice <= Ask and all quotes stay strictly within [V_L, V_H]."""
    receipt = calculate_glosten_milgrom_spread(
        value_low=90.0,
        value_high=110.0,
        informed_fraction_pi=0.35,
        prior_prob_high=0.35,
        order_flow=[1, -1, 1],
    )

    assert 90.0 <= receipt.final_bid <= receipt.final_midprice <= receipt.final_ask <= 110.0


def test_security_input_validation():
    """Verify strict ticker regex allowlist and value_low < value_high bounds."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_glosten_milgrom_spread(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_glosten_milgrom_spread(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="value_low < value_high"):
        calculate_glosten_milgrom_spread(value_low=110.0, value_high=90.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_glosten_milgrom_spread(ticker="AAPL")

    sig_payload = (
        f"{receipt.ticker}:{receipt.value_low:.4f}:{receipt.value_high:.4f}:"
        f"{receipt.informed_fraction_pi:.4f}:{receipt.prior_prob_high:.4f}:"
        f"{receipt.order_flow_steps}:{receipt.posterior_prob_high:.6f}:"
        f"{receipt.initial_spread:.4f}:{receipt.final_spread:.4f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_glosten_milgrom_profile_with_prices():
    """Verify empirical calibration from historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 1.5 * np.cos(i / 2.0) for i in range(40)]

    receipt = evaluate_glosten_milgrom_profile(
        prices=prices,
        ticker="MSFT",
        informed_fraction_pi=0.25,
        uncertainty_band_pct=0.20,
    )

    assert receipt.ticker == "MSFT"
    assert receipt.order_flow_steps > 0
    assert receipt.initial_spread > 0.0


def test_persist_glosten_milgrom_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_gm.db")
    receipt = calculate_glosten_milgrom_spread(ticker="NVDA")

    success = persist_glosten_milgrom_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, initial_spread, final_spread, regime FROM glosten_milgrom_audits WHERE ticker = ?",
            ("NVDA",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "NVDA"
        assert row[1] == pytest.approx(receipt.initial_spread, rel=1e-4)
        assert row[2] == pytest.approx(receipt.final_spread, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_glosten_milgrom_table_execution():
    """Verify CLI invocation for glosten-milgrom command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "glosten-milgrom",
        "--ticker",
        "BOOK",
        "--value-low",
        "90.0",
        "--value-high",
        "110.0",
        "--pi",
        "0.25",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Glosten & Milgrom (1985) Adverse Selection Spread Engine" in res_table.stdout
    assert "BOOK" in res_table.stdout
    assert "Initial Bid / Ask Quote" in res_table.stdout


def test_cli_glosten_milgrom_json_execution():
    """Verify CLI invocation for glosten-milgrom command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "glosten-milgrom",
        "--ticker",
        "SPY",
        "--pi",
        "0.30",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "SPY"
    assert "initial_spread" in parsed
    assert "posterior_prob_high" in parsed
    assert "hmac_signature" in parsed

