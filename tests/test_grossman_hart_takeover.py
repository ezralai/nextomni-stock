"""Unit and integration tests for Grossman & Hart (1980) Takeover Bids & Free-Rider Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.grossman_hart_takeover import (
    GrossmanHartTakeoverReceipt,
    calculate_grossman_hart_takeover,
    evaluate_grossman_hart_profile,
    persist_grossman_hart_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_free_rider_paradox_blocks_efficient_takeover():
    """Verify zero dilution and zero toehold causes value-creating takeover to fail due to free-riding."""
    receipt = calculate_grossman_hart_takeover(
        status_quo_value=100.0,
        restructuring_synergy=30.0,  # Large synergy (30 > 8 cost)
        bidding_cost=8.0,
        dilution_cap_phi=0.0,        # Strictly prohibited minority dilution
        pre_bid_toehold_alpha=0.0,   # No pre-bid stake
        ticker="PARAD",
    )

    assert isinstance(receipt, GrossmanHartTakeoverReceipt)
    assert receipt.ticker == "PARAD"
    assert receipt.equilibrium_tender_price == 130.0  # Shareholders demand full post-takeover v
    assert receipt.bidder_net_profit == pytest.approx(-8.0, rel=1e-6)
    assert receipt.takeover_succeeds is False
    assert receipt.target_bid_premium == 0.0
    assert receipt.missed_synergy_loss == pytest.approx(22.0, rel=1e-6)
    assert receipt.regime == "FREE_RIDER_PARADOX_PARALYSIS"


def test_dilution_cap_unlocks_profitable_takeover():
    """Verify corporate charter dilution allowance phi > c restores bidder profit and target premium."""
    receipt = calculate_grossman_hart_takeover(
        status_quo_value=100.0,
        restructuring_synergy=30.0,
        bidding_cost=8.0,
        dilution_cap_phi=14.0,
        pre_bid_toehold_alpha=0.0,
        ticker="DILU",
    )

    assert receipt.equilibrium_tender_price == 116.0  # v(130) - phi(14)
    assert receipt.bidder_net_profit == pytest.approx(6.0, rel=1e-6)
    assert receipt.takeover_succeeds is True
    assert receipt.target_bid_premium == pytest.approx(16.0, rel=1e-6)
    assert receipt.missed_synergy_loss == 0.0
    assert receipt.regime == "DILUTION_ENABLED_EFFICIENT_TAKEOVER"


def test_shleifer_vishny_toehold_resolution_zero_dilution():
    """Verify pre-bid activist toehold overcomes free-riding even with zero minority dilution."""
    receipt = calculate_grossman_hart_takeover(
        status_quo_value=100.0,
        restructuring_synergy=40.0,
        bidding_cost=6.0,
        dilution_cap_phi=0.0,
        pre_bid_toehold_alpha=0.25,  # 25% * 40 = 10 > 6 cost
        ticker="ACTV",
    )

    assert receipt.takeover_succeeds is True
    assert receipt.bidder_net_profit == pytest.approx(4.0, rel=1e-6)
    assert receipt.target_bid_premium == 40.0
    assert receipt.optimal_min_dilution == 0.0
    assert receipt.regime == "SHLEIFER_VISHNY_TOEHOLD_CONTROL"


def test_uneconomic_bid_insufficient_synergy():
    """Verify negative social surplus bid (synergy <= cost) fails cleanly."""
    receipt = calculate_grossman_hart_takeover(
        status_quo_value=100.0,
        restructuring_synergy=5.0,
        bidding_cost=12.0,
        dilution_cap_phi=10.0,
        ticker="BADM",
    )

    assert receipt.regime == "UNECONOMIC_BID_INSUFFICIENT_SYNERGY"
    assert receipt.takeover_succeeds is False


def test_monotonicity_dilution_vs_tender_premium():
    """Verify expanding dilution cap strictly increases bidder profit and narrows target tender premium."""
    r_low_phi = calculate_grossman_hart_takeover(dilution_cap_phi=10.0, pre_bid_toehold_alpha=0.0, bidding_cost=5.0)
    r_high_phi = calculate_grossman_hart_takeover(dilution_cap_phi=20.0, pre_bid_toehold_alpha=0.0, bidding_cost=5.0)

    assert r_high_phi.bidder_net_profit > r_low_phi.bidder_net_profit
    assert r_high_phi.target_bid_premium < r_low_phi.target_bid_premium


def test_optimal_min_dilution_breakeven_identity():
    """Verify bidding precisely at optimal minimal dilution floor yields zero net bidder profit."""
    base = calculate_grossman_hart_takeover(restructuring_synergy=30.0, bidding_cost=10.0, pre_bid_toehold_alpha=0.10)
    at_floor = calculate_grossman_hart_takeover(
        restructuring_synergy=30.0,
        bidding_cost=10.0,
        pre_bid_toehold_alpha=0.10,
        dilution_cap_phi=base.optimal_min_dilution,
    )
    assert abs(at_floor.bidder_net_profit) < 1e-6


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positivity validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_grossman_hart_takeover(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_grossman_hart_takeover(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="status_quo_value must be finite and positive"):
        calculate_grossman_hart_takeover(status_quo_value=-50.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_grossman_hart_takeover(ticker="TWTR")

    sig_payload = (
        f"{receipt.ticker}:{receipt.status_quo_value:.4f}:{receipt.restructuring_synergy:.4f}:"
        f"{receipt.bidding_cost:.4f}:{receipt.dilution_cap_phi:.4f}:{receipt.pre_bid_toehold_alpha:.4f}:"
        f"{receipt.equilibrium_tender_price:.4f}:{receipt.bidder_net_profit:.4f}:"
        f"{receipt.takeover_succeeds}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_grossman_hart_profile_with_prices():
    """Verify empirical calibration from historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.4 * i + 2.0 * np.sin(i / 2.5) for i in range(40)]

    receipt = evaluate_grossman_hart_profile(
        prices=prices,
        ticker="ATVI",
        dilution_pct=0.15,
        toehold_alpha=0.05,
        synergy_pct=0.30,
    )

    assert receipt.ticker == "ATVI"
    assert receipt.status_quo_value == pytest.approx(float(prices[-1]), rel=1e-4)
    assert receipt.takeover_succeeds is True


def test_persist_grossman_hart_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_gh_takeover.db")
    receipt = calculate_grossman_hart_takeover(ticker="VMW")

    success = persist_grossman_hart_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, bidder_net_profit, equilibrium_tender_price, regime FROM grossman_hart_audits WHERE ticker = ?",
            ("VMW",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "VMW"
        assert row[1] == pytest.approx(receipt.bidder_net_profit, rel=1e-4)
        assert row[2] == pytest.approx(receipt.equilibrium_tender_price, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_takeover_bid_table_execution():
    """Verify CLI invocation for takeover-bid command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "takeover-bid",
        "--ticker",
        "TARGET",
        "--status-quo",
        "100.0",
        "--synergy",
        "30.0",
        "--dilution",
        "14.0",
        "--toehold",
        "0.05",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Grossman & Hart (1980) Takeover Bids & Free-Rider Engine" in res_table.stdout
    assert "TARGET" in res_table.stdout
    assert "Free-Rider Equilibrium Bid" in res_table.stdout


def test_cli_takeover_bid_json_execution():
    """Verify CLI invocation for takeover-bid command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "takeover-bid",
        "--ticker",
        "SAVE",
        "--status-quo",
        "100.0",
        "--dilution",
        "0.0",
        "--toehold",
        "0.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "SAVE"
    assert parsed["takeover_succeeds"] is False
    assert parsed["missed_synergy_loss"] > 0.0
    assert "hmac_signature" in parsed

