"""Unit & CLI integration tests for Dixit (1989) Entry-Exit Hysteresis Band Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.dixit_hysteresis_entry_exit import (
    calculate_dixit_hysteresis,
    evaluate_dixit_hysteresis_profile,
    persist_dixit_hysteresis_record,
    verify_hmac_signature,
)


def test_coupled_value_matching_residual_near_zero_and_band_bounds():
    """Coupled free boundaries P_H and P_L solve value-matching & smooth-pasting cleanly."""
    rec = calculate_dixit_hysteresis(
        current_price_p=1.05,
        operating_cost_c=1.00,
        entry_cost_k=2.00,
        exit_cost_l=1.50,
        discount_rate_r=0.05,
        drift_shortfall_delta=0.05,
        price_volatility_sigma=0.25,
        ticker="DX89",
    )

    assert rec.value_matching_residual < 1e-6
    assert rec.entry_trigger_ph > rec.marshallian_entry_trigger
    assert 0.0 < rec.exit_trigger_pl < rec.marshallian_exit_trigger
    assert rec.hysteresis_band_width > (rec.marshallian_entry_trigger - rec.marshallian_exit_trigger)
    assert rec.option_hysteresis_premium > 0.0


def test_volatility_widens_hysteresis_inaction_band():
    """Higher output volatility widens P_H - P_L asymmetrically (raises P_H, lowers P_L)."""
    low_vol = calculate_dixit_hysteresis(price_volatility_sigma=0.12, ticker="LOWV")
    high_vol = calculate_dixit_hysteresis(price_volatility_sigma=0.40, ticker="HIGHV")

    assert high_vol.entry_trigger_ph > low_vol.entry_trigger_ph
    assert high_vol.exit_trigger_pl < low_vol.exit_trigger_pl
    assert high_vol.hysteresis_band_width > low_vol.hysteresis_band_width


def test_switching_costs_expand_hysteresis_band():
    """Larger sunk entry/exit switching costs expand P_H and depress abandonment trigger P_L."""
    low_cost = calculate_dixit_hysteresis(entry_cost_k=0.5, exit_cost_l=0.5)
    high_cost = calculate_dixit_hysteresis(entry_cost_k=4.0, exit_cost_l=4.0)

    assert high_cost.entry_trigger_ph > low_cost.entry_trigger_ph
    assert high_cost.exit_trigger_pl < low_cost.exit_trigger_pl
    assert high_cost.hysteresis_band_width > low_cost.hysteresis_band_width


def test_frictionless_zero_switching_cost_collapse():
    """When entry_cost_k = 0 and exit_cost_l = 0, hysteresis collapses to Marshallian threshold."""
    rec = calculate_dixit_hysteresis(
        current_price_p=1.05,
        operating_cost_c=1.00,
        entry_cost_k=0.0,
        exit_cost_l=0.0,
        discount_rate_r=0.05,
        drift_shortfall_delta=0.05,
        price_volatility_sigma=0.25,
    )

    assert rec.entry_trigger_ph == pytest.approx(1.00, rel=1e-9)
    assert rec.exit_trigger_pl == pytest.approx(1.00, rel=1e-9)
    assert rec.hysteresis_band_width == 0.0
    assert rec.regime == "FRICTIONLESS_MARSHALLIAN_THRESHOLD"


def test_deterministic_zero_volatility_marshallian_triggers():
    """As sigma -> 0+, optimal thresholds converge exactly to Marshallian P_H = C + r*k, P_L = C - r*l."""
    rec = calculate_dixit_hysteresis(
        operating_cost_c=1.00,
        entry_cost_k=2.00,
        exit_cost_l=1.50,
        discount_rate_r=0.05,
        drift_shortfall_delta=0.05,
        price_volatility_sigma=0.0,
    )

    assert rec.entry_trigger_ph == pytest.approx(1.10, rel=1e-9)
    assert rec.exit_trigger_pl == pytest.approx(0.925, rel=1e-9)


def test_loss_absorbing_hysteresis_and_switching_regimes():
    """Active firm tolerates negative margin P_L < P < C rather than paying exit cost l."""
    loss_absorb = calculate_dixit_hysteresis(
        current_price_p=0.80,
        operating_cost_c=1.00,
        entry_cost_k=2.00,
        exit_cost_l=1.50,
        price_volatility_sigma=0.25,
        current_state="ACTIVE",
    )
    assert loss_absorb.exit_trigger_pl < 0.80 < 1.00
    assert loss_absorb.regime == "LOSS_ABSORBING_HYSTERESIS_PERSISTENCE"

    abandon = calculate_dixit_hysteresis(
        current_price_p=0.30,
        operating_cost_c=1.00,
        entry_cost_k=2.00,
        exit_cost_l=1.50,
        price_volatility_sigma=0.25,
        current_state="ACTIVE",
    )
    assert abandon.regime == "BELOW_EXIT_TRIGGER_ABANDONMENT"
    assert abandon.recommended_action == "ABANDON_EXIT_PRODUCTION_NOW"

    enter = calculate_dixit_hysteresis(
        current_price_p=3.50,
        operating_cost_c=1.00,
        entry_cost_k=2.00,
        exit_cost_l=1.50,
        price_volatility_sigma=0.25,
        current_state="IDLE",
    )
    assert enter.regime == "ABOVE_ENTRY_TRIGGER_ACTIVE_EXPANSION"
    assert enter.recommended_action == "ENTER_PRODUCTION_NOW"


def test_high_exit_cost_forbids_abandonment_recovers_one_sided_trigger():
    """If l >= C/r, abandonment is never optimal (P_L == 0) leaving single upper entry trigger."""
    rec = calculate_dixit_hysteresis(
        operating_cost_c=1.00,
        discount_rate_r=0.05,
        entry_cost_k=2.00,
        exit_cost_l=25.0,  # > C/r = 20.0
    )
    assert rec.exit_trigger_pl == 0.0
    assert rec.entry_trigger_ph > rec.marshallian_entry_trigger


def test_security_input_validation():
    """Verify strict ticker allowlist, current_state allowlist, and positive cost/rate bounds."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_dixit_hysteresis(ticker="DX89;DROP")

    with pytest.raises(ValueError, match="Unsupported current_state"):
        calculate_dixit_hysteresis(current_state="BANKRUPT")

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_dixit_hysteresis(operating_cost_c=-1.0)

    with pytest.raises(ValueError, match="non-negative"):
        calculate_dixit_hysteresis(entry_cost_k=-2.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_dixit_hysteresis(ticker="FCX")
    sig_payload = (
        f"{rec.ticker}:{rec.current_price_p:.4f}:{rec.operating_cost_c:.4f}:"
        f"{rec.entry_cost_k:.4f}:{rec.exit_cost_l:.4f}:{rec.price_volatility_sigma:.4f}:"
        f"{rec.entry_trigger_ph:.4f}:{rec.exit_trigger_pl:.4f}:"
        f"{rec.idle_value_v0:.4f}:{rec.active_value_v1:.4f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_dixit_hysteresis_profile_from_prices():
    """Verify empirical price ratio normalization and volatility estimation."""
    prices = [100.0 + 0.3 * i + 2.0 * np.sin(i / 2.2) for i in range(40)]
    rec = evaluate_dixit_hysteresis_profile(prices=prices, ticker="NEM")

    assert rec.ticker == "NEM"
    assert rec.entry_trigger_ph > rec.exit_trigger_pl


def test_persist_dixit_hysteresis_record_sqlite(tmp_path):
    """Verify parameterized SQLite audit persistence for Dixit hysteresis evaluations."""
    db_file = str(tmp_path / "test_dixit_hysteresis.db")
    rec = calculate_dixit_hysteresis(ticker="SCCO")

    assert persist_dixit_hysteresis_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, entry_trigger_ph, exit_trigger_pl, regime FROM dixit_hysteresis_audits WHERE ticker = ?",
            ("SCCO",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "SCCO"
        assert row[1] == pytest.approx(rec.entry_trigger_ph, rel=1e-4)
        assert row[2] == pytest.approx(rec.exit_trigger_pl, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_entry_exit_table_and_json_execution():
    """Verify CLI execution for entry-exit command in both table and --json modes."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "entry-exit",
        "--ticker",
        "DX89",
        "--price",
        "1.10",
        "--cost",
        "1.00",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Dixit (1989) Entry-Exit Hysteresis Band & Switching Engine" in res_table.stdout
    assert "DX89" in res_table.stdout
    assert "Hysteresis Band Width" in res_table.stdout

    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "entry-exit",
        "--ticker",
        "AA",
        "--vol",
        "0.30",
        "--state",
        "IDLE",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "AA"
    assert parsed["current_state"] == "IDLE"
    assert parsed["hysteresis_band_width"] > 0.0
    assert "hmac_signature" in parsed

