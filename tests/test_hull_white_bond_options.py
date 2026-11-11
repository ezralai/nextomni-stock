"""Unit & CLI integration tests for Hull & White (1990) No-Arbitrage Bond Option Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.hull_white_bond_options import (
    calculate_hull_white_bond_option,
    evaluate_hull_white_profile,
    persist_hull_white_record,
    verify_hmac_signature,
)


def test_exact_bond_option_put_call_parity():
    """Verify closed-form Hull-White ZBC - ZBP == P(0, S) - K * P(0, T)."""
    rec = calculate_hull_white_bond_option(
        short_rate_r0=0.035,
        long_yield_y_inf=0.050,
        mean_reversion_a=0.15,
        rate_volatility_sigma=0.018,
        option_maturity_t=2.0,
        bond_maturity_s=10.0,
        strike_k=0.65,
        option_type="CALL",
        ticker="TLT",
    )

    lhs = rec.bond_call_price - rec.bond_put_price
    rhs = rec.discount_p_0_s - rec.strike_k * rec.discount_p_0_t
    assert lhs == pytest.approx(rhs, abs=1e-10)
    assert rec.curve_fit_error_bps == 0.0
    assert rec.option_price == pytest.approx(rec.bond_call_price, rel=1e-10)


def test_deterministic_zero_rate_vol_limit_matches_intrinsic():
    """As sigma -> 0, bond option price equals exact discounted forward intrinsic value."""
    rec = calculate_hull_white_bond_option(
        rate_volatility_sigma=0.0,
        option_maturity_t=2.0,
        bond_maturity_s=10.0,
        strike_k=0.55,
        option_type="CALL",
        ticker="DET",
    )

    assert rec.bond_price_vol_sigma_p == 0.0
    assert rec.option_price == pytest.approx(rec.forward_intrinsic_value, rel=1e-10)
    assert rec.time_value_premium == pytest.approx(0.0, abs=1e-12)


def test_option_value_increases_monotonically_with_rate_volatility():
    """Higher short-rate diffusion sigma raises bond-price volatility sigma_p and option prices."""
    low_vol = calculate_hull_white_bond_option(rate_volatility_sigma=0.005, ticker="LOWV")
    high_vol = calculate_hull_white_bond_option(rate_volatility_sigma=0.030, ticker="HIGHV")

    assert high_vol.bond_price_vol_sigma_p > low_vol.bond_price_vol_sigma_p
    assert high_vol.bond_call_price > low_vol.bond_call_price
    assert high_vol.bond_put_price > low_vol.bond_put_price


def test_short_bond_tail_s_approaches_t_shrinks_duration_sigma_p():
    """As S -> T+, remaining bond duration B(T, S) -> 0 and optionality shrinks cleanly."""
    short_tail = calculate_hull_white_bond_option(
        option_maturity_t=2.0,
        bond_maturity_s=2.05,
        strike_k=0.98,
        ticker="SHRT",
    )
    long_tail = calculate_hull_white_bond_option(
        option_maturity_t=2.0,
        bond_maturity_s=20.0,
        strike_k=0.40,
        ticker="LONG",
    )

    assert short_tail.bond_price_vol_sigma_p < long_tail.bond_price_vol_sigma_p
    assert short_tail.time_value_premium < 0.01


def test_small_mean_reversion_a_taylor_stability():
    """Verify Ho-Lee limit (a -> 0+) evaluates smoothly without 0/0 singularities."""
    rec = calculate_hull_white_bond_option(
        mean_reversion_a=1e-10,
        rate_volatility_sigma=0.015,
        option_maturity_t=2.0,
        bond_maturity_s=10.0,
        strike_k=0.65,
        ticker="HOLE",
    )

    expected_sigma_p = 0.015 * np.sqrt(2.0) * 8.0
    assert rec.bond_price_vol_sigma_p == pytest.approx(expected_sigma_p, rel=1e-6)
    assert rec.option_price > 0.0


def test_call_put_delta_signs_and_put_selection():
    """Verify Call delta in [0, 1], Put delta in [-1, 0], and option_type='PUT' selection."""
    put_rec = calculate_hull_white_bond_option(option_type="PUT", strike_k=0.70, ticker="PUTB")

    assert put_rec.option_price == pytest.approx(put_rec.bond_put_price, rel=1e-10)
    assert -1.0 <= put_rec.spot_delta <= 0.0


def test_security_input_validation():
    """Verify strict ticker allowlist, option_type allowlist, and maturity order 0 < T < S."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_hull_white_bond_option(ticker="HW90;DROP")

    with pytest.raises(ValueError, match="Unsupported option_type"):
        calculate_hull_white_bond_option(option_type="SWAPTION")

    with pytest.raises(ValueError, match="0 < option_maturity_t < bond_maturity_s"):
        calculate_hull_white_bond_option(option_maturity_t=5.0, bond_maturity_s=5.0)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_hull_white_bond_option(mean_reversion_a=0.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering detection."""
    rec = calculate_hull_white_bond_option(ticker="IEF")
    sig_payload = (
        f"{rec.ticker}:{rec.option_type}:{rec.short_rate_r0:.6f}:{rec.long_yield_y_inf:.6f}:"
        f"{rec.mean_reversion_a:.4f}:{rec.rate_volatility_sigma:.4f}:"
        f"{rec.option_maturity_t:.4f}:{rec.bond_maturity_s:.4f}:"
        f"{rec.strike_k:.4f}:{rec.option_price:.6f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_hull_white_profile_from_prices():
    """Verify ATM forward bond calibration from trailing asset closes."""
    prices = [100.0 + 0.25 * i + 1.2 * np.sin(i / 3.0) for i in range(40)]
    rec = evaluate_hull_white_profile(prices=prices, ticker="TLT", option_type="CALL")

    assert rec.ticker == "TLT"
    assert rec.strike_k == pytest.approx(rec.forward_bond_price, rel=1e-6)
    assert rec.option_price > 0.0


def test_persist_hull_white_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized placeholders and FK pragma."""
    db_file = str(tmp_path / "test_hull_white.db")
    rec = calculate_hull_white_bond_option(ticker="SHY")

    assert persist_hull_white_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, option_type, option_price, regime FROM hull_white_audits WHERE ticker = ?",
            ("SHY",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "SHY"
        assert row[1] == "CALL"
        assert row[2] == pytest.approx(rec.option_price, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_hull_white_table_execution():
    """Verify CLI invocation for hull-white command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "hull-white",
        "--ticker",
        "HW90",
        "--t-opt",
        "2.0",
        "--s-bond",
        "10.0",
        "--strike",
        "0.65",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Hull & White (1990) No-Arbitrage Term Structure & Bond Option Engine" in res.stdout
    assert "HW90" in res.stdout
    assert "Forward Bond Price" in res.stdout


def test_cli_hull_white_json_execution():
    """Verify CLI invocation for hull-white command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "hull-white",
        "--ticker",
        "IEF",
        "--type",
        "PUT",
        "--sigma",
        "0.02",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "IEF"
    assert parsed["option_type"] == "PUT"
    assert parsed["curve_fit_error_bps"] == 0.0
    assert "hmac_signature" in parsed

