"""Unit & CLI integration tests for Fischer Black (1976) Futures Option Pricing Engine."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.black76_futures_options import (
    calculate_black76_option,
    evaluate_black76_profile,
    persist_black76_record,
    verify_hmac_signature,
)


def test_black76_futures_put_call_parity_exact():
    """Verify exact Black-76 Futures Put-Call Parity: C - P == exp(-r*T) * (F - K)."""
    rec = calculate_black76_option(
        futures_price_f=112.0,
        strike_k=100.0,
        maturity_years=1.5,
        risk_free_rate_r=0.05,
        volatility=0.30,
        option_type="CALL",
        ticker="CL",
    )

    lhs = rec.call_price - rec.put_price
    rhs = math.exp(-0.05 * 1.5) * (112.0 - 100.0)
    assert lhs == pytest.approx(rhs, rel=1e-9)
    assert rec.option_price == pytest.approx(rec.call_price, rel=1e-9)


def test_non_positive_futures_rho_for_both_calls_and_puts():
    """Holding futures price F constant, Black-76 discount rho == -T * V <= 0 for both Calls and Puts."""
    call_rec = calculate_black76_option(option_type="CALL", maturity_years=2.0, ticker="NG")
    put_rec = calculate_black76_option(option_type="PUT", maturity_years=2.0, ticker="NG")

    assert call_rec.discount_rho < 0.0
    assert put_rec.discount_rho < 0.0
    assert call_rec.discount_rho == pytest.approx(-2.0 * call_rec.call_price, rel=1e-9)
    assert put_rec.discount_rho == pytest.approx(-2.0 * put_rec.put_price, rel=1e-9)


def test_equivalence_to_spot_bsm_under_cost_of_carry_f_equals_s_ert():
    """When F == S_0 * exp(r*T), Black-76 futures Call matches non-dividend spot BSM price."""
    s0, k, t, r, vol = 105.0, 100.0, 1.0, 0.05, 0.25
    f0 = s0 * math.exp(r * t)
    rec = calculate_black76_option(
        futures_price_f=f0,
        strike_k=k,
        maturity_years=t,
        risk_free_rate_r=r,
        volatility=vol,
        spot_reference_s0=s0,
    )

    def _cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    d1 = (math.log(s0 / k) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    bsm_call = s0 * _cdf(d1) - k * math.exp(-r * t) * _cdf(d2)

    assert rec.call_price == pytest.approx(bsm_call, rel=1e-9)
    assert rec.implied_net_carry_rate == pytest.approx(r, rel=1e-9)


def test_zero_volatility_deterministic_forward_intrinsic():
    """At volatility == 0, option value collapses cleanly to discounted intrinsic without NaNs."""
    call_rec = calculate_black76_option(
        futures_price_f=110.0,
        strike_k=100.0,
        maturity_years=1.0,
        risk_free_rate_r=0.05,
        volatility=0.0,
        option_type="CALL",
    )
    assert call_rec.option_price == pytest.approx(10.0 * math.exp(-0.05), rel=1e-9)
    assert call_rec.time_value_premium == pytest.approx(0.0, abs=1e-12)
    assert call_rec.futures_gamma == 0.0
    assert call_rec.vega == 0.0


def test_futures_greeks_signs_and_put_selection():
    """Verify Call/Put delta signs, positive Gamma and Vega, and Put value equality."""
    call_rec = calculate_black76_option(option_type="CALL", futures_price_f=100.0, strike_k=100.0)
    put_rec = calculate_black76_option(option_type="PUT", futures_price_f=100.0, strike_k=100.0)

    assert 0.0 < call_rec.futures_delta < call_rec.discount_factor
    assert -put_rec.discount_factor < put_rec.futures_delta < 0.0
    assert call_rec.futures_delta - put_rec.futures_delta == pytest.approx(call_rec.discount_factor, rel=1e-9)
    assert call_rec.futures_gamma == pytest.approx(put_rec.futures_gamma, rel=1e-9)
    assert call_rec.futures_gamma > 0.0
    assert call_rec.vega == pytest.approx(put_rec.vega, rel=1e-9)
    assert call_rec.vega > 0.0
    assert put_rec.option_price == pytest.approx(put_rec.put_price, rel=1e-9)


def test_backwardation_convenience_yield_regime_detection():
    """Futures below spot carry (F < S0) triggers CONVENIENCE_YIELD_BACKWARDATION_PARITY."""
    rec = calculate_black76_option(
        futures_price_f=92.0,
        strike_k=92.0,
        spot_reference_s0=100.0,
        maturity_years=1.0,
        risk_free_rate_r=0.05,
        volatility=0.22,
    )
    assert rec.implied_net_carry_rate < 0.0
    assert rec.regime == "CONVENIENCE_YIELD_BACKWARDATION_PARITY"


def test_security_input_validation():
    """Verify strict ticker allowlist, CALL/PUT allowlist, and positive futures/strike/maturity."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_black76_option(ticker="CL;DROP")

    with pytest.raises(ValueError, match="Unsupported option_type"):
        calculate_black76_option(option_type="BINARY")

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_black76_option(futures_price_f=-10.0)

    with pytest.raises(ValueError, match="non-negative"):
        calculate_black76_option(volatility=-0.1)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_black76_option(ticker="GC")
    sig_payload = (
        f"{rec.ticker}:{rec.option_type}:{rec.futures_price_f:.4f}:{rec.strike_k:.4f}:"
        f"{rec.maturity_years:.4f}:{rec.risk_free_rate_r:.6f}:{rec.volatility:.4f}:"
        f"{rec.option_price:.6f}:{rec.futures_delta:.6f}:{rec.discount_rho:.6f}:"
        f"{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_black76_profile_from_prices():
    """Verify empirical volatility and forward strike calibration from trailing market closes."""
    prices = [80.0 + 0.25 * i + 1.5 * np.sin(i / 2.2) for i in range(40)]
    rec = evaluate_black76_profile(prices=prices, ticker="SI")

    assert rec.ticker == "SI"
    assert rec.spot_reference_s0 == pytest.approx(prices[-1], rel=1e-6)
    assert rec.option_price > 0.0


def test_persist_black76_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_black76.db")
    rec = calculate_black76_option(ticker="ZC")

    assert persist_black76_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, option_type, option_price, discount_rho, regime FROM black76_futures_audits WHERE ticker = ?",
            ("ZC",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "ZC"
        assert row[1] == "CALL"
        assert row[2] == pytest.approx(rec.option_price, rel=1e-4)
        assert row[3] == pytest.approx(rec.discount_rho, rel=1e-4)
        assert row[4] == rec.regime


def test_cli_futures_option_table_execution():
    """Verify CLI execution for futures-option command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "futures-option",
        "--ticker",
        "CL",
        "--futures",
        "85",
        "--strike",
        "85",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Fischer Black (1976) Futures & Commodity Option Pricing Engine" in res.stdout
    assert "CL" in res.stdout
    assert "Futures Delta" in res.stdout


def test_cli_futures_option_json_execution():
    """Verify CLI execution for futures-option command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "futures-option",
        "--ticker",
        "GC",
        "--futures",
        "2000",
        "--strike",
        "1950",
        "--type",
        "PUT",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "GC"
    assert parsed["option_type"] == "PUT"
    assert parsed["discount_rho"] < 0.0
    assert "hmac_signature" in parsed
