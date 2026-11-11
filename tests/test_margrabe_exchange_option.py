"""Unit & CLI integration tests for William Margrabe (1978) Exchange Option Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.margrabe_exchange_option import (
    calculate_margrabe_exchange_option,
    evaluate_margrabe_exchange_profile,
    persist_margrabe_exchange_record,
    verify_hmac_signature,
)


def test_euler_degree_one_homogeneity_identity():
    """Verify Euler's theorem: V(S1, S2) == S1*Delta_1 + S2*Delta_2 and scale homogeneity."""
    rec1 = calculate_margrabe_exchange_option(
        spot_asset1=120.0,
        spot_asset2=100.0,
        vol_asset1=0.30,
        vol_asset2=0.22,
        correlation_rho=0.40,
        maturity_years=1.5,
        asset1_ticker="AAPL",
        asset2_ticker="SPY",
    )
    euler_sum = rec1.spot_asset1 * rec1.delta_asset1 + rec1.spot_asset2 * rec1.delta_asset2
    assert rec1.exchange_option_value == pytest.approx(euler_sum, rel=1e-9)
    assert rec1.delta_asset1 > 0.0
    assert rec1.delta_asset2 < 0.0

    rec_scaled = calculate_margrabe_exchange_option(
        spot_asset1=240.0,
        spot_asset2=200.0,
        vol_asset1=0.30,
        vol_asset2=0.22,
        correlation_rho=0.40,
        maturity_years=1.5,
        asset1_ticker="AAPL",
        asset2_ticker="SPY",
    )
    assert rec_scaled.exchange_option_value == pytest.approx(2.0 * rec1.exchange_option_value, rel=1e-9)


def test_correlation_comparative_statics_monotonicity():
    """Higher correlation lowers composite spread volatility and strictly lowers exchange value."""
    low_corr = calculate_margrabe_exchange_option(correlation_rho=-0.20, asset1_ticker="NVDA", asset2_ticker="QQQ")
    mid_corr = calculate_margrabe_exchange_option(correlation_rho=0.50, asset1_ticker="NVDA", asset2_ticker="QQQ")
    high_corr = calculate_margrabe_exchange_option(correlation_rho=0.92, asset1_ticker="NVDA", asset2_ticker="QQQ")

    assert low_corr.composite_volatility > mid_corr.composite_volatility > high_corr.composite_volatility
    assert low_corr.exchange_option_value > mid_corr.exchange_option_value > high_corr.exchange_option_value
    assert mid_corr.correlation_sensitivity < 0.0
    assert mid_corr.relative_vega > 0.0


def test_exchange_option_lower_bound_and_time_value():
    """Option value must satisfy arbitrage lower bound max(S1*exp(-q1*T) - S2*exp(-q2*T), 0)."""
    rec = calculate_margrabe_exchange_option(
        spot_asset1=115.0,
        spot_asset2=100.0,
        div_yield_asset1=0.01,
        div_yield_asset2=0.02,
        correlation_rho=0.30,
        maturity_years=1.0,
    )
    assert rec.exchange_option_value > rec.intrinsic_forward_bound
    assert rec.time_switching_value == pytest.approx(rec.exchange_option_value - rec.intrinsic_forward_bound, rel=1e-9)


def test_degenerate_zero_composite_volatility_safe():
    """Verify exact deterministic limit without division by zero when rho=1 and vol1==vol2."""
    rec = calculate_margrabe_exchange_option(
        spot_asset1=110.0,
        spot_asset2=100.0,
        vol_asset1=0.25,
        vol_asset2=0.25,
        div_yield_asset1=0.0,
        div_yield_asset2=0.0,
        correlation_rho=1.0,
        maturity_years=1.0,
    )
    assert rec.composite_volatility == pytest.approx(0.0, abs=1e-10)
    assert rec.exchange_option_value == pytest.approx(10.0, rel=1e-9)
    assert rec.regime == "TIGHT_COINTEGRATED_SPREAD"


def test_symmetry_parity_identity():
    """Verify Put-Call Exchange Parity: V(S1,S2) - V(S2,S1) == S1*exp(-q1*T) - S2*exp(-q2*T)."""
    import math

    v12 = calculate_margrabe_exchange_option(
        spot_asset1=108.0,
        spot_asset2=100.0,
        vol_asset1=0.28,
        vol_asset2=0.21,
        div_yield_asset1=0.01,
        div_yield_asset2=0.02,
        correlation_rho=0.45,
        maturity_years=1.25,
        asset1_ticker="MSFT",
        asset2_ticker="SPY",
    )
    v21 = calculate_margrabe_exchange_option(
        spot_asset1=100.0,
        spot_asset2=108.0,
        vol_asset1=0.21,
        vol_asset2=0.28,
        div_yield_asset1=0.02,
        div_yield_asset2=0.01,
        correlation_rho=0.45,
        maturity_years=1.25,
        asset1_ticker="SPY",
        asset2_ticker="MSFT",
    )

    lhs = v12.exchange_option_value - v21.exchange_option_value
    rhs = 108.0 * math.exp(-0.01 * 1.25) - 100.0 * math.exp(-0.02 * 1.25)
    assert lhs == pytest.approx(rhs, rel=1e-8)


def test_regime_classification_boundaries():
    """Verify regime labels across dispersion and moneyness extremes."""
    high_disp = calculate_margrabe_exchange_option(
        spot_asset1=100.0, spot_asset2=100.0, vol_asset1=0.35, vol_asset2=0.25, correlation_rho=0.0
    )
    assert high_disp.regime == "HIGH_DISPERSION_SWITCH_OPTIONALITY"

    otm = calculate_margrabe_exchange_option(
        spot_asset1=75.0, spot_asset2=100.0, vol_asset1=0.25, vol_asset2=0.20, correlation_rho=0.5
    )
    assert otm.regime == "OUT_OF_THE_MONEY_SWITCH"


def test_security_input_validation():
    """Verify strict regex validation for both asset tickers and numeric range guards."""
    with pytest.raises(ValueError, match="Invalid asset1_ticker symbol"):
        calculate_margrabe_exchange_option(asset1_ticker="BAD;SQL")

    with pytest.raises(ValueError, match="Invalid asset2_ticker symbol"):
        calculate_margrabe_exchange_option(asset2_ticker="TOO_LONG_TICKER")

    with pytest.raises(ValueError, match="correlation_rho"):
        calculate_margrabe_exchange_option(correlation_rho=1.25)

    with pytest.raises(ValueError, match="must be strictly positive"):
        calculate_margrabe_exchange_option(spot_asset1=-50.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering detection."""
    rec = calculate_margrabe_exchange_option(asset1_ticker="META", asset2_ticker="QQQ")
    sig_payload = (
        f"{rec.asset1_ticker}:{rec.asset2_ticker}:{rec.spot_asset1:.4f}:{rec.spot_asset2:.4f}:"
        f"{rec.vol_asset1:.4f}:{rec.vol_asset2:.4f}:{rec.correlation_rho:.4f}:{rec.maturity_years:.4f}:"
        f"{rec.exchange_option_value:.6f}:{rec.composite_volatility:.6f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_margrabe_exchange_profile_from_prices():
    """Verify trailing price return volatility calibration for Asset 1 exchange profile."""
    prices = [100.0 + 0.3 * i + 1.8 * np.sin(i / 2.2) for i in range(40)]
    rec = evaluate_margrabe_exchange_profile(
        prices=prices,
        asset1_ticker="AAPL",
        asset2_ticker="SPY",
        correlation_rho=0.60,
    )

    assert rec.asset1_ticker == "AAPL"
    assert rec.asset2_ticker == "SPY"
    assert rec.spot_asset1 == pytest.approx(prices[-1], rel=1e-6)
    assert rec.exchange_option_value > 0.0


def test_persist_margrabe_exchange_record_sqlite(tmp_path):
    """Verify SQLite persistence using strictly parameterized queries."""
    db_file = str(tmp_path / "test_margrabe.db")
    rec = calculate_margrabe_exchange_option(asset1_ticker="AMD", asset2_ticker="SOXX")

    assert persist_margrabe_exchange_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT asset1_ticker, asset2_ticker, exchange_option_value, regime FROM margrabe_exchange_audits WHERE asset1_ticker = ?",
            ("AMD",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "AMD"
        assert row[1] == "SOXX"
        assert row[2] == pytest.approx(rec.exchange_option_value, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_exchange_option_table_execution():
    """Verify CLI execution for exchange-option command in table output mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "exchange-option",
        "--asset1",
        "NVDA",
        "--asset2",
        "SPY",
        "--s1",
        "105",
        "--s2",
        "100",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "William Margrabe (1978) Exchange Option & Relative-Performance Engine" in res.stdout
    assert "NVDA vs SPY" in res.stdout
    assert "Exchange Option Value" in res.stdout


def test_cli_exchange_option_json_execution():
    """Verify CLI execution for exchange-option command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "exchange-option",
        "--asset1",
        "AAPL",
        "--asset2",
        "QQQ",
        "--rho",
        "0.40",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["asset1_ticker"] == "AAPL"
    assert parsed["asset2_ticker"] == "QQQ"
    assert parsed["correlation_rho"] == pytest.approx(0.40)
    assert "hmac_signature" in parsed

