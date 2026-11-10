"""Unit and integration tests for Steven L. Heston (1993) Stochastic Volatility Option Pricing Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.heston_stochastic_vol import (
    HestonPricingReceipt,
    calculate_heston_pricing,
    evaluate_heston_profile,
    persist_heston_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_black_scholes_convergence_zero_vol_of_vol():
    """Verify Heston price matches exact Black-Scholes benchmark when vol-of-vol xi = 0."""
    receipt = calculate_heston_pricing(
        spot_price=100.0,
        strike_price=105.0,
        risk_free_rate=0.05,
        time_to_expiry=0.50,
        initial_variance=0.04,
        long_run_variance=0.04,
        mean_reversion_kappa=2.0,
        vol_of_vol_xi=0.0,
        spot_vol_corr_rho=0.0,
        ticker="BSLIM",
    )

    assert isinstance(receipt, HestonPricingReceipt)
    assert receipt.ticker == "BSLIM"
    assert receipt.heston_call_price == pytest.approx(receipt.bs_benchmark_call_price, rel=1e-4)
    assert abs(receipt.volatility_skew_spread) < 1e-3
    assert receipt.regime == "NEAR_CONSTANT_VOLATILITY_BLACK_SCHOLES"


def test_exact_put_call_parity_identity():
    """Verify exact European put-call parity C - P = S_0 - K * exp(-r*T)."""
    s0, k, r, t = 120.0, 115.0, 0.04, 0.75
    receipt = calculate_heston_pricing(
        spot_price=s0,
        strike_price=k,
        risk_free_rate=r,
        time_to_expiry=t,
        initial_variance=0.06,
        long_run_variance=0.05,
        vol_of_vol_xi=0.40,
        spot_vol_corr_rho=-0.60,
        ticker="PCP",
    )

    rhs = s0 - k * np.exp(-r * t)
    assert (receipt.heston_call_price - receipt.heston_put_price) == pytest.approx(rhs, rel=1e-5)


def test_negative_correlation_generates_equity_put_skew():
    """Verify negative spot-vol correlation rho < 0 produces higher OTM put IV than OTM call IV."""
    r_neg = calculate_heston_pricing(
        spot_price=100.0,
        strike_price=100.0,
        vol_of_vol_xi=0.45,
        spot_vol_corr_rho=-0.75,
        ticker="NEGSK",
    )
    r_pos = calculate_heston_pricing(
        spot_price=100.0,
        strike_price=100.0,
        vol_of_vol_xi=0.45,
        spot_vol_corr_rho=0.75,
        ticker="POSSK",
    )

    assert r_neg.volatility_skew_spread > 0.02
    assert r_neg.otm_put_implied_vol > r_neg.otm_call_implied_vol
    assert r_pos.volatility_skew_spread < -0.02


def test_vol_of_vol_amplifies_butterfly_wings():
    """Verify increasing vol-of-vol xi widens butterfly wing convexity spread (fat tails)."""
    r_low_xi = calculate_heston_pricing(vol_of_vol_xi=0.10, spot_vol_corr_rho=0.0, ticker="LOWXI")
    r_high_xi = calculate_heston_pricing(vol_of_vol_xi=0.65, spot_vol_corr_rho=0.0, ticker="HIGHXI")

    assert r_high_xi.butterfly_wing_spread > r_low_xi.butterfly_wing_spread


def test_feller_condition_validation_and_violation():
    """Verify Feller condition 2*kappa*theta > xi^2 detection accurately separates regimes."""
    # Satisfied: 2 * 2.5 * 0.05 = 0.25 > 0.30^2 = 0.09
    r_ok = calculate_heston_pricing(mean_reversion_kappa=2.5, long_run_variance=0.05, vol_of_vol_xi=0.30)
    assert r_ok.feller_condition_satisfied is True
    assert r_ok.feller_ratio > 1.0

    # Violated: 2 * 0.5 * 0.03 = 0.03 < 0.70^2 = 0.49
    r_bad = calculate_heston_pricing(mean_reversion_kappa=0.5, long_run_variance=0.03, vol_of_vol_xi=0.70, spot_vol_corr_rho=-0.30)
    assert r_bad.feller_condition_satisfied is False
    assert r_bad.feller_ratio < 1.0


def test_no_arbitrage_option_price_bounds():
    """Verify call price satisfies max(0, S0 - K*exp(-rT)) <= C <= S0."""
    receipt = calculate_heston_pricing(spot_price=150.0, strike_price=135.0, risk_free_rate=0.04, time_to_expiry=0.5)
    lower = max(0.0, 150.0 - 135.0 * np.exp(-0.04 * 0.5))
    assert lower <= receipt.heston_call_price <= 150.0
    assert receipt.heston_put_price >= 0.0


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positivity validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_heston_pricing(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_heston_pricing(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="spot_price must be finite and positive"):
        calculate_heston_pricing(spot_price=-25.0)

    with pytest.raises(ValueError, match="time_to_expiry must be finite and positive"):
        calculate_heston_pricing(time_to_expiry=0.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_heston_pricing(ticker="VIX")

    sig_payload = (
        f"{receipt.ticker}:{receipt.spot_price:.4f}:{receipt.strike_price:.4f}:"
        f"{receipt.time_to_expiry:.4f}:{receipt.initial_variance:.4f}:{receipt.long_run_variance:.4f}:"
        f"{receipt.mean_reversion_kappa:.4f}:{receipt.vol_of_vol_xi:.4f}:{receipt.spot_vol_corr_rho:.4f}:"
        f"{receipt.heston_call_price:.6f}:{receipt.heston_put_price:.6f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_heston_profile_with_prices():
    """Verify empirical variance calibration from historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.3 * i + 2.0 * np.sin(i / 2.5) for i in range(50)]

    receipt = evaluate_heston_profile(
        prices=prices,
        ticker="SPX",
        vol_of_vol_xi=0.40,
        spot_vol_corr_rho=-0.70,
    )

    assert receipt.ticker == "SPX"
    assert receipt.spot_price == pytest.approx(float(prices[-1]), rel=1e-4)
    assert receipt.heston_call_price > 0.0
    assert receipt.volatility_skew_spread > 0.0


def test_persist_heston_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_heston.db")
    receipt = calculate_heston_pricing(ticker="NVDA")

    success = persist_heston_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, heston_call_price, volatility_skew_spread, regime FROM heston_vol_audits WHERE ticker = ?",
            ("NVDA",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "NVDA"
        assert row[1] == pytest.approx(receipt.heston_call_price, rel=1e-4)
        assert row[2] == pytest.approx(receipt.volatility_skew_spread, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_heston_table_execution():
    """Verify CLI invocation for heston-vol command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "heston-vol",
        "--ticker",
        "VOL",
        "--spot",
        "100.0",
        "--strike",
        "100.0",
        "--xi",
        "0.35",
        "--rho",
        "-0.65",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Heston (1993) Stochastic Volatility Option Pricing Engine" in res_table.stdout
    assert "VOL" in res_table.stdout
    assert "Feller Stability" in res_table.stdout


def test_cli_heston_json_execution():
    """Verify CLI invocation for heston-vol command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "heston-vol",
        "--ticker",
        "SPY",
        "--spot",
        "100.0",
        "--strike",
        "105.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "SPY"
    assert "heston_call_price" in parsed
    assert "volatility_skew_spread" in parsed
    assert "hmac_signature" in parsed

