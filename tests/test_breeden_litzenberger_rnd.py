"""Unit and integration tests for Douglas T. Breeden & Robert H. Litzenberger (1978) Arrow-Debreu State Price & RND Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.breeden_litzenberger_rnd import (
    BreedenLitzenbergerReceipt,
    calculate_breeden_litzenberger_rnd,
    evaluate_breeden_litzenberger_profile,
    generate_skewed_call_surface,
    persist_breeden_litzenberger_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_calculate_breeden_litzenberger_baseline():
    """Verify standard Breeden-Litzenberger state price recovery and moment calculation."""
    strikes, call_prices = generate_skewed_call_surface(
        spot_price=100.0,
        atm_vol=0.20,
        vol_skew=0.0,
        vol_smile=0.0,
        risk_free_rate=0.05,
        time_to_expiry=1.0,
        num_strikes=61,
    )
    receipt = calculate_breeden_litzenberger_rnd(
        spot_price=100.0,
        strikes=strikes,
        call_prices=call_prices,
        risk_free_rate=0.05,
        time_to_expiry=1.0,
        ticker="SPY",
    )

    assert isinstance(receipt, BreedenLitzenbergerReceipt)
    assert receipt.ticker == "SPY"
    assert receipt.spot_price == 100.0
    assert receipt.num_strikes == 61
    assert receipt.theoretical_forward == pytest.approx(100.0 * np.exp(0.05), rel=1e-4)
    # Recovered mean should be close to theoretical forward
    assert receipt.rnd_mean == pytest.approx(receipt.theoretical_forward, rel=0.03)
    assert receipt.normalized_integral == pytest.approx(1.0, rel=1e-5)
    assert receipt.rnd_std > 0.0
    assert receipt.butterfly_arbitrage_detected is False
    assert receipt.butterfly_violation_count == 0


def test_rnd_normalization_integral():
    """Verify that recovered discrete RND integrates exactly to 1.0."""
    strikes, call_prices = generate_skewed_call_surface(
        spot_price=150.0,
        atm_vol=0.28,
        vol_skew=0.15,
        vol_smile=0.02,
        risk_free_rate=0.03,
        time_to_expiry=0.5,
        num_strikes=51,
    )
    receipt = calculate_breeden_litzenberger_rnd(
        spot_price=150.0,
        strikes=strikes,
        call_prices=call_prices,
        risk_free_rate=0.03,
        time_to_expiry=0.5,
        ticker="NVDA",
    )

    assert receipt.normalized_integral == pytest.approx(1.0, rel=1e-5)
    assert 0.0 < receipt.raw_probability_mass <= 1.05


def test_butterfly_arbitrage_detection():
    """Verify that a convexity violation (negative butterfly price) triggers arbitrage flag."""
    strikes = np.linspace(80.0, 120.0, 9)
    # Normal convex call curve: [25, 18, 12, 7, 4, 2, 0.8, 0.2, 0.0]
    # Artificially create a non-convex spike at index 4 (strike 100.0)
    calls = np.array([25.0, 18.0, 12.0, 7.0, 15.0, 2.0, 0.8, 0.2, 0.0])

    receipt = calculate_breeden_litzenberger_rnd(
        spot_price=100.0,
        strikes=strikes,
        call_prices=calls,
        risk_free_rate=0.04,
        time_to_expiry=0.25,
        ticker="ARB",
    )

    assert receipt.butterfly_arbitrage_detected is True
    assert receipt.butterfly_violation_count > 0
    assert receipt.regime == "BUTTERFLY_ARBITRAGE_CONVEXITY_VIOLATION"


def test_volatility_skew_impact_on_rnd_skewness():
    """Verify that steepening implied volatility skew reduces RND skewness (deepens crash fear)."""
    # Zero skew (symmetric Black-Scholes lognormal)
    s_sym, c_sym = generate_skewed_call_surface(spot_price=100.0, vol_skew=0.0, vol_smile=0.0)
    r_sym = calculate_breeden_litzenberger_rnd(100.0, s_sym, c_sym, ticker="SYM")

    # High negative equity put skew (vol rises as strike decreases)
    s_skew, c_skew = generate_skewed_call_surface(spot_price=100.0, vol_skew=0.28, vol_smile=0.0)
    r_skew = calculate_breeden_litzenberger_rnd(100.0, s_skew, c_skew, ticker="SKEW")

    assert abs(r_sym.rnd_skewness) < 0.10
    assert r_sym.regime == "NEAR_LOGNORMAL_RISK_NEUTRAL"
    assert r_skew.rnd_skewness < -0.40
    assert r_skew.regime == "CRASH_FEAR_LEFT_TAIL_SKEW"
    assert r_skew.rnd_skewness < r_sym.rnd_skewness


def test_tail_probabilities_bounds():
    """Verify left and right tail probability bounds lie strictly in [0, 1]."""
    strikes, call_prices = generate_skewed_call_surface(spot_price=200.0, atm_vol=0.30)
    receipt = calculate_breeden_litzenberger_rnd(200.0, strikes, call_prices, ticker="AAPL")

    assert 0.0 <= receipt.left_tail_prob_10pct <= 1.0
    assert 0.0 <= receipt.right_tail_prob_10pct <= 1.0
    assert receipt.left_tail_prob_10pct + receipt.right_tail_prob_10pct <= 1.0


def test_security_input_validation_ticker():
    """Verify strict ticker regex allowlist validation and exception handling."""
    s, c = generate_skewed_call_surface(spot_price=100.0)
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_breeden_litzenberger_rnd(100.0, s, c, ticker="INVALID$$$")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_breeden_litzenberger_rnd(100.0, s, c, ticker="TOOLONGTICKER")


def test_security_input_validation_data():
    """Verify validation on strike/call arrays, length consistency, monotonicity, and positivity."""
    with pytest.raises(ValueError, match="equal length"):
        calculate_breeden_litzenberger_rnd(100.0, [90, 100], [10, 5, 1])

    with pytest.raises(ValueError, match="At least 3 strikes"):
        calculate_breeden_litzenberger_rnd(100.0, [90, 100], [10, 5])

    with pytest.raises(ValueError, match="strictly increasing"):
        calculate_breeden_litzenberger_rnd(100.0, [100, 90, 110], [10, 5, 1])

    with pytest.raises(ValueError, match="Spot price must be finite and positive"):
        calculate_breeden_litzenberger_rnd(-50.0, [90, 100, 110], [15, 8, 2])

    with pytest.raises(ValueError, match="Time to expiry must be positive"):
        calculate_breeden_litzenberger_rnd(100.0, [90, 100, 110], [15, 8, 2], time_to_expiry=-0.1)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    strikes, calls = generate_skewed_call_surface(spot_price=100.0)
    receipt = calculate_breeden_litzenberger_rnd(100.0, strikes, calls, ticker="MSFT")

    sig_payload = (
        f"{receipt.ticker}:{receipt.spot_price:.4f}:{receipt.risk_free_rate:.4f}:"
        f"{receipt.time_to_expiry:.4f}:{receipt.rnd_mean:.4f}:{receipt.rnd_std:.4f}:"
        f"{receipt.rnd_skewness:.4f}:{receipt.rnd_kurtosis:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_breeden_litzenberger_profile_with_prices():
    """Verify empirical profile evaluation with historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.8 * i + 2.5 * np.sin(i / 2.0) for i in range(50)]

    receipt = evaluate_breeden_litzenberger_profile(
        prices=prices,
        ticker="GOOGL",
        vol_skew=0.20,
        vol_smile=0.03,
        risk_free_rate=0.04,
        time_to_expiry=0.25,
    )

    assert receipt.ticker == "GOOGL"
    assert receipt.spot_price == pytest.approx(float(prices[-1]), rel=1e-4)
    assert receipt.rnd_mean > 0.0
    assert receipt.normalized_integral == pytest.approx(1.0, rel=1e-5)


def test_persist_breeden_litzenberger_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_bl_rnd.db")
    strikes, calls = generate_skewed_call_surface(spot_price=120.0)
    receipt = calculate_breeden_litzenberger_rnd(120.0, strikes, calls, ticker="AMD")

    success = persist_breeden_litzenberger_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, spot_price, rnd_mean, rnd_skewness, regime FROM breeden_litzenberger_audits WHERE ticker = ?",
            ("AMD",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "AMD"
        assert row[1] == pytest.approx(120.0, rel=1e-4)
        assert row[2] == pytest.approx(receipt.rnd_mean, rel=1e-4)
        assert row[3] == pytest.approx(receipt.rnd_skewness, rel=1e-4)
        assert row[4] == receipt.regime


def test_cli_breeden_litzenberger_table_execution():
    """Verify CLI invocation for breeden-litzenberger command in standard table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "breeden-litzenberger",
        "--ticker",
        "RND",
        "--spot",
        "100.0",
        "--vol-skew",
        "0.22",
        "--vol-smile",
        "0.04",
        "--rate",
        "0.04",
        "--expiry",
        "0.25",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Breeden & Litzenberger (1978) Arrow-Debreu State Prices & RND Engine" in res_table.stdout
    assert "RND" in res_table.stdout
    assert "Risk-Neutral Skewness" in res_table.stdout


def test_cli_breeden_litzenberger_json_execution():
    """Verify CLI invocation for breeden-litzenberger command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "breeden-litzenberger",
        "--ticker",
        "SPY",
        "--spot",
        "100.0",
        "--vol-skew",
        "0.20",
        "--vol-smile",
        "0.03",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "SPY"
    assert "rnd_mean" in parsed
    assert "rnd_skewness" in parsed
    assert "normalized_integral" in parsed
    assert "hmac_signature" in parsed
