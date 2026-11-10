"""Unit and integration tests for Michael J. Brennan (1998) Two-Beta Agency Asset Pricing Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.brennan_agency_pricing import (
    BrennanAgencyReceipt,
    calculate_brennan_agency_pricing,
    evaluate_brennan_agency_profile,
    persist_brennan_agency_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_calculate_brennan_agency_baseline():
    """Verify standard Brennan agency two-beta pricing equilibrium."""
    receipt = calculate_brennan_agency_pricing(
        beta_market=1.10,
        beta_benchmark=1.40,
        institutional_share=0.75,
        risk_aversion_abs=2.50,
        tracking_aversion_rel=5.00,
        market_risk_premium=0.06,
        benchmark_variance=0.04,
        risk_free_rate=0.04,
        ticker="AAPL",
    )

    assert isinstance(receipt, BrennanAgencyReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.beta_market == 1.10
    assert receipt.beta_benchmark == 1.40
    assert receipt.institutional_share == 0.75
    assert receipt.expected_return_capm == pytest.approx(0.04 + 1.10 * 0.06, rel=1e-5)
    # beta_B > beta_M => agency spread is negative (depressed required return due to benchmark crowding)
    assert receipt.agency_distortion_spread < 0.0
    assert receipt.expected_return_agency < receipt.expected_return_capm
    assert receipt.agency_alpha > 0.0


def test_capm_consistency_zero_institutional_share():
    """Verify that when institutional share is zero, the model collapses strictly to CAPM."""
    receipt = calculate_brennan_agency_pricing(
        beta_market=1.20,
        beta_benchmark=1.80,
        institutional_share=0.0,  # Zero institutional wealth
        ticker="SPY",
    )

    assert receipt.agency_risk_intensity == 0.0
    assert receipt.agency_distortion_spread == 0.0
    assert receipt.expected_return_agency == pytest.approx(receipt.expected_return_capm, rel=1e-6)
    assert receipt.regime == "UNCONSTRAINED_CAPM_EQUILIBRIUM"


def test_capm_consistency_zero_tracking_penalty():
    """Verify that when tracking error penalty is zero, the model collapses strictly to CAPM."""
    receipt = calculate_brennan_agency_pricing(
        beta_market=1.20,
        beta_benchmark=1.80,
        institutional_share=0.80,
        tracking_aversion_rel=0.0,  # Zero tracking error penalty
        ticker="MSFT",
    )

    assert receipt.agency_risk_intensity == 0.0
    assert receipt.agency_distortion_spread == 0.0
    assert receipt.expected_return_agency == pytest.approx(receipt.expected_return_capm, rel=1e-6)
    assert receipt.regime == "UNCONSTRAINED_CAPM_EQUILIBRIUM"


def test_benchmark_congestion_overpricing_regime():
    """Verify benchmark congestion overpricing regime when benchmark beta greatly exceeds market beta."""
    receipt = calculate_brennan_agency_pricing(
        beta_market=1.00,
        beta_benchmark=1.60,
        institutional_share=0.80,
        tracking_aversion_rel=6.0,
        ticker="NVDA",
    )

    assert receipt.agency_distortion_spread <= -0.005
    assert receipt.regime == "BENCHMARK_CONGESTION_OVERPRICING"


def test_out_of_benchmark_alpha_premium_regime():
    """Verify out-of-benchmark alpha premium regime when asset is neglected by the benchmark."""
    receipt = calculate_brennan_agency_pricing(
        beta_market=1.20,
        beta_benchmark=0.50,  # Neglected by benchmark
        institutional_share=0.75,
        tracking_aversion_rel=5.0,
        ticker="RUT",
    )

    assert receipt.agency_distortion_spread >= 0.005
    assert receipt.agency_alpha < 0.0  # Positive expected return spread => higher required return
    assert receipt.regime == "OUT_OF_BENCHMARK_ALPHA_PREMIUM"


def test_mathematical_monotonicity_tracking_aversion():
    """Verify that agency risk intensity lambda_agency strictly increases with tracking aversion."""
    r1 = calculate_brennan_agency_pricing(tracking_aversion_rel=2.0)
    r2 = calculate_brennan_agency_pricing(tracking_aversion_rel=5.0)
    r3 = calculate_brennan_agency_pricing(tracking_aversion_rel=10.0)

    assert r1.agency_risk_intensity < r2.agency_risk_intensity
    assert r2.agency_risk_intensity < r3.agency_risk_intensity


def test_mathematical_monotonicity_institutional_share():
    """Verify that agency risk intensity lambda_agency strictly increases with institutional share."""
    r1 = calculate_brennan_agency_pricing(institutional_share=0.30)
    r2 = calculate_brennan_agency_pricing(institutional_share=0.60)
    r3 = calculate_brennan_agency_pricing(institutional_share=0.90)

    assert r1.agency_risk_intensity < r2.agency_risk_intensity
    assert r2.agency_risk_intensity < r3.agency_risk_intensity


def test_security_input_validation():
    """Verify strict ticker regex allowlist validation and exception handling."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_brennan_agency_pricing(ticker="INVALID$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_brennan_agency_pricing(ticker="TOOLONGTICKERNAME")


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_brennan_agency_pricing(ticker="GOOGL")

    sig_payload = (
        f"{receipt.ticker}:{receipt.beta_market:.4f}:{receipt.beta_benchmark:.4f}:"
        f"{receipt.institutional_share:.4f}:{receipt.agency_risk_intensity:.6f}:"
        f"{receipt.expected_return_agency:.6f}:{receipt.agency_distortion_spread:.6f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_brennan_agency_profile_with_prices():
    """Verify empirical profile evaluation with historical price series."""
    np.random.seed(42)
    asset_p = [100.0 * (1.0 + 0.001 * i + 0.015 * np.random.randn()) for i in range(40)]
    market_p = [300.0 * (1.0 + 0.0008 * i + 0.010 * np.random.randn()) for i in range(40)]
    bench_p = [400.0 * (1.0 + 0.0009 * i + 0.012 * np.random.randn()) for i in range(40)]

    receipt = evaluate_brennan_agency_profile(
        asset_prices=asset_p,
        market_prices=market_p,
        benchmark_prices=bench_p,
        ticker="AMZN",
        institutional_share=0.75,
        tracking_aversion=5.0,
    )

    assert receipt.ticker == "AMZN"
    assert receipt.expected_return_agency > 0.0
    assert receipt.agency_risk_intensity > 0.0


def test_persist_brennan_agency_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_brennan.db")
    receipt = calculate_brennan_agency_pricing(ticker="META")

    success = persist_brennan_agency_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, expected_return_agency, agency_distortion_spread, regime FROM brennan_agency_audits WHERE ticker = ?", ("META",))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "META"
        assert row[1] == pytest.approx(receipt.expected_return_agency, rel=1e-4)
        assert row[2] == pytest.approx(receipt.agency_distortion_spread, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_brennan_agency_execution():
    """Verify CLI invocation for brennan-agency command in standard and JSON modes."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "brennan-agency",
        "--ticker",
        "AAPL",
        "--beta-m",
        "1.10",
        "--beta-b",
        "1.45",
        "--inst-share",
        "0.75",
        "--tracking-aversion",
        "6.0",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Michael J. Brennan (1998) Two-Beta Agency Asset Pricing" in res_table.stdout
    assert "AAPL" in res_table.stdout
    assert "Agency Distortion Spread" in res_table.stdout

    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "brennan-agency",
        "--ticker",
        "AAPL",
        "--beta-m",
        "1.10",
        "--beta-b",
        "1.45",
        "--inst-share",
        "0.75",
        "--tracking-aversion",
        "6.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "AAPL"
    assert "agency_distortion_spread" in parsed
    assert "expected_return_agency" in parsed
    assert "hmac_signature" in parsed
