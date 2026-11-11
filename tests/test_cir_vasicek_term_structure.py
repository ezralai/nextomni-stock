"""Unit and integration tests for CIR (1985) & Vasicek (1977) Affine Term Structure Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.cir_vasicek_term_structure import (
    AffineTermStructureReceipt,
    calculate_affine_term_structure,
    evaluate_term_structure_profile,
    persist_term_structure_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_deterministic_convergence_cir_and_vasicek():
    """Verify CIR (1985) and Vasicek (1977) agree to machine precision when volatility sigma = 0."""
    cir = calculate_affine_term_structure(
        short_rate_r0=0.03,
        long_run_mean_theta=0.05,
        mean_reversion_kappa=0.40,
        rate_volatility_sigma=0.0,
        model_type="CIR",
        ticker="ZCIR",
    )
    vas = calculate_affine_term_structure(
        short_rate_r0=0.03,
        long_run_mean_theta=0.05,
        mean_reversion_kappa=0.40,
        rate_volatility_sigma=0.0,
        model_type="VASICEK",
        ticker="ZVAS",
    )

    assert isinstance(cir, AffineTermStructureReceipt)
    assert cir.bond_price_10y == pytest.approx(vas.bond_price_10y, rel=1e-8)
    assert cir.yield_10y == pytest.approx(vas.yield_10y, rel=1e-8)
    assert cir.convexity_adjustment_bps == pytest.approx(0.0, abs=1e-8)


def test_upward_sloping_normal_yield_curve():
    """Verify when r_0 < theta, bond prices decay monotonically and 10Y-2Y slope is positive."""
    receipt = calculate_affine_term_structure(
        short_rate_r0=0.025,
        long_run_mean_theta=0.055,
        mean_reversion_kappa=0.35,
        rate_volatility_sigma=0.04,
        model_type="CIR",
        ticker="NORM",
    )

    assert receipt.bond_price_1y > receipt.bond_price_2y > receipt.bond_price_5y > receipt.bond_price_10y > receipt.bond_price_30y
    assert 0.0 < receipt.bond_price_10y < 1.0
    assert receipt.yield_30y > receipt.yield_10y > receipt.yield_2y > receipt.yield_3m
    assert receipt.term_slope_10y_2y_bps > 50.0


def test_inverted_yield_curve_recession_regime():
    """Verify restrictive central bank rates (r_0 >> theta) generate inverted negative 10Y-2Y slope."""
    receipt = calculate_affine_term_structure(
        short_rate_r0=0.065,
        long_run_mean_theta=0.035,
        mean_reversion_kappa=0.40,
        rate_volatility_sigma=0.03,
        model_type="CIR",
        ticker="INV",
    )

    assert receipt.yield_3m > receipt.yield_2y > receipt.yield_10y > receipt.yield_30y
    assert receipt.term_slope_10y_2y_bps < -25.0
    assert receipt.regime == "INVERTED_YIELD_CURVE_RECESSION_SIGNAL"


def test_feller_positivity_and_convexity_adjustment():
    """Verify CIR Feller check and Jensen convexity pulling asymptotic long yield below theta."""
    receipt = calculate_affine_term_structure(
        short_rate_r0=0.04,
        long_run_mean_theta=0.05,
        mean_reversion_kappa=0.35,
        rate_volatility_sigma=0.10,
        model_type="CIR",
        ticker="CONV",
    )

    assert receipt.asymptotic_long_yield < receipt.long_run_mean_theta
    assert receipt.convexity_adjustment_bps > 0.0

    bad_feller = calculate_affine_term_structure(
        short_rate_r0=0.04,
        long_run_mean_theta=0.04,
        mean_reversion_kappa=0.15,
        rate_volatility_sigma=0.25,  # 2*0.15*0.04 = 0.012 < 0.0625
        model_type="CIR",
        ticker="FELL",
    )
    assert bad_feller.feller_condition_satisfied is False
    assert bad_feller.regime == "CIR_FELLER_BOUNDARY_ZERO_RATE_RISK"


def test_vasicek_bond_pricing_exactness():
    """Verify Vasicek 10Y yield, bond price identity P = exp(-y*T), and Jensen convexity."""
    receipt = calculate_affine_term_structure(
        short_rate_r0=0.03,
        long_run_mean_theta=0.05,
        mean_reversion_kappa=0.30,
        rate_volatility_sigma=0.02,
        model_type="VASICEK",
        ticker="VAS",
    )

    assert receipt.bond_price_10y == pytest.approx(np.exp(-receipt.yield_10y * 10.0), rel=1e-7)
    assert receipt.duration_10y == pytest.approx((1.0 - np.exp(-0.30 * 10.0)) / 0.30, rel=1e-7)
    assert receipt.asymptotic_long_yield < receipt.long_run_mean_theta


def test_no_overflow_at_long_30y_horizon():
    """Verify overflow-free CIR exponent handling for high mean-reversion kappa and 30Y bonds."""
    receipt = calculate_affine_term_structure(
        short_rate_r0=0.04,
        long_run_mean_theta=0.05,
        mean_reversion_kappa=5.0,  # Large kappa*T = 150 -> naive exp would be huge
        rate_volatility_sigma=0.15,
        model_type="CIR",
        ticker="LONG",
    )

    assert np.isfinite(receipt.bond_price_30y)
    assert 0.0 < receipt.bond_price_30y < 1.0
    assert receipt.yield_30y == pytest.approx(receipt.asymptotic_long_yield, rel=0.05)


def test_security_input_validation():
    """Verify strict ticker allowlist, model_type allowlist, and rate positivity checks."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_affine_term_structure(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Unsupported model_type"):
        calculate_affine_term_structure(model_type="INVALID_SQL_INJ")

    with pytest.raises(ValueError, match="mean_reversion_kappa must be positive"):
        calculate_affine_term_structure(mean_reversion_kappa=-0.1)

    with pytest.raises(ValueError, match="CIR model requires non-negative r0"):
        calculate_affine_term_structure(short_rate_r0=-0.02, model_type="CIR")


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_affine_term_structure(ticker="TLT")

    sig_payload = (
        f"{receipt.ticker}:{receipt.model_type}:{receipt.short_rate_r0:.6f}:"
        f"{receipt.long_run_mean_theta:.6f}:{receipt.mean_reversion_kappa:.4f}:"
        f"{receipt.rate_volatility_sigma:.4f}:{receipt.bond_price_10y:.6f}:"
        f"{receipt.yield_10y:.6f}:{receipt.term_slope_10y_2y_bps:.2f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_term_structure_profile_with_prices():
    """Verify empirical calibration from price volatility series."""
    np.random.seed(42)
    prices = [100.0 + 0.2 * i + 1.0 * np.sin(i / 2.5) for i in range(40)]

    receipt = evaluate_term_structure_profile(
        prices=prices,
        ticker="IEF",
        short_rate_r0=0.040,
        long_run_mean_theta=0.048,
        model_type="CIR",
    )

    assert receipt.ticker == "IEF"
    assert 0.0 < receipt.bond_price_10y < 1.0
    assert receipt.yield_10y > 0.0


def test_persist_term_structure_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_term_structure.db")
    receipt = calculate_affine_term_structure(ticker="SHY")

    success = persist_term_structure_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, yield_10y, term_slope_10y_2y_bps, regime FROM term_structure_audits WHERE ticker = ?",
            ("SHY",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "SHY"
        assert row[1] == pytest.approx(receipt.yield_10y, rel=1e-4)
        assert row[2] == pytest.approx(receipt.term_slope_10y_2y_bps, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_term_structure_table_execution():
    """Verify CLI invocation for term-structure command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "term-structure",
        "--ticker",
        "BOND",
        "--r0",
        "0.035",
        "--theta",
        "0.050",
        "--model",
        "CIR",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "CIR (1985) & Vasicek (1977) Affine Term Structure Engine" in res_table.stdout
    assert "BOND" in res_table.stdout
    assert "10Y-2Y Curve Slope" in res_table.stdout


def test_cli_term_structure_json_execution():
    """Verify CLI invocation for term-structure command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "term-structure",
        "--ticker",
        "TLT",
        "--r0",
        "0.06",
        "--theta",
        "0.04",
        "--model",
        "VASICEK",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "TLT"
    assert parsed["model_type"] == "VASICEK"
    assert parsed["term_slope_10y_2y_bps"] < 0.0
    assert "hmac_signature" in parsed

