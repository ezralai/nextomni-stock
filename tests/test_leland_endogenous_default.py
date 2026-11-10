"""Unit and integration tests for Hayne E. Leland (1994) Endogenous Default & Optimal Capital Structure Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.leland_endogenous_default import (
    LelandCapitalReceipt,
    calculate_leland_equilibrium,
    evaluate_leland_profile,
    persist_leland_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_calculate_leland_baseline_identities():
    """Verify closed-form valuation identities: v(V) = V + TB - BC = D + E."""
    receipt = calculate_leland_equilibrium(
        asset_value=100.0,
        asset_volatility=0.25,
        risk_free_rate=0.05,
        corporate_tax_rate=0.25,
        bankruptcy_cost_alpha=0.35,
        coupon=4.0,
        ticker="DEBT",
    )

    assert isinstance(receipt, LelandCapitalReceipt)
    assert receipt.ticker == "DEBT"
    assert receipt.default_triggered is False
    assert receipt.endogenous_default_barrier < receipt.asset_value
    assert receipt.levered_firm_value == pytest.approx(
        receipt.asset_value + receipt.tax_shield_value - receipt.bankruptcy_cost_value,
        rel=1e-6,
    )
    assert receipt.levered_firm_value == pytest.approx(
        receipt.debt_value + receipt.equity_value,
        rel=1e-6,
    )
    assert receipt.optimal_firm_value >= receipt.levered_firm_value - 1e-6


def test_optimal_coupon_strictly_maximizes_firm_value():
    """Verify that closed-form optimal coupon C* achieves strictly higher firm value than suboptimal coupons."""
    base = calculate_leland_equilibrium(asset_value=100.0, corporate_tax_rate=0.30, bankruptcy_cost_alpha=0.40, coupon=1.0)
    c_star = base.optimal_coupon

    at_opt = calculate_leland_equilibrium(asset_value=100.0, corporate_tax_rate=0.30, bankruptcy_cost_alpha=0.40, coupon=c_star)
    too_low = calculate_leland_equilibrium(asset_value=100.0, corporate_tax_rate=0.30, bankruptcy_cost_alpha=0.40, coupon=c_star * 0.5)
    too_high = calculate_leland_equilibrium(asset_value=100.0, corporate_tax_rate=0.30, bankruptcy_cost_alpha=0.40, coupon=c_star * 1.8)

    assert at_opt.levered_firm_value > too_low.levered_firm_value
    assert at_opt.levered_firm_value > too_high.levered_firm_value
    assert at_opt.levered_firm_value == pytest.approx(at_opt.optimal_firm_value, rel=1e-6)


def test_endogenous_default_trigger():
    """Verify equity value hits zero and debt recovers (1 - alpha)*V when V <= V_B."""
    # First check barrier for coupon=8.0, tau=0.20, r=0.05, sigma=0.25 -> V_B ~ 87.27
    receipt = calculate_leland_equilibrium(
        asset_value=70.0,  # Below endogenous barrier
        asset_volatility=0.25,
        risk_free_rate=0.05,
        corporate_tax_rate=0.20,
        bankruptcy_cost_alpha=0.30,
        coupon=8.0,
        ticker="BK",
    )

    assert receipt.default_triggered is True
    assert receipt.equity_value == 0.0
    assert receipt.debt_value == pytest.approx(0.70 * 70.0, rel=1e-6)
    assert receipt.regime == "ENDOGENOUS_DEFAULT_BANKRUPTCY_TRIGGERED"


def test_modigliani_miller_zero_tax_limit():
    """Verify that with zero corporate taxes and positive bankruptcy costs, optimal coupon is zero."""
    receipt = calculate_leland_equilibrium(
        asset_value=100.0,
        corporate_tax_rate=0.0,
        bankruptcy_cost_alpha=0.35,
        coupon=3.0,
        ticker="MM",
    )

    assert receipt.tax_shield_value == 0.0
    assert receipt.optimal_coupon == 0.0
    assert receipt.optimal_leverage_ratio == 0.0
    assert receipt.levered_firm_value < receipt.asset_value  # Pure bankruptcy deadweight loss


def test_zero_coupon_all_equity_firm():
    """Verify unlevered all-equity limit (coupon=0): v(V) = V = E(V), spread = 0."""
    receipt = calculate_leland_equilibrium(
        asset_value=150.0,
        coupon=0.0,
        ticker="UNLEV",
    )

    assert receipt.endogenous_default_barrier == 0.0
    assert receipt.debt_value == 0.0
    assert receipt.equity_value == 150.0
    assert receipt.levered_firm_value == 150.0
    assert receipt.credit_spread_bps == 0.0
    assert receipt.regime == "CONSERVATIVE_UNDERLEVERED_TAX_FORFEIT"


def test_monotonicity_bankruptcy_cost_and_volatility():
    """Verify higher bankruptcy costs and asset volatility reduce optimal leverage."""
    r_low_alpha = calculate_leland_equilibrium(bankruptcy_cost_alpha=0.15)
    r_high_alpha = calculate_leland_equilibrium(bankruptcy_cost_alpha=0.55)
    assert r_high_alpha.optimal_coupon < r_low_alpha.optimal_coupon
    assert r_high_alpha.optimal_leverage_ratio < r_low_alpha.optimal_leverage_ratio

    r_low_vol = calculate_leland_equilibrium(asset_volatility=0.15)
    r_high_vol = calculate_leland_equilibrium(asset_volatility=0.45)
    assert r_high_vol.credit_spread_bps > r_low_vol.credit_spread_bps


def test_security_input_validation():
    """Verify strict ticker regex and positive finite asset value checks."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_leland_equilibrium(ticker="INVALID$$$TICKER")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_leland_equilibrium(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="Asset value must be finite and positive"):
        calculate_leland_equilibrium(asset_value=-10.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_leland_equilibrium(ticker="IBM")

    sig_payload = (
        f"{receipt.ticker}:{receipt.asset_value:.4f}:{receipt.asset_volatility:.4f}:"
        f"{receipt.risk_free_rate:.4f}:{receipt.corporate_tax_rate:.4f}:"
        f"{receipt.bankruptcy_cost_alpha:.4f}:{receipt.contractual_coupon:.4f}:"
        f"{receipt.endogenous_default_barrier:.4f}:{receipt.levered_firm_value:.4f}:"
        f"{receipt.optimal_leverage_ratio:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_leland_profile_with_prices():
    """Verify empirical profile calibration from historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.4 * i + 2.0 * np.sin(i / 3.0) for i in range(45)]

    receipt = evaluate_leland_profile(
        prices=prices,
        ticker="AAPL",
        corporate_tax_rate=0.21,
        bankruptcy_cost_alpha=0.30,
        target_leverage_proxy=0.35,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.asset_value > float(prices[-1])
    assert receipt.optimal_firm_value >= receipt.levered_firm_value - 1e-6


def test_persist_leland_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_leland.db")
    receipt = calculate_leland_equilibrium(ticker="F")

    success = persist_leland_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, endogenous_default_barrier, optimal_leverage_ratio, regime FROM leland_capital_audits WHERE ticker = ?",
            ("F",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "F"
        assert row[1] == pytest.approx(receipt.endogenous_default_barrier, rel=1e-4)
        assert row[2] == pytest.approx(receipt.optimal_leverage_ratio, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_leland_table_execution():
    """Verify CLI invocation for leland-default command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "leland-default",
        "--ticker",
        "DEBT",
        "--asset-value",
        "100.0",
        "--coupon",
        "4.0",
        "--tax-rate",
        "0.25",
        "--bankruptcy-cost",
        "0.35",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Leland (1994) Endogenous Default & Optimal Capital Structure Engine" in res_table.stdout
    assert "DEBT" in res_table.stdout
    assert "Endogenous Default Barrier" in res_table.stdout


def test_cli_leland_json_execution():
    """Verify CLI invocation for leland-default command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "leland-default",
        "--ticker",
        "GM",
        "--asset-value",
        "100.0",
        "--coupon",
        "4.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "GM"
    assert "endogenous_default_barrier" in parsed
    assert "optimal_leverage_ratio" in parsed
    assert "hmac_signature" in parsed

