"""Unit and integration tests for Merton H. Miller (1977) Debt & Taxes Clientele Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.miller_debt_taxes import (
    MillerTaxEquilibriumReceipt,
    calculate_miller_debt_taxes,
    evaluate_miller_tax_profile,
    persist_miller_tax_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_modigliani_miller_1963_special_case():
    """Verify exact MM (1963) identity G_L = tau_c * B_D when personal taxes are equal (tau_ps = tau_pd)."""
    receipt = calculate_miller_debt_taxes(
        unlevered_firm_value=100.0,
        debt_market_value=50.0,
        corporate_tax_rate=0.30,
        personal_equity_tax_rate=0.15,
        personal_debt_tax_rate=0.15,
        ticker="MM63",
    )

    assert isinstance(receipt, MillerTaxEquilibriumReceipt)
    assert receipt.ticker == "MM63"
    assert receipt.miller_gain_factor == pytest.approx(0.30, rel=1e-6)
    assert receipt.miller_leverage_gain == pytest.approx(15.0, rel=1e-6)
    assert receipt.personal_tax_erosion == pytest.approx(0.0, abs=1e-8)
    assert receipt.levered_firm_value == pytest.approx(115.0, rel=1e-6)
    assert receipt.regime == "STRONG_AFTER_TAX_DEBT_ADVANTAGE"


def test_miller_macro_equilibrium_zero_gain():
    """Verify at break-even personal debt bracket tau_pd*, net leverage tax gain vanishes (G_L = 0)."""
    # tau_c = 0.25, tau_ps = 0.20 -> (1-0.25)*(1-0.20) = 0.60 -> tau_pd* = 0.40
    receipt = calculate_miller_debt_taxes(
        unlevered_firm_value=100.0,
        debt_market_value=50.0,
        corporate_tax_rate=0.25,
        personal_equity_tax_rate=0.20,
        personal_debt_tax_rate=0.40,
        ticker="MILLEQ",
    )

    assert receipt.breakeven_debt_tax_rate == pytest.approx(0.40, rel=1e-6)
    assert receipt.miller_gain_factor == pytest.approx(0.0, abs=1e-6)
    assert receipt.miller_leverage_gain == pytest.approx(0.0, abs=1e-6)
    assert receipt.levered_firm_value == pytest.approx(100.0, rel=1e-6)
    assert receipt.regime == "MILLER_MACRO_INDIFFERENCE_EQUILIBRIUM"


def test_high_tax_bracket_negative_leverage_gain():
    """Verify high-tax investors (tau_pd > tau_pd*) experience negative net debt tax gain."""
    receipt = calculate_miller_debt_taxes(
        unlevered_firm_value=100.0,
        debt_market_value=40.0,
        corporate_tax_rate=0.21,
        personal_equity_tax_rate=0.10,  # tau_pd* = 1 - 0.79*0.90 = 0.289
        personal_debt_tax_rate=0.50,    # Top ordinary bracket > 0.289!
        ticker="HBRK",
    )

    assert receipt.miller_gain_factor < 0.0
    assert receipt.miller_leverage_gain < 0.0
    assert receipt.levered_firm_value < receipt.unlevered_firm_value
    assert receipt.investor_clientele == "HIGH_BRACKET_EQUITY_CLIENTELE"
    assert receipt.regime == "TAX_PENALTY_NEGATIVE_LEVERAGE_GAIN"


def test_tax_exempt_clientele_and_clearing_yield_parity():
    """Verify market clearing taxable yield r_d* strictly satisfies r_d* * (1 - tau_pd*) = r_0."""
    receipt = calculate_miller_debt_taxes(
        corporate_tax_rate=0.25,
        personal_equity_tax_rate=0.10,
        personal_debt_tax_rate=0.0,  # Pension/tax-exempt investor
        tax_free_rate=0.04,
        ticker="MUNI",
    )

    after_tax_yield = receipt.market_clearing_bond_yield * (1.0 - receipt.breakeven_debt_tax_rate)
    assert after_tax_yield == pytest.approx(0.04, rel=1e-6)
    assert receipt.tax_clientele_spread_bps > 0.0
    assert receipt.investor_clientele == "TAX_EXEMPT_BOND_CLIENTELE"


def test_monotonicity_personal_debt_tax_reduces_leverage_gain():
    """Verify escalating bondholder personal tax rate tau_pd strictly reduces leverage tax shield."""
    r_low = calculate_miller_debt_taxes(personal_debt_tax_rate=0.05)
    r_mid = calculate_miller_debt_taxes(personal_debt_tax_rate=0.25)
    r_high = calculate_miller_debt_taxes(personal_debt_tax_rate=0.45)

    assert r_low.miller_leverage_gain > r_mid.miller_leverage_gain > r_high.miller_leverage_gain
    assert r_low.personal_tax_erosion < r_mid.personal_tax_erosion < r_high.personal_tax_erosion


def test_unlevered_zero_debt_regime():
    """Verify zero debt sets all shields and personal tax erosions strictly to zero."""
    receipt = calculate_miller_debt_taxes(debt_market_value=0.0, ticker="UNLEV")
    assert receipt.miller_leverage_gain == 0.0
    assert receipt.mm63_corporate_shield == 0.0
    assert receipt.personal_tax_erosion == 0.0
    assert receipt.regime == "UNLEVERED_ZERO_TAX_SHIELD"


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positivity validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_miller_debt_taxes(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_miller_debt_taxes(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="unlevered_firm_value must be finite and positive"):
        calculate_miller_debt_taxes(unlevered_firm_value=-50.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_miller_debt_taxes(ticker="XOM")

    sig_payload = (
        f"{receipt.ticker}:{receipt.unlevered_firm_value:.4f}:{receipt.debt_market_value:.4f}:"
        f"{receipt.corporate_tax_rate:.4f}:{receipt.personal_equity_tax_rate:.4f}:"
        f"{receipt.personal_debt_tax_rate:.4f}:{receipt.miller_gain_factor:.6f}:"
        f"{receipt.miller_leverage_gain:.4f}:{receipt.breakeven_debt_tax_rate:.4f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_miller_tax_profile_with_prices():
    """Verify empirical calibration from historical price sequence."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 2.0 * np.sin(i / 2.5) for i in range(40)]

    receipt = evaluate_miller_tax_profile(
        prices=prices,
        ticker="CVX",
        corporate_tax_rate=0.25,
        personal_equity_tax_rate=0.12,
        personal_debt_tax_rate=0.20,
        debt_ratio=0.40,
    )

    assert receipt.ticker == "CVX"
    assert receipt.unlevered_firm_value == pytest.approx(float(prices[-1]), rel=1e-4)
    assert receipt.miller_leverage_gain > 0.0


def test_persist_miller_tax_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_miller_tax.db")
    receipt = calculate_miller_debt_taxes(ticker="PG")

    success = persist_miller_tax_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, miller_leverage_gain, breakeven_debt_tax_rate, regime FROM miller_tax_audits WHERE ticker = ?",
            ("PG",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "PG"
        assert row[1] == pytest.approx(receipt.miller_leverage_gain, rel=1e-4)
        assert row[2] == pytest.approx(receipt.breakeven_debt_tax_rate, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_debt_taxes_table_execution():
    """Verify CLI invocation for debt-taxes command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "debt-taxes",
        "--ticker",
        "TAX",
        "--firm-value",
        "100.0",
        "--debt",
        "40.0",
        "--corp-tax",
        "0.25",
        "--equity-tax",
        "0.12",
        "--debt-tax",
        "0.30",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Miller (1977) Debt & Taxes Personal Clientele Engine" in res_table.stdout
    assert "TAX" in res_table.stdout
    assert "Break-Even Bond Bracket" in res_table.stdout


def test_cli_debt_taxes_json_execution():
    """Verify CLI invocation for debt-taxes command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "debt-taxes",
        "--ticker",
        "KO",
        "--debt",
        "50.0",
        "--debt-tax",
        "0.40",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "KO"
    assert "miller_leverage_gain" in parsed
    assert "breakeven_debt_tax_rate" in parsed
    assert "hmac_signature" in parsed

