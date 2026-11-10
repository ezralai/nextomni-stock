"""Unit and integration tests for Stewart C. Myers (1977) Debt Overhang & Real Growth Option Engine."""

import json
import sqlite3
import subprocess
import pytest
import numpy as np

from src.analytics.myers_debt_overhang import (
    MyersOverhangReceipt,
    calculate_myers_debt_overhang,
    evaluate_myers_overhang_profile,
    persist_myers_overhang_record,
    generate_hmac_signature,
    verify_hmac_signature,
)


def test_all_equity_first_best_zero_overhang():
    """Verify unlevered all-equity firm (D = 0) captures 100% of positive-NPV growth options."""
    receipt = calculate_myers_debt_overhang(
        assets_in_place_pv=100.0,
        growth_option_gross_pv=60.0,
        investment_outlay=40.0,
        debt_face_value=0.0,
        ticker="UNLEV",
    )

    assert isinstance(receipt, MyersOverhangReceipt)
    assert receipt.ticker == "UNLEV"
    assert receipt.debt_overhang_triggered is False
    assert receipt.overhang_agency_loss == pytest.approx(0.0, abs=1e-8)
    assert receipt.overhang_probability == 0.0
    assert receipt.levered_growth_captured == pytest.approx(receipt.first_best_growth_value, rel=1e-6)
    assert receipt.levered_equity_value == pytest.approx(receipt.levered_firm_value, rel=1e-6)
    assert receipt.regime == "OVERHANG_FREE_FIRST_BEST_INVESTMENT"


def test_severe_debt_overhang_real_option_forfeiture():
    """Verify heavy senior debt induces rational rejection of positive-NPV growth options."""
    receipt = calculate_myers_debt_overhang(
        assets_in_place_pv=100.0,
        growth_option_gross_pv=55.0,
        investment_outlay=40.0,
        debt_face_value=140.0,  # Debt overhang traps upside!
        asset_volatility=0.30,
        growth_volatility=0.35,
        ticker="OVERH",
    )

    assert receipt.debt_overhang_triggered is True
    assert receipt.overhang_agency_loss > 2.0
    assert receipt.overhang_probability > 0.10
    assert receipt.levered_growth_captured < receipt.first_best_growth_value
    assert receipt.levered_firm_value < receipt.first_best_firm_value
    assert receipt.regime in ("MODERATE_OVERHANG_UNDERINVESTMENT", "SEVERE_DEBT_OVERHANG_GROWTH_PARALYSIS")


def test_value_conservation_and_claim_additivity():
    """Verify exact value conservation identities: E + D = V_lev = V_FB - L_overhang."""
    receipt = calculate_myers_debt_overhang(
        assets_in_place_pv=120.0,
        growth_option_gross_pv=70.0,
        investment_outlay=45.0,
        debt_face_value=110.0,
        ticker="CONS",
    )

    assert receipt.levered_firm_value == pytest.approx(
        receipt.levered_equity_value + receipt.levered_debt_value,
        rel=1e-6,
    )
    assert receipt.overhang_agency_loss == pytest.approx(
        receipt.first_best_firm_value - receipt.levered_firm_value,
        rel=1e-6,
    )
    assert receipt.levered_growth_captured <= receipt.first_best_growth_value + 1e-6


def test_monotonicity_in_senior_debt_face():
    """Verify increasing senior debt face value monotonically worsens overhang loss and growth paralysis."""
    r_low = calculate_myers_debt_overhang(debt_face_value=40.0)
    r_mid = calculate_myers_debt_overhang(debt_face_value=95.0)
    r_high = calculate_myers_debt_overhang(debt_face_value=160.0)

    assert r_high.overhang_agency_loss > r_mid.overhang_agency_loss >= r_low.overhang_agency_loss
    assert r_high.levered_growth_captured < r_mid.levered_growth_captured <= r_low.levered_growth_captured


def test_zero_growth_option_zero_overhang():
    """Verify firm with zero investment growth options (G0=0) suffers zero overhang loss regardless of debt."""
    receipt = calculate_myers_debt_overhang(
        assets_in_place_pv=100.0,
        growth_option_gross_pv=0.0,
        investment_outlay=20.0,
        debt_face_value=120.0,
        ticker="NOGROW",
    )

    assert receipt.first_best_growth_value == 0.0
    assert receipt.overhang_agency_loss == pytest.approx(0.0, abs=1e-8)
    assert receipt.debt_overhang_triggered is False


def test_low_leverage_safe_debt_avoids_overhang():
    """Verify low debt cushion maintains overhang-free exercise of growth options."""
    receipt = calculate_myers_debt_overhang(
        assets_in_place_pv=150.0,
        growth_option_gross_pv=60.0,
        investment_outlay=35.0,
        debt_face_value=15.0,
        asset_volatility=0.10,
        ticker="LOWD",
    )

    assert receipt.overhang_severity_pct < 0.01


def test_security_input_validation():
    """Verify strict ticker regex allowlist and positivity validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_myers_debt_overhang(ticker="BAD$$$NAME")

    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_myers_debt_overhang(ticker="TOOLONGTICKERNAME")

    with pytest.raises(ValueError, match="assets_in_place_pv must be finite and positive"):
        calculate_myers_debt_overhang(assets_in_place_pv=-10.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 signature generation and tampering rejection."""
    receipt = calculate_myers_debt_overhang(ticker="AMZN")

    sig_payload = (
        f"{receipt.ticker}:{receipt.assets_in_place_pv:.4f}:{receipt.growth_option_gross_pv:.4f}:"
        f"{receipt.investment_outlay:.4f}:{receipt.debt_face_value:.4f}:{receipt.asset_volatility:.4f}:"
        f"{receipt.growth_volatility:.4f}:{receipt.first_best_firm_value:.4f}:{receipt.levered_firm_value:.4f}:"
        f"{receipt.overhang_agency_loss:.4f}:{receipt.regime}:{receipt.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", receipt.hmac_signature) is False


def test_evaluate_myers_overhang_profile_with_prices():
    """Verify empirical calibration from historical equity price series."""
    np.random.seed(42)
    prices = [100.0 + 0.5 * i + 2.2 * np.sin(i / 2.5) for i in range(45)]

    receipt = evaluate_myers_overhang_profile(
        prices=prices,
        ticker="TSLA",
        debt_to_assets_ratio=0.90,
        capex_to_assets_ratio=0.35,
    )

    assert receipt.ticker == "TSLA"
    assert receipt.assets_in_place_pv == pytest.approx(float(prices[-1]), rel=1e-4)
    assert receipt.first_best_firm_value > receipt.assets_in_place_pv


def test_persist_myers_overhang_record_sqlite(tmp_path):
    """Verify SQLite database persistence using parameterized queries."""
    db_file = str(tmp_path / "test_overhang.db")
    receipt = calculate_myers_debt_overhang(ticker="BA")

    success = persist_myers_overhang_record(receipt, db_path=db_file)
    assert success is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, overhang_agency_loss, levered_firm_value, regime FROM myers_overhang_audits WHERE ticker = ?",
            ("BA",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "BA"
        assert row[1] == pytest.approx(receipt.overhang_agency_loss, rel=1e-4)
        assert row[2] == pytest.approx(receipt.levered_firm_value, rel=1e-4)
        assert row[3] == receipt.regime


def test_cli_debt_overhang_table_execution():
    """Verify CLI invocation for debt-overhang command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "debt-overhang",
        "--ticker",
        "GROWTH",
        "--assets",
        "100.0",
        "--growth-pv",
        "60.0",
        "--outlay",
        "40.0",
        "--debt",
        "110.0",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Myers (1977) Debt Overhang & Real Growth Option Engine" in res_table.stdout
    assert "GROWTH" in res_table.stdout
    assert "Overhang Agency Loss" in res_table.stdout


def test_cli_debt_overhang_json_execution():
    """Verify CLI invocation for debt-overhang command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "debt-overhang",
        "--ticker",
        "BA",
        "--assets",
        "100.0",
        "--debt",
        "110.0",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "BA"
    assert "overhang_agency_loss" in parsed
    assert "overhang_severity_pct" in parsed
    assert "hmac_signature" in parsed

