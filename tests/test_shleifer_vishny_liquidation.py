"""Unit & CLI integration tests for Shleifer & Vishny (1992) Liquidation Debt Capacity Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.shleifer_vishny_liquidation import (
    calculate_shleifer_vishny_liquidation,
    evaluate_shleifer_vishny_liquidation_profile,
    persist_shleifer_vishny_liquidation_record,
    verify_hmac_signature,
)


def test_idiosyncratic_vs_correlated_industry_fire_sale_discount():
    """Correlated industry shocks compress liquidation price and debt capacity when peers lack slack."""
    idio = calculate_shleifer_vishny_liquidation(
        specialist_asset_value_v1=100.0,
        asset_specificity_discount=0.50,
        peer_debt_capacity_slack=0.20,
        industry_correlation_rho=0.10,
        ticker="IDIO",
    )
    systemic = calculate_shleifer_vishny_liquidation(
        specialist_asset_value_v1=100.0,
        asset_specificity_discount=0.50,
        peer_debt_capacity_slack=0.20,
        industry_correlation_rho=0.85,
        ticker="SYST",
    )

    assert idio.idiosyncratic_liquidation_price == pytest.approx(100.0, rel=1e-9)
    assert systemic.downturn_liquidation_price == pytest.approx(60.0, rel=1e-9)
    assert systemic.expected_distress_liquidation_price < idio.expected_distress_liquidation_price
    assert systemic.risky_debt_capacity < idio.risky_debt_capacity
    assert systemic.fire_sale_haircut_pct > idio.fire_sale_haircut_pct
    assert systemic.regime == "SYSTEMIC_INDUSTRY_FIRE_SALE_DISCOUNT"


def test_generalist_redeployable_assets_preserve_full_debt_capacity():
    """Zero asset specificity (delta_spec = 0) yields zero fire-sale loss even in systemic downturns."""
    rec = calculate_shleifer_vishny_liquidation(
        specialist_asset_value_v1=120.0,
        asset_specificity_discount=0.0,
        peer_debt_capacity_slack=0.0,
        industry_correlation_rho=1.0,
        ticker="REIT",
    )

    assert rec.downturn_liquidation_price == pytest.approx(120.0, rel=1e-9)
    assert rec.expected_distress_liquidation_price == pytest.approx(120.0, rel=1e-9)
    assert rec.risky_debt_capacity == pytest.approx(120.0, rel=1e-9)
    assert rec.fire_sale_haircut_pct == pytest.approx(0.0, abs=1e-12)
    assert rec.regime == "GENERALIST_LIQUID_COLLATERAL_FULL_CAPACITY"


def test_abundant_peer_slack_prevents_outsider_discount():
    """Unconstrained healthy industry peers absorb assets at first-best valuation V_1."""
    rec = calculate_shleifer_vishny_liquidation(
        specialist_asset_value_v1=100.0,
        asset_specificity_discount=0.60,
        peer_debt_capacity_slack=1.0,
        industry_correlation_rho=0.90,
        ticker="CASH",
    )

    assert rec.downturn_liquidation_price == pytest.approx(100.0, rel=1e-9)
    assert rec.collateral_capacity_loss == pytest.approx(0.0, abs=1e-12)
    assert rec.regime == "GENERALIST_LIQUID_COLLATERAL_FULL_CAPACITY"


def test_asset_specificity_monotonically_reduces_debt_capacity():
    """Higher asset specificity strictly widens fire-sale haircuts and reduces pledgeable debt."""
    low_spec = calculate_shleifer_vishny_liquidation(asset_specificity_discount=0.15, ticker="LOWS")
    high_spec = calculate_shleifer_vishny_liquidation(asset_specificity_discount=0.65, ticker="HIGHS")

    assert high_spec.outside_buyer_value_v2 < low_spec.outside_buyer_value_v2
    assert high_spec.risky_debt_capacity < low_spec.risky_debt_capacity
    assert high_spec.debt_capacity_haircut_pct > low_spec.debt_capacity_haircut_pct


def test_downturn_probability_lowers_expected_debt_capacity():
    """Higher downturn likelihood reduces ex-ante risky debt capacity when fire sales occur."""
    safe_ind = calculate_shleifer_vishny_liquidation(downturn_prob_q=0.10, asset_specificity_discount=0.45)
    cyclical_ind = calculate_shleifer_vishny_liquidation(downturn_prob_q=0.55, asset_specificity_discount=0.45)

    assert cyclical_ind.risky_debt_capacity < safe_ind.risky_debt_capacity
    assert cyclical_ind.debt_capacity_haircut_pct > safe_ind.debt_capacity_haircut_pct


def test_regime_classifications_across_correlations():
    """Verify idiosyncratic absorption and procyclical drag regime transitions."""
    idio = calculate_shleifer_vishny_liquidation(industry_correlation_rho=0.20, asset_specificity_discount=0.40)
    assert idio.regime == "IDIOSYNCRATIC_PEER_ABSORPTION_PROTECTED"

    proc = calculate_shleifer_vishny_liquidation(
        asset_specificity_discount=0.35,
        peer_debt_capacity_slack=0.40,
        industry_correlation_rho=0.55,
    )
    assert proc.regime == "PROCYCLICAL_PEER_ILLIQUIDITY_DRAG"


def test_security_input_validation():
    """Verify strict ticker allowlist and probability/specificity boundary validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_shleifer_vishny_liquidation(ticker="SV92;DROP")

    with pytest.raises(ValueError, match="asset_specificity_discount must satisfy"):
        calculate_shleifer_vishny_liquidation(asset_specificity_discount=1.0)

    with pytest.raises(ValueError, match="peer_debt_capacity_slack must satisfy"):
        calculate_shleifer_vishny_liquidation(peer_debt_capacity_slack=1.5)

    with pytest.raises(ValueError, match="downturn_prob_q must satisfy"):
        calculate_shleifer_vishny_liquidation(downturn_prob_q=0.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_shleifer_vishny_liquidation(ticker="UAL")
    sig_payload = (
        f"{rec.ticker}:{rec.specialist_asset_value_v1:.4f}:{rec.asset_specificity_discount:.4f}:"
        f"{rec.peer_debt_capacity_slack:.4f}:{rec.industry_correlation_rho:.4f}:"
        f"{rec.downturn_prob_q:.4f}:{rec.expected_distress_liquidation_price:.4f}:"
        f"{rec.risky_debt_capacity:.4f}:{rec.fire_sale_haircut_pct:.2f}:"
        f"{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_shleifer_vishny_liquidation_profile_from_prices():
    """Verify peer slack calibration from trailing asset drawdowns and volatility."""
    prices = [100.0 + 0.25 * i + 2.5 * np.sin(i / 2.2) for i in range(45)]
    rec = evaluate_shleifer_vishny_liquidation_profile(prices=prices, ticker="DAL")

    assert rec.ticker == "DAL"
    assert rec.specialist_asset_value_v1 == pytest.approx(prices[-1], rel=1e-6)
    assert 0.05 <= rec.peer_debt_capacity_slack <= 0.90


def test_persist_shleifer_vishny_liquidation_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_sv_liquidation.db")
    rec = calculate_shleifer_vishny_liquidation(ticker="AAL")

    assert persist_shleifer_vishny_liquidation_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, risky_debt_capacity, fire_sale_haircut_pct, regime FROM shleifer_vishny_liquidation_audits WHERE ticker = ?",
            ("AAL",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "AAL"
        assert row[1] == pytest.approx(rec.risky_debt_capacity, rel=1e-4)
        assert row[2] == pytest.approx(rec.fire_sale_haircut_pct, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_industry_liquidation_table_execution():
    """Verify CLI execution for industry-liquidation command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "industry-liquidation",
        "--ticker",
        "SV92",
        "--specificity",
        "0.45",
        "--rho",
        "0.75",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Shleifer & Vishny (1992) Liquidation Values & Debt Capacity Engine" in res.stdout
    assert "SV92" in res.stdout
    assert "Fire-Sale Haircut" in res.stdout


def test_cli_industry_liquidation_json_execution():
    """Verify CLI execution for industry-liquidation command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "industry-liquidation",
        "--ticker",
        "UAL",
        "--specificity",
        "0.50",
        "--peer-slack",
        "0.20",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "UAL"
    assert parsed["downturn_liquidation_price"] == pytest.approx(60.0, rel=1e-4)
    assert parsed["risky_debt_capacity"] < 100.0
    assert "hmac_signature" in parsed

