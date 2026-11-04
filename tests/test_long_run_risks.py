"""Tests for Bansal & Yaron (2004) Long-Run Risks (LRR) Pricing Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.long_run_risks import (
    LongRunRisksReceipt,
    calculate_long_run_risks,
    classify_lrr_regime,
    evaluate_long_run_risks_profile,
)
from src.cli import lrr_command
from src.security.validators import SecurityValidationError


def test_calculate_long_run_risks_high_growth_leader() -> None:
    """Verify high LRR cash-flow growth leader when beta_x >= 1.25 and beta_sigma <= -0.25."""
    receipt = calculate_long_run_risks(
        ticker="AAPL",
        latest_price=220.0,
        cashflow_growth_beta=1.65,
        uncertainty_shock_beta=-0.55,
        growth_persistence_rho=0.979,
        intertemporal_elasticity_ies=1.50,
        relative_risk_aversion=10.0,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.early_resolution_preference_theta > 0.0
    assert receipt.wealth_growth_sensitivity_ax > 10.0
    assert receipt.persistent_growth_premium_pct > 0.0
    assert receipt.uncertainty_volatility_premium_pct > 0.0
    assert receipt.bansal_yaron_lrr_impulse >= 0.45
    assert receipt.regime == "HIGH_LRR_CASHFLOW_GROWTH_PREMIUM"
    assert receipt.is_high_lrr_growth_premium_leader is True
    assert receipt.is_macro_uncertainty_safe_haven is False
    assert receipt.expected_lrr_alpha_pct > 4.0
    assert receipt.verify_integrity() is True


def test_calculate_long_run_risks_macro_safe_haven_hedge() -> None:
    """Verify macro uncertainty safe-haven hedge when uncertainty_shock_beta >= 0.35."""
    receipt = calculate_long_run_risks(
        ticker="JNJ",
        latest_price=160.0,
        cashflow_growth_beta=0.55,
        uncertainty_shock_beta=0.65,
        growth_persistence_rho=0.979,
        intertemporal_elasticity_ies=1.50,
        relative_risk_aversion=10.0,
    )
    assert receipt.uncertainty_volatility_premium_pct < 0.0
    assert receipt.regime == "MACRO_UNCERTAINTY_SAFE_HAVEN_HEDGE"
    assert receipt.is_macro_uncertainty_safe_haven is True
    assert receipt.is_high_lrr_growth_premium_leader is False
    assert receipt.verify_integrity() is True


def test_calculate_long_run_risks_uncompensated_fragile_growth_trap() -> None:
    """Verify uncompensated fragile growth trap when beta_x < 0.25 and beta_sigma <= -0.60."""
    receipt = calculate_long_run_risks(
        ticker="INTC",
        latest_price=48.0,
        cashflow_growth_beta=-0.40,
        uncertainty_shock_beta=-0.85,
        growth_persistence_rho=0.979,
        intertemporal_elasticity_ies=1.50,
        relative_risk_aversion=10.0,
    )
    assert receipt.regime == "UNCOMPENSATED_FRAGILE_GROWTH_TRAP"
    assert receipt.is_high_lrr_growth_premium_leader is False
    assert receipt.is_macro_uncertainty_safe_haven is False
    assert receipt.expected_lrr_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_long_run_risks_balanced_equilibrium() -> None:
    """Verify balanced Epstein-Zin equilibrium for moderate exposures."""
    receipt = calculate_long_run_risks(
        ticker="MSFT",
        latest_price=410.0,
        cashflow_growth_beta=0.95,
        uncertainty_shock_beta=-0.15,
        growth_persistence_rho=0.96,
        intertemporal_elasticity_ies=1.50,
        relative_risk_aversion=8.0,
    )
    assert receipt.regime == "BALANCED_EPSTEIN_ZIN_EQUILIBRIUM"
    assert receipt.is_high_lrr_growth_premium_leader is False
    assert receipt.is_macro_uncertainty_safe_haven is False
    assert receipt.verify_integrity() is True


def test_epstein_zin_ies_above_one_increases_wealth_growth_sensitivity() -> None:
    """Verify Bansal & Yaron Theorem: IES > 1 implies A_x > 0, whereas IES < 1 implies A_x < 0."""
    high_ies = calculate_long_run_risks(
        ticker="AAPL",
        latest_price=200.0,
        intertemporal_elasticity_ies=1.50,
    )
    low_ies = calculate_long_run_risks(
        ticker="AAPL",
        latest_price=200.0,
        intertemporal_elasticity_ies=0.65,
    )
    assert high_ies.wealth_growth_sensitivity_ax > 0.0
    assert low_ies.wealth_growth_sensitivity_ax < 0.0


def test_classify_lrr_regime_boundaries() -> None:
    """Verify exact regime classification boundaries."""
    assert (
        classify_lrr_regime(1.40, -0.40, 0.65)
        == "HIGH_LRR_CASHFLOW_GROWTH_PREMIUM"
    )
    assert (
        classify_lrr_regime(0.60, 0.45, 0.10)
        == "MACRO_UNCERTAINTY_SAFE_HAVEN_HEDGE"
    )
    assert (
        classify_lrr_regime(0.10, -0.75, -0.80)
        == "UNCOMPENSATED_FRAGILE_GROWTH_TRAP"
    )
    assert (
        classify_lrr_regime(0.90, -0.10, 0.25)
        == "BALANCED_EPSTEIN_ZIN_EQUILIBRIUM"
    )


def test_long_run_risks_hmac_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity fails if any receipt field is mutated."""
    receipt = calculate_long_run_risks(
        ticker="AAPL",
        latest_price=220.0,
    )
    assert receipt.verify_integrity() is True

    tampered = LongRunRisksReceipt(
        **{
            **receipt.to_dict(),
            "expected_lrr_alpha_pct": 99.99,
        }
    )
    assert tampered.verify_integrity() is False


def test_long_run_risks_input_validation_bounds() -> None:
    """Verify input validation rejects out-of-range parameters."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_long_run_risks("AAPL", 0.0)

    with pytest.raises(ValueError, match="cashflow_growth_beta"):
        calculate_long_run_risks("AAPL", 150.0, cashflow_growth_beta=5.5)

    with pytest.raises(ValueError, match="uncertainty_shock_beta"):
        calculate_long_run_risks("AAPL", 150.0, uncertainty_shock_beta=-4.0)

    with pytest.raises(ValueError, match="growth_persistence_rho"):
        calculate_long_run_risks("AAPL", 150.0, growth_persistence_rho=0.999)

    with pytest.raises(ValueError, match="intertemporal_elasticity_ies"):
        calculate_long_run_risks("AAPL", 150.0, intertemporal_elasticity_ies=0.20)

    with pytest.raises(ValueError, match="relative_risk_aversion"):
        calculate_long_run_risks("AAPL", 150.0, relative_risk_aversion=25.0)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_long_run_risks(malicious, 150.0)


def test_evaluate_long_run_risks_profile_from_sqlite(tmp_path: Path) -> None:
    """Verify end-to-end empirical evaluation from SQLite price observations."""
    db_file = tmp_path / "test_lrr.db"
    db.init_db(db_file)
    db.upsert_stock(
        ticker="AAPL",
        exchange="XNAS",
        company_name="Apple Inc.",
        currency="USD",
        timezone="America/New_York",
        db_path=db_file,
    )

    records = []
    for i in range(30):
        p = 150.0 + i * 1.5
        records.append(
            {
                "ticker": "AAPL",
                "exchange": "XNAS",
                "session_date": f"2026-01-{i+1:02d}",
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "adj_close": p,
                "volume": 50_000_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=db_file)

    receipt = evaluate_long_run_risks_profile(ticker="AAPL", db_path=db_file)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(193.5, abs=1e-2)
    assert receipt.wealth_growth_sensitivity_ax > 0.0
    assert receipt.verify_integrity() is True


def test_evaluate_long_run_risks_profile_override_precedence(tmp_path: Path) -> None:
    """Verify explicit override parameters take precedence over SQLite defaults."""
    db_file = tmp_path / "test_override_lrr.db"
    db.init_db(db_file)
    receipt = evaluate_long_run_risks_profile(
        ticker="AAPL",
        db_path=db_file,
        cashflow_growth_beta=0.45,
        uncertainty_shock_beta=0.80,
    )
    assert receipt.regime == "MACRO_UNCERTAINTY_SAFE_HAVEN_HEDGE"
    assert receipt.is_macro_uncertainty_safe_haven is True


def test_cli_lrr_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI lrr_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        cashflow_growth_beta=1.55,
        uncertainty_shock_beta=-0.50,
        growth_persistence=0.979,
        ies=1.50,
        risk_aversion=10.0,
        json=False,
    )
    lrr_command(args_text)
    captured = capsys.readouterr()
    assert "Bansal & Yaron (2004) Long-Run Risks (LRR) Engine" in captured.out
    assert "Wealth-Growth Sensitivity (A_x):" in captured.out
    assert "Verified: True" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        cashflow_growth_beta=1.55,
        uncertainty_shock_beta=-0.50,
        growth_persistence=0.979,
        ies=1.50,
        risk_aversion=10.0,
        json=True,
    )
    lrr_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "HIGH_LRR_CASHFLOW_GROWTH_PREMIUM"
    assert len(payload["signature"]) == 64

