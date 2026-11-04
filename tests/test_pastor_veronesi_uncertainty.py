"""Tests for Pástor & Veronesi (2003, 2006) Bayesian Learning & Uncertainty Convexity Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.pastor_veronesi_uncertainty import (
    PastorVeronesiUncertaintyReceipt,
    calculate_pastor_veronesi_uncertainty,
    classify_pastor_veronesi_regime,
    evaluate_pastor_veronesi_uncertainty_profile,
)
from src.cli import pastor_veronesi_command
from src.security.validators import SecurityValidationError


def test_calculate_pastor_veronesi_convex_growth_learning_leader() -> None:
    """Verify convex growth learning leader when observed M/B underprices warranted convexity."""
    receipt = calculate_pastor_veronesi_uncertainty(
        ticker="NVDA",
        latest_price=140.0,
        expected_excess_profitability=0.10,
        prior_profitability_uncertainty=0.22,
        firm_learning_age_years=4.0,
        retention_ratio=0.90,
        observed_market_to_book=3.50,
    )
    assert receipt.ticker == "NVDA"
    assert receipt.posterior_profitability_uncertainty < receipt.prior_profitability_uncertainty
    assert receipt.convexity_valuation_multiplier > 1.50
    assert receipt.warranted_market_to_book > receipt.observed_market_to_book
    assert receipt.valuation_mispricing_gap_pct <= -0.12
    assert receipt.pastor_veronesi_learning_impulse >= 0.35
    assert receipt.regime == "CONVEX_GROWTH_LEARNING_UNDERVALUED"
    assert receipt.is_convex_learning_leader is True
    assert receipt.is_uncertainty_bubble_trap is False
    assert receipt.expected_learning_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_pastor_veronesi_uncertainty_bubble_trap() -> None:
    """Verify extrapolative uncertainty bubble trap when observed M/B far exceeds warranted bound."""
    receipt = calculate_pastor_veronesi_uncertainty(
        ticker="CSCO",
        latest_price=80.0,
        expected_excess_profitability=0.03,
        prior_profitability_uncertainty=0.10,
        firm_learning_age_years=12.0,
        retention_ratio=0.60,
        observed_market_to_book=12.0,
    )
    assert receipt.valuation_mispricing_gap_pct >= 0.25
    assert receipt.pastor_veronesi_learning_impulse <= -0.40
    assert receipt.regime == "EXTRAPOLATIVE_UNCERTAINTY_BUBBLE_TRAP"
    assert receipt.is_uncertainty_bubble_trap is True
    assert receipt.is_convex_learning_leader is False
    assert receipt.expected_learning_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_pastor_veronesi_rational_convexity_high_valuation() -> None:
    """Verify rational convexity high valuation when high M/B matches warranted C_PV."""
    receipt = calculate_pastor_veronesi_uncertainty(
        ticker="AMZN",
        latest_price=190.0,
        expected_excess_profitability=0.06,
        prior_profitability_uncertainty=0.18,
        firm_learning_age_years=5.0,
        retention_ratio=0.85,
        observed_market_to_book=12.50,
    )
    assert receipt.convexity_valuation_multiplier >= 1.25
    assert abs(receipt.valuation_mispricing_gap_pct) < 0.25
    assert receipt.regime == "RATIONAL_CONVEXITY_HIGH_VALUATION"
    assert receipt.is_convex_learning_leader is False
    assert receipt.is_uncertainty_bubble_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_pastor_veronesi_mature_posterior_certainty_equilibrium() -> None:
    """Verify mature posterior certainty equilibrium for older firm with low uncertainty."""
    receipt = calculate_pastor_veronesi_uncertainty(
        ticker="PG",
        latest_price=165.0,
        expected_excess_profitability=0.03,
        prior_profitability_uncertainty=0.06,
        firm_learning_age_years=45.0,
        retention_ratio=0.35,
        observed_market_to_book=1.40,
    )
    assert receipt.convexity_valuation_multiplier < 1.25
    assert abs(receipt.valuation_mispricing_gap_pct) < 0.25
    assert receipt.regime == "MATURE_POSTERIOR_CERTAINTY_EQUILIBRIUM"
    assert receipt.is_convex_learning_leader is False
    assert receipt.is_uncertainty_bubble_trap is False
    assert receipt.verify_integrity() is True


def test_bayesian_learning_age_reduces_posterior_uncertainty_and_convexity() -> None:
    """Verify Pástor & Veronesi Theorem: older firm age strictly reduces posterior uncertainty and M/B."""
    young = calculate_pastor_veronesi_uncertainty(
        ticker="AAPL",
        latest_price=200.0,
        expected_excess_profitability=0.06,
        prior_profitability_uncertainty=0.16,
        firm_learning_age_years=2.0,
        retention_ratio=0.85,
        observed_market_to_book=3.5,
    )
    mature = calculate_pastor_veronesi_uncertainty(
        ticker="AAPL",
        latest_price=200.0,
        expected_excess_profitability=0.06,
        prior_profitability_uncertainty=0.16,
        firm_learning_age_years=30.0,
        retention_ratio=0.85,
        observed_market_to_book=3.5,
    )
    assert mature.posterior_profitability_uncertainty < young.posterior_profitability_uncertainty
    assert mature.convexity_valuation_multiplier < young.convexity_valuation_multiplier
    assert mature.warranted_market_to_book < young.warranted_market_to_book


def test_classify_pastor_veronesi_regime_boundaries() -> None:
    """Verify exact regime classification boundaries."""
    assert (
        classify_pastor_veronesi_regime(-0.20, 1.40, 0.50)
        == "CONVEX_GROWTH_LEARNING_UNDERVALUED"
    )
    assert (
        classify_pastor_veronesi_regime(0.35, 1.30, -0.60)
        == "EXTRAPOLATIVE_UNCERTAINTY_BUBBLE_TRAP"
    )
    assert (
        classify_pastor_veronesi_regime(0.05, 1.45, 0.05)
        == "RATIONAL_CONVEXITY_HIGH_VALUATION"
    )
    assert (
        classify_pastor_veronesi_regime(0.04, 1.08, -0.05)
        == "MATURE_POSTERIOR_CERTAINTY_EQUILIBRIUM"
    )


def test_pastor_veronesi_hmac_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity fails if any receipt field is mutated."""
    receipt = calculate_pastor_veronesi_uncertainty(
        ticker="AAPL",
        latest_price=220.0,
    )
    assert receipt.verify_integrity() is True

    tampered = PastorVeronesiUncertaintyReceipt(
        **{
            **receipt.to_dict(),
            "warranted_market_to_book": 99.99,
        }
    )
    assert tampered.verify_integrity() is False


def test_pastor_veronesi_input_validation_bounds() -> None:
    """Verify input validation rejects out-of-range parameters."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_pastor_veronesi_uncertainty("AAPL", 0.0)

    with pytest.raises(ValueError, match="expected_excess_profitability"):
        calculate_pastor_veronesi_uncertainty("AAPL", 150.0, expected_excess_profitability=1.20)

    with pytest.raises(ValueError, match="prior_profitability_uncertainty"):
        calculate_pastor_veronesi_uncertainty("AAPL", 150.0, prior_profitability_uncertainty=0.80)

    with pytest.raises(ValueError, match="firm_learning_age_years"):
        calculate_pastor_veronesi_uncertainty("AAPL", 150.0, firm_learning_age_years=0.1)

    with pytest.raises(ValueError, match="retention_ratio"):
        calculate_pastor_veronesi_uncertainty("AAPL", 150.0, retention_ratio=1.2)

    with pytest.raises(ValueError, match="observed_market_to_book"):
        calculate_pastor_veronesi_uncertainty("AAPL", 150.0, observed_market_to_book=0.05)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_pastor_veronesi_uncertainty(malicious, 150.0)


def test_evaluate_pastor_veronesi_profile_from_sqlite(tmp_path: Path) -> None:
    """Verify end-to-end empirical evaluation from SQLite price observations."""
    db_file = tmp_path / "test_pv.db"
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

    receipt = evaluate_pastor_veronesi_uncertainty_profile(ticker="AAPL", db_path=db_file)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(193.5, abs=1e-2)
    assert receipt.convexity_valuation_multiplier >= 1.0
    assert receipt.verify_integrity() is True


def test_evaluate_pastor_veronesi_profile_override_precedence(tmp_path: Path) -> None:
    """Verify explicit override parameters take precedence over SQLite defaults."""
    db_file = tmp_path / "test_override_pv.db"
    db.init_db(db_file)
    receipt = evaluate_pastor_veronesi_uncertainty_profile(
        ticker="AAPL",
        db_path=db_file,
        expected_excess_profitability=0.02,
        prior_profitability_uncertainty=0.08,
        firm_learning_age_years=15.0,
        retention_ratio=0.50,
        observed_market_to_book=14.0,
    )
    assert receipt.regime == "EXTRAPOLATIVE_UNCERTAINTY_BUBBLE_TRAP"
    assert receipt.is_uncertainty_bubble_trap is True


def test_cli_pastor_veronesi_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI pastor_veronesi_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        expected_excess_profitability=0.09,
        prior_uncertainty=0.20,
        firm_age_years=5.0,
        retention_ratio=0.85,
        observed_mb=3.20,
        json=False,
    )
    pastor_veronesi_command(args_text)
    captured = capsys.readouterr()
    assert "Pástor & Veronesi (2003, 2006) Learning & Convexity Engine" in captured.out
    assert "Convexity Multiplier (C_PV):" in captured.out
    assert "Verified: True" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        expected_excess_profitability=0.09,
        prior_uncertainty=0.20,
        firm_age_years=5.0,
        retention_ratio=0.85,
        observed_mb=3.20,
        json=True,
    )
    pastor_veronesi_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["convexity_valuation_multiplier"] >= 1.0
    assert len(payload["signature"]) == 64
