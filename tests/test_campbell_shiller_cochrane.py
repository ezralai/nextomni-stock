"""Tests for Campbell-Shiller (1988) & Cochrane (2008, 2011) Present-Value Decomposition Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.campbell_shiller_cochrane import (
    CampbellShillerCochraneReceipt,
    calculate_campbell_shiller_cochrane,
    classify_cochrane_regime,
    evaluate_campbell_shiller_cochrane_profile,
)
from src.cli import cochrane_command
from src.security.validators import SecurityValidationError


def test_calculate_campbell_shiller_cochrane_discount_rate_opportunity() -> None:
    """Verify discount-rate opportunity when yield is elevated and S_DR >= 0.70."""
    receipt = calculate_campbell_shiller_cochrane(
        ticker="AAPL",
        latest_price=180.0,
        log_dividend_price_ratio=-2.80,
        mean_log_dividend_price=-3.40,
        valuation_persistence_phi=0.94,
        one_period_return_slope=0.09,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.yield_deviation > 0.50
    assert receipt.discount_rate_variance_share > 0.70
    assert receipt.cochrane_discount_rate_impulse >= 0.40
    assert receipt.regime == "DISCOUNT_RATE_MEAN_REVERSION_OPPORTUNITY"
    assert receipt.is_discount_rate_opportunity is True
    assert receipt.is_compressed_discount_rate_trap is False
    assert receipt.expected_discount_rate_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_campbell_shiller_cochrane_compressed_discount_rate_trap() -> None:
    """Verify compressed discount-rate trap when yield is depressed and driven by low discount rate."""
    receipt = calculate_campbell_shiller_cochrane(
        ticker="TSLA",
        latest_price=240.0,
        log_dividend_price_ratio=-4.20,
        mean_log_dividend_price=-3.40,
        valuation_persistence_phi=0.95,
        one_period_return_slope=0.08,
    )
    assert receipt.yield_deviation < -0.70
    assert receipt.discount_rate_variance_share > 0.70
    assert receipt.cochrane_discount_rate_impulse <= -0.40
    assert receipt.regime == "COMPRESSED_DISCOUNT_RATE_OVERVALUATION_TRAP"
    assert receipt.is_compressed_discount_rate_trap is True
    assert receipt.is_discount_rate_opportunity is False
    assert receipt.expected_discount_rate_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_campbell_shiller_cochrane_cashflow_growth_justified() -> None:
    """Verify cash-flow growth justified regime when cash-flow share S_CF >= 0.35."""
    receipt = calculate_campbell_shiller_cochrane(
        ticker="GOOG",
        latest_price=175.0,
        log_dividend_price_ratio=-3.80,
        mean_log_dividend_price=-3.40,
        valuation_persistence_phi=0.94,
        one_period_return_slope=0.04,
        cashflow_growth_forecast_slope=-0.045,
    )
    assert receipt.cashflow_variance_share >= 0.35
    assert receipt.regime == "CASHFLOW_GROWTH_JUSTIFIED_VALUATION"
    assert receipt.is_discount_rate_opportunity is False
    assert receipt.is_compressed_discount_rate_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_campbell_shiller_cochrane_steady_state_equilibrium() -> None:
    """Verify steady-state present-value equilibrium when yield deviation is small."""
    receipt = calculate_campbell_shiller_cochrane(
        ticker="MSFT",
        latest_price=420.0,
        log_dividend_price_ratio=-3.38,
        mean_log_dividend_price=-3.40,
        valuation_persistence_phi=0.94,
    )
    assert abs(receipt.yield_deviation) < 0.15
    assert receipt.regime == "STEADY_STATE_PRESENT_VALUE_EQUILIBRIUM"
    assert receipt.is_discount_rate_opportunity is False
    assert receipt.is_compressed_discount_rate_trap is False
    assert receipt.verify_integrity() is True


def test_cochrane_present_value_identity_holds_exactly() -> None:
    """Verify Cochrane (2008, 2011) Theorem: S_DR + S_CF == 1.0 (or b_r^{LR} - b_d^{LR} == 1.0)."""
    receipt = calculate_campbell_shiller_cochrane(
        ticker="AAPL",
        latest_price=200.0,
        log_dividend_price_ratio=-3.20,
        mean_log_dividend_price=-3.40,
        valuation_persistence_phi=0.92,
        one_period_return_slope=0.12,
    )
    assert (receipt.discount_rate_variance_share + receipt.cashflow_variance_share) == pytest.approx(1.0, abs=1e-3)


def test_classify_cochrane_regime_boundaries() -> None:
    """Verify exact regime classification boundaries."""
    assert (
        classify_cochrane_regime(0.25, 0.85, 0.15, 0.55)
        == "DISCOUNT_RATE_MEAN_REVERSION_OPPORTUNITY"
    )
    assert (
        classify_cochrane_regime(-0.35, 0.80, 0.20, -0.60)
        == "COMPRESSED_DISCOUNT_RATE_OVERVALUATION_TRAP"
    )
    assert (
        classify_cochrane_regime(-0.10, 0.55, 0.45, -0.10)
        == "CASHFLOW_GROWTH_JUSTIFIED_VALUATION"
    )
    assert (
        classify_cochrane_regime(0.05, 0.90, 0.10, 0.10)
        == "STEADY_STATE_PRESENT_VALUE_EQUILIBRIUM"
    )


def test_campbell_shiller_cochrane_hmac_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity fails if any receipt field is mutated."""
    receipt = calculate_campbell_shiller_cochrane(
        ticker="AAPL",
        latest_price=220.0,
    )
    assert receipt.verify_integrity() is True

    tampered = CampbellShillerCochraneReceipt(
        **{
            **receipt.to_dict(),
            "discount_rate_variance_share": 9.99,
        }
    )
    assert tampered.verify_integrity() is False


def test_campbell_shiller_cochrane_input_validation_bounds() -> None:
    """Verify input validation rejects out-of-range parameters."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_campbell_shiller_cochrane("AAPL", 0.0)

    with pytest.raises(ValueError, match="log_dividend_price_ratio"):
        calculate_campbell_shiller_cochrane("AAPL", 150.0, log_dividend_price_ratio=1.5)

    with pytest.raises(ValueError, match="mean_log_dividend_price"):
        calculate_campbell_shiller_cochrane("AAPL", 150.0, mean_log_dividend_price=-8.0)

    with pytest.raises(ValueError, match="valuation_persistence_phi"):
        calculate_campbell_shiller_cochrane("AAPL", 150.0, valuation_persistence_phi=0.40)

    with pytest.raises(ValueError, match="one_period_return_slope"):
        calculate_campbell_shiller_cochrane("AAPL", 150.0, one_period_return_slope=2.5)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_campbell_shiller_cochrane(malicious, 150.0)


def test_evaluate_campbell_shiller_cochrane_profile_from_sqlite(tmp_path: Path) -> None:
    """Verify end-to-end empirical evaluation from SQLite price observations."""
    db_file = tmp_path / "test_cs.db"
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

    receipt = evaluate_campbell_shiller_cochrane_profile(ticker="AAPL", db_path=db_file)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(193.5, abs=1e-2)
    assert 0.90 <= receipt.campbell_shiller_rho <= 0.995
    assert receipt.verify_integrity() is True


def test_evaluate_campbell_shiller_cochrane_profile_override_precedence(tmp_path: Path) -> None:
    """Verify explicit override parameters take precedence over SQLite defaults."""
    db_file = tmp_path / "test_override_cs.db"
    db.init_db(db_file)
    receipt = evaluate_campbell_shiller_cochrane_profile(
        ticker="AAPL",
        db_path=db_file,
        log_dividend_price_ratio=-2.70,
        mean_log_dividend_price=-3.40,
        valuation_persistence_phi=0.95,
        one_period_return_slope=0.10,
    )
    assert receipt.regime == "DISCOUNT_RATE_MEAN_REVERSION_OPPORTUNITY"
    assert receipt.is_discount_rate_opportunity is True


def test_cli_cochrane_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI cochrane_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        log_dividend_price=-2.90,
        mean_log_dividend_price=-3.40,
        valuation_persistence=0.94,
        return_slope=0.08,
        cashflow_slope=None,
        json=False,
    )
    cochrane_command(args_text)
    captured = capsys.readouterr()
    assert "Campbell-Shiller & Cochrane (2008, 2011) Decomposition" in captured.out
    assert "Discount-Rate Share (S_DR):" in captured.out
    assert "Verified: True" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        log_dividend_price=-2.90,
        mean_log_dividend_price=-3.40,
        valuation_persistence=0.94,
        return_slope=0.08,
        cashflow_slope=None,
        json=True,
    )
    cochrane_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "DISCOUNT_RATE_MEAN_REVERSION_OPPORTUNITY"
    assert len(payload["signature"]) == 64

