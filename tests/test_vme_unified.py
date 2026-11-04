"""Unit and integration tests for Asness, Moskowitz & Pedersen (2013) 'Value and Momentum Everywhere' (VME) Engine."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.vme_unified import (
    VMEReceipt,
    calculate_vme_score,
    classify_vme_regime,
    evaluate_vme_profile,
)
from src.cli import vme_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_vme_synergistic_compounder() -> None:
    """Test twin-engine compounder: attractive valuation + strong 12-2 intermediate momentum."""
    # 60 sessions with strong upward trend (+30% over first 39 sessions)
    prices = [100.0 * (1.0 + 0.008 * i) for i in range(60)]
    receipt = calculate_vme_score(
        ticker="TWIN",
        prices=prices,
        book_to_market=0.95,
        earnings_yield_pct=9.5,
        value_momentum_correlation=-0.52,
    )

    assert receipt.ticker == "TWIN"
    assert receipt.value_z_score > 1.0
    assert receipt.momentum_z_score > 1.0
    assert receipt.vme_combo_z_score > 1.5
    assert receipt.is_twin_engine_compounder is True
    assert receipt.is_dual_headwind_trap is False
    assert receipt.regime == "SYNERGISTIC_VME_COMPOUNDER"
    assert receipt.expected_vme_alpha_pct >= 5.0
    assert receipt.verify_integrity() is True


def test_calculate_vme_hedged_barbell_equilibrium() -> None:
    """Test hedged barbell equilibrium: strong value offsetting mild negative momentum via negative covariance."""
    # Mild downward trend (-2%) so momentum z-score is slightly negative, but deep value (B/M = 1.10)
    prices = [100.0 * (1.0 - 0.0005 * i) for i in range(60)]
    receipt = calculate_vme_score(
        ticker="HDGE",
        prices=prices,
        book_to_market=0.90,
        earnings_yield_pct=8.5,
        value_momentum_correlation=-0.55,
    )

    assert receipt.ticker == "HDGE"
    assert receipt.value_z_score > 0.8
    assert receipt.momentum_z_score < 0.0
    assert receipt.vme_combo_z_score >= 0.25
    assert receipt.regime == "HEDGED_BARBELL_EQUILIBRIUM"
    assert receipt.diversification_benefit_ratio > 1.4
    assert receipt.verify_integrity() is True


def test_calculate_vme_dual_factor_headwind_trap() -> None:
    """Test dual-factor headwind trap: expensive glamour valuation + crashing 12-2 momentum."""
    # Steep downward trend (-30%)
    prices = [100.0 * (1.0 - 0.006 * i) for i in range(60)]
    receipt = calculate_vme_score(
        ticker="TRAP",
        prices=prices,
        book_to_market=0.12,
        earnings_yield_pct=1.2,
        value_momentum_correlation=-0.50,
    )

    assert receipt.ticker == "TRAP"
    assert receipt.value_z_score < -1.0
    assert receipt.momentum_z_score < -1.0
    assert receipt.vme_combo_z_score < -0.50
    assert receipt.is_dual_headwind_trap is True
    assert receipt.is_twin_engine_compounder is False
    assert receipt.regime == "DUAL_FACTOR_HEADWIND_TRAP"
    assert receipt.expected_vme_alpha_pct <= -4.0
    assert receipt.verify_integrity() is True


def test_calculate_vme_neutral_factor_exposure() -> None:
    """Test neutral benchmark valuation and momentum exposure."""
    # ~6% return over first 39 sessions
    prices = [100.0 * (1.0 + 0.0015 * i) for i in range(60)]
    receipt = calculate_vme_score(
        ticker="NEUT",
        prices=prices,
        book_to_market=0.50,
        earnings_yield_pct=5.0,
        value_momentum_correlation=-0.50,
    )

    assert receipt.ticker == "NEUT"
    assert abs(receipt.value_z_score) < 1e-4
    assert abs(receipt.momentum_z_score) < 0.25
    assert receipt.regime == "NEUTRAL_FACTOR_EXPOSURE"
    assert receipt.verify_integrity() is True


def test_skip_month_isolation() -> None:
    """Prove that 12-2 intermediate momentum return strictly excludes the most recent 21 sessions."""
    base_prices = [100.0 * (1.0 + 0.004 * i) for i in range(60)]
    # Create a second series identical up to t - 21, then jumping in the last 20 sessions
    shock_prices = list(base_prices)
    for j in range(40, 60):
        shock_prices[j] = shock_prices[j] * 1.15

    rec_base = calculate_vme_score("SKIP", base_prices, value_momentum_correlation=-0.50)
    rec_shock = calculate_vme_score("SKIP", shock_prices, value_momentum_correlation=-0.50)

    # 12-2 intermediate momentum percentage must be identical because skip_idx = 60 - 21 = 39
    assert rec_base.momentum_12_2_pct == rec_shock.momentum_12_2_pct
    assert rec_shock.short_term_1m_pct > rec_base.short_term_1m_pct


def test_diversification_benefit_ratio_math() -> None:
    """Verify exact mathematical identity DBR = sqrt(2 / (1 + rho))."""
    prices = [100.0 + 0.5 * i for i in range(50)]

    # When rho = -0.50, 1 + rho = 0.50, 2 / 0.50 = 4.0 -> sqrt(4.0) = 2.00
    r_neg = calculate_vme_score("DBR", prices, value_momentum_correlation=-0.50)
    assert abs(r_neg.diversification_benefit_ratio - 2.00) < 1e-4

    # When rho = 0.00, 1 + rho = 1.00, 2 / 1.00 = 2.0 -> sqrt(2.0) = 1.4142
    r_zero = calculate_vme_score("DBR", prices, value_momentum_correlation=0.00)
    assert abs(r_zero.diversification_benefit_ratio - math.sqrt(2.0)) < 1e-3


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 signature verification and tamper detection."""
    prices = [100.0 + 0.4 * i for i in range(50)]
    receipt = calculate_vme_score("HMAC", prices)

    assert receipt.verify_integrity() is True

    # Tamper with combo score
    receipt.vme_combo_z_score = 3.99
    assert receipt.verify_integrity() is False


def test_classify_vme_regime() -> None:
    """Test boundary conditions for all VME regimes."""
    assert classify_vme_regime(1.0, 1.0, 1.5) == "SYNERGISTIC_VME_COMPOUNDER"
    assert classify_vme_regime(1.2, -0.3, 0.6) == "HEDGED_BARBELL_EQUILIBRIUM"
    assert classify_vme_regime(0.0, 0.0, 0.0) == "NEUTRAL_FACTOR_EXPOSURE"
    assert classify_vme_regime(-1.0, -1.0, -1.2) == "DUAL_FACTOR_HEADWIND_TRAP"


def test_evaluate_vme_profile_database(tmp_path: Path) -> None:
    """Test end-to-end evaluation using seeded SQLite database observations."""
    test_db = tmp_path / "test_stock.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="VMET",
        exchange="XNYS",
        company_name="VME Test Corp.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(45):
        p = 100.0 + i * 0.6
        records.append(
            {
                "ticker": "VMET",
                "exchange": "XNYS",
                "session_date": f"2026-01-{(i % 28) + 1:02d}" if i < 28 else f"2026-02-{(i - 28) + 1:02d}",
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "adj_close": p,
                "volume": 600_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    receipt = evaluate_vme_profile(
        ticker="VMET",
        db_path=test_db,
        book_to_market=0.75,
        earnings_yield_pct=7.5,
    )

    assert receipt.ticker == "VMET"
    assert abs(receipt.latest_price - (100.0 + 44 * 0.6)) < 1e-2
    assert receipt.momentum_12_2_pct > 0
    assert receipt.verify_integrity() is True


def test_evaluate_vme_profile_fallback() -> None:
    """Test deterministic fallback when ticker has no database records."""
    receipt = evaluate_vme_profile(
        ticker="XYZ",
        book_to_market=0.80,
        earnings_yield_pct=8.0,
        value_momentum_correlation=-0.52,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price > 0
    assert receipt.book_to_market == 0.80
    assert receipt.verify_integrity() is True


def test_cli_vme_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI vme command output formatting and JSON mode."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        book_to_market=0.65,
        earnings_yield=6.5,
        correlation=-0.52,
        json=False,
    )
    vme_command(args_text)
    captured = capsys.readouterr()
    assert "Asness, Moskowitz & Pedersen (2013) VME Unified Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Diversification Benefit Ratio" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        book_to_market=0.65,
        earnings_yield=6.5,
        correlation=-0.52,
        json=True,
    )
    vme_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "AAPL"
    assert "vme_combo_z_score" in data
    assert "signature" in data


def test_input_validation_vme() -> None:
    """Test strict security input validation on ticker, price lengths, and finite numbers."""
    prices = [100.0 + i for i in range(30)]

    with pytest.raises(SecurityValidationError):
        calculate_vme_score("BAD$TICKER", prices)

    with pytest.raises(ValueError, match="At least 25 price observations"):
        calculate_vme_score("AAPL", [100.0, 101.0, 102.0])

    with pytest.raises(ValueError, match="must be finite"):
        calculate_vme_score("AAPL", prices, book_to_market=float("nan"))
