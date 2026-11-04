"""Unit and integration tests for Hirshleifer, Hou, Teoh & Zhang (2004) Net Operating Assets (NOA) & Balance Sheet Bloat Engine."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.noa_bloat import (
    NOABloatReceipt,
    calculate_noa_bloat,
    classify_noa_regime,
    evaluate_noa_bloat_profile,
)
from src.cli import noa_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_noa_lean_cash_fortress() -> None:
    """Test lean cash-generative fortress with low SNOA and strong operating liability float."""
    # TA = 10,500M, Prior TA = 10,000M, Cash = 3,500M, Debt = 1,500M, Equity = 5,500M
    # NOA = Equity + Debt - Cash = 5,500 + 1,500 - 3,500 = 3,500M -> SNOA = 0.35
    receipt = calculate_noa_bloat(
        ticker="LEAN",
        latest_price=140.0,
        total_assets=10_500.0,
        prior_total_assets=10_000.0,
        cash_and_equivalents=3_500.0,
        total_debt=1_500.0,
        book_equity=5_500.0,
        prior_snoa=0.38,
    )

    assert receipt.ticker == "LEAN"
    assert abs(receipt.net_operating_assets - 3_500.0) < 1e-2
    assert abs(receipt.scaled_noa_ratio - 0.35) < 1e-4
    assert receipt.delta_snoa < 0.0
    assert receipt.is_lean_cash_fortress is True
    assert receipt.is_bloated_balance_sheet_trap is False
    assert receipt.regime == "LEAN_CASH_GENERATIVE_FORTRESS"
    assert receipt.expected_noa_alpha_pct > 2.5
    assert receipt.verify_integrity() is True


def test_calculate_noa_balanced_operating_structure() -> None:
    """Test balanced operating asset structure (0.45 < SNOA < 0.70)."""
    # NOA = 5,000 + 2,000 - 1,200 = 5,800M -> SNOA = 0.58
    receipt = calculate_noa_bloat(
        ticker="BALN",
        latest_price=100.0,
        total_assets=10_500.0,
        prior_total_assets=10_000.0,
        cash_and_equivalents=1_200.0,
        total_debt=2_000.0,
        book_equity=5_000.0,
        prior_snoa=0.57,
    )

    assert 0.45 < receipt.scaled_noa_ratio < 0.70
    assert receipt.regime == "BALANCED_OPERATING_STRUCTURE"
    assert receipt.is_lean_cash_fortress is False
    assert receipt.is_bloated_balance_sheet_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_noa_elevated_accrual_accumulation() -> None:
    """Test elevated accrual accumulation (0.70 <= SNOA < 0.85)."""
    # NOA = 5,500 + 3,000 - 800 = 7,700M -> SNOA = 0.77
    receipt = calculate_noa_bloat(
        ticker="ELEV",
        latest_price=85.0,
        total_assets=11_000.0,
        prior_total_assets=10_000.0,
        cash_and_equivalents=800.0,
        total_debt=3_000.0,
        book_equity=5_500.0,
        prior_snoa=0.72,
    )

    assert 0.70 <= receipt.scaled_noa_ratio < 0.85
    assert receipt.regime == "ELEVATED_ACCRUAL_ACCUMULATION"
    assert receipt.expected_noa_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_noa_acute_balance_sheet_bloat_trap() -> None:
    """Test acute balance sheet bloat trap (SNOA >= 0.85) hitting the accounting sustainability wall."""
    # NOA = 6,500 + 3,500 - 400 = 9,600M -> SNOA = 0.96
    receipt = calculate_noa_bloat(
        ticker="BLOT",
        latest_price=45.0,
        total_assets=12_000.0,
        prior_total_assets=10_000.0,
        cash_and_equivalents=400.0,
        total_debt=3_500.0,
        book_equity=6_500.0,
        prior_snoa=0.82,
    )

    assert receipt.scaled_noa_ratio >= 0.85
    assert receipt.is_bloated_balance_sheet_trap is True
    assert receipt.is_lean_cash_fortress is False
    assert receipt.regime == "ACUTE_BALANCE_SHEET_BLOAT_TRAP"
    assert receipt.expected_noa_alpha_pct <= -6.0
    assert receipt.verify_integrity() is True


def test_hirshleifer_dual_accounting_identity() -> None:
    """Verify exact accounting equivalence: NOA == OA - OL == Book_Equity + Total_Debt - Cash."""
    ta = 15_000.0
    prior_ta = 14_000.0
    cash = 2_500.0
    debt = 4_000.0
    equity = 7_000.0

    rec = calculate_noa_bloat("IDEN", 100.0, ta, prior_ta, cash, debt, equity)
    assert abs(rec.operating_assets - (ta - cash)) < 1e-4
    assert abs(rec.operating_liabilities - (ta - debt - equity)) < 1e-4
    assert abs(rec.net_operating_assets - (rec.operating_assets - rec.operating_liabilities)) < 1e-4
    assert abs(rec.net_operating_assets - (equity + debt - cash)) < 1e-4


def test_delta_snoa_sensitivity() -> None:
    """Verify that rising YoY SNOA penalizes expected alpha."""
    rec_stable = calculate_noa_bloat("DSNO", 100.0, 10_500.0, 10_000.0, 2_000.0, 2_500.0, 5_000.0, prior_snoa=0.55)
    rec_surging = calculate_noa_bloat("DSNO", 100.0, 10_500.0, 10_000.0, 2_000.0, 2_500.0, 5_000.0, prior_snoa=0.35)

    assert rec_surging.delta_snoa > rec_stable.delta_snoa
    assert rec_surging.expected_noa_alpha_pct < rec_stable.expected_noa_alpha_pct


def test_classify_noa_regime_boundaries() -> None:
    """Test regime classification boundary thresholds."""
    assert classify_noa_regime(0.35, -0.02) == "LEAN_CASH_GENERATIVE_FORTRESS"
    assert classify_noa_regime(0.55, 0.01) == "BALANCED_OPERATING_STRUCTURE"
    assert classify_noa_regime(0.75, 0.04) == "ELEVATED_ACCRUAL_ACCUMULATION"
    assert classify_noa_regime(0.60, 0.15) == "ELEVATED_ACCRUAL_ACCUMULATION"
    assert classify_noa_regime(0.90, 0.05) == "ACUTE_BALANCE_SHEET_BLOAT_TRAP"


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 signature verification and tamper detection."""
    receipt = calculate_noa_bloat("HMAC", 100.0, 10_500.0, 10_000.0, 2_400.0, 2_200.0, 4_200.0)
    assert receipt.verify_integrity() is True

    receipt.scaled_noa_ratio = 1.99
    assert receipt.verify_integrity() is False


def test_evaluate_noa_bloat_profile_database(tmp_path: Path) -> None:
    """Test end-to-end evaluation using seeded SQLite database observations."""
    test_db = tmp_path / "test_stock.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="NOAT",
        exchange="XNYS",
        company_name="NOA Test Corp.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(25):
        p = 100.0 + i * 0.5
        records.append(
            {
                "ticker": "NOAT",
                "exchange": "XNYS",
                "session_date": f"2026-01-{i+1:02d}",
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "adj_close": p,
                "volume": 500_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    receipt = evaluate_noa_bloat_profile(
        ticker="NOAT",
        db_path=test_db,
    )

    assert receipt.ticker == "NOAT"
    assert abs(receipt.latest_price - 112.0) < 1e-2
    assert receipt.scaled_noa_ratio == 0.40
    assert receipt.verify_integrity() is True


def test_evaluate_noa_bloat_profile_fallback() -> None:
    """Test deterministic fallback when ticker is not in database."""
    receipt = evaluate_noa_bloat_profile(
        ticker="XYZ",
        total_assets=11_000.0,
        prior_total_assets=10_000.0,
        cash_and_equivalents=3_000.0,
        total_debt=2_000.0,
        book_equity=4_500.0,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price == 100.0
    assert receipt.scaled_noa_ratio == 0.35
    assert receipt.verify_integrity() is True


def test_cli_noa_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI noa command output formatting and JSON mode."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        total_assets=10_500.0,
        prior_assets=10_000.0,
        cash=2_400.0,
        debt=2_200.0,
        equity=4_200.0,
        prior_snoa=0.41,
        json=False,
    )
    noa_command(args_text)
    captured = capsys.readouterr()
    assert "Hirshleifer, Hou, Teoh & Zhang (2004) NOA Bloat Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Scaled NOA Ratio (SNOA)" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        total_assets=10_500.0,
        prior_assets=10_000.0,
        cash=2_400.0,
        debt=2_200.0,
        equity=4_200.0,
        prior_snoa=0.41,
        json=True,
    )
    noa_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "AAPL"
    assert data["scaled_noa_ratio"] == 0.40
    assert "signature" in data


def test_input_validation_noa() -> None:
    """Test strict security input validation on ticker, assets, cash bounds, and finite numbers."""
    with pytest.raises(SecurityValidationError):
        calculate_noa_bloat("BAD$TICKER", 100.0, 10_000.0, 10_000.0, 2_000.0, 2_000.0, 4_000.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_noa_bloat("AAPL", 0.0, 10_000.0, 10_000.0, 2_000.0, 2_000.0, 4_000.0)

    with pytest.raises(ValueError, match="must be strictly positive"):
        calculate_noa_bloat("AAPL", 100.0, -500.0, 10_000.0, 2_000.0, 2_000.0, 4_000.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_noa_bloat("AAPL", 100.0, 10_000.0, 10_000.0, -100.0, 2_000.0, 4_000.0)

    with pytest.raises(ValueError, match="cannot exceed total_assets"):
        calculate_noa_bloat("AAPL", 100.0, 10_000.0, 10_000.0, 12_000.0, 2_000.0, 4_000.0)

    with pytest.raises(ValueError, match="must be finite"):
        calculate_noa_bloat("AAPL", 100.0, float("nan"), 10_000.0, 2_000.0, 2_000.0, 4_000.0)
