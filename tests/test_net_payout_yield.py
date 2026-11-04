"""Unit and integration tests for Boudoukh et al. (2007) & Pontiff-Woodgate (2008) Net Payout Yield Engine."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.net_payout_yield import (
    NetPayoutYieldReceipt,
    calculate_net_payout_yield,
    classify_payout_regime,
    evaluate_net_payout_profile,
)
from src.cli import payout_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_net_payout_share_cannibal_compounder() -> None:
    """Test share-cannibal compounder with high FCF-backed buybacks and shrinking share count."""
    receipt = calculate_net_payout_yield(
        ticker="CANN",
        latest_price=150.0,
        market_cap=10_000.0,
        dividends=200.0,  # 2.0% DY
        share_repurchases=450.0,  # 4.5% BY
        equity_issuance=50.0,  # 0.5% IY -> NPY = 6.0%
        current_shares=95.0,
        prior_shares=100.0,  # CSI = -5.13%
        free_cash_flow=900.0,  # PCR = 900 / 650 = 1.38x
    )

    assert receipt.ticker == "CANN"
    assert abs(receipt.net_payout_yield_pct - 6.0) < 1e-4
    assert receipt.composite_share_issuance_pct <= -1.50
    assert receipt.fcf_payout_coverage_ratio >= 1.0
    assert receipt.is_share_cannibal is True
    assert receipt.is_unsustainable_debt_payout is False
    assert receipt.is_dilution_trap is False
    assert receipt.regime == "SHARE_CANNIBAL_COMPOUNDER"
    assert receipt.expected_payout_alpha_pct >= 6.0
    assert receipt.verify_integrity() is True


def test_calculate_net_payout_unsustainable_debt_trap() -> None:
    """Test debt-funded payout trap where distributions exceed organic Free Cash Flow."""
    receipt = calculate_net_payout_yield(
        ticker="DEBT",
        latest_price=80.0,
        market_cap=10_000.0,
        dividends=350.0,
        share_repurchases=200.0,  # Gross dist = 5.5%
        equity_issuance=20.0,
        current_shares=98.0,
        prior_shares=100.0,
        free_cash_flow=200.0,  # PCR = 200 / 550 = 0.36x < 0.75x
    )

    assert receipt.fcf_payout_coverage_ratio < 0.75
    assert receipt.is_unsustainable_debt_payout is True
    assert receipt.is_share_cannibal is False
    assert receipt.regime == "UNSUSTAINABLE_DEBT_FUNDED_PAYOUT"
    assert receipt.expected_payout_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_net_payout_dilution_trap() -> None:
    """Test shareholder dilution trap with heavy equity issuance and expanding share count."""
    receipt = calculate_net_payout_yield(
        ticker="DILU",
        latest_price=40.0,
        market_cap=10_000.0,
        dividends=0.0,
        share_repurchases=50.0,
        equity_issuance=450.0,  # NPY = -4.0%
        current_shares=106.0,
        prior_shares=100.0,  # CSI = +5.83%
        free_cash_flow=100.0,
    )

    assert receipt.net_payout_yield_pct == -4.0
    assert receipt.composite_share_issuance_pct >= 2.50
    assert receipt.is_dilution_trap is True
    assert receipt.is_share_cannibal is False
    assert receipt.regime == "SHAREHOLDER_DILUTION_TRAP"
    assert receipt.expected_payout_alpha_pct <= -6.0
    assert receipt.verify_integrity() is True


def test_calculate_net_payout_balanced_distribution() -> None:
    """Test moderate sustainable capital return profile."""
    receipt = calculate_net_payout_yield(
        ticker="BALN",
        latest_price=100.0,
        market_cap=10_000.0,
        dividends=150.0,
        share_repurchases=120.0,
        equity_issuance=30.0,  # NPY = +2.4%
        current_shares=99.2,
        prior_shares=100.0,
        free_cash_flow=500.0,
    )

    assert receipt.regime == "BALANCED_CAPITAL_DISTRIBUTION"
    assert receipt.is_share_cannibal is False
    assert receipt.is_dilution_trap is False
    assert receipt.is_unsustainable_debt_payout is False
    assert receipt.verify_integrity() is True


def test_boudoukh_npy_identity_math() -> None:
    """Verify exact Boudoukh et al. (2007) identity NPY = DY + BY - IY."""
    rec = calculate_net_payout_yield(
        ticker="MATH",
        latest_price=100.0,
        market_cap=20_000.0,
        dividends=400.0,  # 2.0%
        share_repurchases=700.0,  # 3.5%
        equity_issuance=200.0,  # 1.0%
        current_shares=97.5,
        prior_shares=100.0,
        free_cash_flow=1_500.0,
    )
    assert abs(rec.dividend_yield_pct - 2.0) < 1e-4
    assert abs(rec.gross_buyback_yield_pct - 3.5) < 1e-4
    assert abs(rec.gross_issuance_yield_pct - 1.0) < 1e-4
    assert abs(rec.net_payout_yield_pct - 4.5) < 1e-4


def test_pontiff_woodgate_csi_log_identity() -> None:
    """Verify exact Pontiff & Woodgate (2008) log share change formula CSI = ln(S_t / S_{t-1}) * 100."""
    rec = calculate_net_payout_yield(
        ticker="CSIM",
        latest_price=100.0,
        market_cap=10_000.0,
        dividends=100.0,
        share_repurchases=200.0,
        equity_issuance=50.0,
        current_shares=90.0,
        prior_shares=100.0,
        free_cash_flow=500.0,
    )
    expected_csi = round(math.log(90.0 / 100.0) * 100.0, 2)
    assert rec.composite_share_issuance_pct == expected_csi


def test_classify_payout_regime_boundaries() -> None:
    """Test regime classification boundary thresholds."""
    assert classify_payout_regime(5.5, -3.0, 1.4, 6.0) == "SHARE_CANNIBAL_COMPOUNDER"
    assert classify_payout_regime(4.5, -2.0, 0.5, 5.0) == "UNSUSTAINABLE_DEBT_FUNDED_PAYOUT"
    assert classify_payout_regime(-1.5, 1.0, 1.2, 1.0) == "SHAREHOLDER_DILUTION_TRAP"
    assert classify_payout_regime(1.0, 3.0, 1.2, 2.0) == "SHAREHOLDER_DILUTION_TRAP"
    assert classify_payout_regime(2.5, -0.5, 1.5, 3.0) == "BALANCED_CAPITAL_DISTRIBUTION"


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 signature verification and tamper detection."""
    receipt = calculate_net_payout_yield(
        "HMAC", 100.0, 10_000.0, 200.0, 300.0, 50.0, 97.0, 100.0, 700.0
    )
    assert receipt.verify_integrity() is True

    receipt.net_payout_yield_pct = 19.99
    assert receipt.verify_integrity() is False


def test_evaluate_net_payout_profile_database(tmp_path: Path) -> None:
    """Test end-to-end evaluation using seeded SQLite database observations."""
    test_db = tmp_path / "test_stock.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="PAYT",
        exchange="XNYS",
        company_name="Payout Test Corp.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(25):
        p = 100.0 + i * 1.0
        records.append(
            {
                "ticker": "PAYT",
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

    receipt = evaluate_net_payout_profile(
        ticker="PAYT",
        db_path=test_db,
    )

    assert receipt.ticker == "PAYT"
    assert abs(receipt.latest_price - 124.0) < 1e-2
    assert receipt.net_payout_yield_pct > 0.0
    assert receipt.verify_integrity() is True


def test_evaluate_net_payout_profile_fallback() -> None:
    """Test deterministic fallback when ticker is not in database."""
    receipt = evaluate_net_payout_profile(
        ticker="XYZ",
        market_cap=10_000.0,
        dividends=200.0,
        share_repurchases=400.0,
        equity_issuance=50.0,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price == 100.0
    assert receipt.net_payout_yield_pct == 5.5
    assert receipt.verify_integrity() is True


def test_cli_payout_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI payout command output formatting and JSON mode."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        market_cap=10_000.0,
        dividends=180.0,
        repurchases=420.0,
        issuance=50.0,
        current_shares=96.0,
        prior_shares=100.0,
        free_cash_flow=850.0,
        json=False,
    )
    payout_command(args_text)
    captured = capsys.readouterr()
    assert "Boudoukh et al. (2007) & Pontiff-Woodgate (2008) Payout Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Total Net Payout Yield (NPY)" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        market_cap=10_000.0,
        dividends=180.0,
        repurchases=420.0,
        issuance=50.0,
        current_shares=96.0,
        prior_shares=100.0,
        free_cash_flow=850.0,
        json=True,
    )
    payout_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "AAPL"
    assert data["net_payout_yield_pct"] == 5.5
    assert "signature" in data


def test_input_validation_payout() -> None:
    """Test strict security input validation on ticker, market cap, shares, and finite numbers."""
    with pytest.raises(SecurityValidationError):
        calculate_net_payout_yield("BAD$TICKER", 100.0, 10_000.0, 100.0, 200.0, 50.0, 98.0, 100.0, 500.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_net_payout_yield("AAPL", 0.0, 10_000.0, 100.0, 200.0, 50.0, 98.0, 100.0, 500.0)

    with pytest.raises(ValueError, match="market_cap must be strictly positive"):
        calculate_net_payout_yield("AAPL", 100.0, -500.0, 100.0, 200.0, 50.0, 98.0, 100.0, 500.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_net_payout_yield("AAPL", 100.0, 10_000.0, -50.0, 200.0, 50.0, 98.0, 100.0, 500.0)

    with pytest.raises(ValueError, match="Share counts must be strictly positive"):
        calculate_net_payout_yield("AAPL", 100.0, 10_000.0, 100.0, 200.0, 50.0, 0.0, 100.0, 500.0)

    with pytest.raises(ValueError, match="must be finite"):
        calculate_net_payout_yield("AAPL", 100.0, float("nan"), 100.0, 200.0, 50.0, 98.0, 100.0, 500.0)
