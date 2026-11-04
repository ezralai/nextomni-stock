"""Unit and integration tests for Victor L. Bernard & Jacob K. Thomas (1989, 1990) PEAD & SUE Engine."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.pead_sue import (
    PEADSUEReceipt,
    calculate_pead_sue,
    classify_pead_regime,
    evaluate_pead_sue_profile,
    map_sue_to_decile,
)
from src.cli import pead_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_calculate_pead_sue_high_positive_drift() -> None:
    """Test top-decile positive SUE within active 60-day drift window."""
    # EPS_t = 2.00, EPS_{t-4} = 1.50, mu = 0.02, sigma = 0.15 -> SUE = 0.48 / 0.15 = +3.20
    receipt = calculate_pead_sue(
        ticker="BEAT",
        latest_price=120.0,
        current_eps=2.00,
        year_ago_eps=1.50,
        surprise_std_eps=0.15,
        drift_mean_eps=0.02,
        days_since_announcement=5,
    )

    assert receipt.ticker == "BEAT"
    assert receipt.sue_score >= 2.0
    assert receipt.sue_decile == 10
    assert receipt.predicted_next_quarter_sue > 0.0
    assert receipt.predicted_lag4_reversal_sue < 0.0
    assert receipt.remaining_drift_pct > 4.0
    assert receipt.is_active_positive_drift is True
    assert receipt.is_active_negative_slump is False
    assert receipt.regime == "HIGH_SUE_POSITIVE_DRIFT"
    assert receipt.expected_pead_alpha_pct > 8.0
    assert receipt.verify_integrity() is True


def test_calculate_pead_sue_low_negative_slump() -> None:
    """Test bottom-decile negative SUE within active 60-day slump window."""
    # EPS_t = 1.00, EPS_{t-4} = 1.50, mu = 0.02, sigma = 0.15 -> SUE = -0.52 / 0.15 = -3.4667
    receipt = calculate_pead_sue(
        ticker="MISS",
        latest_price=80.0,
        current_eps=1.00,
        year_ago_eps=1.50,
        surprise_std_eps=0.15,
        drift_mean_eps=0.02,
        days_since_announcement=5,
    )

    assert receipt.ticker == "MISS"
    assert receipt.sue_score <= -2.0
    assert receipt.sue_decile == 1
    assert receipt.predicted_next_quarter_sue < 0.0
    assert receipt.predicted_lag4_reversal_sue > 0.0
    assert receipt.remaining_drift_pct < -4.0
    assert receipt.is_active_negative_slump is True
    assert receipt.is_active_positive_drift is False
    assert receipt.regime == "LOW_SUE_NEGATIVE_SLUMP"
    assert receipt.expected_pead_alpha_pct < -8.0
    assert receipt.verify_integrity() is True


def test_calculate_pead_sue_exhausted_late_cycle_drift() -> None:
    """Test late-cycle drift exhaustion when days_since_announcement > 50."""
    receipt = calculate_pead_sue(
        ticker="LATE",
        latest_price=110.0,
        current_eps=1.90,
        year_ago_eps=1.50,
        surprise_std_eps=0.15,
        drift_mean_eps=0.02,
        days_since_announcement=65,
    )

    assert receipt.sue_score >= 2.0
    assert receipt.regime == "EXHAUSTED_LATE_CYCLE_DRIFT"
    assert receipt.is_active_positive_drift is False
    assert receipt.remaining_drift_pct < receipt.initial_60d_drift_pct * 0.25
    assert receipt.verify_integrity() is True


def test_calculate_pead_sue_neutral_equilibrium() -> None:
    """Test mild surprise near seasonal random walk expectation."""
    receipt = calculate_pead_sue(
        ticker="NEUT",
        latest_price=100.0,
        current_eps=1.55,
        year_ago_eps=1.50,
        surprise_std_eps=0.15,
        drift_mean_eps=0.02,
        days_since_announcement=10,
    )

    assert abs(receipt.sue_score) < 1.0
    assert receipt.regime == "NEUTRAL_EARNINGS_EQUILIBRIUM"
    assert receipt.is_active_positive_drift is False
    assert receipt.is_active_negative_slump is False
    assert receipt.verify_integrity() is True


def test_exponential_drift_decay_monotonicity() -> None:
    """Verify remaining drift decays monotonically as sessions since announcement increase."""
    drifts = []
    for d in [0, 15, 30, 45, 60, 80]:
        rec = calculate_pead_sue(
            ticker="DCAY",
            latest_price=100.0,
            current_eps=1.95,
            year_ago_eps=1.50,
            days_since_announcement=d,
        )
        drifts.append(rec.remaining_drift_pct)

    for i in range(len(drifts) - 1):
        assert drifts[i] > drifts[i + 1]


def test_bernard_thomas_ar4_autocorrelation_identities() -> None:
    """Verify exact Bernard & Thomas (1990) phi_1 = +0.34 and phi_4 = -0.24 identities."""
    rec = calculate_pead_sue(
        ticker="AUTO",
        latest_price=100.0,
        current_eps=1.82,
        year_ago_eps=1.50,
        surprise_std_eps=0.15,
        drift_mean_eps=0.02,
        days_since_announcement=10,
    )
    # SUE = (1.82 - 1.50 - 0.02) / 0.15 = 2.00
    assert abs(rec.sue_score - 2.00) < 1e-4
    assert abs(rec.predicted_next_quarter_sue - 0.68) < 1e-4
    assert abs(rec.predicted_lag4_reversal_sue - (-0.48)) < 1e-4


def test_map_sue_to_decile_boundaries() -> None:
    """Test all 10 SUE decile boundaries."""
    assert map_sue_to_decile(2.5) == 10
    assert map_sue_to_decile(1.5) == 9
    assert map_sue_to_decile(0.8) == 8
    assert map_sue_to_decile(0.4) == 7
    assert map_sue_to_decile(0.1) == 6
    assert map_sue_to_decile(-0.1) == 5
    assert map_sue_to_decile(-0.4) == 4
    assert map_sue_to_decile(-0.8) == 3
    assert map_sue_to_decile(-1.5) == 2
    assert map_sue_to_decile(-2.5) == 1


def test_receipt_hmac_tamper_resistance() -> None:
    """Test cryptographic HMAC-SHA256 signature verification and tamper detection."""
    receipt = calculate_pead_sue("HMAC", 100.0, 1.80, 1.50)
    assert receipt.verify_integrity() is True

    receipt.sue_score = 5.99
    assert receipt.verify_integrity() is False


def test_evaluate_pead_sue_profile_database(tmp_path: Path) -> None:
    """Test end-to-end evaluation using seeded SQLite database observations."""
    test_db = tmp_path / "test_stock.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="PEAD",
        exchange="XNYS",
        company_name="PEAD Test Inc.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(30):
        p = 100.0 + i * 0.8
        records.append(
            {
                "ticker": "PEAD",
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

    receipt = evaluate_pead_sue_profile(
        ticker="PEAD",
        db_path=test_db,
        current_eps=1.90,
        year_ago_eps=1.50,
    )

    assert receipt.ticker == "PEAD"
    assert abs(receipt.latest_price - (100.0 + 29 * 0.8)) < 1e-2
    assert receipt.sue_score > 1.0
    assert receipt.verify_integrity() is True


def test_evaluate_pead_sue_profile_fallback() -> None:
    """Test deterministic fallback when ticker is not in database."""
    receipt = evaluate_pead_sue_profile(
        ticker="XYZ",
        current_eps=1.80,
        year_ago_eps=1.50,
        days_since_announcement=12,
    )
    assert receipt.ticker == "XYZ"
    assert receipt.latest_price == 100.0
    assert receipt.days_since_announcement == 12
    assert receipt.verify_integrity() is True


def test_cli_pead_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI pead command output formatting and JSON mode."""
    args_text = argparse.Namespace(
        ticker="NVDA",
        current_eps=2.10,
        year_ago_eps=1.50,
        surprise_std=0.15,
        days_since=8,
        json=False,
    )
    pead_command(args_text)
    captured = capsys.readouterr()
    assert "Bernard & Thomas (1989, 1990) PEAD & SUE Autocorrelation Engine" in captured.out
    assert "NVDA" in captured.out
    assert "Standardized Unexpected (SUE)" in captured.out

    args_json = argparse.Namespace(
        ticker="NVDA",
        current_eps=2.10,
        year_ago_eps=1.50,
        surprise_std=0.15,
        days_since=8,
        json=True,
    )
    pead_command(args_json)
    captured_json = capsys.readouterr()
    data = json.loads(captured_json.out)
    assert data["ticker"] == "NVDA"
    assert data["sue_decile"] == 10
    assert "signature" in data


def test_input_validation_pead() -> None:
    """Test strict security input validation on ticker, prices, days, and finite numbers."""
    with pytest.raises(SecurityValidationError):
        calculate_pead_sue("BAD$TICKER", 100.0, 1.80, 1.50)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_pead_sue("AAPL", -10.0, 1.80, 1.50)

    with pytest.raises(ValueError, match="days_since_announcement must be non-negative"):
        calculate_pead_sue("AAPL", 100.0, 1.80, 1.50, days_since_announcement=-5)

    with pytest.raises(ValueError, match="must be finite"):
        calculate_pead_sue("AAPL", 100.0, float("nan"), 1.50)
