"""Unit and integration tests for Zhi Da, Umit G. Gurun & Mitch Warachka (2014)
'Frog in the Pan' (FIP) Continuous Information Arrival & Momentum Quality Engine (Iteration 92).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.frog_in_the_pan import (
    FrogInThePanReceipt,
    calculate_frog_in_the_pan,
    classify_fip_regime,
    evaluate_frog_in_the_pan_profile,
)
from src.cli import fip_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_continuous_boiling_frog_winner_regime() -> None:
    """Verify positive return with frequent small positive days (ID <= -0.15) classifies as CONTINUOUS_BOILING_FROG_WINNER."""
    rec = calculate_frog_in_the_pan(
        ticker="AAPL",
        latest_price=195.0,
        past_cumulative_return=0.22,
        pct_positive_days=0.68,
        pct_negative_days=0.30,
        max_three_day_jump_share=0.18,
    )
    assert abs(rec.information_discreteness_id - (-0.38)) < 1e-6
    assert rec.fip_momentum_quality_score >= 0.62
    assert rec.regime == "CONTINUOUS_BOILING_FROG_WINNER"
    assert rec.is_continuous_frog_winner is True
    assert rec.is_continuous_frog_loser_trap is False
    assert rec.expected_fip_alpha_pct > 2.0
    assert rec.verify_integrity() is True


def test_discrete_jump_exhaustion_watch_regime() -> None:
    """Verify positive cumulative return achieved via few massive jumps (ID > 0 or high jump share) triggers exhaustion watch."""
    rec = calculate_frog_in_the_pan(
        ticker="NVDA",
        latest_price=140.0,
        past_cumulative_return=0.25,
        pct_positive_days=0.38,
        pct_negative_days=0.58,
        max_three_day_jump_share=0.68,
    )
    assert rec.information_discreteness_id > 0.0
    assert rec.regime == "DISCRETE_JUMP_EXHAUSTION_WATCH"
    assert rec.is_continuous_frog_winner is False
    assert rec.is_continuous_frog_loser_trap is False
    assert rec.expected_fip_alpha_pct < 0.0
    assert rec.verify_integrity() is True


def test_continuous_boiling_frog_loser_trap_regime() -> None:
    """Verify negative cumulative return with frequent small negative days (ID <= -0.15) triggers loser trap."""
    rec = calculate_frog_in_the_pan(
        ticker="INTC",
        latest_price=24.0,
        past_cumulative_return=-0.28,
        pct_positive_days=0.28,
        pct_negative_days=0.68,
        max_three_day_jump_share=0.20,
    )
    # sign(-0.28) * (0.68 - 0.28) = -1 * 0.40 = -0.40 <= -0.15
    assert abs(rec.information_discreteness_id - (-0.40)) < 1e-6
    assert rec.regime == "CONTINUOUS_BOILING_FROG_LOSER_TRAP"
    assert rec.is_continuous_frog_loser_trap is True
    assert rec.is_continuous_frog_winner is False
    assert rec.expected_fip_alpha_pct < -2.0
    assert rec.verify_integrity() is True


def test_neutral_information_arrival_regime() -> None:
    """Verify mild cumulative return (< 4%) classifies as NEUTRAL_INFORMATION_ARRIVAL."""
    rec = calculate_frog_in_the_pan(
        ticker="MSFT",
        latest_price=420.0,
        past_cumulative_return=0.015,
        pct_positive_days=0.52,
        pct_negative_days=0.46,
        max_three_day_jump_share=0.25,
    )
    assert rec.regime == "NEUTRAL_INFORMATION_ARRIVAL"
    assert rec.is_continuous_frog_winner is False
    assert rec.is_continuous_frog_loser_trap is False


def test_same_pret_continuous_vs_discrete_alpha_divergence() -> None:
    """Verify Da-Gurun-Warachka (2014) core theorem: identical +30% PRET yields much higher alpha when continuous than discrete."""
    rec_cont = calculate_frog_in_the_pan(
        ticker="AAPL",
        latest_price=190.0,
        past_cumulative_return=0.30,
        pct_positive_days=0.72,
        pct_negative_days=0.25,
        max_three_day_jump_share=0.15,
    )
    rec_disc = calculate_frog_in_the_pan(
        ticker="AAPL",
        latest_price=190.0,
        past_cumulative_return=0.30,
        pct_positive_days=0.35,
        pct_negative_days=0.62,
        max_three_day_jump_share=0.75,
    )
    assert rec_cont.information_discreteness_id < rec_disc.information_discreteness_id
    assert rec_cont.fip_momentum_quality_score > rec_disc.fip_momentum_quality_score
    assert rec_cont.expected_fip_alpha_pct > rec_disc.expected_fip_alpha_pct


def test_classify_fip_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_fip_regime."""
    assert classify_fip_regime(0.12, -0.30, 0.20, 0.72) == "CONTINUOUS_BOILING_FROG_WINNER"
    assert classify_fip_regime(0.12, 0.15, 0.40, 0.48) == "DISCRETE_JUMP_EXHAUSTION_WATCH"
    assert classify_fip_regime(-0.15, -0.32, 0.20, 0.70) == "CONTINUOUS_BOILING_FROG_LOSER_TRAP"
    assert classify_fip_regime(0.01, -0.05, 0.25, 0.60) == "NEUTRAL_INFORMATION_ARRIVAL"


def test_zero_pret_sign_handling() -> None:
    """Verify exact zero past_cumulative_return resolves sign deterministically without NaN."""
    rec = calculate_frog_in_the_pan(
        ticker="AMD",
        latest_price=150.0,
        past_cumulative_return=0.0,
        pct_positive_days=0.55,
        pct_negative_days=0.40,
        max_three_day_jump_share=0.25,
    )
    assert abs(rec.information_discreteness_id - (-0.15)) < 1e-6
    assert abs(rec.pct_zero_days - 0.05) < 1e-6
    assert rec.verify_integrity() is True


def test_evaluate_frog_in_the_pan_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_frog_in_the_pan_profile computes daily sign proportions from SQLite OHLCV history."""
    db_file = tmp_path / "fip_test.db"
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
    for i in range(25):
        p = 180.0 + i * 0.8
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

    rec = evaluate_frog_in_the_pan_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert rec.pct_positive_days == 1.0
    assert rec.information_discreteness_id == -1.0
    assert rec.regime == "CONTINUOUS_BOILING_FROG_WINNER"
    assert rec.verify_integrity() is True


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, PRET, daily sign fractions, simplex sum > 1, and jump share raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_frog_in_the_pan("AAPL", 0.0)
    with pytest.raises(ValueError, match="past_cumulative_return must be in"):
        calculate_frog_in_the_pan("AAPL", 150.0, past_cumulative_return=-1.2)
    with pytest.raises(ValueError, match="pct_positive_days must be in"):
        calculate_frog_in_the_pan("AAPL", 150.0, pct_positive_days=1.2)
    with pytest.raises(ValueError, match="pct_positive_days \\+ pct_negative_days cannot exceed 1.0"):
        calculate_frog_in_the_pan(
            "AAPL", 150.0, pct_positive_days=0.65, pct_negative_days=0.50
        )
    with pytest.raises(ValueError, match="max_three_day_jump_share must be in"):
        calculate_frog_in_the_pan("AAPL", 150.0, max_three_day_jump_share=1.5)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_frog_in_the_pan(malicious, 150.0)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_frog_in_the_pan(
        ticker="GOOG",
        latest_price=175.0,
        past_cumulative_return=0.18,
    )
    assert rec.verify_integrity() is True
    tampered = FrogInThePanReceipt(
        **{**rec.to_dict(), "expected_fip_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_fip_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI fip_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        pret=0.22,
        pct_positive=0.68,
        pct_negative=0.30,
        max_jump_share=0.18,
        json=False,
    )
    fip_command(args_text)
    captured = capsys.readouterr()
    assert "Da, Gurun & Warachka (2014) 'Frog in the Pan' (FIP) Engine" in captured.out
    assert "AAPL" in captured.out
    assert "CONTINUOUS_BOILING_FROG_WINNER" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        pret=0.22,
        pct_positive=0.68,
        pct_negative=0.30,
        max_jump_share=0.18,
        json=True,
    )
    fip_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "CONTINUOUS_BOILING_FROG_WINNER"
    assert payload["is_continuous_frog_winner"] is True
    assert len(payload["signature"]) == 64

