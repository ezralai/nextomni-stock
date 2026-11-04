"""Unit and integration tests for Mark Grinblatt & Bing Han (2005) / Andrea Frazzini (2006)
Prospect Theory Reference Point, Capital Gains Overhang (CGO) & Disposition Effect Engine (Iteration 95).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.capital_gains_overhang import (
    CapitalGainsOverhangReceipt,
    calculate_capital_gains_overhang,
    classify_cgo_regime,
    evaluate_capital_gains_overhang_profile,
)
from src.cli import cgo_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_gain_overhang_continuation_leader_regime() -> None:
    """Verify positive CGO (>= 0.08) aligned with positive news classifies as GAIN_OVERHANG_CONTINUATION_LEADER."""
    rec = calculate_capital_gains_overhang(
        ticker="AAPL",
        latest_price=200.0,
        aggregate_reference_price=160.0,
        avg_daily_turnover=0.006,
        news_surprise_score=1.40,
        institutional_rationality_shield=0.45,
    )
    assert abs(rec.capital_gains_overhang - 0.20) < 1e-6
    assert rec.news_overhang_alignment_mult > 1.5
    assert rec.frazzini_disposition_impulse >= 0.18
    assert rec.regime == "GAIN_OVERHANG_CONTINUATION_LEADER"
    assert rec.is_gain_overhang_leader is True
    assert rec.is_loss_lock_in_trap is False
    assert rec.expected_cgo_alpha_pct > 5.0
    assert rec.verify_integrity() is True


def test_loss_aversion_lock_in_trap_regime() -> None:
    """Verify deep negative CGO (<= -0.10) aligned with negative news triggers LOSS_AVERSION_LOCK_IN_TRAP."""
    rec = calculate_capital_gains_overhang(
        ticker="INTC",
        latest_price=22.0,
        aggregate_reference_price=33.0,
        avg_daily_turnover=0.008,
        news_surprise_score=-1.80,
        institutional_rationality_shield=0.35,
    )
    # CGO = (22 - 33) / 22 = -0.50
    assert abs(rec.capital_gains_overhang - (-0.50)) < 1e-6
    assert rec.frazzini_disposition_impulse <= -0.22
    assert rec.regime == "LOSS_AVERSION_LOCK_IN_TRAP"
    assert rec.is_loss_lock_in_trap is True
    assert rec.is_gain_overhang_leader is False
    assert rec.expected_cgo_alpha_pct < -8.0
    assert rec.verify_integrity() is True


def test_gain_harvesting_supply_watch_regime() -> None:
    """Verify positive CGO (>= 0.08) paired with contradicting negative news (< -0.50) classifies as watch."""
    rec = calculate_capital_gains_overhang(
        ticker="NVDA",
        latest_price=140.0,
        aggregate_reference_price=115.0,
        avg_daily_turnover=0.015,
        news_surprise_score=-1.10,
        institutional_rationality_shield=0.50,
    )
    assert rec.capital_gains_overhang >= 0.08
    assert rec.news_overhang_alignment_mult == 1.0
    assert rec.regime == "GAIN_HARVESTING_SUPPLY_WATCH"
    assert rec.is_gain_overhang_leader is False
    assert rec.is_loss_lock_in_trap is False


def test_anchored_reference_equilibrium_regime() -> None:
    """Verify price near aggregate reference cost basis (|CGO| < 0.08) classifies as equilibrium."""
    rec = calculate_capital_gains_overhang(
        ticker="MSFT",
        latest_price=420.0,
        aggregate_reference_price=412.0,
        avg_daily_turnover=0.007,
        news_surprise_score=0.20,
        institutional_rationality_shield=0.60,
    )
    assert abs(rec.capital_gains_overhang) < 0.08
    assert rec.regime == "ANCHORED_REFERENCE_EQUILIBRIUM"
    assert rec.is_gain_overhang_leader is False
    assert rec.is_loss_lock_in_trap is False


def test_frazzini_news_alignment_amplification() -> None:
    """Verify Frazzini (2006) theorem: same-sign news and CGO amplify disposition impulse vs opposite-sign."""
    rec_opposed = calculate_capital_gains_overhang(
        ticker="AAPL",
        latest_price=200.0,
        aggregate_reference_price=165.0,
        news_surprise_score=-0.30,
    )
    rec_aligned = calculate_capital_gains_overhang(
        ticker="AAPL",
        latest_price=200.0,
        aggregate_reference_price=165.0,
        news_surprise_score=2.00,
    )
    assert rec_aligned.news_overhang_alignment_mult > rec_opposed.news_overhang_alignment_mult
    assert rec_aligned.frazzini_disposition_impulse > rec_opposed.frazzini_disposition_impulse
    assert rec_aligned.expected_cgo_alpha_pct > rec_opposed.expected_cgo_alpha_pct


def test_institutional_shield_dampening_monotonicity() -> None:
    """Verify higher institutional rationality shield dampens disposition underreaction magnitude."""
    rec_retail = calculate_capital_gains_overhang(
        ticker="TSLA",
        latest_price=250.0,
        aggregate_reference_price=190.0,
        news_surprise_score=1.20,
        institutional_rationality_shield=0.10,
    )
    rec_inst = calculate_capital_gains_overhang(
        ticker="TSLA",
        latest_price=250.0,
        aggregate_reference_price=190.0,
        news_surprise_score=1.20,
        institutional_rationality_shield=0.95,
    )
    assert rec_retail.frazzini_disposition_impulse > rec_inst.frazzini_disposition_impulse


def test_classify_cgo_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_cgo_regime."""
    assert classify_cgo_regime(0.15, 0.80, 0.45) == "GAIN_OVERHANG_CONTINUATION_LEADER"
    assert classify_cgo_regime(0.15, -0.80, 0.25) == "GAIN_HARVESTING_SUPPLY_WATCH"
    assert classify_cgo_regime(-0.20, -1.20, -0.55) == "LOSS_AVERSION_LOCK_IN_TRAP"
    assert classify_cgo_regime(0.02, 0.10, 0.04) == "ANCHORED_REFERENCE_EQUILIBRIUM"


def test_evaluate_capital_gains_overhang_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_capital_gains_overhang_profile computes turnover-weighted R_t from SQLite OHLCV."""
    db_file = tmp_path / "cgo_test.db"
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
        p = 160.0 + i * 1.5
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

    rec = evaluate_capital_gains_overhang_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert rec.latest_price > rec.aggregate_reference_price
    assert rec.capital_gains_overhang > 0.08
    assert rec.regime == "GAIN_OVERHANG_CONTINUATION_LEADER"
    assert rec.verify_integrity() is True


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, reference_price, turnover, news_surprise, and shield raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_capital_gains_overhang("AAPL", 0.0)
    with pytest.raises(ValueError, match="aggregate_reference_price must be strictly positive"):
        calculate_capital_gains_overhang("AAPL", 150.0, aggregate_reference_price=0.0)
    with pytest.raises(ValueError, match="avg_daily_turnover must be in"):
        calculate_capital_gains_overhang("AAPL", 150.0, avg_daily_turnover=0.80)
    with pytest.raises(ValueError, match="news_surprise_score must be in"):
        calculate_capital_gains_overhang("AAPL", 150.0, news_surprise_score=6.0)
    with pytest.raises(ValueError, match="institutional_rationality_shield must be in"):
        calculate_capital_gains_overhang("AAPL", 150.0, institutional_rationality_shield=-0.1)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_capital_gains_overhang(malicious, 150.0)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_capital_gains_overhang(
        ticker="GOOG",
        latest_price=180.0,
        aggregate_reference_price=155.0,
    )
    assert rec.verify_integrity() is True
    tampered = CapitalGainsOverhangReceipt(
        **{**rec.to_dict(), "expected_cgo_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_cgo_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI cgo_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        reference_price=160.0,
        daily_turnover=0.006,
        news_surprise=1.40,
        institutional_disposition_shield=0.45,
        json=False,
    )
    cgo_command(args_text)
    captured = capsys.readouterr()
    assert "Grinblatt & Han (2005) & Frazzini (2006) CGO Engine" in captured.out
    assert "AAPL" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        reference_price=80.0,
        daily_turnover=0.006,
        news_surprise=1.40,
        institutional_disposition_shield=0.45,
        json=True,
    )
    cgo_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "GAIN_OVERHANG_CONTINUATION_LEADER"
    assert payload["is_gain_overhang_leader"] is True
    assert len(payload["signature"]) == 64

