"""Unit and integration tests for Diether, Malloy & Scherbina (2002) / Miller (1977)
Analyst Forecast Dispersion & Divergence of Opinion Engine (Iteration 87).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.belief_dispersion import (
    BeliefDispersionReceipt,
    calculate_belief_dispersion,
    classify_dispersion_regime,
    evaluate_belief_dispersion_profile,
)
from src.cli import dispersion_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_high_consensus_compounder_regime() -> None:
    """Verify low dispersion and low short-sale friction classify as HIGH_CONSENSUS_COMPOUNDER."""
    rec = calculate_belief_dispersion(
        ticker="AAPL",
        latest_price=185.0,
        consensus_mean_eps=6.50,
        forecast_std_eps=0.18,
        analyst_count=24,
        short_borrow_fee_pct=0.25,
        institutional_ownership_pct=84.0,
    )
    assert rec.dispersion_ratio <= 0.05
    assert rec.miller_overpricing_wedge <= 0.25
    assert rec.regime == "HIGH_CONSENSUS_COMPOUNDER"
    assert rec.is_high_consensus_compounder is True
    assert rec.is_miller_overpricing_trap is False
    assert rec.expected_dispersion_alpha_pct > 2.0
    assert rec.verify_integrity() is True


def test_miller_overpricing_trap_regime() -> None:
    """Verify wide dispersion coupled with binding short-sale friction triggers MILLER_OVERPRICING_TRAP."""
    rec = calculate_belief_dispersion(
        ticker="NVDA",
        latest_price=135.0,
        consensus_mean_eps=1.20,
        forecast_std_eps=0.55,
        analyst_count=16,
        short_borrow_fee_pct=8.50,
        institutional_ownership_pct=20.0,
    )
    assert rec.dispersion_ratio >= 0.28
    assert rec.short_sale_friction_index >= 0.85
    assert rec.miller_overpricing_wedge >= 1.35
    assert rec.regime == "MILLER_OVERPRICING_TRAP"
    assert rec.is_miller_overpricing_trap is True
    assert rec.is_high_consensus_compounder is False
    assert rec.expected_dispersion_alpha_pct < -5.0
    assert rec.verify_integrity() is True


def test_contested_opinion_watch_regime() -> None:
    """Verify elevated dispersion (0.18 <= D_t < 0.28) with moderate short friction classifies as CONTESTED_OPINION_WATCH."""
    rec = calculate_belief_dispersion(
        ticker="TSLA",
        latest_price=240.0,
        consensus_mean_eps=3.00,
        forecast_std_eps=0.60,
        analyst_count=25,
        short_borrow_fee_pct=0.40,
        institutional_ownership_pct=85.0,
    )
    assert 0.18 <= rec.dispersion_ratio < 0.28
    assert rec.miller_overpricing_wedge < 1.35
    assert rec.regime == "CONTESTED_OPINION_WATCH"
    assert rec.is_miller_overpricing_trap is False
    assert rec.is_high_consensus_compounder is False


def test_moderate_belief_divergence_regime() -> None:
    """Verify normal intermediate dispersion (0.05 < D_t < 0.18) classifies as MODERATE_BELIEF_DIVERGENCE."""
    rec = calculate_belief_dispersion(
        ticker="MSFT",
        latest_price=420.0,
        consensus_mean_eps=10.00,
        forecast_std_eps=0.90,
        analyst_count=20,
        short_borrow_fee_pct=0.35,
        institutional_ownership_pct=78.0,
    )
    assert 0.05 < rec.dispersion_ratio < 0.18
    assert rec.regime == "MODERATE_BELIEF_DIVERGENCE"
    assert rec.is_high_consensus_compounder is False
    assert rec.is_miller_overpricing_trap is False


def test_zero_denominator_guard_diether_malloy() -> None:
    """Verify near-zero consensus mean EPS uses the Diether-Malloy $0.10 denominator floor without ZeroDivisionError."""
    rec = calculate_belief_dispersion(
        ticker="AMD",
        latest_price=150.0,
        consensus_mean_eps=0.001,
        forecast_std_eps=0.02,
        analyst_count=12,
        short_borrow_fee_pct=1.0,
        institutional_ownership_pct=65.0,
    )
    assert abs(rec.dispersion_ratio - (0.02 / 0.10)) < 1e-6
    assert np.isfinite(rec.dispersion_ratio)
    assert rec.verify_integrity() is True


def test_classify_dispersion_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_dispersion_regime."""
    assert classify_dispersion_regime(0.04, 0.20) == "HIGH_CONSENSUS_COMPOUNDER"
    assert classify_dispersion_regime(0.10, 0.50) == "MODERATE_BELIEF_DIVERGENCE"
    assert classify_dispersion_regime(0.20, 0.85) == "CONTESTED_OPINION_WATCH"
    assert classify_dispersion_regime(0.30, 1.40) == "MILLER_OVERPRICING_TRAP"


def test_evaluate_belief_dispersion_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_belief_dispersion_profile reads closing prices from SQLite and computes dispersion."""
    db_file = tmp_path / "dispersion_test.db"
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
        p = 180.0 + i * 0.5
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

    rec = evaluate_belief_dispersion_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert abs(rec.latest_price - 192.0) < 1e-2
    assert rec.analyst_count == 15
    assert rec.verify_integrity() is True


def test_short_sale_friction_monotonicity() -> None:
    """Verify higher borrow fee and lower institutional ownership monotonically increase SSFI and Miller Wedge."""
    rec_low_friction = calculate_belief_dispersion(
        ticker="AAPL",
        latest_price=180.0,
        consensus_mean_eps=5.0,
        forecast_std_eps=0.75,
        analyst_count=15,
        short_borrow_fee_pct=0.25,
        institutional_ownership_pct=90.0,
    )
    rec_high_friction = calculate_belief_dispersion(
        ticker="AAPL",
        latest_price=180.0,
        consensus_mean_eps=5.0,
        forecast_std_eps=0.75,
        analyst_count=15,
        short_borrow_fee_pct=7.50,
        institutional_ownership_pct=15.0,
    )
    assert rec_high_friction.short_sale_friction_index > rec_low_friction.short_sale_friction_index
    assert rec_high_friction.miller_overpricing_wedge > rec_low_friction.miller_overpricing_wedge
    assert rec_high_friction.expected_dispersion_alpha_pct < rec_low_friction.expected_dispersion_alpha_pct


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, std_eps, analyst_count, borrow_fee, and institutional_ownership raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_belief_dispersion("AAPL", 0.0, 5.0, 0.2)
    with pytest.raises(ValueError, match="forecast_std_eps must be non-negative"):
        calculate_belief_dispersion("AAPL", 150.0, 5.0, -0.2)
    with pytest.raises(ValueError, match="analyst_count must be at least 2"):
        calculate_belief_dispersion("AAPL", 150.0, 5.0, 0.2, analyst_count=1)
    with pytest.raises(ValueError, match="short_borrow_fee_pct must be non-negative"):
        calculate_belief_dispersion("AAPL", 150.0, 5.0, 0.2, short_borrow_fee_pct=-1.0)
    with pytest.raises(ValueError, match="institutional_ownership_pct must be between 0 and 100"):
        calculate_belief_dispersion("AAPL", 150.0, 5.0, 0.2, institutional_ownership_pct=120.0)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_belief_dispersion(malicious, 150.0, 5.0, 0.2)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_belief_dispersion(
        ticker="GOOG",
        latest_price=175.0,
        consensus_mean_eps=7.20,
        forecast_std_eps=0.40,
        analyst_count=22,
    )
    assert rec.verify_integrity() is True
    tampered = BeliefDispersionReceipt(
        **{**rec.to_dict(), "expected_dispersion_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_dispersion_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI dispersion_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        mean_eps=6.80,
        std_eps=0.20,
        analysts=28,
        borrow_fee=0.30,
        inst_ownership=82.0,
        json=False,
    )
    dispersion_command(args_text)
    captured = capsys.readouterr()
    assert "Diether et al. (2002) & Miller (1977) Belief Dispersion Engine" in captured.out
    assert "AAPL" in captured.out
    assert "HIGH_CONSENSUS_COMPOUNDER" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        mean_eps=6.80,
        std_eps=0.20,
        analysts=28,
        borrow_fee=0.30,
        inst_ownership=82.0,
        json=True,
    )
    dispersion_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "HIGH_CONSENSUS_COMPOUNDER"
    assert payload["is_high_consensus_compounder"] is True
    assert len(payload["signature"]) == 64
