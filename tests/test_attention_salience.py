"""Unit and integration tests for Brad M. Barber & Terrance Odean (2008) /
Zhi Da, Joseph Engelberg & Pengjie Gao (2011) Retail Attention Spike (ASVI) & Reversal Engine (Iteration 93).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.attention_salience import (
    AttentionSalienceReceipt,
    calculate_attention_salience,
    classify_attention_regime,
    evaluate_attention_salience_profile,
)
from src.cli import attention_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_under_the_radar_steady_compounder_regime() -> None:
    """Verify low ASVI, normal volume, and mild daily return classify as UNDER_THE_RADAR_STEADY_COMPOUNDER."""
    rec = calculate_attention_salience(
        ticker="AAPL",
        latest_price=190.0,
        abnormal_svi_log_change=0.03,
        abnormal_volume_ratio=1.02,
        extreme_day_return=0.01,
        retail_buy_imbalance=0.04,
    )
    assert rec.barber_odean_salience_score <= 0.35
    assert rec.regime == "UNDER_THE_RADAR_STEADY_COMPOUNDER"
    assert rec.is_under_the_radar_compounder is True
    assert rec.is_glitter_buying_climax_trap is False
    assert rec.expected_reversal_alpha_pct > 2.0
    assert rec.verify_integrity() is True


def test_glitter_attention_buying_climax_trap_regime() -> None:
    """Verify extreme ASVI spike, abnormal volume surge, and retail buying frenzy trigger buying climax trap."""
    rec = calculate_attention_salience(
        ticker="GME",
        latest_price=35.0,
        abnormal_svi_log_change=1.80,
        abnormal_volume_ratio=4.50,
        extreme_day_return=0.22,
        retail_buy_imbalance=0.75,
    )
    assert rec.barber_odean_salience_score >= 1.55
    assert rec.retail_reversal_overhang_index >= 1.85
    assert rec.regime == "GLITTER_ATTENTION_BUYING_CLIMAX_TRAP"
    assert rec.is_glitter_buying_climax_trap is True
    assert rec.is_under_the_radar_compounder is False
    assert rec.expected_reversal_alpha_pct < -5.0
    assert rec.verify_integrity() is True


def test_transient_svi_price_pressure_watch_regime() -> None:
    """Verify elevated search volume (ASVI >= 0.45) with moderate salience classifies as watch."""
    rec = calculate_attention_salience(
        ticker="TSLA",
        latest_price=240.0,
        abnormal_svi_log_change=0.60,
        abnormal_volume_ratio=1.60,
        extreme_day_return=0.035,
        retail_buy_imbalance=0.20,
    )
    assert rec.barber_odean_salience_score < 1.55
    assert rec.retail_reversal_overhang_index < 1.85
    assert rec.regime == "TRANSIENT_SVI_PRICE_PRESSURE_WATCH"
    assert rec.short_term_pressure_bps > 15.0
    assert rec.is_glitter_buying_climax_trap is False
    assert rec.is_under_the_radar_compounder is False


def test_normal_attention_equilibrium_regime() -> None:
    """Verify intermediate salience (0.35 < BASS < 0.95) and low ASVI classify as NORMAL_ATTENTION_EQUILIBRIUM."""
    rec = calculate_attention_salience(
        ticker="MSFT",
        latest_price=420.0,
        abnormal_svi_log_change=0.18,
        abnormal_volume_ratio=1.35,
        extreme_day_return=0.025,
        retail_buy_imbalance=0.10,
    )
    assert 0.35 < rec.barber_odean_salience_score < 0.95
    assert rec.regime == "NORMAL_ATTENTION_EQUILIBRIUM"
    assert rec.is_under_the_radar_compounder is False
    assert rec.is_glitter_buying_climax_trap is False


def test_retail_buy_imbalance_asymmetry_monotonicity() -> None:
    """Verify Barber-Odean asymmetric retail buying amplifies reversal overhang and reduces long-horizon alpha."""
    rec_neutral = calculate_attention_salience(
        ticker="NVDA",
        latest_price=140.0,
        abnormal_svi_log_change=0.50,
        abnormal_volume_ratio=2.20,
        extreme_day_return=0.05,
        retail_buy_imbalance=0.0,
    )
    rec_fomo = calculate_attention_salience(
        ticker="NVDA",
        latest_price=140.0,
        abnormal_svi_log_change=0.50,
        abnormal_volume_ratio=2.20,
        extreme_day_return=0.05,
        retail_buy_imbalance=0.80,
    )
    assert rec_fomo.retail_reversal_overhang_index > rec_neutral.retail_reversal_overhang_index
    assert rec_fomo.short_term_pressure_bps > rec_neutral.short_term_pressure_bps
    assert rec_fomo.expected_reversal_alpha_pct < rec_neutral.expected_reversal_alpha_pct


def test_classify_attention_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_attention_regime."""
    assert classify_attention_regime(0.20, 0.05, 0.25) == "UNDER_THE_RADAR_STEADY_COMPOUNDER"
    assert classify_attention_regime(0.55, 0.20, 0.65) == "NORMAL_ATTENTION_EQUILIBRIUM"
    assert classify_attention_regime(1.10, 0.55, 1.35) == "TRANSIENT_SVI_PRICE_PRESSURE_WATCH"
    assert classify_attention_regime(1.75, 1.20, 2.10) == "GLITTER_ATTENTION_BUYING_CLIMAX_TRAP"


def test_negative_extreme_return_salience_symmetry() -> None:
    """Verify Barber & Odean (2008) finding: both extreme positive and extreme negative returns grab attention."""
    rec_pos = calculate_attention_salience(
        ticker="AMD",
        latest_price=155.0,
        extreme_day_return=0.07,
    )
    rec_neg = calculate_attention_salience(
        ticker="AMD",
        latest_price=155.0,
        extreme_day_return=-0.07,
    )
    assert rec_pos.barber_odean_salience_score == rec_neg.barber_odean_salience_score


def test_evaluate_attention_salience_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_attention_salience_profile reads OHLCV history from SQLite and computes salience."""
    db_file = tmp_path / "attention_test.db"
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
        p = 180.0 + i * 0.4
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

    rec = evaluate_attention_salience_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert abs(rec.abnormal_volume_ratio - 1.0) < 1e-3
    assert rec.regime == "UNDER_THE_RADAR_STEADY_COMPOUNDER"
    assert rec.verify_integrity() is True


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, ASVI, volume ratio, extreme return, and retail buy imbalance raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_attention_salience("AAPL", 0.0)
    with pytest.raises(ValueError, match="abnormal_svi_log_change must be in"):
        calculate_attention_salience("AAPL", 150.0, abnormal_svi_log_change=6.0)
    with pytest.raises(ValueError, match="abnormal_volume_ratio must be strictly positive"):
        calculate_attention_salience("AAPL", 150.0, abnormal_volume_ratio=0.0)
    with pytest.raises(ValueError, match="extreme_day_return must be in"):
        calculate_attention_salience("AAPL", 150.0, extreme_day_return=-1.1)
    with pytest.raises(ValueError, match="retail_buy_imbalance must be in"):
        calculate_attention_salience("AAPL", 150.0, retail_buy_imbalance=1.3)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_attention_salience(malicious, 150.0)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_attention_salience(
        ticker="GOOG",
        latest_price=178.0,
        abnormal_svi_log_change=0.05,
    )
    assert rec.verify_integrity() is True
    tampered = AttentionSalienceReceipt(
        **{**rec.to_dict(), "expected_reversal_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_attention_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI attention_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        asvi=0.03,
        abnormal_volume=1.02,
        extreme_return=0.01,
        retail_buy_imbalance=0.04,
        json=False,
    )
    attention_command(args_text)
    captured = capsys.readouterr()
    assert "Barber & Odean (2008) & Da et al. (2011) Attention Engine" in captured.out
    assert "AAPL" in captured.out
    assert "UNDER_THE_RADAR_STEADY_COMPOUNDER" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        asvi=0.03,
        abnormal_volume=1.02,
        extreme_return=0.01,
        retail_buy_imbalance=0.04,
        json=True,
    )
    attention_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "UNDER_THE_RADAR_STEADY_COMPOUNDER"
    assert payload["is_under_the_radar_compounder"] is True
    assert len(payload["signature"]) == 64

