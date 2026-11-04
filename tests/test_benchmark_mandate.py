"""Unit and integration tests for Malcolm Baker, Brendan Bradley & Jeffrey Wurgler (2011)
Institutional Benchmark Mandate, Tracking-Error Leverage Aversion & Low-Risk Anomaly Engine (Iteration 94).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.benchmark_mandate import (
    BenchmarkMandateReceipt,
    calculate_benchmark_mandate,
    classify_benchmark_mandate_regime,
    evaluate_benchmark_mandate_profile,
)
from src.cli import benchmark_mandate_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_neglected_low_beta_alpha_haven_regime() -> None:
    """Verify low beta (<= 0.75) and low IVOL (<= 0.22) classify as NEGLECTED_LOW_BETA_ALPHA_HAVEN."""
    rec = calculate_benchmark_mandate(
        ticker="PG",
        latest_price=168.0,
        capm_beta=0.55,
        market_volatility_ann=0.16,
        idiosyncratic_volatility_ann=0.14,
        expected_market_risk_premium_pct=6.0,
        benchmarked_institutional_share=0.78,
    )
    assert abs(rec.active_return_alpha_hurdle_pct - 2.70) < 1e-2
    assert rec.institutional_crowding_wedge <= -0.35
    assert rec.regime == "NEGLECTED_LOW_BETA_ALPHA_HAVEN"
    assert rec.is_neglected_low_beta_haven is True
    assert rec.is_crowded_high_beta_trap is False
    assert rec.expected_unconstrained_alpha_pct > 2.5
    assert rec.verify_integrity() is True


def test_crowded_high_beta_leverage_trap_regime() -> None:
    """Verify high beta (>= 1.35) and high benchmarked institutional share trigger CROWDED_HIGH_BETA_LEVERAGE_TRAP."""
    rec = calculate_benchmark_mandate(
        ticker="NVDA",
        latest_price=140.0,
        capm_beta=1.65,
        market_volatility_ann=0.18,
        idiosyncratic_volatility_ann=0.38,
        expected_market_risk_premium_pct=6.0,
        benchmarked_institutional_share=0.82,
    )
    assert rec.active_return_alpha_hurdle_pct < 0.0
    assert rec.institutional_crowding_wedge >= 0.85
    assert rec.regime == "CROWDED_HIGH_BETA_LEVERAGE_TRAP"
    assert rec.is_crowded_high_beta_trap is True
    assert rec.is_neglected_low_beta_haven is False
    assert rec.expected_unconstrained_alpha_pct < -5.0
    assert rec.verify_integrity() is True


def test_tracking_error_orphan_watch_regime() -> None:
    """Verify low beta (<= 0.75) paired with high IVOL (> 0.22) classifies as TRACKING_ERROR_ORPHAN_WATCH."""
    rec = calculate_benchmark_mandate(
        ticker="NEM",
        latest_price=48.0,
        capm_beta=0.60,
        market_volatility_ann=0.16,
        idiosyncratic_volatility_ann=0.34,
        expected_market_risk_premium_pct=6.0,
        benchmarked_institutional_share=0.65,
    )
    assert rec.regime == "TRACKING_ERROR_ORPHAN_WATCH"
    assert rec.is_neglected_low_beta_haven is False
    assert rec.is_crowded_high_beta_trap is False


def test_benchmark_neutral_core_regime() -> None:
    """Verify near-unit beta (0.75 < beta < 1.25) classifies as BENCHMARK_NEUTRAL_CORE."""
    rec = calculate_benchmark_mandate(
        ticker="MSFT",
        latest_price=420.0,
        capm_beta=0.98,
        market_volatility_ann=0.16,
        idiosyncratic_volatility_ann=0.18,
        expected_market_risk_premium_pct=6.0,
        benchmarked_institutional_share=0.75,
    )
    assert rec.regime == "BENCHMARK_NEUTRAL_CORE"
    assert rec.is_neglected_low_beta_haven is False
    assert rec.is_crowded_high_beta_trap is False


def test_baker_bradley_wurgler_tracking_error_formula() -> None:
    """Verify exact mathematical calculation of TE_i = sqrt((beta-1)^2 * sigma_M^2 + sigma_e^2)."""
    # beta = 0.50, sigma_M = 0.20 -> (beta-1)*sigma_M = -0.10; sigma_e = 0.24 -> sqrt(0.10^2 + 0.24^2) = 0.26
    rec = calculate_benchmark_mandate(
        ticker="JNJ",
        latest_price=155.0,
        capm_beta=0.50,
        market_volatility_ann=0.20,
        idiosyncratic_volatility_ann=0.24,
        expected_market_risk_premium_pct=6.0,
    )
    assert abs(rec.single_stock_tracking_error - 0.26) < 1e-6
    assert abs(rec.active_return_alpha_hurdle_pct - 3.00) < 1e-6


def test_benchmarked_institutional_share_amplification() -> None:
    """Verify higher benchmarked institutional share amplifies crowding/neglect wedge magnitude."""
    rec_low_inst = calculate_benchmark_mandate(
        ticker="KO",
        latest_price=62.0,
        capm_beta=0.55,
        benchmarked_institutional_share=0.20,
    )
    rec_high_inst = calculate_benchmark_mandate(
        ticker="KO",
        latest_price=62.0,
        capm_beta=0.55,
        benchmarked_institutional_share=0.90,
    )
    assert abs(rec_high_inst.institutional_crowding_wedge) > abs(rec_low_inst.institutional_crowding_wedge)
    assert rec_high_inst.expected_unconstrained_alpha_pct > rec_low_inst.expected_unconstrained_alpha_pct


def test_classify_benchmark_mandate_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_benchmark_mandate_regime."""
    assert classify_benchmark_mandate_regime(0.60, 0.16, -0.80) == "NEGLECTED_LOW_BETA_ALPHA_HAVEN"
    assert classify_benchmark_mandate_regime(0.60, 0.30, -0.80) == "TRACKING_ERROR_ORPHAN_WATCH"
    assert classify_benchmark_mandate_regime(1.02, 0.18, 0.05) == "BENCHMARK_NEUTRAL_CORE"
    assert classify_benchmark_mandate_regime(1.45, 0.28, 1.20) == "CROWDED_HIGH_BETA_LEVERAGE_TRAP"


def test_evaluate_benchmark_mandate_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_benchmark_mandate_profile reads closing prices from SQLite and infers low-vol profile."""
    db_file = tmp_path / "mandate_test.db"
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
        p = 180.0 + i * 0.25
        records.append(
            {
                "ticker": "AAPL",
                "exchange": "XNAS",
                "session_date": f"2026-01-{i+1:02d}",
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "adj_close": p,
                "volume": 50_000_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=db_file)

    rec = evaluate_benchmark_mandate_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert rec.capm_beta <= 0.75
    assert rec.regime == "NEGLECTED_LOW_BETA_ALPHA_HAVEN"
    assert rec.verify_integrity() is True


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, beta, market_vol, idio_vol, ERP, and institutional share raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_benchmark_mandate("AAPL", 0.0)
    with pytest.raises(ValueError, match="capm_beta must be in"):
        calculate_benchmark_mandate("AAPL", 150.0, capm_beta=5.0)
    with pytest.raises(ValueError, match="market_volatility_ann must be in"):
        calculate_benchmark_mandate("AAPL", 150.0, market_volatility_ann=0.0)
    with pytest.raises(ValueError, match="idiosyncratic_volatility_ann must be in"):
        calculate_benchmark_mandate("AAPL", 150.0, idiosyncratic_volatility_ann=-0.1)
    with pytest.raises(ValueError, match="expected_market_risk_premium_pct must be in"):
        calculate_benchmark_mandate("AAPL", 150.0, expected_market_risk_premium_pct=0.0)
    with pytest.raises(ValueError, match="benchmarked_institutional_share must be in"):
        calculate_benchmark_mandate("AAPL", 150.0, benchmarked_institutional_share=1.2)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_benchmark_mandate(malicious, 150.0)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_benchmark_mandate(
        ticker="PG",
        latest_price=168.0,
        capm_beta=0.55,
    )
    assert rec.verify_integrity() is True
    tampered = BenchmarkMandateReceipt(
        **{**rec.to_dict(), "expected_unconstrained_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_benchmark_mandate_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI benchmark_mandate_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="PG",
        beta=0.55,
        market_vol=0.16,
        idio_vol=0.14,
        market_premium=6.0,
        inst_benchmarked_share=0.78,
        json=False,
    )
    benchmark_mandate_command(args_text)
    captured = capsys.readouterr()
    assert "Baker, Bradley & Wurgler (2011) Benchmark Mandate Engine" in captured.out
    assert "PG" in captured.out
    assert "NEGLECTED_LOW_BETA_ALPHA_HAVEN" in captured.out

    args_json = argparse.Namespace(
        ticker="PG",
        beta=0.55,
        market_vol=0.16,
        idio_vol=0.14,
        market_premium=6.0,
        inst_benchmarked_share=0.78,
        json=True,
    )
    benchmark_mandate_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "PG"
    assert payload["regime"] == "NEGLECTED_LOW_BETA_ALPHA_HAVEN"
    assert payload["is_neglected_low_beta_haven"] is True
    assert len(payload["signature"]) == 64

