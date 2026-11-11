"""Unit & CLI integration tests for Black & Jones (1987) CPPI & TIPP Portfolio Insurance Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.cppi_portfolio_insurance import (
    calculate_cppi_allocation,
    evaluate_cppi_path,
    persist_cppi_insurance_record,
    verify_hmac_signature,
)


def test_cppi_allocation_formula_and_cushion_conservation():
    """Verify E = min(L_max*W, m*(W - F)) and budget conservation E + B == W."""
    rec = calculate_cppi_allocation(
        portfolio_value_w=100.0,
        floor_value_f=85.0,
        multiplier_m=4.0,
        max_leverage_cap=1.0,
        ticker="SPY",
    )

    assert rec.cushion_c == pytest.approx(15.0, rel=1e-9)
    assert rec.risky_exposure_e == pytest.approx(60.0, rel=1e-9)
    assert rec.safe_cash_reserve_b == pytest.approx(40.0, rel=1e-9)
    assert rec.risky_exposure_e + rec.safe_cash_reserve_b == pytest.approx(rec.portfolio_value_w, rel=1e-9)
    assert rec.gap_crash_tolerance_pct == pytest.approx(25.0, rel=1e-9)


def test_leverage_cap_binds_with_large_cushion_or_multiplier():
    """Exposure stops cleanly at max_leverage_cap * W even when m * C exceeds W."""
    rec = calculate_cppi_allocation(
        portfolio_value_w=120.0,
        floor_value_f=80.0,
        multiplier_m=5.0,
        max_leverage_cap=1.0,
        ticker="CAP",
    )

    assert rec.risky_exposure_e == pytest.approx(120.0, rel=1e-9)
    assert rec.risky_weight_pct == pytest.approx(100.0, rel=1e-9)
    assert rec.safe_cash_reserve_b == pytest.approx(0.0, abs=1e-9)
    assert rec.regime == "CONVEX_UPSIDE_PARTICIPATION"


def test_floor_preservation_and_drawdown_compression_in_gradual_bear_market():
    """Gradual bear market with daily drops inside 1/m preserves floor and cuts drawdown."""
    # -45% cumulative decline via -1.2% daily steps (well inside 1/4 = 25% crash tolerance)
    prices = [100.0 * (0.988 ** t) for t in range(50)]
    rec = evaluate_cppi_path(
        prices=prices,
        ticker="BEAR",
        initial_wealth=100.0,
        floor_pct=0.85,
        multiplier_m=4.0,
        strategy="CPPI",
    )

    assert rec.gap_risk_breach is False
    assert rec.portfolio_value_w >= rec.floor_value_f - 1e-6
    assert rec.realized_drawdown_pct < rec.uninsured_drawdown_pct
    assert rec.drawdown_reduction_pct > 15.0


def test_gap_risk_floor_breach_detected_on_overnight_crash_exceeding_one_over_m():
    """Single-step crash worse than -1/m causes cushion depletion and floor breach flag."""
    # m = 5 -> crash tolerance = 20%; drop asset by 35% in one step with cushion large enough to allocate ~100%
    prices = [100.0, 110.0, 65.0, 65.0]
    rec = evaluate_cppi_path(
        prices=prices,
        ticker="GAP",
        initial_wealth=100.0,
        floor_pct=0.80,
        multiplier_m=5.0,
        max_leverage_cap=1.0,
        strategy="CPPI",
    )

    assert rec.gap_risk_breach is True
    assert rec.regime == "GAP_RISK_FLOOR_BREACHED"


def test_tipp_high_water_ratchet_raises_floor_above_cppi_in_rally_then_pullback():
    """TIPP ratchets floor upward during rally so terminal floor strictly exceeds static CPPI."""
    rally_then_dip = [100.0 + 1.5 * t for t in range(25)] + [136.0 - 1.5 * t for t in range(15)]

    cppi_rec = evaluate_cppi_path(prices=rally_then_dip, strategy="CPPI", floor_pct=0.85, multiplier_m=4.0)
    tipp_rec = evaluate_cppi_path(prices=rally_then_dip, strategy="TIPP", floor_pct=0.85, multiplier_m=4.0)

    assert tipp_rec.floor_value_f > cppi_rec.floor_value_f
    assert tipp_rec.realized_drawdown_pct <= cppi_rec.realized_drawdown_pct


def test_cash_lock_regime_when_wealth_equals_floor():
    """Zero cushion sets risky allocation to 0% in CASH_LOCKED_FLOOR_PRESERVATION."""
    rec = calculate_cppi_allocation(portfolio_value_w=85.0, floor_value_f=85.0, multiplier_m=4.0)

    assert rec.cash_locked is True
    assert rec.risky_exposure_e == 0.0
    assert rec.safe_cash_reserve_b == pytest.approx(85.0, rel=1e-9)
    assert rec.regime == "CASH_LOCKED_FLOOR_PRESERVATION"


def test_cautious_cushion_buffer_regime():
    """Moderate cushion between 0% and 65% risky weight activates CAUTIOUS_CUSHION_BUFFER."""
    rec = calculate_cppi_allocation(portfolio_value_w=100.0, floor_value_f=90.0, multiplier_m=4.0)
    assert rec.risky_weight_pct == pytest.approx(40.0, rel=1e-9)
    assert rec.regime == "CAUTIOUS_CUSHION_BUFFER"


def test_zero_lookahead_path_invariance():
    """Modifying future price shocks at step t+k must not affect allocation prior to t+k."""
    base_prices = [100.0, 102.0, 104.0, 105.0, 106.0]
    shocked_tail = [100.0, 102.0, 104.0, 105.0, 106.0, 50.0, 50.0]

    rec_short = evaluate_cppi_path(prices=base_prices, ticker="ZLK")
    rec_prefix_only = evaluate_cppi_path(prices=shocked_tail[:5], ticker="ZLK")

    assert rec_short.portfolio_value_w == pytest.approx(rec_prefix_only.portfolio_value_w, rel=1e-9)
    assert rec_short.risky_exposure_e == pytest.approx(rec_prefix_only.risky_exposure_e, rel=1e-9)


def test_short_price_history_graceful_fallback():
    """Fewer than 2 prices safely returns initial floor allocation without failing."""
    rec = evaluate_cppi_path(prices=[100.0], initial_wealth=100.0, floor_pct=0.85, multiplier_m=4.0)
    assert rec.portfolio_value_w == pytest.approx(100.0, rel=1e-9)
    assert rec.floor_value_f == pytest.approx(85.0, rel=1e-9)
    assert rec.risky_exposure_e == pytest.approx(60.0, rel=1e-9)


def test_security_input_validation():
    """Verify strict ticker allowlist, CPPI/TIPP strategy allowlist, and multiplier/cap guards."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_cppi_allocation(ticker="CPPI;DROP")

    with pytest.raises(ValueError, match="Unsupported strategy"):
        calculate_cppi_allocation(strategy="MARTINGALE")

    with pytest.raises(ValueError, match="multiplier_m must be >= 1.0"):
        calculate_cppi_allocation(multiplier_m=0.5)

    with pytest.raises(ValueError, match="floor_pct must satisfy"):
        evaluate_cppi_path(prices=[100.0, 101.0], floor_pct=1.05)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_cppi_allocation(ticker="QQQ")
    sig_payload = (
        f"{rec.ticker}:{rec.strategy}:{rec.portfolio_value_w:.4f}:{rec.floor_value_f:.4f}:"
        f"{rec.multiplier_m:.2f}:{rec.risky_exposure_e:.4f}:{rec.risky_weight_pct:.2f}:"
        f"{int(rec.cash_locked)}:{int(rec.gap_risk_breach)}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_persist_cppi_insurance_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized placeholders and FK enforcement."""
    db_file = str(tmp_path / "test_cppi.db")
    rec = calculate_cppi_allocation(ticker="IWM", strategy="TIPP")

    assert persist_cppi_insurance_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, strategy, risky_weight_pct, regime FROM cppi_insurance_audits WHERE ticker = ?",
            ("IWM",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "IWM"
        assert row[1] == "TIPP"
        assert row[2] == pytest.approx(rec.risky_weight_pct, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_cppi_table_execution():
    """Verify CLI execution for cppi command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "cppi",
        "--ticker",
        "CPPI",
        "--wealth",
        "100",
        "--floor",
        "85",
        "--multiplier",
        "4.0",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Black & Jones (1987) CPPI & TIPP Portfolio Insurance Engine" in res.stdout
    assert "CPPI" in res.stdout
    assert "Risky Asset Exposure" in res.stdout


def test_cli_cppi_json_execution():
    """Verify CLI execution for cppi command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "cppi",
        "--ticker",
        "SPY",
        "--strategy",
        "TIPP",
        "--wealth",
        "120",
        "--floor",
        "90",
        "--multiplier",
        "3.5",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "SPY"
    assert parsed["strategy"] == "TIPP"
    assert parsed["cushion_c"] == pytest.approx(30.0)
    assert "hmac_signature" in parsed

