"""Tests for Kelly (1956) & Thorp (1997) Fractional Kelly Log-Wealth Growth Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.kelly_growth_criterion import (
    _build_hmac_payload,
    calculate_binomial_kelly,
    calculate_kelly_growth,
    evaluate_kelly_growth_profile,
    persist_kelly_growth_record,
    verify_hmac_signature,
)


def test_full_kelly_closed_form_and_half_kelly_tradeoff() -> None:
    """Verify f* = (mu - rf)/sigma^2 and 75% growth / 75% variance reduction at Half-Kelly."""
    rec_full = calculate_kelly_growth(
        expected_return_annual=0.12,
        volatility_annual=0.20,
        risk_free_annual=0.04,
        kelly_fraction=1.0,
        drawdown_threshold=0.50,
        ticker="SPY",
    )
    assert rec_full.full_kelly_fraction == pytest.approx(2.0, abs=1e-6)
    assert rec_full.applied_fraction == pytest.approx(2.0, abs=1e-6)
    assert rec_full.double_kelly_ruin_fraction == pytest.approx(4.0, abs=1e-6)
    assert rec_full.full_kelly_growth_ann == pytest.approx(0.04 + 0.08, abs=1e-6)
    assert rec_full.drawdown_exceedance_prob == pytest.approx(0.50, abs=1e-6)
    assert rec_full.regime == "FULL_KELLY_MAX_GROWTH"

    rec_half = calculate_kelly_growth(
        expected_return_annual=0.12,
        volatility_annual=0.20,
        risk_free_annual=0.04,
        kelly_fraction=0.50,
        drawdown_threshold=0.50,
        ticker="SPY",
    )
    assert rec_half.applied_fraction == pytest.approx(1.0, abs=1e-6)
    assert rec_half.growth_efficiency_ratio == pytest.approx(0.75, abs=1e-6)
    assert rec_half.variance_reduction_ratio == pytest.approx(0.75, abs=1e-6)
    assert rec_half.drawdown_exceedance_prob == pytest.approx(0.125, abs=1e-6)
    assert rec_half.regime == "HALF_KELLY_COMPOUNDING_SWEET_SPOT"


def test_double_kelly_ruin_boundary_and_overbetting_decay() -> None:
    """At k = 2.0 excess growth is 0 and P(DD)=1.0; at k > 2.0 excess growth turns negative."""
    rec_double = calculate_kelly_growth(
        expected_return_annual=0.12,
        volatility_annual=0.20,
        risk_free_annual=0.04,
        kelly_fraction=2.0,
        ticker="QQQ",
    )
    assert rec_double.applied_excess_growth_ann == pytest.approx(0.0, abs=1e-9)
    assert rec_double.growth_efficiency_ratio == pytest.approx(0.0, abs=1e-9)
    assert rec_double.drawdown_exceedance_prob == pytest.approx(1.0, abs=1e-9)
    assert rec_double.regime == "OVERBETTING_VOLATILITY_RUIN"

    rec_over = calculate_kelly_growth(
        expected_return_annual=0.12,
        volatility_annual=0.20,
        risk_free_annual=0.04,
        kelly_fraction=2.5,
        ticker="QQQ",
    )
    assert rec_over.applied_excess_growth_ann < 0.0
    assert rec_over.applied_growth_ann < rec_over.risk_free_annual
    assert rec_over.regime == "OVERBETTING_VOLATILITY_RUIN"


def test_negative_edge_zero_allocation_and_discrete_binomial() -> None:
    """Negative excess return clamps long-only allocation to 0; discrete binomial matches textbook."""
    rec_neg = calculate_kelly_growth(
        expected_return_annual=0.02,
        volatility_annual=0.25,
        risk_free_annual=0.04,
        kelly_fraction=0.50,
        ticker="TLT",
    )
    assert rec_neg.full_kelly_fraction == pytest.approx(0.0, abs=1e-9)
    assert rec_neg.applied_fraction == pytest.approx(0.0, abs=1e-9)
    assert rec_neg.drawdown_exceedance_prob == pytest.approx(0.0, abs=1e-9)
    assert rec_neg.regime == "NEGATIVE_EDGE_ZERO_ALLOCATION"

    # 60% win prob at 1:1 even money odds -> 0.60 - 0.40 = 0.20 (20% bankroll)
    assert calculate_binomial_kelly(0.60, 1.0) == pytest.approx(0.20, abs=1e-9)


def test_suboptimal_over_kelly_and_conservative_regimes() -> None:
    """Verify k in (1.05, 2.0) and k < 0.25 map to expected regimes."""
    rec_over = calculate_kelly_growth(0.14, 0.22, risk_free_annual=0.04, kelly_fraction=1.25)
    assert rec_over.regime == "SUBOPTIMAL_OVER_KELLY_DRAG"
    assert rec_over.applied_excess_growth_ann < rec_over.full_kelly_growth_ann - rec_over.risk_free_annual

    rec_low = calculate_kelly_growth(0.14, 0.22, risk_free_annual=0.04, kelly_fraction=0.15)
    assert rec_low.regime == "CONSERVATIVE_LOW_FRACTION"
    assert rec_low.drawdown_exceedance_prob < 0.001


def test_empirical_price_profile_calibration_and_short_fallback() -> None:
    """Empirical log-return calibration recovers positive drift and fallback works smoothly."""
    t = np.arange(100, dtype=float)
    prices = 100.0 * np.exp(0.0008 * t + 0.005 * np.sin(t * 0.4))
    rec = evaluate_kelly_growth_profile(prices, ticker="AAPL", kelly_fraction=0.50)
    assert rec.ticker == "AAPL"
    assert rec.full_kelly_fraction > 0.0

    rec_short = evaluate_kelly_growth_profile([100.0, 101.0], ticker="MSFT")
    assert rec_short.expected_return_annual == pytest.approx(0.12, abs=1e-6)


def test_input_validation_and_hmac_tamper_detection(tmp_path) -> None:
    """Reject bad tickers/volatilities, verify HMAC seal, and test SQLite persistence."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_kelly_growth(0.12, 0.20, ticker="../BAD")
    with pytest.raises(ValueError):
        calculate_kelly_growth(0.12, 0.0, ticker="SPY")

    rec = calculate_kelly_growth(0.12, 0.20, ticker="NVDA")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        expected_return_annual=rec.expected_return_annual,
        volatility_annual=rec.volatility_annual,
        kelly_multiplier=rec.kelly_multiplier,
        full_kelly_fraction=rec.full_kelly_fraction,
        applied_fraction=rec.applied_fraction,
        applied_growth_ann=rec.applied_growth_ann,
        drawdown_exceedance_prob=rec.drawdown_exceedance_prob,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("NVDA", "TSLA"), rec.hmac_signature) is False

    db_file = str(tmp_path / "kelly_test.db")
    assert persist_kelly_growth_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, regime FROM kelly_growth_audits").fetchall()
    assert rows == [("NVDA", "HALF_KELLY_COMPOUNDING_SWEET_SPOT")]


def test_cli_kelly_growth_json_and_aliases() -> None:
    """Verify CLI primary command and aliases emit valid JSON receipt."""
    for alias in ("kelly-growth", "fractional-kelly", "kt-1956"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--mu",
                "0.12",
                "--vol",
                "0.20",
                "--rf",
                "0.04",
                "--kelly-fraction",
                "0.5",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["full_kelly_fraction"] == pytest.approx(2.0, abs=1e-6)
        assert data["growth_efficiency_ratio"] == pytest.approx(0.75, abs=1e-6)
        assert data["regime"] == "HALF_KELLY_COMPOUNDING_SWEET_SPOT"

