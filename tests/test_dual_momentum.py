"""Unit and integration test suite for Jegadeesh & Titman (1993, 2001) / Antonacci Dual Momentum Engine."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from src.analytics.dual_momentum import (
    DualMomentumReceipt,
    calculate_dual_momentum,
    classify_dual_momentum_regime,
    evaluate_dual_momentum_profile,
)
from src.cli import dual_momentum_command


def test_calculate_dual_momentum_strong_leader() -> None:
    """Test standard strong dual momentum leader with positive drift and steady benchmark."""
    # 252 sessions with 30% upward drift
    n = 252
    base = 100.0
    # Prices increasing linearly from 100 to 140
    prices = [base * (1.0 + 0.40 * (i / n)) for i in range(n)]
    bench_prices = [300.0 * (1.0 + 0.10 * (i / n)) for i in range(n)]

    receipt = calculate_dual_momentum(
        prices=prices,
        benchmark_prices=bench_prices,
        ticker="LEADER",
        benchmark_ticker="SPY",
        rf_rate_pct=4.0,
    )

    assert receipt.ticker == "LEADER"
    assert receipt.sessions_evaluated == 252
    assert receipt.latest_price == pytest.approx(prices[-1], rel=1e-4)
    assert receipt.momentum_12_2_pct > 20.0
    assert receipt.momentum_6_1_pct > 10.0
    assert receipt.total_return_12m_pct > receipt.momentum_12_2_pct
    assert receipt.is_absolute_trend_positive is True
    assert receipt.has_momentum_crash_risk is False
    assert receipt.relative_momentum_spread_pct > 10.0
    assert receipt.regime == "STRONG_DUAL_MOMENTUM_LEADER"
    assert receipt.expected_momentum_alpha_pct > 5.0
    assert receipt.verify_integrity() is True


def test_calculate_dual_momentum_reversal_skip_month_isolation() -> None:
    """Prove that 1-month short-term price movements DO NOT contaminate 12-2 and 6-1 intermediate momentum."""
    n = 252
    base_prices = [100.0 * (1.0 + 0.001 * i) for i in range(n)]

    # Scenario A: Normal continuation in the final 21 sessions
    rec_a = calculate_dual_momentum(base_prices, ticker="TEST_A")

    # Scenario B: Violent drop in the final 21 sessions (from t-21 to t)
    shocked_prices = list(base_prices)
    for idx in range(n - 21, n):
        shocked_prices[idx] = shocked_prices[idx] * 0.85  # 15% shock in month 1

    rec_b = calculate_dual_momentum(shocked_prices, ticker="TEST_B")

    # Mathematical Invariance Proof:
    # Intermediate 12-2 momentum is calculated up to t-21, so rec_a and rec_b MUST be identical!
    assert rec_a.momentum_12_2_pct == pytest.approx(rec_b.momentum_12_2_pct, rel=1e-5)
    assert rec_a.momentum_6_1_pct == pytest.approx(rec_b.momentum_6_1_pct, rel=1e-5)

    # However, short-term reversal R_{1,0} and total return R_{12,0} MUST reflect the recent month shock
    assert rec_a.reversal_1_0_pct > 0.0
    assert rec_b.reversal_1_0_pct < -10.0
    assert rec_b.total_return_12m_pct < rec_a.total_return_12m_pct


def test_momentum_crash_risk_detection() -> None:
    """Test Daniel & Moskowitz (2016) momentum crash warning detection."""
    n = 252
    prices = [100.0] * n
    # Strong bull market from session 0 to 230: price rises from 100 to 160 (+60%)
    for i in range(231):
        prices[i] = 100.0 * (1.0 + 0.60 * (i / 230))
    # Severe liquidation crash from session 231 to 251 (final 21 trading days): price drops by 18%
    p_peak = prices[230]
    for k, i in enumerate(range(231, n)):
        prices[i] = p_peak * (1.0 - 0.18 * ((k + 1) / 21))

    receipt = calculate_dual_momentum(prices, ticker="CRASH_ASSET")

    assert receipt.momentum_12_2_pct >= 20.0
    assert receipt.reversal_1_0_pct <= -8.5
    assert receipt.has_momentum_crash_risk is True
    assert receipt.regime == "MOMENTUM_CRASH_WARNING"
    assert receipt.expected_momentum_alpha_pct == -10.5
    assert "MOMENTUM_CRASH_WARNING" in receipt.summary


def test_bearish_avoid_negative_trend() -> None:
    """Test Antonacci absolute momentum defensive hedge when intermediate return is negative."""
    n = 252
    # Secular bear market: prices decreasing from 100 to 60
    prices = [100.0 * (1.0 - 0.40 * (i / n)) for i in range(n)]

    receipt = calculate_dual_momentum(prices, ticker="BEAR_ASSET", rf_rate_pct=4.0)

    assert receipt.momentum_12_2_pct < 0.0
    assert receipt.is_absolute_trend_positive is False
    assert receipt.regime == "BEARISH_AVOID"
    assert receipt.expected_momentum_alpha_pct < 0.0


def test_industry_vs_idiosyncratic_decomposition() -> None:
    """Test Moskowitz & Grinblatt (1999) market beta and idiosyncratic momentum decomposition."""
    n = 252
    # Benchmark drifts up 15%
    bench_prices = [200.0 * (1.0 + 0.15 * (i / n)) for i in range(n)]
    # Stock drifts up 35% with beta ~ 1.0
    stock_prices = [100.0 * (1.0 + 0.35 * (i / n)) for i in range(n)]

    receipt = calculate_dual_momentum(
        prices=stock_prices,
        benchmark_prices=bench_prices,
        ticker="ALPHA_STOCK",
        benchmark_ticker="BENCH",
    )

    assert receipt.relative_momentum_spread_pct > 15.0
    assert receipt.idiosyncratic_momentum_pct > 10.0
    assert 0.5 <= receipt.beta_to_benchmark <= 2.5
    assert receipt.verify_integrity() is True


def test_volatility_adjusted_sharpe_momentum() -> None:
    """Verify that lower volatility yields higher Sharpe momentum for identical returns."""
    n = 252
    # Smooth deterministic path
    smooth_prices = [100.0 * (1.0 + 0.25 * (i / n)) for i in range(n)]
    # Volatile path ending at the same points
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.03, n)
    noisy_prices = [smooth_prices[i] * (1.0 + noise[i]) for i in range(n)]
    noisy_prices[0] = smooth_prices[0]
    noisy_prices[-22] = smooth_prices[-22]

    rec_smooth = calculate_dual_momentum(smooth_prices, ticker="SMOOTH")
    rec_noisy = calculate_dual_momentum(noisy_prices, ticker="NOISY")

    assert rec_smooth.annualized_volatility_pct < rec_noisy.annualized_volatility_pct
    assert rec_smooth.sharpe_momentum > rec_noisy.sharpe_momentum


def test_dual_momentum_receipt_hmac_tamper() -> None:
    """Test cryptographic integrity and tamper resistance of DualMomentumReceipt."""
    prices = [100.0 * (1.0 + 0.001 * i) for i in range(100)]
    receipt = calculate_dual_momentum(prices, ticker="AUDIT_TEST")

    assert receipt.verify_integrity() is True

    # Tamper with 12-2 momentum
    tampered = DualMomentumReceipt(
        ticker=receipt.ticker,
        benchmark_ticker=receipt.benchmark_ticker,
        sessions_evaluated=receipt.sessions_evaluated,
        latest_price=receipt.latest_price,
        momentum_12_2_pct=receipt.momentum_12_2_pct + 15.0,  # TAMPER
        momentum_6_1_pct=receipt.momentum_6_1_pct,
        reversal_1_0_pct=receipt.reversal_1_0_pct,
        total_return_12m_pct=receipt.total_return_12m_pct,
        sma_200=receipt.sma_200,
        annualized_volatility_pct=receipt.annualized_volatility_pct,
        sharpe_momentum=receipt.sharpe_momentum,
        benchmark_12_2_pct=receipt.benchmark_12_2_pct,
        relative_momentum_spread_pct=receipt.relative_momentum_spread_pct,
        idiosyncratic_momentum_pct=receipt.idiosyncratic_momentum_pct,
        beta_to_benchmark=receipt.beta_to_benchmark,
        is_absolute_trend_positive=receipt.is_absolute_trend_positive,
        has_momentum_crash_risk=receipt.has_momentum_crash_risk,
        expected_momentum_alpha_pct=receipt.expected_momentum_alpha_pct,
        regime=receipt.regime,
        summary=receipt.summary,
        created_at_utc=receipt.created_at_utc,
        signature=receipt.signature,
    )
    assert tampered.verify_integrity() is False


def test_evaluate_dual_momentum_profile_end_to_end() -> None:
    """Test evaluate_dual_momentum_profile with synthetic calibrated fallback and validation."""
    receipt = evaluate_dual_momentum_profile("AAPL", benchmark_ticker="SPY")

    assert receipt.ticker == "AAPL"
    assert receipt.benchmark_ticker == "SPY"
    assert receipt.sessions_evaluated >= 42
    assert receipt.verify_integrity() is True
    assert isinstance(receipt.to_dict(), dict)


def test_cli_dual_momentum_command(capsys: pytest.CaptureFixture) -> None:
    """Test CLI dual-momentum handler with human-readable and JSON output formats."""
    # Test human-readable output
    args_text = argparse.Namespace(
        ticker="MSFT",
        benchmark="SPY",
        rf_rate=4.5,
        json=False,
    )
    dual_momentum_command(args_text)
    captured = capsys.readouterr().out
    assert "NextOmni Stock: Jegadeesh & Titman (1993) / Antonacci Dual Momentum Engine" in captured
    assert "MSFT" in captured
    assert "12-2 Intermediate Momentum" in captured

    # Test JSON output
    args_json = argparse.Namespace(
        ticker="NVDA",
        benchmark="QQQ",
        rf_rate=4.0,
        json=True,
    )
    dual_momentum_command(args_json)
    captured_json = capsys.readouterr().out
    payload = json.loads(captured_json)
    assert payload["ticker"] == "NVDA"
    assert payload["benchmark_ticker"] == "QQQ"
    assert "signature" in payload


def test_dual_momentum_input_validation() -> None:
    """Test security validation against invalid inputs, short series, and non-finite values."""
    # Insufficient length
    with pytest.raises(ValueError, match="requires at least 42 sessions"):
        calculate_dual_momentum([100.0] * 20, ticker="SHORT")

    # Non-finite price
    with pytest.raises(ValueError):
        calculate_dual_momentum([100.0] * 50 + [float("inf")], ticker="INF")

    # Non-positive price
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_dual_momentum([100.0] * 50 + [-10.0], ticker="NEG")

    # Invalid ticker format
    with pytest.raises(ValueError):
        evaluate_dual_momentum_profile("INVALID;DROP TABLE stock_observations;")
