"""Unit and integration test suite for Jegadeesh & Lehmann (1990) Short-Term Reversal Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.short_term_reversal import (
    ShortTermReversalReceipt,
    calculate_short_term_reversal,
    classify_reversal_regime,
    evaluate_short_term_reversal_profile,
)
from src.cli import reversal_command


def test_calculate_short_term_reversal_liquidity_dip_buy() -> None:
    """Test high-conviction liquidity dip in an established intermediate bull trend."""
    n = 120
    # Strong upward trend for the first 114 sessions: price rises from 100 to 145 (+45%)
    prices = [100.0 * (1.0 + 0.45 * (i / 114)) for i in range(115)]
    # Sharp transient 1-week liquidity drop over final 5 sessions (-5.5%)
    peak = prices[-1]
    drop_pct = [0.015, 0.025, 0.038, 0.048, 0.055]
    for d in drop_pct:
        prices.append(peak * (1.0 - d))

    volumes = [1_000_000.0] * (n - 5) + [2_500_000.0] * 5  # Volume expansion on the drop

    receipt = calculate_short_term_reversal(
        prices=prices,
        volumes=volumes,
        ticker="DIP_BUYER",
        benchmark_ticker="SPY",
    )

    assert receipt.ticker == "DIP_BUYER"
    assert receipt.sessions_evaluated == n
    assert receipt.return_1w_pct <= -4.0
    assert receipt.is_bullish_intermediate_trend is True
    assert receipt.is_liquidity_dip_buy is True
    assert receipt.is_falling_knife is False
    assert receipt.volume_shock_zscore > 1.0
    assert receipt.expected_contrarian_alpha_pct > 5.0
    assert receipt.regime == "HIGH_CONVICTION_LIQUIDITY_DIP"
    assert receipt.verify_integrity() is True


def test_calculate_short_term_reversal_falling_knife() -> None:
    """Test falling knife avoidance when 1-week drop occurs within a deteriorating secular bear trend."""
    n = 120
    # Chronic bear market: prices declining from 100 to 70 over 115 sessions, then plunging by 8% over final 5 sessions
    prices = [100.0 * (1.0 - 0.30 * (i / 114)) for i in range(115)]
    base = prices[-1]
    for d in [0.018, 0.035, 0.052, 0.068, 0.082]:
        prices.append(base * (1.0 - d))

    receipt = calculate_short_term_reversal(
        prices=prices,
        ticker="KNIFE",
        benchmark_ticker="SPY",
    )

    assert receipt.return_1w_pct < 0.0
    assert receipt.is_bullish_intermediate_trend is False
    assert receipt.is_falling_knife is True
    assert receipt.is_liquidity_dip_buy is False
    assert receipt.regime == "SECULAR_FALLING_KNIFE"
    assert receipt.expected_contrarian_alpha_pct == -12.5


def test_negative_serial_correlation_proof() -> None:
    """Mathematical verification that oscillating price paths exhibit negative first-order serial correlation."""
    n = 60
    # Mean-reverting zigzag path
    prices = [100.0]
    for i in range(1, n):
        shock = -0.015 if (i % 2 == 1) else +0.016
        prices.append(prices[-1] * (1.0 + shock))

    receipt = calculate_short_term_reversal(prices, ticker="MEAN_REV")

    assert receipt.serial_correlation_rho1 < -0.10
    assert receipt.roll_spread_proxy_pct > 0.0
    assert receipt.verify_integrity() is True


def test_overbought_mean_reversion() -> None:
    """Test overbought parabolic surge triggering short-term mean-reverting pullback warning."""
    n = 60
    prices = [100.0] * (n - 6)
    # Parabolic short squeeze in the final 5 sessions (+10% gain)
    for i in range(1, 7):
        prices.append(100.0 * (1.0 + 0.018 * i))

    receipt = calculate_short_term_reversal(prices, ticker="OVERBOUGHT")

    assert receipt.return_1w_pct >= 7.0
    assert receipt.regime == "OVERBOUGHT_MEAN_REVERSION"
    assert receipt.expected_contrarian_alpha_pct == -8.0


def test_short_term_reversal_receipt_hmac_tamper() -> None:
    """Verify cryptographic authenticity and tamper detection of ShortTermReversalReceipt."""
    prices = [100.0 * (1.0 + 0.001 * i) for i in range(50)]
    receipt = calculate_short_term_reversal(prices, ticker="HMAC_TEST")

    assert receipt.verify_integrity() is True

    # Tamper with 1-week return
    tampered = ShortTermReversalReceipt(
        ticker=receipt.ticker,
        benchmark_ticker=receipt.benchmark_ticker,
        sessions_evaluated=receipt.sessions_evaluated,
        latest_price=receipt.latest_price,
        return_1w_pct=receipt.return_1w_pct + 10.0,  # TAMPER
        return_2w_pct=receipt.return_2w_pct,
        benchmark_1w_pct=receipt.benchmark_1w_pct,
        excess_return_1w_pct=receipt.excess_return_1w_pct,
        serial_correlation_rho1=receipt.serial_correlation_rho1,
        roll_spread_proxy_pct=receipt.roll_spread_proxy_pct,
        volume_shock_zscore=receipt.volume_shock_zscore,
        intermediate_momentum_12_2_pct=receipt.intermediate_momentum_12_2_pct,
        is_bullish_intermediate_trend=receipt.is_bullish_intermediate_trend,
        expected_contrarian_alpha_pct=receipt.expected_contrarian_alpha_pct,
        is_liquidity_dip_buy=receipt.is_liquidity_dip_buy,
        is_falling_knife=receipt.is_falling_knife,
        regime=receipt.regime,
        summary=receipt.summary,
        created_at_utc=receipt.created_at_utc,
        signature=receipt.signature,
    )
    assert tampered.verify_integrity() is False


def test_evaluate_short_term_reversal_profile_end_to_end() -> None:
    """Test evaluate_short_term_reversal_profile end-to-end with database fallback and valid dict schema."""
    receipt = evaluate_short_term_reversal_profile("AAPL", benchmark_ticker="SPY")

    assert receipt.ticker == "AAPL"
    assert receipt.benchmark_ticker == "SPY"
    assert receipt.sessions_evaluated >= 30
    assert receipt.verify_integrity() is True
    assert isinstance(receipt.to_dict(), dict)


def test_cli_reversal_command(capsys: pytest.CaptureFixture) -> None:
    """Test CLI reversal command handler with table formatting and JSON output."""
    # Test human-readable table
    args_text = argparse.Namespace(
        ticker="AMD",
        benchmark="SPY",
        json=False,
    )
    reversal_command(args_text)
    captured = capsys.readouterr().out
    assert "NextOmni Stock: Jegadeesh & Lehmann (1990) Short-Term Reversal Engine" in captured
    assert "AMD" in captured
    assert "1-Week Return" in captured

    # Test JSON output
    args_json = argparse.Namespace(
        ticker="INTC",
        benchmark="QQQ",
        json=True,
    )
    reversal_command(args_json)
    captured_json = capsys.readouterr().out
    payload = json.loads(captured_json)
    assert payload["ticker"] == "INTC"
    assert payload["benchmark_ticker"] == "QQQ"
    assert "return_1w_pct" in payload
    assert "signature" in payload


def test_short_term_reversal_input_validation() -> None:
    """Test defensive security validation on input lengths, non-finite values, and tickers."""
    # Insufficient length (< 30)
    with pytest.raises(ValueError, match="requires at least 30 sessions"):
        calculate_short_term_reversal([100.0] * 15, ticker="SHORT")

    # Non-finite price
    with pytest.raises(ValueError, match="Prices cannot contain NaN or infinite values"):
        calculate_short_term_reversal([100.0] * 40 + [float("nan")], ticker="NAN")

    # Non-positive price
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_short_term_reversal([100.0] * 40 + [-5.0], ticker="NEG")

    # Invalid ticker format
    with pytest.raises(ValueError):
        evaluate_short_term_reversal_profile("INVALID; DROP TABLE predictions;")
