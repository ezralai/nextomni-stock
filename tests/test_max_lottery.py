"""Unit and integration test suite for Bali, Cakici & Whitelaw (2011) MAX Effect & Lottery Demand Alpha Engine."""

import argparse
import io
import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.max_lottery import (
    MaxLotteryReceipt,
    classify_lottery_regime,
    compute_max_effect,
    evaluate_max_lottery_profile,
)
from src.cli import max_lottery_command


def test_max_lottery_synthetic_lottery_stock() -> None:
    """Verify detection of speculative lottery stock with massive upside spikes and positive skew."""
    np.random.seed(42)
    n = 60
    # Normal returns with 3 massive lottery spikes: +18%, +22%, +15%
    rets = np.random.normal(-0.002, 0.015, n)
    rets[10] = 0.18
    rets[25] = 0.22
    rets[45] = 0.15

    prices = 100.0 * np.exp(np.cumsum(rets))

    receipt = evaluate_max_lottery_profile(
        ticker="MEME",
        asset_prices=prices,
        top_n_days=5,
        lookback_window=60,
    )

    # Check order statistics
    assert receipt.max_1_pct >= 20.0  # +22% spike
    assert receipt.max_n_pct >= 8.0  # MAX_5 should exceed 8%
    assert receipt.max_to_min_ratio > 2.0  # Upside tail heavily dominates downside
    assert receipt.return_skewness > 1.0  # Strongly positive skew

    # High lottery attraction score and negative expected alpha drag
    assert receipt.lottery_demand_score >= 70.0
    assert receipt.expected_alpha_drag_ann_pct < -5.0
    assert receipt.lottery_regime == "EXTREME_LOTTERY_TRAP"


def test_max_lottery_steady_compounder() -> None:
    """Verify low-MAX prime steady compounder delivering positive alpha premium."""
    np.random.seed(123)
    n = 60
    # Low-volatility steady upward drift without extreme tail spikes
    rets = np.random.normal(0.0008, 0.006, n)
    prices = 100.0 * np.exp(np.cumsum(rets))

    receipt = evaluate_max_lottery_profile(
        ticker="BRK",
        asset_prices=prices,
        top_n_days=5,
        lookback_window=60,
    )

    # Modest MAX_5 returns
    assert receipt.max_n_pct < 3.0
    assert receipt.max_1_pct < 4.0
    # Low lottery demand score
    assert receipt.lottery_demand_score < 35.0
    # Commands positive low-MAX compounding premium
    assert receipt.expected_alpha_drag_ann_pct > 0.0
    assert receipt.lottery_regime == "PRIME_STEADY_COMPOUNDER"


def test_max_lottery_monotonicity_with_tail_spikes() -> None:
    """Verify that adding larger positive upside spikes strictly increases lottery score and alpha drag."""
    np.random.seed(99)
    n = 50
    base_rets = np.random.normal(0.0, 0.01, n)

    scores: list[float] = []
    max_5s: list[float] = []
    alpha_drags: list[float] = []

    # Inject progressively larger upside lottery spikes
    spike_magnitudes = [0.02, 0.06, 0.12, 0.25]

    for spike in spike_magnitudes:
        r = base_rets.copy()
        r[5] = spike
        r[15] = spike * 0.8
        r[25] = spike * 0.6

        _, max_n, _, _, _, _, l_score, a_drag, _ = compute_max_effect(r, top_n=5)
        scores.append(l_score)
        max_5s.append(max_n)
        alpha_drags.append(a_drag)

    # Monotonicity checks
    for i in range(len(spike_magnitudes) - 1):
        assert max_5s[i] < max_5s[i + 1]
        assert scores[i] < scores[i + 1]
        assert alpha_drags[i] >= alpha_drags[i + 1]


def test_max_lottery_regime_classification() -> None:
    """Verify lottery regime tier boundaries."""
    assert classify_lottery_regime(80.0, 9.0) == "EXTREME_LOTTERY_TRAP"
    assert classify_lottery_regime(60.0, 4.0) == "SPECULATIVE_OVERPRICING"
    assert classify_lottery_regime(45.0, 3.0) == "BALANCED_EQUILIBRIUM"
    assert classify_lottery_regime(25.0, 1.5) == "PRIME_STEADY_COMPOUNDER"


def test_max_lottery_receipt_tamper_integrity() -> None:
    """Verify HMAC-SHA256 signature generation and tamper rejection."""
    np.random.seed(77)
    n = 60
    prices = 150.0 + np.cumsum(np.random.normal(0.05, 1.0, n))

    receipt = evaluate_max_lottery_profile(
        ticker="AAPL",
        asset_prices=prices,
        top_n_days=5,
        lookback_window=50,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.verify_integrity() is True
    assert len(receipt.signature) == 64

    # Tampering with lottery demand score invalidates signature
    receipt.lottery_demand_score += 1.0
    assert receipt.verify_integrity() is False


def test_max_lottery_cli_command_plain(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with formatted terminal report."""
    np.random.seed(88)
    n = 60
    prices = 200.0 + np.cumsum(np.random.normal(0.0, 1.5, n))

    with patch("src.data.db.get_observations") as mock_db:
        mock_df = pd.DataFrame({"adj_close": prices})
        mock_db.return_value = mock_df

        args = argparse.Namespace(
            command="max-lottery",
            ticker="TSLA",
            top_n=5,
            lookback=50,
            json=False,
        )

        max_lottery_command(args)
        captured = capsys.readouterr().out

        assert "Bali et al. (2011) MAX Effect & Lottery Demand Engine" in captured
        assert "Security / Ticker:            TSLA" in captured
        assert "Single Highest Return (MAX_1):" in captured
        assert "Average Top 5 Return (MAX_5):" in captured
        assert "Lottery Demand Score (Λ):" in captured
        assert "Audit Digest (HMAC):" in captured
        assert "Verified: True" in captured


def test_max_lottery_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with JSON output."""
    np.random.seed(88)
    n = 60
    prices = 130.0 + np.cumsum(np.random.normal(0.0, 1.2, n))

    with patch("src.data.db.get_observations") as mock_db:
        mock_df = pd.DataFrame({"adj_close": prices})
        mock_db.return_value = mock_df

        args = argparse.Namespace(
            command="max-lottery",
            ticker="NVDA",
            top_n=5,
            lookback=50,
            json=True,
        )

        max_lottery_command(args)
        captured = capsys.readouterr().out
        data = json.loads(captured)

        assert data["ticker"] == "NVDA"
        assert "max_n_pct" in data
        assert "lottery_demand_score" in data
        assert "expected_alpha_drag_ann_pct" in data
        assert "signature" in data


def test_max_lottery_input_validation() -> None:
    """Verify input validation guards against invalid tickers or parameters."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        evaluate_max_lottery_profile(ticker="BAD$$$")

    with pytest.raises(ValueError, match="Insufficient price observations"):
        evaluate_max_lottery_profile(ticker="AAPL", asset_prices=[100.0] * 10)
