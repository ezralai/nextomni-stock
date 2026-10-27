"""Unit tests for directional win-probability calibration and provider quota telemetry."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.probability import calculate_directional_probability
from src.cli import predict_command
from src.data.db import init_db, save_observations, upsert_stock
from src.data.rate_limiter import ProviderQuotaTracker, get_quota_tracker
from src.models.multi_horizon import MultiHorizonForecaster


def test_calculate_directional_probability_outcomes():
    """Verify Gaussian directional win probability calculations and categorical conviction."""
    # Strong Bullish (+3.0% return with 1.0% residual std -> z = +3.0)
    p_strong_bull, conv_strong_bull = calculate_directional_probability(0.03, 0.01)
    assert p_strong_bull > 0.99
    assert conv_strong_bull == "STRONG BULLISH"

    # Lean Bullish (+0.5% return with 2.0% residual std -> z = +0.25 -> ~59.8%)
    p_lean_bull, conv_lean_bull = calculate_directional_probability(0.005, 0.02)
    assert 0.55 <= p_lean_bull < 0.65
    assert conv_lean_bull == "LEAN BULLISH"

    # Neutral (0.0% return -> z = 0.0 -> 50.0%)
    p_neutral, conv_neutral = calculate_directional_probability(0.0, 0.02)
    assert abs(p_neutral - 0.50) < 1e-3
    assert conv_neutral == "NEUTRAL"

    # Lean Bearish (-0.5% return with 2.0% residual std -> ~40.1%)
    p_lean_bear, conv_lean_bear = calculate_directional_probability(-0.005, 0.02)
    assert 0.35 <= p_lean_bear <= 0.45
    assert conv_lean_bear == "LEAN BEARISH"

    # Strong Bearish (-3.0% return with 1.0% residual std -> < 1%)
    p_strong_bear, conv_strong_bear = calculate_directional_probability(-0.03, 0.01)
    assert p_strong_bear < 0.01
    assert conv_strong_bear == "STRONG BEARISH"


def test_calculate_directional_probability_zero_or_negative_std():
    """Verify zero or invalid residual std uses safety floor without raising error."""
    p_zero_std, conv = calculate_directional_probability(0.01, 0.0)
    assert np.isfinite(p_zero_std)
    assert p_zero_std > 0.99
    assert conv == "STRONG BULLISH"

    p_neg_std, conv_neg = calculate_directional_probability(-0.01, -0.05)
    assert np.isfinite(p_neg_std)
    assert p_neg_std < 0.01
    assert conv_neg == "STRONG BEARISH"


def test_provider_quota_tracker_sliding_window():
    """Verify ProviderQuotaTracker enforces rate limits and tracks telemetry."""
    tracker = ProviderQuotaTracker(provider="TestProvider", max_requests_per_minute=3)
    tracker.reset()

    # 1st, 2nd, 3rd requests succeed
    assert tracker.acquire() is True
    assert tracker.acquire() is True
    assert tracker.acquire() is True

    # 4th request within 60 seconds is throttled
    assert tracker.acquire() is False

    tel = tracker.get_telemetry()
    assert tel.total_requests == 3
    assert tel.requests_last_minute == 3
    assert tel.quota_remaining == 0
    assert tel.is_throttled is True

    # Record failover event
    tracker.record_failover()
    tel2 = tracker.get_telemetry()
    assert tel2.failover_count == 1

    tracker.reset()
    tel3 = tracker.get_telemetry()
    assert tel3.total_requests == 0
    assert tel3.quota_remaining == 3


def test_multi_horizon_predict_directional_odds(tmp_path):
    """Verify MultiHorizonForecaster.predict_all_horizons attaches win prob & conviction."""
    dates = pd.date_range("2024-01-01", periods=65, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = [100.0]
    for r in np.random.normal(0.001, 0.01, size=64):
        prices.append(prices[-1] * (1.0 + r))

    df = pd.DataFrame({
        "ticker": ["AAPL"] * 65,
        "exchange": ["XNYS"] * 65,
        "session_date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "adj_close": prices,
        "volume": [10000] * 65,
        "data_status": ["END_OF_DAY"] * 65,
        "provider": ["Mock"] * 65,
    })

    forecaster = MultiHorizonForecaster(model_family="linear")
    forecaster.fit(df)
    forecasts = forecaster.predict_all_horizons(df, "AAPL", "XNYS")

    for h_key in ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]:
        f = forecasts[h_key]
        assert "directional_win_probability" in f
        assert "conviction_label" in f
        assert 0.0 <= f["directional_win_probability"] <= 1.0
        assert f["conviction_label"] in {
            "STRONG BULLISH",
            "LEAN BULLISH",
            "NEUTRAL",
            "LEAN BEARISH",
            "STRONG BEARISH",
        }


def test_cli_predict_directional_win_prob_output(tmp_path, monkeypatch, capsys):
    """Verify nextomni predict CLI displays Directional Win Prob and Conviction badge."""
    import exchange_calendars as xcals
    cal = xcals.get_calendar("XNYS")
    dates = [dt.strftime("%Y-%m-%d") for dt in cal.sessions_in_range("2024-01-02", "2024-05-01")[:65]]
    prices = [150.0 + i * 0.5 for i in range(len(dates))]
    df = pd.DataFrame({
        "ticker": ["IBM"] * len(dates),
        "exchange": ["XNYS"] * len(dates),
        "session_date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "adj_close": prices,
        "volume": [10000] * len(dates),
        "data_status": ["END_OF_DAY"] * len(dates),
        "provider": ["Mock"] * len(dates),
    })

    db_path = tmp_path / "test_cli_prob.db"
    init_db(db_path)
    upsert_stock("IBM", "XNYS", "IBM Corp", "USD", "America/New_York", db_path=db_path)
    save_observations(df, db_path=db_path)

    # Train in-memory forecaster
    forecaster = MultiHorizonForecaster(model_family="linear")
    forecaster.fit(df)

    monkeypatch.setattr("src.data.db.get_observations", lambda ticker: df)
    monkeypatch.setattr("src.evaluation.tracker.DATABASE_PATH", db_path)
    monkeypatch.setattr("src.models.multi_horizon.MultiHorizonForecaster.load_models", lambda p, t: forecaster)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    args = argparse.Namespace(
        ticker="IBM",
        exchange="XNYS",
        horizon=1,
        model="linear",
        explain=False,
    )
    predict_command(args)

    out = capsys.readouterr().out
    assert "Directional Win Prob:" in out
    assert "% [" in out
