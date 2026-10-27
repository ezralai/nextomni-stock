"""Unit and mathematical invariant tests for market regime detection and term structure monotonicity."""

import math
import numpy as np
import pandas as pd
import pytest

from src.analytics.regime import MarketRegimeReceipt, detect_market_regime
from src.models.multi_horizon import MultiHorizonForecaster


def test_detect_market_regime_insufficient_data():
    """Verify ValueError is raised when series has fewer than 20 sessions."""
    df = pd.DataFrame({"adj_close": [100.0] * 10, "session_date": [f"2024-01-{i:02d}" for i in range(1, 11)]})
    with pytest.raises(ValueError, match="Insufficient observations"):
        detect_market_regime(df)


def test_detect_market_regime_bull_trend():
    """Verify steady upward price series with low vol produces BULL_TREND."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    # Low-vol linear upward climb
    prices = [100.0 + i * 0.5 for i in range(60)]
    df = pd.DataFrame({"adj_close": prices, "session_date": dates})

    receipt = detect_market_regime(df)
    assert isinstance(receipt, MarketRegimeReceipt)
    assert receipt.trend_regime == "BULLISH"
    assert receipt.volatility_regime == "LOW_VOL"
    assert receipt.regime == "BULL_TREND"
    assert receipt.sentiment_score > 0.0


def test_detect_market_regime_panic_crisis():
    """Verify violent downward trend with large volatility produces PANIC_CRISIS."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    # Volatile downward plunge
    prices = [200.0]
    for _ in range(59):
        shock = np.random.normal(-0.02, 0.04)  # High vol downward
        prices.append(max(prices[-1] * (1.0 + shock), 10.0))

    df = pd.DataFrame({"adj_close": prices, "session_date": dates})

    receipt = detect_market_regime(df)
    assert receipt.trend_regime == "BEARISH"
    assert receipt.volatility_regime == "HIGH_VOL"
    assert receipt.regime == "PANIC_CRISIS"
    assert receipt.sentiment_score < 0.0


def test_enforce_term_structure_monotonicity_corrects_inversion():
    """Verify enforce_term_structure_monotonicity expands inverted interval widths."""
    # Synthetic forecast where 1_SESSION interval is wider than 5_SESSIONS and 21_SESSIONS
    mock_forecasts = {
        "1_SESSION": {
            "horizon_sessions": 1,
            "predicted_price": 100.0,
            "lower_bound": 90.0,   # width = 20.0 (half-width 10.0)
            "upper_bound": 110.0,
        },
        "5_SESSIONS": {
            "horizon_sessions": 5,
            "predicted_price": 102.0,
            "lower_bound": 99.0,   # inverted width = 6.0 (half-width 3.0) -> SHOULD EXPAND
            "upper_bound": 105.0,
        },
        "21_SESSIONS": {
            "horizon_sessions": 21,
            "predicted_price": 105.0,
            "lower_bound": 103.0,  # inverted width = 4.0 (half-width 2.0) -> SHOULD EXPAND
            "upper_bound": 107.0,
        },
    }

    corrected = MultiHorizonForecaster.enforce_term_structure_monotonicity(mock_forecasts)

    w1 = corrected["1_SESSION"]["upper_bound"] - corrected["1_SESSION"]["lower_bound"]
    w5 = corrected["5_SESSIONS"]["upper_bound"] - corrected["5_SESSIONS"]["lower_bound"]
    w21 = corrected["21_SESSIONS"]["upper_bound"] - corrected["21_SESSIONS"]["lower_bound"]

    # Term-structure monotonicity must hold: Width(21) >= Width(5) >= Width(1)
    assert w5 >= w1
    assert w21 >= w5

    # Exactly scale-checked against sqrt(5/1) and sqrt(21/5)
    scale_1_to_5 = math.sqrt(5.0 / 1.0)
    expected_half_w5 = 10.0 * scale_1_to_5
    assert abs((corrected["5_SESSIONS"]["upper_bound"] - 102.0) - expected_half_w5) < 0.01


def test_multi_horizon_predict_all_has_monotonic_intervals():
    """Verify live MultiHorizonForecaster outputs mathematically monotonic interval widths."""
    dates = pd.date_range("2024-01-01", periods=65, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = [100.0]
    for r in np.random.normal(0.001, 0.012, size=64):
        prices.append(prices[-1] * (1.0 + r))

    df = pd.DataFrame({
        "ticker": ["TEST"] * 65,
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
    forecasts = forecaster.predict_all_horizons(df, "TEST", "XNYS")

    w1 = forecasts["1_SESSION"]["upper_bound"] - forecasts["1_SESSION"]["lower_bound"]
    w5 = forecasts["5_SESSIONS"]["upper_bound"] - forecasts["5_SESSIONS"]["lower_bound"]
    w21 = forecasts["21_SESSIONS"]["upper_bound"] - forecasts["21_SESSIONS"]["lower_bound"]

    assert w5 >= w1
    assert w21 >= w5
