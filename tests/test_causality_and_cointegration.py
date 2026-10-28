"""Unit and integration tests for cross-market Granger causality and cointegration engine."""

import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.causality import (
    CointegrationReceipt,
    GrangerCausalityReceipt,
    compute_cointegration_spread_zscore,
    evaluate_engle_granger_cointegration,
    evaluate_granger_causality,
)


def test_granger_causality_true_lead_lag():
    """Verify Granger test detects true predictive lead-lag relationship."""
    np.random.seed(42)
    n = 120
    x = np.random.normal(0, 0.02, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + 0.7 * x[t - 1] + np.random.normal(0, 0.005)

    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    s_x = pd.Series(x, index=dates)
    s_y = pd.Series(y, index=dates)

    receipt = evaluate_granger_causality(s_x, s_y, max_lag=3, source_ticker="JPM", target_ticker="1155.KL")
    assert isinstance(receipt, GrangerCausalityReceipt)
    assert receipt.is_causal
    assert receipt.p_value < 0.01
    assert receipt.optimal_lag == 1
    assert receipt.f_statistic > 10.0


def test_granger_causality_independent_series():
    """Verify Granger test correctly identifies independent white noise as non-causal."""
    np.random.seed(123)
    n = 100
    x = np.random.normal(0, 0.02, n)
    y = np.random.normal(0, 0.02, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    receipt = evaluate_granger_causality(pd.Series(x, index=dates), pd.Series(y, index=dates), max_lag=3)
    assert isinstance(receipt, GrangerCausalityReceipt)
    assert not receipt.is_causal
    assert receipt.p_value > 0.05


def test_granger_causality_insufficient_data():
    """Verify graceful handling when sample size is insufficient."""
    s1 = pd.Series([0.01, -0.02], index=["2024-01-01", "2024-01-02"])
    s2 = pd.Series([0.02, 0.01], index=["2024-01-01", "2024-01-02"])
    receipt = evaluate_granger_causality(s1, s2, max_lag=5)
    assert not receipt.is_causal
    assert receipt.p_value == 1.0


def test_engle_granger_cointegration_synthetic_pair():
    """Verify Engle-Granger detects stationary mean-reverting spread between cointegrated assets."""
    np.random.seed(42)
    n = 180
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    # Independent random walk for P2
    p2 = 100.0 + np.cumsum(np.random.normal(0, 1.0, n))

    # Stationary Ornstein-Uhlenbeck spread
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.65 * spread[t - 1] + np.random.normal(0, 0.5)

    beta = 1.8
    alpha = 5.0
    p1 = alpha + beta * p2 + spread

    s_p1 = pd.Series(p1, index=dates)
    s_p2 = pd.Series(p2, index=dates)

    receipt = evaluate_engle_granger_cointegration(s_p1, s_p2, ticker_1="AAPL", ticker_2="MSFT")
    assert isinstance(receipt, CointegrationReceipt)
    assert receipt.is_cointegrated
    assert receipt.adf_statistic < receipt.critical_values["5%"]
    assert np.isclose(receipt.hedge_ratio_beta, beta, atol=0.15)
    assert 1.0 <= receipt.half_life_sessions <= 10.0


def test_engle_granger_cointegration_independent_random_walks():
    """Verify independent random walks are rejected as not cointegrated."""
    np.random.seed(99)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    p1 = 50.0 + np.cumsum(np.random.normal(0, 1.0, n))
    p2 = 100.0 + np.cumsum(np.random.normal(0, 1.0, n))

    receipt = evaluate_engle_granger_cointegration(pd.Series(p1, index=dates), pd.Series(p2, index=dates))
    assert isinstance(receipt, CointegrationReceipt)
    assert not receipt.is_cointegrated
    assert receipt.adf_statistic > receipt.critical_values["5%"]


def test_compute_cointegration_spread_zscore():
    """Verify rolling spread z-score calculation and threshold trading signals."""
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    p1 = pd.Series(np.linspace(100, 140, 40), index=dates)
    p2 = pd.Series(np.linspace(50, 70, 40), index=dates)

    spread_df = compute_cointegration_spread_zscore(p1, p2, hedge_ratio=2.0, intercept=0.0, rolling_window=15)
    assert isinstance(spread_df, pd.DataFrame)
    assert "spread" in spread_df.columns
    assert "z_score" in spread_df.columns
    assert "signal" in spread_df.columns
    assert len(spread_df) == 40
    valid_signals = {"SHORT_SPREAD", "LONG_SPREAD", "EXIT_POSITION", "HOLD", "NEUTRAL"}
    assert set(spread_df["signal"].unique()).issubset(valid_signals)


def test_cli_lead_lag_command(capsys):
    """Verify CLI nextomni lead-lag executes and outputs JSON."""
    from src.cli import main
    import sys

    dates = [f"2024-01-{i+1:02d}" for i in range(50)]
    df_dummy = pd.DataFrame({
        "session_date": dates,
        "adj_close": 100.0 + np.cumsum(np.random.normal(0, 1, 50)),
    })

    with patch("src.data.db.get_observations", return_value=df_dummy), \
         patch.object(sys, "argv", ["nextomni", "lead-lag", "--source", "JPM", "--target", "1155.KL", "--json"]):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["source_ticker"] == "JPM"
    assert data["target_ticker"] == "1155.KL"
    assert "is_causal" in data


def test_cli_cointegrate_command(capsys):
    """Verify CLI nextomni cointegrate executes and outputs JSON."""
    from src.cli import main
    import sys

    dates = [f"2024-01-{i+1:02d}" for i in range(50)]
    df_dummy = pd.DataFrame({
        "session_date": dates,
        "adj_close": 100.0 + np.cumsum(np.random.normal(0, 1, 50)),
    })

    with patch("src.data.db.get_observations", return_value=df_dummy), \
         patch.object(sys, "argv", ["nextomni", "cointegrate", "--ticker1", "JPM", "--ticker2", "1155.KL", "--json"]):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ticker_1"] == "JPM"
    assert data["ticker_2"] == "1155.KL"
    assert "is_cointegrated" in data
    assert "hedge_ratio_beta" in data
