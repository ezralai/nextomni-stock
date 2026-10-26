"""Tests for NextOmni Stock forecasting metrics, chronological splitting, and persistence baseline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.db import init_db, save_observations, upsert_stock
from src.evaluation.metrics import (
    calculate_all_metrics,
    calculate_directional_accuracy,
    calculate_mae,
    calculate_mape,
    calculate_rmse,
)
from src.models.baseline import PersistenceBaselineModel
from src.models.splitter import align_forecasting_target, chronological_train_test_split


def test_metrics_calculations():
    """Verify exact mathematical values for MAE, RMSE, MAPE, and DA."""
    actual = np.array([100.0, 105.0, 110.0, 95.0])
    predicted = np.array([102.0, 103.0, 112.0, 90.0])

    # Absolute errors: [2, 2, 2, 5] -> Mean = 11 / 4 = 2.75
    assert pytest.approx(calculate_mae(actual, predicted), 0.0001) == 2.75

    # Squared errors: [4, 4, 4, 25] -> Mean = 37 / 4 = 9.25 -> sqrt(9.25) ~ 3.04138
    assert pytest.approx(calculate_rmse(actual, predicted), 0.0001) == np.sqrt(9.25)

    # Percentage errors: [2/100, 2/105, 2/110, 5/95]
    expected_mape = np.mean([2 / 100, 2 / 105, 2 / 110, 5 / 95]) * 100
    assert pytest.approx(calculate_mape(actual, predicted), 0.01) == expected_mape


def test_mape_zero_denominator_safeguard():
    """Verify that MAPE does not divide by zero when actual price is 0."""
    actual = np.array([0.0, 10.0])
    predicted = np.array([1.0, 10.0])
    mape = calculate_mape(actual, predicted)
    assert not np.isnan(mape)
    assert not np.isinf(mape)


def test_directional_accuracy():
    """Verify Directional Accuracy against known direction movements."""
    # Origin: [100, 100, 100, 100]
    origin = np.array([100.0, 100.0, 100.0, 100.0])
    # Target went: [UP, DOWN, FLAT, UP]
    actual_target = np.array([105.0, 95.0, 100.0, 110.0])
    # Pred went:   [UP, DOWN, FLAT, DOWN] -> 3 out of 4 correct
    pred_target = np.array([102.0, 98.0, 100.0, 90.0])

    da = calculate_directional_accuracy(origin, actual_target, pred_target)
    assert pytest.approx(da, 0.01) == 75.0


def test_align_forecasting_target():
    """Verify target alignment creates target_price P_{t+1} with no lookahead leakage."""
    df = pd.DataFrame(
        {
            "session_date": ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"],
            "adj_close": [100.0, 102.0, 105.0, 107.0],
        }
    )

    aligned = align_forecasting_target(df, horizon_sessions=1, price_col="adj_close")

    # Row 0 (2024-01-08, 100.0) -> Target is Row 1 (2024-01-09, 102.0)
    assert aligned["target_price"].iloc[0] == 102.0
    assert aligned["target_session_date"].iloc[0] == "2024-01-09"

    # Row 1 (2024-01-09, 102.0) -> Target is Row 2 (2024-01-10, 105.0)
    assert aligned["target_price"].iloc[1] == 105.0

    # Row 3 (last row) has no future observed session, target must be NaN
    assert pd.isna(aligned["target_price"].iloc[3])
    assert pd.isna(aligned["target_session_date"].iloc[3])


def test_chronological_train_test_split():
    """Verify split is strictly temporal and preserves order."""
    dates = [f"2024-01-{i:02d}" for i in range(1, 11)]
    df = pd.DataFrame({"session_date": dates, "adj_close": list(range(10))})

    train_df, test_df = chronological_train_test_split(df, train_ratio=0.8)

    assert len(train_df) == 8
    assert len(test_df) == 2

    # All dates in train must precede all dates in test
    assert train_df["session_date"].max() < test_df["session_date"].min()

    with pytest.raises(ValueError, match="train_ratio must be between"):
        chronological_train_test_split(df, train_ratio=0.99)


def test_persistence_model_evaluation():
    """Verify Persistence Baseline evaluation on test data."""
    dates = [f"2024-01-{i:02d}" for i in range(1, 21)]
    # Trending series: 100, 101, 102...
    prices = [100.0 + i for i in range(20)]
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * 20,
            "exchange": ["XNYS"] * 20,
            "session_date": dates,
            "adj_close": prices,
        }
    )

    model = PersistenceBaselineModel()
    eval_res = model.evaluate(df, horizon_sessions=1, train_ratio=0.7)

    assert eval_res["model_name"] == "PersistenceBaseline"
    assert "metrics" in eval_res
    metrics = eval_res["metrics"]
    # For a series where each step increases by 1, predicting P_t for P_{t+1} always has an error of 1.0
    assert pytest.approx(metrics["mae"], 0.01) == 1.0


def test_persistence_generate_future_forecast(tmp_path: Path):
    """Verify future forecast generation uses exchange calendar to advance trading session."""
    test_db = tmp_path / "test.db"
    init_db(test_db)
    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", test_db)

    # Add a Friday session (e.g. Friday 2024-01-12)
    obs_df = pd.DataFrame(
        {
            "ticker": ["JPM"],
            "exchange": ["XNYS"],
            "session_date": ["2024-01-12"],  # Friday
            "open": [170.0],
            "high": [172.0],
            "low": [169.0],
            "close": [171.0],
            "adj_close": [170.5],
            "volume": [8000000],
            "data_status": ["END_OF_DAY"],
            "provider": ["Mock"],
        }
    )
    save_observations(obs_df, test_db)

    model = PersistenceBaselineModel()
    forecast = model.generate_future_forecast("JPM", "XNYS", horizon_sessions=1, db_path=test_db)

    assert forecast["ticker"] == "JPM"
    assert forecast["forecast_origin_date"] == "2024-01-12"
    assert forecast["forecast_origin_price"] == 170.5
    assert forecast["predicted_price"] == 170.5
    assert forecast["status"] == "PENDING"

    # Next session after Friday 2024-01-12:
    # Monday Jan 15 2024 was Martin Luther King Jr. Day (NYSE Holiday!).
    # So the next valid session must be Tuesday Jan 16 2024!
    assert forecast["target_session_date"] == "2024-01-16"
