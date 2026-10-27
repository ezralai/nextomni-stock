"""Tests for NextOmni Stock prediction tracking, lifecycle transitions, and idempotent evaluation."""

from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import pytest

from src.data.base_provider import BaseMarketDataProvider
from src.data.db import get_connection, get_predictions, init_db, save_observations, upsert_stock
from src.evaluation.tracker import PredictionTracker


class MockEvalProvider(BaseMarketDataProvider):
    """Mock provider with controllable observations for evaluation testing."""

    @property
    def name(self) -> str:
        return "MockEvalProvider"

    def fetch_historical_data(
        self,
        ticker: str,
        exchange: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Return observation for 2024-01-16 at price 172.0
        df = pd.DataFrame(
            {
                "ticker": [ticker],
                "exchange": [exchange],
                "session_date": [start_date],
                "open": [170.0],
                "high": [173.0],
                "low": [169.5],
                "close": [172.0],
                "adj_close": [172.0],
                "volume": [1000000],
                "data_status": ["END_OF_DAY"],
                "provider": [self.name],
            }
        )
        return df, {}

    def fetch_corporate_actions(self, ticker: str, exchange: str) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_latest_quote(self, ticker: str, exchange: str) -> Dict[str, Any]:
        return {}


@pytest.fixture
def test_tracker_db(tmp_path: Path) -> Tuple[PredictionTracker, Path]:
    """Create isolated PredictionTracker with test SQLite database."""
    db_file = tmp_path / "test_tracker.db"
    init_db(db_file)
    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", db_file)
    tracker = PredictionTracker(provider=MockEvalProvider(), db_path=db_file)
    return tracker, db_file


def test_record_prediction(test_tracker_db: Tuple[PredictionTracker, Path]):
    """Verify recording an immutable prediction persists to SQLite with status PENDING."""
    tracker, db_file = test_tracker_db

    record = tracker.record_prediction(
        ticker="JPM",
        exchange="XNYS",
        forecast_origin_session="2024-01-12",
        origin_price=170.0,
        horizon_sessions=1,
        target_session_date="2024-01-16",
        predicted_price=171.5,
        model_name="LinearRegression_Ridge",
        model_version="1.0.0",
        reproducibility_ref="models/linear_JPM.joblib",
    )

    assert record["prediction_id"] is not None
    assert record["status"] == "PENDING"
    assert record["predicted_price"] == 171.5

    df = get_predictions(db_path=db_file)
    assert len(df) == 1
    assert df["prediction_id"].iloc[0] == record["prediction_id"]
    assert df["status"].iloc[0] == "PENDING"
    assert pd.isna(df["actual_price"].iloc[0])


def test_evaluate_pending_future_target(test_tracker_db: Tuple[PredictionTracker, Path]):
    """Verify that predictions whose target date is in the future remain PENDING."""
    tracker, db_file = test_tracker_db

    tracker.record_prediction(
        ticker="JPM",
        exchange="XNYS",
        forecast_origin_session="2026-01-01",
        origin_price=200.0,
        horizon_sessions=1,
        target_session_date="2099-01-01",  # Far future
        predicted_price=202.0,
        model_name="LinearRegression_Ridge",
        model_version="1.0.0",
    )

    summary = tracker.evaluate_pending_predictions()
    assert summary["pending"] == 1
    assert summary["evaluated"] == 0

    df = get_predictions(status="PENDING", db_path=db_file)
    assert len(df) == 1


def test_evaluate_and_reconcile_past_target(test_tracker_db: Tuple[PredictionTracker, Path]):
    """Verify that completed past target session is matched with actual price and evaluated."""
    tracker, db_file = test_tracker_db

    # Origin price 170.0, predicted 171.5, actual returned by MockEvalProvider will be 172.0
    rec = tracker.record_prediction(
        ticker="JPM",
        exchange="XNYS",
        forecast_origin_session="2024-01-12",
        origin_price=170.0,
        horizon_sessions=1,
        target_session_date="2024-01-16",  # Past date
        predicted_price=171.5,
        model_name="LinearRegression_Ridge",
        model_version="1.0.0",
    )

    summary = tracker.evaluate_pending_predictions()
    assert summary["evaluated"] == 1

    df = get_predictions(db_path=db_file)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "EVALUATED"
    assert row["actual_price"] == 172.0

    # Error: |171.5 - 172.0| = 0.5
    assert pytest.approx(row["absolute_error"], 0.0001) == 0.5
    # Percentage error: (0.5 / 172.0) * 100 = ~0.29%
    assert pytest.approx(row["percentage_error"], 0.01) == round((0.5 / 172.0) * 100, 2)
    # Direction: Origin was 170.0, pred was 171.5 (+1), actual was 172.0 (+1) -> Correct!
    assert bool(row["direction_correct"]) is True


def test_evaluation_idempotency(test_tracker_db: Tuple[PredictionTracker, Path]):
    """Verify running evaluation workflow repeatedly does NOT duplicate records or change metrics."""
    tracker, db_file = test_tracker_db

    tracker.record_prediction(
        ticker="JPM",
        exchange="XNYS",
        forecast_origin_session="2024-01-12",
        origin_price=170.0,
        horizon_sessions=1,
        target_session_date="2024-01-16",
        predicted_price=171.5,
        model_name="LinearRegression_Ridge",
        model_version="1.0.0",
    )

    # First run
    s1 = tracker.evaluate_pending_predictions()
    assert s1["evaluated"] == 1

    # Second run immediately after
    s2 = tracker.evaluate_pending_predictions()
    assert s2["evaluated"] == 0  # Nothing newly evaluated

    # Check database rows
    conn = get_connection(db_file)
    eval_count = conn.execute("SELECT COUNT(*) FROM prediction_evaluations;").fetchone()[0]
    pred_count = conn.execute("SELECT COUNT(*) FROM predictions;").fetchone()[0]
    assert eval_count == 1
    assert pred_count == 1
    conn.close()
