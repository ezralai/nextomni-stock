"""Persistence baseline forecasting model for NextOmni Stock.

Implements the standard quantitative benchmark: P_{t+k} = P_t.
Any machine learning model must convincingly outperform this persistence benchmark
to be considered practically useful.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import exchange_calendars as xcals
import pandas as pd

from src.config import DATABASE_PATH, PRICE_COLUMN, SUPPORTED_EXCHANGES
from src.data.db import get_observations
from src.evaluation.metrics import calculate_all_metrics
from src.models.splitter import align_forecasting_target, chronological_train_test_split


class PersistenceBaselineModel:
    """Naive persistence benchmark model: Predicts future price equals current price."""

    def __init__(self, model_version: str = "1.0.0"):
        self.model_name = "PersistenceBaseline"
        self.model_version = model_version

    def predict(
        self,
        df: pd.DataFrame,
        price_col: str = PRICE_COLUMN,
    ) -> pd.Series:
        """Predict future price P_{t+k} as the latest observed price P_t.

        Returns:
            pd.Series: Vector of predictions where each element is P_t.
        """
        return df[price_col].copy()

    def evaluate(
        self,
        df: pd.DataFrame,
        horizon_sessions: int = 1,
        train_ratio: float = 0.8,
    ) -> Dict[str, Any]:
        """Evaluate baseline model on an out-of-sample chronological test set.

        Args:
            df: Historical observations DataFrame.
            horizon_sessions: Forecast horizon in trading sessions (default: 1).
            train_ratio: Fraction for chronological training split (default: 0.8).

        Returns:
            Dict[str, Any]: Performance metrics on out-of-sample test set.
        """
        # 1. Align target price
        aligned = align_forecasting_target(df, horizon_sessions=horizon_sessions)

        # 2. Drop rows where target is NaN (i.e. the very last `horizon_sessions` rows)
        evaluable = aligned.dropna(subset=["target_price"]).copy().reset_index(drop=True)

        # 3. Chronological Train/Test split
        train_df, test_df = chronological_train_test_split(evaluable, train_ratio=train_ratio)

        # 4. Generate predictions on test set: P_pred = P_t
        test_predictions = self.predict(test_df)

        # 5. Calculate metrics
        metrics = calculate_all_metrics(
            actual_origin=test_df[PRICE_COLUMN],
            actual_target=test_df["target_price"],
            predicted_target=test_predictions,
        )

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "horizon_sessions": horizon_sessions,
            "train_sessions_count": len(train_df),
            "test_sessions_count": len(test_df),
            "test_period_start": str(test_df["session_date"].iloc[0]),
            "test_period_end": str(test_df["session_date"].iloc[-1]),
            "metrics": metrics,
            "educational_note": (
                "The persistence baseline predicts P_{t+k} = P_t. In equity markets, daily prices "
                "follow near-random-walk behavior, which gives persistence low RMSE/MAE on price levels "
                "without providing any forward directional edge. Any machine learning model must beat this."
            ),
        }

    def generate_future_forecast(
        self,
        ticker: str,
        exchange: str,
        horizon_sessions: int = 1,
        db_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate out-of-sample forecast for the next trading session from latest DB observation.

        Uses exchange_calendars to resolve the exact target trading session date.
        """
        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange '{exchange}'")

        cfg = SUPPORTED_EXCHANGES[exchange]
        df = get_observations(ticker, db_path=db_path)
        if df.empty:
            raise ValueError(f"No market observations found in database for ticker '{ticker}'.")

        latest_obs = df.iloc[-1]
        origin_session = str(latest_obs["session_date"])
        origin_price = float(latest_obs[PRICE_COLUMN])

        # Resolve exact target date using exchange calendar
        calendar = xcals.get_calendar(exchange)
        origin_ts = pd.Timestamp(origin_session)

        # Advance horizon_sessions forward in the exchange calendar
        sessions_ahead = calendar.sessions_in_range(
            origin_ts, origin_ts + pd.Timedelta(days=horizon_sessions * 7)
        )
        # Find index of origin session
        try:
            origin_idx = list(sessions_ahead).index(origin_ts)
            target_ts = sessions_ahead[origin_idx + horizon_sessions]
        except (ValueError, IndexError):
            # Fallback to next_session if range indexing differs
            target_ts = calendar.next_session(origin_ts)

        target_date_str = target_ts.strftime("%Y-%m-%d")

        return {
            "ticker": ticker,
            "exchange": exchange,
            "currency": cfg.currency,
            "forecast_origin_date": origin_session,
            "forecast_origin_price": origin_price,
            "forecast_horizon": f"{horizon_sessions}_SESSION" if horizon_sessions == 1 else f"{horizon_sessions}_SESSIONS",
            "horizon_sessions": horizon_sessions,
            "target_session_date": target_date_str,
            "predicted_price": origin_price,  # Persistence benchmark: P_{t+1} = P_t
            "model_name": self.model_name,
            "model_version": self.model_version,
            "generated_at": datetime.now().isoformat(),
            "status": "PENDING",
        }
