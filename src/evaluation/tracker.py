"""Prediction Tracking & Idempotent Evaluation Workflow for NextOmni Stock.

Implements immutable prediction recording, lifecycle state transitions
(PENDING -> AWAITING_ACTUAL -> EVALUATED / FAILED), and reconciled scoring.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import exchange_calendars as xcals
import numpy as np
import pandas as pd

from src.config import (
    DATABASE_PATH,
    PRICE_COLUMN,
    STATUS_AWAITING_ACTUAL,
    STATUS_EVALUATED,
    STATUS_FAILED,
    STATUS_PENDING,
    SUPPORTED_EXCHANGES,
)
from src.data.base_provider import BaseMarketDataProvider
from src.data.db import (
    get_connection,
    get_observations,
    get_predictions,
    init_db,
    save_observations,
    save_prediction,
    save_prediction_evaluation,
    update_prediction_status,
)
from src.data.yfinance_provider import YahooFinanceProvider


class PredictionTracker:
    """Manages the full lifecycle of predictions from generation to reconciliation."""

    def __init__(
        self,
        provider: Optional[BaseMarketDataProvider] = None,
        db_path: Optional[Path] = None,
    ):
        self.provider = provider or YahooFinanceProvider()
        self.db_path = db_path or DATABASE_PATH
        init_db(self.db_path)

    def record_prediction(
        self,
        ticker: str,
        exchange: str,
        forecast_origin_session: str,
        origin_price: float,
        horizon_sessions: int,
        target_session_date: str,
        predicted_price: float,
        model_name: str,
        model_version: str,
        reproducibility_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create and store an immutable prediction record."""
        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange '{exchange}'")

        cfg = SUPPORTED_EXCHANGES[exchange]
        prediction_id = str(uuid.uuid4())
        horizon_type = f"{horizon_sessions}_SESSION" if horizon_sessions == 1 else f"{horizon_sessions}_SESSIONS"
        now_ts = datetime.now().isoformat()

        record = {
            "prediction_id": prediction_id,
            "ticker": ticker,
            "exchange": exchange,
            "currency": cfg.currency,
            "forecast_origin_timestamp": now_ts,
            "forecast_origin_session": forecast_origin_session,
            "origin_price": float(origin_price),
            "horizon_type": horizon_type,
            "horizon_sessions": int(horizon_sessions),
            "target_session_date": target_session_date,
            "predicted_price": float(predicted_price),
            "model_name": model_name,
            "model_version": model_version,
            "adjustment_convention": "ADJUSTED_CLOSE",
            "reproducibility_ref": reproducibility_ref or "",
            "status": STATUS_PENDING,
        }

        save_prediction(record, db_path=self.db_path)
        return record

    def evaluate_pending_predictions(self) -> Dict[str, Any]:
        """Evaluate all eligible pending predictions against actual market observations.

        Idempotent: Can be executed repeatedly without corrupting or duplicating evaluations.
        """
        pending_df = get_predictions(db_path=self.db_path)
        if pending_df.empty:
            return {"evaluated": 0, "awaiting": 0, "pending": 0, "failed": 0}

        # Filter for predictions that need evaluation
        unresolved = pending_df[
            pending_df["status"].isin([STATUS_PENDING, STATUS_AWAITING_ACTUAL])
        ]

        evaluated_count = 0
        awaiting_count = 0
        pending_count = 0
        failed_count = 0

        today_str = datetime.now().strftime("%Y-%m-%d")

        for _, pred in unresolved.iterrows():
            pred_id = pred["prediction_id"]
            ticker = pred["ticker"]
            exchange = pred["exchange"]
            target_date = pred["target_session_date"]
            origin_price = float(pred["origin_price"])
            pred_price = float(pred["predicted_price"])

            # 1. Check if target session is in the future
            if target_date > today_str:
                pending_count += 1
                continue

            # 2. Check if actual observation already exists in SQLite
            obs_df = get_observations(
                ticker=ticker,
                start_date=target_date,
                end_date=target_date,
                db_path=self.db_path,
            )

            actual_price: Optional[float] = None
            if not obs_df.empty and not pd.isna(obs_df[PRICE_COLUMN].iloc[0]):
                actual_price = float(obs_df[PRICE_COLUMN].iloc[0])
                source = "SQLite_Cache"
            else:
                # 3. Query provider to see if target session is available
                try:
                    end_fetch = (pd.Timestamp(target_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    fetched_df, _ = self.provider.fetch_historical_data(
                        ticker=ticker,
                        exchange=exchange,
                        start_date=target_date,
                        end_date=end_fetch,
                    )
                    if not fetched_df.empty:
                        save_observations(fetched_df, db_path=self.db_path)
                        # Find the exact row for target_date
                        target_row = fetched_df[fetched_df["session_date"] == target_date]
                        if not target_row.empty:
                            actual_price = float(target_row[PRICE_COLUMN].iloc[0])
                            source = self.provider.name
                        else:
                            actual_price = None
                except Exception:
                    actual_price = None


            # 4. If actual price is still unavailable, mark as awaiting
            if actual_price is None or actual_price <= 0:
                update_prediction_status(
                    pred_id, STATUS_AWAITING_ACTUAL, db_path=self.db_path
                )
                awaiting_count += 1
                continue

            # 5. Price confirmed: calculate exact evaluation metrics
            abs_err = abs(pred_price - actual_price)
            denom = max(actual_price, 1e-8)
            pct_err = (abs_err / denom) * 100.0

            actual_dir = int(np.sign(actual_price - origin_price))
            pred_dir = int(np.sign(pred_price - origin_price))
            dir_correct = bool(actual_dir == pred_dir)

            eval_record = {
                "prediction_id": pred_id,
                "actual_price": actual_price,
                "actual_source": source,
                "actual_observed_at": datetime.now().isoformat(),
                "absolute_error": round(abs_err, 4),
                "percentage_error": round(pct_err, 2),
                "actual_direction": actual_dir,
                "predicted_direction": pred_dir,
                "direction_correct": dir_correct,
            }

            save_prediction_evaluation(eval_record, db_path=self.db_path)
            evaluated_count += 1

        return {
            "evaluated": evaluated_count,
            "awaiting": awaiting_count,
            "pending": pending_count,
            "failed": failed_count,
        }
