"""Walk-forward time-series validation engine for NextOmni Stock.

Implements expanding-window chronological cross-validation with side-by-side
baseline benchmarking. Strictly prevents future lookahead leakage.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import PRICE_COLUMN
from src.evaluation.metrics import calculate_all_metrics
from src.features.builder import FEATURE_COLUMNS, build_feature_matrix
from src.models.baseline import PersistenceBaselineModel
from src.models.linear_model import LinearRegressionForecaster


class WalkForwardValidator:
    """Executes chronological expanding-window walk-forward validation."""

    def __init__(
        self,
        min_train_sessions: int = 120,
        step_size: int = 1,
    ):
        self.min_train_sessions = min_train_sessions
        self.step_size = step_size

    def evaluate(
        self,
        df: pd.DataFrame,
        horizon_sessions: int = 1,
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """Perform walk-forward validation comparing Linear Regression against Persistence Baseline.

        Args:
            df: Historical observations DataFrame.
            horizon_sessions: Forecast horizon in trading sessions (default: 1).
            alpha: Regularization strength for Ridge regression.

        Returns:
            Dict[str, Any]: Comprehensive out-of-sample comparison report.
        """
        # 1. Build features and align target
        feature_df, feat_cols = build_feature_matrix(
            df, include_target=True, horizon_sessions=horizon_sessions
        )

        # Drop the last horizon_sessions rows because their target hasn't happened yet
        valid_df = feature_df.dropna(subset=["target_price"]).copy().reset_index(drop=True)

        if len(valid_df) <= self.min_train_sessions:
            raise ValueError(
                f"Dataset length ({len(valid_df)}) must be greater than min_train_sessions ({self.min_train_sessions})."
            )

        origins = []
        target_dates = []
        actual_origins = []
        actual_targets = []
        model_predictions = []
        baseline_predictions = []

        total_sessions = len(valid_df)

        # Expanding window walk-forward loop
        for t in range(self.min_train_sessions, total_sessions, self.step_size):
            train_partition = valid_df.iloc[:t]
            test_row = valid_df.iloc[t]

            # Train linear model strictly on observations <= t
            forecaster = LinearRegressionForecaster(alpha=alpha)
            forecaster.fit(
                X=train_partition[feat_cols],
                y_target_price=train_partition["target_price"],
                p_origin=train_partition[PRICE_COLUMN],
            )

            # Predict for test row
            p_orig = float(test_row[PRICE_COLUMN])
            p_actual_target = float(test_row["target_price"])
            p_model_pred = float(
                forecaster.predict(
                    X=test_row[feat_cols].to_frame().T,
                    p_origin=p_orig,
                )[0]
            )
            # Baseline benchmark: P_{t+k} = P_t
            p_baseline_pred = p_orig

            origins.append(str(test_row["session_date"]))
            target_dates.append(str(test_row["target_session_date"]))
            actual_origins.append(p_orig)
            actual_targets.append(p_actual_target)
            model_predictions.append(p_model_pred)
            baseline_predictions.append(p_baseline_pred)

        # Calculate metrics for Linear Regression
        model_metrics = calculate_all_metrics(
            actual_origin=actual_origins,
            actual_target=actual_targets,
            predicted_target=model_predictions,
        )

        # Calculate metrics for Persistence Baseline on the EXACT same test origins
        baseline_metrics = calculate_all_metrics(
            actual_origin=actual_origins,
            actual_target=actual_targets,
            predicted_target=baseline_predictions,
        )

        # Compute relative improvements
        base_mae = baseline_metrics["mae"]
        model_mae = model_metrics["mae"]
        mae_improvement_pct = round(
            ((base_mae - model_mae) / base_mae) * 100 if base_mae > 0 else 0.0, 2
        )

        base_rmse = baseline_metrics["rmse"]
        model_rmse = model_metrics["rmse"]
        rmse_improvement_pct = round(
            ((base_rmse - model_rmse) / base_rmse) * 100 if base_rmse > 0 else 0.0, 2
        )

        # Train a final model on all available feature data for coefficients export
        final_forecaster = LinearRegressionForecaster(alpha=alpha)
        final_forecaster.fit(
            X=valid_df[feat_cols],
            y_target_price=valid_df["target_price"],
            p_origin=valid_df[PRICE_COLUMN],
        )

        return {
            "ticker": str(df["ticker"].iloc[0]),
            "exchange": str(df["exchange"].iloc[0]),
            "horizon_sessions": horizon_sessions,
            "walk_forward_steps": len(model_predictions),
            "evaluation_period_start": origins[0],
            "evaluation_period_end": origins[-1],
            "model_metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "mae_improvement_pct": mae_improvement_pct,
            "rmse_improvement_pct": rmse_improvement_pct,
            "linear_model_coefficients": final_forecaster.get_coefficients(),
            "outperforms_baseline_mae": model_mae < base_mae,
            "directional_alpha_edge": (
                model_metrics["directional_accuracy_pct"]
                > baseline_metrics["directional_accuracy_pct"]
            ),
        }
