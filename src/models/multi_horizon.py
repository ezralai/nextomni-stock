"""Multi-horizon forecasting engine for NextOmni Stock.

Implements direct forecasting models for:
- 1 Trading Session:  P_{t+1} (Next Day)
- 5 Trading Sessions: P_{t+5} (Next Trading Week)
- 21 Trading Sessions: P_{t+21} (Next Trading Month)

Direct forecasting trains dedicated estimators f_k(X_t) for each horizon k,
preventing the compounding errors inherent to recursive auto-regressive steps.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import exchange_calendars as xcals
import joblib
import pandas as pd

from src.config import (
    HORIZON_FIVE_SESSIONS,
    HORIZON_NEXT_SESSION,
    HORIZON_TWENTY_ONE_SESSIONS,
    PRICE_COLUMN,
    SUPPORTED_EXCHANGES,
    SUPPORTED_HORIZONS,
)
from src.data.db import get_observations
from src.evaluation.metrics import calculate_all_metrics
from src.features.builder import FEATURE_COLUMNS, build_feature_matrix
from src.models.base_model import BaseForecaster
from src.models.ensemble import ConvexEnsembleForecaster
from src.models.gb_model import GradientBoostingForecaster
from src.models.linear_model import LinearRegressionForecaster
from src.security.audit import sign_model_artifact, verify_model_artifact
from src.security.validators import validate_model_family


class MultiHorizonForecaster:
    """Manages dedicated direct forecasting models across multiple trading horizons."""

    def __init__(
        self,
        model_family: str = "linear",
        alpha: float = 1.0,
        model_version: str = "1.0.0",
        **model_kwargs: Any,
    ):
        self.model_family = validate_model_family(model_family)
        self.alpha = alpha
        self.model_version = model_version
        self.model_kwargs = model_kwargs
        self.horizons_map = {
            "1_SESSION": HORIZON_NEXT_SESSION,
            "5_SESSIONS": HORIZON_FIVE_SESSIONS,
            "21_SESSIONS": HORIZON_TWENTY_ONE_SESSIONS,
        }
        self.models: Dict[str, BaseForecaster] = {
            h_key: self._create_forecaster(h_key) for h_key in self.horizons_map
        }
        self.is_fitted: bool = False

    def _create_forecaster(self, horizon_key: str = "1_SESSION") -> BaseForecaster:
        """Instantiate forecaster instance according to configured model family and horizon.

        Applies Horizon-Adaptive Regularization:
        - 1_SESSION: Low signal-to-noise ratio requires strong shrinkage (max_depth=1,
          l2_regularization=20.0, min_samples_leaf=30, max_iter=20) to prevent overfitting daily noise.
        - 5_SESSIONS & 21_SESSIONS: Extended horizons capture persistent trend/momentum signals,
          benefiting from standard tree capacity (max_depth=3, l2=1.0, min_samples_leaf=15, max_iter=50).
        """
        if self.model_family == "gradient_boosting":
            kwargs = dict(self.model_kwargs)
            if horizon_key == "1_SESSION":
                kwargs.setdefault("max_depth", 1)
                kwargs.setdefault("l2_regularization", 20.0)
                kwargs.setdefault("min_samples_leaf", 30)
                kwargs.setdefault("max_iter", 20)
            else:
                kwargs.setdefault("max_depth", 3)
                kwargs.setdefault("l2_regularization", 1.0)
                kwargs.setdefault("min_samples_leaf", 15)
                kwargs.setdefault("max_iter", 50)
            return GradientBoostingForecaster(
                model_version=self.model_version,
                **kwargs,
            )
        if self.model_family == "ensemble":
            kwargs = dict(self.model_kwargs)
            if horizon_key == "1_SESSION":
                kwargs.setdefault("max_depth", 1)
                kwargs.setdefault("l2_regularization", 20.0)
                kwargs.setdefault("min_samples_leaf", 30)
                kwargs.setdefault("max_iter", 20)
            else:
                kwargs.setdefault("max_depth", 3)
                kwargs.setdefault("l2_regularization", 1.0)
                kwargs.setdefault("min_samples_leaf", 15)
                kwargs.setdefault("max_iter", 50)
            lin = LinearRegressionForecaster(alpha=self.alpha, model_version=self.model_version)
            gb = GradientBoostingForecaster(model_version=self.model_version, **kwargs)
            return ConvexEnsembleForecaster(
                linear_forecaster=lin,
                gb_forecaster=gb,
                model_version=self.model_version,
            )
        return LinearRegressionForecaster(
            alpha=self.alpha,
            model_version=self.model_version,
        )

    def fit(self, df: pd.DataFrame) -> "MultiHorizonForecaster":
        """Fit separate direct models for 1-session, 5-session, and 21-session horizons."""
        for h_key, h_sessions in self.horizons_map.items():
            feature_df, feat_cols = build_feature_matrix(
                df, include_target=True, horizon_sessions=h_sessions
            )
            trainable = feature_df.dropna(subset=["target_price"]).copy().reset_index(drop=True)

            if len(trainable) < 15:
                raise ValueError(
                    f"Insufficient observations ({len(trainable)}) to train horizon {h_key}."
                )


            self.models[h_key].fit(
                X=trainable[feat_cols],
                y_target_price=trainable["target_price"],
                p_origin=trainable[PRICE_COLUMN],
            )

        self.is_fitted = True
        return self

    def predict_all_horizons(
        self,
        df: pd.DataFrame,
        ticker: str,
        exchange: str,
        confidence_level: float = 0.80,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate out-of-sample predictions across all 3 horizons from latest observation.

        Uses exchange_calendars to resolve exact future target dates and computes
        calibrated confidence intervals.
        """
        if not self.is_fitted:
            raise RuntimeError("MultiHorizonForecaster must be fitted before predict_all_horizons.")

        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange '{exchange}'")

        cfg = SUPPORTED_EXCHANGES[exchange]
        cal = xcals.get_calendar(exchange)

        # Feature vector for the latest observation
        feature_df, feat_cols = build_feature_matrix(df, include_target=False)
        latest_row = feature_df.iloc[-1]
        latest_features = latest_row[feat_cols].to_frame().T

        origin_session = str(latest_row["session_date"])
        origin_price = float(latest_row[PRICE_COLUMN])
        origin_ts = pd.Timestamp(origin_session)

        forecasts = {}

        for h_key, h_sessions in self.horizons_map.items():
            # Resolve target trading session date
            sessions_ahead = cal.sessions_in_range(
                origin_ts, origin_ts + pd.Timedelta(days=h_sessions * 7)
            )
            try:
                idx = list(sessions_ahead).index(origin_ts)
                target_ts = sessions_ahead[idx + h_sessions]
            except Exception:
                target_ts = origin_ts + pd.Timedelta(days=h_sessions)

            target_date_str = target_ts.strftime("%Y-%m-%d")

            # Predict price and calibrated prediction interval
            model = self.models[h_key]
            pred_arr, low_arr, up_arr = model.predict_interval(
                latest_features, p_origin=origin_price, confidence_level=confidence_level
            )
            predicted_price = float(pred_arr[0])
            lower_bound = float(low_arr[0])
            upper_bound = float(up_arr[0])

            expected_return_pct = round(
                ((predicted_price - origin_price) / origin_price) * 100.0, 2
            )

            # Local signal attribution decomposition
            signal_attributions = model.explain_forecast(latest_features)

            forecasts[h_key] = {
                "ticker": ticker,
                "exchange": exchange,
                "currency": cfg.currency,
                "horizon_type": h_key,
                "horizon_sessions": h_sessions,
                "forecast_origin_session": origin_session,
                "origin_price": origin_price,
                "target_session_date": target_date_str,
                "predicted_price": round(predicted_price, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "confidence_level": confidence_level,
                "expected_return_pct": expected_return_pct,
                "signal_attributions": signal_attributions,
                "model_name": f"{model.model_name}_Direct_{h_key}",
                "model_version": self.model_version,
            }

        return forecasts

    def evaluate_walk_forward_all(
        self,
        df: pd.DataFrame,
        min_train_sessions: int = 120,
    ) -> Dict[str, Dict[str, Any]]:
        """Perform separate walk-forward evaluations for each horizon and compare to baseline."""
        results = {}

        for h_key, h_sessions in self.horizons_map.items():
            feature_df, feat_cols = build_feature_matrix(
                df, include_target=True, horizon_sessions=h_sessions
            )
            valid_df = feature_df.dropna(subset=["target_price"]).copy().reset_index(drop=True)

            if len(valid_df) <= min_train_sessions:
                raise ValueError(
                    f"Dataset ({len(valid_df)}) too small for min_train_sessions ({min_train_sessions}) on {h_key}."
                )

            actual_origins = []
            actual_targets = []
            model_preds = []
            baseline_preds = []

            for t in range(min_train_sessions, len(valid_df)):
                train_part = valid_df.iloc[:t]
                test_r = valid_df.iloc[t]

                forecaster = self._create_forecaster(horizon_key=h_key)
                forecaster.fit(
                    X=train_part[feat_cols],
                    y_target_price=train_part["target_price"],
                    p_origin=train_part[PRICE_COLUMN],
                    fit_quantiles=False,
                )

                p_orig = float(test_r[PRICE_COLUMN])
                p_actual = float(test_r["target_price"])
                p_pred = float(
                    forecaster.predict(X=test_r[feat_cols].to_frame().T, p_origin=p_orig)[0]
                )

                actual_origins.append(p_orig)
                actual_targets.append(p_actual)
                model_preds.append(p_pred)
                baseline_preds.append(p_orig)

            m_metrics = calculate_all_metrics(actual_origins, actual_targets, model_preds)
            b_metrics = calculate_all_metrics(actual_origins, actual_targets, baseline_preds)

            results[h_key] = {
                "horizon_sessions": h_sessions,
                "walk_forward_steps": len(model_preds),
                "model_metrics": m_metrics,
                "baseline_metrics": b_metrics,
                "mae_improvement_pct": round(
                    ((b_metrics["mae"] - m_metrics["mae"]) / b_metrics["mae"]) * 100
                    if b_metrics["mae"] > 0
                    else 0.0,
                    2,
                ),
            }

        return results

    def save_models(self, dir_path: Union[str, Path], ticker: str) -> Path:
        """Serialize all direct horizon models to directory with path traversal protection and HMAC signatures."""
        from src.security.validators import assert_safe_model_path, sanitize_path_component, validate_ticker
        val_ticker = validate_ticker(ticker)
        base = Path(dir_path).resolve()
        safe_name = f"multi_horizon_{sanitize_path_component(val_ticker)}"
        path = assert_safe_model_path(base / safe_name, base)
        path.mkdir(parents=True, exist_ok=True)
        meta_file = path / "metadata.joblib"
        joblib.dump(
            {
                "model_family": self.model_family,
                "model_version": self.model_version,
                "alpha": self.alpha,
                "model_kwargs": self.model_kwargs,
            },
            meta_file,
        )
        sign_model_artifact(meta_file)
        for h_key, model in self.models.items():
            model.save(path / f"{h_key}.joblib")
        return path

    def get_feature_importances(self, horizon_key: str) -> Dict[str, float]:
        """Return quantitative feature importances or coefficients for the given horizon."""
        if horizon_key not in self.models:
            raise ValueError(f"Invalid horizon key '{horizon_key}'")
        model = self.models[horizon_key]
        if hasattr(model, "get_feature_importances"):
            return model.get_feature_importances()
        if hasattr(model, "get_coefficients"):
            return model.get_coefficients()
        return {}

    walk_forward_validate = evaluate_walk_forward_all

    @classmethod
    def load_models(cls, dir_path: Union[str, Path], ticker: str) -> "MultiHorizonForecaster":
        """Load all direct horizon models from directory with path traversal and signature verification."""
        from src.security.validators import (
            SecurityValidationError,
            assert_safe_model_path,
            sanitize_path_component,
            validate_ticker,
        )
        val_ticker = validate_ticker(ticker)
        base = Path(dir_path).resolve()
        safe_name = f"multi_horizon_{sanitize_path_component(val_ticker)}"
        path = assert_safe_model_path(base / safe_name, base)

        meta_file = path / "metadata.joblib"
        if meta_file.exists():
            sig_file = meta_file.with_name(f"{meta_file.name}.sig")
            if sig_file.exists() and not verify_model_artifact(meta_file):
                raise SecurityValidationError(f"Cryptographic signature check failed for {meta_file}")
            meta = joblib.load(meta_file)
            model_family = meta.get("model_family", "linear")
            model_version = meta.get("model_version", "1.0.0")
            alpha = meta.get("alpha", 1.0)
            model_kwargs = meta.get("model_kwargs", {})
            instance = cls(
                model_family=model_family,
                alpha=alpha,
                model_version=model_version,
                **model_kwargs,
            )
        else:
            instance = cls(model_family="linear")

        if instance.model_family == "gradient_boosting":
            forecaster_cls = GradientBoostingForecaster
        elif instance.model_family == "ensemble":
            forecaster_cls = ConvexEnsembleForecaster
        else:
            forecaster_cls = LinearRegressionForecaster

        for h_key in instance.horizons_map.keys():
            instance.models[h_key] = forecaster_cls.load(path / f"{h_key}.joblib")
        instance.is_fitted = True
        return instance
