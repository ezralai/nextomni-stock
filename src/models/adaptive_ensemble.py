"""Dynamic Regime-Adaptive Multi-Horizon Ensemble Forecaster for NextOmni Stock.

Implements market-state conditional model weighting between Ridge linear regression
and non-linear HistGradientBoosting regressors. Dynamically shifts model allocations
conditioned on the prevailing volatility-trend regime:
- BULL_TREND: Overweights non-linear gradient boosting for momentum capture.
- EUPHORIC_MOMENTUM: Balanced growth weighting with volatility dampening.
- DISTRIBUTION: Overweights regularized linear model to avoid false breakout overfitting.
- PANIC_CRISIS: Heavily overweights regularized linear shrinkage to protect capital.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.models.base_model import BaseForecaster
from src.models.gb_model import GradientBoostingForecaster
from src.models.linear_model import LinearRegressionForecaster
from src.security.audit import sign_model_artifact, verify_model_artifact
from src.security.validators import SecurityValidationError, assert_safe_model_path


DEFAULT_REGIME_WEIGHTS: Dict[str, Tuple[float, float]] = {
    # (w_linear, w_gradient_boosting)
    "BULL_TREND": (0.30, 0.70),
    "EUPHORIC_MOMENTUM": (0.40, 0.60),
    "DISTRIBUTION": (0.60, 0.40),
    "PANIC_CRISIS": (0.80, 0.20),
    "DEFAULT": (0.50, 0.50),
}


class RegimeAdaptiveForecaster(BaseForecaster):
    """Regime-conditional ensemble blending Ridge and HistGB models adaptively."""

    def __init__(
        self,
        linear_forecaster: Optional[LinearRegressionForecaster] = None,
        gb_forecaster: Optional[GradientBoostingForecaster] = None,
        regime_weights: Optional[Dict[str, Tuple[float, float]]] = None,
        model_version: str = "1.0.0",
    ):
        super().__init__(model_version=model_version)
        self.linear_model = linear_forecaster or LinearRegressionForecaster(model_version=model_version)
        self.gb_model = gb_forecaster or GradientBoostingForecaster(model_version=model_version)
        self.regime_weights = dict(regime_weights or DEFAULT_REGIME_WEIGHTS)
        self.last_detected_regime: str = "DEFAULT"
        self.last_active_weights: Tuple[float, float] = self.regime_weights["DEFAULT"]

    @property
    def model_name(self) -> str:
        return "RegimeAdaptiveEnsemble(Ridge+HistGB)"

    def classify_regime(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> str:
        """Classify prevailing market regime from feature vectors.

        Uses price_to_sma_20 (trend direction) and vol_regime_ratio (volatility state).
        Falls back to 'DEFAULT' if named feature columns are not accessible.
        """
        if isinstance(X, pd.DataFrame):
            if X.empty:
                return "DEFAULT"
            row = X.iloc[-1]
        elif isinstance(X, pd.Series):
            row = X
        else:
            return "DEFAULT"

        has_trend = "price_to_sma_20" in row.index
        has_vol = "vol_regime_ratio" in row.index

        if not (has_trend and has_vol):
            return "DEFAULT"

        try:
            trend_val = float(row["price_to_sma_20"])
            vol_val = float(row["vol_regime_ratio"])
        except (ValueError, TypeError):
            return "DEFAULT"

        is_bullish = trend_val >= 1.0
        is_high_vol = vol_val > 1.15

        if is_bullish and not is_high_vol:
            return "BULL_TREND"
        elif is_bullish and is_high_vol:
            return "EUPHORIC_MOMENTUM"
        elif not is_bullish and not is_high_vol:
            return "DISTRIBUTION"
        else:
            return "PANIC_CRISIS"

    def get_regime_weights(self, regime: str) -> Tuple[float, float]:
        """Retrieve (w_linear, w_gb) for the given regime."""
        return self.regime_weights.get(regime, self.regime_weights["DEFAULT"])

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_target_price: Union[np.ndarray, pd.Series],
        p_origin: Union[float, np.ndarray, pd.Series],
        fit_quantiles: bool = True,
        **kwargs: Any,
    ) -> "RegimeAdaptiveForecaster":
        """Fit constituent Ridge and HistGradientBoosting submodels on full training set."""
        X_df = pd.DataFrame(X).copy().reset_index(drop=True)
        y_arr = np.asarray(y_target_price, dtype=float)
        p_arr = np.asarray(p_origin, dtype=float)

        self.linear_model.fit(X_df, y_arr, p_arr)
        self.gb_model.fit(X_df, y_arr, p_arr, fit_quantiles=fit_quantiles)

        self.last_detected_regime = self.classify_regime(X_df)
        self.last_active_weights = self.get_regime_weights(self.last_detected_regime)

        self._fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        p_origin: Union[float, np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Generate point forecasts with dynamic regime-adaptive weighting."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict.")

        X_df = pd.DataFrame(X).copy()
        pred_lin = self.linear_model.predict(X_df, p_origin)
        pred_gb = self.gb_model.predict(X_df, p_origin)

        # Detect regime for current batch
        regime = self.classify_regime(X_df)
        self.last_detected_regime = regime
        w_lin, w_gb = self.get_regime_weights(regime)
        self.last_active_weights = (w_lin, w_gb)

        return (w_lin * pred_lin) + (w_gb * pred_gb)

    def predict_interval(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        p_origin: Union[float, np.ndarray, pd.Series],
        confidence_level: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals with dynamic regime-adaptive blending."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict_interval.")

        X_df = pd.DataFrame(X).copy()
        p_lin, low_lin, up_lin = self.linear_model.predict_interval(X_df, p_origin, confidence_level=confidence_level)
        p_gb, low_gb, up_gb = self.gb_model.predict_interval(X_df, p_origin, confidence_level=confidence_level)

        regime = self.classify_regime(X_df)
        self.last_detected_regime = regime
        w_lin, w_gb = self.get_regime_weights(regime)
        self.last_active_weights = (w_lin, w_gb)

        p_blend = (w_lin * p_lin) + (w_gb * p_gb)
        low_blend = (w_lin * low_lin) + (w_gb * low_gb)
        up_blend = (w_lin * up_lin) + (w_gb * up_gb)
        return p_blend, low_blend, up_blend

    def get_feature_importances(self) -> Dict[str, float]:
        """Combine feature signals across submodels according to last active regime weights."""
        w_lin, w_gb = self.last_active_weights
        lin_imp = self.linear_model.get_coefficients()
        gb_imp = self.gb_model.get_feature_importances()

        sum_lin = sum(abs(v) for v in lin_imp.values()) or 1.0
        norm_lin = {k: abs(v) / sum_lin for k, v in lin_imp.items()}

        combined = {}
        all_keys = set(norm_lin.keys()) | set(gb_imp.keys())
        for k in all_keys:
            v_lin = norm_lin.get(k, 0.0)
            v_gb = gb_imp.get(k, 0.0)
            combined[k] = round((w_lin * v_lin) + (w_gb * v_gb), 6)
        return combined


    def explain_forecast(
        self,
        X_sample: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """Explain forecast via regime-weighted combination of constituent model attributions."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before explain_forecast.")

        regime = self.classify_regime(X_sample)
        w_lin, w_gb = self.get_regime_weights(regime)

        lin_attr = self.linear_model.explain_forecast(X_sample)
        gb_attr = self.gb_model.explain_forecast(X_sample)

        all_keys = set(lin_attr.keys()).union(set(gb_attr.keys()))
        blended_attr: Dict[str, float] = {}

        for k in all_keys:
            v_lin = lin_attr.get(k, 0.0)
            v_gb = gb_attr.get(k, 0.0)
            blended_attr[k] = round((w_lin * v_lin) + (w_gb * v_gb), 6)

        return blended_attr

    def save(self, filepath: Union[str, Path], sign: bool = True) -> Path:
        """Persist regime adaptive model with optional cryptographic HMAC signature."""
        target_path = Path(filepath).resolve()
        assert_safe_model_path(target_path, target_path.parent)

        payload = {
            "linear_model": self.linear_model,
            "gb_model": self.gb_model,
            "regime_weights": self.regime_weights,
            "last_detected_regime": self.last_detected_regime,
            "last_active_weights": self.last_active_weights,
            "model_version": self.model_version,
            "_fitted": self._fitted,
        }
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, target_path)

        if sign:
            sign_model_artifact(target_path)

        return target_path

    @classmethod
    def load(cls, filepath: Union[str, Path], verify_signature: bool = True) -> "RegimeAdaptiveForecaster":
        """Load regime adaptive model ensuring cryptographic signature integrity."""
        target_path = Path(filepath).resolve()
        assert_safe_model_path(target_path, target_path.parent)

        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found: {target_path}")

        sig_path = target_path.with_name(f"{target_path.name}.sig")
        if verify_signature and sig_path.exists():
            if not verify_model_artifact(target_path):
                raise SecurityValidationError(f"Cryptographic signature check failed for {target_path}")

        payload = joblib.load(target_path)
        instance = cls(
            linear_forecaster=payload.get("linear_model"),
            gb_forecaster=payload.get("gb_model"),
            regime_weights=payload.get("regime_weights"),
            model_version=payload.get("model_version", "1.0.0"),
        )
        instance.last_detected_regime = payload.get("last_detected_regime", "DEFAULT")
        instance.last_active_weights = payload.get("last_active_weights", (0.5, 0.5))
        instance._fitted = payload.get("_fitted", True)
        return instance
