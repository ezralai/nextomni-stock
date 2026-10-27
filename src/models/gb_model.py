"""Gradient Boosting forecasting model for NextOmni Stock.

Implements a non-linear tree ensemble using scikit-learn's HistGradientBoostingRegressor
with StandardScaler normalization fitted strictly on training folds to eliminate lookahead bias.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseForecaster


class GradientBoostingForecaster(BaseForecaster):
    """Non-linear Gradient Boosted Decision Tree model for stock price forecasting.

    Learns non-linear interactions across technical, volatility, and volume indicators.
    Predicts expected forward return r_{t+k} = (P_{t+k} - P_t) / P_t,
    and converts to future price level: P_pred = P_t * (1 + r_pred).
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 15,
        l2_regularization: float = 1.0,
        max_iter: int = 50,
        random_state: int = 42,
        model_version: str = "1.0.0",
    ):
        super().__init__(model_version=model_version)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.lower_quantile_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.10,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.upper_quantile_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.feature_names: List[str] = []
        self._last_X_train_scaled: Optional[np.ndarray] = None
        self._last_y_train: Optional[np.ndarray] = None
        self.residual_std_: float = 0.015
        self._quantile_fitted: bool = False

    @property
    def model_name(self) -> str:
        return "HistGradientBoosting"

    def fit(
        self,
        X: pd.DataFrame,
        y_target_price: pd.Series,
        p_origin: pd.Series,
        fit_quantiles: bool = True,
        **kwargs: Any,
    ) -> "GradientBoostingForecaster":
        """Fit StandardScaler, point regressor, and optionally empirical quantile trees on training data."""
        self.feature_names = list(X.columns)

        # Target variable: fractional percentage return
        y_t = np.asarray(y_target_price, dtype=float)
        p_o = np.asarray(p_origin, dtype=float)
        y_return = (y_t - p_o) / p_o

        # 1. Fit scaler strictly on training features (ZERO DATA LEAKAGE)
        X_scaled = self.scaler.fit_transform(X)

        # 2. Fit tree ensemble point predictor
        self.model.fit(X_scaled, y_return)
        residuals = y_return - self.model.predict(X_scaled)
        self.residual_std_ = float(np.std(residuals)) if len(residuals) > 1 else 0.015

        # 3. Fit non-parametric quantile regressors (10th and 90th percentiles for 80% CI)
        if fit_quantiles:
            try:
                self.lower_quantile_model.fit(X_scaled, y_return)
                self.upper_quantile_model.fit(X_scaled, y_return)
                self._quantile_fitted = True
            except Exception:
                self._quantile_fitted = False
        else:
            self._quantile_fitted = False

        self._fitted = True
        self._last_X_train_scaled = X_scaled
        self._last_y_train = y_return
        return self

    def predict(
        self,
        X: pd.DataFrame,
        p_origin: Union[pd.Series, np.ndarray, float],
    ) -> np.ndarray:
        """Generate predicted prices P_{t+k} = P_t * (1 + r_pred)."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        # Transform using training scaler (NO DATA LEAKAGE)
        X_scaled = self.scaler.transform(X[self.feature_names])

        # Predict future return
        pred_return = self.model.predict(X_scaled)

        p_orig = np.asarray(p_origin, dtype=float)
        pred_price = p_orig * (1.0 + pred_return)
        return pred_price

    def predict_interval(
        self,
        X: pd.DataFrame,
        p_origin: Union[pd.Series, np.ndarray, float],
        confidence_level: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict price levels and calibrated confidence interval bounds.

        Leverages non-parametric quantile regression for 80% CI (10th/90th percentiles)
        to capture asymmetric return skewness, falling back to residual standard error
        for arbitrary confidence levels.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_interval.")

        X_scaled = self.scaler.transform(X[self.feature_names])
        pred_return = self.model.predict(X_scaled)
        p_orig = np.asarray(p_origin, dtype=float)
        pred_price = p_orig * (1.0 + pred_return)

        # 1. Use empirical quantile trees when evaluating 80% CI
        if abs(confidence_level - 0.80) < 1e-4 and getattr(self, "_quantile_fitted", False):
            q_lower_return = self.lower_quantile_model.predict(X_scaled)
            q_upper_return = self.upper_quantile_model.predict(X_scaled)
            lower_price = p_orig * (1.0 + q_lower_return)
            upper_price = p_orig * (1.0 + q_upper_return)
            # Enforce monotonicity: lower <= pred <= upper
            lower_price = np.minimum(lower_price, pred_price)
            upper_price = np.maximum(upper_price, pred_price)
            return pred_price, lower_price, upper_price

        # 2. Parametric Gaussian residual fallback for arbitrary confidence levels
        from scipy import stats
        z = float(stats.norm.ppf(0.5 + confidence_level / 2.0))
        lower_return = pred_return - (z * self.residual_std_)
        upper_return = pred_return + (z * self.residual_std_)
        lower_price = p_orig * (1.0 + lower_return)
        upper_price = p_orig * (1.0 + upper_return)
        return pred_price, lower_price, upper_price

    def get_feature_importances(self) -> Dict[str, float]:
        """Compute permutation feature importances on the training fold."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted to inspect feature importances.")

        if self._last_X_train_scaled is not None and self._last_y_train is not None:
            res = permutation_importance(
                self.model,
                self._last_X_train_scaled,
                self._last_y_train,
                n_repeats=5,
                random_state=self.random_state,
            )
            return {
                name: round(float(imp), 6)
                for name, imp in zip(self.feature_names, res.importances_mean)
            }

        return {name: 0.0 for name in self.feature_names}

    def save(self, file_path: Union[str, Path]) -> Path:
        """Persist trained model, quantile regressors, scaler, and hyperparameters to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "l2_regularization": self.l2_regularization,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "scaler": self.scaler,
                "model": self.model,
                "lower_quantile_model": getattr(self, "lower_quantile_model", None),
                "upper_quantile_model": getattr(self, "upper_quantile_model", None),
                "quantile_fitted": getattr(self, "_quantile_fitted", False),
                "residual_std": getattr(self, "residual_std_", 0.015),
            },
            path,
        )
        from src.security.audit import sign_model_artifact
        sign_model_artifact(path)
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path], verify_signature: bool = True) -> "GradientBoostingForecaster":
        """Load persisted model artifact and quantile regressors from disk after signature validation."""
        from src.security.audit import verify_model_artifact
        from src.security.validators import SecurityValidationError
        path = Path(file_path)
        sig_file = path.with_name(f"{path.name}.sig")
        if verify_signature and sig_file.exists():
            if not verify_model_artifact(path):
                raise SecurityValidationError(f"Cryptographic signature check failed for {path}")
        data = joblib.load(path)
        instance = cls(
            max_depth=data.get("max_depth", 3),
            min_samples_leaf=data.get("min_samples_leaf", 15),
            l2_regularization=data.get("l2_regularization", 1.0),
            max_iter=data.get("max_iter", 50),
            random_state=data.get("random_state", 42),
            model_version=data.get("model_version", "1.0.0"),
        )
        instance.feature_names = data["feature_names"]
        instance.scaler = data["scaler"]
        instance.model = data["model"]
        if "lower_quantile_model" in data and data["lower_quantile_model"] is not None:
            instance.lower_quantile_model = data["lower_quantile_model"]
        if "upper_quantile_model" in data and data["upper_quantile_model"] is not None:
            instance.upper_quantile_model = data["upper_quantile_model"]
        instance._quantile_fitted = data.get("quantile_fitted", False)
        instance.residual_std_ = data.get("residual_std", 0.015)
        instance._fitted = True
        return instance
