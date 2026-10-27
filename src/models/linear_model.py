"""Linear regression forecasting model for NextOmni Stock.

Implements an interpretable Ridge/Linear Regression model with StandardScaler
normalization fitted strictly on training data to prevent lookahead leakage.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import PRICE_COLUMN


class LinearRegressionForecaster:
    """Interpretable Linear Regression / Ridge model for stock price forecasting.

    Predicts expected future return r_{t+k} = (P_{t+k} - P_t) / P_t from normalized features,
    and converts it into the target price prediction:
        P_pred = P_t * (1 + r_pred)
    This formulation prevents non-stationarity bias and ensures scale invariance across exchanges.
    """

    def __init__(self, alpha: float = 1.0, model_version: str = "1.0.0"):
        self.model_name = "LinearRegression_Ridge"
        self.model_version = model_version
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=42)
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y_target_price: pd.Series,
        p_origin: pd.Series,
    ) -> "LinearRegressionForecaster":
        """Fit the StandardScaler and Ridge regression strictly on training data.

        Args:
            X: Training feature matrix.
            y_target_price: Actual observed future price P_{t+k}.
            p_origin: Origin price P_t.
        """
        self.feature_names = list(X.columns)

        # Target variable: fractional return from t to t+k
        y_return = (y_target_price.values - p_origin.values) / p_origin.values

        # 1. Fit scaler strictly on training features
        X_scaled = self.scaler.fit_transform(X)

        # 2. Fit Ridge regression on scaled features and target return
        self.model.fit(X_scaled, y_return)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        p_origin: Union[pd.Series, np.ndarray, float],
    ) -> np.ndarray:
        """Generate out-of-sample predicted prices P_{t+k}.

        Applies the previously fitted scaler without re-fitting.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        # Transform using training scaler (NO DATA LEAKAGE)
        X_scaled = self.scaler.transform(X[self.feature_names])

        # Predict future fractional return r_pred
        pred_return = self.model.predict(X_scaled)

        p_orig = np.asarray(p_origin, dtype=float)
        # Convert predicted return to price level: P_pred = P_t * (1 + r_pred)
        pred_price = p_orig * (1.0 + pred_return)
        return pred_price

    def get_coefficients(self) -> Dict[str, float]:
        """Return regression coefficients beta_j mapped to feature names.

        Allows quantitative analysts to inspect feature importance and directional influence.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to inspect coefficients.")

        coef_dict = {
            name: round(float(coef), 6)
            for name, coef in zip(self.feature_names, self.model.coef_)
        }
        coef_dict["intercept"] = round(float(self.model.intercept_), 6)
        return coef_dict

    def save(self, file_path: Union[str, Path]) -> Path:
        """Persist trained model, scaler, and metadata to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "alpha": self.alpha,
                "feature_names": self.feature_names,
                "scaler": self.scaler,
                "model": self.model,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "LinearRegressionForecaster":
        """Load persisted model artifact from disk."""
        path = Path(file_path)
        data = joblib.load(path)
        instance = cls(alpha=data["alpha"], model_version=data["model_version"])
        instance.model_name = data["model_name"]
        instance.feature_names = data["feature_names"]
        instance.scaler = data["scaler"]
        instance.model = data["model"]
        instance.is_fitted = True
        return instance
