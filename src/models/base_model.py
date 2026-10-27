"""Base interface contract for NextOmni Stock forecasting models.

Defines the polymorphic contract for all predictive regressors (Linear,
Gradient Boosting, Deep Learning) to ensure consistent fit, predict, and serialization.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for quantitative financial forecasting models."""

    def __init__(self, model_version: str = "1.0.0"):
        self.model_version = model_version
        self._fitted: bool = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier name of the forecasting model family."""
        pass

    @property
    def is_fitted(self) -> bool:
        """Return True if model parameters have been estimated from training data."""
        return self._fitted

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_target_price: Union[np.ndarray, pd.Series],
        p_origin: Union[float, np.ndarray, pd.Series],
    ) -> "BaseForecaster":
        """Fit model strictly on training features and stationary return targets.

        Args:
            X: Feature matrix.
            y_target_price: Observed future price P_{t+k}.
            p_origin: Origin price P_t.
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        p_origin: Union[float, np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Predict expected future price levels P_{t+k} given current origin prices P_t.

        Returns:
            np.ndarray: Predicted absolute price levels P_t * (1 + r_{t+k}).
        """
        pass

    @abstractmethod
    def predict_interval(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        p_origin: Union[float, np.ndarray, pd.Series],
        confidence_level: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict price levels and calibrated confidence interval bounds.

        Args:
            X: Feature matrix.
            p_origin: Origin price P_t.
            confidence_level: Desired coverage probability (default: 0.80).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                (predicted_price, lower_bound_price, upper_bound_price)
        """
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize model artifact and parameters to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseForecaster":
        """Load serialized model artifact from disk."""
        pass
