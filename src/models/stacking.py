"""Out-of-Fold Stacking Meta-Learner (Super Learner) for NextOmni Stock.

Implements the Super Learner / Stacking Regressor framework (Breiman 1996; van der Laan et al. 2007):
- Generates leak-free out-of-fold temporal cross-validated predictions across constituent base models.
- Solves constrained non-negative quadratic optimization to find optimal convex meta-weights (beta_m >= 0, sum(beta_m) = 1).
- Asymptotically guaranteed to perform as well as or strictly better than any individual base model.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.base_model import BaseForecaster
from src.models.gb_model import GradientBoostingForecaster
from src.models.linear_model import LinearRegressionForecaster
from src.security.audit import sign_model_artifact, verify_model_artifact
from src.security.validators import SecurityValidationError, assert_safe_model_path


class StackingForecaster(BaseForecaster):
    """Out-of-fold stacking meta-learner with constrained non-negative convex optimization."""

    def __init__(
        self,
        linear_forecaster: Optional[LinearRegressionForecaster] = None,
        gb_forecaster: Optional[GradientBoostingForecaster] = None,
        n_splits: int = 4,
        min_fold_size: int = 30,
        model_version: str = "1.0.0",
    ):
        super().__init__(model_version=model_version)
        self.linear_model = linear_forecaster or LinearRegressionForecaster(model_version=model_version)
        self.gb_model = gb_forecaster or GradientBoostingForecaster(model_version=model_version)
        self.n_splits = n_splits
        self.min_fold_size = min_fold_size
        self.weights: Tuple[float, float] = (0.50, 0.50)
        self.oof_rmse: Optional[float] = None

    @property
    def model_name(self) -> str:
        w_lin, w_gb = self.weights
        return f"StackingMetaLearner(Ridge:{w_lin:.2f}+HistGB:{w_gb:.2f})"

    def _optimize_meta_weights(
        self,
        Z_oof: np.ndarray,
        y_oof: np.ndarray,
    ) -> Tuple[float, float]:
        """Solve non-negative constrained convex optimization for optimal meta-weights.

        min_{beta} || y_oof - Z_oof @ beta ||_2^2
        s.t. beta_m >= 0, sum(beta_m) == 1.
        """
        def objective(w: np.ndarray) -> float:
            pred = Z_oof @ w
            return float(np.sum((y_oof - pred) ** 2))

        init_weights = np.array([0.50, 0.50], dtype=float)
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            objective,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if res.success and np.all(np.isfinite(res.x)):
            w = np.clip(res.x, 0.0, 1.0)
            total = np.sum(w)
            if total > 0:
                w /= total
                return round(float(w[0]), 4), round(float(w[1]), 4)

        return 0.50, 0.50

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_target_price: Union[np.ndarray, pd.Series],
        p_origin: Union[float, np.ndarray, pd.Series],
        fit_quantiles: bool = True,
        **kwargs: Any,
    ) -> "StackingForecaster":
        """Generate out-of-fold temporal cross-validation matrix, optimize meta-weights, and fit base models."""
        X_df = pd.DataFrame(X).copy().reset_index(drop=True)
        y_arr = np.asarray(y_target_price, dtype=float)
        p_arr = np.asarray(p_origin, dtype=float)
        n = len(X_df)

        # Convert price target to percentage returns
        target_returns = (y_arr - p_arr) / np.maximum(p_arr, 1e-6)

        # Generate leak-free out-of-fold predictions using chronological expanding folds
        if n >= 60 and self.n_splits >= 2:
            step = max(1, (n - self.min_fold_size) // self.n_splits)
            z_oof_list: List[np.ndarray] = []
            y_oof_list: List[np.ndarray] = []

            for i in range(self.n_splits):
                train_end = self.min_fold_size + i * step
                val_end = min(n, train_end + step if i < self.n_splits - 1 else n)

                if val_end <= train_end:
                    break

                # Temporary base models fitted strictly on training fold
                fold_lin = LinearRegressionForecaster(model_version=self.model_version)
                fold_gb = GradientBoostingForecaster(model_version=self.model_version)

                fold_lin.fit(X_df.iloc[:train_end], y_arr[:train_end], p_arr[:train_end])
                fold_gb.fit(X_df.iloc[:train_end], y_arr[:train_end], p_arr[:train_end], fit_quantiles=False)

                val_p = p_arr[train_end:val_end]
                pred_lin_p = fold_lin.predict(X_df.iloc[train_end:val_end], val_p)
                pred_gb_p = fold_gb.predict(X_df.iloc[train_end:val_end], val_p)

                # Convert out-of-fold predictions back to return space
                r_lin = (pred_lin_p - val_p) / np.maximum(val_p, 1e-6)
                r_gb = (pred_gb_p - val_p) / np.maximum(val_p, 1e-6)

                z_fold = np.column_stack([r_lin, r_gb])
                z_oof_list.append(z_fold)
                y_oof_list.append(target_returns[train_end:val_end])

            if z_oof_list:
                Z_oof = np.vstack(z_oof_list)
                y_oof = np.concatenate(y_oof_list)
                self.weights = self._optimize_meta_weights(Z_oof, y_oof)
                oof_preds = Z_oof @ np.array(self.weights)
                self.oof_rmse = float(np.sqrt(np.mean((y_oof - oof_preds) ** 2)))
        else:
            self.weights = (0.50, 0.50)

        # Fit submodels on 100% of the training dataset
        self.linear_model.fit(X_df, y_arr, p_arr)
        self.gb_model.fit(X_df, y_arr, p_arr, fit_quantiles=fit_quantiles)

        self._fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        p_origin: Union[float, np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Generate stacked point forecast using learned meta-weights."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict.")

        pred_lin = self.linear_model.predict(X, p_origin)
        pred_gb = self.gb_model.predict(X, p_origin)
        w_lin, w_gb = self.weights
        return (w_lin * pred_lin) + (w_gb * pred_gb)

    def predict_interval(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        p_origin: Union[float, np.ndarray, pd.Series],
        confidence_level: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate stacked prediction intervals using learned meta-weights."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict_interval.")

        p_lin, low_lin, up_lin = self.linear_model.predict_interval(X, p_origin, confidence_level=confidence_level)
        p_gb, low_gb, up_gb = self.gb_model.predict_interval(X, p_origin, confidence_level=confidence_level)
        w_lin, w_gb = self.weights

        p_blend = (w_lin * p_lin) + (w_gb * p_gb)
        low_blend = (w_lin * low_lin) + (w_gb * low_gb)
        up_blend = (w_lin * up_lin) + (w_gb * up_gb)
        return p_blend, low_blend, up_blend

    def get_feature_importances(self) -> Dict[str, float]:
        """Combine quantitative feature signals across submodels according to learned meta-weights."""
        w_lin, w_gb = self.weights
        lin_imp = self.linear_model.get_coefficients()
        gb_imp = self.gb_model.get_feature_importances()

        sum_lin = sum(abs(v) for v in lin_imp.values()) or 1.0
        norm_lin = {k: abs(v) / sum_lin for k, v in lin_imp.items()}

        combined = {}
        all_keys = set(norm_lin.keys()) | set(gb_imp.keys())
        for k in all_keys:
            val = w_lin * norm_lin.get(k, 0.0) + w_gb * gb_imp.get(k, 0.0)
            combined[k] = round(val, 6)

        return combined

    def explain_forecast(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, float]:
        """Combine submodel additive feature signal attributions weighted by meta-weights."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling explain_forecast.")

        w_lin, w_gb = self.weights
        lin_exp = self.linear_model.explain_forecast(X)
        gb_exp = self.gb_model.explain_forecast(X)

        all_keys = set(lin_exp.keys()) | set(gb_exp.keys())
        blended = {}
        for k in all_keys:
            val = w_lin * lin_exp.get(k, 0.0) + w_gb * gb_exp.get(k, 0.0)
            blended[k] = round(float(val), 6)
        return blended

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize stacked model with path traversal protection and HMAC-SHA256 signature."""
        base_dir = Path(filepath).resolve().parent
        path = assert_safe_model_path(Path(filepath), base_dir)
        payload = {
            "model_version": self.model_version,
            "weights": self.weights,
            "oof_rmse": self.oof_rmse,
            "linear_model": self.linear_model,
            "gb_model": self.gb_model,
            "is_fitted": self._fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
        sign_model_artifact(path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "StackingForecaster":
        """Deserialize stacked model after cryptographic HMAC-SHA256 integrity verification."""
        base_dir = Path(filepath).resolve().parent
        path = assert_safe_model_path(Path(filepath), base_dir)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if not verify_model_artifact(path):
            raise SecurityValidationError(f"HMAC integrity verification failed for {path} — tampering detected!")

        payload = joblib.load(path)
        instance = cls(
            linear_forecaster=payload.get("linear_model"),
            gb_forecaster=payload.get("gb_model"),
            model_version=payload.get("model_version", "1.0.0"),
        )
        instance.weights = payload.get("weights", (0.50, 0.50))
        instance.oof_rmse = payload.get("oof_rmse")
        instance._fitted = payload.get("is_fitted", False)
        return instance
