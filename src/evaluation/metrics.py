"""Evaluation metrics for NextOmni Stock forecasting models.

Implements MAE, RMSE, MAPE (with non-zero denominator protection),
and Directional Accuracy (DA).
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def calculate_mae(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
) -> float:
    """Mean Absolute Error (MAE): Average magnitude of errors in price units."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
) -> float:
    """Root Mean Squared Error (RMSE): Square root of mean squared error (penalizes outliers)."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mape(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error (MAPE) in percent (0 - 100%).

    Guarded against zero or near-zero denominators using epsilon.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if len(y_true) == 0:
        return 0.0
    denominator = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def calculate_directional_accuracy(
    actual_origin: Union[np.ndarray, pd.Series],
    actual_target: Union[np.ndarray, pd.Series],
    predicted_target: Union[np.ndarray, pd.Series],
) -> float:
    """Directional Accuracy (DA) in percent (0 - 100%).

    Measures the percentage of times the model correctly predicts whether the price
    moves UP or DOWN relative to the price at forecast origin (P_t):
    Actual Direction: sign(P_{t+k} - P_t)
    Predicted Direction: sign(P_pred - P_t)
    """
    p_origin = np.asarray(actual_origin, dtype=float)
    p_target = np.asarray(actual_target, dtype=float)
    p_pred = np.asarray(predicted_target, dtype=float)

    if len(p_origin) == 0:
        return 0.0

    actual_dir = np.sign(p_target - p_origin)
    pred_dir = np.sign(p_pred - p_origin)

    # Correct if signs match
    correct = (actual_dir == pred_dir)
    return float(np.mean(correct) * 100.0)


def calculate_all_metrics(
    actual_origin: Union[np.ndarray, pd.Series],
    actual_target: Union[np.ndarray, pd.Series],
    predicted_target: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Calculate all standard evaluation metrics in a single dictionary."""
    return {
        "mae": round(calculate_mae(actual_target, predicted_target), 4),
        "rmse": round(calculate_rmse(actual_target, predicted_target), 4),
        "mape_pct": round(calculate_mape(actual_target, predicted_target), 2),
        "directional_accuracy_pct": round(
            calculate_directional_accuracy(actual_origin, actual_target, predicted_target),
            2,
        ),
    }
