"""Quantitative conformal prediction calibration and Winkler sharpness scoring for NextOmni Stock.

Implements non-parametric split-conformal prediction intervals with finite-sample coverage guarantees,
Winkler score sharpness evaluation, and historical interval calibration diagnostics.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class CalibrationReceipt:
    """Diagnostic receipt summarizing empirical interval coverage and sharpness."""
    ticker: Optional[str]
    horizon_type: Optional[str]
    sample_size: int
    nominal_confidence_level: float
    empirical_coverage_pct: float
    coverage_error_pct: float
    mean_interval_width: float
    mean_winkler_score: float
    under_breach_pct: float
    over_breach_pct: float
    is_well_calibrated: bool
    calibration_status: str
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_winkler_score(
    lower: Union[float, np.ndarray, pd.Series],
    upper: Union[float, np.ndarray, pd.Series],
    actual: Union[float, np.ndarray, pd.Series],
    alpha: float = 0.20,
) -> Union[float, np.ndarray]:
    """Calculate the Winkler score penalizing both interval width and coverage breaches.

    Formulation:
    W = (U - L) + (2 / alpha) * (L - y) * 1_{y < L} + (2 / alpha) * (y - U) * 1_{y > U}

    Properties:
    - If y in [L, U]: Score is interval width (U - L). Narrower intervals achieve lower (better) scores.
    - If y < L: Penalized by (2 / alpha) * distance below lower bound.
    - If y > U: Penalized by (2 / alpha) * distance above upper bound.

    Args:
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.
        actual: Realized settlement price or return.
        alpha: Nominal significance level (alpha = 1 - confidence_level, default: 0.20 for 80% CI).

    Returns:
        Winkler score(s) as float or array matching input shape.
    """
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"Alpha must be strictly in (0, 1), got {alpha}")

    is_scalar = np.isscalar(lower) and np.isscalar(upper) and np.isscalar(actual)

    l_arr = np.asarray(lower, dtype=float)
    u_arr = np.asarray(upper, dtype=float)
    y_arr = np.asarray(actual, dtype=float)

    width = u_arr - l_arr
    penalty_multiplier = 2.0 / alpha

    lower_breach = np.maximum(0.0, l_arr - y_arr)
    upper_breach = np.maximum(0.0, y_arr - u_arr)

    score = width + penalty_multiplier * (lower_breach + upper_breach)

    if is_scalar:
        return float(score.item())
    return score


def compute_conformal_quantile(
    residuals: np.ndarray,
    confidence_level: float = 0.80,
) -> float:
    """Compute non-parametric conformal quantile radius with finite-sample coverage guarantee.

    For nominal coverage 1 - alpha, the conformal quantile of absolute residuals is:
        q_hat = Quantile(s, ceil((n + 1) * (1 - alpha)) / n)

    Args:
        residuals: 1D array of out-of-fold or train-fold forecast return residuals (y - y_hat).
        confidence_level: Nominal confidence level (e.g. 0.80 for 80% coverage).

    Returns:
        Conformal margin radius q_hat.
    """
    if len(residuals) == 0:
        return 0.05

    alpha = max(0.01, min(0.99, 1.0 - confidence_level))
    abs_res = np.abs(np.asarray(residuals, dtype=float))
    abs_res = abs_res[np.isfinite(abs_res)]
    n = len(abs_res)

    if n < 5:
        # Fall back to sample standard deviation * 1.28
        std_val = float(np.std(abs_res, ddof=1)) if n > 1 else 0.05
        return max(std_val * 1.2816, 0.01)

    # Conformal finite-sample rank: ceil((n + 1) * (1 - alpha)) / n
    target_rank = math.ceil((n + 1) * (1.0 - alpha))
    clamped_rank = min(n, max(1, target_rank))
    conformal_level = clamped_rank / n

    try:
        q_hat = float(np.quantile(abs_res, conformal_level, method="higher"))
    except Exception:
        q_hat = float(np.quantile(abs_res, min(1.0, conformal_level)))

    return max(q_hat, 0.001)


def compute_conformal_prediction_interval(
    origin_price: float,
    expected_return: float,
    train_residuals: np.ndarray,
    confidence_level: float = 0.80,
) -> Tuple[float, float, float]:
    """Calculate non-parametric conformal price interval [P_lower, P_upper] and conformal radius.

    Args:
        origin_price: Current reference price P_0.
        expected_return: Point expected percentage return hat{r}.
        train_residuals: Historical calibration residuals (r_actual - hat{r}).
        confidence_level: Nominal coverage target (default: 0.80).

    Returns:
        Tuple of (lower_price, upper_price, conformal_margin).
    """
    q_hat = compute_conformal_quantile(train_residuals, confidence_level=confidence_level)
    ret_lower = expected_return - q_hat
    ret_upper = expected_return + q_hat

    lower_price = max(0.01, origin_price * (1.0 + ret_lower))
    upper_price = max(lower_price + 0.01, origin_price * (1.0 + ret_upper))

    return lower_price, upper_price, q_hat


def evaluate_interval_calibration(
    predictions_df: pd.DataFrame,
    confidence_level: float = 0.80,
) -> CalibrationReceipt:
    """Audit empirical coverage and sharpness across evaluated predictions.

    Args:
        predictions_df: DataFrame containing at least ['lower_bound', 'upper_bound', 'actual_price']
                        and optionally ['ticker', 'horizon_type'].
        confidence_level: Nominal target coverage (default: 0.80).

    Returns:
        CalibrationReceipt with empirical coverage, breach rates, and mean Winkler score.
    """
    ticker = str(predictions_df["ticker"].iloc[0]) if "ticker" in predictions_df.columns and not predictions_df.empty else None
    horizon = str(predictions_df["horizon_type"].iloc[0]) if "horizon_type" in predictions_df.columns and not predictions_df.empty else None

    # Filter rows with valid bounds and actual settlement price
    valid_mask = (
        predictions_df["lower_bound"].notna()
        & predictions_df["upper_bound"].notna()
        & predictions_df["actual_price"].notna()
        & (predictions_df["lower_bound"] > 0)
        & (predictions_df["upper_bound"] >= predictions_df["lower_bound"])
        & (predictions_df["actual_price"] > 0)
    )
    clean_df = predictions_df[valid_mask].copy()
    n = len(clean_df)

    alpha = 1.0 - confidence_level

    if n < 5:
        return CalibrationReceipt(
            ticker=ticker,
            horizon_type=horizon,
            sample_size=n,
            nominal_confidence_level=confidence_level,
            empirical_coverage_pct=0.0,
            coverage_error_pct=round(confidence_level * 100.0, 2),
            mean_interval_width=0.0,
            mean_winkler_score=0.0,
            under_breach_pct=0.0,
            over_breach_pct=0.0,
            is_well_calibrated=False,
            calibration_status="INSUFFICIENT_DATA",
            diagnostics={"warning": f"Insufficient evaluated samples ({n}/5 required) for calibration audit."},
        )

    l_vals = clean_df["lower_bound"].astype(float).values
    u_vals = clean_df["upper_bound"].astype(float).values
    y_vals = clean_df["actual_price"].astype(float).values

    # Coverage indicators
    contained = (y_vals >= l_vals) & (y_vals <= u_vals)
    under_breach = y_vals < l_vals
    over_breach = y_vals > u_vals

    coverage_pct = float(np.mean(contained) * 100.0)
    coverage_error = abs(coverage_pct - (confidence_level * 100.0))
    under_pct = float(np.mean(under_breach) * 100.0)
    over_pct = float(np.mean(over_breach) * 100.0)

    # Interval width and Winkler sharpness
    widths = u_vals - l_vals
    mean_width = float(np.mean(widths))
    winkler_scores = calculate_winkler_score(l_vals, u_vals, y_vals, alpha=alpha)
    mean_winkler = float(np.mean(winkler_scores))

    # Well-calibrated if empirical coverage within 10 percentage points of target
    is_well_calibrated = coverage_error <= 10.0

    if is_well_calibrated:
        calibration_status = "WELL_CALIBRATED"
    elif coverage_pct < (confidence_level * 100.0 - 10.0):
        calibration_status = "UNDER_COVERAGE"  # Intervals too narrow / fat tails missed
    else:
        calibration_status = "OVER_COVERAGE"   # Intervals excessively wide

    diagnostics = {
        "nominal_coverage_pct": round(confidence_level * 100.0, 1),
        "empirical_coverage_pct": round(coverage_pct, 2),
        "coverage_gap_pct": round(coverage_pct - (confidence_level * 100.0), 2),
        "winkler_score_std": round(float(np.std(winkler_scores)), 3),
        "min_winkler": round(float(np.min(winkler_scores)), 3),
        "max_winkler": round(float(np.max(winkler_scores)), 3),
    }

    return CalibrationReceipt(
        ticker=ticker,
        horizon_type=horizon,
        sample_size=n,
        nominal_confidence_level=confidence_level,
        empirical_coverage_pct=round(coverage_pct, 2),
        coverage_error_pct=round(coverage_error, 2),
        mean_interval_width=round(mean_width, 2),
        mean_winkler_score=round(mean_winkler, 2),
        under_breach_pct=round(under_pct, 2),
        over_breach_pct=round(over_pct, 2),
        is_well_calibrated=is_well_calibrated,
        calibration_status=calibration_status,
        diagnostics=diagnostics,
    )
