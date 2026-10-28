"""Probabilistic forecasting evaluation and Proper Scoring Rules engine for NextOmni Stock.

Implements benchmark proper scoring rules for continuous and quantile density forecasts:
1. Continuous Ranked Probability Score (CRPS, Gneiting et al. 2005):
   Exact analytical closed-form Gaussian CRPS generalizing MAE to continuous distributions.
2. Quantile Pinball Loss (Koenker & Bassett 1978 check function):
   Direct asymmetric quantile verification across 10%, 50% (median), and 90% tails.
3. Winkler Score Sharpness & Empirical Calibration Verification.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.calibration import calculate_winkler_score


@dataclass
class ProbabilisticEvaluationReceipt:
    """Comprehensive diagnostic receipt from proper scoring rules evaluation."""
    ticker: Optional[str]
    model_family: Optional[str]
    horizon_sessions: Optional[int]
    sample_size: int
    mean_crps: float
    mean_mae: float
    mean_pinball_loss_q10: float
    mean_pinball_loss_q50: float
    mean_pinball_loss_q90: float
    mean_winkler_score_80: float
    empirical_coverage_80_pct: float
    is_distribution_superior_to_baseline: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def calculate_continuous_ranked_probability_score(
    actual: Union[float, np.ndarray, pd.Series],
    predicted_mean: Union[float, np.ndarray, pd.Series],
    predicted_std: Union[float, np.ndarray, pd.Series],
) -> float:
    """Calculate the Continuous Ranked Probability Score (CRPS) under Gaussian assumptions.

    CRPS generalizes MAE to probabilistic forecasts, rewarding both sharpness (narrow variance)
    and empirical calibration (coverage of truth):
        CRPS(y, mu, sigma) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma, Phi is standard normal CDF, and phi is standard normal PDF.

    When sigma -> 0, CRPS smoothly converges to MAE: |y - mu|.

    Args:
        actual: Realized target values (prices or returns).
        predicted_mean: Forecasted expected values (mu).
        predicted_std: Forecasted distribution standard deviations (sigma).

    Returns:
        Mean CRPS across sample observations (lower is better, in price/return units).
    """
    y = np.atleast_1d(np.asarray(actual, dtype=float))
    mu = np.atleast_1d(np.asarray(predicted_mean, dtype=float))
    sigma = np.atleast_1d(np.asarray(predicted_std, dtype=float))

    if len(y) == 0:
        return 0.0


    valid_mask = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    if not np.any(valid_mask):
        return 0.0

    y_v = y[valid_mask]
    mu_v = mu[valid_mask]
    sig_v = np.maximum(0.0, sigma[valid_mask])

    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)

    # For near-zero or zero sigma, limit is absolute error |y - mu|
    crps_vals = np.abs(y_v - mu_v)
    pos_sigma_mask = sig_v > 1e-9

    if np.any(pos_sigma_mask):
        y_pos = y_v[pos_sigma_mask]
        mu_pos = mu_v[pos_sigma_mask]
        s_pos = sig_v[pos_sigma_mask]

        z = (y_pos - mu_pos) / s_pos
        phi_z = stats.norm.pdf(z)
        Phi_z = stats.norm.cdf(z)

        crps_pos = s_pos * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - inv_sqrt_pi)
        crps_vals[pos_sigma_mask] = np.maximum(0.0, crps_pos)

    return float(np.mean(crps_vals))


def calculate_pinball_loss(
    actual: Union[float, np.ndarray, pd.Series],
    predicted_quantile: Union[float, np.ndarray, pd.Series],
    quantile: float = 0.50,
) -> float:
    """Calculate the asymmetric quantile check / pinball loss.

    Formula:
        L_q(y, q_hat) = max(q * (y - q_hat), (1 - q) * (q_hat - y))

    Properties:
    - At q = 0.50, pinball loss equals half the MAE: 0.5 * |y - q_hat|.
    - Penalizes under-prediction by factor q and over-prediction by factor (1 - q).

    Args:
        actual: Realized target values.
        predicted_quantile: Forecasted quantile level (e.g. 10th or 90th percentile).
        quantile: Target quantile level strictly in (0, 1).

    Returns:
        Mean pinball loss across valid observations.
    """
    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError(f"Quantile must be in (0, 1), got {quantile}")

    y = np.atleast_1d(np.asarray(actual, dtype=float))
    q_hat = np.atleast_1d(np.asarray(predicted_quantile, dtype=float))

    if len(y) == 0:
        return 0.0


    valid_mask = np.isfinite(y) & np.isfinite(q_hat)
    if not np.any(valid_mask):
        return 0.0

    err = y[valid_mask] - q_hat[valid_mask]
    loss = np.maximum(quantile * err, (quantile - 1.0) * err)
    return float(np.mean(loss))


def evaluate_probabilistic_forecasts(
    predictions_df: pd.DataFrame,
    horizon_sessions: Optional[int] = None,
    model_family: Optional[str] = None,
    nominal_confidence: float = 0.80,
) -> ProbabilisticEvaluationReceipt:
    """Audit and evaluate probabilistic forecasts across reconciled predictions.

    Args:
        predictions_df: DataFrame containing reconciled prediction records with columns:
                        ['actual_target_price', 'predicted_target_price',
                         'prediction_interval_lower', 'prediction_interval_upper']
        horizon_sessions: Optional horizon filter (1, 5, 21).
        model_family: Optional model family filter ('ensemble', 'gradient_boosting', 'linear').
        nominal_confidence: Nominal confidence level (default 0.80 for 80% CI).

    Returns:
        ProbabilisticEvaluationReceipt summarizing CRPS, Pinball Losses, and Winkler Sharpness.
    """
    if predictions_df.empty:
        return ProbabilisticEvaluationReceipt(
            ticker=None,
            model_family=model_family,
            horizon_sessions=horizon_sessions,
            sample_size=0,
            mean_crps=0.0,
            mean_mae=0.0,
            mean_pinball_loss_q10=0.0,
            mean_pinball_loss_q50=0.0,
            mean_pinball_loss_q90=0.0,
            mean_winkler_score_80=0.0,
            empirical_coverage_80_pct=0.0,
            is_distribution_superior_to_baseline=False,
            summary="No predictions available for probabilistic evaluation.",
        )

    df = predictions_df.copy()

    # Apply horizon filter if requested
    if horizon_sessions is not None:
        h_col = "horizon_sessions" if "horizon_sessions" in df.columns else "horizon_step"
        if h_col in df.columns:
            df = df[df[h_col] == horizon_sessions]

    # Apply model family filter if requested
    if model_family is not None and "model_family" in df.columns:
        df = df[df["model_family"] == model_family]

    # Filter to evaluated records with valid actual target price
    target_col = "actual_target_price"
    pred_col = "predicted_target_price"
    lower_col = "prediction_interval_lower"
    upper_col = "prediction_interval_upper"

    required = [target_col, pred_col, lower_col, upper_col]
    for col in required:
        if col not in df.columns:
            return ProbabilisticEvaluationReceipt(
                ticker=None,
                model_family=model_family,
                horizon_sessions=horizon_sessions,
                sample_size=0,
                mean_crps=0.0,
                mean_mae=0.0,
                mean_pinball_loss_q10=0.0,
                mean_pinball_loss_q50=0.0,
                mean_pinball_loss_q90=0.0,
                mean_winkler_score_80=0.0,
                empirical_coverage_80_pct=0.0,
                is_distribution_superior_to_baseline=False,
                summary=f"Missing required prediction column '{col}'.",
            )

    valid_df = df.dropna(subset=required)
    valid_df = valid_df[
        (valid_df[target_col] > 0)
        & (valid_df[pred_col] > 0)
        & (valid_df[lower_col] > 0)
        & (valid_df[upper_col] > 0)
    ]

    n_samples = len(valid_df)
    if n_samples < 3:
        return ProbabilisticEvaluationReceipt(
            ticker=str(valid_df["ticker"].iloc[0]) if "ticker" in valid_df.columns and n_samples > 0 else None,
            model_family=model_family,
            horizon_sessions=horizon_sessions,
            sample_size=n_samples,
            mean_crps=0.0,
            mean_mae=0.0,
            mean_pinball_loss_q10=0.0,
            mean_pinball_loss_q50=0.0,
            mean_pinball_loss_q90=0.0,
            mean_winkler_score_80=0.0,
            empirical_coverage_80_pct=0.0,
            is_distribution_superior_to_baseline=False,
            summary=f"Insufficient evaluated predictions (sample_size={n_samples} < 3).",
        )

    actuals = valid_df[target_col].values
    preds = valid_df[pred_col].values
    lowers = valid_df[lower_col].values
    uppers = valid_df[upper_col].values

    # Approximate implied Gaussian sigma from 80% CI: (Upper - Lower) / (2 * z_0.90)
    alpha = 1.0 - nominal_confidence
    z_score = float(stats.norm.ppf(1.0 - alpha / 2.0))
    implied_sigma = np.maximum(1e-6, (uppers - lowers) / (2.0 * z_score))

    mean_crps = calculate_continuous_ranked_probability_score(actuals, preds, implied_sigma)
    mean_mae = float(np.mean(np.abs(actuals - preds)))
    pinball_10 = calculate_pinball_loss(actuals, lowers, quantile=0.10)
    pinball_50 = calculate_pinball_loss(actuals, preds, quantile=0.50)
    pinball_90 = calculate_pinball_loss(actuals, uppers, quantile=0.90)

    # Winkler score
    winkler_vals = calculate_winkler_score(lowers, uppers, actuals, alpha=alpha)
    mean_winkler = float(np.mean(winkler_vals))

    # Empirical Coverage %
    within_bounds = (actuals >= lowers) & (actuals <= uppers)
    coverage_pct = float(np.mean(within_bounds) * 100.0)

    # Model is superior if CRPS <= MAE (distribution uncertainty adds predictive value)
    is_superior = mean_crps <= (mean_mae * 1.05) and coverage_pct >= (nominal_confidence * 100.0 - 15.0)

    ticker_name = str(valid_df["ticker"].iloc[0]) if "ticker" in valid_df.columns else None

    summary = (
        f"Evaluated {n_samples} reconciled forecasts. Mean CRPS: {mean_crps:.4f} (vs MAE: {mean_mae:.4f}). "
        f"Empirical Coverage: {coverage_pct:.1f}% (target: {nominal_confidence * 100:.0f}%). "
        f"Winkler Sharpness: {mean_winkler:.4f}."
    )

    return ProbabilisticEvaluationReceipt(
        ticker=ticker_name,
        model_family=model_family,
        horizon_sessions=horizon_sessions,
        sample_size=n_samples,
        mean_crps=round(mean_crps, 4),
        mean_mae=round(mean_mae, 4),
        mean_pinball_loss_q10=round(pinball_10, 4),
        mean_pinball_loss_q50=round(pinball_50, 4),
        mean_pinball_loss_q90=round(pinball_90, 4),
        mean_winkler_score_80=round(mean_winkler, 4),
        empirical_coverage_80_pct=round(coverage_pct, 2),
        is_distribution_superior_to_baseline=is_superior,
        summary=summary,
    )
