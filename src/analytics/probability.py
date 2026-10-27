"""Quantitative probability calibration and directional odds estimation for NextOmni Stock.

Implements Gaussian error function calibration and logistic sigmoid transformations
to calculate directional win probabilities P(r > 0) from point forecasts and residual variance.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass
class DirectionalOdds:
    """Directional win-probability receipt."""
    probability: float
    conviction: str
    z_score: float
    expected_return_pct: float
    residual_std_pct: float


def calculate_directional_probability(
    expected_return: float,
    residual_std: float,
    min_std: float = 1e-4,
) -> Tuple[float, str]:
    """Calculate calibrated directional win probability P(r > 0) and categorical conviction.

    Formulation:
    P(r > 0) = Phi(r_hat / sigma_res) = 0.5 * (1 + erf(z / sqrt(2)))
    where z = r_hat / max(sigma_res, min_std).

    Conviction levels:
    - >= 0.65: STRONG BULLISH
    - 0.55 <= p < 0.65: LEAN BULLISH
    - 0.45 <= p < 0.55: NEUTRAL
    - 0.35 <= p < 0.45: LEAN BEARISH
    - < 0.35: STRONG BEARISH

    Args:
        expected_return: Predicted percentage return (e.g. 0.025 for +2.5%).
        residual_std: Standard deviation of out-of-sample forecast residuals.
        min_std: Minimum denominator floor to prevent division-by-zero.

    Returns:
        Tuple of (win_probability, conviction_string).
    """
    safe_std = max(float(residual_std), min_std) if math.isfinite(residual_std) and residual_std > 0 else min_std
    z = float(expected_return) / safe_std

    # Gaussian cumulative distribution function via math.erf
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    prob = max(0.0001, min(0.9999, prob))

    if prob >= 0.65:
        conviction = "STRONG BULLISH"
    elif prob >= 0.55:
        conviction = "LEAN BULLISH"
    elif prob > 0.45:
        conviction = "NEUTRAL"
    elif prob > 0.35:
        conviction = "LEAN BEARISH"
    else:
        conviction = "STRONG BEARISH"

    return round(prob, 4), conviction
