"""Quantitative tail-risk analytics for NextOmni Stock.

Implements Value-at-Risk (Parametric Gaussian, Cornish-Fisher expansion, and Empirical Historical)
and Conditional Value-at-Risk (Expected Shortfall / CVaR) across multi-horizon forecasts.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import PRICE_COLUMN


def _clean_returns_series(returns: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """Extract clean, non-null, finite numpy array of returns."""
    arr = np.asarray(returns, dtype=float)
    clean = arr[np.isfinite(arr)]
    return clean


def calculate_var_gaussian(
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate parametric Gaussian Value-at-Risk (VaR).

    Expresses maximum expected percentage loss over holding period at the specified confidence.

    Args:
        returns: Array of historical returns.
        confidence_level: Risk confidence (e.g., 0.95 for 95% VaR).

    Returns:
        VaR expressed as a positive percentage (e.g. 0.025 for 2.5% loss).
    """
    clean = _clean_returns_series(returns)
    if len(clean) < 5:
        return 0.0

    mu = float(np.mean(clean))
    sigma = float(np.std(clean, ddof=1))
    if sigma <= 1e-9:
        return 0.0

    z = float(stats.norm.ppf(1.0 - confidence_level))
    var = -(mu + z * sigma)
    return round(float(max(var, 0.0)), 6)


def calculate_var_cornish_fisher(
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate Cornish-Fisher expansion VaR, adjusting for skewness and fat-tailed excess kurtosis.

    Accounts for empirical financial asset return asymmetry and tail risk.

    Args:
        returns: Array of historical returns.
        confidence_level: Risk confidence (e.g., 0.95 or 0.99).

    Returns:
        VaR expressed as a positive percentage loss.
    """
    clean = _clean_returns_series(returns)
    if len(clean) < 10:
        return calculate_var_gaussian(clean, confidence_level)

    mu = float(np.mean(clean))
    sigma = float(np.std(clean, ddof=1))
    if sigma <= 1e-9:
        return 0.0

    s = float(stats.skew(clean))
    k = float(stats.kurtosis(clean))  # Fisher excess kurtosis (normal = 0)

    z = float(stats.norm.ppf(1.0 - confidence_level))
    # Cornish-Fisher quantile polynomial
    z_cf = (
        z
        + (s / 6.0) * (z**2 - 1.0)
        + (k / 24.0) * (z**3 - 3.0 * z)
        - ((s**2) / 36.0) * (2.0 * z**3 - 5.0 * z)
    )

    var = -(mu + z_cf * sigma)
    return round(float(max(var, 0.0)), 6)


def calculate_var_historical(
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate empirical Historical Value-at-Risk from the sample quantile.

    Args:
        returns: Array of historical returns.
        confidence_level: Risk confidence level.

    Returns:
        Historical VaR expressed as a positive percentage loss.
    """
    clean = _clean_returns_series(returns)
    if len(clean) < 5:
        return 0.0

    alpha_pct = (1.0 - confidence_level) * 100.0
    cutoff = float(np.percentile(clean, alpha_pct))
    var = -cutoff
    return round(float(max(var, 0.0)), 6)


def calculate_cvar_historical(
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate Conditional Value-at-Risk (Expected Shortfall / CVaR).

    Measures the expected loss given that the loss exceeds the Historical VaR threshold.

    Args:
        returns: Array of historical returns.
        confidence_level: Risk confidence level.

    Returns:
        CVaR expressed as a positive percentage loss.
    """
    clean = _clean_returns_series(returns)
    if len(clean) < 5:
        return 0.0

    alpha_pct = (1.0 - confidence_level) * 100.0
    cutoff = float(np.percentile(clean, alpha_pct))
    tail_losses = clean[clean <= cutoff]

    if len(tail_losses) == 0:
        return calculate_var_historical(clean, confidence_level)

    cvar = -float(np.mean(tail_losses))
    return round(float(max(cvar, 0.0)), 6)


def compute_multi_horizon_risk_profile(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive tail risk metrics across 1, 5, and 21 session horizons.

    Args:
        df: Historical market observations with adj_close prices.
        horizons: List of horizon session integers (defaults to [1, 5, 21]).

    Returns:
        Nested dict mapping horizon key (e.g. '1_SESSION') to risk metrics.
    """
    if horizons is None:
        horizons = [1, 5, 21]

    if df.empty or PRICE_COLUMN not in df.columns:
        return {}

    prices = df[PRICE_COLUMN].dropna().values
    if len(prices) < 15:
        return {}

    daily_returns = np.diff(prices) / prices[:-1]
    results: Dict[str, Dict[str, float]] = {}

    horizon_names = {1: "1_SESSION", 5: "5_SESSIONS", 21: "21_SESSIONS"}

    for h in horizons:
        h_key = horizon_names.get(h, f"{h}_SESSIONS")
        # Empirical k-step forward returns
        if len(prices) > h:
            h_returns = (prices[h:] - prices[:-h]) / prices[:-h]
        else:
            # Fallback square root scaling of daily returns: r_h ~ r_1 * sqrt(h)
            h_returns = daily_returns * np.sqrt(h)

        var_95_g = calculate_var_gaussian(h_returns, 0.95)
        var_95_cf = calculate_var_cornish_fisher(h_returns, 0.95)
        var_99_cf = calculate_var_cornish_fisher(h_returns, 0.99)
        var_95_hist = calculate_var_historical(h_returns, 0.95)
        cvar_95_hist = calculate_cvar_historical(h_returns, 0.95)
        cvar_99_hist = calculate_cvar_historical(h_returns, 0.99)

        results[h_key] = {
            "horizon_sessions": float(h),
            "var_95_gaussian_pct": round(var_95_g * 100.0, 3),
            "var_95_cornish_fisher_pct": round(var_95_cf * 100.0, 3),
            "var_99_cornish_fisher_pct": round(var_99_cf * 100.0, 3),
            "var_95_historical_pct": round(var_95_hist * 100.0, 3),
            "cvar_95_historical_pct": round(cvar_95_hist * 100.0, 3),
            "cvar_99_historical_pct": round(cvar_99_hist * 100.0, 3),
            "skewness": round(float(stats.skew(h_returns)), 3) if len(h_returns) > 3 else 0.0,
            "excess_kurtosis": round(float(stats.kurtosis(h_returns)), 3) if len(h_returns) > 3 else 0.0,
        }

    return results
