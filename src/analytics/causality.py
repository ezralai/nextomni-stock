"""Cross-market lead-lag Granger causality and Engle-Granger cointegration engine for NextOmni Stock.

Implements Vector Autoregressive (VAR) Granger causality F-tests, Engle-Granger two-step
cointegration residual stationarity tests (ADF), mean-reversion half-life estimation,
and rolling spread z-score signal generation across multi-asset universes.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats


@dataclass
class GrangerCausalityReceipt:
    """Diagnostic receipt summarizing Granger causality test results."""
    source_ticker: str
    target_ticker: str
    optimal_lag: int
    f_statistic: float
    p_value: float
    is_causal: bool
    r_squared_unrestricted: float
    r_squared_restricted: float
    sample_size: int
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CointegrationReceipt:
    """Diagnostic receipt summarizing Engle-Granger cointegration and mean-reversion metrics."""
    ticker_1: str
    ticker_2: str
    hedge_ratio_beta: float
    intercept_alpha: float
    adf_statistic: float
    critical_values: Dict[str, float]
    p_value_approx: float
    is_cointegrated: bool
    half_life_sessions: float
    current_spread: float
    current_spread_zscore: float
    sample_size: int
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def align_time_series(
    series_1: pd.Series,
    series_2: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Align two time series by matching index dates, removing NaNs.

    Args:
        series_1: First asset series indexed by date or timestamp.
        series_2: Second asset series indexed by date or timestamp.

    Returns:
        Tuple of aligned (clean_1, clean_2) series.
    """
    df = pd.DataFrame({"s1": series_1, "s2": series_2}).dropna()
    return df["s1"], df["s2"]


def evaluate_granger_causality(
    source_series: pd.Series,
    target_series: pd.Series,
    max_lag: int = 5,
    source_ticker: str = "SOURCE",
    target_ticker: str = "TARGET",
) -> GrangerCausalityReceipt:
    """Perform Granger causality F-test to determine whether source leads target.

    Models:
    - Restricted (AR):   Y_t = a_0 + sum_{i=1}^p beta_i Y_{t-i} + e_{r,t}
    - Unrestricted (VAR): Y_t = a_0 + sum_{i=1}^p beta_i Y_{t-i} + sum_{i=1}^p gamma_i X_{t-i} + e_{u,t}

    Null Hypothesis (H0): gamma_1 = ... = gamma_p = 0 (X does not Granger-cause Y).
    F-statistic: ((SSR_r - SSR_u) / p) / (SSR_u / (N - 2p - 1))

    Args:
        source_series: Potential leading time series (X).
        target_series: Potential lagging time series (Y).
        max_lag: Maximum lag order p to test (default: 5).
        source_ticker: Label identifier for source.
        target_ticker: Label identifier for target.

    Returns:
        GrangerCausalityReceipt with test statistics, p-value, and causality verdict.
    """
    s1, s2 = align_time_series(source_series, target_series)
    x = s1.values.astype(float)
    y = s2.values.astype(float)
    n = len(y)

    if n < max_lag * 3 + 10:
        return GrangerCausalityReceipt(
            source_ticker=source_ticker,
            target_ticker=target_ticker,
            optimal_lag=1,
            f_statistic=0.0,
            p_value=1.0,
            is_causal=False,
            r_squared_unrestricted=0.0,
            r_squared_restricted=0.0,
            sample_size=n,
            summary=f"Insufficient sample observations ({n}) for Granger causality test.",
        )

    best_lag = 1
    best_p_val = 1.0
    best_f_stat = 0.0
    best_r2_u = 0.0
    best_r2_r = 0.0
    min_aic = float("inf")

    effective_max_lag = min(max_lag, (n - 10) // 3)

    for p in range(1, effective_max_lag + 1):
        m = n - p
        y_dep = y[p:]
        tss = float(np.sum((y_dep - np.mean(y_dep)) ** 2))
        tss = max(tss, 1e-12)

        # Build lagged design matrices
        y_lags = [y[p - i : n - i] for i in range(1, p + 1)]
        x_lags = [x[p - i : n - i] for i in range(1, p + 1)]

        x_res = np.column_stack([np.ones(m)] + y_lags)
        x_unres = np.column_stack([np.ones(m)] + y_lags + x_lags)

        # Fit restricted OLS
        b_r, _, _, _ = np.linalg.lstsq(x_res, y_dep, rcond=None)
        ssr_r = float(np.sum((y_dep - x_res @ b_r) ** 2))

        # Fit unrestricted OLS
        b_u, _, _, _ = np.linalg.lstsq(x_unres, y_dep, rcond=None)
        ssr_u = float(np.sum((y_dep - x_unres @ b_u) ** 2))

        df2 = m - (2 * p + 1)
        if df2 <= 0 or ssr_u <= 1e-12:
            continue

        f_stat = ((ssr_r - ssr_u) / p) / (ssr_u / df2)
        f_stat = max(0.0, float(f_stat))
        p_val = float(scipy.stats.f.sf(f_stat, p, df2))

        # Akaike Information Criterion for lag selection
        aic = m * math.log(max(ssr_u / m, 1e-12)) + 2.0 * (2 * p + 1)

        if aic < min_aic:
            min_aic = aic
            best_lag = p
            best_p_val = p_val
            best_f_stat = f_stat
            best_r2_u = max(0.0, 1.0 - ssr_u / tss)
            best_r2_r = max(0.0, 1.0 - ssr_r / tss)

    is_causal = best_p_val < 0.05
    summary = (
        f"{source_ticker} statistically Granger-causes {target_ticker} at {best_lag}-session lag "
        f"(F={best_f_stat:.2f}, p={best_p_val:.4f})."
        if is_causal
        else f"No statistically significant lead-lag causality found from {source_ticker} to {target_ticker} (p={best_p_val:.4f})."
    )

    return GrangerCausalityReceipt(
        source_ticker=source_ticker,
        target_ticker=target_ticker,
        optimal_lag=best_lag,
        f_statistic=round(best_f_stat, 4),
        p_value=round(best_p_val, 6),
        is_causal=is_causal,
        r_squared_unrestricted=round(best_r2_u, 4),
        r_squared_restricted=round(best_r2_r, 4),
        sample_size=n,
        summary=summary,
    )


def evaluate_engle_granger_cointegration(
    series_1: pd.Series,
    series_2: pd.Series,
    ticker_1: str = "ASSET_1",
    ticker_2: str = "ASSET_2",
) -> CointegrationReceipt:
    """Perform two-step Engle-Granger cointegration test with mean-reversion half-life.

    Step 1: OLS regression P_{1,t} = alpha + beta * P_{2,t} + eps_t
    Step 2: Augmented Dickey-Fuller test on residual spread eps_t:
            Delta eps_t = gamma * eps_{t-1} + phi * Delta eps_{t-1} + u_t

    Engle-Granger 2-variable critical values:
    - 1%: -3.90
    - 5%: -3.34
    - 10%: -3.04

    Args:
        series_1: Price series for dependent asset (P_1).
        series_2: Price series for independent asset (P_2).
        ticker_1: Label identifier for asset 1.
        ticker_2: Label identifier for asset 2.

    Returns:
        CointegrationReceipt with hedge ratio, ADF statistic, half-life, and z-score.
    """
    s1, s2 = align_time_series(series_1, series_2)
    p1 = s1.values.astype(float)
    p2 = s2.values.astype(float)
    n = len(p1)

    crit_values = {"1%": -3.90, "5%": -3.34, "10%": -3.04}

    if n < 30:
        return CointegrationReceipt(
            ticker_1=ticker_1,
            ticker_2=ticker_2,
            hedge_ratio_beta=1.0,
            intercept_alpha=0.0,
            adf_statistic=0.0,
            critical_values=crit_values,
            p_value_approx=1.0,
            is_cointegrated=False,
            half_life_sessions=float("inf"),
            current_spread=0.0,
            current_spread_zscore=0.0,
            sample_size=n,
            summary=f"Insufficient sample observations ({n}/30 required) for cointegration test.",
        )

    # Step 1: OLS regression P_1 = alpha + beta * P_2 + eps
    des_mat = np.column_stack([np.ones(n), p2])
    b, _, _, _ = np.linalg.lstsq(des_mat, p1, rcond=None)
    alpha_hat = float(b[0])
    beta_hat = float(b[1])

    spread = p1 - (alpha_hat + beta_hat * p2)

    # Step 2: ADF regression on spread residuals: Delta eps_t = gamma * eps_{t-1} + phi * Delta eps_{t-1} + u_t
    delta_spread = np.diff(spread)
    lagged_spread = spread[:-1]

    m = len(delta_spread) - 1
    dy_dep = delta_spread[1:]
    z_mat = np.column_stack([lagged_spread[1:], delta_spread[:-1]])

    gamma_sol, _, _, _ = np.linalg.lstsq(z_mat, dy_dep, rcond=None)
    gamma = float(gamma_sol[0])

    residuals = dy_dep - z_mat @ gamma_sol
    s2_val = float(np.sum(residuals ** 2) / max(1, m - 2))

    try:
        cov_gamma = s2_val * np.linalg.inv(z_mat.T @ z_mat)
        se_gamma = math.sqrt(max(cov_gamma[0, 0], 1e-12))
        adf_stat = gamma / se_gamma
    except Exception:
        adf_stat = 0.0

    # Half-life of mean reversion: -ln(2) / ln(1 + gamma)
    if -0.999 < gamma < 0.0:
        half_life = float(-math.log(2.0) / math.log(1.0 + gamma))
    elif gamma <= -0.999:
        half_life = 1.0
    else:
        half_life = float("inf")

    # Critical value comparison
    is_cointegrated = adf_stat < crit_values["5%"]

    # Approximate p-value based on MacKinnon asymptotic distribution
    if adf_stat <= -3.90:
        p_val = 0.01
    elif adf_stat <= -3.34:
        p_val = 0.05 - 0.04 * (abs(adf_stat) - 3.34) / (3.90 - 3.34)
    elif adf_stat <= -3.04:
        p_val = 0.10 - 0.05 * (abs(adf_stat) - 3.04) / (3.34 - 3.04)
    else:
        p_val = min(0.99, 0.10 + 0.90 * (1.0 - math.exp(max(-5.0, adf_stat + 3.04))))

    # Current spread and rolling z-score
    curr_spread = float(spread[-1])
    std_spread = float(np.std(spread[-30:], ddof=1)) if len(spread) >= 30 else float(np.std(spread, ddof=1))
    mean_spread = float(np.mean(spread[-30:])) if len(spread) >= 30 else float(np.mean(spread))
    curr_zscore = (curr_spread - mean_spread) / max(std_spread, 1e-6)

    summary = (
        f"{ticker_1} and {ticker_2} are cointegrated (ADF={adf_stat:.2f} < 5% crit {crit_values['5%']:.2f}). "
        f"Hedge Ratio Beta={beta_hat:.4f}, Mean-reversion Half-Life={half_life:.1f} sessions."
        if is_cointegrated
        else f"No stationary cointegrating relationship found between {ticker_1} and {ticker_2} (ADF={adf_stat:.2f} >= 5% crit {crit_values['5%']:.2f})."
    )

    return CointegrationReceipt(
        ticker_1=ticker_1,
        ticker_2=ticker_2,
        hedge_ratio_beta=round(beta_hat, 4),
        intercept_alpha=round(alpha_hat, 4),
        adf_statistic=round(adf_stat, 4),
        critical_values=crit_values,
        p_value_approx=round(p_val, 4),
        is_cointegrated=is_cointegrated,
        half_life_sessions=round(half_life, 2) if math.isfinite(half_life) else float("inf"),
        current_spread=round(curr_spread, 4),
        current_spread_zscore=round(curr_zscore, 2),
        sample_size=n,
        summary=summary,
    )


def compute_cointegration_spread_zscore(
    series_1: pd.Series,
    series_2: pd.Series,
    hedge_ratio: float,
    intercept: float = 0.0,
    rolling_window: int = 30,
) -> pd.DataFrame:
    """Calculate time series of cointegration spread, rolling mean, std, z-score, and signals.

    Spread: S_t = P_{1,t} - (intercept + beta * P_{2,t})
    Z-Score: z_t = (S_t - mu_{S,t}) / sigma_{S,t}

    Trading Rules:
    - z_t > +2.0: SHORT_SPREAD (sell P1, buy P2)
    - z_t < -2.0: LONG_SPREAD (buy P1, sell P2)
    - |z_t| <= 0.5: EXIT_POSITION
    - else: HOLD

    Args:
        series_1: Dependent asset price series (P_1).
        series_2: Independent asset price series (P_2).
        hedge_ratio: Estimated cointegrating beta.
        intercept: Estimated cointegrating alpha.
        rolling_window: Rolling window size in sessions (default: 30).

    Returns:
        DataFrame containing date, spread, rolling_mean, rolling_std, z_score, and signal.
    """
    s1, s2 = align_time_series(series_1, series_2)
    spread = s1 - (intercept + hedge_ratio * s2)

    rolling_mean = spread.rolling(window=rolling_window, min_periods=5).mean()
    rolling_std = spread.rolling(window=rolling_window, min_periods=5).std().fillna(1e-4)

    z_score = (spread - rolling_mean) / np.maximum(rolling_std, 1e-6)

    signals: List[str] = []
    for z in z_score:
        if math.isnan(z):
            signals.append("NEUTRAL")
        elif z > 2.0:
            signals.append("SHORT_SPREAD")
        elif z < -2.0:
            signals.append("LONG_SPREAD")
        elif abs(z) <= 0.5:
            signals.append("EXIT_POSITION")
        else:
            signals.append("HOLD")

    result_df = pd.DataFrame({
        "session_date": s1.index,
        "price_1": s1.values,
        "price_2": s2.values,
        "spread": np.round(spread.values, 4),
        "rolling_mean": np.round(rolling_mean.values, 4),
        "rolling_std": np.round(rolling_std.values, 4),
        "z_score": np.round(z_score.values, 2),
        "signal": signals,
    })

    return result_df
