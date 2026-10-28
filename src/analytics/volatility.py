"""Advanced quantitative realized volatility and extreme value estimators for NextOmni Stock.

Implements benchmark extreme-value, jump-adjusted, and drift-independent volatility estimators:
1. Close-to-Close Standard Deviation (Historical baseline, efficiency ~1.0)
2. Parkinson (1980) High-Low Extreme Value Estimator (efficiency ~5.2)
3. Garman-Klass (1980) OHLC Continuous Diffusion Estimator (efficiency ~7.4)
4. Rogers-Satchell (1991) Non-Zero Drift Intraday Estimator
5. Yang-Zhang (2000) Minimum-Variance Unbiased Estimator (efficiency ~14.0)
6. RiskMetrics EWMA Exponential Variance Filter (lambda=0.94)
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class VolatilityDiagnosticsReceipt:
    """Comprehensive diagnostic receipt from multi-estimator volatility profiling."""
    ticker: str
    sample_sessions: int
    window: int
    close_to_close_ann_pct: float
    parkinson_ann_pct: float
    garman_klass_ann_pct: float
    rogers_satchell_ann_pct: float
    yang_zhang_ann_pct: float
    ewma_ann_pct: float
    volatility_ratio_yz_cc: float
    regime_signal: str
    term_structure_yang_zhang: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def _prepare_ohlc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and validate OHLC structure."""
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    data = df.copy()
    col_map = {str(c).lower(): c for c in data.columns}
    required = ["open", "high", "low", "close"]

    for req in required:
        if req not in col_map:
            raise ValueError(f"Missing required price column '{req}' in dataframe")

    standardized = pd.DataFrame(index=data.index)
    for req in required:
        standardized[req] = pd.to_numeric(data[col_map[req]], errors="coerce")

    # Drop any non-positive or NaN values to prevent logarithm domain errors
    standardized = standardized.dropna()
    standardized = standardized[
        (standardized["open"] > 0)
        & (standardized["high"] > 0)
        & (standardized["low"] > 0)
        & (standardized["close"] > 0)
    ]
    # Enforce High >= Low boundary integrity
    standardized["high"] = np.maximum(standardized["high"], standardized["low"])
    return standardized


def compute_close_to_close_volatility(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252,
) -> pd.Series:
    """Compute annualized Close-to-Close rolling volatility.

    Args:
        df: DataFrame containing 'close' prices.
        window: Rolling observation window (sessions).
        trading_periods: Annualization factor (sessions per year, default 252).

    Returns:
        pd.Series of annualized close-to-close volatility.
    """
    clean_df = _prepare_ohlc_frame(df)
    if len(clean_df) < 2:
        return pd.Series(dtype=float, index=df.index)

    log_returns = np.log(clean_df["close"] / clean_df["close"].shift(1))
    min_p = max(2, min(window, 5))
    rolling_std = log_returns.rolling(window=window, min_periods=min_p).std(ddof=1)
    ann_vol = rolling_std * math.sqrt(trading_periods)
    return ann_vol.reindex(df.index)


def compute_parkinson_volatility(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252,
) -> pd.Series:
    """Compute Parkinson (1980) High-Low extreme value volatility.

    Efficiency is ~5.2x higher than close-to-close variance under pure geometric Brownian motion.

    Formula:
        sigma_P^2 = (1 / (4 * ln(2) * N)) * sum( (ln(H_t / L_t))^2 )

    Args:
        df: DataFrame containing 'high' and 'low' prices.
        window: Rolling window length.
        trading_periods: Annualization factor (default 252).

    Returns:
        pd.Series of annualized Parkinson volatility.
    """
    clean_df = _prepare_ohlc_frame(df)
    if len(clean_df) < 2:
        return pd.Series(dtype=float, index=df.index)

    hl_ratio = np.maximum(1e-12, clean_df["high"] / np.maximum(1e-12, clean_df["low"]))
    log_hl = np.log(hl_ratio)
    factor = 1.0 / (4.0 * math.log(2.0))
    parkinson_var_daily = factor * (log_hl ** 2)

    min_p = max(2, min(window, 5))
    rolling_var = parkinson_var_daily.rolling(window=window, min_periods=min_p).mean()
    ann_vol = np.sqrt(np.maximum(0.0, rolling_var * trading_periods))
    return ann_vol.reindex(df.index)


def compute_garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252,
) -> pd.Series:
    """Compute Garman-Klass (1980) OHLC continuous diffusion volatility.

    Efficiency is ~7.4x higher than close-to-close variance under zero-drift diffusion.

    Formula:
        sigma_GK^2 = (1 / N) * sum( 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 )

    Args:
        df: DataFrame with OHLC columns.
        window: Rolling window length.
        trading_periods: Annualization factor (default 252).

    Returns:
        pd.Series of annualized Garman-Klass volatility.
    """
    clean_df = _prepare_ohlc_frame(df)
    if len(clean_df) < 2:
        return pd.Series(dtype=float, index=df.index)

    hl_ratio = np.maximum(1e-12, clean_df["high"] / np.maximum(1e-12, clean_df["low"]))
    co_ratio = np.maximum(1e-12, clean_df["close"] / np.maximum(1e-12, clean_df["open"]))
    log_hl = np.log(hl_ratio)
    log_co = np.log(co_ratio)

    gk_const = 2.0 * math.log(2.0) - 1.0
    daily_var = 0.5 * (log_hl ** 2) - gk_const * (log_co ** 2)

    min_p = max(2, min(window, 5))
    rolling_var = daily_var.rolling(window=window, min_periods=min_p).mean()
    ann_vol = np.sqrt(np.maximum(0.0, rolling_var * trading_periods))
    return ann_vol.reindex(df.index)


def compute_rogers_satchell_volatility(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252,
) -> pd.Series:
    """Compute Rogers-Satchell (1991) drift-independent volatility.

    Allows non-zero drift and handles trending markets without upward volatility bias.

    Formula:
        sigma_RS^2 = (1 / N) * sum( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )

    Args:
        df: DataFrame with OHLC columns.
        window: Rolling window length.
        trading_periods: Annualization factor (default 252).

    Returns:
        pd.Series of annualized Rogers-Satchell volatility.
    """
    clean_df = _prepare_ohlc_frame(df)
    if len(clean_df) < 2:
        return pd.Series(dtype=float, index=df.index)

    h = clean_df["high"]
    l = clean_df["low"]
    c = clean_df["close"]
    o = clean_df["open"]

    log_hc = np.log(np.maximum(1e-12, h / c))
    log_ho = np.log(np.maximum(1e-12, h / o))
    log_lc = np.log(np.maximum(1e-12, l / c))
    log_lo = np.log(np.maximum(1e-12, l / o))

    daily_var = (log_hc * log_ho) + (log_lc * log_lo)

    min_p = max(2, min(window, 5))
    rolling_var = daily_var.rolling(window=window, min_periods=min_p).mean()
    ann_vol = np.sqrt(np.maximum(0.0, rolling_var * trading_periods))
    return ann_vol.reindex(df.index)


def compute_yang_zhang_volatility(
    df: pd.DataFrame,
    window: int = 20,
    trading_periods: int = 252,
) -> pd.Series:
    """Compute Yang-Zhang (2000) minimum-variance unbiased volatility estimator.

    Combines overnight jump variance, continuous open-to-close variance, and Rogers-Satchell
    intraday variance. Exhibits an efficiency ~14x higher than close-to-close variance.

    Formula:
        sigma_YZ^2 = sigma_o^2 + k * sigma_c^2 + (1 - k) * sigma_RS^2
        k = 0.34 / (1.34 + (N + 1) / (N - 1))

    Args:
        df: DataFrame with OHLC columns.
        window: Rolling window length (sessions).
        trading_periods: Annualization factor (default 252).

    Returns:
        pd.Series of annualized Yang-Zhang volatility.
    """
    clean_df = _prepare_ohlc_frame(df)
    if len(clean_df) < 3:
        return pd.Series(dtype=float, index=df.index)

    o = clean_df["open"]
    h = clean_df["high"]
    l = clean_df["low"]
    c = clean_df["close"]
    c_prev = c.shift(1)

    # Overnight jumps: ln(O_t / C_{t-1})
    log_oc_prev = np.log(np.maximum(1e-12, o / np.maximum(1e-12, c_prev)))
    # Open to close: ln(C_t / O_t)
    log_co = np.log(np.maximum(1e-12, c / np.maximum(1e-12, o)))

    # Rogers-Satchell daily components
    log_hc = np.log(np.maximum(1e-12, h / c))
    log_ho = np.log(np.maximum(1e-12, h / o))
    log_lc = np.log(np.maximum(1e-12, l / c))
    log_lo = np.log(np.maximum(1e-12, l / o))
    rs_daily = (log_hc * log_ho) + (log_lc * log_lo)

    min_p = max(3, min(window, 5))

    # Rolling sample variances with ddof=1
    sigma_o_sq = log_oc_prev.rolling(window=window, min_periods=min_p).var(ddof=1)
    sigma_c_sq = log_co.rolling(window=window, min_periods=min_p).var(ddof=1)
    sigma_rs_sq = rs_daily.rolling(window=window, min_periods=min_p).mean()

    # Optimal weighting constant k
    n_effective = window if window > 2 else 3
    k = 0.34 / (1.34 + (n_effective + 1) / (n_effective - 1))

    yz_daily_var = sigma_o_sq + (k * sigma_c_sq) + ((1.0 - k) * sigma_rs_sq)
    ann_vol = np.sqrt(np.maximum(0.0, yz_daily_var * trading_periods))
    return ann_vol.reindex(df.index)


def compute_ewma_volatility(
    df: pd.DataFrame,
    lambda_param: float = 0.94,
    trading_periods: int = 252,
) -> pd.Series:
    """Compute RiskMetrics Exponentially Weighted Moving Average (EWMA) volatility.

    Accounts for volatility clustering and persistence:
        sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

    Args:
        df: DataFrame containing 'close' prices.
        lambda_param: Decay parameter (default 0.94 for daily market equity data).
        trading_periods: Annualization factor (default 252).

    Returns:
        pd.Series of annualized EWMA volatility.
    """
    clean_df = _prepare_ohlc_frame(df)
    if len(clean_df) < 2:
        return pd.Series(dtype=float, index=df.index)

    log_returns = np.log(clean_df["close"] / clean_df["close"].shift(1)).fillna(0.0)
    returns_sq = log_returns ** 2

    alpha = 1.0 - lambda_param
    ewma_var = returns_sq.ewm(alpha=alpha, adjust=False).mean()
    ann_vol = np.sqrt(np.maximum(0.0, ewma_var * trading_periods))
    return ann_vol.reindex(df.index)


def evaluate_volatility_profile(
    df: pd.DataFrame,
    ticker: str = "TICKER",
    window: int = 20,
    trading_periods: int = 252,
) -> VolatilityDiagnosticsReceipt:
    """Generate a comprehensive multi-estimator realized volatility diagnostic receipt.

    Args:
        df: Historical market observation DataFrame containing OHLC bars.
        ticker: Ticker symbol.
        window: Core estimation window (sessions).
        trading_periods: Annualization factor (default 252).

    Returns:
        VolatilityDiagnosticsReceipt populated with latest values and diagnostics.
    """
    clean_df = _prepare_ohlc_frame(df)
    n_sessions = len(clean_df)

    if n_sessions < 5:
        return VolatilityDiagnosticsReceipt(
            ticker=ticker,
            sample_sessions=n_sessions,
            window=window,
            close_to_close_ann_pct=0.0,
            parkinson_ann_pct=0.0,
            garman_klass_ann_pct=0.0,
            rogers_satchell_ann_pct=0.0,
            yang_zhang_ann_pct=0.0,
            ewma_ann_pct=0.0,
            volatility_ratio_yz_cc=1.0,
            regime_signal="INSUFFICIENT_DATA",
            term_structure_yang_zhang={},
        )

    # Compute series across all 5 estimators
    s_cc = compute_close_to_close_volatility(clean_df, window=window, trading_periods=trading_periods)
    s_p = compute_parkinson_volatility(clean_df, window=window, trading_periods=trading_periods)
    s_gk = compute_garman_klass_volatility(clean_df, window=window, trading_periods=trading_periods)
    s_rs = compute_rogers_satchell_volatility(clean_df, window=window, trading_periods=trading_periods)
    s_yz = compute_yang_zhang_volatility(clean_df, window=window, trading_periods=trading_periods)
    s_ewma = compute_ewma_volatility(clean_df, lambda_param=0.94, trading_periods=trading_periods)

    val_cc = float(s_cc.dropna().iloc[-1]) if not s_cc.dropna().empty else 0.0
    val_p = float(s_p.dropna().iloc[-1]) if not s_p.dropna().empty else 0.0
    val_gk = float(s_gk.dropna().iloc[-1]) if not s_gk.dropna().empty else 0.0
    val_rs = float(s_rs.dropna().iloc[-1]) if not s_rs.dropna().empty else 0.0
    val_yz = float(s_yz.dropna().iloc[-1]) if not s_yz.dropna().empty else 0.0
    val_ewma = float(s_ewma.dropna().iloc[-1]) if not s_ewma.dropna().empty else 0.0

    # Volatility Ratio: Yang-Zhang to Close-to-Close
    vr = (val_yz / val_cc) if val_cc > 1e-6 else 1.0

    # Regime categorization based on ratio and level
    if vr > 1.35:
        regime = "OVERNIGHT_GAP_OR_INTRADAY_SURGE"
    elif vr < 0.75:
        regime = "MEAN_REVERTING_INTRADAY"
    elif val_yz > 0.45:
        regime = "HIGH_VOLATILITY_EXPANSION"
    elif val_yz < 0.15:
        regime = "COMPRESSED_VOLATILITY"
    else:
        regime = "NORMAL_DIFFUSION"

    # Multi-window term structure
    term_structure: Dict[str, float] = {}
    for w in [5, 10, 20, 60]:
        if n_sessions >= w:
            s_w = compute_yang_zhang_volatility(clean_df, window=w, trading_periods=trading_periods)
            term_structure[f"{w}d"] = round(float(s_w.dropna().iloc[-1]) * 100, 2) if not s_w.dropna().empty else 0.0

    return VolatilityDiagnosticsReceipt(
        ticker=ticker,
        sample_sessions=n_sessions,
        window=window,
        close_to_close_ann_pct=round(val_cc * 100, 2),
        parkinson_ann_pct=round(val_p * 100, 2),
        garman_klass_ann_pct=round(val_gk * 100, 2),
        rogers_satchell_ann_pct=round(val_rs * 100, 2),
        yang_zhang_ann_pct=round(val_yz * 100, 2),
        ewma_ann_pct=round(val_ewma * 100, 2),
        volatility_ratio_yz_cc=round(vr, 3),
        regime_signal=regime,
        term_structure_yang_zhang=term_structure,
    )
