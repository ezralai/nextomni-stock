"""Feature engineering module for NextOmni Stock.

Constructs interpretable, leak-free quantitative signals using exclusively
data available at or before session t.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import PRICE_COLUMN
from src.models.splitter import align_forecasting_target

FEATURE_COLUMNS = [
    "lag_return_1",
    "lag_return_2",
    "lag_return_5",
    "price_to_sma_5",
    "price_to_sma_20",
    "rolling_vol_21",
    "volume_ratio_20",
    "gk_vol_21",
    "rsi_14",
    "bb_pct_b_20",
    "vol_regime_ratio",
    "macd_norm",
]


def build_feature_matrix(
    df: pd.DataFrame,
    include_target: bool = True,
    horizon_sessions: int = 1,
    price_col: str = PRICE_COLUMN,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct leak-free features for time-series forecasting.

    Features generated at session t (strictly backward-looking P_{<= t}):
    1. lag_return_1: Daily return from session t-1 to t: (P_t - P_{t-1}) / P_{t-1}
    2. lag_return_2: Daily return from session t-2 to t-1: (P_{t-1} - P_{t-2}) / P_{t-2}
    3. lag_return_5: 5-session momentum: (P_t - P_{t-5}) / P_{t-5}
    4. price_to_sma_5: Ratio of closing price to its 5-session SMA: P_t / SMA_5(P_t)
    5. price_to_sma_20: Ratio of closing price to its 20-session SMA: P_t / SMA_20(P_t)
    6. rolling_vol_21: 21-session rolling standard deviation of daily returns
    7. volume_ratio_20: Ratio of current volume to 20-session average volume: V_t / SMA_20(V_t)
    8. gk_vol_21: 21-session Garman-Klass intra-day volatility estimator (Open, High, Low, Close)
    9. rsi_14: 14-session Relative Strength Index normalized to [0, 1]
    10. bb_pct_b_20: 20-session Bollinger Band %B normalized position
    11. vol_regime_ratio: 5-session to 21-session volatility ratio (regime expansion/contraction)
    12. macd_norm: Normalized 12/26/9 MACD histogram difference scaled by price

    All rolling metrics strictly include observations up to index t, with ZERO forward leakage.

    Args:
        df: Chronologically sorted observations DataFrame.
        include_target: If True, adds 'target_price' (P_{t+k}) and 'target_session_date'.
        horizon_sessions: Forecast horizon in trading sessions (default: 1).
        price_col: Price column name (default: adj_close).

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - DataFrame with engineered features (and target if requested).
            - List of feature column names.
    """
    if len(df) < 25:
        raise ValueError(
            f"DataFrame must contain at least 25 sessions to construct 21-session rolling features. Got {len(df)}."
        )

    data = df.copy()
    data = data.sort_values("session_date", ascending=True).reset_index(drop=True)

    # Base daily returns
    daily_returns = data[price_col].pct_change()

    # 1. Lagged returns
    data["lag_return_1"] = daily_returns
    data["lag_return_2"] = daily_returns.shift(1)
    data["lag_return_5"] = data[price_col].pct_change(periods=5)

    # 2. Moving average ratios (measures momentum and mean-reversion stretch)
    sma_5 = data[price_col].rolling(window=5, min_periods=5).mean()
    sma_20 = data[price_col].rolling(window=20, min_periods=20).mean()
    data["price_to_sma_5"] = data[price_col] / sma_5
    data["price_to_sma_20"] = data[price_col] / sma_20

    # 3. Rolling volatility (measures current market regime volatility)
    data["rolling_vol_21"] = daily_returns.rolling(window=21, min_periods=21).std()

    # 4. Volume pressure
    avg_vol_20 = data["volume"].rolling(window=20, min_periods=20).mean()
    data["volume_ratio_20"] = data["volume"] / np.maximum(avg_vol_20, 1.0)

    # 5. Garman-Klass Volatility (lower-variance intra-day estimator)
    # sigma_GK^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    high_clean = np.maximum(data.get("high", data[price_col]), 1e-4)
    low_clean = np.maximum(data.get("low", data[price_col]), 1e-4)
    close_clean = np.maximum(data.get("close", data[price_col]), 1e-4)
    open_clean = np.maximum(data.get("open", data[price_col]), 1e-4)

    log_hl = np.log(high_clean / low_clean)
    log_co = np.log(close_clean / open_clean)
    gk_variance = 0.5 * (log_hl ** 2) - (2.0 * np.log(2) - 1.0) * (log_co ** 2)
    data["gk_vol_21"] = np.sqrt(np.maximum(gk_variance.rolling(window=21, min_periods=21).mean(), 0.0))

    # 6. Relative Strength Index (RSI 14, normalized to [0, 1])
    gain = np.where(daily_returns > 0, daily_returns, 0.0)
    loss = np.where(daily_returns < 0, -daily_returns, 0.0)
    roll_gain = pd.Series(gain, index=data.index).rolling(window=14, min_periods=14).mean()
    roll_loss = pd.Series(loss, index=data.index).rolling(window=14, min_periods=14).mean()
    rs = roll_gain / np.maximum(roll_loss, 1e-8)
    data["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))) / 100.0

    # 7. Bollinger Band %B (20 sessions, 2 standard deviations)
    std_20 = data[price_col].rolling(window=20, min_periods=20).std()
    upper_b = sma_20 + (2.0 * std_20)
    lower_b = sma_20 - (2.0 * std_20)
    band_width = np.maximum(upper_b - lower_b, 1e-8)
    data["bb_pct_b_20"] = (data[price_col] - lower_b) / band_width

    # 8. Volatility Regime Ratio (short-term 5-day vol / long-term 21-day vol)
    vol_5 = daily_returns.rolling(window=5, min_periods=5).std()
    data["vol_regime_ratio"] = vol_5 / np.maximum(data["rolling_vol_21"], 1e-4)

    # 9. Normalized MACD Histogram Difference (12/26/9 EMA)
    ema_12 = data[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = data[price_col].ewm(span=26, adjust=False).mean()
    macd_line = (ema_12 - ema_26) / data[price_col]
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    data["macd_norm"] = macd_line - macd_signal

    # 10. Target alignment
    if include_target:
        data = align_forecasting_target(
            data, horizon_sessions=horizon_sessions, price_col=price_col
        )

    # Drop the initial warm-up rows where rolling windows are NaN (first 21 sessions)
    feature_ready = data.dropna(subset=FEATURE_COLUMNS).copy().reset_index(drop=True)

    return feature_ready, FEATURE_COLUMNS
