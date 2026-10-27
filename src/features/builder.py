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
]


def build_feature_matrix(
    df: pd.DataFrame,
    include_target: bool = True,
    horizon_sessions: int = 1,
    price_col: str = PRICE_COLUMN,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct leak-free features for time-series forecasting.

    Features generated at session t:
    1. lag_return_1: Daily return from session t-1 to t: (P_t - P_{t-1}) / P_{t-1}
    2. lag_return_2: Daily return from session t-2 to t-1: (P_{t-1} - P_{t-2}) / P_{t-2}
    3. lag_return_5: 5-session momentum: (P_t - P_{t-5}) / P_{t-5}
    4. price_to_sma_5: Ratio of closing price to its 5-session SMA: P_t / SMA_5(P_t)
    5. price_to_sma_20: Ratio of closing price to its 20-session SMA: P_t / SMA_20(P_t)
    6. rolling_vol_21: 21-session rolling standard deviation of daily returns
    7. volume_ratio_20: Ratio of current volume to 20-session average volume: V_t / SMA_20(V_t)

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
    # Guard against zero volume in denominator
    data["volume_ratio_20"] = data["volume"] / np.maximum(avg_vol_20, 1.0)

    # 5. Target alignment
    if include_target:
        data = align_forecasting_target(
            data, horizon_sessions=horizon_sessions, price_col=price_col
        )

    # Drop the initial warm-up rows where rolling windows are NaN (first 21 sessions)
    feature_ready = data.dropna(subset=FEATURE_COLUMNS).copy().reset_index(drop=True)

    return feature_ready, FEATURE_COLUMNS
