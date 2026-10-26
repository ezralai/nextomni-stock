"""Chronological data splitting and target alignment for time-series forecasting.

Strictly preserves temporal ordering to prevent lookahead bias.
"""

from typing import Tuple
import pandas as pd

from src.config import PRICE_COLUMN


def align_forecasting_target(
    df: pd.DataFrame,
    horizon_sessions: int = 1,
    price_col: str = PRICE_COLUMN,
) -> pd.DataFrame:
    """Align features at session t with target price at session t + k.

    Target Definition:
        For horizon k, target_price = P_{t+k}.
        target_session_date = session_date_{t+k}.

    Args:
        df: Chronologically sorted observations DataFrame.
        horizon_sessions: Number of future trading sessions (1, 5, 21).
        price_col: Column name of price to forecast (default: adj_close).

    Returns:
        pd.DataFrame: Contains original columns plus 'target_price' and 'target_session_date'.
        Note: The last `horizon_sessions` rows will have NaN for target_price because
        their future target sessions have not yet occurred.
    """
    if df.empty or len(df) <= horizon_sessions:
        raise ValueError(
            f"DataFrame has {len(df)} rows, which is insufficient for horizon {horizon_sessions}."
        )

    aligned = df.copy()
    aligned = aligned.sort_values("session_date", ascending=True).reset_index(drop=True)

    # Shift backwards by horizon_sessions: row t receives value from row t + horizon_sessions
    aligned["target_price"] = aligned[price_col].shift(-horizon_sessions)
    aligned["target_session_date"] = aligned["session_date"].shift(-horizon_sessions)

    return aligned


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically into train and test sets.

    NEVER randomly shuffles rows. First (train_ratio * 100)% rows become the training set,
    and the remaining rows become the out-of-sample test set.

    Args:
        df: Chronologically sorted DataFrame.
        train_ratio: Fraction of historical observations allocated to training (e.g. 0.8).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if not (0.1 <= train_ratio <= 0.9):
        raise ValueError(f"train_ratio must be between 0.1 and 0.9, got {train_ratio}.")

    if len(df) < 5:
        raise ValueError("DataFrame too small to split chronologically.")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df
