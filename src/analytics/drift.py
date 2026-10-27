"""Quantitative model drift and rolling Information Coefficient (IC) monitor for NextOmni Stock.

Tracks time-varying forecasting skill by measuring rolling Pearson and Spearman rank
correlations between multi-horizon expected returns and realized post-settlement returns,
detecting alpha decay and alerting when models require retraining.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ICDriftReceipt:
    """Quantitative diagnostic receipt summarizing model alpha decay and IC trajectory."""
    ticker: Optional[str]
    horizon_type: Optional[str]
    sample_size: int
    mean_ic: float
    mean_rank_ic: float
    ic_std: float
    information_ratio: float
    t_statistic: float
    trailing_rank_ic: float
    is_decaying: bool
    decay_status: str
    decay_warning: Optional[str]
    history_df: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["history"] = self.history_df.to_dict(orient="records")
        d.pop("history_df", None)
        return d


def calculate_information_coefficient_drift(
    predictions_df: pd.DataFrame,
    window: int = 20,
) -> ICDriftReceipt:
    """Compute rolling Information Coefficient time series and evaluate alpha decay.

    Formulation:
    - Expected Return: hat{r}_t = (P_pred - P_origin) / P_origin
    - Realized Return: r_t = (P_actual - P_origin) / P_origin
    - Information Coefficient: IC_t = Corr(hat{r}, r)
    - Rank IC: Rank_IC_t = Corr_rank(hat{r}, r)
    - Annualized Information Ratio: IR = (mean(Rank_IC) / std(Rank_IC)) * sqrt(252)

    Args:
        predictions_df: DataFrame containing at least ['predicted_price', 'origin_price', 'actual_price']
                        and optionally ['session_date' or 'target_session_date', 'ticker', 'horizon_type'].
        window: Rolling observation window for tracking time-varying decay (default: 20).

    Returns:
        ICDriftReceipt containing rolling metrics, decay indicators, and time series.
    """
    ticker = str(predictions_df["ticker"].iloc[0]) if "ticker" in predictions_df.columns and not predictions_df.empty else None
    horizon = str(predictions_df["horizon_type"].iloc[0]) if "horizon_type" in predictions_df.columns and not predictions_df.empty else None

    # Filter valid evaluated rows
    date_col = "target_session_date" if "target_session_date" in predictions_df.columns else (
        "session_date" if "session_date" in predictions_df.columns else None
    )

    valid_mask = (
        predictions_df["predicted_price"].notna()
        & predictions_df["origin_price"].notna()
        & predictions_df["actual_price"].notna()
        & (predictions_df["origin_price"] > 0)
        & (predictions_df["actual_price"] > 0)
    )
    clean_df = predictions_df[valid_mask].copy()

    if date_col and date_col in clean_df.columns:
        clean_df = clean_df.sort_values(date_col, ascending=True).reset_index(drop=True)
    else:
        clean_df = clean_df.reset_index(drop=True)

    n = len(clean_df)
    empty_history = pd.DataFrame(columns=["target_date", "predicted_return_pct", "realized_return_pct", "rolling_ic", "rolling_rank_ic"])

    if n < 5:
        return ICDriftReceipt(
            ticker=ticker,
            horizon_type=horizon,
            sample_size=n,
            mean_ic=0.0,
            mean_rank_ic=0.0,
            ic_std=0.0,
            information_ratio=0.0,
            t_statistic=0.0,
            trailing_rank_ic=0.0,
            is_decaying=False,
            decay_status="INSUFFICIENT_DATA",
            decay_warning=f"Insufficient evaluated predictions ({n}/5 required) for Information Coefficient calculation.",
            history_df=empty_history,
        )

    p_orig = clean_df["origin_price"].astype(float)
    p_pred = clean_df["predicted_price"].astype(float)
    p_act = clean_df["actual_price"].astype(float)

    pred_ret = (p_pred - p_orig) / p_orig
    real_ret = (p_act - p_orig) / p_orig

    # Global Pearson and Spearman correlations
    std_p = pred_ret.std(ddof=1)
    std_r = real_ret.std(ddof=1)

    if std_p > 1e-8 and std_r > 1e-8:
        global_ic = float(pred_ret.corr(real_ret, method="pearson"))
        global_rank_ic = float(pred_ret.corr(real_ret, method="spearman"))
    else:
        global_ic = 0.0
        global_rank_ic = 0.0

    global_ic = global_ic if math.isfinite(global_ic) else 0.0
    global_rank_ic = global_rank_ic if math.isfinite(global_rank_ic) else 0.0

    # Rolling IC computation
    rolling_ic_series: List[float] = []
    rolling_rank_ic_series: List[float] = []

    effective_w = min(window, n)
    min_obs = min(5, effective_w)

    for i in range(n):
        if i + 1 < min_obs:
            rolling_ic_series.append(global_ic)
            rolling_rank_ic_series.append(global_rank_ic)
        else:
            sub_p = pred_ret.iloc[max(0, i + 1 - effective_w) : i + 1]
            sub_r = real_ret.iloc[max(0, i + 1 - effective_w) : i + 1]
            if sub_p.std() > 1e-8 and sub_r.std() > 1e-8:
                ic_val = float(sub_p.corr(sub_r, method="pearson"))
                rank_ic_val = float(sub_p.corr(sub_r, method="spearman"))
            else:
                ic_val = 0.0
                rank_ic_val = 0.0

            rolling_ic_series.append(ic_val if math.isfinite(ic_val) else 0.0)
            rolling_rank_ic_series.append(rank_ic_val if math.isfinite(rank_ic_val) else 0.0)

    rolling_rank_arr = np.array(rolling_rank_ic_series)
    ic_std = float(np.std(rolling_rank_arr, ddof=1)) if len(rolling_rank_arr) > 1 else 0.0
    safe_std = max(ic_std, 1e-4)

    information_ratio = float((global_rank_ic / safe_std) * math.sqrt(252))
    t_stat = float((global_rank_ic / safe_std) * math.sqrt(n))
    trailing_rank_ic = float(rolling_rank_arr[-1])

    # Decay classification
    if trailing_rank_ic < -0.05 and n >= 10:
        is_decaying = True
        decay_status = "ACUTE_DECAY"
        decay_warning = (
            f"Acute alpha decay alert: Trailing {effective_w}-session Rank IC is negative "
            f"({trailing_rank_ic:+.3f}). Estimator requires retraining."
        )
    elif n >= 20 and trailing_rank_ic < (global_rank_ic - 1.5 * safe_std):
        is_decaying = True
        decay_status = "MILD_DECAY"
        decay_warning = (
            f"Mild alpha decay warning: Trailing Rank IC ({trailing_rank_ic:+.3f}) has fallen "
            f"> 1.5 sigma below inception mean ({global_rank_ic:+.3f})."
        )
    else:
        is_decaying = False
        decay_status = "STABLE_ALPHA"
        decay_warning = None

    dates = clean_df[date_col].tolist() if date_col and date_col in clean_df.columns else [f"t_{i}" for i in range(n)]

    history_df = pd.DataFrame({
        "target_date": dates,
        "predicted_return_pct": np.round(pred_ret.values * 100.0, 2),
        "realized_return_pct": np.round(real_ret.values * 100.0, 2),
        "rolling_ic": np.round(rolling_ic_series, 4),
        "rolling_rank_ic": np.round(rolling_rank_ic_series, 4),
    })

    return ICDriftReceipt(
        ticker=ticker,
        horizon_type=horizon,
        sample_size=n,
        mean_ic=round(global_ic, 4),
        mean_rank_ic=round(global_rank_ic, 4),
        ic_std=round(ic_std, 4),
        information_ratio=round(information_ratio, 3),
        t_statistic=round(t_stat, 3),
        trailing_rank_ic=round(trailing_rank_ic, 4),
        is_decaying=is_decaying,
        decay_status=decay_status,
        decay_warning=decay_warning,
        history_df=history_df,
    )
