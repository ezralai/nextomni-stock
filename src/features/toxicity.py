"""Volume-Synchronized Probability of Toxicity (VPIN) and Flow Imbalance Feature Engine for NextOmni Stock.

Implements the canonical Easley, Lopez de Prado & O'Hara (2012) framework:
- Bulk Volume Classification (BVC) decomposing continuous trading volume into buyer- and seller-initiated flow.
- High-frequency order flow imbalance and Volume-Synchronized Probability of Toxicity (VPIN).
- Quantitative toxicity regime categorization for adverse selection detection.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import PRICE_COLUMN


@dataclass
class ToxicityReceipt:
    """Quantitative diagnostic receipt summarizing order flow toxicity and VPIN metrics."""
    ticker: str
    sample_sessions: int
    window: int
    latest_vpin: float
    mean_vpin: float
    max_vpin: float
    latest_buyer_volume_pct: float
    latest_seller_volume_pct: float
    toxicity_regime: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def classify_bulk_volume(
    prices: Union[np.ndarray, pd.Series, List[float]],
    volumes: Union[np.ndarray, pd.Series, List[float]],
    window: int = 20,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Decompose trading volume into buyer- and seller-initiated flow via Bulk Volume Classification (BVC).

    BVC computes standardized price return innovations Z_t = Delta P_t / sigma_{Delta P}
    and assigns buyer volume as V_t^B = V_t * Phi(Z_t), with seller volume V_t^S = V_t - V_t^B.

    Args:
        prices: Historical price series.
        volumes: Historical volume series.
        window: Rolling observation window for return volatility estimation (default: 20).

    Returns:
        Tuple of (buyer_volumes, seller_volumes, flow_imbalances).
    """
    p_series = pd.Series(prices, dtype=float).reset_index(drop=True)
    v_series = pd.Series(volumes, dtype=float).reset_index(drop=True)

    n = len(p_series)
    if n == 0 or len(v_series) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    # Calculate price change returns
    returns = p_series.pct_change().fillna(0.0)

    # Rolling volatility of price returns
    min_w = max(3, min(window, 5))
    rolling_std = returns.rolling(window=window, min_periods=min_w).std().fillna(returns.std())
    rolling_std = rolling_std.replace(0.0, 1e-4).fillna(1e-4)

    # Standardized return innovations
    z_scores = returns / rolling_std
    z_scores = z_scores.clip(lower=-5.0, upper=5.0)

    # Probability of buyer initiation via standard normal CDF Phi(z)
    p_buyer = pd.Series(stats.norm.cdf(z_scores.values), index=p_series.index)

    # Zero-volume guard: preserve exact volume conservation
    clean_v = v_series.clip(lower=0.0).fillna(0.0)
    v_buyer = clean_v * p_buyer
    v_seller = clean_v * (1.0 - p_buyer)
    imbalance = (v_buyer - v_seller).abs()

    return v_buyer, v_seller, imbalance


def calculate_vpin_series(
    prices: Union[np.ndarray, pd.Series, List[float]],
    volumes: Union[np.ndarray, pd.Series, List[float]],
    window: int = 20,
) -> pd.Series:
    """Calculate rolling Volume-Synchronized Probability of Toxicity (VPIN).

    VPIN measures the fraction of total trading volume driven by directional order flow imbalance:
    VPIN_t = sum_{i=0}^{w-1} |V_{t-i}^B - V_{t-i}^S| / sum_{i=0}^{w-1} V_{t-i}.

    Args:
        prices: Historical price series.
        volumes: Historical volume series.
        window: Rolling observation window in sessions (default: 20).

    Returns:
        pd.Series of rolling VPIN bounded strictly in [0.0, 1.0].
    """
    v_buyer, v_seller, imbalance = classify_bulk_volume(prices, volumes, window=window)
    if imbalance.empty:
        return pd.Series(dtype=float)

    v_total = pd.Series(volumes, dtype=float).clip(lower=0.0).fillna(0.0)
    min_w = max(3, min(window, 5))

    rolling_imbalance = imbalance.rolling(window=window, min_periods=min_w).sum()
    rolling_volume = v_total.rolling(window=window, min_periods=min_w).sum()

    # Guard against zero-volume denominator
    safe_volume = rolling_volume.replace(0.0, np.nan)
    vpin = (rolling_imbalance / safe_volume).fillna(0.0).clip(lower=0.0, upper=1.0)
    return vpin


def evaluate_flow_toxicity(
    df: pd.DataFrame,
    ticker: str = "TICKER",
    window: int = 20,
    price_col: str = PRICE_COLUMN,
) -> ToxicityReceipt:
    """Evaluate asset order flow toxicity and generate diagnostic receipt.

    Args:
        df: Historical market observations DataFrame.
        ticker: Asset ticker symbol.
        window: Rolling observation window in sessions (default: 20).
        price_col: Column containing settlement prices (default: adj_close).

    Returns:
        ToxicityReceipt summarizing VPIN and flow imbalance metrics.
    """
    if df.empty or len(df) < 5:
        return ToxicityReceipt(
            ticker=ticker,
            sample_sessions=len(df),
            window=window,
            latest_vpin=0.0,
            mean_vpin=0.0,
            max_vpin=0.0,
            latest_buyer_volume_pct=50.0,
            latest_seller_volume_pct=50.0,
            toxicity_regime="INSUFFICIENT_DATA",
            summary=f"Insufficient observations (n={len(df)} < 5) to compute order flow toxicity.",
        )

    clean_df = df.copy().sort_values("session_date", ascending=True).reset_index(drop=True)
    p_col = price_col if price_col in clean_df.columns else "close"
    if p_col not in clean_df.columns:
        p_col = clean_df.columns[1]

    v_col = "volume" if "volume" in clean_df.columns else clean_df.columns[-1]

    prices = pd.to_numeric(clean_df[p_col], errors="coerce").ffill().values
    volumes = pd.to_numeric(clean_df[v_col], errors="coerce").fillna(0.0).values
    n = len(prices)

    v_b, v_s, imb = classify_bulk_volume(prices, volumes, window=window)
    vpin_s = calculate_vpin_series(prices, volumes, window=window)

    valid_vpin = vpin_s.dropna().values
    if len(valid_vpin) == 0:
        latest_vpin = 0.0
        mean_vpin = 0.0
        max_vpin = 0.0
    else:
        latest_vpin = float(valid_vpin[-1])
        mean_vpin = float(np.mean(valid_vpin))
        max_vpin = float(np.max(valid_vpin))

    last_b = float(v_b.iloc[-1]) if len(v_b) > 0 else 0.0
    last_s = float(v_s.iloc[-1]) if len(v_s) > 0 else 0.0
    tot_last = last_b + last_s
    if tot_last > 0:
        buy_pct = (last_b / tot_last) * 100.0
        sell_pct = (last_s / tot_last) * 100.0
    else:
        buy_pct = 50.0
        sell_pct = 50.0

    # Categorize toxicity regime
    if latest_vpin >= 0.50:
        regime = "TOXIC_INFORMED_SURGE"
    elif latest_vpin >= 0.35:
        regime = "HIGH_ADVERSE_SELECTION"
    elif latest_vpin >= 0.20:
        regime = "MODERATE_FLOW_IMBALANCE"
    else:
        regime = "LOW_TOXICITY"

    summary = (
        f"Order Flow Toxicity ({ticker}, n={n} sessions, window={window}): "
        f"Latest VPIN = {latest_vpin * 100:.2f}%, Trailing Mean VPIN = {mean_vpin * 100:.2f}%, Peak VPIN = {max_vpin * 100:.2f}%. "
        f"Volume Breakdown: {buy_pct:.1f}% Buyer / {sell_pct:.1f}% Seller. Regime: {regime}."
    )

    return ToxicityReceipt(
        ticker=ticker,
        sample_sessions=n,
        window=window,
        latest_vpin=round(latest_vpin, 4),
        mean_vpin=round(mean_vpin, 4),
        max_vpin=round(max_vpin, 4),
        latest_buyer_volume_pct=round(buy_pct, 2),
        latest_seller_volume_pct=round(sell_pct, 2),
        toxicity_regime=regime,
        summary=summary,
    )
