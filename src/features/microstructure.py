"""Market microstructure liquidity and price impact feature store for NextOmni Stock.

Implements the Amihud (2002) Illiquidity Ratio, Corwin-Schultz (2012) High-Low bid-ask spread
estimator, and Kyle's lambda price impact proxy for multi-asset equity analysis.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import PRICE_COLUMN


@dataclass
class LiquidityReceipt:
    """Quantitative diagnostic receipt summarizing asset market microstructure and liquidity."""
    ticker: str
    sample_size: int
    mean_amihud_illiq: float
    latest_amihud_illiq: float
    mean_corwin_schultz_spread_pct: float
    latest_corwin_schultz_spread_pct: float
    mean_dollar_volume: float
    liquidity_regime: str
    is_liquid: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_amihud_illiquidity(
    returns: pd.Series,
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Calculate the rolling Amihud (2002) Illiquidity Ratio.

    Formula:
        ILLIQ_t = (1 / k) * sum_{i=0}^{k-1} (|r_{t-i}| / DollarVolume_{t-i}) * 10^6
        DollarVolume_t = P_t * V_t

    Higher values indicate greater price impact per dollar traded (lower liquidity).

    Args:
        returns: Daily percentage or log returns (r_t).
        prices: Asset closing or adjusted prices (P_t).
        volumes: Trading volume in shares/contracts (V_t).
        window: Rolling observation window in sessions (default: 21).

    Returns:
        Series of rolling Amihud illiquidity scaled by 10^6.
    """
    dollar_vol = prices.astype(float) * volumes.astype(float)
    safe_dollar_vol = np.maximum(dollar_vol, 1.0)
    abs_ret = np.abs(returns.astype(float))

    daily_illiq = (abs_ret / safe_dollar_vol) * 1e6
    min_p = max(1, min(window, 5))
    rolling_illiq = daily_illiq.rolling(window=window, min_periods=min_p).mean().fillna(0.0)
    return rolling_illiq


def compute_corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Calculate the Corwin & Schultz (2012) High-Low bid-ask spread estimator.

    Uses the variance of 2-day high-low price ratios vs 1-day ratios to estimate
    effective bid-ask spread directly from daily OHLC quotes.

    Formulas:
        beta_t = sum_{j=0}^1 [ln(H_{t-j} / L_{t-j})]^2
        gamma_t = [ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))]^2
        alpha_t = (sqrt(2 * beta_t) - sqrt(beta_t)) / (3 - 2*sqrt(2)) - sqrt(gamma_t / (3 - 2*sqrt(2)))
        Spread_t = 2 * (e^{alpha_t} - 1) / (1 + e^{alpha_t})

    Clamped to [0.0, 1.0] to enforce physical bounds.

    Args:
        high: Daily high prices (H_t).
        low: Daily low prices (L_t).
        window: Rolling smoothing window in sessions (default: 21).

    Returns:
        Series of estimated percentage bid-ask spread in [0, 1].
    """
    h = np.maximum(high.astype(float), 1e-4)
    l = np.maximum(low.astype(float), 1e-4)
    # Ensure high >= low
    h_adj = np.maximum(h, l)

    log_hl = np.log(h_adj / l)
    log_hl_sq = log_hl ** 2

    # 2-day sum of single-day squared ranges
    beta = log_hl_sq + log_hl_sq.shift(1)

    # 2-day combined high and low
    h_2d = np.maximum(h_adj, h_adj.shift(1))
    l_2d = np.minimum(l, l.shift(1))
    gamma = (np.log(h_2d / np.maximum(l_2d, 1e-4))) ** 2

    k = 3.0 - 2.0 * math.sqrt(2.0)
    sqrt_2 = math.sqrt(2.0)

    # alpha estimator
    term1 = (np.sqrt(2.0 * np.maximum(beta, 0.0)) - np.sqrt(np.maximum(beta, 0.0))) / k
    term2 = np.sqrt(np.maximum(gamma / k, 0.0))
    alpha = term1 - term2

    # Spread conversion: 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    exp_alpha = np.exp(np.clip(alpha, -10.0, 10.0))
    raw_spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)

    # Clamp negative values (caused by overnight gap openings) to 0.0
    clamped_spread = np.clip(raw_spread, 0.0, 1.0)

    # Rolling average smoothing
    min_p_cs = max(1, min(window, 5))
    rolling_spread = clamped_spread.rolling(window=window, min_periods=min_p_cs).mean().fillna(0.0)
    return rolling_spread


def compute_kyle_lambda_proxy(
    returns: pd.Series,
    volumes: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Calculate Kyle's lambda price impact proxy (sensitivity per unit volume).

    Args:
        returns: Daily percentage returns.
        volumes: Trading volume.
        window: Rolling observation window in sessions (default: 21).

    Returns:
        Series of price impact sensitivity proxy.
    """
    min_p_k = max(1, min(window, 5))
    mean_v = volumes.rolling(window=window, min_periods=min_p_k).mean().fillna(1.0)
    norm_vol = volumes / np.maximum(mean_v, 1.0)
    abs_ret = np.abs(returns)

    lambda_proxy = abs_ret / np.maximum(norm_vol, 0.01)
    rolling_lambda = lambda_proxy.rolling(window=window, min_periods=min_p_k).mean().fillna(0.0)
    return rolling_lambda


def evaluate_liquidity_profile(
    df: pd.DataFrame,
    ticker: str = "TICKER",
    window: int = 21,
    price_col: str = PRICE_COLUMN,
) -> LiquidityReceipt:
    """Evaluate market microstructure liquidity and price impact diagnostics for an asset.

    Args:
        df: DataFrame containing at least ['session_date', 'volume', price_col] and optionally ['high', 'low'].
        ticker: Asset ticker symbol.
        window: Rolling observation window (default: 21).
        price_col: Price column name (default: adj_close).

    Returns:
        LiquidityReceipt with Amihud illiquidity, Corwin-Schultz spread %, and regime status.
    """
    n = len(df)
    if n < 15:
        return LiquidityReceipt(
            ticker=ticker,
            sample_size=n,
            mean_amihud_illiq=0.0,
            latest_amihud_illiq=0.0,
            mean_corwin_schultz_spread_pct=0.0,
            latest_corwin_schultz_spread_pct=0.0,
            mean_dollar_volume=0.0,
            liquidity_regime="INSUFFICIENT_DATA",
            is_liquid=False,
            summary=f"Insufficient sample observations ({n}/15 required) for liquidity audit.",
        )

    clean = df.sort_values("session_date").reset_index(drop=True)
    prices = clean[price_col].astype(float)
    returns = prices.pct_change().fillna(0.0)
    volumes = clean["volume"].astype(float) if "volume" in clean.columns else pd.Series(10000.0, index=clean.index)

    high = clean["high"].astype(float) if "high" in clean.columns else prices * 1.005
    low = clean["low"].astype(float) if "low" in clean.columns else prices * 0.995

    amihud = compute_amihud_illiquidity(returns, prices, volumes, window=window)
    cs_spread = compute_corwin_schultz_spread(high, low, window=window)
    dollar_vols = prices * volumes

    mean_illiq = float(amihud.mean())
    latest_illiq = float(amihud.iloc[-1])
    mean_spread_pct = float(cs_spread.mean() * 100.0)
    latest_spread_pct = float(cs_spread.iloc[-1] * 100.0)
    mean_dv = float(dollar_vols.mean())

    # Liquidity regime classification
    if latest_spread_pct < 0.50 and latest_illiq < 1.0:
        regime = "HIGH_LIQUIDITY"
        is_liquid = True
    elif latest_spread_pct < 1.50 and latest_illiq < 5.0:
        regime = "MODERATE_LIQUIDITY"
        is_liquid = True
    else:
        regime = "LOW_LIQUIDITY_DRYUP"
        is_liquid = False

    summary = (
        f"{ticker} exhibits {regime.replace('_', ' ')}: "
        f"Corwin-Schultz Spread={latest_spread_pct:.2f}%, "
        f"Amihud Illiquidity={latest_illiq:.4f}, "
        f"Mean Daily Turnover=${mean_dv:,.0f}."
    )

    return LiquidityReceipt(
        ticker=ticker,
        sample_size=n,
        mean_amihud_illiq=round(mean_illiq, 4),
        latest_amihud_illiq=round(latest_illiq, 4),
        mean_corwin_schultz_spread_pct=round(mean_spread_pct, 2),
        latest_corwin_schultz_spread_pct=round(latest_spread_pct, 2),
        mean_dollar_volume=round(mean_dv, 2),
        liquidity_regime=regime,
        is_liquid=is_liquid,
        summary=summary,
    )
