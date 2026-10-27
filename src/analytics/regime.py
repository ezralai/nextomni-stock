"""Statistical market regime detection and macro environment classification for NextOmni Stock.

Categorizes asset market states into quantitative regimes based on rolling volatility
percentiles, moving average trend structures, and normalized momentum sentiment.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.config import PRICE_COLUMN


@dataclass
class MarketRegimeReceipt:
    """Quantitative market regime diagnostic receipt."""
    regime: str
    regime_label: str
    sentiment_score: float
    volatility_regime: str
    trend_regime: str
    current_price: float
    sma_50: float
    sma_200: Optional[float]
    rolling_vol_annualized: float
    vol_percentile_1y: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_market_regime(df: pd.DataFrame) -> MarketRegimeReceipt:
    """Classify current market regime using retrospective price and volatility signals.

    Regimes:
    - BULL_TREND: Low Volatility + Bullish Trend (Orderly growth, risk-on)
    - EUPHORIC_MOMENTUM: High Volatility + Bullish Trend (Fast trend, speculative momentum)
    - DISTRIBUTION: Low Volatility + Bearish Trend (Slow grind down, quiet selling)
    - PANIC_CRISIS: High Volatility + Bearish Trend (Acute selloff, volatility spike)

    Args:
        df: Historical observations DataFrame sorted chronologically.

    Returns:
        MarketRegimeReceipt containing classified regime and underlying factors.

    Raises:
        ValueError: If df has fewer than 20 observations.
    """
    if len(df) < 20:
        raise ValueError(f"Insufficient observations ({len(df)}) for market regime detection (min 20 required).")

    clean = df.copy().sort_values("session_date", ascending=True).reset_index(drop=True)
    prices = clean[PRICE_COLUMN].astype(float)
    n = len(prices)

    current_price = float(prices.iloc[-1])
    returns = prices.pct_change().dropna()

    # Rolling 20-session volatility annualized
    window_20_returns = returns.tail(20)
    current_vol = float(window_20_returns.std(ddof=1) * math.sqrt(252)) if len(window_20_returns) > 1 else 0.15
    if not math.isfinite(current_vol) or current_vol <= 0:
        current_vol = 0.15

    # Volatility percentile over past up to 252 sessions
    hist_window = min(n, 252)
    rolling_vols = returns.tail(hist_window).rolling(window=min(20, len(returns))).std(ddof=1) * math.sqrt(252)
    valid_vols = rolling_vols.dropna()

    if len(valid_vols) >= 5:
        vol_pctile = float((valid_vols < current_vol).mean() * 100.0)
    else:
        vol_pctile = 50.0

    vol_regime = "HIGH_VOL" if vol_pctile >= 60.0 or current_vol >= 0.28 else "LOW_VOL"

    # Moving average trend analysis
    sma_50 = float(prices.tail(min(50, n)).mean())
    sma_200 = float(prices.tail(200).mean()) if n >= 200 else None

    if sma_200 is not None:
        is_bullish = current_price >= sma_50 and sma_50 >= sma_200 * 0.98
    else:
        is_bullish = current_price >= sma_50

    trend_regime = "BULLISH" if is_bullish else "BEARISH"

    # Map combined regime
    if trend_regime == "BULLISH" and vol_regime == "LOW_VOL":
        regime = "BULL_TREND"
        regime_label = "Low-Vol Trending Bullish (Risk-On)"
    elif trend_regime == "BULLISH" and vol_regime == "HIGH_VOL":
        regime = "EUPHORIC_MOMENTUM"
        regime_label = "High-Vol Momentum (Speculative)"
    elif trend_regime == "BEARISH" and vol_regime == "LOW_VOL":
        regime = "DISTRIBUTION"
        regime_label = "Low-Vol Distribution (Slow Grind Down)"
    else:
        regime = "PANIC_CRISIS"
        regime_label = "High-Vol Panic / Crisis (Risk-Off)"

    # Continuous sentiment score in [-1.0, +1.0]
    trend_dist = (current_price - sma_50) / sma_50
    bounded_trend = math.tanh(trend_dist * 5.0)  # scale in [-1, +1]

    if vol_regime == "HIGH_VOL" and trend_regime == "BEARISH":
        sentiment_score = -0.7 - 0.3 * (vol_pctile / 100.0)
    elif vol_regime == "LOW_VOL" and trend_regime == "BULLISH":
        sentiment_score = 0.5 + 0.5 * bounded_trend
    else:
        sentiment_score = bounded_trend

    sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))

    return MarketRegimeReceipt(
        regime=regime,
        regime_label=regime_label,
        sentiment_score=round(sentiment_score, 3),
        volatility_regime=vol_regime,
        trend_regime=trend_regime,
        current_price=round(current_price, 2),
        sma_50=round(sma_50, 2),
        sma_200=round(sma_200, 2) if sma_200 is not None else None,
        rolling_vol_annualized=round(current_vol * 100.0, 2),
        vol_percentile_1y=round(vol_pctile, 1),
    )
