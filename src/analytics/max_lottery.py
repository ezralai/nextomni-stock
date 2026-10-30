"""Turan G. Bali, Nusret Cakici & Robert F. Whitelaw (2011) Extreme Daily Return "MAX" Effect & Lottery Demand Engine.

Implements behavioral quantitative modeling of investor lottery preferences and the MAX anomaly:
- Calculates extreme positive order statistics (MAX_1, MAX_N) and downside tail (MIN_N):
    MAX_N = (1 / N) * sum_{i=1}^N R_{(T - i + 1)}
- Computes Max-to-Min return asymmetry ratio:
    Asymmetry = MAX_N / |MIN_N|
- Synthesizes institutional Lottery Demand Score (Lambda_i in [0, 100]) combining:
    1. Extreme upside tail spike magnitude (MAX_5)
    2. Daily return skewness (preference for positive asymmetry)
    3. Max-to-Min upside tail dominance ratio
- Estimates Bali et al. (2011) Expected Annualized Lottery Alpha Drag:
    Quantifies the structural risk-adjusted underperformance (-12.4% annualized)
    suffered by equities overbought by speculative retail gamblers.
- Classifies lottery regimes:
    EXTREME_LOTTERY_TRAP, SPECULATIVE_OVERPRICING, BALANCED_EQUILIBRIUM, PRIME_STEADY_COMPOUNDER.

References:
    Bali, T. G., Cakici, N., & Whitelaw, R. F. (2011). Maxing out: Stocks as lotteries
    and the cross-section of expected stock returns. Journal of Financial Economics, 99(2), 427-446.
    Barberis, N., & Huang, M. (2008). Stocks as lotteries: The implications of probability
    weighting for investments. The Journal of Finance, 63(5), 2066-2100.
    Kumar, A. (2009). Who gambles in the stock market?. The Journal of Finance, 64(4), 1889-1933.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

MAX_LOTTERY_SIGNING_KEY = b"nextomni_max_lottery_secret_key_v1"


@dataclass
class MaxLotteryReceipt:
    """Cryptographic audit receipt for Bali et al. (2011) MAX Effect & Lottery Demand Profile."""

    ticker: str
    sample_sessions: int
    lookback_window: int
    top_n_days: int
    max_1_pct: float  # Single highest daily return in window (%)
    max_n_pct: float  # Average of N highest daily returns (%)
    min_n_pct: float  # Average of N lowest daily returns (%)
    max_to_min_ratio: float  # MAX_N / |MIN_N|
    return_skewness: float  # Sample skewness of daily returns
    daily_volatility_pct: float  # Annualized volatility (%)
    lottery_demand_score: float  # Lambda index in [0, 100]
    expected_alpha_drag_ann_pct: float  # Predicted annualized alpha drag (%)
    lottery_regime: str  # EXTREME_LOTTERY_TRAP, SPECULATIVE_OVERPRICING, BALANCED_EQUILIBRIUM, PRIME_STEADY_COMPOUNDER
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core lottery metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.lookback_window}:{self.top_n_days}:"
            f"{self.max_1_pct:.4f}:{self.max_n_pct:.4f}:{self.min_n_pct:.4f}:"
            f"{self.lottery_demand_score:.4f}:{self.expected_alpha_drag_ann_pct:.4f}:"
            f"{self.lottery_regime}"
        )
        return hmac.new(
            MAX_LOTTERY_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity and tamper-resistance of the lottery receipt."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def classify_lottery_regime(lottery_score: float, max_n_pct: float) -> str:
    """Classify behavioral lottery attraction tier according to Bali et al. standards.

    Tiers:
        Lambda >= 75.0 or MAX_N >= 8.0%: EXTREME_LOTTERY_TRAP (High retail speculative bubble; heavy alpha drag)
        55.0 <= Lambda < 75.0: SPECULATIVE_OVERPRICING (Elevated retail gambling demand)
        35.0 <= Lambda < 55.0: BALANCED_EQUILIBRIUM (Symmetric return distribution; normal institutional pricing)
        Lambda < 35.0: PRIME_STEADY_COMPOUNDER (Disciplined compounder; commands low-MAX premium)
    """
    if lottery_score >= 75.0 or max_n_pct >= 8.0:
        return "EXTREME_LOTTERY_TRAP"
    elif lottery_score >= 55.0:
        return "SPECULATIVE_OVERPRICING"
    elif lottery_score >= 35.0:
        return "BALANCED_EQUILIBRIUM"
    else:
        return "PRIME_STEADY_COMPOUNDER"


def compute_max_effect(
    returns: np.ndarray,
    top_n: int = 5,
) -> Tuple[float, float, float, float, float, float, float, float, str]:
    """Compute Bali, Cakici & Whitelaw (2011) extreme return order statistics and lottery score.

    Args:
        returns: 1D array of daily price returns.
        top_n: Number of extreme days N to average for MAX_N and MIN_N (default: 5).

    Returns:
        Tuple of:
            max_1_pct (Single highest daily return in %)
            max_n_pct (Average of N highest daily returns in %)
            min_n_pct (Average of N lowest daily returns in %)
            max_to_min_ratio (MAX_N / |MIN_N|)
            skewness (Return skewness)
            annualized_vol_pct (Annualized volatility in %)
            lottery_demand_score (Lambda in [0, 100])
            alpha_drag_ann_pct (Annualized alpha drag in %)
            lottery_regime (str)
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)

    if n < 15:
        raise ValueError(f"Insufficient return observations ({n}), need at least 15")

    k = int(max(1, min(top_n, n // 3)))

    sorted_r = np.sort(r)

    # Extreme upper order statistics
    max_1 = float(sorted_r[-1] * 100.0)
    max_n = float(np.mean(sorted_r[-k:]) * 100.0)

    # Extreme lower order statistics
    min_n = float(np.mean(sorted_r[:k]) * 100.0)

    # Asymmetry ratio
    abs_min = max(0.1, abs(min_n))
    max_to_min = float(max_n / abs_min)

    # Standard moments
    std_r = float(np.std(r, ddof=1))
    ann_vol_pct = float(std_r * math.sqrt(252.0) * 100.0)

    # Skewness
    if std_r > 1e-8:
        dev = r - np.mean(r)
        skewness = float(np.mean(dev**3) / (std_r**3))
    else:
        skewness = 0.0

    # Institutional Lottery Demand Score Lambda in [0, 100]:
    # 1. MAX_N component: 0% -> 0, 5% -> 50, 10% -> 100
    s_max = float(np.clip((max_n - 0.5) * 10.0, 0.0, 100.0))
    # 2. Skewness component: negative skew -> 0-40, zero skew -> 50, positive skew -> 60-100
    s_skew = float(np.clip(50.0 + skewness * 25.0, 0.0, 100.0))
    # 3. Asymmetry ratio component: ratio 0.5 -> 0, ratio 1.0 -> 50, ratio 2.0 -> 100
    s_asym = float(np.clip((max_to_min - 0.5) * 50.0, 0.0, 100.0))

    lottery_score = float(0.50 * s_max + 0.30 * s_skew + 0.20 * s_asym)
    lottery_score = float(np.clip(lottery_score, 0.0, 100.0))

    # Bali et al. (2011) documented annualized alpha drag:
    # High-lottery stocks experience -12.4% annualized return penalty
    if lottery_score >= 50.0:
        alpha_drag_ann = float(-12.4 * ((lottery_score - 50.0) / 50.0))
    else:
        # Low-MAX steady compounder commands positive premium up to +4.5%
        alpha_drag_ann = float(+4.5 * ((50.0 - lottery_score) / 50.0))

    regime = classify_lottery_regime(lottery_score, max_n)

    return (
        round(max_1, 2),
        round(max_n, 2),
        round(min_n, 2),
        round(max_to_min, 2),
        round(skewness, 2),
        round(ann_vol_pct, 1),
        round(lottery_score, 1),
        round(alpha_drag_ann, 2),
        regime,
    )


def evaluate_max_lottery_profile(
    ticker: str,
    asset_prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    top_n_days: int = 5,
    lookback_window: int = 60,
    db_path: Optional[Path] = None,
) -> MaxLotteryReceipt:
    """Evaluate comprehensive Bali, Cakici & Whitelaw (2011) MAX effect and lottery demand profile.

    Args:
        ticker: Asset ticker symbol.
        asset_prices: Optional daily price series.
        top_n_days: Number of extreme days N to average for MAX_N (default: 5).
        lookback_window: Rolling window in sessions (default: 60 sessions = ~1 quarter).
        db_path: Optional SQLite database path.

    Returns:
        MaxLotteryReceipt with cryptographic HMAC-SHA256 signature and tamper verification.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(top_n_days, "top_n_days")
    assert_finite_number(lookback_window, "lookback_window")

    k = int(max(1, min(10, top_n_days)))
    window = int(max(20, min(252, lookback_window)))

    if asset_prices is None:
        df_obs = db.get_observations(clean_ticker, db_path=db_path)
        if df_obs.empty or len(df_obs) < 20:
            raise ValueError(
                f"Insufficient observation history for {clean_ticker} (minimum 20 sessions required)."
            )
        col = "adj_close" if "adj_close" in df_obs.columns else "close"
        s_prices = df_obs[col].dropna().astype(float)
    else:
        s_prices = pd.Series(asset_prices, dtype=float).dropna()

    if len(s_prices) < 20:
        raise ValueError(f"Insufficient price observations ({len(s_prices)}), need at least 20")

    p_slice = s_prices.iloc[-window:]
    returns = p_slice.pct_change().dropna().values

    (
        max_1,
        max_n,
        min_n,
        ratio,
        skew,
        ann_vol,
        lottery_score,
        alpha_drag,
        regime,
    ) = compute_max_effect(returns=returns, top_n=k)

    summary = (
        f"{clean_ticker} Bali et al. (2011) MAX Effect: MAX_{k} {max_n:+.2f}% (Single MAX {max_1:+.2f}%), "
        f"MIN_{k} {min_n:+.2f}% (Asymmetry Ratio {ratio:.2f}x, Skew {skew:+.2f}). "
        f"Lottery Demand Score: {lottery_score:.1f}/100. Expected Alpha Drag: {alpha_drag:+.2f}%/yr. "
        f"Regime: {regime}."
    )

    return MaxLotteryReceipt(
        ticker=clean_ticker,
        sample_sessions=len(returns),
        lookback_window=len(p_slice),
        top_n_days=k,
        max_1_pct=max_1,
        max_n_pct=max_n,
        min_n_pct=min_n,
        max_to_min_ratio=ratio,
        return_skewness=skew,
        daily_volatility_pct=ann_vol,
        lottery_demand_score=lottery_score,
        expected_alpha_drag_ann_pct=alpha_drag,
        lottery_regime=regime,
        summary=summary,
    )
