"""Richard Roll (1984) Effective Bid-Ask Spread & Composite Microstructure Friction Engine.

Implements Richard Roll's (1984) serial covariance implicit bid-ask spread model,
Chu, Stivers & Truong (2011) signed extended Roll estimator, and composite
transaction execution friction metrics (in basis points) for market microstructure analysis.

References:
    Roll, R. (1984). A simple implicit measure of the effective bid-ask spread
    in an efficient market. The Journal of Finance, 39(4), 1127-1139.
    Chu, Q. C., Stivers, C. T., & Truong, N. (2011). The Roll (1984) spread estimator:
    What does it really measure?. Journal of Financial Markets, 14(4), 603-625.
    Hasbrouck, J. (2009). Trading costs and returns for US equities: Estimating
    effective spreads from daily data. The Journal of Finance, 64(3), 1445-1477.
    Goyenko, R. Y., Holden, C. W., & Trzcinka, C. A. (2009). Do liquidity measures
    measure liquidity?. Journal of Financial Economics, 92(2), 153-181.
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

from src.features.microstructure import compute_corwin_schultz_spread
from src.security.validators import assert_finite_number, validate_ticker

ROLL_SIGNING_KEY = b"nextomni_roll_effective_spread_secret_key_v1"


@dataclass
class EffectiveSpreadReceipt:
    """Cryptographic audit receipt for Roll effective spread and execution friction."""

    ticker: str
    sample_sessions: int
    window: int
    mean_price: float
    serial_autocovariance: float  # gamma_1 = Cov(Delta P_t, Delta P_{t-1})
    roll_spread_dollars: float  # Classical Roll spread in dollars: 2 * sqrt(max(0, -gamma_1))
    roll_spread_pct: float  # (Roll spread / mean price) * 100%
    signed_roll_spread_pct: float  # Chu, Stivers & Truong (2011) signed continuous spread %
    corwin_schultz_spread_pct: float  # Corwin-Schultz (2012) High-Low spread %
    composite_round_trip_bps: float  # Inferred round-trip execution friction in basis points
    has_positive_autocovariance: bool  # True if gamma_1 > 0 (trending / momentum regime)
    liquidity_tier: str  # ULTRA_LIQUID, LIQUID, MODERATE_FRICTION, or ILLIQUID
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core microstructure metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.window}:{self.mean_price:.4f}:"
            f"{self.serial_autocovariance:.6f}:{self.roll_spread_pct:.4f}:"
            f"{self.composite_round_trip_bps:.2f}:{self.liquidity_tier}:"
            f"{self.has_positive_autocovariance}"
        )
        return hmac.new(ROLL_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_roll_serial_covariance(price_changes: np.ndarray) -> float:
    """Calculate first-order serial autocovariance of price changes gamma_1.

    Formula:
        gamma_1 = (1 / (M - 1)) * sum_{t=2}^M (Delta P_t - mean(Delta P)) * (Delta P_{t-1} - mean(Delta P))
    """
    dp = np.asarray(price_changes, dtype=float)
    if len(dp) < 2:
        return 0.0

    mean_dp = float(np.mean(dp))
    centered = dp - mean_dp
    cov = float(np.sum(centered[1:] * centered[:-1]) / float(len(dp) - 1))
    return cov


def compute_roll_effective_spread(
    prices: Union[pd.Series, np.ndarray, List[float]],
) -> Tuple[float, float, float, bool]:
    """Calculate Richard Roll's (1984) effective bid-ask spread and signed extension.

    Formulas:
        Delta P_t = P_t - P_{t-1}
        gamma_1 = Cov(Delta P_t, Delta P_{t-1})
        S_{Roll} = 2 * sqrt(max(0, -gamma_1))
        S_{signed} = sign(-gamma_1) * 2 * sqrt(|gamma_1|)
        s_{Roll, pct} = (S_{Roll} / mean(P)) * 100%

    Args:
        prices: Sequence of asset closing prices.

    Returns:
        Tuple of (roll_spread_dollars, roll_spread_pct, signed_roll_spread_pct, has_positive_autocovariance).
    """
    p = np.asarray(prices, dtype=float)
    if len(p) < 3:
        return 0.0, 0.0, 0.0, False

    mean_p = max(1e-4, float(np.mean(p)))
    dp = np.diff(p)
    gamma_1 = compute_roll_serial_covariance(dp)

    has_positive_cov = bool(gamma_1 > 0.0)

    # Classical Roll estimator (bounded at 0 when gamma_1 >= 0)
    if gamma_1 < 0.0:
        roll_dollars = 2.0 * math.sqrt(-gamma_1)
    else:
        roll_dollars = 0.0

    roll_pct = (roll_dollars / mean_p) * 100.0

    # Chu, Stivers & Truong (2011) signed extended Roll estimator
    sign_val = -1.0 if gamma_1 > 0.0 else 1.0
    signed_dollars = sign_val * 2.0 * math.sqrt(abs(gamma_1))
    signed_pct = (signed_dollars / mean_p) * 100.0

    return float(roll_dollars), float(roll_pct), float(signed_pct), has_positive_cov


def evaluate_effective_spread_profile(
    ticker: str,
    prices: Union[pd.Series, np.ndarray, List[float]],
    high: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    low: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    volumes: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    window: int = 21,
) -> EffectiveSpreadReceipt:
    """Evaluate comprehensive Roll (1984) effective spread and composite transaction friction.

    Combines the Roll serial covariance spread, Corwin-Schultz (2012) high-low spread,
    and inferred round-trip execution cost in basis points.

    Args:
        ticker: Validated stock ticker symbol.
        prices: Sequence of closing prices.
        high: Optional sequence of daily high prices.
        low: Optional sequence of daily low prices.
        volumes: Optional sequence of daily trading volumes.
        window: Lookback observation window in sessions (default: 21).

    Returns:
        EffectiveSpreadReceipt with granular microstructure diagnostics.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(window, "window")
    if window < 10:
        raise ValueError(f"Lookback window must be at least 10 sessions, got {window}")

    if isinstance(prices, pd.Series):
        s_prices = prices.dropna().astype(float)
    else:
        s_prices = pd.Series(prices).dropna().astype(float)

    if len(s_prices) < window + 1:
        raise ValueError(
            f"Insufficient price history for {clean_ticker}: got {len(s_prices)} prices, need at least {window + 1}"
        )

    # Window slice
    win_p = s_prices.iloc[-window:].values
    mean_price = max(1e-4, float(np.mean(win_p)))
    dp = np.diff(win_p)
    gamma_1 = compute_roll_serial_covariance(dp)

    roll_dlr, roll_pct, signed_pct, pos_cov = compute_roll_effective_spread(win_p)

    # Corwin-Schultz (2012) High-Low Spread
    cs_spread_pct = 0.0
    if high is not None and low is not None:
        s_high = pd.Series(high).dropna().astype(float).iloc[-window:]
        s_low = pd.Series(low).dropna().astype(float).iloc[-window:]
        if len(s_high) >= 2 and len(s_low) >= 2:
            cs_series = compute_corwin_schultz_spread(s_high, s_low, window=min(window, len(s_high)))
            if not cs_series.empty:
                cs_spread_pct = float(cs_series.iloc[-1]) * 100.0

    # Composite Round-Trip Inferred Friction in Basis Points (1% = 100 bps)
    # If Corwin-Schultz is available, average with Roll; otherwise use Roll
    if cs_spread_pct > 0.0 and roll_pct > 0.0:
        composite_pct = 0.5 * (roll_pct + cs_spread_pct)
    elif cs_spread_pct > 0.0:
        composite_pct = cs_spread_pct
    else:
        composite_pct = max(0.01, roll_pct)

    composite_bps = float(composite_pct * 100.0)

    # Classify Liquidity Tier
    if composite_bps < 15.0:
        liquidity_tier = "ULTRA_LIQUID"
    elif composite_bps < 40.0:
        liquidity_tier = "LIQUID"
    elif composite_bps < 100.0:
        liquidity_tier = "MODERATE_FRICTION"
    else:
        liquidity_tier = "ILLIQUID"

    regime_desc = "Positive Autocovariance / Trend Regime" if pos_cov else "Negative Autocovariance / Dealer Bounce"
    summary = (
        f"Roll (1984) Microstructure Analysis for {clean_ticker} ({len(s_prices)} sessions, {window}D window): "
        f"Roll Spread: {roll_pct:.2f}% (${roll_dlr:.3f}, {regime_desc}). "
        f"Corwin-Schultz Spread: {cs_spread_pct:.2f}%. Composite Slippage: {composite_bps:.1f} bps "
        f"(Tier: {liquidity_tier})."
    )

    receipt = EffectiveSpreadReceipt(
        ticker=clean_ticker,
        sample_sessions=len(s_prices),
        window=window,
        mean_price=mean_price,
        serial_autocovariance=gamma_1,
        roll_spread_dollars=roll_dlr,
        roll_spread_pct=roll_pct,
        signed_roll_spread_pct=signed_pct,
        corwin_schultz_spread_pct=cs_spread_pct,
        composite_round_trip_bps=composite_bps,
        has_positive_autocovariance=pos_cov,
        liquidity_tier=liquidity_tier,
        summary=summary,
    )

    return receipt
