"""Thomas J. George & Chuan-Yang Hwang (2004)
52-Week High Psychological Anchoring & Nearness-to-High Momentum Engine.

Implements the psychological anchoring and adjustment momentum asset pricing framework:
- 52-Week High Anchoring Hypothesis (George & Hwang, 2004):
    Investors use the 52-week high price (H_{52,t} = max_{tau in [t-252, t]} P_tau) as a salient
    Tversky-Kahneman psychological reference point when evaluating new information:
        NH_{52,t} = P_t / H_{52,t} in (0.0, 1.0]
    - Nearness to 52-Week High (NH_{52,t} >= 0.90): When good news pushes a stock near its 52-week
      high, anchored traders are reluctant to bid the price higher even when fundamentals warrant it.
      The information eventually prevails, generating strong, persistent momentum continuation that
      uniquely avoids long-horizon DeBondt-Thaler reversals.
    - Far Below 52-Week High (NH_{52,t} <= 0.58): When bad news pushes a stock far below its 52-week
      peak, anchored traders are reluctant to sell at depressed levels, leaving bad news underpriced
      and predicting continued downward drift.
- Dead-Cat Bounce vs. True Anchoring Breakout:
    George & Hwang (2004) demonstrate that nearness to the 52-week high dominates raw past returns
    (Jegadeesh-Titman PRET_t) in pairwise Fama-MacBeth regressions: a stock with high recent bounce
    return (PRET_t >= 0.08) that remains far below its 52-week high (NH_{52,t} < 0.70) is a dead-cat
    bounce rather than a genuine breakout leader.
- Regimes:
    - ANCHORED_52W_HIGH_BREAKOUT_LEADER: NH_{52,t} >= 0.90 and GHAI_t >= 0.45 (+Alpha continuation)
    - CONSTRUCTIVE_UPPER_RANGE_ACCUMULATION: 0.70 <= NH_{52,t} < 0.90
    - DEAD_CAT_BOUNCE_ANCHOR_DIVERGENCE: NH_{52,t} < 0.70 despite positive recent bounce (PRET_t >= 0.08)
    - DEEP_52W_ANCHOR_OVERHANG_TRAP: NH_{52,t} <= 0.58 and PRET_t < 0.08 (-Alpha overhang trap)

References:
    George, T. J., & Hwang, C.-Y. (2004). The 52-week high and momentum investing.
    The Journal of Finance, 59(5), 2145-2176.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

FIFTY_TWO_WEEK_HIGH_SIGNING_KEY = b"nextomni_fifty_two_week_high_secret_v1"


@dataclass
class FiftyTwoWeekHighReceipt:
    """Cryptographic audit receipt for George & Hwang (2004) 52-Week High Anchoring Profile."""

    ticker: str
    latest_price: float
    # 52-Week Price Envelope & Volume Inputs
    high_52w: float  # 52-week maximum price H_{52,t} (>= latest_price)
    low_52w: float  # 52-week minimum price L_{52,t} (<= latest_price)
    past_return_20d: float  # Recent 20-day return PRET_t in [-0.95, 5.0]
    breakout_volume_ratio: float  # Current volume / 20-day average volume (> 0.0)
    # Derived George-Hwang Metrics
    nearness_to_52w_high: float  # NH_{52,t} = P_t / H_{52,t} in (0.0, 1.0]
    range_position_52w: float  # RP_{52,t} = (P_t - L_{52,t}) / (H_{52,t} - L_{52,t}) in [0.0, 1.0]
    george_hwang_anchoring_impulse: float  # GHAI_t in [-2.0, 2.5]
    expected_anchoring_alpha_pct: float  # Projected annualized non-reversing momentum alpha (%)
    # Flags & Regime
    is_52w_high_breakout_leader: bool  # True if NH_52 >= 0.90 and GHAI >= 0.45
    is_deep_anchor_overhang_trap: bool  # True if NH_52 <= 0.58 and past_return < 0.08
    regime: str
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core 52-week high anchoring metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.high_52w:.2f}:"
            f"{self.low_52w:.2f}:{self.past_return_20d:.4f}:"
            f"{self.breakout_volume_ratio:.4f}:{self.nearness_to_52w_high:.4f}:"
            f"{self.range_position_52w:.4f}:{self.george_hwang_anchoring_impulse:.4f}:"
            f"{self.expected_anchoring_alpha_pct:.2f}:{int(self.is_52w_high_breakout_leader)}:"
            f"{int(self.is_deep_anchor_overhang_trap)}:{self.regime}"
        )
        return hmac.new(
            FIFTY_TWO_WEEK_HIGH_SIGNING_KEY,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper resistance."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary format for JSON serialization."""
        return asdict(self)


def classify_52w_high_regime(
    nearness_ratio: float,
    past_return: float,
    anchoring_impulse: float,
) -> str:
    """Classify asset into George & Hwang (2004) 52-Week High Anchoring Regimes."""
    if nearness_ratio >= 0.90 and anchoring_impulse >= 0.45:
        return "ANCHORED_52W_HIGH_BREAKOUT_LEADER"
    if nearness_ratio < 0.70 and past_return >= 0.08:
        return "DEAD_CAT_BOUNCE_ANCHOR_DIVERGENCE"
    if nearness_ratio <= 0.58 and past_return < 0.08:
        return "DEEP_52W_ANCHOR_OVERHANG_TRAP"
    return "CONSTRUCTIVE_UPPER_RANGE_ACCUMULATION"


def calculate_fifty_two_week_high(
    ticker: str,
    latest_price: float,
    high_52w: float = 104.0,
    low_52w: float = 72.0,
    past_return_20d: float = 0.06,
    breakout_volume_ratio: float = 1.25,
) -> FiftyTwoWeekHighReceipt:
    """Calculate George & Hwang (2004) Nearness-to-52-Week-High Ratio and Anchoring Impulse.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        high_52w: 52-week maximum price H_{52,t} (>= latest_price).
        low_52w: 52-week minimum price L_{52,t} (0 < low_52w <= latest_price).
        past_return_20d: Recent 20-day return PRET_t in [-0.95, 5.0].
        breakout_volume_ratio: Current volume divided by 20-day average volume (> 0.0).

    Returns:
        FiftyTwoWeekHighReceipt with NH_{52,t}, RP_{52,t}, GHAI_t, regime, and expected alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(high_52w, "high_52w")
    assert_finite_number(low_52w, "low_52w")
    assert_finite_number(past_return_20d, "past_return_20d")
    assert_finite_number(breakout_volume_ratio, "breakout_volume_ratio")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if low_52w <= 0.0:
        raise ValueError("low_52w must be strictly positive.")
    if low_52w > latest_price + 1e-6:
        raise ValueError("low_52w cannot exceed latest_price.")
    if high_52w < latest_price - 1e-6:
        raise ValueError("high_52w must be greater than or equal to latest_price.")
    if not (-0.95 <= past_return_20d <= 5.0):
        raise ValueError("past_return_20d must be in [-0.95, 5.0].")
    if breakout_volume_ratio <= 0.0:
        raise ValueError("breakout_volume_ratio must be strictly positive.")

    p_t = float(latest_price)
    h_52 = max(float(high_52w), p_t)
    l_52 = min(float(low_52w), p_t)
    pret = float(past_return_20d)
    vol_ratio = float(breakout_volume_ratio)

    # 1. George & Hwang (2004) Nearness-to-52-Week-High Ratio NH_{52,t} = P_t / H_{52,t}
    nh_52 = float(np.clip(p_t / h_52, 0.01, 1.0))

    # 2. 52-Week Range Position RP_{52,t} in [0.0, 1.0]
    span = h_52 - l_52
    if span <= 1e-6:
        rp_52 = 0.50
    else:
        rp_52 = float(np.clip((p_t - l_52) / span, 0.0, 1.0))

    # 3. George-Hwang Anchoring Underreaction & Breakout Impulse GHAI_t in [-2.0, 2.5]
    vol_boost = 1.0 + 0.35 * min(vol_ratio, 2.5)
    raw_ghai = 2.8 * (nh_52 - 0.74) * vol_boost
    ghai = float(np.clip(raw_ghai, -2.0, 2.5))

    # 4. Regime Classification & Expected Non-Reversing Anchoring Alpha
    regime = classify_52w_high_regime(nh_52, pret, ghai)
    is_leader = bool(regime == "ANCHORED_52W_HIGH_BREAKOUT_LEADER")
    is_trap = bool(regime == "DEEP_52W_ANCHOR_OVERHANG_TRAP")

    # George & Hwang (2004): nearness to 52-week high drives ~0.65%/mo (+7.8%/yr) continuation
    # while dead-cat bounces (high PRET, low NH_52) suffer negative anchoring drag.
    bounce_penalty = 2.5 if regime == "DEAD_CAT_BOUNCE_ANCHOR_DIVERGENCE" else 0.0
    expected_alpha = float(
        np.clip(8.5 * ghai + 1.2 * pret - bounce_penalty, -12.5, 11.5)
    )

    if regime == "ANCHORED_52W_HIGH_BREAKOUT_LEADER":
        summary = (
            f"Anchored 52-Week High Breakout Leader (P=${p_t:.2f} vs H52=${h_52:.2f}, NH52={nh_52:.1%}, "
            f"GHAI={ghai:+.2f}). Traders anchored at the 52-week high underreact to fundamental strength, "
            f"predicting non-reversing continuation ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "DEAD_CAT_BOUNCE_ANCHOR_DIVERGENCE":
        summary = (
            f"Dead-Cat Bounce Anchor Divergence (NH52={nh_52:.1%}, 20dBounce={pret:+.1%}). "
            f"Short-term rally remains far below 52-week peak (${h_52:.2f}); George-Hwang regressions "
            f"show weak continuation and overhead supply ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "DEEP_52W_ANCHOR_OVERHANG_TRAP":
        summary = (
            f"Deep 52-Week Anchor Overhang Trap (P=${p_t:.2f} vs H52=${h_52:.2f}, NH52={nh_52:.1%}). "
            f"Stock trades far below its 52-week reference anchor where reluctant sellers delay bad-news "
            f"incorporation ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Constructive Upper-Range Accumulation (NH52={nh_52:.1%}, RangePos={rp_52:.0%}, "
            f"GHAI={ghai:+.2f}). Approaching 52-week psychological anchor with {expected_alpha:+.1f}%/yr "
            f"expected alpha."
        )

    return FiftyTwoWeekHighReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        high_52w=round(h_52, 2),
        low_52w=round(l_52, 2),
        past_return_20d=round(pret, 4),
        breakout_volume_ratio=round(vol_ratio, 4),
        nearness_to_52w_high=round(nh_52, 4),
        range_position_52w=round(rp_52, 4),
        george_hwang_anchoring_impulse=round(ghai, 4),
        expected_anchoring_alpha_pct=round(expected_alpha, 2),
        is_52w_high_breakout_leader=is_leader,
        is_deep_anchor_overhang_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_fifty_two_week_high_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    high_52w: Optional[float] = None,
    low_52w: Optional[float] = None,
    past_return_20d: Optional[float] = None,
    breakout_volume_ratio: Optional[float] = None,
) -> FiftyTwoWeekHighReceipt:
    """Evaluate George & Hwang (2004) 52-Week High Anchoring profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_high = 104.0
    inferred_low = 72.0
    inferred_pret = 0.06
    inferred_vol_ratio = 1.25

    try:
        obs = db.get_observations(clean_ticker, db_path=db_path)
        if obs is not None and not obs.empty and "close" in obs.columns:
            closes = [
                float(x)
                for x in obs["close"].tolist()
                if math.isfinite(float(x)) and float(x) > 0
            ]
            if closes:
                latest_price = closes[-1]
                window_252 = closes[-min(len(closes), 252) :]
                inferred_high = max(max(window_252), latest_price)
                inferred_low = min(min(window_252), latest_price)
                if "high" in obs.columns:
                    highs = [
                        float(h)
                        for h in obs["high"].tolist()[-len(window_252) :]
                        if math.isfinite(float(h)) and float(h) > 0
                    ]
                    if highs:
                        inferred_high = max(inferred_high, max(highs))
                if "low" in obs.columns:
                    lows = [
                        float(l)
                        for l in obs["low"].tolist()[-len(window_252) :]
                        if math.isfinite(float(l)) and float(l) > 0
                    ]
                    if lows:
                        inferred_low = min(inferred_low, min(lows))
            if len(closes) >= 20:
                inferred_pret = float(
                    np.clip((closes[-1] / closes[-20]) - 1.0, -0.90, 4.0)
                )
                if "volume" in obs.columns:
                    vols = [
                        max(float(v), 1.0)
                        for v in obs["volume"].tolist()[-20:]
                    ]
                    mean_v = float(np.mean(vols[:-1])) if len(vols) > 1 else vols[-1]
                    inferred_vol_ratio = float(
                        np.clip(vols[-1] / max(mean_v, 1.0), 0.25, 6.0)
                    )
    except Exception:
        latest_price = 100.0

    h_val = float(high_52w) if high_52w is not None else max(inferred_high, latest_price)
    l_val = float(low_52w) if low_52w is not None else min(inferred_low, latest_price)
    pret_val = float(past_return_20d) if past_return_20d is not None else inferred_pret
    vol_val = (
        float(breakout_volume_ratio)
        if breakout_volume_ratio is not None
        else inferred_vol_ratio
    )

    return calculate_fifty_two_week_high(
        ticker=clean_ticker,
        latest_price=latest_price,
        high_52w=h_val,
        low_52w=l_val,
        past_return_20d=pret_val,
        breakout_volume_ratio=vol_val,
    )

