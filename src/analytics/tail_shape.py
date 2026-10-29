"""Robust Quantile-Based Tail Shape, Moors Kurtosis & Asymmetric Crash Risk Engine.

Implements Hogg (1974), Moors (1988), and Bowley-Kelly quantile-based estimators to
quantify return distribution fat-tailedness, tail asymmetry, and downside crash
susceptibility without requiring higher-order moment existence.

References:
    Moors, J. J. A. (1988). A quantile alternative for kurtosis.
    The Statistician, 37(1), 25-32.
    Hogg, R. V. (1974). Adaptive robust procedures: A review.
    Journal of the American Statistical Association, 69(348), 909-923.
    Groeneveld, R. A., & Meeden, G. (1984). Measuring skewness and kurtosis.
    The Statistician, 33(4), 391-399.
"""

import hashlib
import hmac
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.db import get_observations
from src.security.validators import assert_finite_number, validate_ticker

TAIL_SHAPE_SIGNING_KEY = b"nextomni_tail_shape_audit_secret_key_v1"

# Standard normal reference constants
GAUSSIAN_MOORS_KURTOSIS = 1.233
GAUSSIAN_TAIL_RATIO = 1.0


@dataclass
class TailShapeReceipt:
    """Cryptographic audit receipt for robust quantile tail shape analysis."""

    ticker: str
    sample_sessions: int
    rolling_window: int
    moors_quantile_kurtosis: float
    moors_excess_kurtosis: float  # Moors - 1.233
    bowley_quantile_skewness: float  # [-1.0, 1.0]
    tail_asymmetry_ratio: float  # Left 1% depth / Right 1% depth
    hogg_tail_weight: float  # Heavy-tail ratio
    left_tail_depth_1pct: float  # Median - Q0.01
    right_tail_depth_1pct: float  # Q0.99 - Median
    tail_shape_regime: str  # HEAVY_TAIL_CRASH_RISK, LEPTOKURTIC_SYMMETRIC, GAUSSIAN_NORMAL, PLATYKURTIC
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core tail metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.rolling_window}:"
            f"{self.moors_quantile_kurtosis:.4f}:{self.bowley_quantile_skewness:.4f}:"
            f"{self.tail_asymmetry_ratio:.4f}:{self.hogg_tail_weight:.4f}:"
            f"{self.tail_shape_regime}"
        )
        return hmac.new(TAIL_SHAPE_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_moors_kurtosis(returns: np.ndarray) -> float:
    """Calculate Moors (1988) robust quantile kurtosis based on empirical octiles.

    Formula:
        Q_Moors = [(E_7 - E_5) + (E_3 - E_1)] / (E_6 - E_2)
    where E_i is the i/8 empirical quantile (octile).

    Normal distribution benchmark: Q_Moors ≈ 1.233.
    """
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 16:
        raise ValueError("Insufficient observations to compute Moors quantile kurtosis (minimum 16).")

    # Octiles: 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8
    octiles = np.percentile(clean, [12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5])
    e1, e2, e3, e4, e5, e6, e7 = octiles

    denom = e6 - e2
    if denom <= 1e-8:
        return 1.233

    numer = (e7 - e5) + (e3 - e1)
    moors = float(numer / denom)
    return float(round(moors, 4))


def compute_bowley_skewness(returns: np.ndarray) -> float:
    """Calculate Bowley-Kelly robust quantile skewness.

    Formula:
        S_q = [(Q_0.90 - Q_0.50) - (Q_0.50 - Q_0.10)] / (Q_0.90 - Q_0.10)
    Bounded strictly in [-1.0, 1.0].
    """
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        raise ValueError("Insufficient observations to compute Bowley skewness.")

    q10, q50, q90 = np.percentile(clean, [10.0, 50.0, 90.0])
    denom = q90 - q10
    if denom <= 1e-8:
        return 0.0

    skew = ((q90 - q50) - (q50 - q10)) / denom
    return float(round(float(np.clip(skew, -1.0, 1.0)), 4))


def compute_hogg_tail_weight(returns: np.ndarray) -> float:
    """Calculate Hogg (1974) tail weight indicator.

    Formula:
        H = [Mean(top 5%) - Mean(bottom 5%)] / [Mean(top 50%) - Mean(bottom 50%)]
    For standard normal, H ≈ 2.58. Higher values indicate heavy fat tails.
    """
    clean = np.asarray(returns, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 20:
        return 2.58

    q05, q95 = np.percentile(clean, [5.0, 95.0])
    q50 = np.median(clean)

    top_5 = clean[clean >= q95]
    bot_5 = clean[clean <= q05]
    top_50 = clean[clean >= q50]
    bot_50 = clean[clean <= q50]

    u_05 = float(np.mean(top_5)) if len(top_5) > 0 else q95
    l_05 = float(np.mean(bot_5)) if len(bot_5) > 0 else q05
    u_50 = float(np.mean(top_50)) if len(top_50) > 0 else q50
    l_50 = float(np.mean(bot_50)) if len(bot_50) > 0 else q50

    denom = u_50 - l_50
    if denom <= 1e-8:
        return 2.58

    hogg = float((u_05 - l_05) / denom)
    return float(round(hogg, 4))


def evaluate_tail_shape_profile(
    ticker: str,
    prices: pd.Series,
    window: int = 120,
) -> TailShapeReceipt:
    """Evaluate comprehensive robust quantile tail shape, kurtosis, and crash asymmetry.

    Args:
        ticker: Validated stock ticker symbol.
        prices: Series of daily stock prices.
        window: Rolling observation window in sessions (default: 120).

    Returns:
        TailShapeReceipt with complete diagnostics and cryptographic proof.
    """
    valid_ticker = validate_ticker(ticker.strip().upper())
    window = max(30, min(window, 504))

    p_clean = prices.dropna().astype(float)
    if len(p_clean) < 30:
        raise ValueError(f"Insufficient price history for {valid_ticker} (minimum 30 required).")

    # Use window tail
    p_tail = p_clean.iloc[-window:]
    returns = p_tail.pct_change().dropna().values
    sample_size = len(returns)

    # 1. Moors Kurtosis
    moors_kurt = compute_moors_kurtosis(returns)
    moors_excess = float(round(moors_kurt - GAUSSIAN_MOORS_KURTOSIS, 4))

    # 2. Bowley Skewness
    bowley_skew = compute_bowley_skewness(returns)

    # 3. Hogg Tail Weight
    hogg_tail = compute_hogg_tail_weight(returns)

    # 4. Tail Asymmetry Ratio (Left 1% Depth vs Right 1% Depth)
    q01, q50, q99 = np.percentile(returns, [1.0, 50.0, 99.0])
    left_depth = float(round(max(0.0, q50 - q01) * 100.0, 2))  # in percent
    right_depth = float(round(max(0.0, q99 - q50) * 100.0, 2))  # in percent

    if right_depth <= 1e-4:
        tail_ratio = 1.0 if left_depth <= 1e-4 else 2.5
    else:
        tail_ratio = float(round(left_depth / right_depth, 3))

    # 5. Tail Shape Regime Categorization
    if moors_kurt >= 1.50 and tail_ratio >= 1.25:
        regime = "HEAVY_TAIL_ASYMMETRIC_CRASH_RISK"
    elif moors_kurt >= 1.45:
        regime = "LEPTOKURTIC_SYMMETRIC"
    elif moors_kurt <= 1.05:
        regime = "PLATYKURTIC_THIN_TAILS"
    else:
        regime = "GAUSSIAN_NORMAL_SHAPE"

    summary = (
        f"Robust Tail Shape Analysis for {valid_ticker} ({sample_size} sessions, window: {window}D): "
        f"Moors Kurtosis = {moors_kurt:.3f} (Excess: {moors_excess:+.3f} vs Normal 1.233), "
        f"Bowley Skewness = {bowley_skew:+.3f}, Hogg Tail Weight = {hogg_tail:.2f}. "
        f"Tail Asymmetry Ratio = {tail_ratio:.2f}x (Left 1%: {left_depth:.2f}% vs Right 1%: {right_depth:.2f}%). "
        f"Regime: {regime}."
    )

    receipt = TailShapeReceipt(
        ticker=valid_ticker,
        sample_sessions=sample_size,
        rolling_window=window,
        moors_quantile_kurtosis=moors_kurt,
        moors_excess_kurtosis=moors_excess,
        bowley_quantile_skewness=bowley_skew,
        tail_asymmetry_ratio=tail_ratio,
        hogg_tail_weight=hogg_tail,
        left_tail_depth_1pct=left_depth,
        right_tail_depth_1pct=right_depth,
        tail_shape_regime=regime,
        summary=summary,
    )
    return receipt
