"""Burke Ratio, Becker Pain Index & Ulcer Performance Index Engine for NextOmni Stock.

Implements path-dependent drawdown analytics quantifying the psychological strain,
depth, duration, and recovery physics of investment strategies:
- Martin (1989) Ulcer Index & Ulcer Performance Index (UPI)
- Burke (1994) Ratio (quadratic penalty on discrete drawdown valleys)
- Becker (2001) Pain Index & Pain Ratio
- Calmar & Sterling Ratios

References:
    Martin, P. R. (1989). The Investor's Guide to Fidelity Funds.
    Burke, G. (1994). A sharper Sharpe ratio. Futures, 23(3), 56-56.
    Becker, T. (2001). Risk and risk-adjusted performance measures.
    Journal of Performance Measurement, 5(3), 63-71.
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

DRAWDOWN_SIGNING_KEY = b"nextomni_drawdown_analytics_secret_key_v1"


@dataclass
class DrawdownAnalyticsReceipt:
    """Cryptographic audit receipt for path-dependent drawdown and pain analytics."""

    ticker: str
    sample_sessions: int
    annualized_return_pct: float
    risk_free_rate_pct: float
    max_drawdown_pct: float
    ulcer_index_pct: float
    ulcer_performance_index: float  # Martin Ratio: (R_p - R_f) / UI
    burke_ratio: float  # (R_p - R_f) / sqrt(sum(DD_valleys^2))
    pain_index_pct: float  # Mean percentage drawdown over time
    pain_ratio: float  # (R_p - R_f) / Pain Index
    calmar_ratio: float  # (R_p - R_f) / MaxDD
    sterling_ratio: float  # (R_p - R_f) / (Mean Top 5 DDs)
    max_drawdown_duration_sessions: int
    current_drawdown_pct: float
    is_in_drawdown: bool
    num_discrete_drawdowns: int
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core drawdown metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.annualized_return_pct:.4f}:"
            f"{self.max_drawdown_pct:.4f}:{self.ulcer_index_pct:.4f}:"
            f"{self.burke_ratio:.4f}:{self.pain_index_pct:.4f}:{self.pain_ratio:.4f}:"
            f"{self.max_drawdown_duration_sessions}"
        )
        return hmac.new(DRAWDOWN_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def identify_drawdown_valleys(drawdown_series: np.ndarray) -> List[float]:
    """Identify peak percentage depths of discrete drawdown episodes.

    Args:
        drawdown_series: 1D array of fractional drawdowns (>= 0).

    Returns:
        List of peak drawdown depths for each completed or active episode.
    """
    valleys: List[float] = []
    current_peak = 0.0
    in_dd = False

    for val in drawdown_series:
        if val > 1e-6:
            in_dd = True
            if val > current_peak:
                current_peak = val
        else:
            if in_dd and current_peak > 1e-6:
                valleys.append(float(current_peak))
                current_peak = 0.0
                in_dd = False

    if in_dd and current_peak > 1e-6:
        valleys.append(float(current_peak))

    return valleys


def compute_drawdown_metrics(
    equity_curve: np.ndarray,
    annualized_return: float,
    risk_free_rate: float = 0.04,
) -> Dict[str, Any]:
    """Calculate comprehensive Martin, Burke, Becker, and Calmar drawdown metrics.

    Args:
        equity_curve: 1D array of cumulative equity values (e.g. cumulative wealth).
        annualized_return: Annualized asset or strategy return (e.g. 0.12 for 12%).
        risk_free_rate: Annualized benchmark risk-free rate.

    Returns:
        Dictionary of calculated drawdown indicators.
    """
    clean_eq = np.asarray(equity_curve, dtype=float)
    clean_eq = clean_eq[np.isfinite(clean_eq)]
    n = len(clean_eq)
    if n < 5:
        raise ValueError("Insufficient equity observations (minimum 5 required).")

    # High Water Mark and Drawdowns
    hwm = np.maximum.accumulate(clean_eq)
    dd_fraction = (hwm - clean_eq) / np.maximum(hwm, 1e-8)
    dd_fraction = np.maximum(dd_fraction, 0.0)

    # Maximum Drawdown
    mdd = float(np.max(dd_fraction))
    current_dd = float(dd_fraction[-1])

    # Martin (1989) Ulcer Index: root-mean-square drawdown
    ui = float(math.sqrt(np.mean(dd_fraction ** 2)))

    # Becker (2001) Pain Index: mean drawdown
    pain_index = float(np.mean(dd_fraction))

    # Excess return
    excess_return = annualized_return - risk_free_rate

    # Ulcer Performance Index (UPI / Martin Ratio)
    upi = float(excess_return / ui) if ui > 1e-6 else float(np.sign(excess_return) * 10.0)

    # Becker Pain Ratio
    pain_ratio = (
        float(excess_return / pain_index)
        if pain_index > 1e-6
        else float(np.sign(excess_return) * 10.0)
    )

    # Calmar Ratio
    calmar = float(excess_return / mdd) if mdd > 1e-6 else float(np.sign(excess_return) * 10.0)

    # Discrete drawdown valleys for Burke Ratio
    valleys = identify_drawdown_valleys(dd_fraction)
    if valleys:
        burke_denom = math.sqrt(sum(v ** 2 for v in valleys))
        burke = float(excess_return / burke_denom) if burke_denom > 1e-6 else 0.0
        # Sterling Ratio: excess return / mean of top 5 drawdowns
        top_valleys = sorted(valleys, reverse=True)[:5]
        mean_top_dd = float(np.mean(top_valleys))
        sterling = float(excess_return / mean_top_dd) if mean_top_dd > 1e-6 else 0.0
    else:
        burke = float(np.sign(excess_return) * 10.0)
        sterling = float(np.sign(excess_return) * 10.0)

    # Max Drawdown Duration (sessions under water)
    max_duration = 0
    curr_duration = 0
    for val in dd_fraction:
        if val > 1e-6:
            curr_duration += 1
            if curr_duration > max_duration:
                max_duration = curr_duration
        else:
            curr_duration = 0

    return {
        "max_drawdown_pct": round(mdd * 100.0, 2),
        "ulcer_index_pct": round(ui * 100.0, 2),
        "ulcer_performance_index": round(upi, 3),
        "burke_ratio": round(burke, 3),
        "pain_index_pct": round(pain_index * 100.0, 2),
        "pain_ratio": round(pain_ratio, 3),
        "calmar_ratio": round(calmar, 3),
        "sterling_ratio": round(sterling, 3),
        "max_drawdown_duration_sessions": max_duration,
        "current_drawdown_pct": round(current_dd * 100.0, 2),
        "is_in_drawdown": current_dd > 1e-4,
        "num_discrete_drawdowns": len(valleys),
    }


def evaluate_drawdown_profile(
    ticker: str,
    prices: pd.Series,
    risk_free_rate: float = 0.04,
) -> DrawdownAnalyticsReceipt:
    """Evaluate institutional path-dependent drawdown, Pain Index, and Burke ratio profile.

    Args:
        ticker: Validated stock ticker symbol.
        prices: Series of daily asset prices.
        risk_free_rate: Annualized benchmark risk-free rate.

    Returns:
        DrawdownAnalyticsReceipt with comprehensive performance metrics and cryptographic HMAC signature.
    """
    valid_ticker = validate_ticker(ticker.strip().upper())
    assert_finite_number(risk_free_rate, "risk_free_rate")

    clean_p = prices.dropna().astype(float)
    if len(clean_p) < 15:
        raise ValueError(f"Insufficient price history for {valid_ticker} (minimum 15 required).")

    # Annualized return
    ret_series = clean_p.pct_change().dropna()
    mean_daily = float(ret_series.mean())
    ann_ret = mean_daily * 252.0

    # Cumulative wealth equity curve starting at 1.0
    equity = np.cumprod(1.0 + ret_series.values)
    equity = np.insert(equity, 0, 1.0)

    metrics = compute_drawdown_metrics(
        equity_curve=equity,
        annualized_return=ann_ret,
        risk_free_rate=risk_free_rate,
    )

    summary = (
        f"Drawdown & Pain Profile for {valid_ticker} ({len(clean_p)} sessions): "
        f"Ann Return = {ann_ret * 100:.1f}%, MaxDD = {metrics['max_drawdown_pct']:.1f}%, "
        f"Pain Index = {metrics['pain_index_pct']:.1f}%, Pain Ratio = {metrics['pain_ratio']:.2f}, "
        f"Burke Ratio = {metrics['burke_ratio']:.2f}, Ulcer Index = {metrics['ulcer_index_pct']:.1f}% "
        f"(UPI: {metrics['ulcer_performance_index']:.2f}). Max Under-Water Duration: {metrics['max_drawdown_duration_sessions']} sessions."
    )

    receipt = DrawdownAnalyticsReceipt(
        ticker=valid_ticker,
        sample_sessions=len(clean_p),
        annualized_return_pct=round(ann_ret * 100.0, 2),
        risk_free_rate_pct=round(risk_free_rate * 100.0, 2),
        max_drawdown_pct=metrics["max_drawdown_pct"],
        ulcer_index_pct=metrics["ulcer_index_pct"],
        ulcer_performance_index=metrics["ulcer_performance_index"],
        burke_ratio=metrics["burke_ratio"],
        pain_index_pct=metrics["pain_index_pct"],
        pain_ratio=metrics["pain_ratio"],
        calmar_ratio=metrics["calmar_ratio"],
        sterling_ratio=metrics["sterling_ratio"],
        max_drawdown_duration_sessions=metrics["max_drawdown_duration_sessions"],
        current_drawdown_pct=metrics["current_drawdown_pct"],
        is_in_drawdown=metrics["is_in_drawdown"],
        num_discrete_drawdowns=metrics["num_discrete_drawdowns"],
        summary=summary,
    )
    return receipt
