"""Mark Grinblatt & Bing Han (2005) / Andrea Frazzini (2006)
Prospect Theory Reference Point, Capital Gains Overhang (CGO) & Disposition Effect Engine.

Implements the prospect-theory mental accounting and disposition-driven underreaction framework:
- Prospect Theory & Mental Accounting (Grinblatt & Han, 2005):
    Investors evaluate stock positions relative to an aggregate turnover-weighted reference cost
    basis R_t. Due to the Kahneman-Tversky S-shaped value function (concave over gains, convex and
    steeper over losses), disposition-prone investors prematurely sell stocks trading above their
    cost basis (P_t > R_t) and stubbornly hold onto stocks trading below their cost basis (P_t < R_t).
- Capital Gains Overhang (CGO_t):
    CGO_t = (P_t - R_t) / P_t
    - Positive CGO_t (>= +0.08): Disposition selling depresses equilibrium price P_t below fundamental
      value, predicting positive subsequent return drift as R_t gradually catches up to P_t.
    - Negative CGO_t (<= -0.10): Loss-averse holders refuse to sell into bad news, propping up P_t
      above fundamental value and predicting negative subsequent return drift ('Disposition Lock-In Trap').
- Disposition Effect & Underreaction to News (Frazzini, 2006):
    Post-event drift is strongest when fundamental news surprise (SUE / news_surprise_score) has the
    same sign as CGO_t (positive news on high-CGO winners vs. negative news on deep negative-CGO losers).
- Regimes:
    - GAIN_OVERHANG_CONTINUATION_LEADER: Positive CGO_t >= 0.08 and FDUI_t >= 0.18 (+Alpha continuation)
    - ANCHORED_REFERENCE_EQUILIBRIUM: Price trading near aggregate cost basis (|CGO_t| < 0.08)
    - GAIN_HARVESTING_SUPPLY_WATCH: Positive CGO_t >= 0.08 with contradicting negative news (< -0.50)
    - LOSS_AVERSION_LOCK_IN_TRAP: Deep negative CGO_t <= -0.10 and FDUI_t <= -0.22 (-Alpha trap)

References:
    Grinblatt, M., & Han, B. (2005). Prospect theory, mental accounting, and momentum.
    Journal of Financial Economics, 78(2), 311-339.
    Frazzini, A. (2006). The disposition effect and underreaction to news.
    The Journal of Finance, 61(4), 2017-2046.
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

CAPITAL_GAINS_OVERHANG_SIGNING_KEY = b"nextomni_capital_gains_overhang_secret_v1"


@dataclass
class CapitalGainsOverhangReceipt:
    """Cryptographic audit receipt for Grinblatt & Han (2005) & Frazzini (2006) CGO Profile."""

    ticker: str
    latest_price: float
    # Reference Cost Basis & News Inputs
    aggregate_reference_price: float  # Turnover-weighted cost basis R_t ($ > 0)
    avg_daily_turnover: float  # Average daily share turnover fraction in (0.0, 0.50]
    news_surprise_score: float  # Standardized fundamental news surprise in [-5.0, 5.0]
    institutional_rationality_shield: float  # Discipline shield against disposition bias in [0.0, 1.0]
    # Derived Grinblatt-Han & Frazzini Metrics
    capital_gains_overhang: float  # CGO_t = (P_t - R_t) / P_t in [-5.0, 0.99]
    news_overhang_alignment_mult: float  # Frazzini (2006) news x CGO sign alignment in [1.0, 4.25]
    frazzini_disposition_impulse: float  # FDUI_t in [-2.50, 2.50]
    expected_cgo_alpha_pct: float  # Projected annualized disposition drift alpha (%)
    # Flags & Regime
    is_gain_overhang_leader: bool  # True if CGO >= 0.08 and FDUI >= 0.18
    is_loss_lock_in_trap: bool  # True if CGO <= -0.10 and FDUI <= -0.22
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
        """Compute deterministic HMAC-SHA256 signature across core capital gains overhang metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.aggregate_reference_price:.2f}:"
            f"{self.avg_daily_turnover:.4f}:{self.news_surprise_score:.4f}:"
            f"{self.institutional_rationality_shield:.4f}:{self.capital_gains_overhang:.4f}:"
            f"{self.news_overhang_alignment_mult:.4f}:{self.frazzini_disposition_impulse:.4f}:"
            f"{self.expected_cgo_alpha_pct:.2f}:{int(self.is_gain_overhang_leader)}:"
            f"{int(self.is_loss_lock_in_trap)}:{self.regime}"
        )
        return hmac.new(
            CAPITAL_GAINS_OVERHANG_SIGNING_KEY,
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


def classify_cgo_regime(
    cgo_ratio: float,
    news_surprise: float,
    disposition_impulse: float,
) -> str:
    """Classify asset into Grinblatt & Han (2005) / Frazzini (2006) Disposition Regimes."""
    if cgo_ratio <= -0.10 and disposition_impulse <= -0.22:
        return "LOSS_AVERSION_LOCK_IN_TRAP"
    if cgo_ratio >= 0.08 and news_surprise < -0.50:
        return "GAIN_HARVESTING_SUPPLY_WATCH"
    if cgo_ratio >= 0.08 and disposition_impulse >= 0.18:
        return "GAIN_OVERHANG_CONTINUATION_LEADER"
    return "ANCHORED_REFERENCE_EQUILIBRIUM"


def calculate_capital_gains_overhang(
    ticker: str,
    latest_price: float,
    aggregate_reference_price: float = 88.0,
    avg_daily_turnover: float = 0.008,
    news_surprise_score: float = 0.65,
    institutional_rationality_shield: float = 0.55,
) -> CapitalGainsOverhangReceipt:
    """Calculate Grinblatt & Han (2005) Capital Gains Overhang (CGO_t) and Frazzini (2006) Impulse.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        aggregate_reference_price: Turnover-weighted aggregate cost basis R_t ($ > 0).
        avg_daily_turnover: Average daily share turnover fraction in (0.0, 0.50].
        news_surprise_score: Standardized fundamental news surprise in [-5.0, 5.0].
        institutional_rationality_shield: Institutional discipline factor in [0.0, 1.0].

    Returns:
        CapitalGainsOverhangReceipt with CGO_t, FDUI_t, regime classification, and expected alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(aggregate_reference_price, "aggregate_reference_price")
    assert_finite_number(avg_daily_turnover, "avg_daily_turnover")
    assert_finite_number(news_surprise_score, "news_surprise_score")
    assert_finite_number(
        institutional_rationality_shield, "institutional_rationality_shield"
    )

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if aggregate_reference_price <= 0.0:
        raise ValueError("aggregate_reference_price must be strictly positive.")
    if not (0.0 < avg_daily_turnover <= 0.50):
        raise ValueError("avg_daily_turnover must be in (0.0, 0.50].")
    if not (-5.0 <= news_surprise_score <= 5.0):
        raise ValueError("news_surprise_score must be in [-5.0, 5.0].")
    if not (0.0 <= institutional_rationality_shield <= 1.0):
        raise ValueError("institutional_rationality_shield must be in [0.0, 1.0].")

    p_t = float(latest_price)
    r_t = float(aggregate_reference_price)
    turn = float(avg_daily_turnover)
    news = float(news_surprise_score)
    shield = float(institutional_rationality_shield)

    # 1. Grinblatt & Han (2005) Capital Gains Overhang CGO_t = (P_t - R_t) / P_t
    raw_cgo = (p_t - r_t) / p_t
    cgo = float(np.clip(raw_cgo, -5.0, 0.99))

    # 2. Frazzini (2006) News x Overhang Sign Alignment Multiplier
    if (cgo * news) > 0.0:
        align_mult = float(np.clip(1.0 + 0.65 * abs(news), 1.0, 4.25))
    else:
        align_mult = 1.0

    # 3. Frazzini Disposition Underreaction Impulse FDUI_t in [-2.5, 2.5]
    # Lower daily turnover means reference price R_t updates more slowly, prolonging disposition drift
    speed_factor = float(np.clip(1.15 - 1.5 * turn, 0.40, 1.15))
    raw_fdui = cgo * align_mult * (1.30 - 0.60 * shield) * speed_factor * 2.2
    fdui = float(np.clip(raw_fdui, -2.5, 2.5))

    # 4. Regime Classification & Expected Disposition Drift Alpha
    regime = classify_cgo_regime(cgo, news, fdui)
    is_leader = bool(regime == "GAIN_OVERHANG_CONTINUATION_LEADER")
    is_trap = bool(regime == "LOSS_AVERSION_LOCK_IN_TRAP")

    expected_alpha = float(np.clip(14.5 * fdui + 0.8 * news, -13.5, 13.5))

    if regime == "GAIN_OVERHANG_CONTINUATION_LEADER":
        summary = (
            f"Gain Overhang Continuation Leader (P=${p_t:.2f} vs CostBasis=${r_t:.2f}, CGO={cgo:+.1%}, "
            f"FDUI={fdui:+.3f}). Premature disposition gain-taking caps price below fundamental value, "
            f"predicting upward continuation ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "GAIN_HARVESTING_SUPPLY_WATCH":
        summary = (
            f"Gain Harvesting Supply Watch (CGO={cgo:+.1%}, News={news:+.2f}). "
            f"Positive unrealized gain cushion faces negative fundamental news divergence; monitor "
            f"profit-taking supply ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "LOSS_AVERSION_LOCK_IN_TRAP":
        summary = (
            f"Loss-Aversion Lock-In Trap (P=${p_t:.2f} vs CostBasis=${r_t:.2f}, CGO={cgo:+.1%}, "
            f"FDUI={fdui:+.3f}). Deep underwater holders refuse to realize losses on bad news, "
            f"creating overhead supply drag ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Anchored Reference Equilibrium (P=${p_t:.2f} vs CostBasis=${r_t:.2f}, CGO={cgo:+.1%}). "
            f"Price trades near aggregate mental-accounting cost basis ({expected_alpha:+.1f}%/yr)."
        )

    return CapitalGainsOverhangReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        aggregate_reference_price=round(r_t, 2),
        avg_daily_turnover=round(turn, 4),
        news_surprise_score=round(news, 4),
        institutional_rationality_shield=round(shield, 4),
        capital_gains_overhang=round(cgo, 4),
        news_overhang_alignment_mult=round(align_mult, 4),
        frazzini_disposition_impulse=round(fdui, 4),
        expected_cgo_alpha_pct=round(expected_alpha, 2),
        is_gain_overhang_leader=is_leader,
        is_loss_lock_in_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_capital_gains_overhang_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    aggregate_reference_price: Optional[float] = None,
    avg_daily_turnover: Optional[float] = None,
    news_surprise_score: Optional[float] = None,
    institutional_rationality_shield: Optional[float] = None,
) -> CapitalGainsOverhangReceipt:
    """Evaluate Grinblatt & Han (2005) & Frazzini (2006) CGO profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_ref_price = 88.0
    inferred_turnover = 0.008
    inferred_news = 0.65

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
            if len(closes) >= 20:
                window_c = np.asarray(closes[-min(len(closes), 60) :], dtype=float)
                # Compute Grinblatt-Han survival-weighted reference price over available lookback
                n_hist = len(window_c) - 1
                v_rate = 0.025
                weights = np.zeros(n_hist, dtype=float)
                surv = 1.0
                for idx in range(n_hist - 1, -1, -1):
                    weights[idx] = v_rate * surv
                    surv *= 1.0 - v_rate
                w_sum = float(np.sum(weights))
                if w_sum > 0:
                    weights /= w_sum
                    inferred_ref_price = float(np.sum(weights * window_c[:-1]))
                ret_20 = (window_c[-1] / window_c[0]) - 1.0
                inferred_news = float(np.clip(ret_20 * 5.0, -3.5, 3.5))
    except Exception:
        latest_price = 100.0

    ref_val = (
        float(aggregate_reference_price)
        if aggregate_reference_price is not None
        else inferred_ref_price
    )
    turn_val = (
        float(avg_daily_turnover)
        if avg_daily_turnover is not None
        else inferred_turnover
    )
    news_val = (
        float(news_surprise_score)
        if news_surprise_score is not None
        else inferred_news
    )
    shield_val = (
        float(institutional_rationality_shield)
        if institutional_rationality_shield is not None
        else 0.55
    )

    return calculate_capital_gains_overhang(
        ticker=clean_ticker,
        latest_price=latest_price,
        aggregate_reference_price=ref_val,
        avg_daily_turnover=turn_val,
        news_surprise_score=news_val,
        institutional_rationality_shield=shield_val,
    )

