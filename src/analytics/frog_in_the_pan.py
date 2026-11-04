"""Zhi Da, Umit G. Gurun & Mitch Warachka (2014)
'Frog in the Pan' (FIP) Continuous Information Arrival & Momentum Quality Engine.

Implements the limited-attention information discreteness and momentum continuation framework:
- Frog in the Pan Hypothesis (Da, Gurun & Warachka, 2014):
    Investors with limited attention underreact to gradual, continuous information arriving in small
    daily increments below perception thresholds ('boiling frog'), whereas they react immediately
    and fully to large, discrete lump-sum jumps that attract media headlines.
- Information Discreteness (ID_t in [-1.0, +1.0]):
    ID_t = sign(PRET_t) * (%Negative_Days_t - %Positive_Days_t)
    - Continuous Information Arrival (ID_t <= -0.15): High frequency of small daily moves in the same
      direction as cumulative return PRET_t -> strong subsequent 6-to-12 month momentum continuation.
    - Discrete Jump Arrival (ID_t > 0.0 or high 3-day jump concentration): Cumulative return driven by
      a few massive outlier sessions -> weak continuation and elevated reversal risk.
- Regimes:
    - CONTINUOUS_BOILING_FROG_WINNER: Positive cumulative return (PRET_t >= 0.04) with continuous daily
      gains (ID_t <= -0.15 and FMQS_t >= 0.62) (+Alpha continuation)
    - DISCRETE_JUMP_EXHAUSTION_WATCH: Positive return driven by lumpy jumps (ID_t > 0.0 or JumpShare >= 0.55)
    - NEUTRAL_INFORMATION_ARRIVAL: Mild cumulative return or balanced daily sign distribution
    - CONTINUOUS_BOILING_FROG_LOSER_TRAP: Negative cumulative return (PRET_t <= -0.04) with continuous
      daily bleed (ID_t <= -0.15) (-Alpha downward continuation trap)

References:
    Da, Z., Gurun, U. G., & Warachka, M. (2014). Frog in the pan: Continuous information and momentum.
    The Review of Financial Studies, 27(7), 2171-2218.
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

FROG_IN_THE_PAN_SIGNING_KEY = b"nextomni_frog_in_the_pan_secret_v1"


@dataclass
class FrogInThePanReceipt:
    """Cryptographic audit receipt for Da, Gurun & Warachka (2014) Frog in the Pan (FIP) Profile."""

    ticker: str
    latest_price: float
    # Return & Daily Sign Frequency Inputs
    past_cumulative_return: float  # Cumulative formation return PRET_t in [-0.95, 10.0]
    pct_positive_days: float  # Fraction of formation sessions with r_d > 0 in [0.0, 1.0]
    pct_negative_days: float  # Fraction of formation sessions with r_d < 0 in [0.0, 1.0]
    max_three_day_jump_share: float  # Share of total |r_d| concentrated in top 3 days in [0.0, 1.0]
    # Derived Da-Gurun-Warachka Metrics
    pct_zero_days: float  # 1 - pct_pos - pct_neg in [0.0, 1.0]
    information_discreteness_id: float  # ID_t = sign(PRET) * (%neg - %pos) in [-1.0, 1.0]
    fip_momentum_quality_score: float  # FMQS_t in [0.0, 1.0]
    expected_fip_alpha_pct: float  # Projected annualized momentum continuation alpha (%)
    # Flags & Regime
    is_continuous_frog_winner: bool  # True if PRET >= 0.04, ID <= -0.15, FMQS >= 0.62
    is_continuous_frog_loser_trap: bool  # True if PRET <= -0.04 and ID <= -0.15
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
        """Compute deterministic HMAC-SHA256 signature across core Frog in the Pan metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.past_cumulative_return:.4f}:"
            f"{self.pct_positive_days:.4f}:{self.pct_negative_days:.4f}:"
            f"{self.max_three_day_jump_share:.4f}:{self.pct_zero_days:.4f}:"
            f"{self.information_discreteness_id:.4f}:{self.fip_momentum_quality_score:.4f}:"
            f"{self.expected_fip_alpha_pct:.2f}:{int(self.is_continuous_frog_winner)}:"
            f"{int(self.is_continuous_frog_loser_trap)}:{self.regime}"
        )
        return hmac.new(
            FROG_IN_THE_PAN_SIGNING_KEY,
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


def classify_fip_regime(
    past_return: float,
    info_discreteness: float,
    jump_share: float,
    quality_score: float,
) -> str:
    """Classify asset into Da, Gurun & Warachka (2014) Frog in the Pan Regimes."""
    if past_return <= -0.04 and info_discreteness <= -0.15:
        return "CONTINUOUS_BOILING_FROG_LOSER_TRAP"
    if past_return >= 0.04 and (info_discreteness > 0.0 or jump_share >= 0.55):
        return "DISCRETE_JUMP_EXHAUSTION_WATCH"
    if past_return >= 0.04 and info_discreteness <= -0.15 and quality_score >= 0.62:
        return "CONTINUOUS_BOILING_FROG_WINNER"
    return "NEUTRAL_INFORMATION_ARRIVAL"


def calculate_frog_in_the_pan(
    ticker: str,
    latest_price: float,
    past_cumulative_return: float = 0.14,
    pct_positive_days: float = 0.62,
    pct_negative_days: float = 0.35,
    max_three_day_jump_share: float = 0.22,
) -> FrogInThePanReceipt:
    """Calculate Da, Gurun & Warachka (2014) Information Discreteness (ID_t) and FIP Momentum Quality.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price ($).
        past_cumulative_return: Cumulative formation-period return PRET_t in [-0.95, 10.0].
        pct_positive_days: Fraction of positive return sessions in [0.0, 1.0].
        pct_negative_days: Fraction of negative return sessions in [0.0, 1.0].
        max_three_day_jump_share: Share of total absolute daily returns in top 3 sessions in [0.0, 1.0].

    Returns:
        FrogInThePanReceipt with ID_t, FMQS_t, regime classification, and expected continuation alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(past_cumulative_return, "past_cumulative_return")
    assert_finite_number(pct_positive_days, "pct_positive_days")
    assert_finite_number(pct_negative_days, "pct_negative_days")
    assert_finite_number(max_three_day_jump_share, "max_three_day_jump_share")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (-0.95 <= past_cumulative_return <= 10.0):
        raise ValueError("past_cumulative_return must be in [-0.95, 10.0].")
    if not (0.0 <= pct_positive_days <= 1.0):
        raise ValueError("pct_positive_days must be in [0.0, 1.0].")
    if not (0.0 <= pct_negative_days <= 1.0):
        raise ValueError("pct_negative_days must be in [0.0, 1.0].")
    if (pct_positive_days + pct_negative_days) > 1.0 + 1e-6:
        raise ValueError("pct_positive_days + pct_negative_days cannot exceed 1.0.")
    if not (0.0 <= max_three_day_jump_share <= 1.0):
        raise ValueError("max_three_day_jump_share must be in [0.0, 1.0].")

    p_pos = float(pct_positive_days)
    p_neg = float(pct_negative_days)
    p_zero = float(max(0.0, 1.0 - p_pos - p_neg))
    pret = float(past_cumulative_return)
    jump_share = float(max_three_day_jump_share)

    # 1. Da, Gurun & Warachka (2014) Information Discreteness ID_t
    if pret > 0.0:
        ret_sign = 1.0
    elif pret < 0.0:
        ret_sign = -1.0
    else:
        ret_sign = 1.0 if p_pos >= p_neg else -1.0

    id_metric = float(np.clip(ret_sign * (p_neg - p_pos), -1.0, 1.0))

    # 2. FIP Momentum Quality Score FMQS_t in [0.0, 1.0]
    cont_score = (1.0 - id_metric) / 2.0
    fmqs = float(np.clip(0.65 * cont_score + 0.35 * (1.0 - jump_share), 0.0, 1.0))

    # 3. Regime Classification & Expected FIP Continuation Alpha
    regime = classify_fip_regime(pret, id_metric, jump_share, fmqs)
    is_winner = bool(regime == "CONTINUOUS_BOILING_FROG_WINNER")
    is_loser_trap = bool(regime == "CONTINUOUS_BOILING_FROG_LOSER_TRAP")

    # Continuous information amplifies momentum continuation in the direction of PRET_t;
    # discrete jump arrival dampens or reverses it.
    direction = 1.0 if pret >= 0.0 else -1.0
    mag = min(abs(pret), 0.60)
    raw_alpha = direction * mag * (22.0 * (-id_metric) + 12.0 * (0.50 - jump_share))
    expected_alpha = float(np.clip(raw_alpha, -12.5, 12.5))

    if regime == "CONTINUOUS_BOILING_FROG_WINNER":
        summary = (
            f"Continuous 'Boiling Frog' Winner (PRET={pret:+.1%}, ID={id_metric:+.3f}, Quality={fmqs:.2f}). "
            f"Frequent small positive sessions ({p_pos:.0%} up vs {p_neg:.0%} down) under limited attention "
            f"predict strong momentum continuation ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "DISCRETE_JUMP_EXHAUSTION_WATCH":
        summary = (
            f"Discrete Jump Exhaustion Watch (PRET={pret:+.1%}, ID={id_metric:+.3f}, JumpShare={jump_share:.0%}). "
            f"Cumulative gain concentrated in lumpy headline jumps; high attention leaves minimal drift "
            f"({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "CONTINUOUS_BOILING_FROG_LOSER_TRAP":
        summary = (
            f"Continuous 'Boiling Frog' Loser Trap (PRET={pret:+.1%}, ID={id_metric:+.3f}, DownDays={p_neg:.0%}). "
            f"Persistent gradual daily erosion ignored by inattentive holders predicts continued underperformance "
            f"({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Neutral Information Arrival (PRET={pret:+.1%}, ID={id_metric:+.3f}, Quality={fmqs:.2f}). "
            f"Balanced daily return arrival with {expected_alpha:+.1f}%/yr expected FIP drift."
        )

    return FrogInThePanReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        past_cumulative_return=round(pret, 4),
        pct_positive_days=round(p_pos, 4),
        pct_negative_days=round(p_neg, 4),
        max_three_day_jump_share=round(jump_share, 4),
        pct_zero_days=round(p_zero, 4),
        information_discreteness_id=round(id_metric, 4),
        fip_momentum_quality_score=round(fmqs, 4),
        expected_fip_alpha_pct=round(expected_alpha, 2),
        is_continuous_frog_winner=is_winner,
        is_continuous_frog_loser_trap=is_loser_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_frog_in_the_pan_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    past_cumulative_return: Optional[float] = None,
    pct_positive_days: Optional[float] = None,
    pct_negative_days: Optional[float] = None,
    max_three_day_jump_share: Optional[float] = None,
) -> FrogInThePanReceipt:
    """Evaluate Da, Gurun & Warachka (2014) FIP profile from SQLite OHLCV history and optional overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_pret = 0.14
    inferred_pos = 0.62
    inferred_neg = 0.35
    inferred_jump = 0.22

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
                window = np.asarray(closes[-min(len(closes), 60) :], dtype=float)
                rets = np.diff(window) / window[:-1]
                n_days = len(rets)
                if n_days > 0:
                    inferred_pret = float(
                        np.clip((window[-1] / window[0]) - 1.0, -0.90, 5.0)
                    )
                    inferred_pos = float(np.sum(rets > 1e-6)) / float(n_days)
                    inferred_neg = float(np.sum(rets < -1e-6)) / float(n_days)
                    abs_rets = np.abs(rets)
                    tot_abs = float(np.sum(abs_rets))
                    if tot_abs > 0:
                        top3 = float(np.sum(np.sort(abs_rets)[-3:]))
                        inferred_jump = float(np.clip(top3 / tot_abs, 0.05, 0.98))
    except Exception:
        latest_price = 100.0

    pret_val = (
        float(past_cumulative_return)
        if past_cumulative_return is not None
        else inferred_pret
    )
    pos_val = (
        float(pct_positive_days) if pct_positive_days is not None else inferred_pos
    )
    neg_val = (
        float(pct_negative_days) if pct_negative_days is not None else inferred_neg
    )
    if (pos_val + neg_val) > 1.0:
        scale = 1.0 / (pos_val + neg_val)
        pos_val *= scale
        neg_val *= scale
    jump_val = (
        float(max_three_day_jump_share)
        if max_three_day_jump_share is not None
        else inferred_jump
    )

    return calculate_frog_in_the_pan(
        ticker=clean_ticker,
        latest_price=latest_price,
        past_cumulative_return=pret_val,
        pct_positive_days=pos_val,
        pct_negative_days=neg_val,
        max_three_day_jump_share=jump_val,
    )

