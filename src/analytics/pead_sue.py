"""Victor L. Bernard & Jacob K. Thomas (1989, 1990) Post-Earnings-Announcement Drift (PEAD) & SUE Engine.

Implements the empirical accounting and autocorrelation framework of Bernard & Thomas (1989, 1990):
- Standardized Unexpected Earnings (SUE_t):
    SUE_t = (EPS_t - EPS_{t-4} - mu_delta_EPS) / sigma_delta_EPS
    Quantifies the magnitude of quarterly earnings surprise relative to a seasonal random walk with drift.
- Seasonal Quarterly Autocorrelation Pattern (Foster, 1977; Bernard & Thomas, 1990):
    Seasonally differenced quarterly earnings exhibit positive autocorrelation at lags 1, 2, and 3:
    phi_1 = +0.34, phi_2 = +0.19, phi_3 = +0.06
    and negative seasonal mean-reversion at lag 4:
    phi_4 = -0.24
- Naive Expectation Underreaction & 60-Session Drift:
    Market participants anchor on a naive seasonal random walk (ignoring phi_1 = +0.34), causing prices
    to drift in the direction of the surprise for ~60 trading sessions after the announcement,
    with remaining drift decaying exponentially as sessions since announcement (d) elapse.
- Regimes:
    - HIGH_SUE_POSITIVE_DRIFT: Strong positive surprise (SUE >= 1.0) within active drift window (d <= 50)
    - LOW_SUE_NEGATIVE_SLUMP: Severe negative surprise (SUE <= -1.0) within active drift window (d <= 50)
    - EXHAUSTED_LATE_CYCLE_DRIFT: Large surprise (|SUE| >= 1.0) with d > 50 sessions elapsed
    - NEUTRAL_EARNINGS_EQUILIBRIUM: Mild surprise (|SUE| < 1.0) with contained abnormal drift

References:
    Bernard, V. L., & Thomas, J. K. (1989). Post-earnings-announcement drift: Delayed price response or risk premium?
    Journal of Accounting Research, 27, 1-36.
    Bernard, V. L., & Thomas, J. K. (1990). Evidence that stock prices do not fully reflect the implications of current earnings for future earnings.
    Journal of Accounting and Economics, 13(4), 305-340.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

PEAD_SUE_SIGNING_KEY = b"nextomni_pead_sue_secret_v1"

# Bernard & Thomas (1990) Empirical Seasonal Earnings Autocorrelation Coefficients
PHI_LAG_1 = 0.34
PHI_LAG_2 = 0.19
PHI_LAG_3 = 0.06
PHI_LAG_4 = -0.24


@dataclass
class PEADSUEReceipt:
    """Cryptographic audit receipt for Bernard & Thomas (1989, 1990) PEAD & SUE Profile."""

    ticker: str
    latest_price: float
    # Earnings Inputs & SUE Metrics
    current_eps: float
    year_ago_eps: float
    drift_mean_eps: float
    surprise_std_eps: float
    sue_score: float  # Standardized Unexpected Earnings in [-6.0, +6.0]
    sue_decile: int  # Cross-sectional SUE decile rank in [1, 10]
    days_since_announcement: int  # Trading sessions elapsed since announcement (d >= 0)
    # Autocorrelation & Drift Inferences
    predicted_next_quarter_sue: float  # phi_1 * SUE_t (+0.34 * SUE_t)
    predicted_lag4_reversal_sue: float  # phi_4 * SUE_t (-0.24 * SUE_t)
    initial_60d_drift_pct: float  # Full 60-session CAR potential at d = 0 (%)
    remaining_drift_pct: float  # Decay-adjusted remaining 60-session CAR (%)
    expected_pead_alpha_pct: float  # Annualized expected PEAD excess return (%)
    # Flags & Regime
    is_active_positive_drift: bool  # True if SUE >= 1.0 and d <= 50
    is_active_negative_slump: bool  # True if SUE <= -1.0 and d <= 50
    regime: str  # HIGH_SUE_POSITIVE_DRIFT, LOW_SUE_NEGATIVE_SLUMP, EXHAUSTED_LATE_CYCLE_DRIFT, NEUTRAL_EARNINGS_EQUILIBRIUM
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core PEAD & SUE metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.current_eps:.4f}:"
            f"{self.year_ago_eps:.4f}:{self.drift_mean_eps:.4f}:{self.surprise_std_eps:.4f}:"
            f"{self.sue_score:.4f}:{self.sue_decile}:{self.days_since_announcement}:"
            f"{self.predicted_next_quarter_sue:.4f}:{self.predicted_lag4_reversal_sue:.4f}:"
            f"{self.initial_60d_drift_pct:.2f}:{self.remaining_drift_pct:.2f}:"
            f"{self.expected_pead_alpha_pct:.2f}:{int(self.is_active_positive_drift)}:"
            f"{int(self.is_active_negative_slump)}:{self.regime}"
        )
        return hmac.new(
            PEAD_SUE_SIGNING_KEY,
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


def map_sue_to_decile(sue: float) -> int:
    """Map Standardized Unexpected Earnings (SUE) to Bernard-Thomas decile rank [1, 10]."""
    if sue >= 2.00:
        return 10
    if sue >= 1.20:
        return 9
    if sue >= 0.60:
        return 8
    if sue >= 0.25:
        return 7
    if sue >= 0.00:
        return 6
    if sue >= -0.25:
        return 5
    if sue >= -0.60:
        return 4
    if sue >= -1.20:
        return 3
    if sue >= -2.00:
        return 2
    return 1


def classify_pead_regime(
    sue_score: float,
    days_since_announcement: int,
) -> str:
    """Classify asset into Bernard & Thomas (1989, 1990) PEAD lifecycle regime."""
    if abs(sue_score) < 1.0:
        return "NEUTRAL_EARNINGS_EQUILIBRIUM"
    if days_since_announcement > 50:
        return "EXHAUSTED_LATE_CYCLE_DRIFT"
    if sue_score >= 1.0:
        return "HIGH_SUE_POSITIVE_DRIFT"
    return "LOW_SUE_NEGATIVE_SLUMP"


def calculate_pead_sue(
    ticker: str,
    latest_price: float,
    current_eps: float,
    year_ago_eps: float,
    surprise_std_eps: float = 0.15,
    drift_mean_eps: float = 0.02,
    days_since_announcement: int = 10,
) -> PEADSUEReceipt:
    """Calculate Bernard & Thomas (1989, 1990) SUE score, AR(4) expectations, and 60-day PEAD.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price.
        current_eps: Quarterly diluted EPS for current quarter t.
        year_ago_eps: Quarterly diluted EPS for same quarter prior year t-4.
        surprise_std_eps: Historical standard deviation of seasonal EPS changes (sigma).
        drift_mean_eps: Historical mean trend of seasonal EPS changes (mu).
        days_since_announcement: Trading sessions elapsed since earnings release (d >= 0).

    Returns:
        PEADSUEReceipt with SUE decile, autocorrelation forecasts, and remaining drift.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(current_eps, "current_eps")
    assert_finite_number(year_ago_eps, "year_ago_eps")
    assert_finite_number(surprise_std_eps, "surprise_std_eps")
    assert_finite_number(drift_mean_eps, "drift_mean_eps")
    assert_finite_number(float(days_since_announcement), "days_since_announcement")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if days_since_announcement < 0:
        raise ValueError("days_since_announcement must be non-negative.")

    # Safe denominator floor for earnings surprise volatility
    sigma = max(float(surprise_std_eps), 0.01)
    unexpected_eps = float(current_eps) - float(year_ago_eps) - float(drift_mean_eps)
    raw_sue = unexpected_eps / sigma
    sue_score = float(np.clip(raw_sue, -6.0, 6.0))

    decile = map_sue_to_decile(sue_score)

    # Bernard & Thomas (1990) Seasonal Autocorrelation Projections
    pred_next_q_sue = PHI_LAG_1 * sue_score
    pred_lag4_sue = PHI_LAG_4 * sue_score

    # Initial 60-session cumulative abnormal drift potential (Bernard & Thomas: ~+4.2% to +8.5% for top decile)
    initial_drift = float(np.clip(2.15 * sue_score, -10.0, 10.0))

    # Exponential drift decay across trading sessions since announcement (tau = 35 sessions)
    d = int(days_since_announcement)
    decay_factor = math.exp(-float(d) / 35.0)
    remaining_drift = initial_drift * decay_factor

    # Annualized Expected PEAD Alpha
    expected_alpha = float(np.clip(remaining_drift * 2.2, -14.0, 14.0))

    regime = classify_pead_regime(sue_score, d)
    is_pos_drift = bool(regime == "HIGH_SUE_POSITIVE_DRIFT")
    is_neg_slump = bool(regime == "LOW_SUE_NEGATIVE_SLUMP")

    if regime == "HIGH_SUE_POSITIVE_DRIFT":
        summary = (
            f"High-SUE Positive Drift (SUE={sue_score:+.2f}, Decile {decile}/10, d={d}d). "
            f"Market underreacts to positive AR(1) earnings persistence (E[SUE_t+1]={pred_next_q_sue:+.2f}). "
            f"Remaining 60d drift: {remaining_drift:+.2f}% (Annualized Alpha: {expected_alpha:+.1f}%/yr)."
        )
    elif regime == "LOW_SUE_NEGATIVE_SLUMP":
        summary = (
            f"Low-SUE Negative Slump (SUE={sue_score:+.2f}, Decile {decile}/10, d={d}d). "
            f"Persistent negative earnings autocorrelation (E[SUE_t+1]={pred_next_q_sue:+.2f}). "
            f"Remaining 60d downside drift: {remaining_drift:+.2f}% (Annualized Drag: {expected_alpha:+.1f}%/yr)."
        )
    elif regime == "EXHAUSTED_LATE_CYCLE_DRIFT":
        summary = (
            f"Exhausted Late-Cycle Drift (SUE={sue_score:+.2f}, Decile {decile}/10, d={d}d). "
            f"Primary 60-day announcement drift largely realized (Remaining: {remaining_drift:+.2f}%); "
            f"monitor lag-4 seasonal reversal (E[SUE_t+4]={pred_lag4_sue:+.2f})."
        )
    else:
        summary = (
            f"Neutral Earnings Equilibrium (SUE={sue_score:+.2f}, Decile {decile}/10, d={d}d). "
            f"Earnings surprise near seasonal expectation; minimal abnormal post-announcement drift."
        )

    return PEADSUEReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        current_eps=round(float(current_eps), 4),
        year_ago_eps=round(float(year_ago_eps), 4),
        drift_mean_eps=round(float(drift_mean_eps), 4),
        surprise_std_eps=round(sigma, 4),
        sue_score=round(sue_score, 4),
        sue_decile=decile,
        days_since_announcement=d,
        predicted_next_quarter_sue=round(pred_next_q_sue, 4),
        predicted_lag4_reversal_sue=round(pred_lag4_sue, 4),
        initial_60d_drift_pct=round(initial_drift, 2),
        remaining_drift_pct=round(remaining_drift, 2),
        expected_pead_alpha_pct=round(expected_alpha, 2),
        is_active_positive_drift=is_pos_drift,
        is_active_negative_slump=is_neg_slump,
        regime=regime,
        summary=summary,
    )


def evaluate_pead_sue_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    current_eps: Optional[float] = None,
    year_ago_eps: Optional[float] = None,
    surprise_std_eps: Optional[float] = None,
    drift_mean_eps: Optional[float] = None,
    days_since_announcement: Optional[int] = None,
) -> PEADSUEReceipt:
    """Evaluate Bernard & Thomas (1989, 1990) PEAD & SUE profile using database and heuristics."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_sue_shift = 0.25
    try:
        obs = db.get_observations(clean_ticker, db_path=db_path)
        if obs is not None and not obs.empty and "close" in obs.columns:
            closes = [float(x) for x in obs["close"].tolist() if math.isfinite(float(x)) and float(x) > 0]
            if closes:
                latest_price = closes[-1]
            if len(closes) >= 20:
                ret_20d = (closes[-1] - closes[-20]) / closes[-20]
                inferred_sue_shift = float(np.clip(ret_20d * 2.5, -0.60, 0.80))
    except Exception:
        latest_price = 100.0

    y_eps = float(year_ago_eps) if year_ago_eps is not None else 1.50
    c_eps = float(current_eps) if current_eps is not None else (y_eps + 0.02 + inferred_sue_shift)
    std_eps = float(surprise_std_eps) if surprise_std_eps is not None else 0.15
    mu_eps = float(drift_mean_eps) if drift_mean_eps is not None else 0.02
    days = int(days_since_announcement) if days_since_announcement is not None else 10

    return calculate_pead_sue(
        ticker=clean_ticker,
        latest_price=latest_price,
        current_eps=c_eps,
        year_ago_eps=y_eps,
        surprise_std_eps=std_eps,
        drift_mean_eps=mu_eps,
        days_since_announcement=days,
    )
