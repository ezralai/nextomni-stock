"""Diether, Malloy & Scherbina (2002) / Miller (1977) Analyst Forecast Dispersion & Divergence of Opinion Engine.

Implements the heterogeneous-beliefs and short-sale constraint asset pricing framework:
- Analyst Forecast Dispersion Anomaly (Diether, Malloy & Scherbina, 2002):
    D_t = sigma(EPS_Forecasts_t) / max(|Mean(EPS_Forecast_t)|, 0.10)
    Stocks with higher dispersion in analysts' earnings forecasts earn significantly lower future
    returns than otherwise similar stocks (-9.48% annualized spread between top and bottom quintiles).
- Miller (1977) Optimistic Marginal Buyer Hypothesis:
    When investors hold heterogeneous beliefs (high D_t) and pessimistic investors face short-sale
    frictions (high borrow fees or low institutional lending supply), pessimistic views are excluded
    from price discovery and market prices reflect only the valuation of the most optimistic buyers.
- Earnings Resolution & Disagreement Collapse:
    Subsequent earnings announcements resolve disagreement toward the true fundamental mean, triggering
    sharp downward corrections in high-dispersion stocks ('Miller Overpricing Trap').
- Regimes:
    - HIGH_CONSENSUS_COMPOUNDER: Tight analyst agreement (D_t <= 0.05) with minimal overpricing wedge (+Alpha)
    - MODERATE_BELIEF_DIVERGENCE: Normal cross-sectional forecast dispersion (0.05 < D_t < 0.18)
    - CONTESTED_OPINION_WATCH: Elevated disagreement requiring earnings-release caution
    - MILLER_OVERPRICING_TRAP: Wide forecast dispersion (D_t >= 0.28) amplified by short-sale constraints (-Alpha trap)

References:
    Diether, K. B., Malloy, C. J., & Scherbina, A. (2002). Differences of opinion and the cross section of stock returns.
    The Journal of Finance, 57(5), 2113-2141.
    Miller, E. M. (1977). Risk, uncertainty, and divergence of opinion.
    The Journal of Finance, 32(4), 1151-1168.
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

BELIEF_DISPERSION_SIGNING_KEY = b"nextomni_belief_dispersion_secret_v1"


@dataclass
class BeliefDispersionReceipt:
    """Cryptographic audit receipt for Diether et al. (2002) & Miller (1977) Belief Dispersion Profile."""

    ticker: str
    latest_price: float
    # Forecast & Short-Friction Inputs
    consensus_mean_eps: float
    forecast_std_eps: float
    analyst_count: int
    short_borrow_fee_pct: float
    institutional_ownership_pct: float
    # Dispersion & Miller Overpricing Metrics
    dispersion_ratio: float  # Diether-Malloy-Scherbina D_t in [0.0, 5.0]
    short_sale_friction_index: float  # Binding short impediment score in [0.0, 1.0]
    miller_overpricing_wedge: float  # Optimist valuation bias index in [0.0, 3.5]
    expected_dispersion_alpha_pct: float  # Projected annualized excess return (%)
    # Flags & Regime
    is_high_consensus_compounder: bool  # True if D_t <= 0.05 and MOW <= 0.25
    is_miller_overpricing_trap: bool  # True if D_t >= 0.28 or MOW >= 1.35
    regime: str  # HIGH_CONSENSUS_COMPOUNDER, MODERATE_BELIEF_DIVERGENCE, CONTESTED_OPINION_WATCH, MILLER_OVERPRICING_TRAP
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core belief dispersion metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.consensus_mean_eps:.4f}:"
            f"{self.forecast_std_eps:.4f}:{self.analyst_count}:{self.short_borrow_fee_pct:.2f}:"
            f"{self.institutional_ownership_pct:.2f}:{self.dispersion_ratio:.4f}:"
            f"{self.short_sale_friction_index:.4f}:{self.miller_overpricing_wedge:.4f}:"
            f"{self.expected_dispersion_alpha_pct:.2f}:{int(self.is_high_consensus_compounder)}:"
            f"{int(self.is_miller_overpricing_trap)}:{self.regime}"
        )
        return hmac.new(
            BELIEF_DISPERSION_SIGNING_KEY,
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


def classify_dispersion_regime(
    dispersion_ratio: float,
    miller_wedge: float,
) -> str:
    """Classify asset into Diether, Malloy & Scherbina (2002) / Miller (1977) regimes."""
    if dispersion_ratio >= 0.28 or miller_wedge >= 1.35:
        return "MILLER_OVERPRICING_TRAP"
    if dispersion_ratio >= 0.18 or miller_wedge >= 0.90:
        return "CONTESTED_OPINION_WATCH"
    if dispersion_ratio <= 0.05 and miller_wedge <= 0.25:
        return "HIGH_CONSENSUS_COMPOUNDER"
    return "MODERATE_BELIEF_DIVERGENCE"


def calculate_belief_dispersion(
    ticker: str,
    latest_price: float,
    consensus_mean_eps: float,
    forecast_std_eps: float,
    analyst_count: int = 12,
    short_borrow_fee_pct: float = 0.50,
    institutional_ownership_pct: float = 75.0,
) -> BeliefDispersionReceipt:
    """Calculate Diether, Malloy & Scherbina (2002) Dispersion and Miller (1977) Overpricing Wedge.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price.
        consensus_mean_eps: Mean analyst consensus EPS forecast ($).
        forecast_std_eps: Standard deviation across analyst EPS forecasts ($).
        analyst_count: Number of covering analysts (N >= 2).
        short_borrow_fee_pct: Annualized stock loan borrow fee (%).
        institutional_ownership_pct: Institutional ownership share of float (%).

    Returns:
        BeliefDispersionReceipt with D_t, SSFI, Miller Overpricing Wedge, and expected alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(consensus_mean_eps, "consensus_mean_eps")
    assert_finite_number(forecast_std_eps, "forecast_std_eps")
    assert_finite_number(float(analyst_count), "analyst_count")
    assert_finite_number(short_borrow_fee_pct, "short_borrow_fee_pct")
    assert_finite_number(institutional_ownership_pct, "institutional_ownership_pct")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if forecast_std_eps < 0.0:
        raise ValueError("forecast_std_eps must be non-negative.")
    if analyst_count < 2:
        raise ValueError("analyst_count must be at least 2 to compute dispersion.")
    if short_borrow_fee_pct < 0.0:
        raise ValueError("short_borrow_fee_pct must be non-negative.")
    if not (0.0 <= institutional_ownership_pct <= 100.0):
        raise ValueError("institutional_ownership_pct must be between 0 and 100.")

    # 1. Diether, Malloy & Scherbina (2002) Forecast Dispersion Ratio D_t
    denom_eps = max(abs(float(consensus_mean_eps)), 0.10)
    raw_disp = float(forecast_std_eps) / denom_eps
    disp_ratio = float(np.clip(raw_disp, 0.0, 5.0))

    # 2. Short-Sale Binding Friction Index in [0.0, 1.0]
    fee_comp = min(float(short_borrow_fee_pct) / 8.0, 1.0)
    io_comp = 1.0 - (float(institutional_ownership_pct) / 100.0)
    ssfi = float(np.clip(0.55 * fee_comp + 0.45 * io_comp, 0.0, 1.0))

    # 3. Miller (1977) Optimism Overpricing Wedge in [0.0, 3.5]
    n_cov = int(analyst_count)
    cov_weight = math.sqrt(float(n_cov - 1) / float(n_cov))
    raw_wedge = disp_ratio * cov_weight * (1.0 + 1.8 * ssfi) * 2.5
    mow = float(np.clip(raw_wedge, 0.0, 3.5))

    # 4. Regime Classification & Expected Dispersion Alpha
    regime = classify_dispersion_regime(disp_ratio, mow)
    is_consensus = bool(regime == "HIGH_CONSENSUS_COMPOUNDER")
    is_trap = bool(regime == "MILLER_OVERPRICING_TRAP")

    expected_alpha = float(np.clip(3.8 - 6.5 * mow, -11.5, 4.5))

    if regime == "HIGH_CONSENSUS_COMPOUNDER":
        summary = (
            f"High-Consensus Compounder (D_t={disp_ratio:.3f}, Wedge={mow:.2f}, N={n_cov}). "
            f"Tight analyst earnings agreement and deep institutional lendable float prevent optimistic bias. "
            f"Expected dispersion alpha: {expected_alpha:+.1f}%/yr."
        )
    elif regime == "MODERATE_BELIEF_DIVERGENCE":
        summary = (
            f"Moderate Belief Divergence (D_t={disp_ratio:.3f}, Wedge={mow:.2f}, N={n_cov}). "
            f"Normal cross-sectional forecast dispersion with {expected_alpha:+.1f}%/yr expected alpha."
        )
    elif regime == "CONTESTED_OPINION_WATCH":
        summary = (
            f"Contested Opinion Watch (D_t={disp_ratio:.3f}, Wedge={mow:.2f}, Short Friction={ssfi:.2f}). "
            f"Elevated analyst disagreement; monitor subsequent earnings release for downward resolution "
            f"({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Miller Optimism Overpricing Trap (D_t={disp_ratio:.3f}, Wedge={mow:.2f}, Short Friction={ssfi:.2f}). "
            f"Severe belief divergence combined with short-sale constraints prices stock at optimist extremes "
            f"({expected_alpha:+.1f}%/yr)."
        )

    return BeliefDispersionReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        consensus_mean_eps=round(float(consensus_mean_eps), 4),
        forecast_std_eps=round(float(forecast_std_eps), 4),
        analyst_count=n_cov,
        short_borrow_fee_pct=round(float(short_borrow_fee_pct), 2),
        institutional_ownership_pct=round(float(institutional_ownership_pct), 2),
        dispersion_ratio=round(disp_ratio, 4),
        short_sale_friction_index=round(ssfi, 4),
        miller_overpricing_wedge=round(mow, 4),
        expected_dispersion_alpha_pct=round(expected_alpha, 2),
        is_high_consensus_compounder=is_consensus,
        is_miller_overpricing_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_belief_dispersion_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    consensus_mean_eps: Optional[float] = None,
    forecast_std_eps: Optional[float] = None,
    analyst_count: Optional[int] = None,
    short_borrow_fee_pct: Optional[float] = None,
    institutional_ownership_pct: Optional[float] = None,
) -> BeliefDispersionReceipt:
    """Evaluate Diether et al. (2002) & Miller (1977) Belief Dispersion profile using database and heuristics."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_std = 0.08
    try:
        obs = db.get_observations(clean_ticker, db_path=db_path)
        if obs is not None and not obs.empty and "close" in obs.columns:
            closes = [float(x) for x in obs["close"].tolist() if math.isfinite(float(x)) and float(x) > 0]
            if closes:
                latest_price = closes[-1]
            if len(closes) >= 20:
                arr = np.asarray(closes[-20:], dtype=float)
                rets = np.diff(arr) / arr[:-1]
                vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.015
                inferred_std = float(np.clip(vol * 5.0, 0.04, 0.60))
    except Exception:
        latest_price = 100.0

    mean_eps = float(consensus_mean_eps) if consensus_mean_eps is not None else 2.50
    std_eps = float(forecast_std_eps) if forecast_std_eps is not None else inferred_std
    n_analysts = int(analyst_count) if analyst_count is not None else 15
    fee = float(short_borrow_fee_pct) if short_borrow_fee_pct is not None else 0.45
    inst_own = float(institutional_ownership_pct) if institutional_ownership_pct is not None else 78.0

    return calculate_belief_dispersion(
        ticker=clean_ticker,
        latest_price=latest_price,
        consensus_mean_eps=mean_eps,
        forecast_std_eps=std_eps,
        analyst_count=n_analysts,
        short_borrow_fee_pct=fee,
        institutional_ownership_pct=inst_own,
    )

