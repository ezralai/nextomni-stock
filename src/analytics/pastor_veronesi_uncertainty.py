"""Ľuboš Pástor & Pietro Veronesi (2003, 2006)
Stock Valuation, Bayesian Learning about Profitability & Parameter Uncertainty Convexity Engine.

Implements the rational Bayesian learning and parameter uncertainty convexity framework:
- Pástor & Veronesi (2003, Journal of Finance; 2006, Journal of Financial Economics):
    A firm's long-run average excess profitability psi_i is unobserved and learned by investors
    over time via Bayesian updating as realized earnings and return signals arrive:
        sigma_{psi,post} = sigma_{psi,0} / sqrt(1 + lambda_learn * T_age)
    Because future book equity compounds exponentially with profitability over the growth horizon
    T_growth, the fundamental Market-to-Book ratio (M/B) is strictly convex in unknown psi_i.
    By Jensen's inequality, parameter uncertainty about average profitability (sigma_{psi,post}^2)
    rationally increases fundamental valuation:
        C_PV = exp(0.5 * gamma_eff * T_growth^2 * b_ret * sigma_{psi,post}^2)
        (M/B)^*_PV = clip(exp(T_growth * psi_hat) * C_PV, 0.35, 35.0)
- Valuation Mispricing Gap & Learning Impulse:
    Comparing the observed Market-to-Book ratio (M/B)_obs against the Pástor-Veronesi warranted
    uncertainty-adjusted valuation (M/B)^*_PV distinguishes between:
    1. CONVEX_GROWTH_LEARNING_UNDERVALUED: High-retention, high-uncertainty growth firm whose
       observed valuation underprices its rational Jensen's inequality convexity premium
       (Delta_MB <= -0.12 and PVLI_t >= 0.35).
    2. EXTRAPOLATIVE_UNCERTAINTY_BUBBLE_TRAP: Glamour stock whose observed M/B far exceeds even
       the generous Pástor-Veronesi uncertainty convexity bound (Delta_MB >= 0.25 and PVLI_t <= -0.40).
    3. RATIONAL_CONVEXITY_HIGH_VALUATION: Elevated M/B that is rationally justified by high
       parameter uncertainty and earnings retention (|Delta_MB| < 0.25 and C_PV >= 1.25).
    4. MATURE_POSTERIOR_CERTAINTY_EQUILIBRIUM: Mature firm where Bayesian learning has resolved
       most parameter uncertainty (C_PV < 1.25 and |Delta_MB| < 0.25).

References:
    Pástor, Ľ., & Veronesi, P. (2003). Stock valuation and learning about profitability.
    The Journal of Finance, 58(5), 1749-1789.
    Pástor, Ľ., & Veronesi, P. (2006). Was there a Nasdaq bubble in the late 1990s?
    Journal of Financial Economics, 81(1), 61-100.
"""

from __future__ import annotations

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

PASTOR_VERONESI_SIGNING_KEY = b"nextomni_pastor_veronesi_uncertainty_v1"


@dataclass
class PastorVeronesiUncertaintyReceipt:
    """Cryptographic audit receipt for Pástor & Veronesi (2003, 2006) Learning & Uncertainty Profile."""

    ticker: str
    latest_price: float
    # Fundamental & Learning Inputs
    expected_excess_profitability: float  # Expected ROE - r_e: psi_hat in [-0.50, 0.80]
    prior_profitability_uncertainty: float  # Prior uncertainty sigma_{psi,0} in [0.005, 0.50]
    firm_learning_age_years: float  # Firm learning age T_age in [0.5, 100.0] years
    retention_ratio: float  # Earnings retention ratio (1 - payout) b_ret in [0.0, 1.0]
    observed_market_to_book: float  # Observed M/B ratio in [0.20, 50.0]
    # Derived Pástor-Veronesi Bayesian & Convexity Metrics
    posterior_profitability_uncertainty: float  # Bayesian updated sigma_{psi,post}
    convexity_valuation_multiplier: float  # Jensen's inequality multiplier C_PV >= 1.0
    warranted_market_to_book: float  # Rational Pástor-Veronesi (M/B)^*_PV
    valuation_mispricing_gap_pct: float  # ((M/B)_obs - (M/B)^*_PV) / (M/B)^*_PV
    pastor_veronesi_learning_impulse: float  # PVLI_t in [-2.5, 2.5]
    expected_learning_alpha_pct: float  # Projected annualized alpha (%)
    # Flags & Regime
    is_convex_learning_leader: bool  # True if undervalued relative to rational convexity bound
    is_uncertainty_bubble_trap: bool  # True if M/B exceeds even rational uncertainty bound
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
        """Compute deterministic HMAC-SHA256 signature across all Pástor-Veronesi metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:"
            f"{self.expected_excess_profitability:.4f}:{self.prior_profitability_uncertainty:.4f}:"
            f"{self.firm_learning_age_years:.2f}:{self.retention_ratio:.4f}:"
            f"{self.observed_market_to_book:.4f}:{self.posterior_profitability_uncertainty:.4f}:"
            f"{self.convexity_valuation_multiplier:.4f}:{self.warranted_market_to_book:.4f}:"
            f"{self.valuation_mispricing_gap_pct:.4f}:{self.pastor_veronesi_learning_impulse:.4f}:"
            f"{self.expected_learning_alpha_pct:.2f}:{int(self.is_convex_learning_leader)}:"
            f"{int(self.is_uncertainty_bubble_trap)}:{self.regime}"
        )
        return hmac.new(
            PASTOR_VERONESI_SIGNING_KEY,
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


def classify_pastor_veronesi_regime(
    valuation_mispricing_gap_pct: float,
    convexity_valuation_multiplier: float,
    learning_impulse: float,
) -> str:
    """Classify asset into Pástor & Veronesi (2003, 2006) Learning & Convexity Regimes."""
    if valuation_mispricing_gap_pct <= -0.12 and learning_impulse >= 0.35:
        return "CONVEX_GROWTH_LEARNING_UNDERVALUED"
    if valuation_mispricing_gap_pct >= 0.25 and learning_impulse <= -0.40:
        return "EXTRAPOLATIVE_UNCERTAINTY_BUBBLE_TRAP"
    if abs(valuation_mispricing_gap_pct) < 0.25 and convexity_valuation_multiplier >= 1.25:
        return "RATIONAL_CONVEXITY_HIGH_VALUATION"
    return "MATURE_POSTERIOR_CERTAINTY_EQUILIBRIUM"


def calculate_pastor_veronesi_uncertainty(
    ticker: str,
    latest_price: float,
    expected_excess_profitability: float = 0.06,
    prior_profitability_uncertainty: float = 0.14,
    firm_learning_age_years: float = 8.0,
    retention_ratio: float = 0.80,
    observed_market_to_book: float = 3.20,
) -> PastorVeronesiUncertaintyReceipt:
    """Calculate Pástor & Veronesi (2003, 2006) Bayesian Learning & Uncertainty Convexity Valuation.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        expected_excess_profitability: Expected excess ROE over cost of equity psi_hat in [-0.50, 0.80].
        prior_profitability_uncertainty: Prior uncertainty sigma_{psi,0} in [0.005, 0.50].
        firm_learning_age_years: Effective firm learning age T_age in [0.5, 100.0] years.
        retention_ratio: Earnings retention ratio b_ret = (1 - payout) in [0.0, 1.0].
        observed_market_to_book: Observed Market-to-Book ratio (M/B)_obs in [0.20, 50.0].

    Returns:
        PastorVeronesiUncertaintyReceipt with posterior uncertainty, C_PV, (M/B)^*_PV, and alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(expected_excess_profitability, "expected_excess_profitability")
    assert_finite_number(prior_profitability_uncertainty, "prior_profitability_uncertainty")
    assert_finite_number(firm_learning_age_years, "firm_learning_age_years")
    assert_finite_number(retention_ratio, "retention_ratio")
    assert_finite_number(observed_market_to_book, "observed_market_to_book")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (-0.50 <= expected_excess_profitability <= 0.80):
        raise ValueError("expected_excess_profitability must be in [-0.50, 0.80].")
    if not (0.005 <= prior_profitability_uncertainty <= 0.50):
        raise ValueError("prior_profitability_uncertainty must be in [0.005, 0.50].")
    if not (0.5 <= firm_learning_age_years <= 100.0):
        raise ValueError("firm_learning_age_years must be in [0.5, 100.0].")
    if not (0.0 <= retention_ratio <= 1.0):
        raise ValueError("retention_ratio must be in [0.0, 1.0].")
    if not (0.20 <= observed_market_to_book <= 50.0):
        raise ValueError("observed_market_to_book must be in [0.20, 50.0].")

    p_t = float(latest_price)
    psi_hat = float(expected_excess_profitability)
    sigma_0 = float(prior_profitability_uncertainty)
    t_age = float(firm_learning_age_years)
    b_ret = float(retention_ratio)
    mb_obs = float(observed_market_to_book)

    # 1. Bayesian Posterior Profitability Uncertainty sigma_{psi,post}
    # Learning speed lambda_learn = 0.18 per year reduces posterior variance as firm matures
    lambda_learn = 0.18
    sigma_post = float(sigma_0 / math.sqrt(1.0 + lambda_learn * t_age))

    # 2. Pástor-Veronesi Jensen's Inequality Convexity Multiplier C_PV
    # Effective compounding horizon T_growth = 15.0 years; convexity scaled by retention ratio b_ret
    t_growth = 15.0
    convexity_exponent = 0.5 * (t_growth ** 2) * (0.35 + 0.65 * b_ret) * (sigma_post ** 2)
    c_pv = float(np.clip(math.exp(convexity_exponent), 1.0, 8.0))

    # 3. Warranted Market-to-Book Ratio (M/B)^*_PV
    certainty_equiv_mb = math.exp(t_growth * psi_hat * (0.45 + 0.55 * b_ret))
    mb_warranted = float(np.clip(certainty_equiv_mb * c_pv, 0.35, 35.0))

    # 4. Valuation Mispricing Gap Delta_MB = ((M/B)_obs - (M/B)^*_PV) / (M/B)^*_PV
    mispricing_gap = float((mb_obs - mb_warranted) / max(mb_warranted, 1e-6))

    # 5. Pástor-Veronesi Learning & Valuation Impulse PVLI_t in [-2.5, 2.5]
    # Negative mispricing gap (observed < warranted) + high convexity -> positive impulse
    raw_impulse = -2.2 * mispricing_gap + 0.45 * (c_pv - 1.15)
    pvli = float(np.clip(raw_impulse, -2.5, 2.5))

    # 6. Regime Classification & Expected Alpha
    regime = classify_pastor_veronesi_regime(mispricing_gap, c_pv, pvli)
    is_leader = bool(regime == "CONVEX_GROWTH_LEARNING_UNDERVALUED")
    is_trap = bool(regime == "EXTRAPOLATIVE_UNCERTAINTY_BUBBLE_TRAP")

    expected_alpha = float(np.clip(4.8 * pvli, -12.0, 12.0))

    if regime == "CONVEX_GROWTH_LEARNING_UNDERVALUED":
        summary = (
            f"Convex Growth Learning Leader (Obs M/B={mb_obs:.2f}x vs Warranted={mb_warranted:.2f}x, "
            f"C_PV={c_pv:.2f}x, Gap={mispricing_gap:+.1%}). Market underprices Jensen's inequality "
            f"profitability convexity ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "EXTRAPOLATIVE_UNCERTAINTY_BUBBLE_TRAP":
        summary = (
            f"Extrapolative Uncertainty Bubble Trap (Obs M/B={mb_obs:.2f}x vs Warranted={mb_warranted:.2f}x, "
            f"Gap={mispricing_gap:+.1%}). Valuation exceeds even the Pástor-Veronesi uncertainty bound; "
            f"Bayesian learning resolution predicts multiple compression ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "RATIONAL_CONVEXITY_HIGH_VALUATION":
        summary = (
            f"Rational Convexity High Valuation (Obs M/B={mb_obs:.2f}x vs Warranted={mb_warranted:.2f}x, "
            f"C_PV={c_pv:.2f}x, SigmaPost={sigma_post:.1%}). Elevated valuation multiple is rationally "
            f"justified by parameter uncertainty and high retention ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Mature Posterior Certainty Equilibrium (Obs M/B={mb_obs:.2f}x vs Warranted={mb_warranted:.2f}x, "
            f"Age={t_age:.1f}y, SigmaPost={sigma_post:.1%}). Bayesian learning has largely resolved "
            f"profitability uncertainty ({expected_alpha:+.1f}%/yr)."
        )

    return PastorVeronesiUncertaintyReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        expected_excess_profitability=round(psi_hat, 4),
        prior_profitability_uncertainty=round(sigma_0, 4),
        firm_learning_age_years=round(t_age, 2),
        retention_ratio=round(b_ret, 4),
        observed_market_to_book=round(mb_obs, 4),
        posterior_profitability_uncertainty=round(sigma_post, 4),
        convexity_valuation_multiplier=round(c_pv, 4),
        warranted_market_to_book=round(mb_warranted, 4),
        valuation_mispricing_gap_pct=round(mispricing_gap, 4),
        pastor_veronesi_learning_impulse=round(pvli, 4),
        expected_learning_alpha_pct=round(expected_alpha, 2),
        is_convex_learning_leader=is_leader,
        is_uncertainty_bubble_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_pastor_veronesi_uncertainty_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    expected_excess_profitability: Optional[float] = None,
    prior_profitability_uncertainty: Optional[float] = None,
    firm_learning_age_years: Optional[float] = None,
    retention_ratio: Optional[float] = None,
    observed_market_to_book: Optional[float] = None,
) -> PastorVeronesiUncertaintyReceipt:
    """Evaluate Pástor & Veronesi (2003, 2006) Learning & Uncertainty profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_psi = 0.06
    inferred_sigma0 = 0.14
    inferred_age = 8.0
    inferred_retention = 0.80
    inferred_mb = 3.20

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
                arr = np.asarray(closes[-min(len(closes), 252) :], dtype=float)
                rets = np.diff(arr) / arr[:-1]
                ann_ret = float(np.mean(rets) * 252.0)
                ann_vol = float(np.std(rets, ddof=1) * math.sqrt(252.0)) if len(rets) > 1 else 0.25

                # Infer expected excess profitability psi_hat from drift relative to 8% cost of equity
                inferred_psi = float(np.clip(0.35 * (ann_ret - 0.08) + 0.05, -0.25, 0.35))
                # Infer prior profitability uncertainty from annualized return volatility
                inferred_sigma0 = float(np.clip(0.45 * ann_vol, 0.04, 0.38))
                # Infer effective learning age from sample depth and volatility stability
                inferred_age = float(np.clip(len(closes) / 28.0, 1.5, 35.0))
                # High volatility / growth stocks retain more earnings
                inferred_retention = float(np.clip(0.55 + 0.70 * ann_vol, 0.25, 0.95))
                # Infer observed M/B from cumulative price appreciation over window
                price_ratio = float(arr[-1] / max(arr[0], 1e-6))
                inferred_mb = float(np.clip(2.4 * price_ratio, 0.60, 18.0))
    except Exception:
        latest_price = 100.0

    psi_val = (
        float(expected_excess_profitability)
        if expected_excess_profitability is not None
        else inferred_psi
    )
    sigma0_val = (
        float(prior_profitability_uncertainty)
        if prior_profitability_uncertainty is not None
        else inferred_sigma0
    )
    age_val = (
        float(firm_learning_age_years)
        if firm_learning_age_years is not None
        else inferred_age
    )
    ret_val = (
        float(retention_ratio)
        if retention_ratio is not None
        else inferred_retention
    )
    mb_val = (
        float(observed_market_to_book)
        if observed_market_to_book is not None
        else inferred_mb
    )

    return calculate_pastor_veronesi_uncertainty(
        ticker=clean_ticker,
        latest_price=latest_price,
        expected_excess_profitability=psi_val,
        prior_profitability_uncertainty=sigma0_val,
        firm_learning_age_years=age_val,
        retention_ratio=ret_val,
        observed_market_to_book=mb_val,
    )

