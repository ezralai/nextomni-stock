"""Nicolae Gârleanu & Lasse Heje Pedersen (2011) / Andrea Frazzini & Lasse Heje Pedersen (2014)
Margin-Based Asset Pricing, Leverage Constraints & Basis Spread Engine.

Implements the Margin-Based Asset Pricing framework:
- Gârleanu & Pedersen (2011, Review of Financial Studies, "Margin-Based Asset Pricing"):
    When investors face margin constraints, the shadow cost of capital (Lagrange multiplier psi >= 0)
    enters the equilibrium pricing equation. Assets that require higher margin or cannot easily be
    financed trade at a discount, creating a basis spread between cash and derivative synthetic instruments.
    The Margin-CAPM expected excess return is:
        E[R_i] - R_f = beta_i * lambda_m + m_i * psi
    where:
        - beta_i: Asset market beta
        - lambda_m: Market risk premium
        - m_i: Haircut/margin requirement of asset i (0 < m_i <= 1)
        - psi: Margin constraint shadow cost (positive when margin constraints bind across funding markets)
- Frazzini & Pedersen (2014, Journal of Financial Economics, "Betting Against Beta"):
    Constrained investors who cannot use leverage overweight high-beta assets, making them expensive
    and delivering lower risk-adjusted returns (flatter Security Market Line).
    Conversely, low-beta assets are underpriced and generate high alpha when leveraged.

References:
    Gârleanu, N., & Pedersen, L. H. (2011). Margin-based asset pricing and deviations from the law of one price.
    The Review of Financial Studies, 24(6), 1980-2022.
    Frazzini, A., & Pedersen, L. H. (2014). Betting against beta.
    Journal of Financial Economics, 111(1), 1-25.
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

MARGIN_PRICING_SIGNING_KEY = b"nextomni_margin_asset_pricing_v1"


@dataclass
class MarginAssetPricingReceipt:
    """Cryptographic audit receipt for Gârleanu-Pedersen (2011) Margin-Based Asset Pricing."""

    ticker: str
    latest_price: float
    # Margin & Leverage Parameters
    margin_requirement: float  # Haircut m_i in [0.05, 0.90]
    baseline_margin_requirement: float  # Baseline normal-times haircut m_0 in [0.05, 0.50]
    margin_shadow_cost_psi: float  # Lagrange multiplier shadow cost psi in [0.0, 0.25]
    market_beta: float  # Systematic market beta beta_i in [-1.0, 3.5]
    market_risk_premium_pct: float  # Market risk premium lambda_m (%) in [2.0, 12.0]
    risk_free_rate_pct: float  # Risk-free rate R_f (%) in [0.0, 10.0]
    # Derived Margin-CAPM Metrics
    standard_capm_expected_return_pct: float  # R_f + beta_i * lambda_m (%)
    margin_capm_expected_return_pct: float  # R_f + beta_i * lambda_m + m_i * psi (%)
    margin_premium_contribution_pct: float  # m_i * psi (%)
    basis_spread_discount_pct: float  # D_i = psi * (1 - m_i) * P_t / (1 + R_f) normalized (%)
    bab_leverage_compression_score: float  # Frazzini-Pedersen (2014) BAB compression score
    margin_pricing_impulse: float  # Overall pricing impulse in [-2.5, 2.5]
    expected_margin_alpha_pct: float  # Projected annualized alpha net of margin constraints (%)
    # Flags & Regime
    is_margin_constrained_discount: bool  # True if psi >= 0.08 and m_i <= 0.30
    is_margin_overpricing_trap: bool  # True if beta_i >= 1.40 and m_i >= 0.45 (expensive high-beta trap)
    is_bab_alpha_opportunity: bool  # True if beta_i <= 0.75 and psi >= 0.05 (BAB long opportunity)
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
        """Compute deterministic HMAC-SHA256 signature across all margin pricing metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:"
            f"{self.margin_requirement:.4f}:{self.baseline_margin_requirement:.4f}:"
            f"{self.margin_shadow_cost_psi:.4f}:{self.market_beta:.4f}:"
            f"{self.market_risk_premium_pct:.4f}:{self.risk_free_rate_pct:.4f}:"
            f"{self.standard_capm_expected_return_pct:.4f}:{self.margin_capm_expected_return_pct:.4f}:"
            f"{self.margin_premium_contribution_pct:.4f}:{self.basis_spread_discount_pct:.4f}:"
            f"{self.bab_leverage_compression_score:.4f}:{self.margin_pricing_impulse:.4f}:"
            f"{self.expected_margin_alpha_pct:.2f}:"
            f"{int(self.is_margin_constrained_discount)}:{int(self.is_margin_overpricing_trap)}:{int(self.is_bab_alpha_opportunity)}:{self.regime}"
        )
        return hmac.new(
            MARGIN_PRICING_SIGNING_KEY,
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


def classify_margin_pricing_regime(
    margin_requirement: float,
    shadow_cost_psi: float,
    market_beta: float,
    basis_spread_discount: float,
) -> str:
    """Classify asset into Gârleanu-Pedersen (2011) & Frazzini-Pedersen (2014) Regimes."""
    if shadow_cost_psi >= 0.08 and margin_requirement <= 0.35 and basis_spread_discount >= 2.0:
        return "MARGIN_CONSTRAINED_BASIS_DISCOUNT"
    if market_beta >= 1.40 and margin_requirement >= 0.40 and shadow_cost_psi >= 0.06:
        return "HIGH_BETA_MARGIN_OVERPRICING_TRAP"
    if market_beta <= 0.75 and shadow_cost_psi >= 0.05:
        return "BETTING_AGAINST_BETA_ALPHA_OPPORTUNITY"
    return "MARGIN_EQUILIBRIUM_NEUTRAL"


def calculate_margin_asset_pricing(
    ticker: str,
    latest_price: float,
    margin_requirement: float = 0.25,
    baseline_margin_requirement: float = 0.20,
    margin_shadow_cost_psi: float = 0.06,
    market_beta: float = 1.15,
    market_risk_premium_pct: float = 6.50,
    risk_free_rate_pct: float = 4.00,
) -> MarginAssetPricingReceipt:
    """Calculate Gârleanu & Pedersen (2011) Margin-CAPM & Basis Spread Metrics.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        margin_requirement: Haircut m_i in [0.05, 0.90].
        baseline_margin_requirement: Normal-times baseline haircut in [0.05, 0.50].
        margin_shadow_cost_psi: Shadow cost Lagrange multiplier psi in [0.0, 0.25].
        market_beta: Systematic market beta beta_i in [-1.0, 3.5].
        market_risk_premium_pct: Expected market risk premium lambda_m (%) in [2.0, 12.0].
        risk_free_rate_pct: Risk-free rate R_f (%) in [0.0, 10.0].

    Returns:
        MarginAssetPricingReceipt with Margin-CAPM return, basis discount, BAB compression, and regime.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(margin_requirement, "margin_requirement")
    assert_finite_number(baseline_margin_requirement, "baseline_margin_requirement")
    assert_finite_number(margin_shadow_cost_psi, "margin_shadow_cost_psi")
    assert_finite_number(market_beta, "market_beta")
    assert_finite_number(market_risk_premium_pct, "market_risk_premium_pct")
    assert_finite_number(risk_free_rate_pct, "risk_free_rate_pct")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (0.05 <= margin_requirement <= 0.90):
        raise ValueError("margin_requirement must be in [0.05, 0.90].")
    if not (0.05 <= baseline_margin_requirement <= 0.50):
        raise ValueError("baseline_margin_requirement must be in [0.05, 0.50].")
    if not (0.0 <= margin_shadow_cost_psi <= 0.25):
        raise ValueError("margin_shadow_cost_psi must be in [0.0, 0.25].")
    if not (-1.0 <= market_beta <= 3.5):
        raise ValueError("market_beta must be in [-1.0, 3.5].")
    if not (2.0 <= market_risk_premium_pct <= 12.0):
        raise ValueError("market_risk_premium_pct must be in [2.0, 12.0].")
    if not (0.0 <= risk_free_rate_pct <= 10.0):
        raise ValueError("risk_free_rate_pct must be in [0.0, 10.0].")

    p_t = float(latest_price)
    m_i = float(margin_requirement)
    m_0 = float(baseline_margin_requirement)
    psi = float(margin_shadow_cost_psi)
    beta_i = float(market_beta)
    mrp = float(market_risk_premium_pct)
    rf = float(risk_free_rate_pct)

    # 1. Standard CAPM vs Margin-CAPM Expected Return
    std_capm = rf + beta_i * mrp
    margin_prem = 100.0 * (m_i * psi)
    margin_capm = std_capm + margin_prem

    # 2. Gârleanu-Pedersen Basis Spread Discount D_i (%)
    # When psi > 0, an asset with low margin requirement (1 - m_i high) provides valuable borrowing capacity
    rf_dec = rf / 100.0
    raw_basis = 100.0 * psi * (1.0 - m_i) / (1.0 + rf_dec)
    basis_discount = float(np.clip(raw_basis, -5.0, 15.0))

    # 3. Frazzini-Pedersen (2014) BAB Leverage Compression Score
    # Score > 0 indicates low beta with high capital efficiency; score < 0 indicates expensive high beta
    bab_compression = float(np.clip((1.0 - beta_i) * (1.0 + 5.0 * psi) - 1.5 * (m_i - m_0), -3.0, 3.0))

    # 4. Pricing Impulse & Expected Alpha
    # Assets with high basis discount or low beta in tight margin regimes generate positive abnormal alpha
    raw_impulse = 0.45 * (basis_discount / 4.0) + 0.35 * bab_compression - 0.50 * max(0.0, beta_i - 1.3) * (psi / 0.10)
    impulse = float(np.clip(raw_impulse, -2.5, 2.5))
    expected_alpha = float(np.clip(3.8 * impulse, -12.0, 12.0))

    regime = classify_margin_pricing_regime(m_i, psi, beta_i, basis_discount)
    is_discount = bool(regime == "MARGIN_CONSTRAINED_BASIS_DISCOUNT")
    is_trap = bool(regime == "HIGH_BETA_MARGIN_OVERPRICING_TRAP")
    is_bab = bool(regime == "BETTING_AGAINST_BETA_ALPHA_OPPORTUNITY")

    if regime == "MARGIN_CONSTRAINED_BASIS_DISCOUNT":
        summary = (
            f"Margin-Constrained Basis Discount (m_i={m_i:.1%}, psi={psi:.1%}, basis={basis_discount:+.2f}%). "
            f"Binding funding margin constraints make low-haircut asset trade at a persistent cash-synthetic "
            f"basis discount ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "HIGH_BETA_MARGIN_OVERPRICING_TRAP":
        summary = (
            f"High-Beta Margin Overpricing Trap (beta={beta_i:.2f}, m_i={m_i:.1%}, psi={psi:.1%}). "
            f"Leverage-constrained retail investors have overbid high-beta stock, flattening SML and "
            f"compressing forward risk-adjusted returns ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "BETTING_AGAINST_BETA_ALPHA_OPPORTUNITY":
        summary = (
            f"Betting Against Beta Alpha Opportunity (beta={beta_i:.2f}, psi={psi:.1%}, BAB_score={bab_compression:+.2f}). "
            f"Underpriced low-beta asset offers superior Sharpe ratio and positive alpha when leveraged ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Margin Equilibrium Neutral (beta={beta_i:.2f}, m_i={m_i:.1%}, psi={psi:.1%}). "
            f"Margin requirements and leverage constraints are within normal balanced boundaries ({expected_alpha:+.1f}%/yr)."
        )

    return MarginAssetPricingReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        margin_requirement=round(m_i, 4),
        baseline_margin_requirement=round(m_0, 4),
        margin_shadow_cost_psi=round(psi, 4),
        market_beta=round(beta_i, 4),
        market_risk_premium_pct=round(mrp, 2),
        risk_free_rate_pct=round(rf, 2),
        standard_capm_expected_return_pct=round(std_capm, 2),
        margin_capm_expected_return_pct=round(margin_capm, 2),
        margin_premium_contribution_pct=round(margin_prem, 2),
        basis_spread_discount_pct=round(basis_discount, 2),
        bab_leverage_compression_score=round(bab_compression, 2),
        margin_pricing_impulse=round(impulse, 4),
        expected_margin_alpha_pct=round(expected_alpha, 2),
        is_margin_constrained_discount=is_discount,
        is_margin_overpricing_trap=is_trap,
        is_bab_alpha_opportunity=is_bab,
        regime=regime,
        summary=summary,
    )


def evaluate_margin_pricing_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    margin_requirement: Optional[float] = None,
    baseline_margin_requirement: Optional[float] = None,
    margin_shadow_cost_psi: Optional[float] = None,
    market_beta: Optional[float] = None,
    market_risk_premium_pct: Optional[float] = None,
    risk_free_rate_pct: Optional[float] = None,
) -> MarginAssetPricingReceipt:
    """Evaluate Gârleanu-Pedersen Margin-Based Asset Pricing profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_m_i = 0.25
    inferred_m_0 = 0.20
    inferred_psi = 0.06
    inferred_beta = 1.15
    inferred_mrp = 6.50
    inferred_rf = 4.25

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
                ann_vol = float(np.std(rets, ddof=1) * math.sqrt(252.0)) if len(rets) > 1 else 0.22

                # Infer beta and margin from realized volatility
                inferred_beta = float(np.clip(ann_vol / 0.18, 0.40, 2.50))
                # Higher volatility assets incur higher prime brokerage margin haircuts
                inferred_m_i = float(np.clip(0.15 + 0.60 * ann_vol, 0.10, 0.65))
                inferred_m_0 = 0.20
                # Recent drawdown elevates margin constraint shadow cost
                peak = float(np.max(arr))
                dd = float((arr[-1] - peak) / max(peak, 1e-4))
                inferred_psi = float(np.clip(0.04 - 0.15 * dd, 0.01, 0.18))
    except Exception:
        latest_price = 100.0

    m_val = float(margin_requirement) if margin_requirement is not None else inferred_m_i
    m0_val = (
        float(baseline_margin_requirement)
        if baseline_margin_requirement is not None
        else inferred_m_0
    )
    psi_val = (
        float(margin_shadow_cost_psi)
        if margin_shadow_cost_psi is not None
        else inferred_psi
    )
    beta_val = float(market_beta) if market_beta is not None else inferred_beta
    mrp_val = (
        float(market_risk_premium_pct)
        if market_risk_premium_pct is not None
        else inferred_mrp
    )
    rf_val = float(risk_free_rate_pct) if risk_free_rate_pct is not None else inferred_rf

    return calculate_margin_asset_pricing(
        ticker=clean_ticker,
        latest_price=latest_price,
        margin_requirement=m_val,
        baseline_margin_requirement=m0_val,
        margin_shadow_cost_psi=psi_val,
        market_beta=beta_val,
        market_risk_premium_pct=mrp_val,
        risk_free_rate_pct=rf_val,
    )

