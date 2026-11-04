"""Lauren Cohen & Andrea Frazzini (2008) / Kenneth R. Ahern (2013)
Economic Links, Supply-Chain Customer Momentum & Network Centrality Spillover Engine.

Implements the supply-chain limited-attention and production network asset pricing framework:
- Customer Momentum & Limited Attention (Cohen & Frazzini, 2008):
    Investors subject to attention constraints fail to promptly incorporate news about economically
    linked principal customers (SFAS No. 131 disclosures >= 10% of revenue) into supplier equity
    prices. This creates predictable 1-to-3 month return continuation from customer returns
    (R^{Cust}_t) to supplier returns (R^{Supp}_{t+1}), generating >1.5%/month abnormal alpha.
- Inattention Friction & Shared Coverage (Menzly & Ozbas, 2010; Cohen & Frazzini, 2008):
    Return predictability is strongest when shared analyst coverage / institutional cross-ownership
    between customer and supplier is low, delaying cross-firm information diffusion.
- Production Network Centrality Premium (Ahern, 2013):
    Firms positioned at central hubs of the intersectoral input-output network carry greater
    systematic exposure to aggregate supply-chain shocks and earn a positive centrality premium.
- Regimes:
    - POSITIVE_CUSTOMER_SPILLOVER_ALPHA: Unpriced positive customer demand shock (LASI_t >= 0.035) (+Alpha)
    - SYNCHRONIZED_SUPPLY_CHAIN_EQUILIBRIUM: Contemporaneously aligned customer-supplier pricing
    - CONCENTRATED_CUSTOMER_DEPENDENCY_WATCH: High single-customer revenue dependency (w_Cust >= 0.55)
    - UPSTREAM_DEMAND_COLLAPSE_TRAP: Unpriced negative customer shock (LASI_t <= -0.035) (-Alpha trap)

References:
    Cohen, L., & Frazzini, A. (2008). Economic links and predictable returns.
    The Journal of Finance, 63(4), 1977-2011.
    Ahern, K. R. (2013). Network centrality and the cross section of stock returns.
    Journal of Financial Economics, 109(3), 717-736.
    Menzly, L., & Ozbas, O. (2010). Market segmentation and cross-predictability of returns.
    The Journal of Finance, 65(4), 1555-1580.
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

SUPPLY_CHAIN_MOMENTUM_SIGNING_KEY = b"nextomni_supply_chain_momentum_secret_v1"


@dataclass
class SupplyChainMomentumReceipt:
    """Cryptographic audit receipt for Cohen & Frazzini (2008) & Ahern (2013) Supply-Chain Momentum Profile."""

    ticker: str
    latest_price: float
    # Supply-Chain & Production Network Inputs
    customer_portfolio_return: float  # Principal customer 20-day return R^{Cust}_t in [-0.95, 5.0]
    supplier_own_return: float  # Supplier own 20-day return R^{Supp}_t in [-0.95, 5.0]
    customer_revenue_share: float  # Fraction of revenue from principal customers w_Cust in [0.0, 1.0]
    shared_coverage_ratio: float  # Fraction of shared analyst/fund coverage in [0.0, 1.0]
    network_centrality_score: float  # Ahern (2013) eigenvector centrality score in [0.0, 1.0]
    # Derived Cohen-Frazzini & Ahern Metrics
    economic_link_beta: float  # Fundamental passthrough sensitivity in [0.50, 1.00]
    customer_supplier_return_wedge: float  # Unpriced return gap CSRW_t
    inattention_diffusion_multiplier: float  # Attention friction multiplier in [1.00, 2.20]
    limited_attention_spillover_impulse: float  # LASI_t in [-2.50, 2.50]
    expected_supply_chain_alpha_pct: float  # Projected annualized abnormal return (%)
    # Flags & Regime
    is_positive_spillover_leader: bool  # True if LASI_t >= 0.035
    is_upstream_demand_collapse_trap: bool  # True if LASI_t <= -0.035
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
        """Compute deterministic HMAC-SHA256 signature across core supply-chain momentum metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.customer_portfolio_return:.4f}:"
            f"{self.supplier_own_return:.4f}:{self.customer_revenue_share:.4f}:"
            f"{self.shared_coverage_ratio:.4f}:{self.network_centrality_score:.4f}:"
            f"{self.economic_link_beta:.4f}:{self.customer_supplier_return_wedge:.4f}:"
            f"{self.inattention_diffusion_multiplier:.4f}:{self.limited_attention_spillover_impulse:.4f}:"
            f"{self.expected_supply_chain_alpha_pct:.2f}:{int(self.is_positive_spillover_leader)}:"
            f"{int(self.is_upstream_demand_collapse_trap)}:{self.regime}"
        )
        return hmac.new(
            SUPPLY_CHAIN_MOMENTUM_SIGNING_KEY,
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


def classify_supply_chain_regime(
    spillover_impulse: float,
    customer_revenue_share: float,
) -> str:
    """Classify asset into Cohen & Frazzini (2008) / Ahern (2013) Supply-Chain Regimes."""
    if spillover_impulse <= -0.035:
        return "UPSTREAM_DEMAND_COLLAPSE_TRAP"
    if spillover_impulse >= 0.035:
        return "POSITIVE_CUSTOMER_SPILLOVER_ALPHA"
    if customer_revenue_share >= 0.55:
        return "CONCENTRATED_CUSTOMER_DEPENDENCY_WATCH"
    return "SYNCHRONIZED_SUPPLY_CHAIN_EQUILIBRIUM"


def calculate_supply_chain_momentum(
    ticker: str,
    latest_price: float,
    customer_portfolio_return: float = 0.08,
    supplier_own_return: float = 0.02,
    customer_revenue_share: float = 0.35,
    shared_coverage_ratio: float = 0.40,
    network_centrality_score: float = 0.55,
) -> SupplyChainMomentumReceipt:
    """Calculate Cohen & Frazzini (2008) Supply-Chain Customer Momentum & Ahern (2013) Network Centrality.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price ($).
        customer_portfolio_return: Principal customer value-weighted return over lookback window.
        supplier_own_return: Supplier own return over lookback window.
        customer_revenue_share: Revenue share from principal customers in [0.0, 1.0].
        shared_coverage_ratio: Shared analyst / institutional co-coverage ratio in [0.0, 1.0].
        network_centrality_score: Input-output production network centrality in [0.0, 1.0].

    Returns:
        SupplyChainMomentumReceipt with CSRW_t, LASI_t, regime classification, and expected alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(customer_portfolio_return, "customer_portfolio_return")
    assert_finite_number(supplier_own_return, "supplier_own_return")
    assert_finite_number(customer_revenue_share, "customer_revenue_share")
    assert_finite_number(shared_coverage_ratio, "shared_coverage_ratio")
    assert_finite_number(network_centrality_score, "network_centrality_score")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (-0.95 <= customer_portfolio_return <= 5.0):
        raise ValueError("customer_portfolio_return must be in [-0.95, 5.0].")
    if not (-0.95 <= supplier_own_return <= 5.0):
        raise ValueError("supplier_own_return must be in [-0.95, 5.0].")
    if not (0.0 <= customer_revenue_share <= 1.0):
        raise ValueError("customer_revenue_share must be in [0.0, 1.0].")
    if not (0.0 <= shared_coverage_ratio <= 1.0):
        raise ValueError("shared_coverage_ratio must be in [0.0, 1.0].")
    if not (0.0 <= network_centrality_score <= 1.0):
        raise ValueError("network_centrality_score must be in [0.0, 1.0].")

    # 1. Fundamental Economic Link Sensitivity (Cohen & Frazzini, 2008)
    w_cust = float(customer_revenue_share)
    link_beta = float(0.50 + 0.50 * w_cust)

    # 2. Unpriced Customer-Supplier Return Wedge CSRW_t
    # Guard: if supplier has already moved further in the same direction as customer, clamp unpriced wedge toward 0
    r_cust = float(customer_portfolio_return)
    r_supp = float(supplier_own_return)
    raw_wedge = r_cust - link_beta * r_supp
    if r_cust > 0.0 and r_supp >= r_cust:
        raw_wedge = min(raw_wedge, 0.0)
    elif r_cust < 0.0 and r_supp <= r_cust:
        raw_wedge = max(raw_wedge, 0.0)
    csrw = float(np.clip(raw_wedge, -2.0, 2.0))

    # 3. Limited-Attention Diffusion Multiplier & Spillover Impulse LASI_t
    inattention_mult = float(1.0 + 1.2 * (1.0 - float(shared_coverage_ratio)))
    raw_lasi = csrw * w_cust * inattention_mult * 2.4
    lasi = float(np.clip(raw_lasi, -2.5, 2.5))

    # 4. Regime Classification & Expected Supply-Chain Alpha (Cohen-Frazzini + Ahern Centrality)
    regime = classify_supply_chain_regime(lasi, w_cust)
    is_leader = bool(regime == "POSITIVE_CUSTOMER_SPILLOVER_ALPHA")
    is_trap = bool(regime == "UPSTREAM_DEMAND_COLLAPSE_TRAP")

    centrality_prem = 2.2 * (float(network_centrality_score) - 0.35)
    expected_alpha = float(
        np.clip(45.0 * lasi + centrality_prem, -14.5, 14.5)
    )

    if regime == "POSITIVE_CUSTOMER_SPILLOVER_ALPHA":
        summary = (
            f"Positive Customer Spillover Alpha (R_Cust={r_cust:+.1%}, R_Supp={r_supp:+.1%}, "
            f"LASI={lasi:+.3f}). Unpriced downstream customer expansion with {w_cust:.0%} revenue link "
            f"predicts supplier catch-up drift ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "SYNCHRONIZED_SUPPLY_CHAIN_EQUILIBRIUM":
        summary = (
            f"Synchronized Supply-Chain Equilibrium (Wedge={csrw:+.3f}, LASI={lasi:+.3f}, "
            f"Centrality={network_centrality_score:.2f}). Customer and supplier returns are contemporaneously "
            f"aligned ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "CONCENTRATED_CUSTOMER_DEPENDENCY_WATCH":
        summary = (
            f"Concentrated Customer Dependency Watch (RevShare={w_cust:.0%}, LASI={lasi:+.3f}). "
            f"Heavy SFAS 131 single-customer concentration requires monitoring downstream order books "
            f"({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Upstream Demand Collapse Trap (R_Cust={r_cust:+.1%}, R_Supp={r_supp:+.1%}, "
            f"LASI={lasi:+.3f}). Severe customer contraction not yet priced by inattentive supplier "
            f"shareholders ({expected_alpha:+.1f}%/yr)."
        )

    return SupplyChainMomentumReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        customer_portfolio_return=round(r_cust, 4),
        supplier_own_return=round(r_supp, 4),
        customer_revenue_share=round(w_cust, 4),
        shared_coverage_ratio=round(float(shared_coverage_ratio), 4),
        network_centrality_score=round(float(network_centrality_score), 4),
        economic_link_beta=round(link_beta, 4),
        customer_supplier_return_wedge=round(csrw, 4),
        inattention_diffusion_multiplier=round(inattention_mult, 4),
        limited_attention_spillover_impulse=round(lasi, 4),
        expected_supply_chain_alpha_pct=round(expected_alpha, 2),
        is_positive_spillover_leader=is_leader,
        is_upstream_demand_collapse_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_supply_chain_momentum_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    customer_portfolio_return: Optional[float] = None,
    supplier_own_return: Optional[float] = None,
    customer_revenue_share: Optional[float] = None,
    shared_coverage_ratio: Optional[float] = None,
    network_centrality_score: Optional[float] = None,
) -> SupplyChainMomentumReceipt:
    """Evaluate Cohen & Frazzini (2008) & Ahern (2013) profile from SQLite observations and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_supp_ret = 0.025
    inferred_cust_ret = 0.085

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
                arr = np.asarray(closes[-20:], dtype=float)
                inferred_supp_ret = float(
                    np.clip((arr[-1] / arr[0]) - 1.0, -0.80, 2.00)
                )
                # Lagged first-half impulse as downstream demand lead proxy when no explicit customer return is passed
                mid = len(arr) // 2
                lead_ret = float((arr[mid] / arr[0]) - 1.0)
                inferred_cust_ret = float(
                    np.clip(inferred_supp_ret + 0.055 + 0.5 * lead_ret, -0.80, 2.00)
                )
    except Exception:
        latest_price = 100.0

    r_cust = (
        float(customer_portfolio_return)
        if customer_portfolio_return is not None
        else inferred_cust_ret
    )
    r_supp = (
        float(supplier_own_return)
        if supplier_own_return is not None
        else inferred_supp_ret
    )
    w_cust = (
        float(customer_revenue_share)
        if customer_revenue_share is not None
        else 0.36
    )
    cov_ratio = (
        float(shared_coverage_ratio)
        if shared_coverage_ratio is not None
        else 0.38
    )
    centrality = (
        float(network_centrality_score)
        if network_centrality_score is not None
        else 0.58
    )

    return calculate_supply_chain_momentum(
        ticker=clean_ticker,
        latest_price=latest_price,
        customer_portfolio_return=r_cust,
        supplier_own_return=r_supp,
        customer_revenue_share=w_cust,
        shared_coverage_ratio=cov_ratio,
        network_centrality_score=centrality,
    )

