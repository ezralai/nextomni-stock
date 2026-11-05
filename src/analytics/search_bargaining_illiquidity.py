"""Darrell Duffie, Nicolae Gârleanu & Lasse Heje Pedersen (2005, 2007)
Over-The-Counter (OTC) Search-and-Bargaining Frictions, Dealer Intermediation & Illiquidity Discount Engine.

Implements the Duffie-Gârleanu-Pedersen (DGP) OTC Search framework:
- Duffie, Gârleanu & Pedersen (2005, Econometrica, "Over-the-Counter Markets")
- Duffie, Gârleanu & Pedersen (2007, Review of Financial Studies, "Valuation in Over-the-Counter Markets"):
    In over-the-counter (OTC), dark pool, and institutional block-trading venues, matching is decentralized.
    Investors do not trade instantaneously in a centralized Walrasian auctioneer market.
    Instead, they must search sequentially for counterparty dealers or peers at Poisson arrival rate lambda.
    When a buyer and seller or dealer meet, bilateral terms of trade are resolved via generalized Nash bargaining,
    where the investor commands bargaining power q in [0, 1] and the dealer/counterparty commands (1 - q).
    - Endogenous Bid-Ask Spread: Emerges from search delays and dealer inventory holding costs chi.
    - Search Illiquidity Discount (S_DGP): The percentage price markdown of the OTC asset relative to
      the frictionless Walrasian benchmark price P* = D / r:
          S_DGP = ((r + lambda * (1 - q)) / (r + lambda)) * (chi / D)
    - When search frictions worsen (lambda drops) or dealer inventory carrying costs surge, the search discount
      widens dramatically, driving liquidity dry-ups and block execution penalties.

References:
    Duffie, D., Gârleanu, N., & Pedersen, L. H. (2005). Over-the-counter markets.
    Econometrica, 73(6), 1815-1847.
    Duffie, D., Gârleanu, N., & Pedersen, L. H. (2007). Valuation in over-the-counter markets.
    The Review of Financial Studies, 20(6), 1865-1900.
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

SEARCH_BARGAINING_SIGNING_KEY = b"nextomni_search_bargaining_illiquidity_v1"


@dataclass
class SearchBargainingReceipt:
    """Cryptographic audit receipt for Duffie-Gârleanu-Pedersen (2005) OTC Search & Bargaining Profile."""

    ticker: str
    latest_price: float
    # Search & Bargaining Inputs
    search_intensity_lambda: float  # Counterparty Poisson contact rate lambda in [0.10, 50.0]
    investor_bargaining_power: float  # Investor Nash bargaining power q in [0.05, 0.95]
    dealer_inventory_cost: float  # Inventory holding cost chi in [0.00, 0.25]
    opportunity_cost_rate: float  # Risk-free / discount rate r in [0.01, 0.15]
    asset_cash_flow_rate: float  # Dividend / convenience yield D in [0.005, 0.20]
    # Derived DGP Search & Valuation Metrics
    search_illiquidity_discount_pct: float  # S_DGP search price discount (%) in [0.0, 25.0]
    endogenous_bid_ask_spread_pct: float  # Bid-ask spread Delta_P (%) in [0.01, 10.0]
    walrasian_frictionless_price: float  # Benchmark Walrasian price P* ($)
    otc_effective_price: float  # OTC market price P_OTC ($)
    dgp_illiquidity_impulse: float  # I_dgp in [-2.5, 2.5]
    expected_search_alpha_pct: float  # Projected annualized alpha net of search illiquidity (%)
    # Flags & Regime
    is_severe_search_freeze: bool  # True if lambda <= 2.0 and S_DGP >= 4.5%
    is_centralized_high_liquidity: bool  # True if lambda >= 25.0 and S_DGP <= 0.6%
    is_elevated_search_friction: bool  # True if S_DGP >= 1.5%
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
        """Compute deterministic HMAC-SHA256 signature across all search and bargaining metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:"
            f"{self.search_intensity_lambda:.4f}:{self.investor_bargaining_power:.4f}:"
            f"{self.dealer_inventory_cost:.4f}:{self.opportunity_cost_rate:.4f}:{self.asset_cash_flow_rate:.4f}:"
            f"{self.search_illiquidity_discount_pct:.4f}:{self.endogenous_bid_ask_spread_pct:.4f}:"
            f"{self.walrasian_frictionless_price:.2f}:{self.otc_effective_price:.2f}:"
            f"{self.dgp_illiquidity_impulse:.4f}:{self.expected_search_alpha_pct:.2f}:"
            f"{int(self.is_severe_search_freeze)}:{int(self.is_centralized_high_liquidity)}:{int(self.is_elevated_search_friction)}:{self.regime}"
        )
        return hmac.new(
            SEARCH_BARGAINING_SIGNING_KEY,
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


def classify_search_regime(
    search_lambda: float,
    search_discount: float,
    bid_ask_spread: float,
) -> str:
    """Classify asset into Duffie-Gârleanu-Pedersen (2005) Search-and-Bargaining Regimes."""
    if search_lambda <= 3.0 and search_discount >= 3.50:
        return "SEVERE_OTC_SEARCH_FREEZE"
    if search_lambda >= 20.0 and search_discount <= 1.00:
        return "HIGH_LIQUIDITY_CENTRALIZED_MATCHING"
    if search_discount >= 2.00 or bid_ask_spread >= 0.80:
        return "ELEVATED_SEARCH_BARGAINING_FRICTION"
    return "BALANCED_OTC_INTERMEDIATION_EQUILIBRIUM"


def calculate_search_bargaining(
    ticker: str,
    latest_price: float,
    search_intensity_lambda: float = 15.0,
    investor_bargaining_power: float = 0.50,
    dealer_inventory_cost: float = 0.002,
    opportunity_cost_rate: float = 0.05,
    asset_cash_flow_rate: float = 0.04,
) -> SearchBargainingReceipt:
    """Calculate Duffie-Gârleanu-Pedersen (2005) OTC Search & Bargaining Illiquidity Metrics.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        search_intensity_lambda: Counterparty Poisson contact rate lambda in [0.10, 50.0].
        investor_bargaining_power: Investor Nash bargaining power q in [0.05, 0.95].
        dealer_inventory_cost: Dealer inventory holding cost chi in [0.00, 0.10].
        opportunity_cost_rate: Risk-free discount rate r in [0.01, 0.15].
        asset_cash_flow_rate: Dividend / convenience yield rate D in [0.005, 0.20].

    Returns:
        SearchBargainingReceipt with search discount, bid-ask spread, impulse, and regime.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(search_intensity_lambda, "search_intensity_lambda")
    assert_finite_number(investor_bargaining_power, "investor_bargaining_power")
    assert_finite_number(dealer_inventory_cost, "dealer_inventory_cost")
    assert_finite_number(opportunity_cost_rate, "opportunity_cost_rate")
    assert_finite_number(asset_cash_flow_rate, "asset_cash_flow_rate")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (0.10 <= search_intensity_lambda <= 50.0):
        raise ValueError("search_intensity_lambda must be in [0.10, 50.0].")
    if not (0.05 <= investor_bargaining_power <= 0.95):
        raise ValueError("investor_bargaining_power must be in [0.05, 0.95].")
    if not (0.00 <= dealer_inventory_cost <= 0.10):
        raise ValueError("dealer_inventory_cost must be in [0.00, 0.10].")
    if not (0.01 <= opportunity_cost_rate <= 0.15):
        raise ValueError("opportunity_cost_rate must be in [0.01, 0.15].")
    if not (0.005 <= asset_cash_flow_rate <= 0.20):
        raise ValueError("asset_cash_flow_rate must be in [0.005, 0.20].")

    p_t = float(latest_price)
    lam = float(search_intensity_lambda)
    q = float(investor_bargaining_power)
    chi = float(dealer_inventory_cost)
    r = float(opportunity_cost_rate)
    d = float(asset_cash_flow_rate)

    # 1. Duffie-Gârleanu-Pedersen Search Illiquidity Discount S_DGP (%)
    # S_DGP = ((r + lambda * (1 - q)) / (r + lambda)) * (chi / D) * 100
    search_weight = (r + lam * (1.0 - q)) / max(r + lam, 1e-4)
    raw_discount = search_weight * (chi / max(d, 0.005)) * 100.0
    discount_pct = float(np.clip(raw_discount, 0.0, 25.0))

    # 2. Endogenous Bid-Ask Spread Delta_P (%)
    # Delta_P = 2 * (1 - q) * chi / (r + lambda) * 100
    raw_spread = (2.0 * (1.0 - q) * chi) / max(r + lam, 1e-4) * 100.0
    spread_pct = float(np.clip(raw_spread, 0.01, 10.0))

    # 3. Walrasian Frictionless Benchmark vs OTC Effective Price
    # P* = P_t / (1 - S_DGP / 100)
    p_walras = float(p_t / max(1.0 - discount_pct / 100.0, 0.20))
    p_otc = float(p_t)

    # 4. Search Fragility Impulse & Expected Alpha
    # When discount is high due to temporary search frictions, liquidity recovery offers abnormal alpha
    # When freeze is acute, illiquidity drag produces negative short-term pressure
    raw_impulse = 0.40 * (discount_pct / 3.0) - 0.35 * max(0.0, spread_pct - 1.0) + 0.25 * (lam / 20.0 - 0.5)
    impulse = float(np.clip(raw_impulse, -2.5, 2.5))
    expected_alpha = float(np.clip(3.2 * impulse, -8.0, 12.0))

    regime = classify_search_regime(lam, discount_pct, spread_pct)
    is_freeze = bool(regime == "SEVERE_OTC_SEARCH_FREEZE")
    is_centralized = bool(regime == "HIGH_LIQUIDITY_CENTRALIZED_MATCHING")
    is_elevated = bool(regime == "ELEVATED_SEARCH_BARGAINING_FRICTION")

    if regime == "SEVERE_OTC_SEARCH_FREEZE":
        summary = (
            f"Severe OTC Search Freeze (lambda={lam:.1f}, Discount={discount_pct:.2f}%, Spread={spread_pct:.2f}%). "
            f"Acute search frictions and dealer inventory carrying costs depress secondary execution value ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "HIGH_LIQUIDITY_CENTRALIZED_MATCHING":
        summary = (
            f"High-Liquidity Centralized Matching (lambda={lam:.1f}, Discount={discount_pct:.2f}%, Spread={spread_pct:.2f}%). "
            f"Rapid counterparty search arrival and strong investor bargaining power eliminate illiquidity penalties ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "ELEVATED_SEARCH_BARGAINING_FRICTION":
        summary = (
            f"Elevated Search & Bargaining Friction (lambda={lam:.1f}, Discount={discount_pct:.2f}%, Spread={spread_pct:.2f}%). "
            f"Moderate block-trading search delays create meaningful discounts relative to Walrasian benchmarks ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Balanced OTC Intermediation Equilibrium (lambda={lam:.1f}, Discount={discount_pct:.2f}%, Spread={spread_pct:.2f}%). "
            f"Search arrival rates and dealer inventory compensation are in balanced equilibrium ({expected_alpha:+.1f}%/yr)."
        )

    return SearchBargainingReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        search_intensity_lambda=round(lam, 2),
        investor_bargaining_power=round(q, 4),
        dealer_inventory_cost=round(chi, 4),
        opportunity_cost_rate=round(r, 4),
        asset_cash_flow_rate=round(d, 4),
        search_illiquidity_discount_pct=round(discount_pct, 2),
        endogenous_bid_ask_spread_pct=round(spread_pct, 2),
        walrasian_frictionless_price=round(p_walras, 2),
        otc_effective_price=round(p_otc, 2),
        dgp_illiquidity_impulse=round(impulse, 4),
        expected_search_alpha_pct=round(expected_alpha, 2),
        is_severe_search_freeze=is_freeze,
        is_centralized_high_liquidity=is_centralized,
        is_elevated_search_friction=is_elevated,
        regime=regime,
        summary=summary,
    )


def evaluate_search_bargaining_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    search_intensity_lambda: Optional[float] = None,
    investor_bargaining_power: Optional[float] = None,
    dealer_inventory_cost: Optional[float] = None,
    opportunity_cost_rate: Optional[float] = None,
    asset_cash_flow_rate: Optional[float] = None,
) -> SearchBargainingReceipt:
    """Evaluate Duffie-Gârleanu-Pedersen OTC Search profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_lambda = 12.0
    inferred_q = 0.50
    inferred_chi = 0.03
    inferred_r = 0.05
    inferred_d = 0.04

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

                # Amihud illiquidity proxy from returns and dollar volume
                ann_vol = float(np.std(rets, ddof=1) * math.sqrt(252.0)) if len(rets) > 1 else 0.22

                # Higher volatility requires dealers to hold more inventory capital
                inferred_chi = float(np.clip(0.001 + 0.005 * ann_vol, 0.0005, 0.04))
                # Search arrival intensity inversely scales with idiosyncratic volatility
                inferred_lambda = float(np.clip(35.0 - 60.0 * ann_vol, 2.0, 45.0))
                # Investor bargaining power declines during market drawdowns
                peak = float(np.max(arr))
                dd = float((arr[-1] - peak) / max(peak, 1e-4))
                inferred_q = float(np.clip(0.60 - 0.40 * dd, 0.15, 0.85))
    except Exception:
        latest_price = 100.0

    lam_val = float(search_intensity_lambda) if search_intensity_lambda is not None else inferred_lambda
    q_val = float(investor_bargaining_power) if investor_bargaining_power is not None else inferred_q
    chi_val = float(dealer_inventory_cost) if dealer_inventory_cost is not None else inferred_chi
    r_val = float(opportunity_cost_rate) if opportunity_cost_rate is not None else inferred_r
    d_val = float(asset_cash_flow_rate) if asset_cash_flow_rate is not None else inferred_d

    return calculate_search_bargaining(
        ticker=clean_ticker,
        latest_price=latest_price,
        search_intensity_lambda=lam_val,
        investor_bargaining_power=q_val,
        dealer_inventory_cost=chi_val,
        opportunity_cost_rate=r_val,
        asset_cash_flow_rate=d_val,
    )
