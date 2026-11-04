"""Cooper, Gulen & Schill (2008) / Titman, Wei & Xie (2004) Corporate Asset Growth & Empire-Building Engine.

Implements the corporate investment, asset expansion, and agency-cost framework:
- Total Asset Growth Anomaly (Cooper, Gulen & Schill, 2008):
    AG_t = (Total_Assets_t - Total_Assets_{t-1}) / Total_Assets_{t-1}
    Year-over-year corporate asset expansion is one of the strongest negative predictors of future
    cross-sectional stock returns (-8% to -13% annualized spread between high and low asset growth deciles).
- Abnormal Capital Investment (Titman, Wei & Xie, 2004):
    ACI_t = (CapEx_t / Sales_t) / Mean_3yr(CapEx / Sales) - 1
    Firms that substantially increase capital expenditures relative to their 3-year historical benchmark
    subsequently underperform, as investors over-extrapolate empire-building expansion.
- Jensen (1986) Free Cash Flow Agency Amplification:
    The negative return anomaly of abnormal investment is concentrated in firms with high free cash flow
    and low debt oversight, where managers have greater discretion to engage in empire building.
- Regimes:
    - DISCIPLINED_CAPITAL_ALLOCATOR: Restrained asset growth / asset shrinkage with disciplined capex (+Alpha)
    - SUSTAINABLE_ORGANIC_GROWTH: Balanced reinvestment aligned with historical productivity
    - ELEVATED_EXPANSION_WATCH: Above-average balance sheet expansion requiring ROI monitoring
    - EMPIRE_BUILDING_OVERINVESTOR: Aggressive asset bloat + abnormal capex surge (-Alpha trap)

References:
    Cooper, M. J., Gulen, H., & Schill, M. J. (2008). Asset growth and the cross-section of stock returns.
    The Journal of Finance, 63(4), 1609-1651.
    Titman, S., Wei, K. C. J., & Xie, F. (2004). Capital investments and stock returns.
    Journal of Financial and Quantitative Analysis, 39(4), 677-700.
    Jensen, M. C. (1986). Agency costs of free cash flow, corporate finance, and takeovers.
    American Economic Review, 76(2), 323-329.
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

ASSET_GROWTH_EMPIRE_SIGNING_KEY = b"nextomni_asset_growth_empire_secret_v1"


@dataclass
class AssetGrowthEmpireReceipt:
    """Cryptographic audit receipt for Cooper et al. (2008) & Titman et al. (2004) Asset Growth Profile."""

    ticker: str
    latest_price: float
    # Balance Sheet & CapEx Inputs
    current_total_assets: float
    prior_total_assets: float
    asset_growth_pct: float  # YoY Total Asset Growth (%)
    current_capex_to_sales_pct: float  # Current CapEx / Sales (%)
    historical_3yr_capex_to_sales_pct: float  # 3-Year Benchmark CapEx / Sales (%)
    abnormal_capital_investment: float  # Titman-Wei-Xie ACI ratio
    # Agency & Overinvestment Inferences
    jensen_agency_discretion_index: float  # Free cash flow / low-leverage agency index in [0, 1]
    empire_building_score: float  # Composite Overinvestment Z-Score in [-3.0, +3.0]
    expected_allocation_alpha_pct: float  # Projected annualized excess return (%)
    # Flags & Regime
    is_disciplined_allocator: bool  # True if disciplined asset/capex restraint
    is_empire_builder_trap: bool  # True if aggressive asset bloat + abnormal capex surge
    regime: str  # DISCIPLINED_CAPITAL_ALLOCATOR, SUSTAINABLE_ORGANIC_GROWTH, ELEVATED_EXPANSION_WATCH, EMPIRE_BUILDING_OVERINVESTOR
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core asset growth metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.current_total_assets:.2f}:"
            f"{self.prior_total_assets:.2f}:{self.asset_growth_pct:.4f}:"
            f"{self.current_capex_to_sales_pct:.4f}:{self.historical_3yr_capex_to_sales_pct:.4f}:"
            f"{self.abnormal_capital_investment:.4f}:{self.jensen_agency_discretion_index:.4f}:"
            f"{self.empire_building_score:.4f}:{self.expected_allocation_alpha_pct:.2f}:"
            f"{int(self.is_disciplined_allocator)}:{int(self.is_empire_builder_trap)}:{self.regime}"
        )
        return hmac.new(
            ASSET_GROWTH_EMPIRE_SIGNING_KEY,
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


def classify_empire_regime(
    empire_building_score: float,
    asset_growth_pct: float,
    abnormal_capex: float,
) -> str:
    """Classify corporate capital allocation regime under Cooper et al. (2008) & Titman et al. (2004)."""
    if empire_building_score >= 1.25:
        return "EMPIRE_BUILDING_OVERINVESTOR"
    if empire_building_score >= 0.65:
        return "ELEVATED_EXPANSION_WATCH"
    if empire_building_score <= -0.40 or (asset_growth_pct <= 5.0 and abnormal_capex <= 0.0):
        return "DISCIPLINED_CAPITAL_ALLOCATOR"
    return "SUSTAINABLE_ORGANIC_GROWTH"


def calculate_asset_growth_empire(
    ticker: str,
    latest_price: float,
    current_total_assets: float,
    prior_total_assets: float,
    current_capex_to_sales_pct: float = 6.0,
    historical_3yr_capex_to_sales_pct: float = 6.0,
    fcf_to_assets_pct: float = 6.0,
    debt_to_assets_pct: float = 30.0,
) -> AssetGrowthEmpireReceipt:
    """Compute Cooper et al. (2008) Asset Growth and Titman et al. (2004) Abnormal Capital Investment.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price.
        current_total_assets: Total assets in current fiscal period ($M).
        prior_total_assets: Total assets in prior fiscal period ($M).
        current_capex_to_sales_pct: Current capital expenditure to sales ratio (%).
        historical_3yr_capex_to_sales_pct: Trailing 3-year average CapEx-to-Sales ratio (%).
        fcf_to_assets_pct: Free cash flow to total assets ratio (%).
        debt_to_assets_pct: Total debt to total assets ratio (%).

    Returns:
        AssetGrowthEmpireReceipt with AG%, ACI, Jensen Agency Index, and expected alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(current_total_assets, "current_total_assets")
    assert_finite_number(prior_total_assets, "prior_total_assets")
    assert_finite_number(current_capex_to_sales_pct, "current_capex_to_sales_pct")
    assert_finite_number(historical_3yr_capex_to_sales_pct, "historical_3yr_capex_to_sales_pct")
    assert_finite_number(fcf_to_assets_pct, "fcf_to_assets_pct")
    assert_finite_number(debt_to_assets_pct, "debt_to_assets_pct")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if current_total_assets <= 0.0 or prior_total_assets <= 0.0:
        raise ValueError("Total assets must be strictly positive.")
    if current_capex_to_sales_pct < 0.0 or historical_3yr_capex_to_sales_pct < 0.0:
        raise ValueError("CapEx-to-Sales ratios must be non-negative.")

    eps = 1e-6
    prior_ta = max(float(prior_total_assets), eps)
    cur_ta = max(float(current_total_assets), eps)

    # 1. Cooper, Gulen & Schill (2008) Year-over-Year Total Asset Growth (%)
    ag_pct = ((cur_ta - prior_ta) / prior_ta) * 100.0

    # 2. Titman, Wei & Xie (2004) Abnormal Capital Investment (ACI)
    bench_capex = max(float(historical_3yr_capex_to_sales_pct), 0.10)
    raw_aci = (float(current_capex_to_sales_pct) / bench_capex) - 1.0
    aci = float(np.clip(raw_aci, -1.0, 5.0))

    # 3. Jensen (1986) Free Cash Flow Agency Discretion Index in [0.0, 1.0]
    fcf_ratio = float(fcf_to_assets_pct) / 100.0
    debt_ratio = float(debt_to_assets_pct) / 100.0
    raw_jai = 0.50 + 3.0 * (fcf_ratio - 0.05) - 0.8 * (debt_ratio - 0.30)
    jai = float(np.clip(raw_jai, 0.0, 1.0))

    # 4. Composite Empire-Building Overinvestment Score in [-3.0, +3.0]
    ag_component = (ag_pct - 8.0) / 15.0
    aci_component = aci / 0.35
    agency_multiplier = 0.70 + 0.60 * jai
    raw_ebos = (0.60 * ag_component + 0.40 * aci_component) * agency_multiplier
    ebos = float(np.clip(raw_ebos, -3.0, 3.0))

    # 5. Regime & Expected Capital Allocation Alpha
    regime = classify_empire_regime(ebos, ag_pct, aci)
    is_disciplined = bool(regime == "DISCIPLINED_CAPITAL_ALLOCATOR")
    is_trap = bool(regime == "EMPIRE_BUILDING_OVERINVESTOR")

    expected_alpha = float(np.clip(-4.2 * ebos, -13.0, 10.0))

    if regime == "DISCIPLINED_CAPITAL_ALLOCATOR":
        summary = (
            f"Disciplined Capital Allocator (AG={ag_pct:+.1f}%, ACI={aci:+.2f}, EBOS={ebos:+.2f}). "
            f"Restrained balance sheet growth and capex discipline avoid overinvestment drag. "
            f"Expected capital allocation alpha: {expected_alpha:+.1f}%/yr."
        )
    elif regime == "SUSTAINABLE_ORGANIC_GROWTH":
        summary = (
            f"Sustainable Organic Growth (AG={ag_pct:+.1f}%, ACI={aci:+.2f}, EBOS={ebos:+.2f}). "
            f"Balanced corporate reinvestment aligned with historical capital intensity."
        )
    elif regime == "ELEVATED_EXPANSION_WATCH":
        summary = (
            f"Elevated Expansion Watch (AG={ag_pct:+.1f}%, ACI={aci:+.2f}, EBOS={ebos:+.2f}). "
            f"Accelerating asset and capital expenditure expansion; monitor incremental ROIC."
        )
    else:
        summary = (
            f"Empire-Building Overinvestor Trap (AG={ag_pct:+.1f}%, ACI={aci:+.2f}, EBOS={ebos:+.2f}, "
            f"Agency={jai:.2f}). Severe asset bloat and abnormal capex surge predict negative future returns "
            f"({expected_alpha:+.1f}%/yr)."
        )

    return AssetGrowthEmpireReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        current_total_assets=round(cur_ta, 2),
        prior_total_assets=round(prior_ta, 2),
        asset_growth_pct=round(ag_pct, 2),
        current_capex_to_sales_pct=round(float(current_capex_to_sales_pct), 2),
        historical_3yr_capex_to_sales_pct=round(float(historical_3yr_capex_to_sales_pct), 2),
        abnormal_capital_investment=round(aci, 4),
        jensen_agency_discretion_index=round(jai, 4),
        empire_building_score=round(ebos, 4),
        expected_allocation_alpha_pct=round(expected_alpha, 2),
        is_disciplined_allocator=is_disciplined,
        is_empire_builder_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_asset_growth_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    current_total_assets: Optional[float] = None,
    prior_total_assets: Optional[float] = None,
    current_capex_to_sales_pct: Optional[float] = None,
    historical_3yr_capex_to_sales_pct: Optional[float] = None,
    fcf_to_assets_pct: Optional[float] = None,
    debt_to_assets_pct: Optional[float] = None,
) -> AssetGrowthEmpireReceipt:
    """Evaluate Cooper et al. (2008) & Titman et al. (2004) profile using database and heuristics."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    try:
        obs = db.get_observations(clean_ticker, db_path=db_path)
        if obs is not None and not obs.empty and "close" in obs.columns:
            closes = [float(x) for x in obs["close"].tolist() if math.isfinite(float(x)) and float(x) > 0]
            if closes:
                latest_price = closes[-1]
    except Exception:
        latest_price = 100.0

    prior_ta = float(prior_total_assets) if prior_total_assets is not None else 10_000.0
    cur_ta = float(current_total_assets) if current_total_assets is not None else 10_400.0
    cur_capex = float(current_capex_to_sales_pct) if current_capex_to_sales_pct is not None else 5.8
    hist_capex = float(historical_3yr_capex_to_sales_pct) if historical_3yr_capex_to_sales_pct is not None else 6.2
    fcf_pct = float(fcf_to_assets_pct) if fcf_to_assets_pct is not None else 6.5
    debt_pct = float(debt_to_assets_pct) if debt_to_assets_pct is not None else 28.0

    return calculate_asset_growth_empire(
        ticker=clean_ticker,
        latest_price=latest_price,
        current_total_assets=cur_ta,
        prior_total_assets=prior_ta,
        current_capex_to_sales_pct=cur_capex,
        historical_3yr_capex_to_sales_pct=hist_capex,
        fcf_to_assets_pct=fcf_pct,
        debt_to_assets_pct=debt_pct,
    )
