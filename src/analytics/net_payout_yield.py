"""Boudoukh, Michaely, Richardson & Roberts (2007) / Pontiff & Woodgate (2008) Net Payout Yield Engine.

Implements the corporate capital distribution and share issuance asset pricing framework:
- Total Net Payout Yield (Boudoukh, Michaely, Richardson & Roberts, 2007):
    NPY_t = (Dividends_t + Share_Repurchases_t - Equity_Issuance_t) / Market_Cap_t
    Following SEC Rule 10b-18, share repurchases became a dominant channel of corporate cash distribution.
    Total Net Payout Yield restores and doubles cross-sectional return predictability over dividend yield alone.
- Composite Share Issuance (Daniel & Titman, 2006; Pontiff & Woodgate, 2008):
    CSI_t = ln(Shares_t / Shares_{t-1}) * 100%
    Corporate insiders time equity markets: net share repurchasers (cannibals, CSI < 0) significantly
    outperform net equity issuers / diluters (SEOs, stock M&A, heavy SBC dilution, CSI > 0) by +6.5% to +11.2%/yr.
- Free Cash Flow Payout Coverage Sustainability Guard:
    Distinguishes organic free-cash-flow funded distributions (PCR >= 1.0) from debt-funded unsustainable
    payout traps (PCR < 0.75).
- Regimes:
    - SHARE_CANNIBAL_COMPOUNDER: High sustainable net payout (NPY >= 4.5%) + net share count shrinkage (CSI <= -1.5%)
    - UNSUSTAINABLE_DEBT_FUNDED_PAYOUT: High payout (NPY >= 3.0%) uncovered by free cash flow (PCR < 0.75)
    - SHAREHOLDER_DILUTION_TRAP: Negative net payout (NPY < 0%) or heavy share dilution (CSI >= +2.5%)
    - BALANCED_CAPITAL_DISTRIBUTION: Moderate sustainable dividend/buyback profile

References:
    Boudoukh, J., Michaely, R., Richardson, M., & Roberts, M. R. (2007). On the importance of measuring payout yield:
    Implications for empirical asset pricing. The Journal of Finance, 62(2), 877-915.
    Pontiff, J., & Woodgate, A. (2008). Share issuance and cross-sectional returns.
    The Journal of Finance, 63(2), 921-945.
    Daniel, K., & Titman, S. (2006). Market reactions to tangible and intangible information.
    The Journal of Finance, 61(4), 1605-1643.
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

NET_PAYOUT_YIELD_SIGNING_KEY = b"nextomni_net_payout_yield_secret_v1"


@dataclass
class NetPayoutYieldReceipt:
    """Cryptographic audit receipt for Boudoukh et al. (2007) & Pontiff & Woodgate (2008) Payout Profile."""

    ticker: str
    latest_price: float
    market_cap: float
    # Yield Components (%)
    dividend_yield_pct: float
    gross_buyback_yield_pct: float
    gross_issuance_yield_pct: float
    net_payout_yield_pct: float  # Boudoukh et al. NPY = DY + BY - IY (%)
    # Share Count & Coverage Metrics
    current_shares: float
    prior_shares: float
    composite_share_issuance_pct: float  # Pontiff & Woodgate CSI = ln(S_t / S_{t-1}) * 100 (%)
    fcf_payout_coverage_ratio: float  # Free Cash Flow / (Dividends + Repurchases)
    expected_payout_alpha_pct: float  # Projected annualized shareholder-yield excess return (%)
    # Flags & Regime
    is_share_cannibal: bool  # True if high sustainable NPY and shrinking share count
    is_unsustainable_debt_payout: bool  # True if payout exceeds FCF capacity (PCR < 0.75)
    is_dilution_trap: bool  # True if NPY < 0 or CSI >= 2.5%
    regime: str  # SHARE_CANNIBAL_COMPOUNDER, UNSUSTAINABLE_DEBT_FUNDED_PAYOUT, SHAREHOLDER_DILUTION_TRAP, BALANCED_CAPITAL_DISTRIBUTION
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core payout & issuance metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.market_cap:.2f}:"
            f"{self.dividend_yield_pct:.4f}:{self.gross_buyback_yield_pct:.4f}:"
            f"{self.gross_issuance_yield_pct:.4f}:{self.net_payout_yield_pct:.4f}:"
            f"{self.current_shares:.4f}:{self.prior_shares:.4f}:"
            f"{self.composite_share_issuance_pct:.4f}:{self.fcf_payout_coverage_ratio:.4f}:"
            f"{self.expected_payout_alpha_pct:.2f}:{int(self.is_share_cannibal)}:"
            f"{int(self.is_unsustainable_debt_payout)}:{int(self.is_dilution_trap)}:{self.regime}"
        )
        return hmac.new(
            NET_PAYOUT_YIELD_SIGNING_KEY,
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


def classify_payout_regime(
    net_payout_yield_pct: float,
    composite_share_issuance_pct: float,
    fcf_coverage_ratio: float,
    gross_distribution_pct: float,
) -> str:
    """Classify corporate payout and share issuance regime."""
    if net_payout_yield_pct < 0.0 or composite_share_issuance_pct >= 2.50:
        return "SHAREHOLDER_DILUTION_TRAP"
    if gross_distribution_pct >= 3.0 and fcf_coverage_ratio < 0.75:
        return "UNSUSTAINABLE_DEBT_FUNDED_PAYOUT"
    if net_payout_yield_pct >= 4.50 and composite_share_issuance_pct <= -1.50:
        return "SHARE_CANNIBAL_COMPOUNDER"
    return "BALANCED_CAPITAL_DISTRIBUTION"


def calculate_net_payout_yield(
    ticker: str,
    latest_price: float,
    market_cap: float,
    dividends: float,
    share_repurchases: float,
    equity_issuance: float,
    current_shares: float,
    prior_shares: float,
    free_cash_flow: float,
) -> NetPayoutYieldReceipt:
    """Calculate Boudoukh et al. (2007) Net Payout Yield and Pontiff & Woodgate (2008) Share Issuance.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price.
        market_cap: Total equity market capitalization ($M).
        dividends: Annual cash dividends paid ($M).
        share_repurchases: Annual gross share buybacks ($M).
        equity_issuance: Annual gross equity issuance / SBC dilution ($M).
        current_shares: Split-adjusted diluted shares in current period (M).
        prior_shares: Split-adjusted diluted shares in prior period (M).
        free_cash_flow: Annual operating cash flow minus capex ($M).

    Returns:
        NetPayoutYieldReceipt with NPY%, CSI%, FCF coverage, and expected alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(market_cap, "market_cap")
    assert_finite_number(dividends, "dividends")
    assert_finite_number(share_repurchases, "share_repurchases")
    assert_finite_number(equity_issuance, "equity_issuance")
    assert_finite_number(current_shares, "current_shares")
    assert_finite_number(prior_shares, "prior_shares")
    assert_finite_number(free_cash_flow, "free_cash_flow")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if market_cap <= 0.0:
        raise ValueError("market_cap must be strictly positive.")
    if dividends < 0.0 or share_repurchases < 0.0 or equity_issuance < 0.0:
        raise ValueError("Dividends, repurchases, and issuance amounts must be non-negative.")
    if current_shares <= 0.0 or prior_shares <= 0.0:
        raise ValueError("Share counts must be strictly positive.")

    eps = 1e-6
    mcap = max(float(market_cap), eps)

    # 1. Yield Components (%)
    dy_pct = (float(dividends) / mcap) * 100.0
    by_pct = (float(share_repurchases) / mcap) * 100.0
    iy_pct = (float(equity_issuance) / mcap) * 100.0
    npy_pct = dy_pct + by_pct - iy_pct
    gross_dist_pct = dy_pct + by_pct

    # 2. Pontiff & Woodgate (2008) Composite Share Issuance (%)
    csi_pct = math.log(float(current_shares) / float(prior_shares)) * 100.0

    # 3. Free Cash Flow Payout Coverage Ratio
    total_gross_payout = float(dividends) + float(share_repurchases)
    if total_gross_payout > eps:
        raw_pcr = float(free_cash_flow) / total_gross_payout
    else:
        raw_pcr = 2.0
    pcr = float(np.clip(raw_pcr, -2.0, 5.0))

    # 4. Regime Classification & Expected Shareholder-Yield Alpha
    regime = classify_payout_regime(npy_pct, csi_pct, pcr, gross_dist_pct)
    is_cannibal = bool(regime == "SHARE_CANNIBAL_COMPOUNDER")
    is_unsustainable = bool(regime == "UNSUSTAINABLE_DEBT_FUNDED_PAYOUT")
    is_dilution = bool(regime == "SHAREHOLDER_DILUTION_TRAP")

    if is_unsustainable:
        # Penalize debt-funded payout traps where FCF cannot cover distributions
        expected_alpha = float(np.clip(-2.5 - 2.0 * (0.75 - pcr), -6.5, -1.0))
    else:
        raw_alpha = 1.15 * (npy_pct - 2.0) - 0.85 * csi_pct
        expected_alpha = float(np.clip(raw_alpha, -11.0, 11.0))

    if regime == "SHARE_CANNIBAL_COMPOUNDER":
        summary = (
            f"Share-Cannibal Compounder (NPY={npy_pct:+.2f}%, CSI={csi_pct:+.2f}%, FCF Cov={pcr:.2f}x). "
            f"High FCF-backed capital return and aggressive share count reduction drive "
            f"expected shareholder-yield alpha of {expected_alpha:+.1f}%/yr."
        )
    elif regime == "UNSUSTAINABLE_DEBT_FUNDED_PAYOUT":
        summary = (
            f"Unsustainable Debt-Funded Payout Trap (NPY={npy_pct:+.2f}%, FCF Cov={pcr:.2f}x < 0.75x). "
            f"Distributions exceed organic free cash flow capacity; elevated dividend/buyback cut risk "
            f"({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "SHAREHOLDER_DILUTION_TRAP":
        summary = (
            f"Shareholder Dilution Trap (NPY={npy_pct:+.2f}%, CSI={csi_pct:+.2f}%). "
            f"Net equity issuance and share count expansion dilute per-share intrinsic value "
            f"({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Balanced Capital Distribution (NPY={npy_pct:+.2f}%, CSI={csi_pct:+.2f}%, FCF Cov={pcr:.2f}x). "
            f"Sustainable capital return profile with {expected_alpha:+.1f}%/yr expected alpha."
        )

    return NetPayoutYieldReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        market_cap=round(mcap, 2),
        dividend_yield_pct=round(dy_pct, 2),
        gross_buyback_yield_pct=round(by_pct, 2),
        gross_issuance_yield_pct=round(iy_pct, 2),
        net_payout_yield_pct=round(npy_pct, 2),
        current_shares=round(float(current_shares), 4),
        prior_shares=round(float(prior_shares), 4),
        composite_share_issuance_pct=round(csi_pct, 2),
        fcf_payout_coverage_ratio=round(pcr, 2),
        expected_payout_alpha_pct=round(expected_alpha, 2),
        is_share_cannibal=is_cannibal,
        is_unsustainable_debt_payout=is_unsustainable,
        is_dilution_trap=is_dilution,
        regime=regime,
        summary=summary,
    )


def evaluate_net_payout_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    market_cap: Optional[float] = None,
    dividends: Optional[float] = None,
    share_repurchases: Optional[float] = None,
    equity_issuance: Optional[float] = None,
    current_shares: Optional[float] = None,
    prior_shares: Optional[float] = None,
    free_cash_flow: Optional[float] = None,
) -> NetPayoutYieldReceipt:
    """Evaluate Boudoukh et al. (2007) & Pontiff & Woodgate (2008) profile using database and heuristics."""
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

    cur_sh = float(current_shares) if current_shares is not None else 97.0
    pri_sh = float(prior_shares) if prior_shares is not None else 100.0
    mcap = float(market_cap) if market_cap is not None else (latest_price * cur_sh)
    div = float(dividends) if dividends is not None else (mcap * 0.018)
    buybacks = float(share_repurchases) if share_repurchases is not None else (mcap * 0.038)
    issuance = float(equity_issuance) if equity_issuance is not None else (mcap * 0.006)
    fcf = float(free_cash_flow) if free_cash_flow is not None else (mcap * 0.075)

    return calculate_net_payout_yield(
        ticker=clean_ticker,
        latest_price=latest_price,
        market_cap=mcap,
        dividends=div,
        share_repurchases=buybacks,
        equity_issuance=issuance,
        current_shares=cur_sh,
        prior_shares=pri_sh,
        free_cash_flow=fcf,
    )
