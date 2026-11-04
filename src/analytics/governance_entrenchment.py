"""Paul A. Gompers, Joy L. Ishii & Andrew Metrick (2003) / Lucian Bebchuk, Alma Cohen & Allen Ferrell (2009)
Corporate Governance, Managerial Entrenchment (G-Index / E-Index) & Shareholder Rights Engine.

Implements the corporate governance and managerial agency asset pricing framework:
- Gompers, Ishii & Metrick (2003) Governance Index (G-Index in [0, 24]):
    Counts 24 corporate charter, bylaw, and state anti-takeover provisions restricting shareholder rights.
    A long-short portfolio buying 'Shareholder Democracy' firms (G <= 5) and shorting 'Managerial
    Dictatorship' firms (G >= 14) earns +8.5% annualized abnormal return.
- Bebchuk, Cohen & Ferrell (2009) Entrenchment Index (E-Index in [0, 6]):
    Isolates the 6 core constitutional provisions that drive the governance valuation discount:
    (1) Staggered/Classified Boards, (2) Limits to Shareholder Bylaw Amendments, (3) Supermajority
    Requirements for Mergers, (4) Supermajority Requirements for Charter Amendments, (5) Poison Pills,
    and (6) Golden Parachutes.
- Jensen (1986) Free-Cash-Flow & Dual-Class Voting Wedge Amplification:
    Managerial entrenchment destroys the most shareholder value when combined with high unreturned
    free cash flow or unequal dual-class insider voting wedges, insulating value-destroying empire
    building from market-for-corporate-control discipline.
- Regimes:
    - SHAREHOLDER_DEMOCRACY_COMPOUNDER: Strong shareholder rights (E <= 1, G <= 6, CMES <= 0.22) (+Alpha)
    - BALANCED_GOVERNANCE_STRUCTURE: Standard corporate charter provisions (0.22 < CMES < 0.55)
    - ENTRENCHED_BOARD_AGENCY_WATCH: Elevated anti-takeover insulation (E == 4 or 0.55 <= CMES < 0.72)
    - MANAGERIAL_DICTATORSHIP_TRAP: Severe entrenchment (E >= 5, G >= 14, or CMES >= 0.72) (-Alpha trap)

References:
    Gompers, P., Ishii, J., & Metrick, A. (2003). Corporate governance and equity prices.
    The Quarterly Journal of Economics, 118(1), 107-156.
    Bebchuk, L., Cohen, A., & Ferrell, A. (2009). What matters in corporate governance?
    The Review of Financial Studies, 22(2), 783-827.
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

GOVERNANCE_ENTRENCHMENT_SIGNING_KEY = b"nextomni_governance_entrenchment_secret_v1"


@dataclass
class GovernanceEntrenchmentReceipt:
    """Cryptographic audit receipt for Gompers et al. (2003) & Bebchuk et al. (2009) Governance Profile."""

    ticker: str
    latest_price: float
    # Constitutional Governance & Agency Inputs
    bebchuk_e_index: int  # Bebchuk-Cohen-Ferrell Entrenchment Index in [0, 6]
    gompers_g_index: int  # Gompers-Ishii-Metrick Governance Index in [0, 24] (G >= E)
    dual_class_voting_wedge: float  # Insider voting share minus cash-flow share in [0.0, 0.95]
    board_independence_ratio: float  # Independent directors / total board size in [0.0, 1.0]
    unreturned_fcf_ratio: float  # Retained FCF not paid out to shareholders in [0.0, 1.0]
    # Derived Governance & Agency Metrics
    other_irrc_provisions: int  # O = G - E in [0, 18]
    composite_entrenchment_score: float  # CMES_t in [0.0, 1.0]
    agency_cost_wedge_index: float  # Jensen FCF-amplified agency wedge ACWI_t in [0.0, 2.5]
    expected_governance_alpha_pct: float  # Projected annualized abnormal return (%)
    # Flags & Regime
    is_shareholder_democracy: bool  # True if E <= 1, G <= 6, CMES <= 0.22
    is_managerial_dictatorship_trap: bool  # True if E >= 5, G >= 14, or CMES >= 0.72
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
        """Compute deterministic HMAC-SHA256 signature across core corporate governance metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.bebchuk_e_index}:"
            f"{self.gompers_g_index}:{self.dual_class_voting_wedge:.4f}:"
            f"{self.board_independence_ratio:.4f}:{self.unreturned_fcf_ratio:.4f}:"
            f"{self.other_irrc_provisions}:{self.composite_entrenchment_score:.4f}:"
            f"{self.agency_cost_wedge_index:.4f}:{self.expected_governance_alpha_pct:.2f}:"
            f"{int(self.is_shareholder_democracy)}:{int(self.is_managerial_dictatorship_trap)}:{self.regime}"
        )
        return hmac.new(
            GOVERNANCE_ENTRENCHMENT_SIGNING_KEY,
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


def classify_governance_regime(
    e_index: int,
    g_index: int,
    composite_entrenchment: float,
) -> str:
    """Classify corporate governance into Gompers et al. (2003) / Bebchuk et al. (2009) Regimes."""
    if e_index >= 5 or g_index >= 14 or composite_entrenchment >= 0.72:
        return "MANAGERIAL_DICTATORSHIP_TRAP"
    if e_index == 4 or composite_entrenchment >= 0.55:
        return "ENTRENCHED_BOARD_AGENCY_WATCH"
    if e_index <= 1 and g_index <= 6 and composite_entrenchment <= 0.22:
        return "SHAREHOLDER_DEMOCRACY_COMPOUNDER"
    return "BALANCED_GOVERNANCE_STRUCTURE"


def calculate_governance_entrenchment(
    ticker: str,
    latest_price: float,
    bebchuk_e_index: int = 1,
    gompers_g_index: int = 5,
    dual_class_voting_wedge: float = 0.0,
    board_independence_ratio: float = 0.85,
    unreturned_fcf_ratio: float = 0.20,
) -> GovernanceEntrenchmentReceipt:
    """Calculate Gompers et al. (2003) G-Index & Bebchuk et al. (2009) E-Index Governance metrics.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price ($).
        bebchuk_e_index: Bebchuk-Cohen-Ferrell Entrenchment Index in [0, 6].
        gompers_g_index: Gompers-Ishii-Metrick Governance Index in [0, 24] (must be >= bebchuk_e_index).
        dual_class_voting_wedge: Voting control share minus economic equity share in [0.0, 0.95].
        board_independence_ratio: Independent board member fraction in [0.0, 1.0].
        unreturned_fcf_ratio: Fraction of free cash flow retained without payout in [0.0, 1.0].

    Returns:
        GovernanceEntrenchmentReceipt with CMES_t, ACWI_t, regime classification, and expected alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(float(bebchuk_e_index), "bebchuk_e_index")
    assert_finite_number(float(gompers_g_index), "gompers_g_index")
    assert_finite_number(dual_class_voting_wedge, "dual_class_voting_wedge")
    assert_finite_number(board_independence_ratio, "board_independence_ratio")
    assert_finite_number(unreturned_fcf_ratio, "unreturned_fcf_ratio")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    e_val = int(bebchuk_e_index)
    g_val = int(gompers_g_index)
    if not (0 <= e_val <= 6):
        raise ValueError("bebchuk_e_index must be an integer in [0, 6].")
    if not (0 <= g_val <= 24):
        raise ValueError("gompers_g_index must be an integer in [0, 24].")
    if g_val < e_val:
        raise ValueError("gompers_g_index must be greater than or equal to bebchuk_e_index.")
    if (g_val - e_val) > 18:
        raise ValueError("other_irrc_provisions (g_index - e_index) cannot exceed 18.")
    if not (0.0 <= dual_class_voting_wedge <= 0.95):
        raise ValueError("dual_class_voting_wedge must be in [0.0, 0.95].")
    if not (0.0 <= board_independence_ratio <= 1.0):
        raise ValueError("board_independence_ratio must be in [0.0, 1.0].")
    if not (0.0 <= unreturned_fcf_ratio <= 1.0):
        raise ValueError("unreturned_fcf_ratio must be in [0.0, 1.0].")

    # 1. Decompose G-Index into E-Index (6 core provisions) and Other IRRC Provisions (O = G - E <= 18)
    other_irrc = g_val - e_val

    # 2. Composite Managerial Entrenchment Score CMES_t in [0.0, 1.0]
    e_comp = float(e_val) / 6.0
    o_comp = float(other_irrc) / 18.0
    wedge_comp = min(float(dual_class_voting_wedge) / 0.50, 1.0)
    indep_bonus = max(float(board_independence_ratio) - 0.75, 0.0)
    raw_cmes = 0.55 * e_comp + 0.20 * o_comp + 0.25 * wedge_comp - 0.10 * indep_bonus
    cmes = float(np.clip(raw_cmes, 0.0, 1.0))

    # 3. Jensen (1986) Free-Cash-Flow Agency Cost Wedge Index ACWI_t in [0.0, 2.5]
    raw_acwi = cmes * (1.0 + 1.5 * float(unreturned_fcf_ratio)) * 2.0
    acwi = float(np.clip(raw_acwi, 0.0, 2.5))

    # 4. Regime Classification & Expected Governance Alpha (Gompers et al. +8.5%/yr Democracy-Dictatorship spread)
    regime = classify_governance_regime(e_val, g_val, cmes)
    is_democracy = bool(regime == "SHAREHOLDER_DEMOCRACY_COMPOUNDER")
    is_dictatorship = bool(regime == "MANAGERIAL_DICTATORSHIP_TRAP")

    expected_alpha = float(
        np.clip(4.6 - 5.2 * acwi - 3.5 * cmes, -10.5, 5.0)
    )

    if regime == "SHAREHOLDER_DEMOCRACY_COMPOUNDER":
        summary = (
            f"Shareholder Democracy Compounder (E-Index={e_val}/6, G-Index={g_val}/24, CMES={cmes:.2f}). "
            f"Strong shareholder voting accountability and board independence ({board_independence_ratio:.0%}) "
            f"command a Gompers-Bebchuk governance premium ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "BALANCED_GOVERNANCE_STRUCTURE":
        summary = (
            f"Balanced Governance Structure (E-Index={e_val}/6, G-Index={g_val}/24, CMES={cmes:.2f}). "
            f"Moderate charter anti-takeover provisions with {expected_alpha:+.1f}%/yr expected governance alpha."
        )
    elif regime == "ENTRENCHED_BOARD_AGENCY_WATCH":
        summary = (
            f"Entrenched Board Agency Watch (E-Index={e_val}/6, G-Index={g_val}/24, AgencyWedge={acwi:.2f}). "
            f"Elevated takeover insulation; monitor capital reinvestment and executive compensation "
            f"({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Managerial Dictatorship Trap (E-Index={e_val}/6, G-Index={g_val}/24, CMES={cmes:.2f}, "
            f"AgencyWedge={acwi:.2f}). Severe constitutional entrenchment and voting insulation destroy "
            f"shareholder value ({expected_alpha:+.1f}%/yr)."
        )

    return GovernanceEntrenchmentReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        bebchuk_e_index=e_val,
        gompers_g_index=g_val,
        dual_class_voting_wedge=round(float(dual_class_voting_wedge), 4),
        board_independence_ratio=round(float(board_independence_ratio), 4),
        unreturned_fcf_ratio=round(float(unreturned_fcf_ratio), 4),
        other_irrc_provisions=other_irrc,
        composite_entrenchment_score=round(cmes, 4),
        agency_cost_wedge_index=round(acwi, 4),
        expected_governance_alpha_pct=round(expected_alpha, 2),
        is_shareholder_democracy=is_democracy,
        is_managerial_dictatorship_trap=is_dictatorship,
        regime=regime,
        summary=summary,
    )


def evaluate_governance_entrenchment_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    bebchuk_e_index: Optional[int] = None,
    gompers_g_index: Optional[int] = None,
    dual_class_voting_wedge: Optional[float] = None,
    board_independence_ratio: Optional[float] = None,
    unreturned_fcf_ratio: Optional[float] = None,
) -> GovernanceEntrenchmentReceipt:
    """Evaluate Gompers et al. (2003) & Bebchuk et al. (2009) Governance profile from SQLite and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
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
    except Exception:
        latest_price = 100.0

    e_val = int(bebchuk_e_index) if bebchuk_e_index is not None else 1
    g_val = int(gompers_g_index) if gompers_g_index is not None else max(5, e_val)
    wedge = (
        float(dual_class_voting_wedge)
        if dual_class_voting_wedge is not None
        else 0.0
    )
    board_indep = (
        float(board_independence_ratio)
        if board_independence_ratio is not None
        else 0.85
    )
    fcf_ret = (
        float(unreturned_fcf_ratio)
        if unreturned_fcf_ratio is not None
        else 0.22
    )

    return calculate_governance_entrenchment(
        ticker=clean_ticker,
        latest_price=latest_price,
        bebchuk_e_index=e_val,
        gompers_g_index=g_val,
        dual_class_voting_wedge=wedge,
        board_independence_ratio=board_indep,
        unreturned_fcf_ratio=fcf_ret,
    )

