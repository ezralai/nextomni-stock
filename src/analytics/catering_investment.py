"""Christopher Polk & Paola Sapienza (2009) Catering Theory of Corporate Investment & Valuation Mispricing Engine.

Implements the behavioral corporate investment and catering asset pricing framework:
- Catering Theory of Investment (Polk & Sapienza, 2009):
    When equity prices deviate from fundamental value (measured via discretionary accruals or valuation
    mispricing proxies) and shareholders have short investment horizons (high share turnover), rational
    managers cater to investor sentiment by over-investing in physical capital (CapEx_t / K_{t-1}) even
    without issuing new shares.
- Neoclassical Q vs. Residual Abnormal Investment Decomposition:
    IK_t = CapEx_t / K_{t-1}
    IK_hat_t = 0.08 + 0.045 * max(Tobins_Q_t - 1.0, 0.0) + 0.15 * CashFlowRatio_t
    RAI_t = clip(IK_t - IK_hat_t, -1.0, 1.5)
- Catering Distortion Index (CDI_t):
    Amplifies residual abnormal investment by positive equity mispricing (DMP_t) and short-horizon
    shareholder turnover multiplier (SHTM_t in [0.5, 2.0]).
- Regimes:
    - DISCIPLINED_FUNDAMENTAL_ALLOCATOR: Restrained abnormal investment (RAI_t <= 0.02) and low catering distortion (+Alpha)
    - NEOCLASSICAL_Q_EXPANSION: High capital investment (IK_t >= 0.20) justified by Tobin's Q and cash flow
    - SENTIMENT_CATERING_WATCH: Moderate catering distortion (0.45 <= CDI_t < 1.10) requiring capital allocation scrutiny
    - OVERVALUED_CATERING_CAPEX_TRAP: Severe catering-driven over-investment (CDI_t >= 1.10 or (RAI_t >= 0.15 and DMP_t >= 0.45)) (-Alpha trap)

References:
    Polk, C., & Sapienza, P. (2009). The stock market and corporate investment: A test of catering theory.
    The Review of Financial Studies, 22(1), 187-217.
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

CATERING_INVESTMENT_SIGNING_KEY = b"nextomni_catering_investment_secret_v1"


@dataclass
class CateringInvestmentReceipt:
    """Cryptographic audit receipt for Polk & Sapienza (2009) Catering Theory of Investment Profile."""

    ticker: str
    latest_price: float
    # Capital Investment & Valuation Inputs
    capex_amount: float  # Capital expenditures in $M
    lagged_net_ppe: float  # Lagged net Property, Plant & Equipment K_{t-1} in $M
    tobins_q: float  # Market-to-book asset valuation Q_t
    cash_flow_to_capital: float  # Operating cash flow / K_{t-1}
    discretionary_mispricing_score: float  # Equity overvaluation proxy DMP_t in [-1.0, 1.0]
    annual_turnover_pct: float  # Annualized share turnover (%)
    # Decomposed Catering Metrics
    investment_to_capital_ratio: float  # IK_t = CapEx_t / K_{t-1}
    fundamental_q_investment_rate: float  # Neoclassical Q & CF benchmark investment rate
    residual_abnormal_investment: float  # RAI_t = IK_t - IK_hat_t
    short_horizon_turnover_multiplier: float  # SHTM_t in [0.5, 2.0]
    catering_distortion_index: float  # CDI_t in [-1.5, 3.0]
    expected_catering_alpha_pct: float  # Projected annualized abnormal return (%)
    # Flags & Regime
    is_disciplined_allocator: bool  # True if RAI_t <= 0.02 and CDI_t <= 0.15 and DMP_t <= 0.15
    is_catering_capex_trap: bool  # True if CDI_t >= 1.10 or (RAI_t >= 0.15 and DMP_t >= 0.45)
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
        """Compute deterministic HMAC-SHA256 signature across core catering investment metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.capex_amount:.2f}:"
            f"{self.lagged_net_ppe:.2f}:{self.tobins_q:.4f}:{self.cash_flow_to_capital:.4f}:"
            f"{self.discretionary_mispricing_score:.4f}:{self.annual_turnover_pct:.2f}:"
            f"{self.investment_to_capital_ratio:.4f}:{self.fundamental_q_investment_rate:.4f}:"
            f"{self.residual_abnormal_investment:.4f}:{self.short_horizon_turnover_multiplier:.4f}:"
            f"{self.catering_distortion_index:.4f}:{self.expected_catering_alpha_pct:.2f}:"
            f"{int(self.is_disciplined_allocator)}:{int(self.is_catering_capex_trap)}:{self.regime}"
        )
        return hmac.new(
            CATERING_INVESTMENT_SIGNING_KEY,
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


def classify_catering_regime(
    investment_to_capital: float,
    residual_abnormal_inv: float,
    mispricing_score: float,
    catering_distortion_index: float,
) -> str:
    """Classify corporate capital allocation into Polk & Sapienza (2009) Catering Regimes."""
    if catering_distortion_index >= 1.10 or (
        residual_abnormal_inv >= 0.15 and mispricing_score >= 0.45
    ):
        return "OVERVALUED_CATERING_CAPEX_TRAP"
    if catering_distortion_index >= 0.45:
        return "SENTIMENT_CATERING_WATCH"
    if (
        residual_abnormal_inv <= 0.02
        and catering_distortion_index <= 0.15
        and mispricing_score <= 0.15
    ):
        return "DISCIPLINED_FUNDAMENTAL_ALLOCATOR"
    if investment_to_capital >= 0.20 and catering_distortion_index < 0.45:
        return "NEOCLASSICAL_Q_EXPANSION"
    return "NEOCLASSICAL_Q_EXPANSION"


def calculate_catering_investment(
    ticker: str,
    latest_price: float,
    capex_amount: float,
    lagged_net_ppe: float,
    tobins_q: float = 1.80,
    cash_flow_to_capital: float = 0.35,
    discretionary_mispricing_score: float = 0.05,
    annual_turnover_pct: float = 120.0,
) -> CateringInvestmentReceipt:
    """Calculate Polk & Sapienza (2009) Catering Theory of Investment and Valuation Mispricing metrics.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price ($).
        capex_amount: Annual capital expenditures in $M (CapEx_t >= 0).
        lagged_net_ppe: Prior-period net Property, Plant & Equipment K_{t-1} in $M (> 0).
        tobins_q: Tobin's Q ratio (> 0).
        cash_flow_to_capital: Operating cash flow scaled by lagged net PPE (CF_t / K_{t-1}).
        discretionary_mispricing_score: Discretionary accrual / valuation mispricing proxy in [-1.0, 1.0].
        annual_turnover_pct: Annualized share turnover in % (>= 0).

    Returns:
        CateringInvestmentReceipt with IK_t, RAI_t, CDI_t, regime classification, and expected alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(capex_amount, "capex_amount")
    assert_finite_number(lagged_net_ppe, "lagged_net_ppe")
    assert_finite_number(tobins_q, "tobins_q")
    assert_finite_number(cash_flow_to_capital, "cash_flow_to_capital")
    assert_finite_number(discretionary_mispricing_score, "discretionary_mispricing_score")
    assert_finite_number(annual_turnover_pct, "annual_turnover_pct")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if capex_amount < 0.0:
        raise ValueError("capex_amount must be non-negative.")
    if lagged_net_ppe <= 0.0:
        raise ValueError("lagged_net_ppe must be strictly positive.")
    if tobins_q <= 0.0:
        raise ValueError("tobins_q must be strictly positive.")
    if not (-1.0 <= discretionary_mispricing_score <= 1.0):
        raise ValueError("discretionary_mispricing_score must be in [-1.0, 1.0].")
    if annual_turnover_pct < 0.0:
        raise ValueError("annual_turnover_pct must be non-negative.")

    # 1. Capital Investment Rate IK_t = CapEx_t / K_{t-1}
    raw_ik = float(capex_amount) / float(lagged_net_ppe)
    ik_ratio = float(np.clip(raw_ik, 0.0, 2.50))

    # 2. Neoclassical Fundamental Q & Cash-Flow Benchmark Investment Rate
    q_excess = max(float(tobins_q) - 1.0, 0.0)
    cf_component = float(np.clip(cash_flow_to_capital, -0.50, 2.00))
    ik_hat = float(np.clip(0.08 + 0.045 * q_excess + 0.15 * cf_component, 0.04, 1.20))

    # 3. Residual Abnormal Investment RAI_t
    rai = float(np.clip(ik_ratio - ik_hat, -1.0, 1.5))

    # 4. Short-Horizon Shareholder Turnover Multiplier SHTM_t in [0.5, 2.0]
    shtm = float(np.clip(0.50 + (float(annual_turnover_pct) / 250.0), 0.50, 2.00))

    # 5. Catering Distortion Index CDI_t in [-1.5, 3.0]
    pos_mispricing = max(float(discretionary_mispricing_score), 0.0)
    raw_cdi = rai * (1.0 + 1.8 * pos_mispricing) * shtm * 2.2
    cdi = float(np.clip(raw_cdi, -1.5, 3.0))

    # 6. Regime Classification & Expected Catering Alpha
    regime = classify_catering_regime(
        ik_ratio, rai, float(discretionary_mispricing_score), cdi
    )
    is_disciplined = bool(regime == "DISCIPLINED_FUNDAMENTAL_ALLOCATOR")
    is_trap = bool(regime == "OVERVALUED_CATERING_CAPEX_TRAP")

    expected_alpha = float(
        np.clip(
            3.2 - 5.8 * cdi - 2.4 * max(float(discretionary_mispricing_score), 0.0),
            -11.5,
            5.5,
        )
    )

    if regime == "DISCIPLINED_FUNDAMENTAL_ALLOCATOR":
        summary = (
            f"Disciplined Fundamental Allocator (IK={ik_ratio:.2f}, RAI={rai:+.2f}, CDI={cdi:+.2f}). "
            f"Capital expenditure aligns with neoclassical Tobin's Q ({tobins_q:.2f}) without catering "
            f"to short-horizon sentiment. Expected catering alpha: {expected_alpha:+.1f}%/yr."
        )
    elif regime == "NEOCLASSICAL_Q_EXPANSION":
        summary = (
            f"Neoclassical Q Expansion (IK={ik_ratio:.2f}, Benchmark={ik_hat:.2f}, CDI={cdi:+.2f}). "
            f"Elevated physical investment supported by marginal Q ({tobins_q:.2f}) and internal cash flow "
            f"({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "SENTIMENT_CATERING_WATCH":
        summary = (
            f"Sentiment Catering Watch (IK={ik_ratio:.2f}, RAI={rai:+.2f}, CDI={cdi:+.2f}). "
            f"Abnormal CapEx expansion amid elevated share turnover ({annual_turnover_pct:.0f}%/yr); "
            f"monitor ROIC realization ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Overvalued Catering CapEx Trap (IK={ik_ratio:.2f}, RAI={rai:+.2f}, CDI={cdi:+.2f}). "
            f"Polk-Sapienza (2009) catering distortion: heavy abnormal CapEx driven by equity overvaluation "
            f"(DMP={discretionary_mispricing_score:+.2f}) and short-horizon turnover ({expected_alpha:+.1f}%/yr)."
        )

    return CateringInvestmentReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        capex_amount=round(float(capex_amount), 2),
        lagged_net_ppe=round(float(lagged_net_ppe), 2),
        tobins_q=round(float(tobins_q), 4),
        cash_flow_to_capital=round(float(cash_flow_to_capital), 4),
        discretionary_mispricing_score=round(float(discretionary_mispricing_score), 4),
        annual_turnover_pct=round(float(annual_turnover_pct), 2),
        investment_to_capital_ratio=round(ik_ratio, 4),
        fundamental_q_investment_rate=round(ik_hat, 4),
        residual_abnormal_investment=round(rai, 4),
        short_horizon_turnover_multiplier=round(shtm, 4),
        catering_distortion_index=round(cdi, 4),
        expected_catering_alpha_pct=round(expected_alpha, 2),
        is_disciplined_allocator=is_disciplined,
        is_catering_capex_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_catering_investment_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    capex_amount: Optional[float] = None,
    lagged_net_ppe: Optional[float] = None,
    tobins_q: Optional[float] = None,
    cash_flow_to_capital: Optional[float] = None,
    discretionary_mispricing_score: Optional[float] = None,
    annual_turnover_pct: Optional[float] = None,
) -> CateringInvestmentReceipt:
    """Evaluate Polk & Sapienza (2009) Catering Investment profile from SQLite observations and optional overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_mispricing = 0.04
    inferred_turnover = 115.0

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
                mom_20 = (arr[-1] / arr[0]) - 1.0
                inferred_mispricing = float(np.clip(mom_20 * 1.2, -0.60, 0.60))
    except Exception:
        latest_price = 100.0

    capex = float(capex_amount) if capex_amount is not None else 420.0
    ppe_lag = float(lagged_net_ppe) if lagged_net_ppe is not None else 2_400.0
    q_val = float(tobins_q) if tobins_q is not None else 2.10
    cf_ratio = float(cash_flow_to_capital) if cash_flow_to_capital is not None else 0.38
    dmp = (
        float(discretionary_mispricing_score)
        if discretionary_mispricing_score is not None
        else inferred_mispricing
    )
    turnover = (
        float(annual_turnover_pct)
        if annual_turnover_pct is not None
        else inferred_turnover
    )

    return calculate_catering_investment(
        ticker=clean_ticker,
        latest_price=latest_price,
        capex_amount=capex,
        lagged_net_ppe=ppe_lag,
        tobins_q=q_val,
        cash_flow_to_capital=cf_ratio,
        discretionary_mispricing_score=dmp,
        annual_turnover_pct=turnover,
    )

