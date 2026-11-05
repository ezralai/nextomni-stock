"""John Y. Campbell, Jens Hilscher & Jan Szilagyi (2008)
Financial Distress Puzzle, Dynamic Failure Probability & Negative Distress Premium Engine.

Implements the Campbell-Hilscher-Szilagyi (CHS) Failure Prediction & Distress Anomaly framework:
- Campbell, Hilscher & Szilagyi (2008, Journal of Finance, "In Search of Distress Risk"):
    Traditional asset pricing (CAPM, Merton 1974 structural credit models) posits that financially distressed
    firms must offer elevated expected returns to compensate risk-averse investors for bearing default hazard.
    Campbell, Hilscher, and Szilagyi uncover a celebrated empirical contradiction—the "Financial Distress Puzzle":
    firms with high multi-period failure probabilities deliver anomalously low subsequent stock returns,
    underperforming safe firms across size, book-to-market, and momentum controls.
    - Dynamic Multi-Period Logit Model:
      Predicts 12-month failure hazard P(Failure) from accounting profitability, leverage, market returns,
      and volatility:
          Score = -9.16 - 20.26 * NIMTA + 1.42 * TLMTA - 7.13 * EXRET + 1.41 * SIGMA
                  - 0.045 * RSIZE - 2.13 * CASHMTA + 0.075 * MB - 0.058 * PRICE
          P(Failure) = 1 / (1 + exp(-Score))
      where:
          - NIMTA: Net income divided by market total assets (MTA = Total Liabilities + Market Cap)
          - TLMTA: Total liabilities divided by market total assets
          - CASHMTA: Cash and short-term investments divided by market total assets
          - EXRET: Log stock return minus S&P 500 benchmark return
          - SIGMA: Daily idiosyncratic standard deviation annualized
          - RSIZE: Log ratio of firm equity to total S&P 500 market equity
          - MB: Market-to-book equity ratio
          - PRICE: Price per share capped at $15

References:
    Campbell, J. Y., Hilscher, J., & Szilagyi, J. (2008). In search of distress risk.
    The Journal of Finance, 63(6), 2899-2939.
    Campbell, J. Y., Giglio, S., Polk, C., & Turley, R. (2018). An intertemporal CAPM with
    stochastic volatility. Journal of Financial Economics, 128(2), 207-233.
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

CAMPBELL_DISTRESS_SIGNING_KEY = b"nextomni_campbell_distress_puzzle_v1"


@dataclass
class CampbellDistressReceipt:
    """Cryptographic audit receipt for Campbell-Hilscher-Szilagyi (2008) Financial Distress Profile."""

    ticker: str
    latest_price: float
    # Accounting & Market Ratio Inputs
    net_income_to_market_assets: float  # NIMTA in [-1.0, 0.50]
    total_liabilities_to_market_assets: float  # TLMTA in [0.0, 0.98]
    cash_to_market_assets: float  # CASHMTA in [0.0, 0.80]
    trailing_excess_return: float  # EXRET in [-1.50, 1.50]
    idiosyncratic_volatility_sigma: float  # SIGMA in [0.05, 1.50]
    relative_firm_size: float  # RSIZE in [-12.0, 5.0]
    market_to_book_ratio: float  # MB in [0.10, 25.0]
    # Derived CHS Inferences
    chs_logit_score: float  # Dynamic CHS distress logit index
    failure_probability_12m_pct: float  # P(Failure) over 12 months (%) in [0.001, 60.0]
    distress_anomaly_alpha_drag_pct: float  # Projected negative alpha drag from distress anomaly (%)
    chs_fragility_impulse: float  # I_chs in [-2.5, 2.5]
    expected_distress_alpha_pct: float  # Total annualized expected alpha net of failure drag (%)
    # Flags & Regime
    is_distress_anomaly_trap: bool  # True if P(Failure) >= 2.50% (overpriced low-return lottery)
    is_solvent_financial_fortress: bool  # True if P(Failure) <= 0.10% (resilient blue chip)
    is_elevated_distress_watch: bool  # True if P(Failure) >= 0.80%
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
        """Compute deterministic HMAC-SHA256 signature across all CHS distress metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:"
            f"{self.net_income_to_market_assets:.4f}:{self.total_liabilities_to_market_assets:.4f}:"
            f"{self.cash_to_market_assets:.4f}:{self.trailing_excess_return:.4f}:{self.idiosyncratic_volatility_sigma:.4f}:"
            f"{self.relative_firm_size:.4f}:{self.market_to_book_ratio:.4f}:{self.chs_logit_score:.4f}:"
            f"{self.failure_probability_12m_pct:.4f}:{self.distress_anomaly_alpha_drag_pct:.4f}:"
            f"{self.chs_fragility_impulse:.4f}:{self.expected_distress_alpha_pct:.2f}:"
            f"{int(self.is_distress_anomaly_trap)}:{int(self.is_solvent_financial_fortress)}:{int(self.is_elevated_distress_watch)}:{self.regime}"
        )
        return hmac.new(
            CAMPBELL_DISTRESS_SIGNING_KEY,
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


def classify_distress_regime(
    failure_probability_pct: float,
    tlmta: float,
    cashmta: float,
) -> str:
    """Classify asset into Campbell-Hilscher-Szilagyi (2008) Distress Regimes."""
    if failure_probability_pct >= 2.50:
        return "DISTRESS_ANOMALY_OVERPRICED_TRAP"
    if failure_probability_pct <= 0.10 and cashmta >= 0.15:
        return "SOLVENT_FINANCIAL_FORTRESS"
    if failure_probability_pct >= 0.80 or tlmta >= 0.70:
        return "ELEVATED_FINANCIAL_DISTRESS_WATCH"
    return "NORMAL_SOLVENCY_EQUILIBRIUM"


def calculate_campbell_distress(
    ticker: str,
    latest_price: float,
    net_income_to_market_assets: float = 0.03,
    total_liabilities_to_market_assets: float = 0.40,
    cash_to_market_assets: float = 0.15,
    trailing_excess_return: float = 0.05,
    idiosyncratic_volatility_sigma: float = 0.22,
    relative_firm_size: float = -6.50,
    market_to_book_ratio: float = 2.80,
) -> CampbellDistressReceipt:
    """Calculate Campbell-Hilscher-Szilagyi (2008) Dynamic Failure Hazard & Distress Puzzle Metrics.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        net_income_to_market_assets: NIMTA in [-1.0, 0.50].
        total_liabilities_to_market_assets: TLMTA in [0.0, 0.98].
        cash_to_market_assets: CASHMTA in [0.0, 0.80].
        trailing_excess_return: EXRET in [-1.50, 1.50].
        idiosyncratic_volatility_sigma: SIGMA in [0.05, 1.50].
        relative_firm_size: RSIZE in [-12.0, 5.0].
        market_to_book_ratio: MB in [0.10, 25.0].

    Returns:
        CampbellDistressReceipt with CHS logit score, 12M failure probability, alpha drag, and regime.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(net_income_to_market_assets, "net_income_to_market_assets")
    assert_finite_number(total_liabilities_to_market_assets, "total_liabilities_to_market_assets")
    assert_finite_number(cash_to_market_assets, "cash_to_market_assets")
    assert_finite_number(trailing_excess_return, "trailing_excess_return")
    assert_finite_number(idiosyncratic_volatility_sigma, "idiosyncratic_volatility_sigma")
    assert_finite_number(relative_firm_size, "relative_firm_size")
    assert_finite_number(market_to_book_ratio, "market_to_book_ratio")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (-1.0 <= net_income_to_market_assets <= 0.50):
        raise ValueError("net_income_to_market_assets must be in [-1.0, 0.50].")
    if not (0.0 <= total_liabilities_to_market_assets <= 0.98):
        raise ValueError("total_liabilities_to_market_assets must be in [0.0, 0.98].")
    if not (0.0 <= cash_to_market_assets <= 0.80):
        raise ValueError("cash_to_market_assets must be in [0.0, 0.80].")
    if not (-1.50 <= trailing_excess_return <= 1.50):
        raise ValueError("trailing_excess_return must be in [-1.50, 1.50].")
    if not (0.05 <= idiosyncratic_volatility_sigma <= 1.50):
        raise ValueError("idiosyncratic_volatility_sigma must be in [0.05, 1.50].")
    if not (-12.0 <= relative_firm_size <= 5.0):
        raise ValueError("relative_firm_size must be in [-12.0, 5.0].")
    if not (0.10 <= market_to_book_ratio <= 25.0):
        raise ValueError("market_to_book_ratio must be in [0.10, 25.0].")

    p_t = float(latest_price)
    nimta = float(net_income_to_market_assets)
    tlmta = float(total_liabilities_to_market_assets)
    cashmta = float(cash_to_market_assets)
    exret = float(trailing_excess_return)
    sigma = float(idiosyncratic_volatility_sigma)
    rsize = float(relative_firm_size)
    mb = float(market_to_book_ratio)

    # 1. Campbell-Hilscher-Szilagyi (2008, Table IV, Model 3) Dynamic Logit Score
    price_capped = min(p_t, 15.0)
    raw_score = (
        -9.16
        - 20.26 * nimta
        + 1.42 * tlmta
        - 7.13 * exret
        + 1.41 * sigma
        - 0.045 * rsize
        - 2.13 * cashmta
        + 0.075 * mb
        - 0.058 * price_capped
    )
    chs_score = float(np.clip(raw_score, -15.0, 5.0))

    # 2. 12-Month Failure Probability (%)
    # P(Failure) = 1 / (1 + exp(-Score)) * 100
    prob_failure = float(100.0 / (1.0 + math.exp(-chs_score)))
    prob_failure = float(np.clip(prob_failure, 0.001, 60.0))

    # 3. Distress Puzzle Alpha Drag
    # Firms with elevated failure probability suffer from negative alpha drag
    # Solvent firms (low hazard, strong cash buffer) earn positive quality alpha
    log_hazard_ratio = math.log(max(prob_failure, 0.001) / 0.10)
    alpha_drag = float(
        np.clip(
            -3.5 * max(0.0, log_hazard_ratio) + 1.2 * max(0.0, cashmta - tlmta),
            -12.0,
            8.0,
        )
    )

    # 4. CHS Fragility Impulse
    raw_impulse = -0.45 * (prob_failure / 2.0) + 0.35 * max(0.0, cashmta - 0.10) - 0.30 * max(0.0, tlmta - 0.50)
    impulse = float(np.clip(raw_impulse, -2.5, 2.5))
    expected_alpha = float(np.clip(alpha_drag + 1.8 * impulse, -14.0, 10.0))

    regime = classify_distress_regime(prob_failure, tlmta, cashmta)
    is_trap = bool(regime == "DISTRESS_ANOMALY_OVERPRICED_TRAP")
    is_fortress = bool(regime == "SOLVENT_FINANCIAL_FORTRESS")
    is_watch = bool(regime == "ELEVATED_FINANCIAL_DISTRESS_WATCH")

    if regime == "DISTRESS_ANOMALY_OVERPRICED_TRAP":
        summary = (
            f"Distress Anomaly Overpriced Trap (Failure_Prob={prob_failure:.2f}%, CHS_Score={chs_score:.2f}, Drag={alpha_drag:+.1f}%). "
            f"High dynamic failure probability induces severe retail lottery speculation and underperforming forward alpha ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "SOLVENT_FINANCIAL_FORTRESS":
        summary = (
            f"Solvent Financial Fortress (Failure_Prob={prob_failure:.3f}%, CashMTA={cashmta:.1%}, TLMTA={tlmta:.1%}). "
            f"Negligible failure probability and substantial cash reserves insulate equity from distress drag ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "ELEVATED_FINANCIAL_DISTRESS_WATCH":
        summary = (
            f"Elevated Financial Distress Watch (Failure_Prob={prob_failure:.2f}%, TLMTA={tlmta:.1%}, CHS_Score={chs_score:.2f}). "
            f"Moderate balance sheet leverage and declining profit buffers warrant strict solvency monitoring ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Normal Solvency Equilibrium (Failure_Prob={prob_failure:.2f}%, NIMTA={nimta:.1%}, CashMTA={cashmta:.1%}). "
            f"Solvency and cash flow metrics are within healthy prudential boundaries ({expected_alpha:+.1f}%/yr)."
        )

    return CampbellDistressReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        net_income_to_market_assets=round(nimta, 4),
        total_liabilities_to_market_assets=round(tlmta, 4),
        cash_to_market_assets=round(cashmta, 4),
        trailing_excess_return=round(exret, 4),
        idiosyncratic_volatility_sigma=round(sigma, 4),
        relative_firm_size=round(rsize, 4),
        market_to_book_ratio=round(mb, 2),
        chs_logit_score=round(chs_score, 4),
        failure_probability_12m_pct=round(prob_failure, 4),
        distress_anomaly_alpha_drag_pct=round(alpha_drag, 2),
        chs_fragility_impulse=round(impulse, 4),
        expected_distress_alpha_pct=round(expected_alpha, 2),
        is_distress_anomaly_trap=is_trap,
        is_solvent_financial_fortress=is_fortress,
        is_elevated_distress_watch=is_watch,
        regime=regime,
        summary=summary,
    )


def evaluate_campbell_distress_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    net_income_to_market_assets: Optional[float] = None,
    total_liabilities_to_market_assets: Optional[float] = None,
    cash_to_market_assets: Optional[float] = None,
    trailing_excess_return: Optional[float] = None,
    idiosyncratic_volatility_sigma: Optional[float] = None,
    relative_firm_size: Optional[float] = None,
    market_to_book_ratio: Optional[float] = None,
) -> CampbellDistressReceipt:
    """Evaluate Campbell-Hilscher-Szilagyi distress profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_nimta = 0.03
    inferred_tlmta = 0.35
    inferred_cashmta = 0.15
    inferred_exret = 0.05
    inferred_sigma = 0.22
    inferred_rsize = -6.50
    inferred_mb = 2.80

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

                # Annualized return and idiosyncratic volatility
                inferred_sigma = float(np.clip(np.std(rets, ddof=1) * math.sqrt(252.0), 0.05, 1.50)) if len(rets) > 1 else 0.22
                ann_ret = float(np.mean(rets) * 252.0)
                inferred_exret = float(np.clip(ann_ret - 0.08, -1.50, 1.50))

                # Drawdown proxy for distress leverage elevation
                peak = float(np.max(arr))
                dd = float((arr[-1] - peak) / max(peak, 1e-4))
                inferred_tlmta = float(np.clip(0.35 - 0.50 * dd, 0.10, 0.85))
                inferred_nimta = float(np.clip(0.04 + 0.10 * dd, -0.40, 0.25))
                inferred_cashmta = float(np.clip(0.18 + 0.15 * dd, 0.02, 0.60))
    except Exception:
        latest_price = 100.0

    nimta_val = float(net_income_to_market_assets) if net_income_to_market_assets is not None else inferred_nimta
    tlmta_val = float(total_liabilities_to_market_assets) if total_liabilities_to_market_assets is not None else inferred_tlmta
    cash_val = float(cash_to_market_assets) if cash_to_market_assets is not None else inferred_cashmta
    exret_val = float(trailing_excess_return) if trailing_excess_return is not None else inferred_exret
    sigma_val = float(idiosyncratic_volatility_sigma) if idiosyncratic_volatility_sigma is not None else inferred_sigma
    rsize_val = float(relative_firm_size) if relative_firm_size is not None else inferred_rsize
    mb_val = float(market_to_book_ratio) if market_to_book_ratio is not None else inferred_mb

    return calculate_campbell_distress(
        ticker=clean_ticker,
        latest_price=latest_price,
        net_income_to_market_assets=nimta_val,
        total_liabilities_to_market_assets=tlmta_val,
        cash_to_market_assets=cash_val,
        trailing_excess_return=exret_val,
        idiosyncratic_volatility_sigma=sigma_val,
        relative_firm_size=rsize_val,
        market_to_book_ratio=mb_val,
    )

