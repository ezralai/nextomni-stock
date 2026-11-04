"""Malcolm Baker, Brendan Bradley & Jeffrey Wurgler (2011)
Institutional Benchmark Mandate, Tracking-Error Leverage Aversion & Low-Risk Anomaly Engine.

Implements the institutional benchmark mandate and tracking-error limits-to-arbitrage framework:
- Benchmarks as Limits to Arbitrage (Baker, Bradley & Wurgler, 2011):
    Fixed-weight benchmark mandates (e.g., S&P 500 or Russell 1000 relative performance contracts)
    combined with leverage restrictions distort institutional portfolio choice. An unlevered manager
    evaluated on Information Ratio IR_i = E[R_i - R_B] / TE_i faces single-stock tracking error:
        TE_i = sqrt((beta_i - 1.0)^2 * sigma_M^2 + sigma_{epsilon, i}^2)
    and active return:
        E[R_i - R_B] = alpha_i + (beta_i - 1.0) * E[R_M - r_f]
- Alpha Hurdle for Low-Beta Stocks:
    For any stock with beta_i < 1.0, overweighting it reduces expected benchmark-relative return
    unless its CAPM alpha exceeds the hurdle:
        alpha^{hurdle}_i = (1.0 - beta_i) * E[R_M - r_f]
    Consequently, benchmarked institutions systematically shun low-beta, low-IVOL compounders and
    crowd into high-beta stocks (beta_i > 1.0) as an implicit substitute for leverage.
- Regimes:
    - NEGLECTED_LOW_BETA_ALPHA_HAVEN: Low beta (beta_i <= 0.75) and low IVOL (sigma_e <= 0.22) shunned
      by benchmark-huggers, offering superior unconstrained Sharpe/alpha (+Alpha)
    - BENCHMARK_NEUTRAL_CORE: Near-unit beta (0.75 < beta_i < 1.25) with modest tracking error
    - TRACKING_ERROR_ORPHAN_WATCH: Low beta (beta_i <= 0.75) paired with elevated IVOL (sigma_e > 0.22)
    - CROWDED_HIGH_BETA_LEVERAGE_TRAP: High beta (beta_i >= 1.35 or IBCW_t >= 0.85) overpriced by
      leverage-seeking benchmarked institutions (-Alpha trap)

References:
    Baker, M., Bradley, B., & Wurgler, J. (2011). Benchmarks as limits to arbitrage: Understanding
    the low-volatility anomaly. Financial Analysts Journal, 67(1), 40-54.
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

BENCHMARK_MANDATE_SIGNING_KEY = b"nextomni_benchmark_mandate_secret_v1"


@dataclass
class BenchmarkMandateReceipt:
    """Cryptographic audit receipt for Baker, Bradley & Wurgler (2011) Benchmark Mandate Profile."""

    ticker: str
    latest_price: float
    # Beta, Volatility & Institutional Mandate Inputs
    capm_beta: float  # Stock CAPM beta_i in [-1.0, 4.0]
    market_volatility_ann: float  # Annualized market benchmark volatility sigma_M (> 0)
    idiosyncratic_volatility_ann: float  # Annualized idiosyncratic volatility sigma_{e,i} (> 0)
    expected_market_risk_premium_pct: float  # Expected market excess return E[R_M - r_f] in % (> 0)
    benchmarked_institutional_share: float  # Share of float held by benchmarked institutions in [0.0, 1.0]
    # Derived Baker-Bradley-Wurgler Metrics
    single_stock_tracking_error: float  # TE_i = sqrt((beta-1)^2 * sigma_M^2 + sigma_e^2)
    active_return_alpha_hurdle_pct: float  # Required CAPM alpha (1 - beta) * ERP to break even vs index (%)
    institutional_crowding_wedge: float  # IBCW_t in [-2.5, 2.5] (<0 neglected low-beta, >0 crowded high-beta)
    expected_unconstrained_alpha_pct: float  # Projected annualized unconstrained abnormal return (%)
    # Flags & Regime
    is_neglected_low_beta_haven: bool  # True if beta <= 0.75, sigma_e <= 0.22, IBCW <= -0.35
    is_crowded_high_beta_trap: bool  # True if beta >= 1.35 or IBCW >= 0.85
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
        """Compute deterministic HMAC-SHA256 signature across core benchmark mandate metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.capm_beta:.4f}:"
            f"{self.market_volatility_ann:.4f}:{self.idiosyncratic_volatility_ann:.4f}:"
            f"{self.expected_market_risk_premium_pct:.2f}:{self.benchmarked_institutional_share:.4f}:"
            f"{self.single_stock_tracking_error:.4f}:{self.active_return_alpha_hurdle_pct:.2f}:"
            f"{self.institutional_crowding_wedge:.4f}:{self.expected_unconstrained_alpha_pct:.2f}:"
            f"{int(self.is_neglected_low_beta_haven)}:{int(self.is_crowded_high_beta_trap)}:{self.regime}"
        )
        return hmac.new(
            BENCHMARK_MANDATE_SIGNING_KEY,
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


def classify_benchmark_mandate_regime(
    capm_beta: float,
    idio_vol: float,
    crowding_wedge: float,
) -> str:
    """Classify asset into Baker, Bradley & Wurgler (2011) Institutional Benchmark Mandate Regimes."""
    if capm_beta >= 1.35 or crowding_wedge >= 0.85:
        return "CROWDED_HIGH_BETA_LEVERAGE_TRAP"
    if capm_beta <= 0.75 and idio_vol <= 0.22 and crowding_wedge <= -0.35:
        return "NEGLECTED_LOW_BETA_ALPHA_HAVEN"
    if capm_beta <= 0.75 and idio_vol > 0.22:
        return "TRACKING_ERROR_ORPHAN_WATCH"
    return "BENCHMARK_NEUTRAL_CORE"


def calculate_benchmark_mandate(
    ticker: str,
    latest_price: float,
    capm_beta: float = 0.62,
    market_volatility_ann: float = 0.16,
    idiosyncratic_volatility_ann: float = 0.18,
    expected_market_risk_premium_pct: float = 6.0,
    benchmarked_institutional_share: float = 0.72,
) -> BenchmarkMandateReceipt:
    """Calculate Baker, Bradley & Wurgler (2011) Tracking Error, Alpha Hurdle, and Crowding Wedge.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price ($).
        capm_beta: Stock CAPM beta in [-1.0, 4.0].
        market_volatility_ann: Annualized market volatility sigma_M in (0.0, 2.0].
        idiosyncratic_volatility_ann: Annualized idiosyncratic volatility sigma_{e,i} in (0.0, 3.0].
        expected_market_risk_premium_pct: Expected market risk premium E[R_M - r_f] in % in (0.0, 25.0].
        benchmarked_institutional_share: Fraction of institutional ownership tied to fixed benchmarks in [0.0, 1.0].

    Returns:
        BenchmarkMandateReceipt with TE_i, alpha hurdle, crowding wedge, and unconstrained alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(capm_beta, "capm_beta")
    assert_finite_number(market_volatility_ann, "market_volatility_ann")
    assert_finite_number(idiosyncratic_volatility_ann, "idiosyncratic_volatility_ann")
    assert_finite_number(
        expected_market_risk_premium_pct, "expected_market_risk_premium_pct"
    )
    assert_finite_number(
        benchmarked_institutional_share, "benchmarked_institutional_share"
    )

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (-1.0 <= capm_beta <= 4.0):
        raise ValueError("capm_beta must be in [-1.0, 4.0].")
    if not (0.0 < market_volatility_ann <= 2.0):
        raise ValueError("market_volatility_ann must be in (0.0, 2.0].")
    if not (0.0 < idiosyncratic_volatility_ann <= 3.0):
        raise ValueError("idiosyncratic_volatility_ann must be in (0.0, 3.0].")
    if not (0.0 < expected_market_risk_premium_pct <= 25.0):
        raise ValueError("expected_market_risk_premium_pct must be in (0.0, 25.0].")
    if not (0.0 <= benchmarked_institutional_share <= 1.0):
        raise ValueError("benchmarked_institutional_share must be in [0.0, 1.0].")

    beta = float(capm_beta)
    sig_m = float(market_volatility_ann)
    sig_e = float(idiosyncratic_volatility_ann)
    erp = float(expected_market_risk_premium_pct)
    inst_share = float(benchmarked_institutional_share)

    # 1. Baker, Bradley & Wurgler (2011) Single-Stock Benchmark Tracking Error TE_i
    raw_te = math.sqrt(((beta - 1.0) ** 2) * (sig_m**2) + (sig_e**2))
    te_i = float(max(raw_te, 0.005))

    # 2. Active-Return Break-Even Alpha Hurdle (%) = (1.0 - beta_i) * E[R_M - r_f]
    alpha_hurdle = float((1.0 - beta) * erp)

    # 3. Institutional Benchmark Neglect / Crowding Wedge IBCW_t in [-2.5, 2.5]
    raw_wedge = (beta - 1.0) * (1.0 + 1.4 * inst_share) * (1.0 + (sig_e / 0.25))
    ibcw = float(np.clip(raw_wedge, -2.5, 2.5))

    # 4. Regime Classification & Expected Unconstrained Alpha
    regime = classify_benchmark_mandate_regime(beta, sig_e, ibcw)
    is_haven = bool(regime == "NEGLECTED_LOW_BETA_ALPHA_HAVEN")
    is_trap = bool(regime == "CROWDED_HIGH_BETA_LEVERAGE_TRAP")

    # Unconstrained investors harvest the positive alpha hurdle left on the table in low-beta,
    # low-IVOL stocks and avoid the negative alpha of crowded high-beta lottery/leverage substitutes.
    ivol_drag = max(sig_e - 0.22, 0.0) * 12.0
    expected_alpha = float(
        np.clip(-3.8 * ibcw - ivol_drag + 0.35 * alpha_hurdle, -11.5, 6.5)
    )

    if regime == "NEGLECTED_LOW_BETA_ALPHA_HAVEN":
        summary = (
            f"Neglected Low-Beta Alpha Haven (Beta={beta:.2f}, IVOL={sig_e:.1%}, Hurdle={alpha_hurdle:+.2f}%). "
            f"Shunned by fixed-benchmark institutions due to tracking-error aversion; offers superior "
            f"unconstrained alpha ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "BENCHMARK_NEUTRAL_CORE":
        summary = (
            f"Benchmark-Neutral Core (Beta={beta:.2f}, TE={te_i:.1%}, Wedge={ibcw:+.2f}). "
            f"Near-unit beta aligns with institutional index mandates ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "TRACKING_ERROR_ORPHAN_WATCH":
        summary = (
            f"Tracking-Error Orphan Watch (Beta={beta:.2f}, IVOL={sig_e:.1%}, TE={te_i:.1%}). "
            f"Low systematic beta but elevated idiosyncratic volatility penalizes institutional IR "
            f"({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Crowded High-Beta Leverage Trap (Beta={beta:.2f}, IVOL={sig_e:.1%}, Wedge={ibcw:+.2f}). "
            f"Baker-Bradley-Wurgler (2011) institutional crowding: benchmarked managers bid up high-beta "
            f"stock as implicit leverage ({expected_alpha:+.1f}%/yr)."
        )

    return BenchmarkMandateReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        capm_beta=round(beta, 4),
        market_volatility_ann=round(sig_m, 4),
        idiosyncratic_volatility_ann=round(sig_e, 4),
        expected_market_risk_premium_pct=round(erp, 2),
        benchmarked_institutional_share=round(inst_share, 4),
        single_stock_tracking_error=round(te_i, 4),
        active_return_alpha_hurdle_pct=round(alpha_hurdle, 2),
        institutional_crowding_wedge=round(ibcw, 4),
        expected_unconstrained_alpha_pct=round(expected_alpha, 2),
        is_neglected_low_beta_haven=is_haven,
        is_crowded_high_beta_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_benchmark_mandate_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    capm_beta: Optional[float] = None,
    market_volatility_ann: Optional[float] = None,
    idiosyncratic_volatility_ann: Optional[float] = None,
    expected_market_risk_premium_pct: Optional[float] = None,
    benchmarked_institutional_share: Optional[float] = None,
) -> BenchmarkMandateReceipt:
    """Evaluate Baker, Bradley & Wurgler (2011) Benchmark Mandate profile from SQLite and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_ivol = 0.18
    inferred_beta = 0.62

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
                arr = np.asarray(closes[-60:], dtype=float)
                rets = np.diff(arr) / arr[:-1]
                if len(rets) > 1:
                    ann_vol = float(np.std(rets, ddof=1) * math.sqrt(252.0))
                    inferred_ivol = float(np.clip(ann_vol * 0.75, 0.06, 1.50))
                    inferred_beta = float(np.clip(ann_vol / 0.22, 0.35, 2.40))
    except Exception:
        latest_price = 100.0

    beta_val = float(capm_beta) if capm_beta is not None else inferred_beta
    sig_m_val = (
        float(market_volatility_ann)
        if market_volatility_ann is not None
        else 0.16
    )
    sig_e_val = (
        float(idiosyncratic_volatility_ann)
        if idiosyncratic_volatility_ann is not None
        else inferred_ivol
    )
    erp_val = (
        float(expected_market_risk_premium_pct)
        if expected_market_risk_premium_pct is not None
        else 6.0
    )
    inst_val = (
        float(benchmarked_institutional_share)
        if benchmarked_institutional_share is not None
        else 0.72
    )

    return calculate_benchmark_mandate(
        ticker=clean_ticker,
        latest_price=latest_price,
        capm_beta=beta_val,
        market_volatility_ann=sig_m_val,
        idiosyncratic_volatility_ann=sig_e_val,
        expected_market_risk_premium_pct=erp_val,
        benchmarked_institutional_share=inst_val,
    )

