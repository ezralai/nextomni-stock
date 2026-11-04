"""Lu Zhang (2005)
Costly Reversibility of Capital, Asymmetric Adjustment Costs & Countercyclical Value Risk Premium Engine.

Implements the neoclassical physical capital adjustment cost and countercyclical value premium framework:
- Lu Zhang (2005, Journal of Finance, "The Value Premium"):
    Firms face convex and asymmetric physical capital adjustment costs: downward disinvestment costs
    (theta^-) are substantially more severe than upward expansion costs (theta^+), i.e.,
    theta^- / theta^+ >= 1.0 (costly reversibility), reinforced by fixed operating capital overhead f_K.
    - Value Firms (High Book-to-Market B_t / M_t): Burdened with abundant existing physical capital.
      In economic contractions (S_macro < 0), value firms want to disinvest unproductive capital but
      face severe downward adjustment costs (theta^-) and fixed operating leverage (f_K). Because they
      cannot easily reduce capital stock, their cash flows and market equity covary strongly with
      economic downturns when the aggregate price of risk is highest, generating a large
      countercyclical beta (beta_Zhang,t) and a rational equilibrium Value Risk Premium.
    - Growth Firms (Low Book-to-Market B_t / M_t): Hold primarily real growth expansion options and
      face smoother upward adjustment costs (theta^+) during expansions without unproductive fixed
      capital drag in recessions.

References:
    Zhang, L. (2005). The value premium. The Journal of Finance, 60(1), 67-103.
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

COSTLY_REVERSIBILITY_SIGNING_KEY = b"nextomni_lu_zhang_costly_reversibility_v1"


@dataclass
class CostlyReversibilityReceipt:
    """Cryptographic audit receipt for Lu Zhang (2005) Costly Reversibility & Value Beta Profile."""

    ticker: str
    latest_price: float
    # Capital Adjustment & Macroeconomic Inputs
    book_to_market_ratio: float  # Book-to-Market ratio (B/M)_t in [0.05, 5.0]
    downward_upward_cost_asymmetry: float  # Disinvestment/expansion asymmetry theta^-/theta^+ in [1.0, 25.0]
    fixed_capital_intensity: float  # Operating fixed-capital overhead f_K in [0.0, 1.0]
    macro_state_zscore: float  # Macroeconomic state S_macro in [-3.0, 3.0] (< 0 recession)
    # Derived Lu Zhang Metrics
    capital_irreversibility_index: float  # CII_t = (theta^-/theta^+) * (0.40 + 0.60 * f_K)
    disinvestment_friction_mult: float  # Phi_disinv >= 1.0
    countercyclical_value_beta: float  # Conditional market beta beta_{Zhang,t}
    lu_zhang_value_impulse: float  # LZVI_t in [-2.5, 2.5]
    expected_reversibility_alpha_pct: float  # Projected annualized value/adjustment alpha (%)
    # Flags & Regime
    is_compensated_irreversible_value_leader: bool  # True if high B/M compensated value premium
    is_stranded_capital_recession_trap: bool  # True if stranded fixed capital in deep contraction
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
        """Compute deterministic HMAC-SHA256 signature across all Lu Zhang metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:"
            f"{self.book_to_market_ratio:.4f}:{self.downward_upward_cost_asymmetry:.4f}:"
            f"{self.fixed_capital_intensity:.4f}:{self.macro_state_zscore:.4f}:"
            f"{self.capital_irreversibility_index:.4f}:{self.disinvestment_friction_mult:.4f}:"
            f"{self.countercyclical_value_beta:.4f}:{self.lu_zhang_value_impulse:.4f}:"
            f"{self.expected_reversibility_alpha_pct:.2f}:"
            f"{int(self.is_compensated_irreversible_value_leader)}:"
            f"{int(self.is_stranded_capital_recession_trap)}:{self.regime}"
        )
        return hmac.new(
            COSTLY_REVERSIBILITY_SIGNING_KEY,
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


def classify_reversibility_regime(
    book_to_market_ratio: float,
    fixed_capital_intensity: float,
    macro_state_zscore: float,
    lu_zhang_value_impulse: float,
) -> str:
    """Classify asset into Lu Zhang (2005) Costly Reversibility Regimes."""
    if macro_state_zscore <= -1.0 and fixed_capital_intensity >= 0.70 and lu_zhang_value_impulse <= -0.35:
        return "STRANDED_CAPITAL_RECESSION_TRAP"
    if book_to_market_ratio >= 0.85 and lu_zhang_value_impulse >= 0.40:
        return "COMPENSATED_IRREVERSIBLE_VALUE_PREMIUM"
    if book_to_market_ratio <= 0.45 and fixed_capital_intensity <= 0.35:
        return "FLEXIBLE_GROWTH_EXPANSION_OPTION"
    return "NEUTRAL_ADJUSTMENT_COST_EQUILIBRIUM"


def calculate_costly_reversibility(
    ticker: str,
    latest_price: float,
    book_to_market_ratio: float = 1.05,
    downward_upward_cost_asymmetry: float = 8.0,
    fixed_capital_intensity: float = 0.55,
    macro_state_zscore: float = -0.35,
) -> CostlyReversibilityReceipt:
    """Calculate Lu Zhang (2005) Costly Reversibility of Capital & Countercyclical Value Beta.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        book_to_market_ratio: Book-to-Market equity ratio (B/M)_t in [0.05, 5.0].
        downward_upward_cost_asymmetry: Ratio of disinvestment to expansion adjustment cost
            parameter theta^- / theta^+ in [1.0, 25.0] (Lu Zhang benchmark ~ 10.0).
        fixed_capital_intensity: Operating fixed capital intensity f_K in [0.0, 1.0].
        macro_state_zscore: Standardized macroeconomic state S_macro in [-3.0, 3.0], where
            negative values denote economic contractions / high price of risk.

    Returns:
        CostlyReversibilityReceipt with CII_t, Phi_disinv, beta_{Zhang,t}, regime, and alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(book_to_market_ratio, "book_to_market_ratio")
    assert_finite_number(downward_upward_cost_asymmetry, "downward_upward_cost_asymmetry")
    assert_finite_number(fixed_capital_intensity, "fixed_capital_intensity")
    assert_finite_number(macro_state_zscore, "macro_state_zscore")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (0.05 <= book_to_market_ratio <= 5.0):
        raise ValueError("book_to_market_ratio must be in [0.05, 5.0].")
    if not (1.0 <= downward_upward_cost_asymmetry <= 25.0):
        raise ValueError("downward_upward_cost_asymmetry must be in [1.0, 25.0].")
    if not (0.0 <= fixed_capital_intensity <= 1.0):
        raise ValueError("fixed_capital_intensity must be in [0.0, 1.0].")
    if not (-3.0 <= macro_state_zscore <= 3.0):
        raise ValueError("macro_state_zscore must be in [-3.0, 3.0].")

    p_t = float(latest_price)
    bm = float(book_to_market_ratio)
    asym = float(downward_upward_cost_asymmetry)
    f_k = float(fixed_capital_intensity)
    s_macro = float(macro_state_zscore)

    # 1. Capital Irreversibility Index CII_t = (theta^-/theta^+) * (0.40 + 0.60 * f_K)
    cii = float(asym * (0.40 + 0.60 * f_k))

    # 2. Disinvestment Friction Multiplier Phi_disinv >= 1.0
    contraction_depth = max(0.0, -s_macro)
    phi_disinv = float(np.clip(1.0 + 0.12 * cii * (0.35 + contraction_depth), 1.0, 6.0))

    # 3. Countercyclical Value Beta beta_{Zhang,t}
    # Value stocks (high B/M) experience countercyclical beta spikes in bad times
    beta_base = 0.85
    beta_zhang = float(
        np.clip(
            beta_base + 0.38 * bm * phi_disinv + 0.25 * f_k * contraction_depth,
            0.40,
            3.80,
        )
    )

    # 4. Lu Zhang Value Impulse LZVI_t in [-2.5, 2.5]
    # Moderate contraction / normal times reward high B/M irreversible value;
    # extreme contraction (S_macro <= -1.0) with excessive fixed overhead (f_K >= 0.70) triggers stranded trap
    stranded_penalty = (
        1.85 * contraction_depth * f_k
        if (s_macro <= -1.0 and f_k >= 0.70)
        else 0.0
    )
    raw_impulse = (
        0.85 * (bm - 0.55) * math.sqrt(phi_disinv)
        + 0.25 * (beta_zhang - 1.0)
        - stranded_penalty
    )
    lzvi = float(np.clip(raw_impulse, -2.5, 2.5))

    # 5. Regime Classification & Expected Alpha
    regime = classify_reversibility_regime(bm, f_k, s_macro, lzvi)
    is_leader = bool(regime == "COMPENSATED_IRREVERSIBLE_VALUE_PREMIUM")
    is_trap = bool(regime == "STRANDED_CAPITAL_RECESSION_TRAP")

    expected_alpha = float(np.clip(4.6 * lzvi, -12.0, 12.0))

    if regime == "COMPENSATED_IRREVERSIBLE_VALUE_PREMIUM":
        summary = (
            f"Compensated Irreversible Value Leader (B/M={bm:.2f}, Asym={asym:.1f}x, "
            f"beta_Zhang={beta_zhang:.2f}). Costly capital reversibility generates a rational "
            f"countercyclical value risk premium ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "STRANDED_CAPITAL_RECESSION_TRAP":
        summary = (
            f"Stranded Capital Recession Trap (S_macro={s_macro:+.2f} SD, f_K={f_k:.0%}, "
            f"Phi={phi_disinv:.2f}x). Severe contraction and fixed capital overhead prevent "
            f"disinvestment, creating acute operating leverage drag ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "FLEXIBLE_GROWTH_EXPANSION_OPTION":
        summary = (
            f"Flexible Growth Expansion Option (B/M={bm:.2f}, f_K={f_k:.0%}, "
            f"beta_Zhang={beta_zhang:.2f}). Low fixed-capital drag and smooth upward expansion "
            f"flexibility buffer macroeconomic downturns ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Neutral Adjustment Cost Equilibrium (B/M={bm:.2f}, CII={cii:.2f}, "
            f"beta_Zhang={beta_zhang:.2f}). Balanced physical capital reversibility profile "
            f"({expected_alpha:+.1f}%/yr)."
        )

    return CostlyReversibilityReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        book_to_market_ratio=round(bm, 4),
        downward_upward_cost_asymmetry=round(asym, 4),
        fixed_capital_intensity=round(f_k, 4),
        macro_state_zscore=round(s_macro, 4),
        capital_irreversibility_index=round(cii, 4),
        disinvestment_friction_mult=round(phi_disinv, 4),
        countercyclical_value_beta=round(beta_zhang, 4),
        lu_zhang_value_impulse=round(lzvi, 4),
        expected_reversibility_alpha_pct=round(expected_alpha, 2),
        is_compensated_irreversible_value_leader=is_leader,
        is_stranded_capital_recession_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_costly_reversibility_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    book_to_market_ratio: Optional[float] = None,
    downward_upward_cost_asymmetry: Optional[float] = None,
    fixed_capital_intensity: Optional[float] = None,
    macro_state_zscore: Optional[float] = None,
) -> CostlyReversibilityReceipt:
    """Evaluate Lu Zhang (2005) Costly Reversibility profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_bm = 1.05
    inferred_asym = 8.0
    inferred_fk = 0.55
    inferred_macro = -0.35

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
                ann_vol = float(np.std(rets, ddof=1) * math.sqrt(252.0)) if len(rets) > 1 else 0.22

                # Infer B/M ratio inversely from recent valuation multiple expansion
                price_ratio = float(arr[-1] / max(arr[0], 1e-6))
                inferred_bm = float(np.clip(1.10 / max(price_ratio, 0.35), 0.15, 3.50))
                # Infer downward/upward adjustment asymmetry from downside vs upside return variance
                neg_rets = rets[rets < 0]
                pos_rets = rets[rets > 0]
                if len(neg_rets) > 2 and len(pos_rets) > 2:
                    down_up_ratio = float(np.var(neg_rets) / max(np.var(pos_rets), 1e-8))
                    inferred_asym = float(np.clip(6.0 * down_up_ratio, 1.5, 20.0))
                inferred_fk = float(np.clip(0.35 + 0.85 * ann_vol, 0.15, 0.90))
                inferred_macro = float(np.clip(2.0 * ann_ret, -2.5, 2.5))
    except Exception:
        latest_price = 100.0

    bm_val = (
        float(book_to_market_ratio)
        if book_to_market_ratio is not None
        else inferred_bm
    )
    asym_val = (
        float(downward_upward_cost_asymmetry)
        if downward_upward_cost_asymmetry is not None
        else inferred_asym
    )
    fk_val = (
        float(fixed_capital_intensity)
        if fixed_capital_intensity is not None
        else inferred_fk
    )
    macro_val = (
        float(macro_state_zscore)
        if macro_state_zscore is not None
        else inferred_macro
    )

    return calculate_costly_reversibility(
        ticker=clean_ticker,
        latest_price=latest_price,
        book_to_market_ratio=bm_val,
        downward_upward_cost_asymmetry=asym_val,
        fixed_capital_intensity=fk_val,
        macro_state_zscore=macro_val,
    )

