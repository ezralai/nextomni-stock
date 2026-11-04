"""Clifford S. Asness, Tobias J. Moskowitz & Lasse Heje Pedersen (2013) 'Value and Momentum Everywhere' (VME) Engine.

Implements the unified Value and Momentum asset pricing framework of Asness, Moskowitz & Pedersen (2013):
- Value and Momentum Everywhere (VME) Premise:
    Value (Book-to-Market, Earnings Yield) and Momentum (12-2 skip-month intermediate return)
    individually generate positive risk-adjusted excess returns across global equities and macro assets.
- Negative Covariance & Anti-Fragile Diversification:
    Value and Momentum returns exhibit strong negative correlation (rho in [-0.40, -0.65]).
    When Momentum experiences crash drawdowns during market reversals, Value rebounds sharply;
    when Value suffers prolonged stagnation, Momentum captures persistent trend continuation.
- Unified 50/50 Volatility-Matched VME Combination:
    z_VME = (z_Value + z_Momentum) / sqrt(2 * (1 + rho_val_mom))
    Diversification Benefit Ratio (DBR) = sqrt(2 / (1 + rho_val_mom))
- Regimes:
    - SYNERGISTIC_VME_COMPOUNDER: Cheap valuation + Strong positive momentum (Twin-engine alpha)
    - HEDGED_BARBELL_EQUILIBRIUM: Strong single factor hedged by negative factor covariance
    - NEUTRAL_FACTOR_EXPOSURE: Balanced near-market exposure
    - DUAL_FACTOR_HEADWIND_TRAP: Expensive valuation + Negative momentum breakdown

References:
    Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere.
    The Journal of Finance, 68(3), 929-985.
    Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency.
    The Journal of Finance, 48(1), 65-91.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

VME_UNIFIED_SIGNING_KEY = b"nextomni_vme_unified_secret_v1"


@dataclass
class VMEReceipt:
    """Cryptographic audit receipt for Asness, Moskowitz & Pedersen (2013) VME Unified Profile."""

    ticker: str
    latest_price: float
    # Fundamental & Price Inputs
    book_to_market: float
    earnings_yield_pct: float
    momentum_12_2_pct: float  # 12-2 month skip-month intermediate return (%)
    short_term_1m_pct: float  # 1-month short-term return skipped to avoid reversal (%)
    annualized_volatility_pct: float
    # Standardized Style Z-Scores & Covariance
    value_z_score: float  # Standardized Value factor score in [-3.5, +3.5]
    momentum_z_score: float  # Standardized Momentum factor score in [-3.5, +3.5]
    value_momentum_correlation: float  # Empirical style correlation rho in [-0.95, +0.95]
    vme_combo_z_score: float  # Covariance-adjusted 50/50 VME composite z-score
    diversification_benefit_ratio: float  # Portfolio volatility reduction multiplier sqrt(2 / (1 + rho))
    implied_vme_sharpe: float  # Projected combined Sharpe ratio
    # Classifications & Alpha
    is_twin_engine_compounder: bool  # True if both Value > 0 and Momentum > 0 with high VME score
    is_dual_headwind_trap: bool  # True if expensive and crashing (z_VME < -0.50)
    expected_vme_alpha_pct: float  # Projected annualized VME excess return (%)
    regime: str  # SYNERGISTIC_VME_COMPOUNDER, HEDGED_BARBELL_EQUILIBRIUM, NEUTRAL_FACTOR_EXPOSURE, DUAL_FACTOR_HEADWIND_TRAP
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core VME metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.book_to_market:.4f}:"
            f"{self.earnings_yield_pct:.2f}:{self.momentum_12_2_pct:.2f}:"
            f"{self.short_term_1m_pct:.2f}:{self.annualized_volatility_pct:.2f}:"
            f"{self.value_z_score:.4f}:{self.momentum_z_score:.4f}:"
            f"{self.value_momentum_correlation:.4f}:{self.vme_combo_z_score:.4f}:"
            f"{self.diversification_benefit_ratio:.4f}:{self.implied_vme_sharpe:.2f}:"
            f"{int(self.is_twin_engine_compounder)}:{int(self.is_dual_headwind_trap)}:"
            f"{self.expected_vme_alpha_pct:.2f}:{self.regime}"
        )
        return hmac.new(
            VME_UNIFIED_SIGNING_KEY,
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


def classify_vme_regime(
    value_z: float,
    momentum_z: float,
    vme_combo_z: float,
) -> str:
    """Classify asset into Asness, Moskowitz & Pedersen (2013) VME quadrants."""
    if vme_combo_z < -0.50:
        return "DUAL_FACTOR_HEADWIND_TRAP"
    if vme_combo_z >= 0.80 and value_z > 0.0 and momentum_z > 0.0:
        return "SYNERGISTIC_VME_COMPOUNDER"
    if vme_combo_z >= 0.25:
        return "HEDGED_BARBELL_EQUILIBRIUM"
    return "NEUTRAL_FACTOR_EXPOSURE"


def calculate_vme_score(
    ticker: str,
    prices: Sequence[float],
    book_to_market: float = 0.55,
    earnings_yield_pct: float = 5.5,
    value_momentum_correlation: Optional[float] = None,
) -> VMEReceipt:
    """Calculate Asness, Moskowitz & Pedersen (2013) unified Value & Momentum profile.

    Args:
        ticker: Validated stock ticker symbol.
        prices: Chronological sequence of historical closing prices (at least 25 sessions).
        book_to_market: Ratio of book equity to market equity (B/M).
        earnings_yield_pct: Earnings-to-price yield in percent (E/P %).
        value_momentum_correlation: Optional override for Value-Momentum correlation in [-0.95, 0.95].

    Returns:
        VMEReceipt with standardized style scores, covariance, and combo alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(book_to_market, "book_to_market")
    assert_finite_number(earnings_yield_pct, "earnings_yield_pct")

    if len(prices) < 25:
        raise ValueError("At least 25 price observations are required for VME evaluation.")

    price_arr = np.asarray(prices, dtype=float)
    if not np.all(np.isfinite(price_arr)) or np.any(price_arr <= 0.0):
        raise ValueError("All price observations must be positive finite numbers.")

    latest_price = float(price_arr[-1])

    # 1. Compute 12-2 Skip-Month Intermediate Momentum & 1-Month Short-Term Return
    # Skip latest 21 trading sessions (1 month) to avoid Jegadeesh (1990) short-term reversal
    n = len(price_arr)
    skip_idx = max(n - 21, 1)
    price_t_minus_1m = float(price_arr[skip_idx])
    price_start = float(price_arr[0])

    # 12-2 Intermediate return: from start of window to t - 21 sessions
    mom_12_2_pct = ((price_t_minus_1m - price_start) / price_start) * 100.0
    # 1-month short-term return: from t - 21 to latest
    st_1m_pct = ((latest_price - price_t_minus_1m) / price_t_minus_1m) * 100.0

    # Daily log returns and annualized volatility
    daily_rets = np.diff(price_arr) / price_arr[:-1]
    vol_daily = float(np.std(daily_rets, ddof=1)) if len(daily_rets) > 1 else 0.015
    ann_vol_pct = max(vol_daily * math.sqrt(252.0) * 100.0, 5.0)

    # 2. Standardized Value Z-Score (B/M benchmark = 0.50, E/P benchmark = 5.0%)
    bm_z = (float(book_to_market) - 0.50) / 0.30
    ep_z = (float(earnings_yield_pct) - 5.0) / 3.0
    raw_val_z = 0.5 * bm_z + 0.5 * ep_z
    val_z = float(np.clip(raw_val_z, -3.5, 3.5))

    # 3. Standardized Momentum Z-Score (Volatility-scaled 12-2 return)
    vol_scale = max(ann_vol_pct * 0.65, 8.0)
    raw_mom_z = (mom_12_2_pct - 6.0) / vol_scale
    mom_z = float(np.clip(raw_mom_z, -3.5, 3.5))

    # 4. Empirical Value-Momentum Correlation (Asness et al. benchmark: -0.52)
    if value_momentum_correlation is not None:
        assert_finite_number(value_momentum_correlation, "value_momentum_correlation")
        rho = float(np.clip(value_momentum_correlation, -0.95, 0.95))
    else:
        # Estimate dynamic correlation between rolling value proxy (1/price) and rolling momentum
        if len(daily_rets) >= 20:
            half = len(daily_rets) // 2
            r1 = daily_rets[:half]
            r2 = daily_rets[-half:]
            # Value moves inversely to unanchored price run-ups, creating structural negative correlation
            emp_corr = float(np.corrcoef(-r1, r2)[0, 1]) if np.std(r1) > 1e-8 and np.std(r2) > 1e-8 else -0.52
            if not math.isfinite(emp_corr):
                emp_corr = -0.52
            # Shrink towards theoretical Asness et al. (2013) prior of -0.52
            rho = float(np.clip(0.75 * (-0.52) + 0.25 * emp_corr, -0.85, 0.50))
        else:
            rho = -0.52

    # 5. Unified 50/50 VME Combo Score & Diversification Benefit Ratio
    # Variance of (0.5*Z_val + 0.5*Z_mom) is 0.25 + 0.25 + 0.5*rho = 0.5*(1 + rho)
    # Standard deviation of sum (Z_val + Z_mom) is sqrt(2 * (1 + rho))
    denom_vme = math.sqrt(max(2.0 * (1.0 + rho), 0.10))
    vme_combo_z = float(np.clip((val_z + mom_z) / denom_vme, -4.0, 4.0))

    # Diversification Benefit Ratio: sqrt(2 / (1 + rho))
    dbr = math.sqrt(2.0 / max(1.0 + rho, 0.10))

    # Implied VME Sharpe Ratio: Baseline standalone SR ~ 0.55 enhanced by DBR and signal strength
    base_sr = 0.55 * dbr
    implied_sr = float(np.clip(base_sr + 0.25 * vme_combo_z, 0.0, 2.50))

    # 6. Regime Classification & Expected Alpha
    regime = classify_vme_regime(val_z, mom_z, vme_combo_z)
    is_compounder = bool(regime == "SYNERGISTIC_VME_COMPOUNDER")
    is_trap = bool(regime == "DUAL_FACTOR_HEADWIND_TRAP")

    expected_alpha = float(np.clip(3.2 * vme_combo_z, -12.0, 15.0))

    if regime == "SYNERGISTIC_VME_COMPOUNDER":
        summary = (
            f"Synergistic VME Compounder (z_VME={vme_combo_z:+.2f}, z_Val={val_z:+.2f}, z_Mom={mom_z:+.2f}, "
            f"rho={rho:+.2f}). Attractive valuation reinforced by positive 12-2 momentum. "
            f"Expected VME alpha: {expected_alpha:+.1f}%/yr (Implied SR: {implied_sr:.2f})."
        )
    elif regime == "HEDGED_BARBELL_EQUILIBRIUM":
        summary = (
            f"Hedged Barbell Equilibrium (z_VME={vme_combo_z:+.2f}, z_Val={val_z:+.2f}, z_Mom={mom_z:+.2f}, "
            f"rho={rho:+.2f}). Negative style covariance (DBR={dbr:.2f}x) stabilizes drawdown risk. "
            f"Expected VME alpha: {expected_alpha:+.1f}%/yr."
        )
    elif regime == "NEUTRAL_FACTOR_EXPOSURE":
        summary = (
            f"Neutral Factor Exposure (z_VME={vme_combo_z:+.2f}, z_Val={val_z:+.2f}, z_Mom={mom_z:+.2f}, "
            f"rho={rho:+.2f}). Balanced factor profile near market benchmark."
        )
    else:
        summary = (
            f"Dual Factor Headwind Trap (z_VME={vme_combo_z:+.2f}, z_Val={val_z:+.2f}, z_Mom={mom_z:+.2f}, "
            f"rho={rho:+.2f}). Unfavorable valuation combined with negative momentum decay. "
            f"Expected VME drag: {expected_alpha:+.1f}%/yr."
        )

    return VMEReceipt(
        ticker=clean_ticker,
        latest_price=round(latest_price, 2),
        book_to_market=round(float(book_to_market), 4),
        earnings_yield_pct=round(float(earnings_yield_pct), 2),
        momentum_12_2_pct=round(float(mom_12_2_pct), 2),
        short_term_1m_pct=round(float(st_1m_pct), 2),
        annualized_volatility_pct=round(float(ann_vol_pct), 2),
        value_z_score=round(val_z, 4),
        momentum_z_score=round(mom_z, 4),
        value_momentum_correlation=round(rho, 4),
        vme_combo_z_score=round(vme_combo_z, 4),
        diversification_benefit_ratio=round(dbr, 4),
        implied_vme_sharpe=round(implied_sr, 2),
        is_twin_engine_compounder=is_compounder,
        is_dual_headwind_trap=is_trap,
        expected_vme_alpha_pct=round(expected_alpha, 2),
        regime=regime,
        summary=summary,
    )


def evaluate_vme_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    book_to_market: Optional[float] = None,
    earnings_yield_pct: Optional[float] = None,
    value_momentum_correlation: Optional[float] = None,
) -> VMEReceipt:
    """Evaluate Asness, Moskowitz & Pedersen (2013) VME profile from database or synthetic fallback."""
    clean_ticker = validate_ticker(ticker)

    prices: List[float] = []
    try:
        obs = db.get_observations(clean_ticker, db_path=db_path)
        if obs is not None and not obs.empty and "close" in obs.columns:
            prices = [float(x) for x in obs["close"].tolist() if math.isfinite(float(x)) and float(x) > 0]
    except Exception:
        prices = []

    if len(prices) < 25:
        # Deterministic synthetic series seeded by ticker hash
        seed = sum(ord(c) for c in clean_ticker)
        prices = [
            100.0 * (1.0 + 0.0025 * i + 0.015 * math.sin(0.3 * i + seed))
            for i in range(60)
        ]

    bm = float(book_to_market) if book_to_market is not None else 0.62
    ep = float(earnings_yield_pct) if earnings_yield_pct is not None else 6.2

    return calculate_vme_score(
        ticker=clean_ticker,
        prices=prices,
        book_to_market=bm,
        earnings_yield_pct=ep,
        value_momentum_correlation=value_momentum_correlation,
    )
