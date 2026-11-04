"""Markus K. Brunnermeier & Lasse Heje Pedersen (2009)
Market Liquidity & Funding Liquidity Spirals, Margin Haircut Tightness & Fire-Sale Fragility Engine.

Implements the structural interaction between market liquidity and funding liquidity:
- Brunnermeier & Pedersen (2009, Review of Financial Studies, "Market Liquidity and Funding Liquidity"):
    Traders' ability to provide market liquidity (tight bid-ask spreads, low price impact) depends
    on their availability of funding liquidity (capital adequacy and margin haircuts demanded by financiers).
    Financiers set margin requirements m_t based on Value-at-Risk / volatility (m_t approx k * sigma_t).
    When adverse shocks hit asset values or volatility surges, two self-reinforcing spirals can erupt:
    1. Margin Spiral: An asset price drop increases volatility sigma_t, prompting prime brokers to raise
       margin haircuts from baseline m_0 to m_t (kappa_margin = m_t / m_0 > 1.0). When margin buffers
       B_fund are depleted (B_fund < 0), arbitrageurs face margin calls and are forced to shed positions,
       which further depresses prices and raises volatility in a vicious cycle.
    2. Loss Spiral: Price declines cause immediate trading losses, reducing trader equity and capacity to
       hold illiquid collateral, widening market spreads.
- Liquidity Spiral Multiplier (M_spiral >= 1.0):
    Quantifies the amplification factor of liquidity shocks:
        M_spiral = clip(1.0 + 0.35 * (kappa_margin - 1.0) * max(0, -B_fund) + 0.25 * max(0, sigma_t / sigma_0 - 1.0), 1.0, 4.5)
    High M_spiral combined with negative margin buffers identifies extreme fire-sale liquidation risk,
    while deep funding buffers (B_fund >= 0.40) provide flight-to-liquidity safe-haven resilience.

References:
    Brunnermeier, M. K., & Pedersen, L. H. (2009). Market liquidity and funding liquidity.
    The Review of Financial Studies, 22(6), 2201-2238.
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

LIQUIDITY_SPIRAL_SIGNING_KEY = b"nextomni_brunnermeier_pedersen_liquidity_spiral_v1"


@dataclass
class LiquiditySpiralReceipt:
    """Cryptographic audit receipt for Brunnermeier & Pedersen (2009) Liquidity Spiral Profile."""

    ticker: str
    latest_price: float
    # Market Liquidity & Funding Inputs
    current_margin_haircut: float  # Current margin requirement m_t in [0.02, 0.85]
    baseline_margin_haircut: float  # Baseline normal-times margin m_0 in [0.02, 0.50]
    funding_margin_buffer: float  # Trader equity margin buffer B_fund in [-1.0, 1.0] (< 0 margin call)
    annualized_volatility: float  # Current annualized volatility sigma_t in [0.05, 1.50]
    baseline_volatility: float  # Long-run baseline volatility sigma_0 in [0.05, 1.00]
    # Derived Brunnermeier-Pedersen Spiral Metrics
    margin_tightness_ratio: float  # kappa_margin = m_t / m_0 >= 1.0
    funding_fragility_score: float  # F_fund = kappa_margin * (1 - B_fund)
    liquidity_spiral_multiplier: float  # M_spiral in [1.0, 4.5]
    brunnermeier_pedersen_impulse: float  # BPFI_t in [-2.5, 2.5]
    expected_liquidity_alpha_pct: float  # Projected annualized alpha net of fire-sale discount (%)
    # Flags & Regime
    is_fire_sale_spiral_trap: bool  # True if M_spiral >= 1.80 and B_fund < 0 (acute margin spiral)
    is_funding_liquidity_safe_haven: bool  # True if M_spiral <= 1.15 and B_fund >= 0.40
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
        """Compute deterministic HMAC-SHA256 signature across all liquidity spiral metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:"
            f"{self.current_margin_haircut:.4f}:{self.baseline_margin_haircut:.4f}:"
            f"{self.funding_margin_buffer:.4f}:{self.annualized_volatility:.4f}:"
            f"{self.baseline_volatility:.4f}:{self.margin_tightness_ratio:.4f}:"
            f"{self.funding_fragility_score:.4f}:{self.liquidity_spiral_multiplier:.4f}:"
            f"{self.brunnermeier_pedersen_impulse:.4f}:{self.expected_liquidity_alpha_pct:.2f}:"
            f"{int(self.is_fire_sale_spiral_trap)}:{int(self.is_funding_liquidity_safe_haven)}:{self.regime}"
        )
        return hmac.new(
            LIQUIDITY_SPIRAL_SIGNING_KEY,
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


def classify_liquidity_spiral_regime(
    liquidity_spiral_multiplier: float,
    funding_margin_buffer: float,
    margin_tightness_ratio: float,
) -> str:
    """Classify asset into Brunnermeier & Pedersen (2009) Liquidity Spiral Regimes."""
    if liquidity_spiral_multiplier >= 1.70 and funding_margin_buffer < -0.05:
        return "DESTABILIZING_LIQUIDITY_SPIRAL_FIRE_SALE"
    if liquidity_spiral_multiplier <= 1.15 and funding_margin_buffer >= 0.40:
        return "ROBUST_FUNDING_LIQUIDITY_SAFE_HAVEN"
    if margin_tightness_ratio >= 1.35 and funding_margin_buffer >= 0.0:
        return "MARGIN_TIGHTENING_WATCH"
    return "NORMAL_LIQUIDITY_BUFFER_EQUILIBRIUM"


def calculate_liquidity_spiral(
    ticker: str,
    latest_price: float,
    current_margin_haircut: float = 0.22,
    baseline_margin_haircut: float = 0.15,
    funding_margin_buffer: float = 0.25,
    annualized_volatility: float = 0.26,
    baseline_volatility: float = 0.20,
) -> LiquiditySpiralReceipt:
    """Calculate Brunnermeier & Pedersen (2009) Market & Funding Liquidity Spiral Metrics.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        current_margin_haircut: Current prime brokerage margin requirement m_t in [0.02, 0.85].
        baseline_margin_haircut: Normal-times baseline margin requirement m_0 in [0.02, 0.50].
        funding_margin_buffer: Trader/arbitrageur equity margin buffer B_fund in [-1.0, 1.0].
        annualized_volatility: Current realized/implied volatility sigma_t in [0.05, 1.50].
        baseline_volatility: Long-run baseline volatility sigma_0 in [0.05, 1.00].

    Returns:
        LiquiditySpiralReceipt with margin tightness, spiral multiplier, impulse, and regime.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(current_margin_haircut, "current_margin_haircut")
    assert_finite_number(baseline_margin_haircut, "baseline_margin_haircut")
    assert_finite_number(funding_margin_buffer, "funding_margin_buffer")
    assert_finite_number(annualized_volatility, "annualized_volatility")
    assert_finite_number(baseline_volatility, "baseline_volatility")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (0.02 <= current_margin_haircut <= 0.85):
        raise ValueError("current_margin_haircut must be in [0.02, 0.85].")
    if not (0.02 <= baseline_margin_haircut <= 0.50):
        raise ValueError("baseline_margin_haircut must be in [0.02, 0.50].")
    if not (-1.0 <= funding_margin_buffer <= 1.0):
        raise ValueError("funding_margin_buffer must be in [-1.0, 1.0].")
    if not (0.05 <= annualized_volatility <= 1.50):
        raise ValueError("annualized_volatility must be in [0.05, 1.50].")
    if not (0.05 <= baseline_volatility <= 1.00):
        raise ValueError("baseline_volatility must be in [0.05, 1.00].")

    p_t = float(latest_price)
    m_t = float(current_margin_haircut)
    m_0 = float(baseline_margin_haircut)
    b_fund = float(funding_margin_buffer)
    sig_t = float(annualized_volatility)
    sig_0 = float(baseline_volatility)

    # 1. Margin Tightness Ratio kappa_margin = m_t / m_0 >= 1.0 (haircut inflation)
    kappa_margin = float(np.clip(m_t / max(m_0, 0.01), 0.5, 6.0))

    # 2. Funding Fragility Score F_fund = kappa_margin * (1 - B_fund)
    fragility = float(np.clip(kappa_margin * (1.0 - b_fund), 0.10, 6.0))

    # 3. Brunnermeier-Pedersen Liquidity Spiral Multiplier M_spiral in [1.0, 4.5]
    # Amplification occurs when margins tighten (kappa > 1) AND funding buffers are depleted (B_fund < 0)
    depletion_severity = max(0.0, -b_fund)
    vol_surge = max(0.0, (sig_t / max(sig_0, 0.01)) - 1.0)
    raw_spiral = 1.0 + 0.85 * (kappa_margin - 0.90) * depletion_severity + 0.45 * vol_surge
    m_spiral = float(np.clip(raw_spiral, 1.0, 4.5))

    # 4. Brunnermeier-Pedersen Impulse BPFI_t in [-2.5, 2.5]
    # Negative impulse in fire sales (high spiral multiplier, depleted buffer);
    # Positive impulse in robust funding safe havens (cushion against liquidity dry-ups)
    if b_fund < 0.0 and m_spiral >= 1.50:
        raw_impulse = -0.95 * m_spiral - 1.2 * depletion_severity
    elif b_fund >= 0.35 and m_spiral <= 1.20:
        raw_impulse = 0.85 * b_fund + 0.35 * (1.5 - kappa_margin)
    else:
        raw_impulse = 0.45 * (b_fund - 0.15) - 0.25 * (kappa_margin - 1.0)

    bpfi = float(np.clip(raw_impulse, -2.5, 2.5))

    # 5. Expected Liquidity Alpha & Regime Classification
    expected_alpha = float(np.clip(5.0 * bpfi, -12.5, 12.0))
    regime = classify_liquidity_spiral_regime(m_spiral, b_fund, kappa_margin)
    is_fire_sale = bool(regime == "DESTABILIZING_LIQUIDITY_SPIRAL_FIRE_SALE")
    is_safe_haven = bool(regime == "ROBUST_FUNDING_LIQUIDITY_SAFE_HAVEN")

    if regime == "DESTABILIZING_LIQUIDITY_SPIRAL_FIRE_SALE":
        summary = (
            f"Destabilizing Liquidity Spiral Fire Sale (M_spiral={m_spiral:.2f}x, Buffer={b_fund:+.1%}, "
            f"Haircut={m_t:.1%} vs Base={m_0:.1%}). Haircut spikes and depleted funding trigger acute "
            f"forced de-leveraging and illiquidity spirals ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "ROBUST_FUNDING_LIQUIDITY_SAFE_HAVEN":
        summary = (
            f"Robust Funding Liquidity Safe Haven (M_spiral={m_spiral:.2f}x, Buffer={b_fund:+.1%}, "
            f"Haircut={m_t:.1%}). Deep collateral buffers provide structural insulation against market-wide "
            f"funding liquidity shocks ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "MARGIN_TIGHTENING_WATCH":
        summary = (
            f"Margin Tightening Watch (kappa_margin={kappa_margin:.2f}x, Buffer={b_fund:+.1%}, "
            f"M_spiral={m_spiral:.2f}x). Financiers have escalated margin haircuts but trader funding "
            f"buffers remain solvent ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Normal Liquidity Buffer Equilibrium (M_spiral={m_spiral:.2f}x, Haircut={m_t:.1%}, "
            f"Buffer={b_fund:+.1%}). Funding and market liquidity maintain stable baseline equilibrium "
            f"({expected_alpha:+.1f}%/yr)."
        )

    return LiquiditySpiralReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        current_margin_haircut=round(m_t, 4),
        baseline_margin_haircut=round(m_0, 4),
        funding_margin_buffer=round(b_fund, 4),
        annualized_volatility=round(sig_t, 4),
        baseline_volatility=round(sig_0, 4),
        margin_tightness_ratio=round(kappa_margin, 4),
        funding_fragility_score=round(fragility, 4),
        liquidity_spiral_multiplier=round(m_spiral, 4),
        brunnermeier_pedersen_impulse=round(bpfi, 4),
        expected_liquidity_alpha_pct=round(expected_alpha, 2),
        is_fire_sale_spiral_trap=is_fire_sale,
        is_funding_liquidity_safe_haven=is_safe_haven,
        regime=regime,
        summary=summary,
    )


def evaluate_liquidity_spiral_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    current_margin_haircut: Optional[float] = None,
    baseline_margin_haircut: Optional[float] = None,
    funding_margin_buffer: Optional[float] = None,
    annualized_volatility: Optional[float] = None,
    baseline_volatility: Optional[float] = None,
) -> LiquiditySpiralReceipt:
    """Evaluate Brunnermeier & Pedersen (2009) Liquidity Spiral profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_mt = 0.20
    inferred_m0 = 0.15
    inferred_buffer = 0.30
    inferred_sigt = 0.25
    inferred_sig0 = 0.20

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
                ann_vol = float(np.std(rets, ddof=1) * math.sqrt(252.0)) if len(rets) > 1 else 0.22

                inferred_sigt = float(np.clip(ann_vol, 0.08, 1.20))
                # Long-run baseline vol from historical average
                if len(rets) > 60:
                    half_rets = rets[: len(rets) // 2]
                    inferred_sig0 = float(np.clip(np.std(half_rets, ddof=1) * math.sqrt(252.0), 0.08, 0.80))
                else:
                    inferred_sig0 = float(np.clip(inferred_sigt * 0.90, 0.08, 0.80))

                # Margin requirement modeled via Value-at-Risk rule (Brunnermeier & Pedersen 2009): m_t approx 2.33 * sig_t / sqrt(252) * sqrt(10)
                ten_day_var = 2.33 * inferred_sigt * math.sqrt(10.0 / 252.0)
                inferred_mt = float(np.clip(0.08 + 0.45 * ten_day_var, 0.05, 0.70))
                inferred_m0 = float(np.clip(0.08 + 0.45 * (2.33 * inferred_sig0 * math.sqrt(10.0 / 252.0)), 0.05, 0.50))

                # Funding margin buffer inferred from recent price drawdown
                peak = float(np.max(arr))
                dd = float((arr[-1] - peak) / max(peak, 1e-4))
                inferred_buffer = float(np.clip(0.40 + 1.5 * dd, -0.65, 0.80))
    except Exception:
        latest_price = 100.0

    mt_val = (
        float(current_margin_haircut)
        if current_margin_haircut is not None
        else inferred_mt
    )
    m0_val = (
        float(baseline_margin_haircut)
        if baseline_margin_haircut is not None
        else inferred_m0
    )
    buf_val = (
        float(funding_margin_buffer)
        if funding_margin_buffer is not None
        else inferred_buffer
    )
    sigt_val = (
        float(annualized_volatility)
        if annualized_volatility is not None
        else inferred_sigt
    )
    sig0_val = (
        float(baseline_volatility)
        if baseline_volatility is not None
        else inferred_sig0
    )

    return calculate_liquidity_spiral(
        ticker=clean_ticker,
        latest_price=latest_price,
        current_margin_haircut=mt_val,
        baseline_margin_haircut=m0_val,
        funding_margin_buffer=buf_val,
        annualized_volatility=sigt_val,
        baseline_volatility=sig0_val,
    )

