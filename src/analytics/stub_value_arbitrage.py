"""Owen A. Lamont & Richard H. Thaler (2003) / Mark Mitchell, Todd Pulvino & Erik Stafford (2002)
Limited Arbitrage, Equity Carve-Out Stub Value & Short-Sale Constraints Engine.

Implements the Law of One Price (LOOP) violation and equity carve-out stub arbitrage framework:
- Lamont & Thaler (2003, Journal of Finance, "Can the Market Add and Subtract?"):
    In an equity carve-out or parent-subsidiary corporate structure, a parent firm owns alpha shares
    of a publicly traded subsidiary per parent share. The implied market value of the parent company's
    remaining non-subsidiary core operating business (the "stub value") is:
        S_{stub,t} = P_{parent,t} - alpha * P_{sub,t}
    Rationally, limited liability dictates that S_{stub,t} >= 0.
    However, when retail or speculative demand causes extreme overvaluation of the subsidiary and
    shorting the subsidiary is impeded by high borrow fees, rebate rate penalties, institutional recall
    risk, or supply shortages, S_{stub,t} turns negative (Phi_{stub,t} = S_{stub,t} / P_{parent,t} < 0).
- Mitchell, Pulvino & Stafford (2002, Journal of Finance, "Limited Arbitrage in Equity Markets"):
    Negative stub values represent blatant, persistent violations of the Law of One Price. Rational
    arbitrageurs seeking to buy the cheap parent and short the expensive subsidiary face severe limits
    to arbitrage: collateral haircuts, borrow fee erosion, and path-dependent termination risk.
    Consequently, negative stub values predict massive long-run underperformance of the subsidiary and
    gradual convergence back to non-negative stub parity.

References:
    Lamont, O. A., & Thaler, R. H. (2003). Can the market add and subtract? Mispricings in tech
    stock carve-outs. The Journal of Finance, 58(6), 2271-2301.
    Mitchell, M., Pulvino, T., & Stafford, E. (2002). Limited arbitrage in equity markets: The
    case of negative stub values. The Journal of Finance, 57(2), 551-584.
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

STUB_VALUE_ARBITRAGE_SIGNING_KEY = b"nextomni_lamont_thaler_stub_arbitrage_v1"


@dataclass
class StubValueArbitrageReceipt:
    """Cryptographic audit receipt for Lamont & Thaler (2003) Stub Value Arbitrage Profile."""

    parent_ticker: str
    latest_parent_price: float
    # Parent-Subsidiary & Carve-Out Inputs
    subsidiary_ticker: str
    subsidiary_price: float  # P_{sub,t} > 0
    shares_per_parent_ratio: float  # alpha in [0.01, 10.0]
    annual_borrow_fee_pct: float  # Annualized short borrowing fee c_{borrow} in [0.0, 100.0]%
    # Derived Lamont-Thaler & Mitchell-Pulvino-Stafford Metrics
    implied_subsidiary_value_per_share: float  # alpha * P_{sub,t}
    implied_stub_value: float  # S_{stub,t} = P_{parent,t} - alpha * P_{sub,t}
    stub_fraction_of_parent: float  # Phi_{stub,t} = S_{stub,t} / P_{parent,t} in [-2.5, 1.0]
    short_sale_friction_multiplier: float  # Psi_{short} >= 1.0
    lamont_thaler_arbitrage_impulse: float  # LTAI_t in [-2.5, 2.5]
    expected_convergence_alpha_pct: float  # Projected annualized convergence alpha (%)
    # Flags & Regime
    is_negative_stub_violation: bool  # True if Phi_{stub} < -0.05 (flagrant LOOP violation)
    is_compressed_stub_opportunity: bool  # True if 0.0 <= Phi_{stub} <= 0.15 (cheap stub convergence)
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
        """Compute deterministic HMAC-SHA256 signature across all stub arbitrage metrics."""
        payload = (
            f"{self.parent_ticker}:{self.latest_parent_price:.2f}:{self.subsidiary_ticker}:"
            f"{self.subsidiary_price:.2f}:{self.shares_per_parent_ratio:.4f}:"
            f"{self.annual_borrow_fee_pct:.4f}:{self.implied_subsidiary_value_per_share:.2f}:"
            f"{self.implied_stub_value:.2f}:{self.stub_fraction_of_parent:.4f}:"
            f"{self.short_sale_friction_multiplier:.4f}:{self.lamont_thaler_arbitrage_impulse:.4f}:"
            f"{self.expected_convergence_alpha_pct:.2f}:"
            f"{int(self.is_negative_stub_violation)}:{int(self.is_compressed_stub_opportunity)}:{self.regime}"
        )
        return hmac.new(
            STUB_VALUE_ARBITRAGE_SIGNING_KEY,
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


def classify_stub_regime(
    stub_fraction: float,
    borrow_fee_pct: float,
    impulse: float,
) -> str:
    """Classify asset into Lamont & Thaler (2003) Stub Value Arbitrage Regimes."""
    if stub_fraction < -0.05:
        return "NEGATIVE_STUB_SHORT_SALE_IMPEDIMENT_TRAP"
    if 0.0 <= stub_fraction <= 0.15 and impulse >= 0.35:
        return "COMPRESSED_STUB_ARBITRAGE_CONVERGENCE"
    return "NORMAL_STUB_OPERATING_EQUILIBRIUM"


def calculate_stub_value_arbitrage(
    parent_ticker: str,
    latest_parent_price: float,
    subsidiary_ticker: str = "SUB",
    subsidiary_price: float = 85.0,
    shares_per_parent_ratio: float = 1.20,
    annual_borrow_fee_pct: float = 18.0,
) -> StubValueArbitrageReceipt:
    """Calculate Lamont & Thaler (2003) & Mitchell-Pulvino-Stafford (2002) Stub Value Arbitrage Metrics.

    Args:
        parent_ticker: Validated parent stock ticker symbol.
        latest_parent_price: Latest parent stock price P_{parent,t} ($ > 0).
        subsidiary_ticker: Validated subsidiary stock ticker symbol.
        subsidiary_price: Carved-out subsidiary stock price P_{sub,t} ($ > 0).
        shares_per_parent_ratio: Number of subsidiary shares owned per parent share alpha in [0.01, 10.0].
        annual_borrow_fee_pct: Annualized subsidiary short borrowing fee c_{borrow} in [0.0, 100.0]%.

    Returns:
        StubValueArbitrageReceipt with implied stub value, stub fraction, friction multiplier, and regime.
    """
    clean_parent = validate_ticker(parent_ticker)
    clean_sub = validate_ticker(subsidiary_ticker)
    assert_finite_number(latest_parent_price, "latest_parent_price")
    assert_finite_number(subsidiary_price, "subsidiary_price")
    assert_finite_number(shares_per_parent_ratio, "shares_per_parent_ratio")
    assert_finite_number(annual_borrow_fee_pct, "annual_borrow_fee_pct")

    if latest_parent_price <= 0.0:
        raise ValueError("latest_parent_price must be strictly positive.")
    if subsidiary_price <= 0.0:
        raise ValueError("subsidiary_price must be strictly positive.")
    if not (0.01 <= shares_per_parent_ratio <= 10.0):
        raise ValueError("shares_per_parent_ratio must be in [0.01, 10.0].")
    if not (0.0 <= annual_borrow_fee_pct <= 100.0):
        raise ValueError("annual_borrow_fee_pct must be in [0.0, 100.0]%.")

    p_parent = float(latest_parent_price)
    p_sub = float(subsidiary_price)
    alpha = float(shares_per_parent_ratio)
    borrow_fee = float(annual_borrow_fee_pct)

    # 1. Implied Subsidiary Value Per Parent Share & Implied Stub Value
    sub_val_per_parent = float(alpha * p_sub)
    stub_val = float(p_parent - sub_val_per_parent)
    stub_frac = float(np.clip(stub_val / max(p_parent, 1e-4), -2.5, 1.0))

    # 2. Short-Sale Friction Multiplier Psi_{short}
    # Borrow fee creates a hurdle for long-parent/short-sub convergence arbitrage
    borrow_dec = borrow_fee / 100.0
    friction_mult = float(1.0 + 2.4 * borrow_dec * (1.0 if stub_frac < 0.0 else 0.4))

    # 3. Lamont-Thaler Arbitrage Impulse LTAI_t in [-2.5, 2.5]
    if stub_frac < -0.05:
        # Flagrant negative stub: trapped by short-sale constraints, predicting severe subsidiary collapse
        raw_impulse = -1.8 - 0.7 * min(abs(stub_frac), 1.5) - 0.4 * min(friction_mult - 1.0, 2.0)
    elif 0.0 <= stub_frac <= 0.15:
        # Compressed positive stub: parent core business trading at a steep discount, attractive convergence
        raw_impulse = 0.85 + 2.5 * (0.15 - stub_frac)
    else:
        # Normal operating stub equilibrium
        raw_impulse = 0.25 * (stub_frac - 0.35)

    ltai = float(np.clip(raw_impulse, -2.5, 2.5))

    # 4. Expected Convergence Alpha & Regime Classification
    expected_alpha = float(np.clip(5.2 * ltai, -12.5, 12.0))
    regime = classify_stub_regime(stub_frac, borrow_fee, ltai)
    is_neg_stub = bool(regime == "NEGATIVE_STUB_SHORT_SALE_IMPEDIMENT_TRAP")
    is_compressed = bool(regime == "COMPRESSED_STUB_ARBITRAGE_CONVERGENCE")

    if regime == "NEGATIVE_STUB_SHORT_SALE_IMPEDIMENT_TRAP":
        summary = (
            f"Negative Stub Short-Sale Impediment Trap (S_stub=${stub_val:.2f}, Phi_stub={stub_frac:+.1%}, "
            f"BorrowFee={borrow_fee:.1f}%). Flagrant Law of One Price violation where subsidiary is wildly "
            f"overvalued relative to parent ({clean_parent}); short constraints impede arbitrage ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "COMPRESSED_STUB_ARBITRAGE_CONVERGENCE":
        summary = (
            f"Compressed Stub Arbitrage Convergence (S_stub=${stub_val:.2f}, Phi_stub={stub_frac:+.1%}, "
            f"alpha={alpha:.2f}x). Parent core operating business is deeply discounted relative to embedded "
            f"subsidiary holdings, projecting convergence alpha ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Normal Stub Operating Equilibrium (S_stub=${stub_val:.2f}, Phi_stub={stub_frac:+.1%}). "
            f"Parent-subsidiary market values reflect rational unbundled operating enterprise value "
            f"({expected_alpha:+.1f}%/yr)."
        )

    return StubValueArbitrageReceipt(
        parent_ticker=clean_parent,
        latest_parent_price=round(p_parent, 2),
        subsidiary_ticker=clean_sub,
        subsidiary_price=round(p_sub, 2),
        shares_per_parent_ratio=round(alpha, 4),
        annual_borrow_fee_pct=round(borrow_fee, 2),
        implied_subsidiary_value_per_share=round(sub_val_per_parent, 2),
        implied_stub_value=round(stub_val, 2),
        stub_fraction_of_parent=round(stub_frac, 4),
        short_sale_friction_multiplier=round(friction_mult, 4),
        lamont_thaler_arbitrage_impulse=round(ltai, 4),
        expected_convergence_alpha_pct=round(expected_alpha, 2),
        is_negative_stub_violation=is_neg_stub,
        is_compressed_stub_opportunity=is_compressed,
        regime=regime,
        summary=summary,
    )


def evaluate_stub_value_arbitrage_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    subsidiary_ticker: Optional[str] = None,
    subsidiary_price: Optional[float] = None,
    shares_per_parent_ratio: Optional[float] = None,
    annual_borrow_fee_pct: Optional[float] = None,
) -> StubValueArbitrageReceipt:
    """Evaluate Lamont & Thaler (2003) Stub Value Arbitrage profile from SQLite OHLCV and overrides."""
    clean_parent = validate_ticker(ticker)

    latest_price = 100.0
    inferred_sub_ticker = "SUB"
    inferred_sub_price = 70.0
    inferred_ratio = 1.0
    inferred_fee = 5.0

    try:
        obs = db.get_observations(clean_parent, db_path=db_path)
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

                # Infer baseline subsidiary price and borrow fee from parent volatility
                inferred_sub_price = float(np.clip(latest_price * (0.65 + 0.50 * ann_vol), 10.0, 1000.0))
                inferred_fee = float(np.clip(4.0 + 35.0 * ann_vol, 1.0, 65.0))
    except Exception:
        latest_price = 100.0

    sub_tick = (
        validate_ticker(subsidiary_ticker)
        if subsidiary_ticker is not None
        else inferred_sub_ticker
    )
    sub_p = (
        float(subsidiary_price)
        if subsidiary_price is not None
        else inferred_sub_price
    )
    ratio_val = (
        float(shares_per_parent_ratio)
        if shares_per_parent_ratio is not None
        else inferred_ratio
    )
    fee_val = (
        float(annual_borrow_fee_pct)
        if annual_borrow_fee_pct is not None
        else inferred_fee
    )

    return calculate_stub_value_arbitrage(
        parent_ticker=clean_parent,
        latest_parent_price=latest_price,
        subsidiary_ticker=sub_tick,
        subsidiary_price=sub_p,
        shares_per_parent_ratio=ratio_val,
        annual_borrow_fee_pct=fee_val,
    )

