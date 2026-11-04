"""Tarun Chordia, Richard Roll & Avanidhar Subrahmanyam (2000, 2001, 2002)
Commonality in Liquidity & Order Imbalance Market-Wide Friction Engine.

Implements the market microstructure co-movement and order-flow imbalance asset pricing framework:
- Commonality in Liquidity (Chordia, Roll & Subrahmanyam, 2000):
    Delta_S_{i,t} = alpha_i + beta_{Delta_S,i} * Delta_S_{M,t} + epsilon_{i,t}
    Individual stock bid-ask spreads and quoted depth co-move significantly with market-wide and
    industry-wide liquidity variations. High commonality beta (beta^{eff}_{L,i}) exposes investors
    to non-diversifiable transaction cost spikes during market stress.
- Asymmetric Down-Market Illiquidity Surge (Chordia, Roll & Subrahmanyam, 2001):
    Market liquidity plummets asymmetrically in down markets as dealer inventory financing tightens.
- Order Imbalance & Institutional Order-Splitting (Chordia, Roll & Subrahmanyam, 2002):
    Signed order imbalance OIB_t = (BuyVol_t - SellVol_t) / TotalVol_t exhibits high positive
    first-order autocorrelation (rho_{OIB}) due to institutional order-splitting, straining market
    maker inventory and driving short-horizon price pressure followed by liquidity-induced reversals.
- Regimes:
    - IDIOSYNCRATIC_LIQUIDITY_HAVEN: Low spread/depth co-movement (beta^{eff}_{L} <= 0.45 and SICI_t <= 0.65)
    - NORMAL_MARKET_COMOVEMENT: Moderate market-wide liquidity sensitivity
    - ORDER_IMBALANCE_PRESSURE_WATCH: Persistent one-sided order-splitting (|OIB_t| >= 0.28 and rho_{OIB} >= 0.35)
    - SYSTEMIC_LIQUIDITY_CONTAGION_TRAP: Severe spread commonality (beta^{eff}_{L} >= 1.35 or SICI_t >= 1.75)

References:
    Chordia, T., Roll, R., & Subrahmanyam, A. (2000). Commonality in liquidity.
    Journal of Financial Economics, 56(1), 3-28.
    Chordia, T., Roll, R., & Subrahmanyam, A. (2001). Market liquidity and trading activity.
    The Journal of Finance, 56(2), 501-530.
    Chordia, T., Roll, R., & Subrahmanyam, A. (2002). Order imbalance, liquidity, and market returns.
    Journal of Financial Economics, 65(1), 111-130.
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

LIQUIDITY_COMMONALITY_SIGNING_KEY = b"nextomni_liquidity_commonality_secret_v1"


@dataclass
class LiquidityCommonalityReceipt:
    """Cryptographic audit receipt for Chordia, Roll & Subrahmanyam (2000, 2001) Liquidity Commonality Profile."""

    ticker: str
    latest_price: float
    # Microstructure Commonality & Order Imbalance Inputs
    spread_commonality_beta: float  # Co-movement of stock spread changes with market spread changes
    depth_commonality_beta: float  # Co-movement of stock depth with market depth
    liquidity_market_r2: float  # Market model explanatory R^2_L in [0.0, 1.0]
    signed_order_imbalance: float  # OIB_t = (BuyVol - SellVol) / TotalVol in [-1.0, 1.0]
    oib_autocorrelation: float  # First-order OIB autocorrelation rho_{OIB} in [-1.0, 1.0]
    down_market_illiquidity_surge: float  # Asymmetric down-market spread multiplier (>= 0.5)
    # Derived Chordia-Roll-Subrahmanyam Metrics
    effective_commonality_beta: float  # 0.65 * beta_spread + 0.35 * beta_depth
    order_splitting_pressure_score: float  # |OIB_t| * (1 + max(rho_{OIB}, 0)) in [0.0, 2.0]
    systemic_inventory_contagion_index: float  # SICI_t in [0.0, 3.5]
    expected_commonality_alpha_pct: float  # Projected net execution-adjusted alpha / cost drag (%)
    # Flags & Regime
    is_idiosyncratic_liquidity_haven: bool  # True if beta_eff <= 0.45 and SICI_t <= 0.65
    is_systemic_contagion_trap: bool  # True if beta_eff >= 1.35 or SICI_t >= 1.75
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
        """Compute deterministic HMAC-SHA256 signature across core liquidity commonality metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.spread_commonality_beta:.4f}:"
            f"{self.depth_commonality_beta:.4f}:{self.liquidity_market_r2:.4f}:"
            f"{self.signed_order_imbalance:.4f}:{self.oib_autocorrelation:.4f}:"
            f"{self.down_market_illiquidity_surge:.4f}:{self.effective_commonality_beta:.4f}:"
            f"{self.order_splitting_pressure_score:.4f}:{self.systemic_inventory_contagion_index:.4f}:"
            f"{self.expected_commonality_alpha_pct:.2f}:{int(self.is_idiosyncratic_liquidity_haven)}:"
            f"{int(self.is_systemic_contagion_trap)}:{self.regime}"
        )
        return hmac.new(
            LIQUIDITY_COMMONALITY_SIGNING_KEY,
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


def classify_commonality_regime(
    effective_beta: float,
    signed_oib: float,
    oib_autocorr: float,
    contagion_index: float,
) -> str:
    """Classify asset into Chordia, Roll & Subrahmanyam (2000, 2001, 2002) Liquidity Regimes."""
    if effective_beta >= 1.35 or contagion_index >= 1.75:
        return "SYSTEMIC_LIQUIDITY_CONTAGION_TRAP"
    if abs(signed_oib) >= 0.28 and oib_autocorr >= 0.35:
        return "ORDER_IMBALANCE_PRESSURE_WATCH"
    if effective_beta <= 0.45 and contagion_index <= 0.65:
        return "IDIOSYNCRATIC_LIQUIDITY_HAVEN"
    return "NORMAL_MARKET_COMOVEMENT"


def calculate_liquidity_commonality(
    ticker: str,
    latest_price: float,
    spread_commonality_beta: float = 0.72,
    depth_commonality_beta: float = 0.65,
    liquidity_market_r2: float = 0.22,
    signed_order_imbalance: float = 0.08,
    oib_autocorrelation: float = 0.18,
    down_market_illiquidity_surge: float = 1.15,
) -> LiquidityCommonalityReceipt:
    """Calculate Chordia, Roll & Subrahmanyam (2000, 2001, 2002) Commonality in Liquidity & Order Imbalance.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price ($).
        spread_commonality_beta: Sensitivity of proportional spread changes to market spread changes.
        depth_commonality_beta: Sensitivity of quoted depth changes to market depth changes.
        liquidity_market_r2: Market-model R^2 for spread/depth co-movement in [0.0, 1.0].
        signed_order_imbalance: Net signed volume order imbalance in [-1.0, 1.0].
        oib_autocorrelation: First-order autocorrelation of daily order imbalance in [-1.0, 1.0].
        down_market_illiquidity_surge: Asymmetric spread widening multiplier in down markets (>= 0.5).

    Returns:
        LiquidityCommonalityReceipt with effective beta, order-splitting pressure, SICI_t, and regime.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(spread_commonality_beta, "spread_commonality_beta")
    assert_finite_number(depth_commonality_beta, "depth_commonality_beta")
    assert_finite_number(liquidity_market_r2, "liquidity_market_r2")
    assert_finite_number(signed_order_imbalance, "signed_order_imbalance")
    assert_finite_number(oib_autocorrelation, "oib_autocorrelation")
    assert_finite_number(down_market_illiquidity_surge, "down_market_illiquidity_surge")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (0.0 <= liquidity_market_r2 <= 1.0):
        raise ValueError("liquidity_market_r2 must be in [0.0, 1.0].")
    if not (-1.0 <= signed_order_imbalance <= 1.0):
        raise ValueError("signed_order_imbalance must be in [-1.0, 1.0].")
    if not (-1.0 <= oib_autocorrelation <= 1.0):
        raise ValueError("oib_autocorrelation must be in [-1.0, 1.0].")
    if down_market_illiquidity_surge < 0.5:
        raise ValueError("down_market_illiquidity_surge must be at least 0.5.")

    # 1. Effective Commonality Beta (Chordia, Roll & Subrahmanyam, 2000)
    eff_beta = float(
        np.clip(
            0.65 * float(spread_commonality_beta) + 0.35 * float(depth_commonality_beta),
            -0.50,
            3.50,
        )
    )

    # 2. Order-Splitting Pressure Score (Chordia, Roll & Subrahmanyam, 2002)
    pos_autocorr = max(float(oib_autocorrelation), 0.0)
    oib_pressure = float(
        np.clip(abs(float(signed_order_imbalance)) * (1.0 + pos_autocorr), 0.0, 2.0)
    )

    # 3. Systemic Inventory Contagion Index SICI_t in [0.0, 3.5]
    raw_sici = (
        max(eff_beta, 0.0)
        * (1.0 + 1.2 * float(liquidity_market_r2))
        * float(down_market_illiquidity_surge)
        + 0.85 * oib_pressure
    )
    sici = float(np.clip(raw_sici, 0.0, 3.5))

    # 4. Regime Classification & Expected Execution-Adjusted Alpha
    regime = classify_commonality_regime(
        eff_beta, float(signed_order_imbalance), float(oib_autocorrelation), sici
    )
    is_haven = bool(regime == "IDIOSYNCRATIC_LIQUIDITY_HAVEN")
    is_trap = bool(regime == "SYSTEMIC_LIQUIDITY_CONTAGION_TRAP")

    expected_alpha = float(
        np.clip(3.4 - 3.6 * sici - 1.5 * oib_pressure, -10.5, 4.5)
    )

    if regime == "IDIOSYNCRATIC_LIQUIDITY_HAVEN":
        summary = (
            f"Idiosyncratic Liquidity Haven (Beta_L={eff_beta:.2f}, R2={liquidity_market_r2:.2f}, SICI={sici:.2f}). "
            f"Low co-movement with market-wide spread widening insulates execution during systemic selloffs. "
            f"Net liquidity resilience alpha: {expected_alpha:+.1f}%/yr."
        )
    elif regime == "NORMAL_MARKET_COMOVEMENT":
        summary = (
            f"Normal Market Co-Movement (Beta_L={eff_beta:.2f}, OIB={signed_order_imbalance:+.2f}, SICI={sici:.2f}). "
            f"Typical Chordia-Roll-Subrahmanyam market liquidity sensitivity ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "ORDER_IMBALANCE_PRESSURE_WATCH":
        summary = (
            f"Order Imbalance Pressure Watch (OIB={signed_order_imbalance:+.2f}, Autocorr={oib_autocorrelation:+.2f}, "
            f"Pressure={oib_pressure:.2f}). Persistent institutional order-splitting strains dealer inventory; "
            f"watch for short-horizon reversal once flow subsides ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Systemic Liquidity Contagion Trap (Beta_L={eff_beta:.2f}, DownSurge={down_market_illiquidity_surge:.2f}x, "
            f"SICI={sici:.2f}). Severe spread commonality and asymmetric down-market depth evaporation create "
            f"execution fragility ({expected_alpha:+.1f}%/yr)."
        )

    return LiquidityCommonalityReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        spread_commonality_beta=round(float(spread_commonality_beta), 4),
        depth_commonality_beta=round(float(depth_commonality_beta), 4),
        liquidity_market_r2=round(float(liquidity_market_r2), 4),
        signed_order_imbalance=round(float(signed_order_imbalance), 4),
        oib_autocorrelation=round(float(oib_autocorrelation), 4),
        down_market_illiquidity_surge=round(float(down_market_illiquidity_surge), 4),
        effective_commonality_beta=round(eff_beta, 4),
        order_splitting_pressure_score=round(oib_pressure, 4),
        systemic_inventory_contagion_index=round(sici, 4),
        expected_commonality_alpha_pct=round(expected_alpha, 2),
        is_idiosyncratic_liquidity_haven=is_haven,
        is_systemic_contagion_trap=is_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_liquidity_commonality_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    spread_commonality_beta: Optional[float] = None,
    depth_commonality_beta: Optional[float] = None,
    liquidity_market_r2: Optional[float] = None,
    signed_order_imbalance: Optional[float] = None,
    oib_autocorrelation: Optional[float] = None,
    down_market_illiquidity_surge: Optional[float] = None,
) -> LiquidityCommonalityReceipt:
    """Evaluate Chordia, Roll & Subrahmanyam (2000, 2001) profile from SQLite OHLCV history and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_oib = 0.06
    inferred_autocorr = 0.14
    inferred_down_surge = 1.12

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
            if len(closes) >= 20 and "volume" in obs.columns:
                vols = [
                    max(float(v), 1.0)
                    for v in obs["volume"].tolist()[-len(closes) :]
                ]
                arr_c = np.asarray(closes[-20:], dtype=float)
                arr_v = np.asarray(vols[-20:], dtype=float)
                rets = np.diff(arr_c) / arr_c[:-1]
                signs = np.sign(rets)
                signed_vols = signs * arr_v[1:]
                total_vol = float(np.sum(arr_v[1:]))
                if total_vol > 0:
                    inferred_oib = float(
                        np.clip(np.sum(signed_vols) / total_vol, -0.95, 0.95)
                    )
                if len(signs) >= 6 and np.std(signs[:-1]) > 0 and np.std(signs[1:]) > 0:
                    corr = float(np.corrcoef(signs[:-1], signs[1:])[0, 1])
                    if math.isfinite(corr):
                        inferred_autocorr = float(np.clip(corr, -0.90, 0.90))
                down_rets = rets[rets < 0]
                up_rets = rets[rets >= 0]
                if len(down_rets) >= 2 and len(up_rets) >= 2:
                    ratio = float(np.mean(np.abs(down_rets))) / max(
                        float(np.mean(np.abs(up_rets))), 1e-4
                    )
                    inferred_down_surge = float(np.clip(ratio, 0.75, 2.50))
    except Exception:
        latest_price = 100.0

    b_spread = (
        float(spread_commonality_beta)
        if spread_commonality_beta is not None
        else 0.38
    )
    b_depth = (
        float(depth_commonality_beta) if depth_commonality_beta is not None else 0.35
    )
    r2_l = float(liquidity_market_r2) if liquidity_market_r2 is not None else 0.16
    oib = (
        float(signed_order_imbalance)
        if signed_order_imbalance is not None
        else inferred_oib
    )
    rho_oib = (
        float(oib_autocorrelation)
        if oib_autocorrelation is not None
        else inferred_autocorr
    )
    surge = (
        float(down_market_illiquidity_surge)
        if down_market_illiquidity_surge is not None
        else inferred_down_surge
    )

    return calculate_liquidity_commonality(
        ticker=clean_ticker,
        latest_price=latest_price,
        spread_commonality_beta=b_spread,
        depth_commonality_beta=b_depth,
        liquidity_market_r2=r2_l,
        signed_order_imbalance=oib,
        oib_autocorrelation=rho_oib,
        down_market_illiquidity_surge=surge,
    )

