"""Kyle's (1985) Informed Trading Lambda & Market Price Impact Engine.

Implements Albert S. Kyle's (1985) continuous auction market microstructure model
to quantify price impact, adverse selection risk, and market depth from signed order flow.

References:
    Kyle, A. S. (1985). Continuous auctions and informed trader.
    Econometrica, 53(6), 1315-1335.
    Hasbrouck, J. (1991). Measuring the information content of stock trades.
    The Journal of Finance, 46(1), 179-207.
    Bouchaud, J. P., Gefen, Y., Potters, M., & Wyart, M. (2004). Fluctuations and
    response in financial markets: the subtle nature of 'random' price changes.
    Quantitative Finance, 4(2), 176-190.
"""

import hashlib
import hmac
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from src.data.db import get_observations
from src.security.validators import assert_finite_number, validate_ticker

PRICE_IMPACT_SIGNING_KEY = b"nextomni_price_impact_audit_secret_key_v1"


@dataclass
class PriceImpactReceipt:
    """Cryptographic audit receipt for Kyle's Lambda and price impact estimation."""

    ticker: str
    sample_sessions: int
    rolling_window: int
    kyle_lambda_pct_per_million: float  # Percentage return per $1M net dollar volume
    kyle_lambda_abs_per_k_shares: float  # Price change ($) per 1,000 shares
    market_depth_millions: float  # Dollar volume ($M) to move price by 1%
    average_daily_volume_shares: float
    average_daily_dollar_volume: float
    r_squared: float
    adverse_selection_regime: str  # LOW_ADVERSE_SELECTION, MODERATE_IMPACT, HIGH_ADVERSE_SELECTION
    sqrt_impact_1pct_adv_bps: float  # Basis points impact for trade of 1% ADV
    sqrt_impact_5pct_adv_bps: float  # Basis points impact for trade of 5% ADV
    sqrt_impact_10pct_adv_bps: float  # Basis points impact for trade of 10% ADV
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core microstructure metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.rolling_window}:"
            f"{self.kyle_lambda_pct_per_million:.6f}:{self.kyle_lambda_abs_per_k_shares:.6f}:"
            f"{self.market_depth_millions:.4f}:{self.adverse_selection_regime}:"
            f"{self.sqrt_impact_1pct_adv_bps:.2f}"
        )
        return hmac.new(PRICE_IMPACT_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_kyle_lambda(
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 60,
) -> Tuple[float, float, float, float]:
    """Estimate Kyle's Lambda via robust Huber regression on signed order flow.

    Regression equations:
        1. r_t = alpha + lambda_pct * (Q_t^$ / 10^6) + epsilon_t
        2. delta_P_t = alpha + lambda_abs * (Q_t / 10^3) + epsilon_t

    Args:
        prices: Series of asset closing prices.
        volumes: Series of daily traded volumes.
        window: Regression observation window.

    Returns:
        Tuple of (lambda_pct_per_million, lambda_abs_per_k_shares, r_squared, market_depth_millions).
    """
    p_clean = prices.astype(float).values
    v_clean = volumes.astype(float).values

    if len(p_clean) < 15 or len(v_clean) < 15:
        raise ValueError("Insufficient observations to estimate Kyle's Lambda (minimum 15 required).")

    # Use tail window observations
    p_tail = p_clean[-window:]
    v_tail = v_clean[-window:]

    # Returns and price differences
    delta_p = np.diff(p_tail)
    p_prev = p_tail[:-1]
    ret = delta_p / np.maximum(p_prev, 1e-8)

    # Signed order flow proxy: sign(delta_P) * volume
    sign_flow = np.sign(delta_p)
    v_eff = v_tail[1:]
    signed_shares = (sign_flow * v_eff) / 1000.0  # in 1,000 shares
    dollar_vol = p_prev * v_eff
    signed_dollar_m = (sign_flow * dollar_vol) / 1e6  # in $1,000,000

    # Handle zero or constant order flow
    if np.all(signed_dollar_m == 0.0) or np.all(signed_shares == 0.0):
        return 0.0, 0.0, 0.0, 1e6

    # 1. Percent Lambda: r_t ~ signed_dollar_m
    x_pct = signed_dollar_m.reshape(-1, 1)
    y_pct = ret * 100.0  # in percent return

    try:
        huber_pct = HuberRegressor(max_iter=200, alpha=0.0)
        huber_pct.fit(x_pct, y_pct)
        lambda_pct = float(max(huber_pct.coef_[0], 0.0))
        # R-squared
        y_pred = huber_pct.predict(x_pct)
        ss_tot = np.sum((y_pct - np.mean(y_pct)) ** 2)
        ss_res = np.sum((y_pct - y_pred) ** 2)
        r2 = float(max(0.0, 1.0 - (ss_res / max(ss_tot, 1e-8))))
    except Exception:
        # Fallback to robust median slope
        pos_mask = np.abs(signed_dollar_m) > 1e-4
        if np.any(pos_mask):
            slopes = (y_pct[pos_mask] / signed_dollar_m[pos_mask])
            lambda_pct = float(max(np.median(slopes), 0.0))
        else:
            lambda_pct = 0.0
        r2 = 0.0

    # 2. Absolute Lambda: delta_P ~ signed_shares
    x_abs = signed_shares.reshape(-1, 1)
    try:
        huber_abs = HuberRegressor(max_iter=200, alpha=0.0)
        huber_abs.fit(x_abs, delta_p)
        lambda_abs = float(max(huber_abs.coef_[0], 0.0))
    except Exception:
        pos_mask_abs = np.abs(signed_shares) > 1e-4
        if np.any(pos_mask_abs):
            slopes_abs = delta_p[pos_mask_abs] / signed_shares[pos_mask_abs]
            lambda_abs = float(max(np.median(slopes_abs), 0.0))
        else:
            lambda_abs = 0.0

    # Market depth: $M required to shift price by 1%
    depth = float(1.0 / max(lambda_pct, 1e-6))

    return round(lambda_pct, 6), round(lambda_abs, 6), round(r2, 4), round(depth, 2)


def compute_square_root_impact(
    daily_volatility: float,
    adv_shares: float,
    trade_pct_adv: float,
    y_constant: float = 0.7,
) -> float:
    """Calculate expected price impact via Bouchaud's square-root law in basis points.

    Formula:
        I_bps = Y * sigma_daily * sqrt(trade_shares / ADV) * 10,000

    Args:
        daily_volatility: Daily price return volatility (sigma).
        adv_shares: 21-session average daily volume.
        trade_pct_adv: Trade size expressed as fraction of ADV (e.g. 0.01 for 1%).
        y_constant: Universal dimensionless scaling factor (typically ~0.7).

    Returns:
        Expected execution price impact in basis points (bps).
    """
    assert_finite_number(daily_volatility, "daily_volatility")
    assert_finite_number(adv_shares, "adv_shares")
    assert_finite_number(trade_pct_adv, "trade_pct_adv")

    fraction = max(trade_pct_adv, 0.0)
    impact_dec = y_constant * daily_volatility * math.sqrt(fraction)
    return float(round(impact_dec * 10000.0, 2))


def evaluate_price_impact_profile(
    ticker: str,
    prices: pd.Series,
    volumes: pd.Series,
    window: int = 60,
) -> PriceImpactReceipt:
    """Evaluate comprehensive Kyle's Lambda price impact and adverse selection profile.

    Args:
        ticker: Validated stock ticker symbol.
        prices: Series of closing/adjusted stock prices.
        volumes: Series of traded share volumes.
        window: Regression rolling window (sessions).

    Returns:
        PriceImpactReceipt with institutional execution analytics and HMAC signature.
    """
    valid_ticker = validate_ticker(ticker.strip().upper())
    window = max(20, min(window, 252))

    p_clean = prices.dropna().astype(float)
    v_clean = volumes.dropna().astype(float)
    common_idx = p_clean.index.intersection(v_clean.index)
    if len(common_idx) < 20:
        raise ValueError(f"Insufficient aligned price and volume observations for {valid_ticker}.")

    p_aligned = p_clean.loc[common_idx]
    v_aligned = v_clean.loc[common_idx]
    sample_size = len(p_aligned)

    # Kyle's Lambda and Depth
    lam_pct, lam_abs, r2, depth_m = estimate_kyle_lambda(p_aligned, v_aligned, window=window)

    # 21-session ADV and volatility
    tail_21_p = p_aligned.iloc[-21:]
    tail_21_v = v_aligned.iloc[-21:]
    adv = float(tail_21_v.mean())
    avg_dollar_vol = float((tail_21_p * tail_21_v).mean())

    daily_returns = tail_21_p.pct_change().dropna()
    sigma_daily = float(daily_returns.std()) if len(daily_returns) > 1 else 0.015
    if math.isnan(sigma_daily) or sigma_daily <= 1e-6:
        sigma_daily = 0.015

    # Square-root impact calculations for 1%, 5%, 10% ADV trades
    impact_1pct = compute_square_root_impact(sigma_daily, adv, 0.01)
    impact_5pct = compute_square_root_impact(sigma_daily, adv, 0.05)
    impact_10pct = compute_square_root_impact(sigma_daily, adv, 0.10)

    # Adverse Selection Regime Classification
    # Baseline benchmark: typical large-cap lambda_pct is < 0.05% per $1M
    if lam_pct < 0.02:
        regime = "LOW_ADVERSE_SELECTION_HIGH_LIQUIDITY"
    elif lam_pct <= 0.10:
        regime = "MODERATE_IMPACT_NORMAL_LIQUIDITY"
    else:
        regime = "HIGH_ADVERSE_SELECTION_FRAGILE"

    summary = (
        f"Kyle's Price Impact Profile for {valid_ticker} ({sample_size} sessions, window: {window}D): "
        f"Lambda = {lam_pct:.4f}%/$1M net order flow (Depth: ${depth_m:.1f}M/1% shift, R2: {r2:.3f}). "
        f"ADV: {adv:,.0f} shares (${avg_dollar_vol / 1e6:.1f}M). "
        f"Square-Root Impact: 1% ADV = {impact_1pct:.1f} bps | 5% ADV = {impact_5pct:.1f} bps. "
        f"Regime: {regime}."
    )

    receipt = PriceImpactReceipt(
        ticker=valid_ticker,
        sample_sessions=sample_size,
        rolling_window=window,
        kyle_lambda_pct_per_million=lam_pct,
        kyle_lambda_abs_per_k_shares=lam_abs,
        market_depth_millions=depth_m,
        average_daily_volume_shares=round(adv, 0),
        average_daily_dollar_volume=round(avg_dollar_vol, 2),
        r_squared=r2,
        adverse_selection_regime=regime,
        sqrt_impact_1pct_adv_bps=impact_1pct,
        sqrt_impact_5pct_adv_bps=impact_5pct,
        sqrt_impact_10pct_adv_bps=impact_10pct,
        summary=summary,
    )
    return receipt
