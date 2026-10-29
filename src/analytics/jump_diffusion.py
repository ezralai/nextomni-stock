"""Barndorff-Nielsen & Shephard (2004, 2006) Bipower Variation & Jump Detection Engine.

Implements continuous sample path variance vs. discrete jump variation decomposition,
realized bipower variation (BV), tri-power quarticity (TQ), and the asymptotic BNS (2006)
nonparametric jump test statistic to isolate unhedgeable price jump risks.

References:
    Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and bipower variation with
    quantum jumps and stochastic volatility. Econometrica, 72(4), 1133-1164.
    Barndorff-Nielsen, O. E., & Shephard, N. (2006). Econometrics of testing for jumps
    in financial economics using bipower variation. Journal of Financial Econometrics, 4(1), 1-30.
    Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing it up: Including
    jump components in the measurement, modeling, and forecasting of return volatility.
    The Review of Economics and Statistics, 89(4), 701-720.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import norm

from src.config import PRICE_COLUMN
from src.data.db import get_observations
from src.security.validators import assert_finite_number, validate_ticker

JUMP_SIGNING_KEY = b"nextomni_jump_diffusion_secret_key_v1"

# BNS (2004, 2006) Asymptotic Constants
# mu_1 = E[|Z|] = sqrt(2 / pi)
MU_1 = math.sqrt(2.0 / math.pi)
MU_1_INV_SQ = math.pi / 2.0  # ~1.5707963

# mu_{4/3} = 2^{2/3} * Gamma(7/6) / Gamma(1/2)
MU_4_3 = (2.0 ** (2.0 / 3.0)) * gamma(7.0 / 6.0) / math.sqrt(math.pi)
MU_4_3_INV_CUBE = float(MU_4_3 ** (-3.0))  # ~1.74737

# Asymptotic standardization constant: vartheta = (pi / 2)^2 + pi - 3
BNS_VARTHETA = (math.pi / 2.0) ** 2 + math.pi - 3.0  # ~2.609067


@dataclass
class JumpDiffusionReceipt:
    """Cryptographic audit receipt for BNS Bipower Jump Decomposition."""

    ticker: str
    sample_sessions: int
    window: int
    annualized_total_vol_pct: float
    annualized_continuous_vol_pct: float
    annualized_jump_vol_pct: float
    realized_variance: float  # RV
    bipower_variation: float  # BV
    jump_variation: float  # max(0, RV - BV)
    jump_contribution_ratio: float  # JV / RV in [0, 1]
    tripower_quarticity: float  # TQ
    bns_z_statistic: float  # Asymptotic standard normal test statistic
    p_value: float  # P(Z >= z_obs)
    has_significant_jump: bool  # True if p_value < 0.05
    jump_regime: str  # CONTINUOUS_DIFFUSION, MODERATE_JUMP, or SEVERE_DISCONTINUITY
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core jump metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.window}:"
            f"{self.annualized_total_vol_pct:.4f}:{self.annualized_continuous_vol_pct:.4f}:"
            f"{self.annualized_jump_vol_pct:.4f}:{self.jump_contribution_ratio:.4f}:"
            f"{self.bns_z_statistic:.4f}:{self.has_significant_jump}:{self.jump_regime}"
        )
        return hmac.new(JUMP_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_realized_variance(returns: np.ndarray) -> float:
    """Calculate Realized Variance (RV) over return series.

    Formula:
        RV = sum_{i=1}^M r_i^2
    """
    ret = np.asarray(returns, dtype=float)
    return float(np.sum(ret ** 2))


def compute_bipower_variation(returns: np.ndarray) -> float:
    """Calculate Barndorff-Nielsen & Shephard (2004) Realized Bipower Variation (BV).

    Asymptotically converges to continuous integrated variance, immune to jumps.

    Formula:
        BV = (pi / 2) * (M / (M - 1)) * sum_{i=2}^M |r_i| * |r_{i-1}|
    """
    ret = np.asarray(returns, dtype=float)
    m = len(ret)
    if m < 2:
        return 0.0

    abs_ret = np.abs(ret)
    adjacent_prod = abs_ret[1:] * abs_ret[:-1]
    adj_sum = float(np.sum(adjacent_prod))

    finite_sample_adj = float(m) / float(m - 1)
    bv = MU_1_INV_SQ * finite_sample_adj * adj_sum
    return max(0.0, bv)


def compute_tripower_quarticity(returns: np.ndarray) -> float:
    """Calculate Realized Tri-power Quarticity (TQ) for asymptotic variance scaling.

    Formula:
        TQ = M * mu_{4/3}^{-3} * (M / (M - 2)) * sum_{i=3}^M |r_i|^{4/3} * |r_{i-1}|^{4/3} * |r_{i-2}|^{4/3}
    """
    ret = np.asarray(returns, dtype=float)
    m = len(ret)
    if m < 3:
        return 0.0

    abs_ret = np.abs(ret) ** (4.0 / 3.0)
    tripower_prod = abs_ret[2:] * abs_ret[1:-1] * abs_ret[:-2]
    tripower_sum = float(np.sum(tripower_prod))

    finite_sample_adj = float(m) / float(m - 2)
    tq = float(m) * MU_4_3_INV_CUBE * finite_sample_adj * tripower_sum
    return max(0.0, tq)


def compute_bns_jump_test_statistic(
    rv: float,
    bv: float,
    tq: float,
    sample_size: int,
) -> Tuple[float, float, bool]:
    """Compute Barndorff-Nielsen & Shephard (2006) Ratio Jump Test Z-statistic.

    Formula:
        Z = ((RV - BV) / RV) / sqrt(vartheta * (1 / M) * max(1, TQ / BV^2))

    Returns:
        Tuple of (z_statistic, p_value, has_significant_jump).
    """
    if rv <= 1e-12 or bv <= 1e-12 or sample_size < 3:
        return 0.0, 1.0, False

    relative_diff = (rv - bv) / rv
    ratio_quarticity = max(1.0, tq / (bv ** 2))
    var_scaling = BNS_VARTHETA * (1.0 / float(sample_size)) * ratio_quarticity
    denom = math.sqrt(max(1e-12, var_scaling))

    z_stat = relative_diff / denom
    # One-sided p-value: P(Z >= z_stat)
    p_val = float(1.0 - norm.cdf(z_stat))
    p_val = float(np.clip(p_val, 0.0, 1.0))

    has_jump = bool(p_val < 0.05 and z_stat > 1.64485)
    return float(z_stat), p_val, has_jump


def evaluate_jump_diffusion_profile(
    ticker: str,
    prices: Union[pd.Series, np.ndarray, List[float]],
    window: int = 21,
    db_path: Optional[Path] = None,
) -> JumpDiffusionReceipt:
    """Evaluate Barndorff-Nielsen & Shephard (2004, 2006) Jump Diffusion Profile.

    Args:
        ticker: Validated stock ticker symbol.
        prices: Series or array of asset closing prices.
        window: Lookback observation window in sessions (default: 21).
        db_path: Optional SQLite database path.

    Returns:
        JumpDiffusionReceipt with continuous vs jump decomposition and test statistics.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(window, "window")
    if window < 10:
        raise ValueError(f"Observation window must be at least 10 sessions, got {window}")

    if isinstance(prices, pd.Series):
        s_prices = prices.dropna().astype(float)
    else:
        s_prices = pd.Series(prices).dropna().astype(float)

    if len(s_prices) < window + 1:
        raise ValueError(
            f"Insufficient price history for {clean_ticker}: got {len(s_prices)} prices, need at least {window + 1}"
        )

    # Compute daily log-returns
    log_returns = np.diff(np.log(s_prices.values))
    sample_sessions = len(log_returns)

    # Window sample returns
    window_returns = log_returns[-window:]
    m = len(window_returns)

    rv = compute_realized_variance(window_returns)
    bv = compute_bipower_variation(window_returns)
    tq = compute_tripower_quarticity(window_returns)

    # Continuous and Jump Variation
    cv = min(rv, bv)
    jv = max(0.0, rv - bv)

    # Jump Contribution Ratio J = JV / RV
    jump_ratio = (jv / rv) if rv > 1e-12 else 0.0
    jump_ratio = float(np.clip(jump_ratio, 0.0, 1.0))

    # Annualization factor
    ann_factor = 252.0 / float(m)
    ann_total_vol = math.sqrt(max(0.0, rv * ann_factor)) * 100.0
    ann_cont_vol = math.sqrt(max(0.0, cv * ann_factor)) * 100.0
    ann_jump_vol = math.sqrt(max(0.0, jv * ann_factor)) * 100.0

    # BNS Asymptotic Jump Test
    z_stat, p_val, has_jump = compute_bns_jump_test_statistic(
        rv=rv, bv=bv, tq=tq, sample_size=m
    )

    # Classification Regime
    if not has_jump or jump_ratio < 0.10:
        jump_regime = "CONTINUOUS_DIFFUSION"
    elif jump_ratio < 0.35:
        jump_regime = "MODERATE_JUMP"
    else:
        jump_regime = "SEVERE_DISCONTINUITY"

    summary = (
        f"BNS Jump Diffusion Analysis for {clean_ticker} ({sample_sessions} sessions, {window}D window): "
        f"Total Vol: {ann_total_vol:.1f}% Ann (Cont: {ann_cont_vol:.1f}%, Jump: {ann_jump_vol:.1f}%). "
        f"Jump Ratio: {jump_ratio * 100:.1f}%. BNS Z-Score: {z_stat:.2f} (p={p_val:.4f}). "
        f"Regime: {jump_regime} ({'Significant Discontinuity Detected' if has_jump else 'Pure Continuous Path'})."
    )

    receipt = JumpDiffusionReceipt(
        ticker=clean_ticker,
        sample_sessions=sample_sessions,
        window=window,
        annualized_total_vol_pct=ann_total_vol,
        annualized_continuous_vol_pct=ann_cont_vol,
        annualized_jump_vol_pct=ann_jump_vol,
        realized_variance=rv,
        bipower_variation=bv,
        jump_variation=jv,
        jump_contribution_ratio=jump_ratio,
        tripower_quarticity=tq,
        bns_z_statistic=z_stat,
        p_value=p_val,
        has_significant_jump=has_jump,
        jump_regime=jump_regime,
        summary=summary,
    )

    return receipt
