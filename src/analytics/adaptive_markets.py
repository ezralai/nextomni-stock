"""Andrew W. Lo (2004, 2005, 2017)
Adaptive Markets Hypothesis (AMH), Evolutionary Market Efficiency & Punctuated Equilibrium Engine.

Implements the Adaptive Markets Hypothesis framework:
- Lo (2004, Journal of Portfolio Management, "The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective")
- Lo (2005, Financial Analysts Journal, "Reconciling Efficient Markets with Behavioral Finance: The Adaptive Markets Hypothesis")
- Lo (2017, Princeton University Press, "Adaptive Markets: Financial Evolution at the Speed of Thought"):
    Financial market efficiency is not a static binary state (efficient vs. inefficient), but an evolutionary,
    time-varying equilibrium governed by the principles of competition, mutation, reproduction, and natural selection.
    - Market Ecology: Heterogeneous participant species (fundamental value investors, quantitative hedge funds,
      noise traders, market makers) compete for scarce financial resources.
    - Punctuated Equilibrium: Periods of calm, high efficiency are periodically punctuated by environmental shocks,
      regime shifts, and crisis dislocations where behavioral heuristics (fear, panic, greed) dominate rationality.
    - Dynamic Arbitrage Carrying Capacity (K_arbitrage): Arbitrage opportunities wax and wane in cycles as hedge funds
      flock to profitable strategies until saturation occurs, after which capital withdrawal regenerates alpha.

References:
    Lo, A. W. (2004). The adaptive markets hypothesis: Market efficiency from an evolutionary perspective.
    The Journal of Portfolio Management, 30(5), 15-29.
    Lo, A. W. (2005). Reconciling efficient markets with behavioral finance: the adaptive markets hypothesis.
    Financial Analysts Journal, 61(6), 21-44.
    Lo, A. W. (2017). Adaptive markets: Financial evolution at the speed of thought. Princeton University Press.
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

AMH_SIGNING_KEY = b"nextomni_adaptive_markets_hypothesis_v1"


@dataclass
class AdaptiveMarketsReceipt:
    """Cryptographic audit receipt for Andrew W. Lo (2004, 2005) Adaptive Markets Hypothesis Profile."""

    ticker: str
    latest_price: float
    # Evolutionary & Microstructure Inputs
    lo_mackinlay_variance_ratio: float  # VR(k) variance ratio in [0.30, 2.50]
    serial_autocorrelation_rho1: float  # First-order autocorrelation rho_1 in [-0.80, 0.80]
    ecological_stress_drawdown: float  # Maximum drawdown stress in [0.0, 0.85]
    realized_volatility: float  # Annualized volatility in [0.05, 1.50]
    baseline_volatility: float  # Normal-times baseline volatility in [0.05, 0.80]
    # Derived AMH Metrics
    adaptive_market_efficiency_index: float  # E_AMH in [0.05, 1.00]
    arbitrage_carrying_capacity: float  # K_arbitrage in [0.10, 4.00]
    evolutionary_adaptation_pressure: float  # Pi_evol in [-2.0, 3.5]
    amh_arbitrage_impulse: float  # I_amh in [-2.5, 2.5]
    expected_adaptive_alpha_pct: float  # Projected annualized alpha net of ecological competition (%)
    # Flags & Regime
    is_competitive_saturation: bool  # True if E_AMH >= 0.85 (high efficiency, compressed alpha)
    is_punctuated_equilibrium: bool  # True if E_AMH <= 0.40 (severe ecology dislocation, wide alpha)
    is_adaptive_recovery: bool  # True if 0.40 < E_AMH < 0.70
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
        """Compute deterministic HMAC-SHA256 signature across all AMH metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:"
            f"{self.lo_mackinlay_variance_ratio:.4f}:{self.serial_autocorrelation_rho1:.4f}:"
            f"{self.ecological_stress_drawdown:.4f}:{self.realized_volatility:.4f}:{self.baseline_volatility:.4f}:"
            f"{self.adaptive_market_efficiency_index:.4f}:{self.arbitrage_carrying_capacity:.4f}:"
            f"{self.evolutionary_adaptation_pressure:.4f}:{self.amh_arbitrage_impulse:.4f}:"
            f"{self.expected_adaptive_alpha_pct:.2f}:"
            f"{int(self.is_competitive_saturation)}:{int(self.is_punctuated_equilibrium)}:{int(self.is_adaptive_recovery)}:{self.regime}"
        )
        return hmac.new(
            AMH_SIGNING_KEY,
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


def classify_amh_regime(
    adaptive_efficiency: float,
    arbitrage_capacity: float,
    adaptation_pressure: float,
) -> str:
    """Classify asset into Andrew W. Lo (2004, 2005) Adaptive Markets Hypothesis Regimes."""
    if adaptive_efficiency >= 0.85:
        return "HIGH_EFFICIENCY_COMPETITIVE_SATURATION"
    if adaptive_efficiency <= 0.40 or adaptation_pressure >= 1.50:
        return "BEHAVIORAL_PUNCTUATED_EQUILIBRIUM"
    if adaptive_efficiency < 0.70:
        return "ADAPTIVE_ARBITRAGE_RECOVERY"
    return "CYCLICAL_ECOLOGICAL_EQUILIBRIUM"


def calculate_adaptive_markets(
    ticker: str,
    latest_price: float,
    lo_mackinlay_variance_ratio: float = 1.05,
    serial_autocorrelation_rho1: float = 0.04,
    ecological_stress_drawdown: float = 0.08,
    realized_volatility: float = 0.22,
    baseline_volatility: float = 0.20,
) -> AdaptiveMarketsReceipt:
    """Calculate Andrew W. Lo (2004, 2005) Adaptive Markets Hypothesis & Evolutionary Efficiency Metrics.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        lo_mackinlay_variance_ratio: Lo-MacKinlay VR(k) in [0.30, 2.50].
        serial_autocorrelation_rho1: First-order return autocorrelation in [-0.80, 0.80].
        ecological_stress_drawdown: Recent peak drawdown in [0.0, 0.85].
        realized_volatility: Annualized return volatility in [0.05, 1.50].
        baseline_volatility: Normal-times baseline volatility in [0.05, 0.80].

    Returns:
        AdaptiveMarketsReceipt with adaptive efficiency index, carrying capacity, impulse, and regime.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(lo_mackinlay_variance_ratio, "lo_mackinlay_variance_ratio")
    assert_finite_number(serial_autocorrelation_rho1, "serial_autocorrelation_rho1")
    assert_finite_number(ecological_stress_drawdown, "ecological_stress_drawdown")
    assert_finite_number(realized_volatility, "realized_volatility")
    assert_finite_number(baseline_volatility, "baseline_volatility")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (0.30 <= lo_mackinlay_variance_ratio <= 2.50):
        raise ValueError("lo_mackinlay_variance_ratio must be in [0.30, 2.50].")
    if not (-0.80 <= serial_autocorrelation_rho1 <= 0.80):
        raise ValueError("serial_autocorrelation_rho1 must be in [-0.80, 0.80].")
    if not (0.0 <= ecological_stress_drawdown <= 0.85):
        raise ValueError("ecological_stress_drawdown must be in [0.0, 0.85].")
    if not (0.05 <= realized_volatility <= 1.50):
        raise ValueError("realized_volatility must be in [0.05, 1.50].")
    if not (0.05 <= baseline_volatility <= 0.80):
        raise ValueError("baseline_volatility must be in [0.05, 0.80].")

    p_t = float(latest_price)
    vr = float(lo_mackinlay_variance_ratio)
    rho = float(serial_autocorrelation_rho1)
    dd = float(ecological_stress_drawdown)
    vol = float(realized_volatility)
    vol_0 = float(baseline_volatility)

    # 1. Adaptive Market Efficiency Index E_AMH
    # Measures departure from pure random walk (VR = 1.0, rho = 0.0)
    departure = 0.70 * abs(vr - 1.0) + 0.30 * abs(rho)
    raw_eff = 1.0 - departure
    e_amh = float(np.clip(raw_eff, 0.05, 1.00))

    # 2. Arbitrage Carrying Capacity K_arbitrage
    # When efficiency is low, carrying capacity is high (fertile alpha environment);
    # high ecological drawdown temporarily reduces available risk capital
    raw_capacity = (1.0 / max(e_amh, 0.20)) * max(0.20, 1.0 - 0.75 * dd)
    k_arb = float(np.clip(raw_capacity, 0.10, 4.00))

    # 3. Evolutionary Adaptation Pressure Pi_evol
    # Elevated volatility shocks and low efficiency create high evolutionary selection pressure
    vol_excess = max(-0.5, (vol - vol_0) / max(vol_0, 0.05))
    raw_pressure = 1.5 * (1.0 - e_amh) + 0.5 * vol_excess
    pi_evol = float(np.clip(raw_pressure, -2.0, 3.5))

    # 4. AMH Arbitrage Impulse & Expected Alpha
    # High capacity and moderate efficiency allow quantitative strategies to extract abnormal returns
    raw_impulse = 0.55 * (1.0 - e_amh) * min(2.0, k_arb) - 0.35 * max(0.0, e_amh - 0.80)
    impulse = float(np.clip(raw_impulse, -2.5, 2.5))
    expected_alpha = float(np.clip(4.2 * impulse, -6.0, 14.0))

    regime = classify_amh_regime(e_amh, k_arb, pi_evol)
    is_sat = bool(regime == "HIGH_EFFICIENCY_COMPETITIVE_SATURATION")
    is_punc = bool(regime == "BEHAVIORAL_PUNCTUATED_EQUILIBRIUM")
    is_rec = bool(regime == "ADAPTIVE_ARBITRAGE_RECOVERY")

    if regime == "HIGH_EFFICIENCY_COMPETITIVE_SATURATION":
        summary = (
            f"High-Efficiency Competitive Saturation (E_AMH={e_amh:.2f}, VR={vr:.2f}, K_arb={k_arb:.2f}x). "
            f"Fierce quantitative competition has saturated arbitrage channels; returns conform closely "
            f"to random walk with compressed alpha potential ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "BEHAVIORAL_PUNCTUATED_EQUILIBRIUM":
        summary = (
            f"Behavioral Punctuated Equilibrium (E_AMH={e_amh:.2f}, Pressure={pi_evol:+.2f}, DD={dd:.1%}). "
            f"Environmental disruption triggers behavioral panic and heuristic pricing, creating wide "
            f"adaptive arbitrage opportunities ({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "ADAPTIVE_ARBITRAGE_RECOVERY":
        summary = (
            f"Adaptive Arbitrage Recovery (E_AMH={e_amh:.2f}, K_arb={k_arb:.2f}x, VR={vr:.2f}). "
            f"Evolving market ecology where capital re-entry generates favorable cyclical risk-adjusted returns ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Cyclical Ecological Equilibrium (E_AMH={e_amh:.2f}, VR={vr:.2f}, rho={rho:+.2f}). "
            f"Market ecology is balanced with stable evolutionary turnover and moderate predictability ({expected_alpha:+.1f}%/yr)."
        )

    return AdaptiveMarketsReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        lo_mackinlay_variance_ratio=round(vr, 4),
        serial_autocorrelation_rho1=round(rho, 4),
        ecological_stress_drawdown=round(dd, 4),
        realized_volatility=round(vol, 4),
        baseline_volatility=round(vol_0, 4),
        adaptive_market_efficiency_index=round(e_amh, 4),
        arbitrage_carrying_capacity=round(k_arb, 2),
        evolutionary_adaptation_pressure=round(pi_evol, 4),
        amh_arbitrage_impulse=round(impulse, 4),
        expected_adaptive_alpha_pct=round(expected_alpha, 2),
        is_competitive_saturation=is_sat,
        is_punctuated_equilibrium=is_punc,
        is_adaptive_recovery=is_rec,
        regime=regime,
        summary=summary,
    )


def evaluate_amh_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    lo_mackinlay_variance_ratio: Optional[float] = None,
    serial_autocorrelation_rho1: Optional[float] = None,
    ecological_stress_drawdown: Optional[float] = None,
    realized_volatility: Optional[float] = None,
    baseline_volatility: Optional[float] = None,
) -> AdaptiveMarketsReceipt:
    """Evaluate Andrew W. Lo Adaptive Markets profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_vr = 1.05
    inferred_rho = 0.04
    inferred_dd = 0.08
    inferred_vol = 0.22
    inferred_vol0 = 0.20

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

                # First-order serial autocorrelation
                if len(rets) > 5:
                    r_centered = rets - np.mean(rets)
                    var_0 = np.dot(r_centered, r_centered)
                    cov_1 = np.dot(r_centered[:-1], r_centered[1:])
                    inferred_rho = float(np.clip(cov_1 / max(var_0, 1e-6), -0.75, 0.75))

                # 5-session Lo-MacKinlay Variance Ratio
                k = 5
                if len(rets) >= k * 3:
                    # 5-period overlapping log returns
                    log_closes = np.log(arr)
                    r_1 = np.diff(log_closes)
                    var_1 = float(np.var(r_1, ddof=1))
                    r_k = log_closes[k:] - log_closes[:-k]
                    var_k = float(np.var(r_k, ddof=1))
                    if var_1 > 1e-8:
                        inferred_vr = float(np.clip(var_k / (k * var_1), 0.40, 2.20))

                # Peak drawdown
                peak = float(np.max(arr))
                inferred_dd = float(np.clip((peak - arr[-1]) / max(peak, 1e-4), 0.0, 0.85))

                inferred_vol = float(np.clip(np.std(rets, ddof=1) * math.sqrt(252.0), 0.05, 1.50)) if len(rets) > 1 else 0.22
    except Exception:
        latest_price = 100.0

    vr_val = float(lo_mackinlay_variance_ratio) if lo_mackinlay_variance_ratio is not None else inferred_vr
    rho_val = float(serial_autocorrelation_rho1) if serial_autocorrelation_rho1 is not None else inferred_rho
    dd_val = float(ecological_stress_drawdown) if ecological_stress_drawdown is not None else inferred_dd
    vol_val = float(realized_volatility) if realized_volatility is not None else inferred_vol
    vol0_val = float(baseline_volatility) if baseline_volatility is not None else inferred_vol0

    return calculate_adaptive_markets(
        ticker=clean_ticker,
        latest_price=latest_price,
        lo_mackinlay_variance_ratio=vr_val,
        serial_autocorrelation_rho1=rho_val,
        ecological_stress_drawdown=dd_val,
        realized_volatility=vol_val,
        baseline_volatility=vol0_val,
    )
