"""Ravi Bansal & Amir Yaron (2004) / Bansal, Dittmar & Lundblad (2005)
Long-Run Risks (LRR), Persistent Cash-Flow Growth & Stochastic Economic Uncertainty Pricing Engine.

Implements the Epstein-Zin-Weil recursive utility Long-Run Risks asset pricing framework:
- Bansal & Yaron (2004, Journal of Finance) & Bansal, Dittmar & Lundblad (2005, Journal of Finance):
    Aggregate consumption and dividend growth contain:
    1. A small but highly persistent predictable long-run growth component x_t with persistence
       rho_x in [0.50, 0.995].
    2. Time-varying macroeconomic uncertainty (stochastic consumption volatility sigma_t^2).
    When the representative investor has Epstein-Zin recursive preferences with relative risk
    aversion gamma_RRA > 1 and intertemporal elasticity of substitution psi_IES > 1, the investor
    prefers early resolution of uncertainty (Theta_EZ = gamma_RRA - 1/psi_IES > 0):
        A_x = (1 - 1 / psi_IES) / (1 - kappa_1 * rho_x)
    Consequently, two distinct macroeconomic channels price cross-sectional expected returns:
    - Persistent Growth Risk Channel (lambda_x * beta_{x,i}): Assets whose cash flows covary
      strongly with persistent long-run growth shocks (high beta_{x,i}) command a substantial
      positive risk premium.
    - Stochastic Economic Uncertainty Channel (lambda_sigma * (-beta_{sigma,i})): Because marginal
      utility rises when macroeconomic uncertainty spikes (for psi_IES > 1, gamma_RRA > 1), assets
      that covary negatively with uncertainty shocks (beta_{sigma,i} < 0) are risky and command a
      positive uncertainty risk premium, whereas assets that pay off during volatility spikes
      (beta_{sigma,i} > 0) serve as macroeconomic safe-haven hedges.

References:
    Bansal, R., & Yaron, A. (2004). Risks for the long run: A potential resolution of asset
    pricing puzzles. The Journal of Finance, 59(4), 1481-1509.
    Bansal, R., Dittmar, R. F., & Lundblad, C. T. (2005). Consumption, dividends, and the
    cross section of equity returns. The Journal of Finance, 60(4), 1639-1672.
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

LONG_RUN_RISKS_SIGNING_KEY = b"nextomni_bansal_yaron_long_run_risks_v1"


@dataclass
class LongRunRisksReceipt:
    """Cryptographic audit receipt for Bansal & Yaron (2004) Long-Run Risks (LRR) Profile."""

    ticker: str
    latest_price: float
    # Cash-Flow Betas & Epstein-Zin Preference Inputs
    cashflow_growth_beta: float  # Persistent growth beta beta_{x,i} in [-2.0, 4.0]
    uncertainty_shock_beta: float  # Macroeconomic volatility beta beta_{sigma,i} in [-3.0, 3.0]
    growth_persistence_rho: float  # Long-run growth persistence rho_x in [0.50, 0.995]
    intertemporal_elasticity_ies: float  # Epstein-Zin IES psi_IES in [0.50, 3.0]
    relative_risk_aversion: float  # Epstein-Zin RRA gamma_RRA in [1.5, 20.0]
    # Derived Bansal-Yaron LRR Metrics
    early_resolution_preference_theta: float  # Theta_EZ = gamma_RRA - 1/psi_IES
    wealth_growth_sensitivity_ax: float  # A_x = (1 - 1/psi_IES) / (1 - kappa_1 * rho_x)
    persistent_growth_premium_pct: float  # Annualized premium from beta_{x,i} (%)
    uncertainty_volatility_premium_pct: float  # Annualized premium from -beta_{sigma,i} (%)
    bansal_yaron_lrr_impulse: float  # Composite LRR pricing impulse LRRI_t in [-2.5, 2.5]
    expected_lrr_alpha_pct: float  # Total projected LRR risk premium / alpha (% / year)
    # Flags & Regime
    is_high_lrr_growth_premium_leader: bool  # True if compensated persistent growth leader
    is_macro_uncertainty_safe_haven: bool  # True if positive uncertainty beta hedge
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
        """Compute deterministic HMAC-SHA256 signature across all Bansal-Yaron LRR metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:"
            f"{self.cashflow_growth_beta:.4f}:{self.uncertainty_shock_beta:.4f}:"
            f"{self.growth_persistence_rho:.4f}:{self.intertemporal_elasticity_ies:.4f}:"
            f"{self.relative_risk_aversion:.4f}:{self.early_resolution_preference_theta:.4f}:"
            f"{self.wealth_growth_sensitivity_ax:.4f}:{self.persistent_growth_premium_pct:.4f}:"
            f"{self.uncertainty_volatility_premium_pct:.4f}:{self.bansal_yaron_lrr_impulse:.4f}:"
            f"{self.expected_lrr_alpha_pct:.2f}:{int(self.is_high_lrr_growth_premium_leader)}:"
            f"{int(self.is_macro_uncertainty_safe_haven)}:{self.regime}"
        )
        return hmac.new(
            LONG_RUN_RISKS_SIGNING_KEY,
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


def classify_lrr_regime(
    cashflow_growth_beta: float,
    uncertainty_shock_beta: float,
    lrr_impulse: float,
) -> str:
    """Classify asset into Bansal & Yaron (2004) Long-Run Risks Regimes."""
    if cashflow_growth_beta >= 1.25 and uncertainty_shock_beta <= -0.25 and lrr_impulse >= 0.45:
        return "HIGH_LRR_CASHFLOW_GROWTH_PREMIUM"
    if uncertainty_shock_beta >= 0.35:
        return "MACRO_UNCERTAINTY_SAFE_HAVEN_HEDGE"
    if cashflow_growth_beta < 0.25 and uncertainty_shock_beta <= -0.60:
        return "UNCOMPENSATED_FRAGILE_GROWTH_TRAP"
    return "BALANCED_EPSTEIN_ZIN_EQUILIBRIUM"


def calculate_long_run_risks(
    ticker: str,
    latest_price: float,
    cashflow_growth_beta: float = 1.45,
    uncertainty_shock_beta: float = -0.45,
    growth_persistence_rho: float = 0.979,
    intertemporal_elasticity_ies: float = 1.50,
    relative_risk_aversion: float = 10.0,
) -> LongRunRisksReceipt:
    """Calculate Bansal & Yaron (2004) Long-Run Risks (LRR) Cash-Flow & Uncertainty Pricing.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price P_t ($ > 0).
        cashflow_growth_beta: Exposure to persistent cash-flow growth beta_{x,i} in [-2.0, 4.0].
        uncertainty_shock_beta: Exposure to macro volatility shocks beta_{sigma,i} in [-3.0, 3.0].
        growth_persistence_rho: Persistence of long-run growth rho_x in [0.50, 0.995].
        intertemporal_elasticity_ies: Epstein-Zin IES psi_IES in [0.50, 3.0].
        relative_risk_aversion: Epstein-Zin RRA gamma_RRA in [1.5, 20.0].

    Returns:
        LongRunRisksReceipt with Epstein-Zin multipliers, growth/uncertainty premia, and regime.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(cashflow_growth_beta, "cashflow_growth_beta")
    assert_finite_number(uncertainty_shock_beta, "uncertainty_shock_beta")
    assert_finite_number(growth_persistence_rho, "growth_persistence_rho")
    assert_finite_number(intertemporal_elasticity_ies, "intertemporal_elasticity_ies")
    assert_finite_number(relative_risk_aversion, "relative_risk_aversion")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (-2.0 <= cashflow_growth_beta <= 4.0):
        raise ValueError("cashflow_growth_beta must be in [-2.0, 4.0].")
    if not (-3.0 <= uncertainty_shock_beta <= 3.0):
        raise ValueError("uncertainty_shock_beta must be in [-3.0, 3.0].")
    if not (0.50 <= growth_persistence_rho <= 0.995):
        raise ValueError("growth_persistence_rho must be in [0.50, 0.995].")
    if not (0.50 <= intertemporal_elasticity_ies <= 3.0):
        raise ValueError("intertemporal_elasticity_ies must be in [0.50, 3.0].")
    if not (1.5 <= relative_risk_aversion <= 20.0):
        raise ValueError("relative_risk_aversion must be in [1.5, 20.0].")

    p_t = float(latest_price)
    beta_x = float(cashflow_growth_beta)
    beta_sig = float(uncertainty_shock_beta)
    rho_x = float(growth_persistence_rho)
    psi_ies = float(intertemporal_elasticity_ies)
    gamma_rra = float(relative_risk_aversion)

    # 1. Epstein-Zin Preference for Early Resolution of Uncertainty Theta_EZ
    theta_ez = float(gamma_rra - (1.0 / psi_ies))

    # 2. Bansal-Yaron Wealth-Consumption Sensitivity to Persistent Growth A_x
    # Campbell-Shiller linearization constant kappa_1 = 0.997 (monthly calibrated)
    kappa_1 = 0.997
    denom = max(1.0 - kappa_1 * rho_x, 0.005)
    a_x = float((1.0 - (1.0 / psi_ies)) / denom)

    # 3. Persistent Long-Run Growth Risk Premium (%)
    # Scales with gamma_RRA, persistence amplification, and asset cashflow_growth_beta
    persistence_scale = min(max(a_x / 12.0, -1.5), 3.0)
    lambda_x = 0.32 * gamma_rra * max(persistence_scale, 0.25)
    growth_premium = float(np.clip(lambda_x * beta_x, -8.0, 10.0))

    # 4. Stochastic Economic Uncertainty Risk Premium (%)
    # When Theta_EZ > 0 and psi_IES > 1, negative uncertainty beta commands positive risk premium
    early_res_scale = np.clip(theta_ez / 9.33, -1.0, 2.2)
    lambda_sigma = 2.10 * early_res_scale
    uncertainty_premium = float(np.clip(lambda_sigma * (-beta_sig), -6.0, 6.0))

    # 5. Composite Bansal-Yaron Long-Run Risks Impulse LRRI_t in [-2.5, 2.5]
    fragile_penalty = 1.65 if (beta_x < 0.25 and beta_sig <= -0.60) else 0.0
    raw_impulse = 0.22 * (growth_premium + uncertainty_premium) - fragile_penalty
    lrri = float(np.clip(raw_impulse, -2.5, 2.5))

    # 6. Regime Classification & Expected Alpha
    regime = classify_lrr_regime(beta_x, beta_sig, lrri)
    is_leader = bool(regime == "HIGH_LRR_CASHFLOW_GROWTH_PREMIUM")
    is_safe_haven = bool(regime == "MACRO_UNCERTAINTY_SAFE_HAVEN_HEDGE")

    expected_alpha = float(
        np.clip(growth_premium + uncertainty_premium - 3.5 * fragile_penalty, -12.0, 12.5)
    )

    if regime == "HIGH_LRR_CASHFLOW_GROWTH_PREMIUM":
        summary = (
            f"High Long-Run Risks Cash-Flow Growth Leader (beta_x={beta_x:.2f}, beta_sigma={beta_sig:+.2f}, "
            f"A_x={a_x:.1f}). High sensitivity to persistent growth under Epstein-Zin early resolution "
            f"preferences commands {expected_alpha:+.2f}%/yr risk premium."
        )
    elif regime == "MACRO_UNCERTAINTY_SAFE_HAVEN_HEDGE":
        summary = (
            f"Macroeconomic Uncertainty Safe-Haven Hedge (beta_sigma={beta_sig:+.2f}, beta_x={beta_x:.2f}). "
            f"Asset pays off during macroeconomic volatility spikes, providing defensive Epstein-Zin "
            f"insurance ({expected_alpha:+.2f}%/yr equilibrium premium)."
        )
    elif regime == "UNCOMPENSATED_FRAGILE_GROWTH_TRAP":
        summary = (
            f"Uncompensated Fragile Growth Trap (beta_x={beta_x:.2f}, beta_sigma={beta_sig:+.2f}). "
            f"Weak persistent cash-flow growth combined with severe macroeconomic volatility drawdown "
            f"exposure ({expected_alpha:+.2f}%/yr)."
        )
    else:
        summary = (
            f"Balanced Epstein-Zin Equilibrium (beta_x={beta_x:.2f}, beta_sigma={beta_sig:+.2f}, "
            f"Theta_EZ={theta_ez:.2f}). Moderate persistent growth and uncertainty exposures yield "
            f"{expected_alpha:+.2f}%/yr expected LRR premium."
        )

    return LongRunRisksReceipt(
        ticker=clean_ticker,
        latest_price=round(p_t, 2),
        cashflow_growth_beta=round(beta_x, 4),
        uncertainty_shock_beta=round(beta_sig, 4),
        growth_persistence_rho=round(rho_x, 4),
        intertemporal_elasticity_ies=round(psi_ies, 4),
        relative_risk_aversion=round(gamma_rra, 4),
        early_resolution_preference_theta=round(theta_ez, 4),
        wealth_growth_sensitivity_ax=round(a_x, 4),
        persistent_growth_premium_pct=round(growth_premium, 4),
        uncertainty_volatility_premium_pct=round(uncertainty_premium, 4),
        bansal_yaron_lrr_impulse=round(lrri, 4),
        expected_lrr_alpha_pct=round(expected_alpha, 2),
        is_high_lrr_growth_premium_leader=is_leader,
        is_macro_uncertainty_safe_haven=is_safe_haven,
        regime=regime,
        summary=summary,
    )


def evaluate_long_run_risks_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    cashflow_growth_beta: Optional[float] = None,
    uncertainty_shock_beta: Optional[float] = None,
    growth_persistence_rho: Optional[float] = None,
    intertemporal_elasticity_ies: Optional[float] = None,
    relative_risk_aversion: Optional[float] = None,
) -> LongRunRisksReceipt:
    """Evaluate Bansal & Yaron (2004) Long-Run Risks profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_beta_x = 1.45
    inferred_beta_sig = -0.45
    inferred_rho = 0.979
    inferred_ies = 1.50
    inferred_rra = 10.0

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

                # Infer persistent cash-flow growth beta from medium-horizon trend vs volatility
                inferred_beta_x = float(np.clip(1.0 + 1.8 * ann_ret + 0.6 * ann_vol, -1.5, 3.5))
                # Infer uncertainty shock beta from return skewness/downside sensitivity to vol spikes
                sq_rets = rets ** 2
                if len(rets) > 5 and float(np.std(sq_rets)) > 1e-10:
                    corr = float(np.corrcoef(rets, sq_rets)[0, 1])
                    if math.isfinite(corr):
                        inferred_beta_sig = float(np.clip(1.2 * corr - 0.25, -2.5, 2.5))
    except Exception:
        latest_price = 100.0

    bx_val = (
        float(cashflow_growth_beta)
        if cashflow_growth_beta is not None
        else inferred_beta_x
    )
    bsig_val = (
        float(uncertainty_shock_beta)
        if uncertainty_shock_beta is not None
        else inferred_beta_sig
    )
    rho_val = (
        float(growth_persistence_rho)
        if growth_persistence_rho is not None
        else inferred_rho
    )
    ies_val = (
        float(intertemporal_elasticity_ies)
        if intertemporal_elasticity_ies is not None
        else inferred_ies
    )
    rra_val = (
        float(relative_risk_aversion)
        if relative_risk_aversion is not None
        else inferred_rra
    )

    return calculate_long_run_risks(
        ticker=clean_ticker,
        latest_price=latest_price,
        cashflow_growth_beta=bx_val,
        uncertainty_shock_beta=bsig_val,
        growth_persistence_rho=rho_val,
        intertemporal_elasticity_ies=ies_val,
        relative_risk_aversion=rra_val,
    )

