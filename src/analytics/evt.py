"""Extreme Value Theory (EVT) Peaks-Over-Threshold (POT) Tail Risk Engine.

Implements the Pickands-Balkema-de Haan (1974, 1975) Generalized Pareto Distribution (GPD)
asymptotic tail framework for accurate modeling of catastrophic financial tail drawdowns,
semi-parametric Value-at-Risk (VaR), and Expected Shortfall (CVaR).

References:
    Pickands, J. (1975). Statistical inference using extreme order statistics.
    Annals of Statistics, 3(1), 119-131.
    McNeil, A. J., Frey, R., & Embrechts, P. (2015). Quantitative Risk Management:
    Concepts, Techniques and Tools. Princeton University Press, Chapter 5: "Extreme Value Theory".
"""

import hashlib
import hmac
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

EVT_SIGNING_KEY = b"nextomni_evt_secret_audit_key_v1"


@dataclass
class EVTTailReceipt:
    """Cryptographic audit receipt for Extreme Value Theory tail risk profile."""

    ticker: str
    sample_sessions: int
    threshold_u: float
    threshold_quantile: float
    exceedances_count: int
    gpd_shape_xi: float
    gpd_scale_beta: float
    tail_fatness_regime: str
    var_95_pct: float
    cvar_95_pct: float
    var_99_pct: float
    cvar_99_pct: float
    gaussian_var_99_pct: float
    cornish_fisher_var_99_pct: float
    tail_risk_underestimation_pct: float
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core tail risk metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.threshold_u:.6f}:{self.threshold_quantile:.4f}:"
            f"{self.exceedances_count}:{self.gpd_shape_xi:.4f}:{self.gpd_scale_beta:.6f}:"
            f"{self.var_99_pct:.4f}:{self.cvar_99_pct:.4f}:{self.tail_fatness_regime}"
        )
        return hmac.new(EVT_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt against unauthorized record modification."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def fit_generalized_pareto(
    excesses: np.ndarray,
) -> Tuple[float, float]:
    """Fit Generalized Pareto Distribution (GPD) parameters (xi, beta) to excesses y = X - u.

    Log-Likelihood for exceedances y_1, ..., y_m:
        ln L(xi, beta) = -m * ln(beta) - (1 + 1/xi) * sum_{i=1}^m ln(1 + xi * y_i / beta)

    Args:
        excesses: Positive loss excess values y_i = L_i - u > 0.

    Returns:
        Tuple of (xi_shape, beta_scale) where xi > 0 represents fat tails.
    """
    y = np.asarray(excesses, dtype=float)
    y = y[y > 1e-12]
    m = len(y)
    if m < 5:
        return 0.1, float(np.mean(y) if m > 0 else 0.01)

    mean_y = float(np.mean(y))
    var_y = float(np.var(y))

    # Initial parameter guess via Method of Moments:
    # E[Y] = beta / (1 - xi), Var[Y] = beta^2 / ((1 - xi)^2 * (1 - 2*xi))
    if var_y > 1e-12 and mean_y > 1e-12:
        ratio = (mean_y ** 2) / var_y
        xi_init = float(np.clip(0.5 * (1.0 - ratio), -0.5, 0.5))
        beta_init = float(max(1e-4, mean_y * (1.0 - xi_init)))
    else:
        xi_init = 0.1
        beta_init = max(1e-4, mean_y)

    def neg_log_likelihood(params: np.ndarray) -> float:
        xi, beta = params[0], params[1]
        if beta <= 1e-8:
            return 1e12

        # Support constraint: 1 + xi * y / beta > 0 for all y
        terms = 1.0 + (xi * y) / beta
        if np.any(terms <= 1e-12):
            return 1e12

        if abs(xi) < 1e-6:
            # Exponential limit as xi -> 0
            nll = m * np.log(beta) + np.sum(y) / beta
        else:
            nll = m * np.log(beta) + (1.0 + 1.0 / xi) * np.sum(np.log(terms))

        if not np.isfinite(nll):
            return 1e12
        return float(nll)

    bounds = [(-0.5, 0.95), (1e-6, None)]
    res = optimize.minimize(
        neg_log_likelihood,
        x0=np.array([xi_init, beta_init]),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if res.success and np.isfinite(res.x[0]) and np.isfinite(res.x[1]):
        xi_hat = float(np.clip(res.x[0], -0.5, 0.95))
        beta_hat = float(max(1e-6, res.x[1]))
        return xi_hat, beta_hat

    return xi_init, beta_init


def calculate_evt_var_cvar(
    losses: np.ndarray,
    quantile_threshold: float = 0.95,
    alpha: float = 0.99,
) -> Dict[str, float]:
    """Calculate semi-parametric EVT Peaks-Over-Threshold VaR and Expected Shortfall (CVaR).

    Mathematical Closed-Form Formulas:
        VaR_alpha^EVT = u + (beta / xi) * [ ((N / N_u) * (1 - alpha))^(-xi) - 1 ]
        ES_alpha^EVT  = (VaR_alpha^EVT / (1 - xi)) + ((beta - xi * u) / (1 - xi))

    Args:
        losses: Array of positive percentage losses L = -r.
        quantile_threshold: Threshold quantile q in (0.80, 0.98) to define threshold u.
        alpha: Confidence level alpha in (0.90, 0.999).

    Returns:
        Dictionary with threshold u, xi, beta, var, cvar, and count of exceedances.
    """
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    if n < 20:
        return {
            "u": 0.0,
            "xi": 0.1,
            "beta": 0.01,
            "n_u": 0,
            "var": 0.0,
            "cvar": 0.0,
        }

    q = float(np.clip(quantile_threshold, 0.80, 0.98))
    u = float(np.quantile(losses, q))

    excesses = losses[losses > u] - u
    n_u = len(excesses)

    if n_u < 5:
        # Fallback to empirical quantile if exceedances are too few
        emp_var = float(np.quantile(losses, alpha))
        return {
            "u": u,
            "xi": 0.0,
            "beta": float(np.mean(excesses) if n_u > 0 else 0.01),
            "n_u": n_u,
            "var": emp_var,
            "cvar": emp_var * 1.25,
        }

    xi, beta = fit_generalized_pareto(excesses)

    # Probability scaling: prob = (N / N_u) * (1 - alpha)
    prob_ratio = (float(n) / float(n_u)) * (1.0 - alpha)

    if abs(xi) < 1e-6:
        var_evt = u - beta * np.log(max(1e-12, prob_ratio))
        cvar_evt = var_evt + beta
    else:
        var_evt = u + (beta / xi) * (np.power(max(1e-12, prob_ratio), -xi) - 1.0)
        # ES requires xi < 1 for finite expectation
        xi_clamped = min(0.95, xi)
        cvar_evt = (var_evt / (1.0 - xi_clamped)) + ((beta - xi_clamped * u) / (1.0 - xi_clamped))

    # Guard guarantee: ES must strictly exceed VaR
    if cvar_evt < var_evt:
        cvar_evt = var_evt + max(1e-4, beta)

    return {
        "u": u,
        "xi": xi,
        "beta": beta,
        "n_u": n_u,
        "var": float(var_evt),
        "cvar": float(cvar_evt),
    }


def evaluate_evt_tail_profile(
    ticker: str,
    prices: Union[pd.Series, np.ndarray, List[float]],
    threshold_quantile: float = 0.95,
) -> EVTTailReceipt:
    """Evaluate full Extreme Value Theory tail risk profile and issue audit receipt.

    Args:
        ticker: Asset ticker symbol.
        prices: Historical price series.
        threshold_quantile: Quantile defining high loss threshold u (default: 0.95).

    Returns:
        Cryptographically signed EVTTailReceipt.
    """
    if isinstance(prices, pd.Series):
        s = prices.dropna().astype(float)
    else:
        s = pd.Series(prices).dropna().astype(float)

    n_sessions = len(s)
    if n_sessions < 30:
        return EVTTailReceipt(
            ticker=ticker.upper().strip(),
            sample_sessions=n_sessions,
            threshold_u=0.0,
            threshold_quantile=threshold_quantile,
            exceedances_count=0,
            gpd_shape_xi=0.0,
            gpd_scale_beta=0.0,
            tail_fatness_regime="INSUFFICIENT_DATA",
            var_95_pct=0.0,
            cvar_95_pct=0.0,
            var_99_pct=0.0,
            cvar_99_pct=0.0,
            gaussian_var_99_pct=0.0,
            cornish_fisher_var_99_pct=0.0,
            tail_risk_underestimation_pct=0.0,
            summary=f"Insufficient history ({n_sessions} sessions, minimum 30 required) to evaluate EVT tail risk.",
        )

    # Daily percentage losses: L = -(P_t - P_{t-1}) / P_{t-1} * 100
    ret = s.pct_change().dropna().values
    losses = -ret * 100.0  # express as positive percentages for losses

    # Compute EVT VaR & CVaR at 95% and 99%
    res_95 = calculate_evt_var_cvar(losses, quantile_threshold=threshold_quantile, alpha=0.95)
    res_99 = calculate_evt_var_cvar(losses, quantile_threshold=threshold_quantile, alpha=0.99)

    xi = res_99["xi"]
    beta = res_99["beta"]
    u = res_99["u"]
    n_u = res_99["n_u"]

    # Benchmark: Gaussian VaR 99%
    mu_loss = float(np.mean(losses))
    std_loss = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
    gauss_var_99 = mu_loss + 2.3263 * std_loss

    # Benchmark: Cornish-Fisher VaR 99%
    skew_loss = float(stats.skew(losses)) if len(losses) > 2 else 0.0
    kurt_loss = float(stats.kurtosis(losses)) if len(losses) > 3 else 0.0
    z = 2.3263
    cf_z = z + (z**2 - 1.0) * skew_loss / 6.0 + (z**3 - 3.0 * z) * kurt_loss / 24.0 - (2.0 * z**3 - 5.0 * z) * (skew_loss**2) / 36.0
    cf_var_99 = mu_loss + cf_z * std_loss

    # Underestimation %: (EVT_VaR - Gauss_VaR) / Gauss_VaR * 100
    if gauss_var_99 > 1e-4:
        underestimation_pct = float(((res_99["var"] - gauss_var_99) / gauss_var_99) * 100.0)
    else:
        underestimation_pct = 0.0

    # Tail fatness classification
    if xi > 0.15:
        tail_regime = "HEAVY_PARETO_TAIL"
        regime_desc = f"Heavy Pareto tail (ξ = {xi:.3f} > 0.15), indicating severe vulnerability to black swan tail shocks"
    elif xi >= -0.05:
        tail_regime = "MODERATE_EXPONENTIAL"
        regime_desc = f"Moderate exponential tail (ξ = {xi:.3f}), tail risk decays near exponential rate"
    else:
        tail_regime = "BOUNDED_LIGHT_TAIL"
        regime_desc = f"Bounded thin tail (ξ = {xi:.3f} < 0), finite upper bound on extreme losses"

    summary = (
        f"{ticker}: EVT 99% 1-day VaR is {res_99['var']:.2f}% and Expected Shortfall is {res_99['cvar']:.2f}%. "
        f"GPD threshold u = {u:.2f}% ({n_u} exceedances). Standard Gaussian models underestimate tail risk by {underestimation_pct:+.1f}%. "
        f"{regime_desc}."
    )

    return EVTTailReceipt(
        ticker=ticker.upper().strip(),
        sample_sessions=n_sessions,
        threshold_u=round(u, 4),
        threshold_quantile=threshold_quantile,
        exceedances_count=n_u,
        gpd_shape_xi=round(xi, 4),
        gpd_scale_beta=round(beta, 6),
        tail_fatness_regime=tail_regime,
        var_95_pct=round(res_95["var"], 2),
        cvar_95_pct=round(res_95["cvar"], 2),
        var_99_pct=round(res_99["var"], 2),
        cvar_99_pct=round(res_99["cvar"], 2),
        gaussian_var_99_pct=round(gauss_var_99, 2),
        cornish_fisher_var_99_pct=round(cf_var_99, 2),
        tail_risk_underestimation_pct=round(underestimation_pct, 1),
        summary=summary,
    )
