"""Asymmetric GJR-GARCH(1, 1) Volatility Forecasting & Leverage Effect Engine.

Implements Glosten, Jagannathan & Runkle (1993) asymmetric GJR-GARCH(1, 1)
conditional variance modeling to capture the leverage effect (Black 1976),
and analytical forward-horizon volatility term structure forecasting.

References:
    Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between
    the expected value and the volatility of the nominal excess return on stocks.
    The Journal of Finance, 48(5), 1779-1801.
    Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
    Journal of Econometrics, 31(3), 307-327.
"""

import hashlib
import hmac
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize

GARCH_SIGNING_KEY = b"nextomni_garch_secret_audit_key_v1"


@dataclass
class GARCHReceipt:
    """Cryptographic audit receipt for GJR-GARCH conditional volatility estimation."""

    ticker: str
    sample_sessions: int
    model_type: str
    omega: float
    alpha: float
    gamma_asymmetry: float
    beta: float
    persistence_phi: float
    leverage_asymmetry_ratio: float
    unconditional_volatility_ann_pct: float
    current_conditional_volatility_ann_pct: float
    term_structure_forecast_ann_pct: Dict[str, float]
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core volatility metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.model_type}:{self.omega:.8e}:"
            f"{self.alpha:.6f}:{self.gamma_asymmetry:.6f}:{self.beta:.6f}:"
            f"{self.persistence_phi:.6f}:{self.current_conditional_volatility_ann_pct:.4f}"
        )
        return hmac.new(GARCH_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt provenance against unauthorized record modification."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def fit_gjr_garch(
    returns: np.ndarray,
    asymmetric: bool = True,
) -> Dict[str, Any]:
    """Fit GJR-GARCH(1, 1) or standard GARCH(1, 1) via Quasi-Maximum Likelihood Estimation.

    Variance Equation:
        sigma_t^2 = omega + (alpha + gamma * I_{t-1}) * eps_{t-1}^2 + beta * sigma_{t-1}^2
        where I_{t-1} = 1 if eps_{t-1} < 0 else 0.

    Args:
        returns: Array of historical asset returns (e.g. daily percentage or fractional returns).
        asymmetric: Whether to include the GJR asymmetric leverage parameter gamma.

    Returns:
        Dictionary with parameters: omega, alpha, gamma, beta, persistence, final_sigma, final_eps, var_unconditional.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)

    sample_var = float(np.var(r, ddof=1)) if n > 1 else 1e-4
    sample_var = max(1e-8, sample_var)

    # De-mean residuals
    mu = float(np.mean(r))
    eps = r - mu

    # Variance targeting initial parameter guesses
    alpha_0 = 0.05
    gamma_0 = 0.05 if asymmetric else 0.0
    beta_0 = 0.85
    phi_0 = alpha_0 + 0.5 * gamma_0 + beta_0
    omega_0 = max(1e-8, sample_var * (1.0 - phi_0))

    if asymmetric:
        init_params = np.array([omega_0, alpha_0, gamma_0, beta_0], dtype=float)
        # Bounds: omega > 0, alpha >= 0, gamma >= 0, beta >= 0
        bounds = [(1e-8, None), (0.0, 0.4), (0.0, 0.5), (0.0, 0.99)]
    else:
        init_params = np.array([omega_0, alpha_0, 0.0, beta_0], dtype=float)
        bounds = [(1e-8, None), (0.0, 0.4), (0.0, 0.0), (0.0, 0.99)]

    def neg_log_likelihood(params: np.ndarray) -> float:
        omega, alpha, gamma, beta = params[0], params[1], params[2], params[3]

        # Enforce stationarity condition: alpha + gamma / 2 + beta < 1
        phi = alpha + 0.5 * gamma + beta
        if phi >= 0.999 or omega <= 0.0 or alpha < 0.0 or gamma < 0.0 or beta < 0.0:
            return 1e12

        # Filter conditional variance recursion
        sigma2 = np.empty(n, dtype=float)
        sigma2[0] = sample_var

        for t in range(1, n):
            eps_prev = eps[t - 1]
            i_prev = 1.0 if eps_prev < 0.0 else 0.0
            sigma2[t] = omega + (alpha + gamma * i_prev) * (eps_prev ** 2) + beta * sigma2[t - 1]
            if sigma2[t] <= 1e-12:
                sigma2[t] = 1e-12

        # Negative log-likelihood (ignoring constant term n/2 * ln(2*pi))
        nll = 0.5 * np.sum(np.log(sigma2) + (eps ** 2) / sigma2)
        if not np.isfinite(nll):
            return 1e12
        return float(nll)

    try:
        res = optimize.minimize(
            neg_log_likelihood,
            x0=init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if res.success and np.all(np.isfinite(res.x)):
            omega_hat = float(max(1e-8, res.x[0]))
            alpha_hat = float(max(0.0, res.x[1]))
            gamma_hat = float(max(0.0, res.x[2])) if asymmetric else 0.0
            beta_hat = float(max(0.0, res.x[3]))
        else:
            omega_hat, alpha_hat, gamma_hat, beta_hat = omega_0, alpha_0, gamma_0, beta_0
    except Exception:
        omega_hat, alpha_hat, gamma_hat, beta_hat = omega_0, alpha_0, gamma_0, beta_0

    phi_hat = alpha_hat + 0.5 * gamma_hat + beta_hat
    denom = max(1e-4, 1.0 - phi_hat)
    uncond_var = float(omega_hat / denom)

    # Reconstruct final conditional variance
    curr_sigma2 = sample_var
    for t in range(1, n):
        eps_prev = eps[t - 1]
        i_prev = 1.0 if eps_prev < 0.0 else 0.0
        curr_sigma2 = omega_hat + (alpha_hat + gamma_hat * i_prev) * (eps_prev ** 2) + beta_hat * curr_sigma2

    return {
        "omega": omega_hat,
        "alpha": alpha_hat,
        "gamma": gamma_hat,
        "beta": beta_hat,
        "persistence_phi": float(phi_hat),
        "unconditional_variance": uncond_var,
        "final_sigma2": float(curr_sigma2),
        "final_eps": float(eps[-1] if n > 0 else 0.0),
    }


def forecast_garch_term_structure(
    fit_res: Dict[str, Any],
    horizons: List[int] = [1, 5, 21],
) -> Dict[str, float]:
    """Forecast forward cumulative annualized conditional volatility across multi-step horizons.

    Multi-Step Conditional Variance Projection:
        sigma_{T+1}^2 = omega + (alpha + gamma * I_T) * eps_T^2 + beta * sigma_T^2
        For k >= 2: E[sigma_{T+k}^2] = bar{sigma}^2 + phi^{k-1} * (sigma_{T+1}^2 - bar{sigma}^2)

    Args:
        fit_res: Fitted GARCH parameter dictionary from fit_gjr_garch.
        horizons: Forward session horizons to project (default: [1, 5, 21]).

    Returns:
        Dictionary mapping horizon keys to annualized volatility percentages.
    """
    omega = fit_res["omega"]
    alpha = fit_res["alpha"]
    gamma = fit_res["gamma"]
    beta = fit_res["beta"]
    phi = fit_res["persistence_phi"]
    bar_sigma2 = fit_res["unconditional_variance"]
    final_sigma2 = fit_res["final_sigma2"]
    final_eps = fit_res["final_eps"]

    # Next session 1-step forecast
    i_final = 1.0 if final_eps < 0.0 else 0.0
    sigma_1_sq = float(omega + (alpha + gamma * i_final) * (final_eps ** 2) + beta * final_sigma2)
    sigma_1_sq = max(1e-8, sigma_1_sq)

    max_h = max(horizons)
    step_variances = np.empty(max_h, dtype=float)
    step_variances[0] = sigma_1_sq

    for k in range(2, max_h + 1):
        # E[sigma_{T+k}^2] = bar{sigma}^2 + phi^{k-1} * (sigma_{T+1}^2 - bar{sigma}^2)
        decay = math.pow(max(0.0, phi), k - 1)
        step_variances[k - 1] = float(bar_sigma2 + decay * (sigma_1_sq - bar_sigma2))

    term_forecasts: Dict[str, float] = {}
    for h in horizons:
        # Cumulative average variance over horizon h
        avg_var = float(np.mean(step_variances[:h]))
        # Annualized volatility %: sqrt(avg_var * 252) * 100
        ann_vol_pct = float(math.sqrt(avg_var * 252.0) * 100.0)
        label = f"{h}_session" if h == 1 else f"{h}_sessions"
        term_forecasts[label] = round(ann_vol_pct, 2)

    return term_forecasts


def evaluate_garch_profile(
    ticker: str,
    prices: Union[pd.Series, np.ndarray, List[float]],
    asymmetric: bool = True,
) -> GARCHReceipt:
    """Evaluate full GJR-GARCH conditional volatility and forward term structure.

    Args:
        ticker: Asset ticker symbol.
        prices: Historical price series.
        asymmetric: Whether to fit asymmetric GJR-GARCH (default: True) or standard GARCH.

    Returns:
        Cryptographically signed GARCHReceipt.
    """
    if isinstance(prices, pd.Series):
        s = prices.dropna().astype(float)
    else:
        s = pd.Series(prices).dropna().astype(float)

    n_sessions = len(s)
    if n_sessions < 40:
        return GARCHReceipt(
            ticker=ticker.upper().strip(),
            sample_sessions=n_sessions,
            model_type="GJR-GARCH(1,1)" if asymmetric else "Standard GARCH(1,1)",
            omega=0.0,
            alpha=0.0,
            gamma_asymmetry=0.0,
            beta=0.0,
            persistence_phi=0.0,
            leverage_asymmetry_ratio=1.0,
            unconditional_volatility_ann_pct=0.0,
            current_conditional_volatility_ann_pct=0.0,
            term_structure_forecast_ann_pct={},
            summary=f"Insufficient history ({n_sessions} sessions, minimum 40 required) for GARCH volatility estimation.",
        )

    # Fractional returns for volatility modeling
    returns = s.pct_change().dropna().values

    fit_res = fit_gjr_garch(returns=returns, asymmetric=asymmetric)
    term_forecasts = forecast_garch_term_structure(fit_res, horizons=[1, 5, 21])

    omega = fit_res["omega"]
    alpha = fit_res["alpha"]
    gamma = fit_res["gamma"]
    beta = fit_res["beta"]
    phi = fit_res["persistence_phi"]
    uncond_var = fit_res["unconditional_variance"]
    final_sigma2 = fit_res["final_sigma2"]

    uncond_vol_ann = float(math.sqrt(max(1e-8, uncond_var) * 252.0) * 100.0)
    current_vol_ann = float(math.sqrt(max(1e-8, final_sigma2) * 252.0) * 100.0)

    # Leverage effect asymmetry ratio: (alpha + gamma) / max(1e-4, alpha)
    asymmetry_ratio = float((alpha + gamma) / max(1e-4, alpha)) if asymmetric else 1.0

    model_label = "GJR-GARCH(1,1)" if asymmetric else "Standard GARCH(1,1)"

    if asymmetric and gamma > 0.02:
        leverage_desc = f"Statistically prominent leverage asymmetry (γ = {gamma:.3f}, ratio = {asymmetry_ratio:.2f}x bad news volatility amplification)"
    elif asymmetric:
        leverage_desc = f"Mild leverage asymmetry (γ = {gamma:.3f})"
    else:
        leverage_desc = "Symmetric volatility response"

    summary = (
        f"{ticker}: {model_label} conditional volatility is {current_vol_ann:.2f}% annualized (long-term mean: {uncond_vol_ann:.2f}%). "
        f"Variance persistence φ = {phi:.3f}. {leverage_desc}. "
        f"Forward term structure: 1-Day {term_forecasts.get('1_session', 0.0)}%, 5-Day {term_forecasts.get('5_sessions', 0.0)}%, 21-Day {term_forecasts.get('21_sessions', 0.0)}%."
    )

    return GARCHReceipt(
        ticker=ticker.upper().strip(),
        sample_sessions=n_sessions,
        model_type=model_label,
        omega=float(omega),
        alpha=round(float(alpha), 4),
        gamma_asymmetry=round(float(gamma), 4),
        beta=round(float(beta), 4),
        persistence_phi=round(float(phi), 4),
        leverage_asymmetry_ratio=round(asymmetry_ratio, 2),
        unconditional_volatility_ann_pct=round(uncond_vol_ann, 2),
        current_conditional_volatility_ann_pct=round(current_vol_ann, 2),
        term_structure_forecast_ann_pct=term_forecasts,
        summary=summary,
    )
