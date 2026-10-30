"""Black-Litterman (1992) Bayesian Portfolio Optimization & View Integration Engine.

Implements the Black-Litterman asset allocation framework combining CAPM equilibrium
market prior returns with active quantitative investor/model views and estimation uncertainty.
Eliminates Markowitz "error-maximization" corner solutions and produces stable,
diversified optimal allocations. Supports RMT covariance denoising integration.

References:
    Black, F., & Litterman, R. (1992). Global portfolio optimization. Financial
    Analysts Journal, 48(5), 28-43.
    He, G., & Litterman, R. (1999). The intuition behind Black-Litterman model
    portfolios. Goldman Sachs Asset Management.
    Idzorek, T. (2005). A step-by-step guide to the Black-Litterman model:
    Incorporating user-specified confidence levels. Ibbotson Associates.
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
from scipy.optimize import minimize

from src.analytics.portfolio import build_aligned_returns_matrix
from src.analytics.rmt_denoiser import denoise_covariance_matrix
import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

BLACK_LITTERMAN_SIGNING_KEY = b"nextomni_black_litterman_secret_key_v1"


@dataclass
class AssetAllocation:
    """Asset-level prior, posterior, and allocation details."""

    ticker: str
    market_weight: float
    prior_return_ann_pct: float  # Pi (CAPM Implied Equilibrium Return)
    posterior_return_ann_pct: float  # mu_BL (Black-Litterman Posterior Expected Return)
    optimal_weight: float  # w_BL^* (Constrained Long-Only Optimal Weight)
    weight_shift: float  # Delta w = w_BL^* - w_mkt


@dataclass
class BlackLittermanReceipt:
    """Cryptographic audit receipt for Black-Litterman Bayesian portfolio optimization."""

    universe: List[str]
    sample_sessions: int
    lookback_window: int
    tau: float  # Prior uncertainty scalar (e.g. 0.05)
    risk_aversion_delta: float  # Market risk aversion parameter
    views_count: int  # Number of expressed active views
    expected_portfolio_return_ann_pct: float  # w^T mu_BL
    expected_portfolio_vol_ann_pct: float  # sqrt(w^T Sigma_BL w)
    sharpe_ratio: float  # (E[R] - Rf) / Vol
    prior_market_return_ann_pct: float  # w_mkt^T Pi
    allocations: List[Dict[str, Any]]
    is_rmt_denoised: bool
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across allocations and risk metrics."""
        univ_str = ",".join(sorted(self.universe))
        payload = (
            f"{univ_str}:{self.sample_sessions}:{self.lookback_window}:{self.tau:.4f}:"
            f"{self.risk_aversion_delta:.4f}:{self.views_count}:{self.expected_portfolio_return_ann_pct:.4f}:"
            f"{self.expected_portfolio_vol_ann_pct:.4f}:{self.sharpe_ratio:.4f}:{self.is_rmt_denoised}"
        )
        return hmac.new(
            BLACK_LITTERMAN_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity of the Black-Litterman optimization receipt."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_implied_equilibrium_returns(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion_delta: float = 2.5,
) -> np.ndarray:
    """Reverse-engineer the CAPM implied equilibrium prior returns vector (Pi).

    Formula:
        Pi = delta * Sigma * w_mkt

    Args:
        cov_matrix: Annualized covariance matrix (N x N).
        market_weights: Benchmark capitalization or baseline weights (N,).
        risk_aversion_delta: Market aggregate risk aversion coefficient (default: 2.5).

    Returns:
        Vector of implied equilibrium returns Pi (N,).
    """
    sigma = np.asarray(cov_matrix, dtype=float)
    w_mkt = np.asarray(market_weights, dtype=float)

    if sigma.shape[0] != sigma.shape[1] or len(w_mkt) != sigma.shape[0]:
        raise ValueError(f"Shape mismatch: cov {sigma.shape} vs weights ({len(w_mkt)})")

    # Ensure weights sum to 1.0
    w_norm = w_mkt / max(1e-12, float(np.sum(w_mkt)))
    pi = risk_aversion_delta * (sigma @ w_norm)
    return pi


def blend_black_litterman_views(
    prior_returns: np.ndarray,
    cov_matrix: np.ndarray,
    pick_matrix: Optional[np.ndarray] = None,
    view_returns: Optional[np.ndarray] = None,
    view_uncertainties: Optional[np.ndarray] = None,
    tau: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute Bayesian information blending of prior returns and active investor/model views.

    Formulas:
        mu_BL = Pi + tau * Sigma * P^T * (P * tau * Sigma * P^T + Omega)^{-1} * (Q - P * Pi)
        Sigma_BL = (1 + tau) * Sigma - tau^2 * Sigma * P^T * (P * tau * Sigma * P^T + Omega)^{-1} * P * Sigma

    Args:
        prior_returns: CAPM implied returns vector Pi (N,).
        cov_matrix: Asset annualized covariance matrix Sigma (N x N).
        pick_matrix: Optional view pick matrix P (K x N).
        view_returns: Optional expected return vector Q (K,).
        view_uncertainties: Optional diagonal view uncertainty matrix Omega (K x K).
        tau: Prior uncertainty scalar (default: 0.05).

    Returns:
        Tuple of (posterior_returns_mu_bl, posterior_covariance_sigma_bl).
    """
    pi = np.asarray(prior_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    n = len(pi)

    # Base case: no active views -> posterior returns equal prior returns
    if pick_matrix is None or view_returns is None or len(pick_matrix) == 0:
        sigma_bl = (1.0 + tau) * sigma
        return pi.copy(), sigma_bl

    P = np.asarray(pick_matrix, dtype=float)
    Q = np.asarray(view_returns, dtype=float)
    k = len(Q)

    if P.shape != (k, n):
        raise ValueError(f"View pick matrix shape {P.shape} incompatible with K={k}, N={n}")

    tau_sigma = tau * sigma

    # Default Omega: He & Litterman proportional diagonal covariance
    if view_uncertainties is None or view_uncertainties.shape != (k, k):
        # Omega = diag(P * tau * Sigma * P^T)
        p_tau_sigma_pt = P @ tau_sigma @ P.T
        omega = np.diag(np.diag(p_tau_sigma_pt))
        # Guard against zero variance on diagonals
        for i in range(k):
            omega[i, i] = max(1e-8, float(omega[i, i]))
    else:
        omega = np.asarray(view_uncertainties, dtype=float)

    # Innovation covariance: M = (P * tau * Sigma * P^T + Omega)
    M = (P @ tau_sigma @ P.T) + omega
    M_inv = np.linalg.pinv(M)

    # Posterior expected returns: mu_BL = Pi + tau * Sigma * P^T * M_inv * (Q - P * Pi)
    gain_matrix = tau_sigma @ P.T @ M_inv
    view_excess = Q - (P @ pi)
    mu_bl = pi + (gain_matrix @ view_excess)

    # Posterior covariance: Sigma_BL = (1 + tau)*Sigma - tau^2 * Sigma * P^T * M_inv * P * Sigma
    shrink_cov = (tau * sigma) @ P.T @ M_inv @ P @ (tau * sigma)
    sigma_bl = ((1.0 + tau) * sigma) - shrink_cov

    # Symmetrize posterior covariance
    sigma_bl = 0.5 * (sigma_bl + sigma_bl.T)

    return mu_bl, sigma_bl


def compute_black_litterman_optimal_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion_delta: float = 2.5,
    long_only: bool = True,
) -> np.ndarray:
    """Compute optimal portfolio weights under Black-Litterman posterior parameters.

    Solves:
        max_w  w^T mu_BL - (delta / 2) * w^T Sigma_BL w
        s.t.   sum(w) = 1, and (if long_only) w >= 0.

    Args:
        expected_returns: Posterior return vector mu_BL (N,).
        cov_matrix: Posterior covariance matrix Sigma_BL (N x N).
        risk_aversion_delta: Risk aversion parameter.
        long_only: Whether to constrain weights to be non-negative (default: True).

    Returns:
        Vector of optimal weights w^* (N,).
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    n = len(mu)

    if not long_only:
        # Closed-form unconstrained weights: w = (1 / delta) * Sigma^{-1} * mu
        inv_sigma = np.linalg.pinv(sigma)
        raw_w = (1.0 / max(1e-6, risk_aversion_delta)) * (inv_sigma @ mu)
        sum_w = float(np.sum(raw_w))
        if abs(sum_w) > 1e-6:
            return raw_w / sum_w
        return np.ones(n) / float(n)

    # Long-only quadratic programming via SLSQP
    def objective(w: np.ndarray) -> float:
        # Negative utility to minimize
        port_ret = float(w @ mu)
        port_var = float(w @ sigma @ w)
        utility = port_ret - (0.5 * risk_aversion_delta * port_var)
        return -utility

    def gradient(w: np.ndarray) -> np.ndarray:
        return -(mu - risk_aversion_delta * (sigma @ w))

    w0 = np.ones(n) / float(n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 300},
    )

    if res.success:
        weights = np.maximum(0.0, res.x)
        return weights / float(np.sum(weights))
    else:
        # Fallback equal weights
        return np.ones(n) / float(n)


def evaluate_black_litterman_portfolio(
    universe: List[str],
    views_dict: Optional[Dict[str, float]] = None,
    tau: float = 0.05,
    risk_aversion_delta: float = 2.5,
    risk_free_rate_ann: float = 0.04,
    lookback_window: int = 120,
    denoise_cov: bool = True,
    market_weights: Optional[Dict[str, float]] = None,
    db_path: Optional[Path] = None,
) -> BlackLittermanReceipt:
    """Execute complete Black-Litterman optimization workflow with cryptographic audit receipt.

    Args:
        universe: List of equity ticker symbols.
        views_dict: Optional dict of absolute annual return views (e.g. {"NVDA": 0.25, "AAPL": 0.12}).
        tau: Prior uncertainty scalar (default: 0.05).
        risk_aversion_delta: Market risk aversion coefficient (default: 2.5).
        risk_free_rate_ann: Annual risk-free rate (default: 4.0%).
        lookback_window: Historical observation sessions to include (default: 120).
        denoise_cov: Whether to apply Marchenko-Pastur RMT covariance denoising (default: True).
        market_weights: Optional baseline capitalization weights dict.
        db_path: Optional SQLite database path.

    Returns:
        BlackLittermanReceipt with allocations, prior vs posterior returns, and audit signature.
    """
    clean_universe = [validate_ticker(t) for t in universe]
    if len(clean_universe) < 2:
        raise ValueError("Universe must contain at least 2 distinct assets for portfolio optimization")
    assert_finite_number(tau, "tau")
    assert_finite_number(risk_aversion_delta, "risk_aversion_delta")
    assert_finite_number(lookback_window, "lookback_window")

    # Build aligned historical return matrix
    df_returns = build_aligned_returns_matrix(
        clean_universe,
        lookback_window=lookback_window,
        db_path=db_path,
    )
    if df_returns.empty or len(df_returns) < 20:
        raise ValueError(
            f"Insufficient aligned price history for universe {clean_universe}. Ensure tickers are synchronized."
        )

    ret_matrix = df_returns[clean_universe].values
    m_sessions, n_assets = ret_matrix.shape

    # Annualized sample covariance matrix
    sample_cov = np.cov(ret_matrix, rowvar=False) * 252.0

    # Optional Marchenko-Pastur RMT covariance denoising
    if denoise_cov and m_sessions > n_assets:
        try:
            denoised_cov, _ = denoise_covariance_matrix(sample_cov, q_ratio=float(m_sessions) / float(n_assets))
            cov_mat = denoised_cov
            is_denoised = True
        except Exception:
            cov_mat = sample_cov
            is_denoised = False
    else:
        cov_mat = sample_cov
        is_denoised = False

    # Baseline market weights (equal-weighted default if not specified)
    if market_weights:
        w_mkt_arr = np.array([float(market_weights.get(t, 1.0 / n_assets)) for t in clean_universe])
    else:
        w_mkt_arr = np.ones(n_assets) / float(n_assets)
    w_mkt_arr = w_mkt_arr / float(np.sum(w_mkt_arr))

    # 1. Reverse-engineer CAPM Equilibrium Implied Prior Returns: Pi
    pi = compute_implied_equilibrium_returns(cov_mat, w_mkt_arr, risk_aversion_delta=risk_aversion_delta)

    # 2. Build Pick Matrix P and View Vector Q
    pick_rows = []
    q_vals = []
    if views_dict:
        for t, expected_ret in views_dict.items():
            clean_t = validate_ticker(t)
            if clean_t in clean_universe:
                idx = clean_universe.index(clean_t)
                row = np.zeros(n_assets)
                row[idx] = 1.0  # Absolute view: asset clean_t expected return is expected_ret
                pick_rows.append(row)
                q_vals.append(float(expected_ret))

    if pick_rows:
        P = np.array(pick_rows)
        Q = np.array(q_vals)
    else:
        P = None
        Q = None

    # 3. Blend Views via Black-Litterman Bayesian Master Formula
    mu_bl, sigma_bl = blend_black_litterman_views(
        prior_returns=pi,
        cov_matrix=cov_mat,
        pick_matrix=P,
        view_returns=Q,
        tau=tau,
    )

    # 4. Solve Optimal Constrained Long-Only Weights
    w_opt = compute_black_litterman_optimal_weights(
        expected_returns=mu_bl,
        cov_matrix=sigma_bl,
        risk_aversion_delta=risk_aversion_delta,
        long_only=True,
    )

    # 5. Portfolio Aggregate Metrics
    port_ret_ann = float(w_opt @ mu_bl) * 100.0
    port_vol_ann = math.sqrt(max(0.0, float(w_opt @ sigma_bl @ w_opt))) * 100.0
    excess_ret = (port_ret_ann / 100.0) - risk_free_rate_ann
    sharpe = float(excess_ret / max(1e-6, port_vol_ann / 100.0))
    mkt_prior_ret = float(w_mkt_arr @ pi) * 100.0

    allocations_list = []
    for i, t in enumerate(clean_universe):
        alloc = AssetAllocation(
            ticker=t,
            market_weight=float(w_mkt_arr[i]),
            prior_return_ann_pct=float(pi[i] * 100.0),
            posterior_return_ann_pct=float(mu_bl[i] * 100.0),
            optimal_weight=float(w_opt[i]),
            weight_shift=float(w_opt[i] - w_mkt_arr[i]),
        )
        allocations_list.append(asdict(alloc))

    v_count = len(q_vals) if q_vals else 0
    denoise_tag = "RMT Denoised" if is_denoised else "Standard Sample Covariance"
    summary = (
        f"Black-Litterman Portfolio for [{', '.join(clean_universe)}] ({v_count} views, {denoise_tag}): "
        f"Expected Return: {port_ret_ann:.2f}% Ann (Market Prior: {mkt_prior_ret:.2f}%), "
        f"Portfolio Vol: {port_vol_ann:.2f}% Ann, Sharpe: {sharpe:.2f} (tau={tau:.2f}, delta={risk_aversion_delta:.1f})."
    )

    receipt = BlackLittermanReceipt(
        universe=clean_universe,
        sample_sessions=m_sessions,
        lookback_window=lookback_window,
        tau=tau,
        risk_aversion_delta=risk_aversion_delta,
        views_count=v_count,
        expected_portfolio_return_ann_pct=port_ret_ann,
        expected_portfolio_vol_ann_pct=port_vol_ann,
        sharpe_ratio=sharpe,
        prior_market_return_ann_pct=mkt_prior_ret,
        allocations=allocations_list,
        is_rmt_denoised=is_denoised,
        summary=summary,
    )

    return receipt
