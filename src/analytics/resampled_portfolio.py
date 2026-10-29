"""Michaud (1998) Resampled Efficiency & Robust Frontier Optimization Engine.

Implements Richard Michaud's (1998) Resampled Efficient Frontier algorithm to
overcome Markowitz "Error Maximization", producing robust, turnover-dampened,
and estimation-error resilient asset allocations.

References:
    Michaud, R. O. (1989). The Markowitz optimization enigma: Is 'optimized'
    optimal?. Financial Analysts Journal, 45(1), 31-42.
    Michaud, R. O. (1998). Efficient Asset Management: A Practical Guide to
    Stock Portfolio Optimization and Asset Allocation. Harvard Business School Press.
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
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.analytics.portfolio import (
    build_aligned_returns_matrix,
    compute_ledoit_wolf_covariance,
)
from src.security.validators import assert_finite_number, validate_ticker

RESAMPLED_SIGNING_KEY = b"nextomni_resampled_frontier_secret_key_v1"


@dataclass
class FrontierPoint:
    """A single evaluated risk-return point on an efficient frontier."""

    rank_index: int
    expected_return_ann_pct: float
    volatility_ann_pct: float
    sharpe_ratio: float
    effective_assets: float
    weights: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResampledFrontierReceipt:
    """Cryptographic audit receipt for Michaud Resampled Efficient Frontier."""

    tickers: List[str]
    sample_sessions: int
    num_simulations: int
    num_frontier_points: int
    risk_free_rate_pct: float
    shrinkage_intensity: float
    classical_max_sharpe: FrontierPoint
    resampled_max_sharpe: FrontierPoint
    classical_min_variance: FrontierPoint
    resampled_min_variance: FrontierPoint
    classical_frontier: List[FrontierPoint]
    resampled_frontier: List[FrontierPoint]
    michaud_diversification_gain_pct: float
    weight_stability_dispersion: Dict[str, float]
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core frontier metrics."""
        payload = (
            f"{','.join(self.tickers)}:{self.sample_sessions}:{self.num_simulations}:"
            f"{self.num_frontier_points}:{self.risk_free_rate_pct:.4f}:"
            f"{self.classical_max_sharpe.sharpe_ratio:.4f}:{self.resampled_max_sharpe.sharpe_ratio:.4f}:"
            f"{self.michaud_diversification_gain_pct:.4f}"
        )
        return hmac.new(RESAMPLED_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["classical_max_sharpe"] = self.classical_max_sharpe.to_dict()
        d["resampled_max_sharpe"] = self.resampled_max_sharpe.to_dict()
        d["classical_min_variance"] = self.classical_min_variance.to_dict()
        d["resampled_min_variance"] = self.resampled_min_variance.to_dict()
        d["classical_frontier"] = [p.to_dict() for p in self.classical_frontier]
        d["resampled_frontier"] = [p.to_dict() for p in self.resampled_frontier]
        return d


def _calculate_effective_assets(weights: np.ndarray) -> float:
    """Calculate the Inverse Herfindahl-Hirschman index (Effective Number of Assets)."""
    herfindahl = float(np.sum(weights ** 2))
    return float(1.0 / max(herfindahl, 1e-8))


def _solve_portfolio_for_target_return(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Solve for minimum variance portfolio weights achieving at least target_return."""
    n = len(mu)
    if n == 1:
        return np.array([1.0])

    w0 = np.ones(n) / n
    bounds = tuple((0.0, max_weight) for _ in range(n))
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: float(w.T @ mu) - target_return},
    ]

    def var_obj(w: np.ndarray) -> float:
        return float(w.T @ cov @ w)

    res = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        w = np.maximum(res.x, 0.0)
        s = np.sum(w)
        return w / s if s > 0 else w0
    return w0


def _solve_min_variance_weights(
    cov: np.ndarray,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Solve for global minimum variance portfolio."""
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    w0 = np.ones(n) / n
    bounds = tuple((0.0, max_weight) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def var_obj(w: np.ndarray) -> float:
        return float(w.T @ cov @ w)

    res = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        w = np.maximum(res.x, 0.0)
        s = np.sum(w)
        return w / s if s > 0 else w0
    return w0


def trace_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: List[str],
    num_points: int = 15,
    risk_free_rate: float = 0.04,
) -> Tuple[List[FrontierPoint], np.ndarray]:
    """Trace an efficient frontier across num_points return levels.

    Args:
        mu: Expected annualized returns vector (N,).
        cov: Annualized covariance matrix (N, N).
        tickers: List of ticker symbols.
        num_points: Number of discrete frontier points.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Tuple of (List of FrontierPoint, 2D array of weights shape (num_points, N)).
    """
    n = len(tickers)
    if n == 1:
        w1 = np.array([1.0])
        exp_ret = float(mu[0])
        vol = float(math.sqrt(max(cov[0, 0], 1e-8)))
        sr = float((exp_ret - risk_free_rate) / max(vol, 1e-6))
        pt = FrontierPoint(
            rank_index=0,
            expected_return_ann_pct=round(exp_ret * 100.0, 2),
            volatility_ann_pct=round(vol * 100.0, 2),
            sharpe_ratio=round(sr, 3),
            effective_assets=1.0,
            weights={tickers[0]: 1.0},
        )
        return [pt] * num_points, np.ones((num_points, 1))

    w_min = _solve_min_variance_weights(cov)
    r_min = float(w_min.T @ mu)
    r_max = float(np.max(mu))

    if r_max <= r_min:
        r_max = r_min + 0.01

    target_returns = np.linspace(r_min, r_max, num_points)
    points: List[FrontierPoint] = []
    weights_matrix = np.zeros((num_points, n))

    for idx, r_target in enumerate(target_returns):
        w_k = _solve_portfolio_for_target_return(mu, cov, r_target)
        weights_matrix[idx, :] = w_k

        port_ret = float(w_k.T @ mu)
        port_vol = float(math.sqrt(max(w_k.T @ cov @ w_k, 1e-8)))
        sharpe = float((port_ret - risk_free_rate) / max(port_vol, 1e-6))
        eff_n = _calculate_effective_assets(w_k)

        w_dict = {tickers[i]: round(float(w_k[i]), 4) for i in range(n)}
        pt = FrontierPoint(
            rank_index=idx,
            expected_return_ann_pct=round(port_ret * 100.0, 2),
            volatility_ann_pct=round(port_vol * 100.0, 2),
            sharpe_ratio=round(sharpe, 3),
            effective_assets=round(eff_n, 2),
            weights=w_dict,
        )
        points.append(pt)

    return points, weights_matrix


def compute_resampled_efficient_frontier(
    returns_df: pd.DataFrame,
    num_simulations: int = 100,
    num_frontier_points: int = 15,
    risk_free_rate: float = 0.04,
    random_state: int = 42,
) -> ResampledFrontierReceipt:
    """Compute Michaud (1998) Resampled Efficient Frontier via Monte Carlo resampling.

    Args:
        returns_df: DataFrame of synchronized daily asset returns.
        num_simulations: Number of Monte Carlo resampling paths (B >= 20).
        num_frontier_points: Number of discrete points K along the frontier.
        risk_free_rate: Annualized risk-free benchmark rate.
        random_state: Random seed for reproducibility.

    Returns:
        ResampledFrontierReceipt with classical vs resampled allocations and audit metrics.
    """
    tickers = list(returns_df.columns)
    for t in tickers:
        validate_ticker(t)

    assert_finite_number(risk_free_rate, "risk_free_rate")
    num_simulations = max(20, min(num_simulations, 500))
    num_frontier_points = max(5, min(num_frontier_points, 50))

    t_obs, n_assets = returns_df.shape
    if n_assets == 0 or t_obs < 10:
        raise ValueError("Insufficient return observations to compute resampled efficient frontier.")

    # 1. Base estimation on true historical data
    lw_receipt = compute_ledoit_wolf_covariance(returns_df)
    cov_daily = lw_receipt.shrunk_covariance.values
    mu_daily = returns_df.mean().values

    # Annualize moments
    mu_ann = mu_daily * 252.0
    cov_ann = cov_daily * 252.0

    # Ensure cov_ann is symmetric and positive definite
    cov_ann = 0.5 * (cov_ann + cov_ann.T)
    min_eig = np.min(np.real(np.linalg.eigvals(cov_ann)))
    if min_eig < 1e-6:
        cov_ann += np.eye(n_assets) * (1e-6 - min_eig)

    # 2. Classical Markowitz Efficient Frontier
    classical_pts, classical_weights = trace_frontier(
        mu=mu_ann,
        cov=cov_ann,
        tickers=tickers,
        num_points=num_frontier_points,
        risk_free_rate=risk_free_rate,
    )

    # 3. Michaud Monte Carlo Resampling Simulation
    rng = np.random.RandomState(random_state)
    simulated_weights = np.zeros((num_simulations, num_frontier_points, n_assets))

    # Pre-compute Cholesky factor for fast multivariate normal draws
    try:
        chol_daily = np.linalg.cholesky(cov_daily)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(cov_daily)
        evals = np.maximum(evals, 1e-7)
        chol_daily = evecs @ np.diag(np.sqrt(evals))

    for b in range(num_simulations):
        # Draw T synthetic daily returns: R_b = mu_daily + Z @ chol^T
        z = rng.standard_normal(size=(t_obs, n_assets))
        sim_returns = mu_daily + (z @ chol_daily.T)

        # Estimate resampled moments
        sim_mu_daily = np.mean(sim_returns, axis=0)
        sim_mu_ann = sim_mu_daily * 252.0

        # Shrunk covariance for draw
        lw_sim = LedoitWolf()
        lw_sim.fit(sim_returns)
        sim_cov_ann = lw_sim.covariance_ * 252.0
        sim_cov_ann = 0.5 * (sim_cov_ann + sim_cov_ann.T)

        # Trace frontier on resampled data draw
        _, draw_weights = trace_frontier(
            mu=sim_mu_ann,
            cov=sim_cov_ann,
            tickers=tickers,
            num_points=num_frontier_points,
            risk_free_rate=risk_free_rate,
        )
        simulated_weights[b, :, :] = draw_weights

    # 4. Ensemble Average Across Simulations for Each Frontier Point
    # w_k^RE = 1/B * sum_{b=1}^B w_{k,b}
    resampled_weights = np.mean(simulated_weights, axis=0)
    # Re-normalize to enforce sum(w) = 1.0 exactly
    sums = np.sum(resampled_weights, axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    resampled_weights = resampled_weights / sums

    # Weight dispersion (standard deviation of weights across simulations)
    weight_stds = np.std(simulated_weights, axis=0)  # shape (K, N)
    avg_weight_dispersion = np.mean(weight_stds, axis=0)
    dispersion_dict = {
        tickers[i]: round(float(avg_weight_dispersion[i]), 4) for i in range(n_assets)
    }

    # 5. Evaluate Resampled Portfolios Against True Empirical Moments
    resampled_pts: List[FrontierPoint] = []
    for k in range(num_frontier_points):
        w_re = resampled_weights[k, :]
        p_ret = float(w_re.T @ mu_ann)
        p_vol = float(math.sqrt(max(w_re.T @ cov_ann @ w_re, 1e-8)))
        p_sr = float((p_ret - risk_free_rate) / max(p_vol, 1e-6))
        eff_n = _calculate_effective_assets(w_re)

        w_dict = {tickers[i]: round(float(w_re[i]), 4) for i in range(n_assets)}
        pt = FrontierPoint(
            rank_index=k,
            expected_return_ann_pct=round(p_ret * 100.0, 2),
            volatility_ann_pct=round(p_vol * 100.0, 2),
            sharpe_ratio=round(p_sr, 3),
            effective_assets=round(eff_n, 2),
            weights=w_dict,
        )
        resampled_pts.append(pt)

    # 6. Extract Key Benchmark Portfolios
    # Max Sharpe
    classical_best_sr = max(classical_pts, key=lambda p: p.sharpe_ratio)
    resampled_best_sr = max(resampled_pts, key=lambda p: p.sharpe_ratio)

    # Min Variance
    classical_min_var = min(classical_pts, key=lambda p: p.volatility_ann_pct)
    resampled_min_var = min(resampled_pts, key=lambda p: p.volatility_ann_pct)

    # Michaud diversification gain: % improvement in Effective Number of Assets at Max Sharpe
    eff_classical = classical_best_sr.effective_assets
    eff_resampled = resampled_best_sr.effective_assets
    div_gain_pct = (
        round(((eff_resampled - eff_classical) / max(eff_classical, 1e-4)) * 100.0, 2)
        if eff_classical > 0
        else 0.0
    )

    summary = (
        f"Michaud Resampled Efficient Frontier ({len(tickers)} assets, {t_obs} sessions, "
        f"{num_simulations} paths). Resampled Max Sharpe = {resampled_best_sr.sharpe_ratio:.3f} "
        f"(Vol: {resampled_best_sr.volatility_ann_pct:.1f}%, Ret: {resampled_best_sr.expected_return_ann_pct:.1f}%) "
        f"vs Classical Max Sharpe = {classical_best_sr.sharpe_ratio:.3f}. "
        f"Diversification Gain = +{div_gain_pct:.1f}% (N_eff: {eff_classical:.1f} -> {eff_resampled:.1f})."
    )

    receipt = ResampledFrontierReceipt(
        tickers=tickers,
        sample_sessions=t_obs,
        num_simulations=num_simulations,
        num_frontier_points=num_frontier_points,
        risk_free_rate_pct=round(risk_free_rate * 100.0, 2),
        shrinkage_intensity=lw_receipt.shrinkage_intensity,
        classical_max_sharpe=classical_best_sr,
        resampled_max_sharpe=resampled_best_sr,
        classical_min_variance=classical_min_var,
        resampled_min_variance=resampled_min_var,
        classical_frontier=classical_pts,
        resampled_frontier=resampled_pts,
        michaud_diversification_gain_pct=div_gain_pct,
        weight_stability_dispersion=dispersion_dict,
        summary=summary,
    )
    return receipt


def evaluate_resampled_frontier_profile(
    tickers: List[str],
    db_path: Optional[Path] = None,
    num_simulations: int = 100,
    num_frontier_points: int = 15,
    risk_free_rate: float = 0.04,
) -> ResampledFrontierReceipt:
    """High-level orchestrator: loads data, aligns returns, and computes resampled frontier."""
    cleaned_tickers = [validate_ticker(t.strip().upper()) for t in tickers if t.strip()]
    if len(cleaned_tickers) < 2:
        raise ValueError("At least 2 validated ticker symbols are required for portfolio frontier analysis.")

    returns_df = build_aligned_returns_matrix(cleaned_tickers, db_path=db_path, min_sessions=30)
    if returns_df.empty or len(returns_df) < 20:
        raise ValueError(
            f"Insufficient synchronized session observations for tickers: {cleaned_tickers}. "
            "Ensure market quotes are ingested."
        )

    return compute_resampled_efficient_frontier(
        returns_df=returns_df,
        num_simulations=num_simulations,
        num_frontier_points=num_frontier_points,
        risk_free_rate=risk_free_rate,
    )
