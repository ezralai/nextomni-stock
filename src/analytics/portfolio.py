"""Quantitative portfolio optimization and Ledoit-Wolf shrinkage covariance engine for NextOmni Stock.

Implements analytical Ledoit-Wolf shrinkage covariance matrix estimation,
bounded Markowitz optimization (Max Sharpe, Min Variance, Equal Risk Contribution),
and marginal asset risk attribution across multi-asset forecast universes.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize
import scipy.spatial.distance as ssd
from sklearn.covariance import LedoitWolf

from src.analytics.screener import ScreenerRecord, screen_asset_universe
from src.config import PRICE_COLUMN
from src.data.db import get_observations
from src.security.validators import assert_finite_number, validate_exchange



@dataclass
class LedoitWolfReceipt:
    """Diagnostic receipt from analytical Ledoit-Wolf shrinkage covariance estimation."""
    tickers: List[str]
    sample_size: int
    shrinkage_intensity: float
    sample_condition_number: float
    shrunk_condition_number: float
    shrunk_covariance: pd.DataFrame
    shrunk_correlation: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shrunk_covariance"] = self.shrunk_covariance.to_dict()
        d["shrunk_correlation"] = self.shrunk_correlation.to_dict()
        return d


@dataclass
class PortfolioReceipt:
    """Comprehensive portfolio allocation receipt."""
    mode: str
    tickers: List[str]
    weights: Dict[str, float]
    expected_annual_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    shrinkage_intensity: float
    risk_contributions_pct: Dict[str, float]
    effective_number_of_assets: float  # Herfindahl inverse: 1 / sum(w_i^2)
    horizon_sessions: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_aligned_returns_matrix(
    tickers: List[str],
    db_path: Optional[Path] = None,
    min_sessions: int = 40,
) -> pd.DataFrame:
    """Construct a synchronized, date-aligned daily return matrix across tickers.

    Args:
        tickers: List of validated stock ticker symbols.
        db_path: Optional SQLite database path.
        min_sessions: Minimum overlapping session count required.

    Returns:
        DataFrame of aligned percentage daily returns indexed by session date.
    """
    returns_dict: Dict[str, pd.Series] = {}

    for ticker in tickers:
        obs_df = get_observations(ticker, db_path=db_path)
        if obs_df.empty or len(obs_df) < min_sessions:
            continue
        clean = obs_df.sort_values("session_date").drop_duplicates(subset=["session_date"])
        prices = clean[PRICE_COLUMN].astype(float)
        ret = prices.pct_change().dropna()
        ret.index = clean["session_date"].iloc[1:]
        returns_dict[ticker] = ret

    if not returns_dict:
        return pd.DataFrame()

    returns_df = pd.DataFrame(returns_dict).dropna(how="any")
    return returns_df


def compute_ledoit_wolf_covariance(returns_df: pd.DataFrame) -> LedoitWolfReceipt:
    """Compute analytical Ledoit-Wolf shrinkage covariance matrix and correlation.

    Shrinks sample covariance S toward constant correlation / identity target F:
        Sigma* = delta * F + (1 - delta) * S

    Args:
        returns_df: Date-aligned DataFrame of daily returns for N assets.

    Returns:
        LedoitWolfReceipt with shrunk covariance, correlation, and condition metrics.
    """
    tickers = list(returns_df.columns)
    n_obs, n_assets = returns_df.shape

    if n_assets == 0 or n_obs < 5:
        empty_mat = pd.DataFrame()
        return LedoitWolfReceipt(
            tickers=tickers,
            sample_size=n_obs,
            shrinkage_intensity=0.0,
            sample_condition_number=1.0,
            shrunk_condition_number=1.0,
            shrunk_covariance=empty_mat,
            shrunk_correlation=empty_mat,
        )

    # Analytical Ledoit-Wolf shrinkage
    lw = LedoitWolf()
    lw.fit(returns_df.values)
    shrunk_cov = lw.covariance_
    shrinkage = float(lw.shrinkage_)

    # Sample covariance for condition comparison
    sample_cov = np.cov(returns_df.values, rowvar=False)
    if n_assets == 1:
        sample_cov = np.array([[float(sample_cov)]])
        shrunk_cov = np.array([[float(shrunk_cov)]])

    try:
        sample_cond = float(np.linalg.cond(sample_cov))
    except Exception:
        sample_cond = 1e6

    try:
        shrunk_cond = float(np.linalg.cond(shrunk_cov))
    except Exception:
        shrunk_cond = 1.0

    # Derive correlation matrix from shrunk covariance
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(shrunk_cov), 1e-12)))
    shrunk_corr = d_inv_sqrt @ shrunk_cov @ d_inv_sqrt
    np.fill_diagonal(shrunk_corr, 1.0)

    cov_df = pd.DataFrame(shrunk_cov, index=tickers, columns=tickers)
    corr_df = pd.DataFrame(shrunk_corr, index=tickers, columns=tickers)

    return LedoitWolfReceipt(
        tickers=tickers,
        sample_size=n_obs,
        shrinkage_intensity=round(shrinkage, 4),
        sample_condition_number=round(sample_cond, 2),
        shrunk_condition_number=round(shrunk_cond, 2),
        shrunk_covariance=cov_df,
        shrunk_correlation=corr_df,
    )


def calculate_portfolio_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """Calculate percentage marginal risk contribution of each asset to portfolio risk.

    RC_i = w_i * (Sigma * w)_i / sigma_p^2
    sum(RC_i) = 1.0

    Args:
        weights: 1D array of portfolio weights (sums to 1.0).
        cov_matrix: 2D covariance matrix.

    Returns:
        1D array of risk contributions summing to 1.0.
    """
    port_var = float(weights.T @ cov_matrix @ weights)
    if port_var <= 1e-12:
        n = len(weights)
        return np.ones(n) / n

    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib / port_var
    return risk_contrib


def compute_hrp_weights(
    cov_matrix: np.ndarray,
    corr_matrix: Optional[np.ndarray] = None,
    max_weight: Optional[float] = None,
) -> np.ndarray:
    """Compute Hierarchical Risk Parity (HRP) weights via tree clustering and recursive bisection.

    Follows Marcos López de Prado (2016):
    1. Distance metric: D_ij = sqrt(0.5 * (1 - rho_ij))
    2. Single-linkage hierarchical agglomerative tree clustering
    3. Quasi-diagonalization to order leaves along cluster hierarchy
    4. Top-down recursive bisection allocating weights inversely proportional to cluster variance

    Args:
        cov_matrix: 2D covariance matrix (n x n).
        corr_matrix: Optional 2D correlation matrix (n x n). Computed from cov if None.
        max_weight: Optional maximum allowable weight per asset. Excess is iteratively redistributed.

    Returns:
        1D array of portfolio weights (n elements) summing strictly to 1.0.
    """
    n = len(cov_matrix)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    if corr_matrix is None:
        std = np.sqrt(np.maximum(1e-12, np.diag(cov_matrix)))
        corr_matrix = cov_matrix / np.outer(std, std)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)

    # 1. Tree clustering distance
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr_matrix)))
    np.fill_diagonal(dist, 0.0)
    condensed = ssd.squareform(dist, checks=False)
    link = sch.linkage(condensed, method="single")

    # 2. Quasi-diagonal leaf order
    ordered_leaves = sch.leaves_list(link).tolist()

    # 3. Recursive bisection
    def _get_cluster_var(cluster_indices: List[int]) -> float:
        sub_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
        diag_inv = 1.0 / np.maximum(1e-12, np.diag(sub_cov))
        inv_w = diag_inv / np.sum(diag_inv)
        return float(inv_w.T @ sub_cov @ inv_w)

    weights_by_idx = {i: 1.0 for i in ordered_leaves}
    active_clusters = [ordered_leaves]

    while active_clusters:
        active_clusters = [c for c in active_clusters if len(c) > 1]
        next_clusters = []
        for c in active_clusters:
            mid = len(c) // 2
            c1 = c[:mid]
            c2 = c[mid:]
            v1 = _get_cluster_var(c1)
            v2 = _get_cluster_var(c2)
            total_v = max(v1 + v2, 1e-12)
            alpha1 = v2 / total_v
            alpha2 = 1.0 - alpha1
            for idx in c1:
                weights_by_idx[idx] *= alpha1
            for idx in c2:
                weights_by_idx[idx] *= alpha2
            if len(c1) > 1:
                next_clusters.append(c1)
            if len(c2) > 1:
                next_clusters.append(c2)
        active_clusters = next_clusters

    raw_weights = np.array([weights_by_idx[i] for i in range(n)])
    raw_weights = np.maximum(0.0, raw_weights)
    total_w = np.sum(raw_weights)
    w = raw_weights / total_w if total_w > 0 else np.ones(n) / n

    # Apply upper bound constraint with iterative redistribution if specified
    if max_weight is not None and max_weight > 0:
        upper_bound = max(1.0 / n, max_weight)
        for _ in range(25):
            excess = np.sum(np.maximum(0.0, w - upper_bound))
            if excess <= 1e-8:
                break
            w = np.minimum(w, upper_bound)
            unconstrained = w < upper_bound - 1e-8
            if np.any(unconstrained):
                w[unconstrained] += excess * (w[unconstrained] / np.sum(w[unconstrained]))
            else:
                break
        w = np.maximum(0.0, w)
        w /= np.sum(w)

    return w


def optimize_alpha_portfolio(
    screener_records: List[ScreenerRecord],
    returns_df: Optional[pd.DataFrame] = None,
    mode: str = "max_sharpe",
    risk_free_rate: float = 0.04,
    max_weight: float = 0.40,
    db_path: Optional[Path] = None,
) -> PortfolioReceipt:
    """Optimize multi-asset portfolio weights combining alpha forecasts and shrunk covariance.

    Args:
        screener_records: List of ScreenerRecord objects from screen_asset_universe.
        returns_df: Optional precomputed aligned returns DataFrame.
        mode: Allocation mode ('max_sharpe', 'min_variance', 'risk_parity', 'equal_weight', 'hrp').
        risk_free_rate: Annualized risk-free benchmark rate (default: 0.04).
        max_weight: Maximum allowable weight per asset in long-only allocation.
        db_path: Optional SQLite database path.

    Returns:
        PortfolioReceipt containing optimal weights, risk/return metrics, and diagnostics.
    """
    valid_modes = {"max_sharpe", "min_variance", "risk_parity", "equal_weight", "hrp"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {valid_modes}")


    if not screener_records:
        raise ValueError("screener_records cannot be empty.")

    tickers = [r.ticker for r in screener_records]
    horizon = screener_records[0].horizon_sessions
    annual_factor = 252.0 / max(horizon, 1)

    # Expected annual return vector (mu)
    exp_returns = np.array([r.expected_return_pct / 100.0 * annual_factor for r in screener_records])

    # If single asset, trivial allocation
    if len(tickers) == 1:
        t = tickers[0]
        obs = get_observations(t, db_path=db_path)
        vol = 0.20
        if not obs.empty and len(obs) > 10:
            vol = float(obs[PRICE_COLUMN].pct_change().std() * math.sqrt(252))
        vol = max(vol, 0.01)
        ret_pct = float(exp_returns[0] * 100.0)
        sharpe = (exp_returns[0] - risk_free_rate) / vol
        return PortfolioReceipt(
            mode=mode,
            tickers=[t],
            weights={t: 1.0},
            expected_annual_return_pct=round(ret_pct, 2),
            annualized_volatility_pct=round(vol * 100.0, 2),
            sharpe_ratio=round(sharpe, 3),
            shrinkage_intensity=0.0,
            risk_contributions_pct={t: 100.0},
            effective_number_of_assets=1.0,
            horizon_sessions=horizon,
        )

    # Build aligned returns matrix if not supplied
    if returns_df is None or returns_df.empty:
        returns_df = build_aligned_returns_matrix(tickers, db_path=db_path)

    # Common tickers available in returns_df
    common_tickers = [t for t in tickers if t in returns_df.columns]
    if len(common_tickers) < 2:
        # Fall back to equal weight on requested tickers
        n = len(tickers)
        eq_w = {t: round(1.0 / n, 4) for t in tickers}
        eq_ret = float(np.mean(exp_returns) * 100.0)
        return PortfolioReceipt(
            mode="equal_weight",
            tickers=tickers,
            weights=eq_w,
            expected_annual_return_pct=round(eq_ret, 2),
            annualized_volatility_pct=20.0,
            sharpe_ratio=round((eq_ret / 100.0 - risk_free_rate) / 0.20, 3),
            shrinkage_intensity=0.0,
            risk_contributions_pct={t: round(100.0 / n, 2) for t in tickers},
            effective_number_of_assets=float(n),
            horizon_sessions=horizon,
        )

    # Reindex expected returns to common tickers
    ticker_indices = [tickers.index(t) for t in common_tickers]
    sub_returns = returns_df[common_tickers]
    mu = exp_returns[ticker_indices]
    n = len(common_tickers)

    # Compute Ledoit-Wolf annualized covariance: Sigma_ann = Sigma_daily * 252
    lw_receipt = compute_ledoit_wolf_covariance(sub_returns)
    cov_daily = lw_receipt.shrunk_covariance.values
    cov_ann = cov_daily * 252.0

    # Bounds: w_i in [0, max(1/N, max_weight)]
    upper_bound = max(1.0 / n, max_weight)
    bounds = tuple((0.0, upper_bound) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.ones(n) / n

    if mode == "equal_weight":
        opt_weights = w0
    elif mode == "hrp":
        opt_weights = compute_hrp_weights(
            cov_matrix=cov_ann,
            corr_matrix=lw_receipt.shrunk_correlation.values,
            max_weight=upper_bound,
        )
    elif mode == "min_variance":

        def var_objective(w: np.ndarray) -> float:
            return float(w.T @ cov_ann @ w)

        res = minimize(var_objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = res.x if res.success else w0
    elif mode == "risk_parity":
        # Target equal risk contribution: 1/n for each asset
        def risk_parity_objective(w: np.ndarray) -> float:
            port_var = float(w.T @ cov_ann @ w)
            if port_var <= 1e-12:
                return 1e6
            marginal = cov_ann @ w
            rc = w * marginal / port_var
            target = 1.0 / n
            return float(np.sum((rc - target) ** 2))

        res = minimize(risk_parity_objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = res.x if res.success else w0
    else:  # max_sharpe
        def neg_sharpe(w: np.ndarray) -> float:
            p_ret = float(w.T @ mu)
            p_var = float(w.T @ cov_ann @ w)
            p_vol = math.sqrt(max(p_var, 1e-8))
            return -float((p_ret - risk_free_rate) / p_vol)

        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = res.x if res.success else w0

    # Clean and normalize weights
    opt_weights = np.maximum(opt_weights, 0.0)
    opt_weights /= np.sum(opt_weights)

    # Compute realized portfolio metrics
    port_exp_return = float(opt_weights.T @ mu)
    port_vol = float(math.sqrt(max(opt_weights.T @ cov_ann @ opt_weights, 1e-8)))
    sharpe = float((port_exp_return - risk_free_rate) / port_vol)
    risk_contribs = calculate_portfolio_risk_contributions(opt_weights, cov_ann)
    herfindahl = float(np.sum(opt_weights ** 2))
    effective_n = float(1.0 / herfindahl) if herfindahl > 0 else float(n)

    weights_dict = {t: round(float(opt_weights[i]), 4) for i, t in enumerate(common_tickers)}
    rc_dict = {t: round(float(risk_contribs[i] * 100.0), 2) for i, t in enumerate(common_tickers)}

    return PortfolioReceipt(
        mode=mode,
        tickers=common_tickers,
        weights=weights_dict,
        expected_annual_return_pct=round(port_exp_return * 100.0, 2),
        annualized_volatility_pct=round(port_vol * 100.0, 2),
        sharpe_ratio=round(sharpe, 3),
        shrinkage_intensity=lw_receipt.shrinkage_intensity,
        risk_contributions_pct=rc_dict,
        effective_number_of_assets=round(effective_n, 2),
        horizon_sessions=horizon,
    )
