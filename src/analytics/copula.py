"""Asymmetric Lower-Tail Dependence (lambda_L) & Joint Crash Fragility Engine for NextOmni Stock.

Implements non-parametric and parametric copula lower-tail dependence estimation:
- Empirical Probability Integral Transform (PIT) mapping continuous asset returns to uniform margins.
- Empirical lower-tail quantile co-exceedance: P(U_1 <= q, U_2 <= q) / q.
- Analytical Clayton copula lower-tail dependence: lambda_L = 2^(-1/theta) via Kendall's rank tau.
- Systemic joint crash fragility scoring across multi-asset investment universes.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.portfolio import build_aligned_returns_matrix
from src.data.db import get_all_stocks
from src.security.validators import validate_exchange


@dataclass
class TailDependenceReceipt:
    """Quantitative diagnostic receipt summarizing asymmetric lower-tail co-crash dependence."""
    tickers: List[str]
    sample_sessions: int
    tail_threshold: float
    tail_dependence_matrix: pd.DataFrame
    systemic_fragility_scores: Dict[str, float]
    highest_risk_pair: Tuple[str, str, float]
    market_joint_crash_regime: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        d = asdict(self)
        d["tail_dependence_matrix"] = self.tail_dependence_matrix.to_dict()
        return d


def compute_empirical_pit(
    series: Union[np.ndarray, pd.Series, List[float]],
) -> np.ndarray:
    """Map continuous returns to uniform margins U in (0, 1) via empirical rank transform.

    Formula:
        U_t = rank(r_t) / (T + 1)

    Args:
        series: Array or Series of historical asset returns.

    Returns:
        1D numpy array of uniform copula marginals in (0, 1).
    """
    arr = np.asarray(series, dtype=float)
    clean = arr[np.isfinite(arr)]
    n = len(clean)
    if n == 0:
        return np.array([], dtype=float)

    # Rank with average tie handling
    ranks = stats.rankdata(clean, method="average")
    u = ranks / (n + 1.0)
    return np.clip(u, 1e-6, 1.0 - 1e-6)


def calculate_bivariate_lower_tail_dependence(
    u1: np.ndarray,
    u2: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[float, float, float]:
    """Calculate empirical and parametric Clayton lower-tail dependence between uniform margins.

    Args:
        u1: Uniform copula marginals for asset 1 in (0, 1).
        u2: Uniform copula marginals for asset 2 in (0, 1).
        threshold: Tail quantile cutoff q (default: 0.05 for 5% extreme crash tail).

    Returns:
        Tuple of (hybrid_lambda, empirical_lambda, clayton_lambda).
    """
    n = min(len(u1), len(u2))
    if n < 10 or threshold <= 0.0 or threshold >= 1.0:
        return 0.0, 0.0, 0.0

    u1_c = u1[:n]
    u2_c = u2[:n]

    # Perfect correlation identity
    if np.allclose(u1_c, u2_c):
        return 1.0, 1.0, 1.0

    # 1. Empirical non-parametric co-exceedance: P(U1 <= q, U2 <= q) / q
    co_exceedances = np.sum((u1_c <= threshold) & (u2_c <= threshold))
    lambda_emp = float(co_exceedances / (n * threshold))
    lambda_emp = float(np.clip(lambda_emp, 0.0, 1.0))

    # 2. Clayton copula lower-tail dependence: lambda_L = 2^(-1/theta)
    # Kendall's tau rank correlation
    tau, _ = stats.kendalltau(u1_c, u2_c)
    if math.isnan(tau) or tau <= 1e-4:
        lambda_clayton = 0.0
    else:
        # Clayton parameter theta = 2 * tau / (1 - tau)
        theta = (2.0 * tau) / max(1e-4, 1.0 - tau)
        if theta > 1e-4:
            lambda_clayton = float(2.0 ** (-1.0 / theta))
        else:
            lambda_clayton = 0.0
    lambda_clayton = float(np.clip(lambda_clayton, 0.0, 1.0))

    # 3. Hybrid regularized estimate
    lambda_hybrid = 0.5 * lambda_emp + 0.5 * lambda_clayton
    return round(lambda_hybrid, 4), round(lambda_emp, 4), round(lambda_clayton, 4)


def compute_lower_tail_dependence_matrix(
    returns_df: pd.DataFrame,
    threshold: float = 0.05,
    method: str = "hybrid",
) -> pd.DataFrame:
    """Compute N x N symmetric lower-tail dependence matrix across date-aligned asset returns.

    Args:
        returns_df: Date-aligned DataFrame of daily asset returns.
        threshold: Tail quantile cutoff (default: 0.05).
        method: Estimation method ('hybrid', 'empirical', or 'clayton').

    Returns:
        Symmetric N x N DataFrame with diagonal = 1.0 and values in [0.0, 1.0].
    """
    tickers = list(returns_df.columns)
    p = len(tickers)
    mat = np.eye(p, dtype=float)

    if p < 2 or len(returns_df) < 10:
        return pd.DataFrame(mat, index=tickers, columns=tickers)

    # Compute empirical PIT for each asset
    u_dict = {col: compute_empirical_pit(returns_df[col].values) for col in tickers}

    for i in range(p):
        for j in range(i + 1, p):
            u_i = u_dict[tickers[i]]
            u_j = u_dict[tickers[j]]
            hyb, emp, cla = calculate_bivariate_lower_tail_dependence(
                u_i, u_j, threshold=threshold
            )

            if method == "empirical":
                val = emp
            elif method == "clayton":
                val = cla
            else:
                val = hyb

            mat[i, j] = val
            mat[j, i] = val

    return pd.DataFrame(mat, index=tickers, columns=tickers)


def evaluate_universe_tail_dependence(
    tickers: Optional[List[str]] = None,
    exchange: Optional[str] = None,
    threshold: float = 0.05,
    method: str = "hybrid",
    db_path: Optional[Path] = None,
) -> TailDependenceReceipt:
    """Evaluate multi-asset lower-tail dependence and compute systemic joint crash fragility.

    Args:
        tickers: Optional list of ticker symbols. If None, queries all active database assets.
        exchange: Optional exchange MIC filter (XNYS, XKLS).
        threshold: Tail quantile cutoff (default: 0.05).
        method: Lower tail estimator ('hybrid', 'empirical', or 'clayton').
        db_path: Optional SQLite database path.

    Returns:
        TailDependenceReceipt containing tail matrix, fragility scores, and systemic risk summary.
    """
    if exchange:
        exchange = validate_exchange(exchange)

    if not tickers:
        stocks_df = get_all_stocks(db_path=db_path)
        if exchange and not stocks_df.empty:
            stocks_df = stocks_df[stocks_df["exchange"] == exchange]
        tickers = stocks_df["ticker"].tolist() if not stocks_df.empty else []

    if len(tickers) < 2:
        empty_mat = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
        return TailDependenceReceipt(
            tickers=tickers,
            sample_sessions=0,
            tail_threshold=threshold,
            tail_dependence_matrix=empty_mat,
            systemic_fragility_scores={t: 0.0 for t in tickers},
            highest_risk_pair=("NONE", "NONE", 0.0),
            market_joint_crash_regime="INSUFFICIENT_ASSETS",
            summary=f"Insufficient multi-asset universe (n={len(tickers)} < 2) for tail dependence evaluation.",
        )

    returns_df = build_aligned_returns_matrix(tickers, db_path=db_path, min_sessions=20)
    aligned_tickers = list(returns_df.columns)
    p = len(aligned_tickers)
    n_sessions = len(returns_df)

    if p < 2 or n_sessions < 15:
        empty_mat = pd.DataFrame(np.eye(p), index=aligned_tickers, columns=aligned_tickers)
        return TailDependenceReceipt(
            tickers=aligned_tickers,
            sample_sessions=n_sessions,
            tail_threshold=threshold,
            tail_dependence_matrix=empty_mat,
            systemic_fragility_scores={t: 0.0 for t in aligned_tickers},
            highest_risk_pair=("NONE", "NONE", 0.0),
            market_joint_crash_regime="INSUFFICIENT_DATA",
            summary=f"Insufficient overlapping sessions ({n_sessions} < 15) across {p} assets.",
        )

    tail_mat = compute_lower_tail_dependence_matrix(
        returns_df, threshold=threshold, method=method
    )

    # Compute systemic fragility score (mean off-diagonal tail dependence per asset)
    fragility_scores: Dict[str, float] = {}
    for t in aligned_tickers:
        row_vals = [tail_mat.loc[t, other] for other in aligned_tickers if other != t]
        fragility_scores[t] = round(float(np.mean(row_vals)), 4) if row_vals else 0.0

    # Identify highest risk co-crash pair
    highest_pair = ("NONE", "NONE", 0.0)
    max_val = -1.0
    for i in range(p):
        for j in range(i + 1, p):
            val = float(tail_mat.iloc[i, j])
            if val > max_val:
                max_val = val
                highest_pair = (aligned_tickers[i], aligned_tickers[j], round(val, 4))

    # Market-wide off-diagonal average
    off_diag_vals = [
        tail_mat.iloc[i, j]
        for i in range(p)
        for j in range(i + 1, p)
    ]
    avg_tail_dep = float(np.mean(off_diag_vals)) if off_diag_vals else 0.0

    if avg_tail_dep >= 0.35:
        regime = "HIGH_CONTAGION_RISK"
    elif avg_tail_dep >= 0.15:
        regime = "MODERATE_TAIL_DEPENDENCE"
    else:
        regime = "LOW_CO_CRASH_RISK"

    summary = (
        f"Evaluated {p} assets ({n_sessions} overlapping sessions, tail threshold q={threshold}). "
        f"Average Market Lower-Tail Dependence: {avg_tail_dep * 100:.1f}%. "
        f"Highest Co-Crash Pair: {highest_pair[0]} & {highest_pair[1]} ({highest_pair[2] * 100:.1f}% joint tail dependence). "
        f"Regime: {regime}."
    )

    return TailDependenceReceipt(
        tickers=aligned_tickers,
        sample_sessions=n_sessions,
        tail_threshold=threshold,
        tail_dependence_matrix=tail_mat,
        systemic_fragility_scores=fragility_scores,
        highest_risk_pair=highest_pair,
        market_joint_crash_regime=regime,
        summary=summary,
    )
