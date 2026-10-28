"""Bayesian Online Change-Point Detection (BOCPD) and Structural Break Engine for NextOmni Stock.

Implements the exact online Bayesian change-point detection algorithm (Adams & MacKay 2007)
using conjugate Normal-Inverse-Gamma priors and Student-t predictive distributions:
- Recursively estimates the posterior distribution of run-length r_t (sessions elapsed since last break).
- Detects discrete regime breaks when posterior run-length collapses to zero.
- Emits real-time change-point probabilities and structural stability alerts.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import PRICE_COLUMN


@dataclass
class ChangePointReceipt:
    """Comprehensive diagnostic receipt from Bayesian change-point detection."""
    ticker: str
    sample_sessions: int
    hazard_lambda: int
    current_run_length: int
    current_change_point_prob: float
    detected_break_dates: List[str]
    detected_break_indices: List[int]
    regime_stability_status: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_bocpd_series(
    returns: Union[np.ndarray, pd.Series, List[float]],
    hazard_lambda: int = 100,
    max_run_length: int = 250,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Compute recursive Bayesian Online Change-Point Detection across return series.

    Args:
        returns: 1D array of percentage or log returns.
        hazard_lambda: Expected timescale / mean run-length between structural breaks (default: 100 sessions).
        max_run_length: Maximum tracked run length to maintain bounded O(1) memory per step.

    Returns:
        Tuple of (mode_run_lengths, change_point_probs, detected_change_point_indices).
    """
    arr = np.asarray(returns, dtype=float)
    clean_mask = np.isfinite(arr)
    data = arr[clean_mask]
    n = len(data)

    if n < 5:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=float), []

    hazard = 1.0 / max(hazard_lambda, 5)

    # Initial prior hyperparameters from burn-in observations
    burn_in = min(n, 20)
    mu0 = 0.0
    var0 = float(np.var(data[:burn_in]))
    if var0 <= 1e-8:
        var0 = 1e-4

    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = alpha0 * var0

    # Sufficient statistics vectors
    mu_T = np.array([mu0])
    kappa_T = np.array([kappa0])
    alpha_T = np.array([alpha0])
    beta_T = np.array([beta0])

    # R represents probability distribution of run lengths at current step
    R = np.array([1.0])

    mode_run_lengths: List[int] = []
    change_point_probs: List[float] = []
    detected_indices: List[int] = []

    for t in range(n):
        x = data[t]
        n_r = len(R)

        # Student-t predictive distribution: df = 2*alpha, scale = sqrt(beta*(kappa+1)/(alpha*kappa))
        df = 2.0 * alpha_T
        scale = np.sqrt(np.maximum(1e-12, beta_T * (kappa_T + 1.0) / (alpha_T * kappa_T)))
        pred_probs = stats.t.pdf(x, df=df, loc=mu_T, scale=scale)
        pred_probs = np.maximum(pred_probs, 1e-25)

        # 1. Growth probability: r_t = r_{t-1} + 1
        growth = R * pred_probs * (1.0 - hazard)

        # 2. Change-point probability: r_t = 0 (hazard reset)
        cp_prob = float(np.sum(R * pred_probs * hazard))

        # Form next run-length distribution vector
        R_next = np.empty(n_r + 1)
        R_next[0] = cp_prob
        R_next[1:] = growth

        # Normalize posterior distribution
        evidence = np.sum(R_next)
        if evidence > 0:
            R_next /= evidence

        # Truncate to max_run_length to maintain bounded O(1) memory
        if len(R_next) > max_run_length:
            R_next = R_next[:max_run_length]
            R_next /= np.sum(R_next)

        post = R_next
        mode_r = int(np.argmax(post))
        mode_run_lengths.append(mode_r)

        # Recent change-point probability mass (run length <= 3)
        cp_mass = float(np.sum(post[: min(4, len(post))]))
        change_point_probs.append(round(cp_mass, 4))

        # Detect discrete structural break when run length collapses from >= 10 down to <= 2 with high probability mass
        if t > 0 and mode_run_lengths[t - 1] >= 10 and mode_r <= 2 and cp_mass > 0.40:
            detected_indices.append(t)

        # Update sufficient statistics
        new_kappa = kappa_T + 1.0
        new_mu = (kappa_T * mu_T + x) / new_kappa
        new_alpha = alpha_T + 0.5
        new_beta = beta_T + 0.5 * (kappa_T * (x - mu_T) ** 2) / new_kappa

        mu_T = np.concatenate([[mu0], new_mu])
        kappa_T = np.concatenate([[kappa0], new_kappa])
        alpha_T = np.concatenate([[alpha0], new_alpha])
        beta_T = np.concatenate([[beta0], new_beta])

        if len(mu_T) > max_run_length:
            mu_T = mu_T[:max_run_length]
            kappa_T = kappa_T[:max_run_length]
            alpha_T = alpha_T[:max_run_length]
            beta_T = beta_T[:max_run_length]

        R = R_next

    return np.array(mode_run_lengths), np.array(change_point_probs), detected_indices


def detect_bayesian_changepoints(
    df: pd.DataFrame,
    ticker: str = "TICKER",
    hazard_lambda: int = 100,
    price_col: str = PRICE_COLUMN,
) -> ChangePointReceipt:
    """Analyze historical price observations to detect structural change points and regime breaks.

    Args:
        df: Historical market observations DataFrame.
        ticker: Asset ticker symbol.
        hazard_lambda: Expected timescale between breaks (default: 100 sessions).
        price_col: Column name representing settlement prices (default: adj_close).

    Returns:
        ChangePointReceipt containing latest break probabilities and historical event dates.
    """
    if df.empty or len(df) < 15:
        return ChangePointReceipt(
            ticker=ticker,
            sample_sessions=len(df),
            hazard_lambda=hazard_lambda,
            current_run_length=0,
            current_change_point_prob=0.0,
            detected_break_dates=[],
            detected_break_indices=[],
            regime_stability_status="INSUFFICIENT_DATA",
            summary=f"Insufficient observations (sample_size={len(df)} < 15) for Bayesian change-point detection.",
        )

    clean_df = df.copy().sort_values("session_date", ascending=True).reset_index(drop=True)
    p_col = price_col if price_col in clean_df.columns else "close"
    if p_col not in clean_df.columns:
        p_col = clean_df.columns[1]

    prices = pd.to_numeric(clean_df[p_col], errors="coerce").dropna()
    returns = prices.pct_change().dropna().values
    n = len(returns)

    mode_r, cp_probs, break_indices = compute_bocpd_series(
        returns, hazard_lambda=hazard_lambda
    )

    current_r = int(mode_r[-1]) if len(mode_r) > 0 else 0
    current_p = float(cp_probs[-1]) if len(cp_probs) > 0 else 0.0

    # Map break indices back to calendar session dates
    dates = clean_df["session_date"].values if "session_date" in clean_df.columns else []
    break_dates: List[str] = []
    for idx in break_indices:
        # idx in returns corresponds to session idx+1 in clean_df
        d_idx = min(idx + 1, len(clean_df) - 1)
        if len(dates) > d_idx:
            break_dates.append(str(dates[d_idx]))
        else:
            break_dates.append(f"session_{idx}")

    # Determine status
    if current_p > 0.40 or current_r <= 2:
        status = "STRUCTURAL_BREAK_DETECTED"
    elif current_p > 0.20 or current_r <= 5:
        status = "REGIME_UNCERTAINTY"
    else:
        status = "STABLE_PERSISTENCE"

    summary = (
        f"Analyzed {n} daily sessions (hazard lambda={hazard_lambda}). "
        f"Current Run-Length: {current_r} sessions. Break Probability: {current_p * 100:.1f}%. "
        f"Historical Structural Breaks Detected: {len(break_dates)} ({', '.join(break_dates[-3:]) if break_dates else 'none'}). "
        f"Regime Status: {status}."
    )

    return ChangePointReceipt(
        ticker=ticker,
        sample_sessions=n,
        hazard_lambda=hazard_lambda,
        current_run_length=current_r,
        current_change_point_prob=round(current_p, 4),
        detected_break_dates=break_dates,
        detected_break_indices=break_indices,
        regime_stability_status=status,
        summary=summary,
    )
