"""Hamilton (1989) Two-State Markov-Switching Dynamic Regime Engine.

Implements James D. Hamilton's (1989) two-state Markov-switching model with
Expectation-Maximization (EM) parameter estimation, Hamilton forward filter,
and Kim (1994) backward smoother to infer latent volatility regime transitions.

References:
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. Econometrica, 57(2), 357-384.
    Kim, C. J. (1994). Dynamic linear models with Markov-switching.
    Journal of Econometrics, 60(1-2), 1-22.
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

from src.data.db import get_observations
from src.security.validators import assert_finite_number, validate_ticker

MARKOV_SIGNING_KEY = b"nextomni_markov_regime_audit_key_v1"


@dataclass
class MarkovRegimeReceipt:
    """Cryptographic audit receipt for two-state Markov-switching regime estimation."""

    ticker: str
    sample_sessions: int
    rolling_window: int
    current_regime: str  # LOW_VOLATILITY_EXPANSION or HIGH_VOLATILITY_CRISIS
    current_crisis_probability: float  # P(S_T = 1 | F_T)
    p00_persistence: float  # P(S_t = 0 | S_{t-1} = 0)
    p11_persistence: float  # P(S_t = 1 | S_{t-1} = 1)
    expected_low_vol_duration_sessions: float  # 1 / (1 - p00)
    expected_high_vol_duration_sessions: float  # 1 / (1 - p11)
    state_0_mean_ann_pct: float
    state_0_volatility_ann_pct: float
    state_1_mean_ann_pct: float
    state_1_volatility_ann_pct: float
    volatility_ratio_state1_to_state0: float
    log_likelihood: float
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core regime metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.rolling_window}:"
            f"{self.current_regime}:{self.current_crisis_probability:.4f}:"
            f"{self.p00_persistence:.4f}:{self.p11_persistence:.4f}:"
            f"{self.state_0_volatility_ann_pct:.4f}:{self.state_1_volatility_ann_pct:.4f}"
        )
        return hmac.new(MARKOV_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gaussian_density(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """Compute Gaussian probability density with numerical floor."""
    v = max(var, 1e-8)
    diff = x - mu
    denom = math.sqrt(2.0 * math.pi * v)
    return np.exp(-0.5 * (diff ** 2) / v) / denom


def run_hamilton_filter(
    returns: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    trans_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute Hamilton forward filter recursion.

    Args:
        returns: 1D array of daily returns (T,).
        mu: State means [mu_0, mu_1].
        var: State variances [var_0, var_1].
        trans_matrix: 2x2 transition matrix P where P[i, j] = P(S_t = j | S_{t-1} = i).

    Returns:
        Tuple of (filtered_probs (T, 2), predicted_probs (T, 2), total_log_likelihood).
    """
    t_len = len(returns)
    filtered = np.zeros((t_len, 2))
    predicted = np.zeros((t_len, 2))
    total_log_lik = 0.0

    # Ergodic stationary distribution as t=0 prior: pi * P = pi
    p00 = trans_matrix[0, 0]
    p11 = trans_matrix[1, 1]
    denom = (2.0 - p00 - p11)
    if denom > 1e-6:
        prior = np.array([(1.0 - p11) / denom, (1.0 - p00) / denom])
    else:
        prior = np.array([0.5, 0.5])

    curr_filtered = prior

    for t in range(t_len):
        # 1. Prediction step: pi_{t|t-1} = P^T @ pi_{t-1|t-1}
        pi_pred = trans_matrix.T @ curr_filtered
        pi_pred = np.maximum(pi_pred, 1e-12)
        pi_pred /= np.sum(pi_pred)
        predicted[t] = pi_pred

        # 2. Emission densities
        eta = np.array([
            _gaussian_density(returns[t], mu[0], var[0]),
            _gaussian_density(returns[t], mu[1], var[1]),
        ])
        eta = np.maximum(eta, 1e-15)

        # 3. Update step: pi_{t|t} = (pi_{t|t-1} * eta) / marginal_lik
        joint = pi_pred * eta
        marginal_lik = float(np.sum(joint))
        if marginal_lik <= 1e-15:
            marginal_lik = 1e-15

        curr_filtered = joint / marginal_lik
        curr_filtered /= np.sum(curr_filtered)
        filtered[t] = curr_filtered

        total_log_lik += math.log(marginal_lik)

    return filtered, predicted, float(total_log_lik)


def run_kim_smoother(
    filtered: np.ndarray,
    predicted: np.ndarray,
    trans_matrix: np.ndarray,
) -> np.ndarray:
    """Execute Kim (1994) backward smoothing recursion for full-sample smoothed state probabilities."""
    t_len = len(filtered)
    smoothed = np.zeros_like(filtered)
    smoothed[-1] = filtered[-1]

    for t in range(t_len - 2, -1, -1):
        for s in range(2):
            term = 0.0
            for j in range(2):
                p_trans = trans_matrix[s, j]
                p_pred = predicted[t + 1, j]
                ratio = smoothed[t + 1, j] / max(p_pred, 1e-12)
                term += p_trans * ratio
            smoothed[t, s] = filtered[t, s] * term

        # Normalize
        s_sum = np.sum(smoothed[t])
        if s_sum > 0:
            smoothed[t] /= s_sum
        else:
            smoothed[t] = np.array([0.5, 0.5])

    return smoothed


def fit_two_state_markov_switching(
    returns: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a two-state Markov-switching model via Expectation-Maximization (EM).

    Args:
        returns: 1D array of daily asset returns.
        max_iter: Maximum EM iterations (default: 50).
        tol: Log-likelihood convergence tolerance.

    Returns:
        Tuple of (mu, var, trans_matrix, filtered_probs, smoothed_probs, log_likelihood).
    """
    clean_ret = np.asarray(returns, dtype=float)
    clean_ret = clean_ret[np.isfinite(clean_ret)]
    n = len(clean_ret)
    if n < 30:
        raise ValueError("Insufficient returns observations for Markov-switching model (minimum 30 required).")

    # Smart initialization based on split volatility
    overall_mean = float(np.mean(clean_ret))
    overall_std = float(np.std(clean_ret))

    # State 0: Low Volatility (std * 0.7)
    # State 1: High Volatility (std * 1.6)
    mu = np.array([overall_mean, overall_mean])
    var = np.array([(overall_std * 0.7) ** 2, (overall_std * 1.6) ** 2])
    var = np.maximum(var, 1e-8)

    trans_matrix = np.array([
        [0.95, 0.05],
        [0.10, 0.90],
    ])

    prev_log_lik = -float("inf")
    filtered = np.zeros((n, 2))
    smoothed = np.zeros((n, 2))

    for iteration in range(max_iter):
        # E-Step: Forward Hamilton filter & Kim smoother
        filtered, predicted, log_lik = run_hamilton_filter(clean_ret, mu, var, trans_matrix)
        smoothed = run_kim_smoother(filtered, predicted, trans_matrix)

        if math.isnan(log_lik) or math.isinf(log_lik):
            break

        if abs(log_lik - prev_log_lik) < tol:
            break
        prev_log_lik = log_lik

        # M-Step: Update parameters from smoothed posterior weights
        w0 = smoothed[:, 0]
        w1 = smoothed[:, 1]
        sum_w0 = float(np.sum(w0))
        sum_w1 = float(np.sum(w1))

        if sum_w0 > 1e-4:
            mu[0] = float(np.sum(w0 * clean_ret) / sum_w0)
            var[0] = float(np.sum(w0 * ((clean_ret - mu[0]) ** 2)) / sum_w0)
        if sum_w1 > 1e-4:
            mu[1] = float(np.sum(w1 * clean_ret) / sum_w1)
            var[1] = float(np.sum(w1 * ((clean_ret - mu[1]) ** 2)) / sum_w1)

        var = np.maximum(var, 1e-8)

        # Update transition probabilities
        p00_numer = float(np.sum(smoothed[:-1, 0] * (trans_matrix[0, 0] * smoothed[1:, 0] / np.maximum(predicted[1:, 0], 1e-12))))
        p11_numer = float(np.sum(smoothed[:-1, 1] * (trans_matrix[1, 1] * smoothed[1:, 1] / np.maximum(predicted[1:, 1], 1e-12))))

        p00 = max(0.50, min(p00_numer / max(sum_w0, 1e-4), 0.995))
        p11 = max(0.50, min(p11_numer / max(sum_w1, 1e-4), 0.995))

        trans_matrix = np.array([
            [p00, 1.0 - p00],
            [1.0 - p11, p11],
        ])

    # Enforce standard identification: State 0 is Low Volatility, State 1 is High Volatility
    if var[0] > var[1]:
        mu = mu[::-1]
        var = var[::-1]
        trans_matrix = np.array([
            [trans_matrix[1, 1], trans_matrix[1, 0]],
            [trans_matrix[0, 1], trans_matrix[0, 0]],
        ])
        filtered = filtered[:, ::-1]
        smoothed = smoothed[:, ::-1]

    return mu, var, trans_matrix, filtered, smoothed, prev_log_lik


def evaluate_markov_regime_profile(
    ticker: str,
    prices: pd.Series,
    window: int = 120,
) -> MarkovRegimeReceipt:
    """Evaluate two-state Markov-switching dynamic regime model for an asset.

    Args:
        ticker: Validated stock ticker symbol.
        prices: Series of asset closing prices.
        window: Regression observation window (sessions).

    Returns:
        MarkovRegimeReceipt with regime probabilities, expected durations, and cryptographic HMAC signature.
    """
    valid_ticker = validate_ticker(ticker.strip().upper())
    window = max(40, min(window, 504))

    clean_p = prices.dropna().astype(float)
    if len(clean_p) < 40:
        raise ValueError(f"Insufficient price history for {valid_ticker} (minimum 40 required).")

    # Use window tail
    tail_p = clean_p.iloc[-window:]
    returns = tail_p.pct_change().dropna().values
    sample_size = len(returns)

    mu, var, trans_mat, filtered, smoothed, log_lik = fit_two_state_markov_switching(returns)

    # Annualize moments
    mu_ann_0 = float(mu[0] * 252.0 * 100.0)
    vol_ann_0 = float(math.sqrt(var[0] * 252.0) * 100.0)
    mu_ann_1 = float(mu[1] * 252.0 * 100.0)
    vol_ann_1 = float(math.sqrt(var[1] * 252.0) * 100.0)
    vol_ratio = float(round(vol_ann_1 / max(vol_ann_0, 1e-4), 2))

    # Transition parameters
    p00 = float(trans_mat[0, 0])
    p11 = float(trans_mat[1, 1])

    # Expected durations: 1 / (1 - p)
    dur_0 = float(round(1.0 / max(1.0 - p00, 0.005), 1))
    dur_1 = float(round(1.0 / max(1.0 - p11, 0.005), 1))

    # Current filtered state probability at t=T
    current_crisis_prob = float(filtered[-1, 1])
    regime = (
        "HIGH_VOLATILITY_CRISIS"
        if current_crisis_prob >= 0.50
        else "LOW_VOLATILITY_EXPANSION"
    )

    summary = (
        f"Hamilton Two-State Markov Regime for {valid_ticker} ({sample_size} sessions, window: {window}D): "
        f"Current State: {regime} (P(Crisis) = {current_crisis_prob * 100:.1f}%). "
        f"Low-Vol State: {vol_ann_0:.1f}% Ann Vol, Expected Duration: {dur_0:.1f} sessions (p00: {p00:.3f}). "
        f"High-Vol Crisis State: {vol_ann_1:.1f}% Ann Vol ({vol_ratio:.1f}x vol), Expected Duration: {dur_1:.1f} sessions (p11: {p11:.3f})."
    )

    receipt = MarkovRegimeReceipt(
        ticker=valid_ticker,
        sample_sessions=sample_size,
        rolling_window=window,
        current_regime=regime,
        current_crisis_probability=round(current_crisis_prob, 4),
        p00_persistence=round(p00, 4),
        p11_persistence=round(p11, 4),
        expected_low_vol_duration_sessions=dur_0,
        expected_high_vol_duration_sessions=dur_1,
        state_0_mean_ann_pct=round(mu_ann_0, 2),
        state_0_volatility_ann_pct=round(vol_ann_0, 2),
        state_1_mean_ann_pct=round(mu_ann_1, 2),
        state_1_volatility_ann_pct=round(vol_ann_1, 2),
        volatility_ratio_state1_to_state0=vol_ratio,
        log_likelihood=round(log_lik, 2),
        summary=summary,
    )
    return receipt
