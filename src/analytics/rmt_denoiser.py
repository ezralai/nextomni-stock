"""Marchenko-Pastur (1967) Random Matrix Theory (RMT) Covariance Denoising & Detrending Engine.

Implements Random Matrix Theory (RMT) covariance matrix denoising and market mode detrending
to eliminate in-sample noise eigenvalues and stabilize portfolio optimization, risk attribution,
and clustering algorithms.

References:
    Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets
    of random matrices. Sbornik: Mathematics, 1(4), 457-483.
    López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
    López de Prado, M. (2020). Machine Learning for Asset Managers. Cambridge University Press.
    Bouchaud, J. P., & Potters, M. (2009). Financial applications of random matrix theory:
    a short review. arXiv preprint arXiv:0910.1205.
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
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde

from src.analytics.portfolio import build_aligned_returns_matrix
from src.analytics.screener import screen_asset_universe
from src.security.validators import assert_finite_number, validate_exchange, validate_ticker

RMT_SIGNING_KEY = b"nextomni_rmt_denoiser_secret_key_v1"


@dataclass
class RmtDenoiserReceipt:
    """Cryptographic audit receipt for Random Matrix Theory covariance denoising."""

    tickers: List[str]
    sample_sessions: int
    num_assets: int
    q_ratio: float  # T / N
    noise_cutoff_lambda_plus: float  # Upper bound of Marchenko-Pastur null distribution
    noise_floor_lambda_minus: float  # Lower bound of Marchenko-Pastur null distribution
    fitted_noise_variance: float  # Empirical sigma^2
    num_signal_eigenvalues: int  # Eigenvalues exceeding lambda_plus
    num_noise_eigenvalues: int  # Eigenvalues within or below noise band
    signal_variance_ratio_pct: float  # Percentage of total variance in signal modes
    condition_number_raw: float  # Condition number of empirical correlation matrix
    condition_number_denoised: float  # Condition number of denoised correlation matrix
    condition_number_improvement_ratio: float  # Raw / Denoised
    method: str  # constant_residual or targeted_shrinkage
    is_detrended: bool  # Whether dominant market mode was removed
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core RMT metrics."""
        payload = (
            f"{','.join(self.tickers)}:{self.sample_sessions}:{self.num_assets}:"
            f"{self.q_ratio:.4f}:{self.noise_cutoff_lambda_plus:.4f}:"
            f"{self.num_signal_eigenvalues}:{self.condition_number_raw:.2f}:"
            f"{self.condition_number_denoised:.2f}:{self.method}:{self.is_detrended}"
        )
        return hmac.new(RMT_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_marchenko_pastur_pdf(
    var: float,
    q: float,
    pts: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Calculate theoretical Marchenko-Pastur probability density function.

    Formula:
        lambda_{\\pm} = \\sigma^2 (1 \\pm \\sqrt{1 / q})^2
        f(\\lambda) = \\frac{q}{2\\pi \\sigma^2 \\lambda} \\sqrt{(\\lambda_+ - \\lambda)(\\lambda - \\lambda_-)}

    Args:
        var: Variance of the random variables (\\sigma^2).
        q: Ratio of observations to dimensions (T / N >= 1.0).
        pts: Number of evaluation points across [lambda_-, lambda_+].

    Returns:
        Tuple of (lambda_domain, density_pdf, lambda_minus, lambda_plus).
    """
    assert_finite_number(var, "var")
    assert_finite_number(q, "q")
    if var <= 0.0:
        raise ValueError(f"Variance must be positive, got {var}")
    if q <= 0.0:
        raise ValueError(f"Q ratio must be positive, got {q}")

    inv_q_sqrt = math.sqrt(1.0 / q)
    lambda_minus = var * (1.0 - inv_q_sqrt) ** 2
    lambda_plus = var * (1.0 + inv_q_sqrt) ** 2

    # Generate domain across bounds
    lambdas = np.linspace(lambda_minus, lambda_plus, pts)
    # Density formula
    diff_plus = np.maximum(0.0, lambda_plus - lambdas)
    diff_minus = np.maximum(0.0, lambdas - lambda_minus)
    radicand = diff_plus * diff_minus

    safe_lambdas = np.maximum(lambdas, 1e-12)
    denom = 2.0 * math.pi * var * safe_lambdas
    pdf = (q / denom) * np.sqrt(radicand)
    # Handle zero domain points
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

    return lambdas, pdf, float(lambda_minus), float(lambda_plus)


def fit_marchenko_pastur_bounds(
    eigenvalues: np.ndarray,
    q: float,
    bwidth: float = 0.05,
) -> Tuple[float, float, float]:
    """Fit empirical noise variance \\sigma^2 to the bulk of empirical eigenvalues.

    Finds \\sigma^2 \\in (0, 1] minimizing the squared error between empirical KDE
    and theoretical Marchenko-Pastur distribution over [lambda_-, lambda_+].

    Args:
        eigenvalues: 1D array of sorted descending empirical eigenvalues.
        q: Ratio of observations to dimensions (T / N).
        bwidth: Bandwidth scaling parameter for kernel density estimation.

    Returns:
        Tuple of (fitted_sigma2, lambda_minus, lambda_plus).
    """
    evals = np.asarray(eigenvalues, dtype=float)
    if len(evals) < 2:
        raise ValueError(f"At least 2 eigenvalues required, got {len(evals)}")

    # KDE over empirical eigenvalues
    kde = gaussian_kde(evals, bw_method=bwidth)

    def _loss(var_candidate: float) -> float:
        if var_candidate <= 1e-6:
            return 1e6
        lambdas, mp_pdf, l_min, l_max = compute_marchenko_pastur_pdf(var_candidate, q, pts=200)
        emp_pdf = kde.evaluate(lambdas)
        sse = float(np.sum((emp_pdf - mp_pdf) ** 2))
        return sse

    # Optimize bounded variance in (0.01, 1.0]
    res = minimize_scalar(_loss, bounds=(0.01, 1.0), method="bounded")
    best_var = float(res.x)

    _, _, l_minus, l_plus = compute_marchenko_pastur_pdf(best_var, q, pts=10)
    return best_var, l_minus, l_plus


def get_covariance_from_correlation(
    corr: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Reconstruct covariance matrix from correlation matrix and standard deviations."""
    std_diag = np.diag(stds)
    cov = std_diag @ corr @ std_diag
    # Guarantee exact symmetry
    cov = 0.5 * (cov + cov.T)
    return cov


def get_correlation_from_covariance(
    cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose covariance matrix into correlation matrix and standard deviations."""
    variances = np.diag(cov)
    safe_variances = np.maximum(variances, 1e-12)
    stds = np.sqrt(safe_variances)
    inv_stds = np.diag(1.0 / stds)
    corr = inv_stds @ cov @ inv_stds
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr, stds


def denoise_covariance_matrix(
    cov_matrix: Union[pd.DataFrame, np.ndarray],
    sample_sessions: int,
    method: str = "constant_residual",
    shrinkage_alpha: float = 0.0,
) -> Tuple[np.ndarray, float, float, int, float, float]:
    """Denoise empirical covariance matrix via Random Matrix Theory eigenvalue filtering.

    Replaces noise eigenvalues (\\lambda_i \\le \\lambda_+) with their constant residual
    mean, preserving the trace of the correlation matrix and enforcing positive-definiteness.

    Args:
        cov_matrix: Empirical NxN covariance matrix (DataFrame or numpy array).
        sample_sessions: Number of historical time-series observations (T).
        method: Denoising algorithm: 'constant_residual' or 'targeted_shrinkage'.
        shrinkage_alpha: Shrinkage intensity in [0, 1] if targeted_shrinkage is selected.

    Returns:
        Tuple of:
            - denoised_cov (np.ndarray): Positive-definite denoised NxN covariance matrix.
            - lambda_plus (float): Upper Marchenko-Pastur cutoff.
            - fitted_sigma2 (float): Fitted empirical noise variance.
            - num_signals (int): Number of signal eigenvalues (\\lambda > \\lambda_+).
            - raw_cond_number (float): Condition number of raw correlation matrix.
            - denoised_cond_number (float): Condition number of denoised correlation matrix.
    """
    if isinstance(cov_matrix, pd.DataFrame):
        raw_cov = cov_matrix.values.astype(float)
    else:
        raw_cov = np.asarray(cov_matrix, dtype=float)

    n_assets = raw_cov.shape[0]
    if raw_cov.shape[0] != raw_cov.shape[1]:
        raise ValueError(f"Covariance matrix must be square, got {raw_cov.shape}")
    if n_assets < 2:
        raise ValueError(f"At least 2 assets required for RMT denoising, got {n_assets}")
    if sample_sessions < 5:
        raise ValueError(f"Insufficient sample sessions ({sample_sessions})")

    q = float(sample_sessions) / float(n_assets)

    # Convert to correlation and standard deviations
    raw_corr, stds = get_correlation_from_covariance(raw_cov)

    # Spectral decomposition of correlation matrix
    evals, evecs = np.linalg.eigh(raw_corr)
    # Sort descending
    sort_idx = np.argsort(evals)[::-1]
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    # Calculate raw condition number
    raw_min_eval = max(float(evals[-1]), 1e-12)
    raw_cond_number = float(evals[0]) / raw_min_eval

    # Fit Marchenko-Pastur bounds
    fitted_sigma2, l_minus, l_plus = fit_marchenko_pastur_bounds(evals, q)

    # Partition eigenvalues into signal vs noise
    is_signal = evals > l_plus
    num_signals = int(np.sum(is_signal))

    # At least the top eigenvalue is kept as signal to prevent complete degeneracy
    if num_signals == 0:
        num_signals = 1
        is_signal[0] = True

    denoised_evals = evals.copy()
    noise_indices = np.where(~is_signal)[0]

    if len(noise_indices) > 0:
        # Constant residual mean of noise eigenvalues to conserve trace
        mean_noise = float(np.mean(evals[noise_indices]))
        if method == "constant_residual":
            denoised_evals[noise_indices] = mean_noise
        elif method == "targeted_shrinkage":
            alpha = float(np.clip(shrinkage_alpha, 0.0, 1.0))
            denoised_evals[noise_indices] = (
                alpha * mean_noise + (1.0 - alpha) * evals[noise_indices]
            )
        else:
            raise ValueError(
                f"Unknown denoising method '{method}'. Must be 'constant_residual' or 'targeted_shrinkage'"
            )

    # Guarantee all eigenvalues are strictly positive
    denoised_evals = np.maximum(denoised_evals, 1e-8)

    # Reconstruct correlation matrix: C_tilde = V * Lambda_tilde * V^T
    lambda_mat = np.diag(denoised_evals)
    c_tilde = evecs @ lambda_mat @ evecs.T

    # Rescale to ensure unit diagonals: C_hat = D^{-1/2} C_tilde D^{-1/2}
    diag_c = np.diag(c_tilde)
    safe_diag_c = np.maximum(diag_c, 1e-12)
    inv_sqrt_d = np.diag(1.0 / np.sqrt(safe_diag_c))
    denoised_corr = inv_sqrt_d @ c_tilde @ inv_sqrt_d
    denoised_corr = 0.5 * (denoised_corr + denoised_corr.T)
    np.fill_diagonal(denoised_corr, 1.0)

    # Condition number of denoised correlation matrix
    denoised_eigen_decomp = np.linalg.eigvalsh(denoised_corr)
    denoised_min_eval = max(float(np.min(denoised_eigen_decomp)), 1e-12)
    denoised_cond_number = float(np.max(denoised_eigen_decomp)) / denoised_min_eval

    # Reconstruct covariance matrix
    denoised_cov = get_covariance_from_correlation(denoised_corr, stds)

    return (
        denoised_cov,
        float(l_plus),
        float(fitted_sigma2),
        num_signals,
        raw_cond_number,
        denoised_cond_number,
    )


def detrend_covariance_matrix(
    cov_matrix: Union[pd.DataFrame, np.ndarray],
    sample_sessions: int,
) -> np.ndarray:
    """Detrend empirical covariance matrix by shrinking the dominant market mode (lambda_1).

    Isolates idiosyncratic, sector, and residual cross-asset relationships by removing
    the systematic market-wide factor from the correlation structure.

    Args:
        cov_matrix: NxN covariance matrix.
        sample_sessions: Number of historical time-series observations.

    Returns:
        Detrended NxN covariance matrix.
    """
    if isinstance(cov_matrix, pd.DataFrame):
        raw_cov = cov_matrix.values.astype(float)
    else:
        raw_cov = np.asarray(cov_matrix, dtype=float)

    n_assets = raw_cov.shape[0]
    raw_corr, stds = get_correlation_from_covariance(raw_cov)

    evals, evecs = np.linalg.eigh(raw_corr)
    sort_idx = np.argsort(evals)[::-1]
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    # Market mode is evals[0]. Replace market mode with average of remaining eigenvalues
    if len(evals) > 1:
        evals[0] = float(np.mean(evals[1:]))

    lambda_mat = np.diag(evals)
    c_tilde = evecs @ lambda_mat @ evecs.T

    # Rescale to unit diagonal
    diag_c = np.diag(c_tilde)
    safe_diag_c = np.maximum(diag_c, 1e-12)
    inv_sqrt_d = np.diag(1.0 / np.sqrt(safe_diag_c))
    detrended_corr = inv_sqrt_d @ c_tilde @ inv_sqrt_d
    detrended_corr = 0.5 * (detrended_corr + detrended_corr.T)
    np.fill_diagonal(detrended_corr, 1.0)

    detrended_cov = get_covariance_from_correlation(detrended_corr, stds)
    return detrended_cov


def evaluate_rmt_denoising_profile(
    tickers: Optional[List[str]] = None,
    exchange: Optional[str] = None,
    lookback_sessions: int = 252,
    db_path: Optional[Path] = None,
    method: str = "constant_residual",
    detrend: bool = False,
) -> Tuple[RmtDenoiserReceipt, pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end Random Matrix Theory covariance denoising across universe tickers.

    Args:
        tickers: Optional explicit list of stock tickers.
        exchange: Optional MIC exchange code for screener auto-selection (default XNYS).
        lookback_sessions: Number of overlapping trading sessions (default 252).
        db_path: Optional SQLite database path.
        method: Denoising algorithm: 'constant_residual' or 'targeted_shrinkage'.
        detrend: Whether to detrend the dominant market eigenvalue mode.

    Returns:
        Tuple of (RmtDenoiserReceipt, raw_covariance_df, denoised_covariance_df).
    """
    assert_finite_number(lookback_sessions, "lookback_sessions")
    if lookback_sessions < 40:
        raise ValueError(f"Lookback sessions must be at least 40, got {lookback_sessions}")

    validated_tickers: List[str] = []
    if tickers:
        for t in tickers:
            validated_tickers.append(validate_ticker(t))
    else:
        mic = validate_exchange(exchange or "XNYS")
        screener_records = screen_asset_universe(
            exchange=mic, min_sessions=lookback_sessions, db_path=db_path
        )
        validated_tickers = [r.ticker for r in screener_records[:12]]

    if len(validated_tickers) < 2:
        raise ValueError(
            f"At least 2 validated assets required for RMT denoising, got {len(validated_tickers)}"
        )

    # Build aligned returns matrix
    returns_df = build_aligned_returns_matrix(
        validated_tickers, db_path=db_path, min_sessions=lookback_sessions
    )

    if returns_df.empty or len(returns_df) < 40 or returns_df.shape[1] < 2:
        raise ValueError(
            f"Insufficient aligned returns data for universe (rows: {len(returns_df)}, cols: {returns_df.shape[1]})"
        )

    # Truncate to recent lookback
    aligned_sample = returns_df.tail(lookback_sessions)
    n_sessions = len(aligned_sample)
    n_assets = aligned_sample.shape[1]
    active_tickers = list(aligned_sample.columns)

    # Empirical covariance matrix
    raw_cov = aligned_sample.cov()

    # Denoise
    (
        denoised_cov_arr,
        l_plus,
        fitted_sigma2,
        num_signals,
        raw_cond,
        denoised_cond,
    ) = denoise_covariance_matrix(
        raw_cov.values,
        sample_sessions=n_sessions,
        method=method,
    )

    if detrend:
        denoised_cov_arr = detrend_covariance_matrix(
            denoised_cov_arr,
            sample_sessions=n_sessions,
        )

    denoised_cov_df = pd.DataFrame(
        denoised_cov_arr,
        index=active_tickers,
        columns=active_tickers,
    )

    q = float(n_sessions) / float(n_assets)
    l_minus = fitted_sigma2 * (1.0 - math.sqrt(1.0 / q)) ** 2
    num_noise = n_assets - num_signals
    improvement_ratio = raw_cond / max(denoised_cond, 1e-12)

    # Compute variance proportion in signal modes
    corr_evals = np.linalg.eigvalsh(aligned_sample.corr().values)[::-1]
    signal_variance_pct = (
        float(np.sum(corr_evals[:num_signals])) / float(np.sum(corr_evals)) * 100.0
    )

    summary = (
        f"RMT Covariance Denoising ({n_assets} assets, {n_sessions} sessions, q={q:.2f}): "
        f"{num_signals} Signal Modes ({signal_variance_pct:.1f}% Variance), {num_noise} Noise Modes. "
        f"Noise Cutoff Lambda+: {l_plus:.3f}. Condition Number: {raw_cond:.1f} -> {denoised_cond:.1f} "
        f"({improvement_ratio:.1f}x condition improvement)."
    )

    receipt = RmtDenoiserReceipt(
        tickers=active_tickers,
        sample_sessions=n_sessions,
        num_assets=n_assets,
        q_ratio=q,
        noise_cutoff_lambda_plus=l_plus,
        noise_floor_lambda_minus=float(l_minus),
        fitted_noise_variance=fitted_sigma2,
        num_signal_eigenvalues=num_signals,
        num_noise_eigenvalues=num_noise,
        signal_variance_ratio_pct=signal_variance_pct,
        condition_number_raw=raw_cond,
        condition_number_denoised=denoised_cond,
        condition_number_improvement_ratio=improvement_ratio,
        method=method,
        is_detrended=detrend,
        summary=summary,
    )

    return receipt, raw_cov, denoised_cov_df
