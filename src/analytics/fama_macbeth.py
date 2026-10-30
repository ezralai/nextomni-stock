"""Eugene Fama & James MacBeth (1973) Two-Pass Cross-Sectional Asset Pricing Engine.

Implements the classic two-pass regression methodology to test multi-factor asset pricing
models (CAPM, Fama-French, Carhart) and determine whether factor loadings command statistically
significant risk premia (lambda_k) across the cross-section of equities.
Features:
- Pass 1: Multivariate time-series regression of asset excess returns on factor returns to estimate betas.
- Pass 2: Cross-sectional regressions at every time step t of asset returns on estimated betas.
- Pass 3: Time-series aggregation of lambda_t with standard Fama-MacBeth and Shanken (1992)
  errors-in-variables corrections for generated regressor uncertainty.

References:
    Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests.
    Journal of Political Economy, 81(3), 607-636.
    Shanken, J. (1992). On the estimation of beta-pricing models. The Review of
    Financial Studies, 5(1), 1-33.
    Cochrane, J. H. (2005). Asset Pricing (Revised Edition). Princeton University Press.
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

from src.analytics.fama_french import construct_synthetic_factor_proxies
from src.analytics.portfolio import build_aligned_returns_matrix
import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

FAMA_MACBETH_SIGNING_KEY = b"nextomni_fama_macbeth_secret_key_v1"


@dataclass
class FactorPremiumStat:
    """Statistical summary for an individual factor risk premium."""

    factor_name: str
    risk_premium_ann_pct: float  # Annualized lambda_k * 100
    t_stat_fm: float  # Classic Fama-MacBeth t-statistic
    t_stat_shanken: float  # Shanken (1992) corrected t-statistic
    p_value_shanken: float
    is_significant_5pct: bool
    shanken_correction_factor: float


@dataclass
class FamaMacBethReceipt:
    """Cryptographic audit receipt for Fama-MacBeth Two-Pass Cross-Sectional Pricing."""

    universe: List[str]
    sample_sessions: int
    lookback_window: int
    factors_tested: List[str]
    factor_premia: List[Dict[str, Any]]
    cross_sectional_r_squared: float  # Average cross-sectional R^2
    pricing_error_rmse_ann_pct: float  # Annualized root mean squared pricing error alpha
    is_model_rejected_5pct: bool  # Chi-squared test of joint pricing errors
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across cross-sectional pricing metrics."""
        univ_str = ",".join(sorted(self.universe))
        factors_str = ",".join(sorted(self.factors_tested))
        payload = (
            f"{univ_str}:{factors_str}:{self.sample_sessions}:{self.lookback_window}:"
            f"{self.cross_sectional_r_squared:.4f}:{self.pricing_error_rmse_ann_pct:.4f}:"
            f"{self.is_model_rejected_5pct}"
        )
        return hmac.new(
            FAMA_MACBETH_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity of the Fama-MacBeth pricing receipt."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def estimate_time_series_betas(
    asset_returns_matrix: np.ndarray,
    factor_returns_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute Pass 1: Multivariate time-series regressions for each asset.

    Model for asset i:
        R_{i,t} = alpha_i + beta_{i,1} F_{1,t} + ... + beta_{i,K} F_{K,t} + eps_{i,t}

    Args:
        asset_returns_matrix: Matrix of asset returns (T x N).
        factor_returns_matrix: Matrix of factor returns (T x K).

    Returns:
        Tuple of:
            - beta_matrix: Estimated factor loadings (N x K).
            - alpha_vector: Estimated time-series intercepts (N,).
            - residuals_matrix: Time-series regression residuals (T x N).
    """
    T, N = asset_returns_matrix.shape
    T_f, K = factor_returns_matrix.shape
    if T != T_f:
        raise ValueError(f"Time-series length mismatch: assets ({T}) vs factors ({T_f})")
    if T <= K + 2:
        raise ValueError(f"Insufficient time sessions ({T}) for {K} factors (need at least {K + 3})")

    # Design matrix: [Constant, Factors]
    X = np.column_stack([np.ones(T), factor_returns_matrix])
    XtX_inv = np.linalg.pinv(X.T @ X)

    # Solve across all N assets simultaneously: B = (X^T X)^{-1} X^T Y
    # Shape: (K+1, N)
    B = XtX_inv @ (X.T @ asset_returns_matrix)

    alpha_vector = B[0, :]
    beta_matrix = B[1:, :].T  # Shape: (N, K)

    # Compute residuals: Y - X B
    y_pred = X @ B
    residuals_matrix = asset_returns_matrix - y_pred

    return beta_matrix, alpha_vector, residuals_matrix


def run_cross_sectional_pass(
    asset_returns_matrix: np.ndarray,
    beta_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute Pass 2: Cross-sectional regressions at every session t.

    Model at session t across all N assets:
        R_{i,t} = lambda_{0,t} + beta_{i,1} lambda_{1,t} + ... + beta_{i,K} lambda_{K,t} + eta_{i,t}

    Args:
        asset_returns_matrix: Matrix of asset returns (T x N).
        beta_matrix: Estimated factor loadings (N x K).

    Returns:
        Tuple of:
            - lambda_zero: Time series of zero-beta / intercept rates (T,).
            - lambda_premia: Time series of factor risk premia (T x K).
            - cross_sectional_r2: Time series of cross-sectional R^2 (T,).
    """
    T, N = asset_returns_matrix.shape
    N_b, K = beta_matrix.shape
    if N != N_b:
        raise ValueError(f"Cross-sectional asset count mismatch: returns ({N}) vs betas ({N_b})")
    if N <= K + 1:
        raise ValueError(f"Cross-sectional degrees of freedom deficit: N={N} assets must exceed K+1={K+1}")

    # Design matrix for cross-sectional regression: [Constant, Betas]
    Z = np.column_stack([np.ones(N), beta_matrix])  # Shape: (N, K+1)
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)

    # Regress at each time step t: lambda_t = (Z^T Z)^{-1} Z^T R_t
    # Z_cross: (K+1, N)
    Z_proj = ZtZ_inv @ Z.T
    # Lambda_all: (K+1, T)
    lambda_all = Z_proj @ asset_returns_matrix.T

    lambda_zero = lambda_all[0, :]
    lambda_premia = lambda_all[1:, :].T  # Shape: (T, K)

    # Compute cross-sectional R^2 at each t
    pred_returns = (Z @ lambda_all).T  # Shape: (T, N)
    resids = asset_returns_matrix - pred_returns

    r2_list = []
    for t in range(T):
        ss_tot = float(np.sum((asset_returns_matrix[t, :] - np.mean(asset_returns_matrix[t, :])) ** 2))
        ss_res = float(np.sum(resids[t, :] ** 2))
        r2 = max(0.0, 1.0 - (ss_res / max(1e-12, ss_tot)))
        r2_list.append(r2)

    return lambda_zero, lambda_premia, np.array(r2_list)


def compute_fama_macbeth_statistics(
    factor_names: List[str],
    lambda_premia: np.ndarray,
    factor_returns_matrix: np.ndarray,
) -> List[FactorPremiumStat]:
    """Execute Pass 3: Time-series aggregation and Shanken (1992) errors-in-variables correction.

    Formula:
        bar_lambda = mean(lambda_t)
        Var_FM(bar_lambda) = (1 / (T*(T-1))) * sum((lambda_t - bar_lambda)^2)
        Shanken correction factor: c = bar_lambda^T Sigma_F^{-1} bar_lambda
        Var_Shanken = (1 + c) * Var_FM(bar_lambda)

    Args:
        factor_names: List of factor names (K,).
        lambda_premia: Estimated factor risk premia across time (T x K).
        factor_returns_matrix: Factor returns matrix (T x K).

    Returns:
        List of FactorPremiumStat objects.
    """
    T, K = lambda_premia.shape
    mean_lambda = np.mean(lambda_premia, axis=0)  # (K,)

    # Factor covariance matrix Sigma_F (annualized or daily consistent)
    sigma_f = np.cov(factor_returns_matrix, rowvar=False)
    if K == 1:
        sigma_f = np.array([[float(sigma_f)]])

    sigma_f_inv = np.linalg.pinv(sigma_f)

    # Shanken quadratic form correction factor c: lambda^T Sigma_f^{-1} lambda
    # Ensure dimensions align
    c_shanken = float(mean_lambda @ sigma_f_inv @ mean_lambda)
    c_shanken = max(0.0, c_shanken)
    shanken_mult = 1.0 + c_shanken

    stats_list = []
    for k in range(K):
        series_k = lambda_premia[:, k]
        mean_k = float(mean_lambda[k])
        ann_premium_pct = mean_k * 252.0 * 100.0

        # Classic Fama-MacBeth sample variance of the mean
        sample_var = float(np.var(series_k, ddof=1)) if T > 1 else 1e-6
        se_fm = math.sqrt(sample_var / float(T))
        t_fm = float(mean_k / max(1e-12, se_fm))

        # Shanken corrected standard error
        se_shanken = se_fm * math.sqrt(shanken_mult)
        t_shanken = float(mean_k / max(1e-12, se_shanken))

        # Gaussian two-tailed p-value
        z = abs(t_shanken)
        p_val_shanken = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))
        is_sig = p_val_shanken < 0.05

        stats_list.append(
            FactorPremiumStat(
                factor_name=factor_names[k],
                risk_premium_ann_pct=ann_premium_pct,
                t_stat_fm=t_fm,
                t_stat_shanken=t_shanken,
                p_value_shanken=p_val_shanken,
                is_significant_5pct=is_sig,
                shanken_correction_factor=shanken_mult,
            )
        )

    return stats_list


def evaluate_fama_macbeth_profile(
    universe: List[str],
    benchmark_ticker: str = "SPY",
    lookback_window: int = 120,
    risk_free_rate_ann: float = 0.04,
    db_path: Optional[Path] = None,
) -> FamaMacBethReceipt:
    """Execute complete Fama-MacBeth (1973) Two-Pass Asset Pricing workflow with Shanken correction.

    Args:
        universe: List of equity ticker symbols (minimum 4 for 2-factor, 6 for 4-factor).
        benchmark_ticker: Market benchmark symbol (default: 'SPY').
        lookback_window: Historical observation sessions to include (default: 120).
        risk_free_rate_ann: Annual risk-free rate (default: 4.0%).
        db_path: Optional SQLite database path.

    Returns:
        FamaMacBethReceipt with cross-sectional factor premia, Shanken t-statistics, and HMAC audit signature.
    """
    clean_universe = [validate_ticker(t) for t in universe]
    clean_bench = validate_ticker(benchmark_ticker)
    if clean_bench not in clean_universe:
        all_tickers = clean_universe + [clean_bench]
    else:
        all_tickers = clean_universe

    if len(clean_universe) < 4:
        raise ValueError("Universe must contain at least 4 distinct assets for cross-sectional regression")
    assert_finite_number(lookback_window, "lookback_window")
    assert_finite_number(risk_free_rate_ann, "risk_free_rate_ann")

    # Load aligned returns
    df_returns = build_aligned_returns_matrix(
        all_tickers,
        lookback_window=lookback_window,
        db_path=db_path,
    )
    if df_returns.empty or len(df_returns) < 25:
        raise ValueError(
            f"Insufficient aligned price history for universe {clean_universe}. Ensure tickers are synchronized."
        )

    # Check that benchmark exists in returns
    if clean_bench not in df_returns.columns:
        raise ValueError(f"Benchmark {clean_bench} not found in synchronized aligned returns")

    # Excess returns
    rf_daily = risk_free_rate_ann / 252.0
    r_bench = df_returns[clean_bench].values - rf_daily

    # Asset excess returns (T x N)
    Y_assets = df_returns[clean_universe].values - rf_daily
    T, N = Y_assets.shape

    # Construct standard Carhart (1997) 4-factor proxies [MKT, SMB, HML, UMD]
    df_factors = construct_synthetic_factor_proxies(r_bench)
    if N >= 6:
        factor_names = ["MKT", "SMB", "HML", "UMD"]
    elif N == 5:
        factor_names = ["MKT", "SMB", "HML"]
    else:  # N == 4
        factor_names = ["MKT", "SMB"]
    F_factors = df_factors[factor_names].values

    # Pass 1: Time-series regressions
    betas, alphas_ts, residuals_ts = estimate_time_series_betas(Y_assets, F_factors)

    # Pass 2: Cross-sectional regressions
    lambda_zero, lambda_premia, r2_series = run_cross_sectional_pass(Y_assets, betas)

    # Pass 3: Time-series aggregation & Shanken correction
    factor_stats = compute_fama_macbeth_statistics(factor_names, lambda_premia, F_factors)

    # Cross-sectional pricing error RMSE (unexplained cross-sectional alpha)
    # Pricing error = mean(Y_i) - (mean(lambda_0) + sum(beta_ik * mean(lambda_k)))
    mean_y = np.mean(Y_assets, axis=0)  # (N,)
    mean_l0 = float(np.mean(lambda_zero))
    mean_lk = np.mean(lambda_premia, axis=0)  # (K,)
    implied_ret = mean_l0 + (betas @ mean_lk)
    pricing_errors = mean_y - implied_ret
    pricing_rmse_ann = math.sqrt(float(np.mean(pricing_errors**2)) * 252.0) * 100.0

    avg_r2 = float(np.mean(r2_series))

    # Joint test of pricing errors: Chi-squared Wald test
    # If pricing errors are jointly zero, model cannot be rejected
    cov_resid = np.cov(residuals_ts, rowvar=False)
    cov_resid_inv = np.linalg.pinv(cov_resid)
    shanken_c = factor_stats[0].shanken_correction_factor
    # Wald stat = T * (1 + c)^{-1} * alpha^T * Sigma^{-1} * alpha
    wald_stat = float((T / max(1e-6, shanken_c)) * (pricing_errors @ cov_resid_inv @ pricing_errors))
    dof_wald = N - len(factor_names)
    # Approximate critical value at 5% level (normal approx for chi2 if dof > 1)
    crit_val = float(dof_wald + 1.96 * math.sqrt(2.0 * max(1, dof_wald)))
    is_rejected = wald_stat > crit_val

    # Convert factor stats to dict list
    premia_dicts = [asdict(s) for s in factor_stats]

    sig_factors = [s.factor_name for s in factor_stats if s.is_significant_5pct]
    sig_summary = ", ".join(sig_factors) if sig_factors else "None"
    rejection_str = "REJECTED (significant pricing errors)" if is_rejected else "SUPPORTED (pricing errors not statistically significant)"

    summary = (
        f"Fama-MacBeth (1973) Two-Pass Pricing for {N} assets across {T} sessions: "
        f"Cross-Sectional R²: {avg_r2 * 100:.1f}%, Pricing RMSE: {pricing_rmse_ann:.2f}% Ann. "
        f"Significant Factors (Shanken 5%): [{sig_summary}]. Model Status: {rejection_str}."
    )

    receipt = FamaMacBethReceipt(
        universe=clean_universe,
        sample_sessions=T,
        lookback_window=lookback_window,
        factors_tested=factor_names,
        factor_premia=premia_dicts,
        cross_sectional_r_squared=avg_r2,
        pricing_error_rmse_ann_pct=pricing_rmse_ann,
        is_model_rejected_5pct=is_rejected,
        summary=summary,
    )

    return receipt
