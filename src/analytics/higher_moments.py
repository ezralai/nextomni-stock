"""Campbell R. Harvey & Akhtar Siddique (2000) Co-Skewness & Robert Dittmar (2002) Co-Kurtosis Engine.

Implements higher-order systematic risk co-moments:
- Systematic Co-Skewness (beta_SKS, Kraus-Litzenberger 1976 / Harvey-Siddique 2000):
  Measures asset covariance with market volatility. Negative co-skewness assets crash
  precisely when aggregate market volatility spikes, demanding a +3.6% annual risk premium.
- Systematic Co-Kurtosis (beta_CKT, Dittmar 2002 / Fang-Lai 1997):
  Measures asset covariance with market extreme tails, quantifying sensitivity to market fat-tail panics.
- Four-Moment CAPM:
  Multivariate orthogonal polynomial regression estimating linear market beta, quadratic co-skewness beta,
  and cubic co-kurtosis beta with White (HC0) heteroskedasticity-robust standard errors.

References:
    Harvey, C. R., & Siddique, A. (2000). Conditional skewness in asset pricing tests.
    The Journal of Finance, 55(3), 1263-1295.
    Dittmar, R. (2002). Nonlinear pricing kernels, Kurtosis preference, and evidence from
    the cross section of equity returns. The Journal of Finance, 57(1), 369-403.
    Kraus, A., & Litzenberger, R. H. (1976). Skewness and investors' decisions.
    The Journal of Finance, 31(4), 1085-1100.
    Fang, H., & Lai, T. Y. (1997). Co-kurtosis and capital asset pricing.
    The Financial Review, 32(2), 293-307.
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

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

HIGHER_MOMENTS_SIGNING_KEY = b"nextomni_higher_moments_secret_key_v1"


@dataclass
class HigherMomentsReceipt:
    """Cryptographic audit receipt for Systematic Co-Skewness & Co-Kurtosis Profile."""

    ticker: str
    benchmark_ticker: str
    sample_sessions: int
    lookback_window: int
    market_beta: float  # beta_M (linear covariance)
    coskewness_beta: float  # beta_SKS (Harvey & Siddique 2000)
    coskewness_t_stat: float
    coskewness_p_value: float
    is_coskewness_significant: bool  # p < 0.05
    cokurtosis_beta: float  # beta_CKT (Dittmar 2002)
    cokurtosis_t_stat: float
    cokurtosis_p_value: float
    is_cokurtosis_significant: bool  # p < 0.05
    annualized_alpha_pct: float  # 4-Moment Alpha
    expected_coskewness_premium_pct: float  # -beta_SKS * 3.6% (Negative SKS commands positive premium)
    higher_moment_risk_regime: str  # SEVERE_CRASH_VULNERABILITY, ASYMMETRIC_HEDGE, NEUTRAL_GAUSSIAN
    four_moment_r_squared: float
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across higher-moment risk metrics."""
        payload = (
            f"{self.ticker}:{self.benchmark_ticker}:{self.sample_sessions}:{self.lookback_window}:"
            f"{self.market_beta:.4f}:{self.coskewness_beta:.4f}:{self.coskewness_t_stat:.4f}:"
            f"{self.cokurtosis_beta:.4f}:{self.cokurtosis_t_stat:.4f}:"
            f"{self.expected_coskewness_premium_pct:.4f}:{self.higher_moment_risk_regime}"
        )
        return hmac.new(
            HIGHER_MOMENTS_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity of the higher moments receipt."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_standardized_comoments(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute Kraus-Litzenberger (1976) / Harvey-Siddique (2000) standardized co-moments.

    Formulas:
        beta_M   = Cov(r_i, r_m) / Var(r_m)
        beta_SKS = E[(r_i - mu_i) * (r_m - mu_m)^2] / (std(r_i) * Var(r_m))
        beta_CKT = E[(r_i - mu_i) * (r_m - mu_m)^3] / (std(r_i) * Var(r_m)^(3/2))

    Note: When asset == market:
        beta_M(M, M)   = 1.0
        beta_SKS(M, M) = Skew(M)
        beta_CKT(M, M) = Kurt(M)

    Args:
        asset_returns: 1D array of daily asset returns.
        market_returns: 1D array of daily market returns.

    Returns:
        Tuple of (beta_M, beta_SKS, beta_CKT).
    """
    ri = np.asarray(asset_returns, dtype=float)
    rm = np.asarray(market_returns, dtype=float)
    n = len(ri)

    if n != len(rm):
        raise ValueError(f"Length mismatch: asset ({n}) vs market ({len(rm)})")
    if n < 15:
        raise ValueError(f"Insufficient observations ({n}), need at least 15 for higher moments")

    dev_i = ri - np.mean(ri)
    dev_m = rm - np.mean(rm)

    var_i = max(1e-12, float(np.mean(dev_i**2)))
    var_m = max(1e-12, float(np.mean(dev_m**2)))
    std_i = math.sqrt(var_i)
    std_m = math.sqrt(var_m)

    # 1. Linear Beta
    cov_im = float(np.mean(dev_i * dev_m))
    beta_m = cov_im / var_m

    # 2. Systematic Co-Skewness: E[dev_i * dev_m^2] / (std_i * var_m)
    coskew_num = float(np.mean(dev_i * (dev_m**2)))
    beta_sks = coskew_num / max(1e-12, std_i * var_m)

    # 3. Systematic Co-Kurtosis: E[dev_i * dev_m^3] / (std_i * std_m^3)
    cokurt_num = float(np.mean(dev_i * (dev_m**3)))
    beta_ckt = cokurt_num / max(1e-12, std_i * (var_m**1.5))

    return beta_m, beta_sks, beta_ckt


def estimate_four_moment_capm_regression(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    risk_free_rate_ann: float = 0.04,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Execute Four-Moment CAPM regression with White HC0 robust standard errors.

    Model with orthogonal polynomial market terms:
        r_{i,t} - r_f = alpha + beta_M * (r_{m,t} - r_f) + gamma_SKS * z_sks + delta_CKT * z_ckt + eps_t
    where z_sks and z_ckt are Gram-Schmidt orthogonalized quadratic and cubic market deviations.

    Args:
        asset_returns: 1D array of daily asset returns.
        market_returns: 1D array of daily market returns.
        risk_free_rate_ann: Annual risk-free rate (default: 4.0%).

    Returns:
        Tuple of:
            (alpha_ann_pct, beta_M, gamma_SKS, t_stat_sks, p_val_sks,
             delta_CKT, t_stat_ckt, p_val_ckt, r_squared)
    """
    ri = np.asarray(asset_returns, dtype=float)
    rm = np.asarray(market_returns, dtype=float)
    n = len(ri)

    rf_daily = risk_free_rate_ann / 252.0
    y = ri - rf_daily
    mkt_excess = rm - rf_daily

    dev_m = rm - np.mean(rm)
    quad_m = dev_m**2
    cube_m = dev_m**3

    # Gram-Schmidt orthogonalization to eliminate collinearity between mkt, quad, and cube
    # 1. Orthogonalize quad_m against mkt_excess
    beta_q_m = float(np.cov(quad_m, mkt_excess)[0, 1]) / max(1e-10, float(np.var(mkt_excess, ddof=1)))
    z_sks = quad_m - (beta_q_m * mkt_excess)
    z_sks = (z_sks - np.mean(z_sks)) / max(1e-6, float(np.std(z_sks, ddof=1)))

    # 2. Orthogonalize cube_m against [mkt_excess, z_sks]
    X_sub = np.column_stack([mkt_excess, z_sks])
    b_cube = np.linalg.pinv(X_sub) @ cube_m
    z_ckt = cube_m - (X_sub @ b_cube)
    z_ckt = (z_ckt - np.mean(z_ckt)) / max(1e-6, float(np.std(z_ckt, ddof=1)))

    # Design matrix: [Constant, mkt_excess, z_sks, z_ckt]
    X = np.column_stack([np.ones(n), mkt_excess, z_sks, z_ckt])
    k_params = 4

    # OLS estimation
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)

    alpha_daily = float(beta_hat[0])
    beta_M = float(beta_hat[1])
    gamma_SKS = float(beta_hat[2])
    delta_CKT = float(beta_hat[3])

    # White (HC0) heteroskedasticity-consistent covariance matrix
    residuals = y - (X @ beta_hat)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residuals**2))
    r_squared = float(max(0.0, 1.0 - (ss_res / max(1e-12, ss_tot))))

    S = np.zeros((k_params, k_params))
    for t in range(n):
        x_t = X[t : t + 1]
        S += (residuals[t] ** 2) * (x_t.T @ x_t)
    hc0_cov = XtX_inv @ S @ XtX_inv

    se_sks = math.sqrt(max(1e-12, float(hc0_cov[2, 2])))
    t_stat_sks = float(gamma_SKS / se_sks)
    z_sks_val = abs(t_stat_sks)
    p_val_sks = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z_sks_val / math.sqrt(2.0)))))

    se_ckt = math.sqrt(max(1e-12, float(hc0_cov[3, 3])))
    t_stat_ckt = float(delta_CKT / se_ckt)
    z_ckt_val = abs(t_stat_ckt)
    p_val_ckt = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z_ckt_val / math.sqrt(2.0)))))

    alpha_ann_pct = float(alpha_daily * 252.0 * 100.0)

    return (
        alpha_ann_pct,
        beta_M,
        gamma_SKS,
        t_stat_sks,
        p_val_sks,
        delta_CKT,
        t_stat_ckt,
        p_val_ckt,
        r_squared,
    )


def classify_higher_moment_risk_regime(
    coskewness_beta: float,
    cokurtosis_beta: float,
) -> Tuple[str, float]:
    """Classify higher-moment systematic risk regime and calculate expected co-skewness premium.

    Harvey & Siddique (2000) show that an asset with negative co-skewness commands an annualized
    expected return premium of ~3.6% per unit of negative co-skewness.

    Args:
        coskewness_beta: Standardized systematic co-skewness (beta_SKS).
        cokurtosis_beta: Standardized systematic co-kurtosis (beta_CKT).

    Returns:
        Tuple of (risk_regime, expected_coskewness_premium_pct).
    """
    # Negative co-skewness commands positive risk premium: Premium = -beta_SKS * 3.6%
    expected_premium = float(round(-coskewness_beta * 3.60, 2))

    if coskewness_beta < -0.25 and cokurtosis_beta > 1.20:
        regime = "SEVERE_CRASH_VULNERABILITY"
    elif coskewness_beta > 0.20:
        regime = "ASYMMETRIC_TURBULENCE_HEDGE"
    elif coskewness_beta < -0.15:
        regime = "ELEVATED_CO_VOLATILITY_RISK"
    else:
        regime = "NEUTRAL_GAUSSIAN_PROFILE"

    return regime, expected_premium


def evaluate_higher_moments_profile(
    ticker: str,
    asset_prices: Union[pd.Series, np.ndarray, List[float]],
    benchmark_ticker: str = "SPY",
    benchmark_prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    lookback_window: int = 60,
    risk_free_rate_ann: float = 0.04,
    db_path: Optional[Path] = None,
) -> HigherMomentsReceipt:
    """Evaluate complete Harvey-Siddique Co-Skewness & Dittmar Co-Kurtosis profile.

    Args:
        ticker: Asset ticker symbol.
        asset_prices: Daily closing price series.
        benchmark_ticker: Benchmark ticker symbol (default: 'SPY').
        benchmark_prices: Optional benchmark price series. If None, loaded from SQLite.
        lookback_window: Rolling observation window in sessions (default: 60).
        risk_free_rate_ann: Annual risk-free rate (default: 4.0%).
        db_path: Optional SQLite database path.

    Returns:
        HigherMomentsReceipt with cryptographic HMAC-SHA256 signature and tamper verification.
    """
    clean_ticker = validate_ticker(ticker)
    clean_bench = validate_ticker(benchmark_ticker)
    assert_finite_number(lookback_window, "lookback_window")
    assert_finite_number(risk_free_rate_ann, "risk_free_rate_ann")

    s_asset = pd.Series(asset_prices, dtype=float).dropna()
    if len(s_asset) < 20:
        raise ValueError(f"Insufficient asset price observations ({len(s_asset)}), need at least 20")

    # Load benchmark data if not provided
    if benchmark_prices is None:
        df_bench = db.get_observations(clean_bench, db_path=db_path)
        if df_bench.empty or len(df_bench) < 20:
            raise ValueError(
                f"Insufficient benchmark observations for {clean_bench}. Synchronize {clean_bench} first."
            )
        col = "adj_close" if "adj_close" in df_bench.columns else "close"
        s_bench = df_bench[col].dropna().astype(float)
    else:
        s_bench = pd.Series(benchmark_prices, dtype=float).dropna()

    # Align dates / index safely
    r_asset = s_asset.pct_change().dropna()
    r_bench = s_bench.pct_change().dropna()

    if isinstance(r_asset.index, pd.DatetimeIndex) and isinstance(r_bench.index, pd.DatetimeIndex):
        aligned = pd.concat([r_asset, r_bench], axis=1, join="inner").dropna()
        y_ret = aligned.iloc[:, 0].values
        m_ret = aligned.iloc[:, 1].values
    else:
        min_len = min(len(r_asset), len(r_bench))
        y_ret = r_asset.values[-min_len:]
        m_ret = r_bench.values[-min_len:]

    total_sessions = len(y_ret)
    if total_sessions < 16:
        raise ValueError(f"Insufficient aligned return observations ({total_sessions}), need at least 16")

    # Slice lookback window
    window = min(lookback_window, total_sessions)
    y_slice = y_ret[-window:]
    m_slice = m_ret[-window:]

    # 1. Standardized Co-Moments (Kraus-Litzenberger / Harvey-Siddique / Dittmar)
    beta_m_std, beta_sks, beta_ckt = compute_standardized_comoments(y_slice, m_slice)

    # 2. Four-Moment CAPM Regression with White HC0 Robust Inference
    (
        alpha_ann,
        beta_m_reg,
        gamma_sks,
        t_stat_sks,
        p_val_sks,
        delta_ckt,
        t_stat_ckt,
        p_val_ckt,
        r_sq,
    ) = estimate_four_moment_capm_regression(
        y_slice,
        m_slice,
        risk_free_rate_ann=risk_free_rate_ann,
    )

    is_sks_sig = p_val_sks < 0.05
    is_ckt_sig = p_val_ckt < 0.05

    # 3. Classify Risk Regime & Expected Co-Skewness Premium
    regime, expected_prem = classify_higher_moment_risk_regime(beta_sks, beta_ckt)

    sig_sks_label = "YES (p < 0.05)" if is_sks_sig else "no"
    sig_ckt_label = "YES (p < 0.05)" if is_ckt_sig else "no"

    summary = (
        f"Higher-Moment Asset Pricing for {clean_ticker} vs {clean_bench} ({window} sessions): "
        f"Market Beta: {beta_m_std:.2f}, Co-Skewness (β_SKS): {beta_sks:+.3f} (t={t_stat_sks:+.2f}, Sig: {sig_sks_label}), "
        f"Co-Kurtosis (β_CKT): {beta_ckt:.3f} (t={t_stat_ckt:+.2f}, Sig: {sig_ckt_label}), "
        f"4-Moment Alpha: {alpha_ann:+.2f}% Ann (R²: {r_sq * 100:.1f}%). "
        f"Regime: {regime} (Expected Co-Skewness Premium: {expected_prem:+.2f}%/yr)."
    )

    receipt = HigherMomentsReceipt(
        ticker=clean_ticker,
        benchmark_ticker=clean_bench,
        sample_sessions=total_sessions,
        lookback_window=window,
        market_beta=beta_m_std,
        coskewness_beta=beta_sks,
        coskewness_t_stat=t_stat_sks,
        coskewness_p_value=p_val_sks,
        is_coskewness_significant=is_sks_sig,
        cokurtosis_beta=beta_ckt,
        cokurtosis_t_stat=t_stat_ckt,
        cokurtosis_p_value=p_val_ckt,
        is_cokurtosis_significant=is_ckt_sig,
        annualized_alpha_pct=alpha_ann,
        expected_coskewness_premium_pct=expected_prem,
        higher_moment_risk_regime=regime,
        four_moment_r_squared=r_sq,
        summary=summary,
    )

    return receipt
