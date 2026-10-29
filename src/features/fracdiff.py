"""Fractionally Differentiated Features (FFD) Engine for NextOmni Stock.

Implements Marcos López de Prado's (2018) Fixed-Width Window Fractional
Differentiation (FFD) filter to resolve the fundamental trade-off between
time-series stationarity and memory preservation.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    John Wiley & Sons, Chapter 5: "Fractionally Differentiated Features".
"""

import hashlib
import hmac
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import interpolate

FRACDIFF_SIGNING_KEY = b"nextomni_fracdiff_secret_audit_key_v1"

# Standard MacKinnon (2010) asymptotic critical values for OLS regression with constant (no trend)
ADF_CRITICAL_1PCT = -3.434
ADF_CRITICAL_5PCT = -2.863
ADF_CRITICAL_10PCT = -2.568

# MacKinnon response surface anchor points for p-value monotonic PCHIP interpolation
_MACKINNON_QUANTILES_T = np.array([
    -6.00, -5.00, -4.50, -3.90, -3.434, -3.120, -2.863, -2.568, -2.230,
    -1.940, -1.620, -1.280, -0.900, -0.450, 0.000, 0.500, 1.000, 2.000, 4.000
])
_MACKINNON_QUANTILES_P = np.array([
    0.00001, 0.00005, 0.0001, 0.0010, 0.0100, 0.0250, 0.0500, 0.1000, 0.2000,
    0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 0.9700, 0.9990, 0.9999
])
_PVALUE_INTERPOLATOR = interpolate.PchipInterpolator(_MACKINNON_QUANTILES_T, _MACKINNON_QUANTILES_P)


@dataclass
class FracDiffReceipt:
    """Cryptographic audit receipt for memory-preserving fractional differentiation."""

    ticker: str
    sample_sessions: int
    optimal_d: float
    selected_d: float
    threshold: float
    window_length: int
    retained_memory_corr: float
    raw_to_diff1_corr: float
    adf_stat: float
    adf_pvalue: float
    is_stationary: bool
    scan_points: List[Dict[str, float]]
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core quantitative metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.optimal_d:.4f}:{self.selected_d:.4f}:"
            f"{self.threshold:.6f}:{self.retained_memory_corr:.4f}:{self.raw_to_diff1_corr:.4f}:"
            f"{self.adf_stat:.4f}:{self.adf_pvalue:.4f}:{1 if self.is_stationary else 0}"
        )
        return hmac.new(FRACDIFF_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt against unauthorized record modification."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def get_fractional_weights(
    d: float,
    threshold: float = 1e-4,
    max_l: int = 2000,
) -> np.ndarray:
    """Compute binomial expansion weights for fractional differentiation filter (1 - B)^d.

    Mathematical Recursive Formulation:
        w_0 = 1.0
        w_k = -w_{k-1} * (d - k + 1) / k,   for k = 1, 2, ...
    
    Weights are truncated once |w_k| < threshold, yielding fixed-width window length l.

    Args:
        d: Order of fractional differentiation (typically in [0.0, 1.0]).
        threshold: Absolute weight cutoff threshold tau (default: 1e-4).
        max_l: Maximum allowable window length to prevent infinite expansion.

    Returns:
        NumPy array of expansion weights w = [w_0, w_1, ..., w_{l-1}].
    """
    if d == 0.0:
        return np.array([1.0], dtype=float)
    if d == 1.0:
        return np.array([1.0, -1.0], dtype=float)

    weights = [1.0]
    k = 1
    while k < max_l:
        w_k = -weights[-1] * (d - k + 1.0) / float(k)
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1

    return np.array(weights, dtype=float)


def fractional_differentiation_ffd(
    series: Union[pd.Series, np.ndarray, List[float]],
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """Compute Fixed-Width Window Fractional Differentiation (FFD).

    Guarantees causal zero-leakage transformation: each value X_t^(d) is computed
    strictly from past observations P_{t - l + 1 : t} with a constant history length l.

    Args:
        series: Time-series price sequence.
        d: Degree of fractional differentiation.
        threshold: Weight truncation threshold.

    Returns:
        pd.Series aligned with original index, containing NaNs for the initial l - 1 entries.
    """
    if isinstance(series, pd.Series):
        s_vals = series.values.astype(float)
        idx = series.index
    else:
        s_vals = np.asarray(series, dtype=float)
        idx = pd.RangeIndex(len(s_vals))

    n = len(s_vals)
    weights = get_fractional_weights(d=d, threshold=threshold)
    l_win = len(weights)

    if n < l_win or l_win == 0:
        return pd.Series(np.full(n, np.nan), index=idx)

    # Causal 1D convolution: np.convolve internally reflects the kernel,
    # so passing weights directly computes: sum_{k=0}^{l-1} weights[k] * s_vals[t - k]
    valid_conv = np.convolve(s_vals, weights, mode="valid")

    result = np.full(n, np.nan)
    result[l_win - 1:] = valid_conv

    return pd.Series(result, index=idx)


def augmented_dickey_fuller_test(
    series: Union[pd.Series, np.ndarray, List[float]],
    lags: int = 1,
) -> Tuple[float, float, bool]:
    """Perform Augmented Dickey-Fuller (ADF) unit root test for time-series stationarity.

    OLS Specification with constant:
        Delta y_t = alpha + gamma * y_{t-1} + sum_{i=1}^p phi_i * Delta y_{t-i} + eps_t

    Null Hypothesis (H0): gamma = 0 (series has a unit root / non-stationary).
    Alternative (H1): gamma < 0 (series is stationary).

    Args:
        series: Array or Series to test for stationarity.
        lags: Number of autoregressive lag differences (default: 1).

    Returns:
        Tuple of (adf_statistic, p_value, is_stationary_at_5pct).
    """
    if isinstance(series, pd.Series):
        vals = series.dropna().values.astype(float)
    else:
        arr = np.asarray(series, dtype=float)
        vals = arr[~np.isnan(arr)]

    n = len(vals)
    # Require at least 25 observations for reliable regression
    if n < max(25, lags + 5):
        return 0.0, 1.0, False

    diff = np.diff(vals)
    n_diff = len(diff)

    # Build regression matrices
    # Dependent variable: Delta y_t from t = lags to end
    y_reg = diff[lags:]
    n_samples = len(y_reg)

    if n_samples < lags + 3:
        return 0.0, 1.0, False

    # Regressors: [constant, y_{t-1}, Delta y_{t-1}, ..., Delta y_{t-lags}]
    cols = [np.ones(n_samples)]
    # y_{t-1} level
    cols.append(vals[lags: lags + n_samples])

    for i in range(1, lags + 1):
        cols.append(diff[lags - i: lags - i + n_samples])

    x_mat = np.column_stack(cols)

    try:
        # Solve OLS via SVD / least squares
        beta, residuals, rank, s = np.linalg.lstsq(x_mat, y_reg, rcond=None)
        if rank < x_mat.shape[1]:
            return 0.0, 1.0, False

        res = y_reg - x_mat @ beta
        rss = np.sum(res ** 2)
        df_e = n_samples - x_mat.shape[1]
        if df_e <= 0 or rss <= 1e-14:
            return 0.0, 1.0, False

        sigma_sq = rss / df_e
        inv_xtx = np.linalg.inv(x_mat.T @ x_mat)
        se_gamma = np.sqrt(max(1e-12, sigma_sq * inv_xtx[1, 1]))

        gamma_hat = beta[1]
        adf_stat = float(gamma_hat / se_gamma)

        # Estimate p-value using MacKinnon quantile response surface
        if adf_stat <= _MACKINNON_QUANTILES_T[0]:
            p_val = 0.00001
        elif adf_stat >= _MACKINNON_QUANTILES_T[-1]:
            p_val = 0.9999
        else:
            p_val = float(np.clip(_PVALUE_INTERPOLATOR(adf_stat), 0.0, 1.0))

        is_stat = (adf_stat <= ADF_CRITICAL_5PCT) and (p_val < 0.05)
        return adf_stat, p_val, is_stat

    except Exception:
        return 0.0, 1.0, False


def find_optimal_d(
    series: Union[pd.Series, np.ndarray, List[float]],
    threshold: float = 1e-4,
    d_step: float = 0.05,
    max_d: float = 1.0,
) -> Dict[str, Any]:
    """Scan order d in [0.0, max_d] to find the minimum d* that achieves stationarity.

    Balances stationarity with memory preservation:
        d* = argmin_{d} { d in [0, 1] : p_{ADF}(d) < 0.05 }

    Args:
        series: Original price series.
        threshold: Fractional differentiation weight truncation threshold.
        d_step: Grid step size for scanning d (default: 0.05).
        max_d: Maximum order to evaluate (default: 1.0).

    Returns:
        Dictionary containing scan results, optimal d*, and correlation metrics.
    """
    if isinstance(series, pd.Series):
        s = series.dropna().astype(float)
    else:
        s = pd.Series(series).dropna().astype(float)

    if len(s) < 30:
        return {
            "optimal_d": 1.0,
            "retained_memory_corr": 0.0,
            "raw_to_diff1_corr": 0.0,
            "adf_stat": 0.0,
            "adf_pvalue": 1.0,
            "is_stationary": False,
            "scan_points": [],
        }

    d_values = np.round(np.arange(0.0, max_d + 1e-6, d_step), 3)
    scan_points: List[Dict[str, float]] = []

    # First difference benchmark (d=1.0)
    diff1_series = s.diff().dropna()
    aligned_s_diff1 = s.loc[diff1_series.index]
    if len(aligned_s_diff1) > 5 and np.std(diff1_series) > 1e-8:
        raw_to_diff1_corr = float(np.corrcoef(aligned_s_diff1, diff1_series)[0, 1])
    else:
        raw_to_diff1_corr = 0.0

    optimal_d: Optional[float] = None
    optimal_stat: float = 0.0
    optimal_pval: float = 1.0
    optimal_corr: float = 0.0
    min_pval_found: float = 1.0
    best_candidate_d: float = 1.0
    best_candidate_stat: float = 0.0
    best_candidate_corr: float = 0.0

    for d in d_values:
        ffd = fractional_differentiation_ffd(s, d=float(d), threshold=threshold)
        valid_idx = ffd.dropna().index
        if len(valid_idx) < 25:
            continue

        ffd_valid = ffd.loc[valid_idx].values
        raw_valid = s.loc[valid_idx].values

        if np.std(ffd_valid) > 1e-8 and np.std(raw_valid) > 1e-8:
            corr = float(np.corrcoef(raw_valid, ffd_valid)[0, 1])
        else:
            corr = 0.0

        stat, pval, is_stat = augmented_dickey_fuller_test(ffd_valid)

        scan_points.append({
            "d": float(d),
            "adf_stat": float(stat),
            "p_value": float(pval),
            "memory_corr": float(corr),
            "is_stationary": 1.0 if is_stat else 0.0,
        })

        if pval < min_pval_found:
            min_pval_found = pval
            best_candidate_d = float(d)
            best_candidate_stat = float(stat)
            best_candidate_corr = float(corr)

        if optimal_d is None and is_stat:
            optimal_d = float(d)
            optimal_stat = float(stat)
            optimal_pval = float(pval)
            optimal_corr = float(corr)

    if optimal_d is None:
        optimal_d = best_candidate_d
        optimal_stat = best_candidate_stat
        optimal_pval = min_pval_found
        optimal_corr = best_candidate_corr

    return {
        "optimal_d": optimal_d,
        "retained_memory_corr": optimal_corr,
        "raw_to_diff1_corr": raw_to_diff1_corr,
        "adf_stat": optimal_stat,
        "adf_pvalue": optimal_pval,
        "is_stationary": optimal_pval < 0.05,
        "scan_points": scan_points,
    }


def evaluate_fracdiff_profile(
    ticker: str,
    prices: Union[pd.Series, np.ndarray, List[float]],
    threshold: float = 1e-4,
    auto_d: bool = True,
    target_d: Optional[float] = None,
) -> FracDiffReceipt:
    """Evaluate comprehensive fractional differentiation profile and issue audit receipt.

    Args:
        ticker: Asset ticker symbol.
        prices: Historical price series.
        threshold: FFD weight threshold.
        auto_d: Whether to automatically discover optimal d*.
        target_d: Optional manual differentiation order to evaluate.

    Returns:
        Cryptographically signed FracDiffReceipt.
    """
    if isinstance(prices, pd.Series):
        s = prices.dropna().astype(float)
    else:
        s = pd.Series(prices).dropna().astype(float)

    n_sessions = len(s)
    scan_res = find_optimal_d(s, threshold=threshold)

    optimal_d = scan_res["optimal_d"]
    if auto_d or target_d is None:
        selected_d = optimal_d
        selected_stat = scan_res["adf_stat"]
        selected_pval = scan_res["adf_pvalue"]
        selected_corr = scan_res["retained_memory_corr"]
        is_stat = scan_res["is_stationary"]
    else:
        selected_d = float(target_d)
        ffd_target = fractional_differentiation_ffd(s, d=selected_d, threshold=threshold)
        v_idx = ffd_target.dropna().index
        if len(v_idx) >= 25:
            ffd_v = ffd_target.loc[v_idx].values
            raw_v = s.loc[v_idx].values
            selected_corr = float(np.corrcoef(raw_v, ffd_v)[0, 1]) if np.std(ffd_v) > 1e-8 else 0.0
            selected_stat, selected_pval, is_stat = augmented_dickey_fuller_test(ffd_v)
        else:
            selected_stat, selected_pval, is_stat, selected_corr = 0.0, 1.0, False, 0.0

    w = get_fractional_weights(selected_d, threshold=threshold)
    win_len = len(w)

    return FracDiffReceipt(
        ticker=ticker.upper().strip(),
        sample_sessions=n_sessions,
        optimal_d=optimal_d,
        selected_d=selected_d,
        threshold=threshold,
        window_length=win_len,
        retained_memory_corr=selected_corr,
        raw_to_diff1_corr=scan_res["raw_to_diff1_corr"],
        adf_stat=selected_stat,
        adf_pvalue=selected_pval,
        is_stationary=is_stat,
        scan_points=scan_res["scan_points"],
    )
