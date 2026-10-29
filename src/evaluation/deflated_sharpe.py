"""Bailey & López de Prado (2012, 2014) Probabilistic & Deflated Sharpe Ratio Engine.

Implements the Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) to correct
for non-normality (skewness, fat tails), selection bias, multiple testing (data snooping),
and determine the Minimum Track Record Length (MinTRL) to reject backtest overfitting.

References:
    Bailey, D. H., & López de Prado, M. (2012). The Sharpe ratio efficient frontier.
    Journal of Risk, 15(2), 13-44.
    Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
    selection bias, backtest overfitting and non-normality. Journal of Portfolio
    Management, 40(5), 94-107.
    López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
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
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as scipy_skew

from src.security.validators import assert_finite_number, validate_ticker

DSR_SIGNING_KEY = b"nextomni_deflated_sharpe_secret_key_v1"
EULER_MASCHERONI = 0.57721566490153286


@dataclass
class DeflatedSharpeReceipt:
    """Cryptographic audit receipt for Deflated Sharpe Ratio and backtest overfitting analysis."""

    ticker: str
    sample_sessions: int
    annualized_sharpe_ratio: float  # Classical point-estimate annualized Sharpe
    daily_sharpe_ratio: float
    skewness: float  # Pearson skewness gamma_3
    kurtosis: float  # Pearson kurtosis gamma_4 (normal = 3.0)
    benchmark_sharpe_ann: float  # Hurdle annualized Sharpe
    probabilistic_sharpe_ratio: float  # PSR in [0, 1]
    num_trials: int  # Number of model/parameter trials tested
    trial_sharpe_variance: float  # Variance across tested strategy trials
    expected_max_sharpe_ann: float  # Theoretical E[max SR] under null of zero skill
    deflated_sharpe_ratio: float  # DSR in [0, 1]
    is_statistically_significant: bool  # DSR >= 0.95 (at 95% confidence)
    overfitting_probability_pct: float  # (1 - DSR) * 100%
    min_track_record_sessions: float  # MinTRL in sessions (or inf)
    overfitting_status: str  # ROBUST_GENUINE_ALPHA, SUSPECT_MULTIPLE_TESTING, or SEVERELY_OVERFIT
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core DSR metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.annualized_sharpe_ratio:.4f}:"
            f"{self.skewness:.4f}:{self.kurtosis:.4f}:{self.probabilistic_sharpe_ratio:.4f}:"
            f"{self.num_trials}:{self.deflated_sharpe_ratio:.4f}:"
            f"{self.is_statistically_significant}:{self.overfitting_status}"
        )
        return hmac.new(DSR_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_sharpe_standard_error(
    sr_daily: float,
    n_sessions: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Calculate Mertens (2002) standard error of the daily Sharpe ratio under non-normality.

    Formula:
        sigma_{SR} = sqrt((1 / (T - 1)) * (1 - gamma_3 * SR + ((gamma_4 - 1) / 4) * SR^2))

    Args:
        sr_daily: Estimated daily Sharpe ratio.
        n_sessions: Number of historical return observations (T).
        skewness: Skewness of daily return distribution (gamma_3).
        kurtosis: Pearson kurtosis of daily return distribution (gamma_4, normal = 3.0).

    Returns:
        Standard error sigma_{SR} > 0.
    """
    if n_sessions <= 2:
        return 1.0

    t_minus_1 = float(n_sessions - 1)
    variance_term = 1.0 - skewness * sr_daily + ((kurtosis - 1.0) / 4.0) * (sr_daily ** 2)
    # Guarantee real positive variance floor
    safe_var = max(1e-8, variance_term)
    se = math.sqrt(safe_var / t_minus_1)
    return max(1e-6, se)


def compute_probabilistic_sharpe_ratio(
    observed_sr_daily: float,
    benchmark_sr_daily: float,
    n_sessions: int,
    skewness: float,
    kurtosis: float,
) -> Tuple[float, float]:
    """Calculate the Bailey & López de Prado (2012) Probabilistic Sharpe Ratio (PSR).

    Calculates the probability that the true Sharpe ratio exceeds a benchmark SR*,
    incorporating skewness and kurtosis corrections.

    Formula:
        Z = (SR_obs - SR*) / sigma_{SR}
        PSR(SR*) = Phi(Z)

    Returns:
        Tuple of (psr_probability in [0, 1], z_score).
    """
    se = compute_sharpe_standard_error(
        sr_daily=observed_sr_daily,
        n_sessions=n_sessions,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    z_score = (observed_sr_daily - benchmark_sr_daily) / se
    psr = float(norm.cdf(z_score))
    psr = float(np.clip(psr, 0.0, 1.0))
    return psr, float(z_score)


def compute_expected_maximum_sharpe(
    num_trials: int,
    trial_variance: float,
    benchmark_sr_daily: float = 0.0,
) -> float:
    """Calculate theoretical expected maximum Sharpe ratio under null hypothesis of zero skill.

    Uses Extreme Value Theory expansion (López de Prado 2014, Eq. 6):
        E[max_N {SR}] = sqrt(V) * ((1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N * e)))

    Args:
        num_trials: Number of independent/correlated backtests or parameter trials evaluated (N).
        trial_variance: Variance of the Sharpe ratio estimates across trials (V).
        benchmark_sr_daily: Base benchmark daily Sharpe ratio (default: 0.0).

    Returns:
        Expected maximum daily Sharpe ratio under multiple testing.
    """
    n = max(1, int(num_trials))
    if n <= 1 or trial_variance <= 1e-12:
        return benchmark_sr_daily

    v = max(1e-12, float(trial_variance))
    std_v = math.sqrt(v)

    p1 = 1.0 - (1.0 / float(n))
    p2 = 1.0 - (1.0 / (float(n) * math.e))

    # Guard against numerical boundary in ppf
    p1 = min(p1, 1.0 - 1e-12)
    p2 = min(p2, 1.0 - 1e-12)

    z1 = float(norm.ppf(p1))
    z2 = float(norm.ppf(p2))

    expected_max = benchmark_sr_daily + std_v * (
        (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    )
    return float(expected_max)


def compute_deflated_sharpe_ratio(
    observed_sr_daily: float,
    num_trials: int,
    trial_variance: float,
    n_sessions: int,
    skewness: float,
    kurtosis: float,
    benchmark_sr_daily: float = 0.0,
) -> Tuple[float, float, float]:
    """Calculate the Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR).

    Evaluates PSR where the hurdle rate is set to the expected maximum Sharpe ratio
    achieved under multiple testing:
        DSR = PSR(SR* = E[max_N {SR}])

    Returns:
        Tuple of (dsr_probability in [0, 1], expected_max_daily_sr, z_score).
    """
    exp_max_daily = compute_expected_maximum_sharpe(
        num_trials=num_trials,
        trial_variance=trial_variance,
        benchmark_sr_daily=benchmark_sr_daily,
    )

    dsr, z_score = compute_probabilistic_sharpe_ratio(
        observed_sr_daily=observed_sr_daily,
        benchmark_sr_daily=exp_max_daily,
        n_sessions=n_sessions,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    return dsr, exp_max_daily, z_score


def compute_min_track_record_length(
    observed_sr_daily: float,
    benchmark_sr_daily: float,
    skewness: float,
    kurtosis: float,
    alpha: float = 0.05,
) -> float:
    """Calculate Minimum Track Record Length (MinTRL) in trading sessions.

    Formula:
        MinTRL = 1 + (1 - gamma_3 * SR + ((gamma_4 - 1) / 4) * SR^2) * (z_{1 - alpha} / (SR - SR*))^2

    Returns:
        Minimum number of trading sessions required for statistical significance at (1 - alpha),
        or float('inf') if observed_sr <= benchmark_sr.
    """
    diff = observed_sr_daily - benchmark_sr_daily
    if diff <= 1e-8:
        return float("inf")

    z_alpha = float(norm.ppf(1.0 - alpha))
    variance_term = 1.0 - skewness * observed_sr_daily + ((kurtosis - 1.0) / 4.0) * (observed_sr_daily ** 2)
    safe_var = max(1e-8, variance_term)

    mintr_sessions = 1.0 + safe_var * ((z_alpha / diff) ** 2)
    return float(max(1.0, mintr_sessions))


def evaluate_deflated_sharpe_profile(
    ticker: str,
    returns: Union[pd.Series, np.ndarray, List[float]],
    num_trials: int = 10,
    trial_sr_variance: Optional[float] = None,
    benchmark_sharpe_ann: float = 0.0,
    risk_free_rate_ann: float = 0.04,
) -> DeflatedSharpeReceipt:
    """Execute comprehensive Bailey & López de Prado DSR audit across a return series.

    Args:
        ticker: Validated stock or strategy ticker symbol.
        returns: Series of daily returns.
        num_trials: Number of model configurations or strategy trials tested (default: 10).
        trial_sr_variance: Optional variance of Sharpe ratios across trials. If None,
            calibrated to theoretical variance 1 / (T - 1).
        benchmark_sharpe_ann: Hurdle annualized Sharpe ratio (default: 0.0).
        risk_free_rate_ann: Annualized risk-free rate (default: 0.04).

    Returns:
        DeflatedSharpeReceipt containing non-normality metrics, PSR, DSR, and MinTRL.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(num_trials, "num_trials")
    assert_finite_number(benchmark_sharpe_ann, "benchmark_sharpe_ann")

    if isinstance(returns, pd.Series):
        s_ret = returns.dropna().astype(float)
    else:
        s_ret = pd.Series(returns).dropna().astype(float)

    n_sessions = len(s_ret)
    if n_sessions < 10:
        raise ValueError(
            f"Insufficient return history for {clean_ticker}: got {n_sessions} sessions, need at least 10"
        )

    # Daily risk-free rate
    rf_daily = (1.0 + risk_free_rate_ann) ** (1.0 / 252.0) - 1.0
    excess_returns = s_ret.values - rf_daily

    mean_excess = float(np.mean(excess_returns))
    std_excess = float(np.std(excess_returns, ddof=1))
    safe_std = max(1e-8, std_excess)

    daily_sr = mean_excess / safe_std
    ann_sr = daily_sr * math.sqrt(252.0)

    # Statistical moments: Skewness and Pearson Kurtosis (where normal = 3.0)
    skew = float(scipy_skew(excess_returns, bias=False))
    kurt = float(scipy_kurtosis(excess_returns, fisher=False, bias=False))

    # Benchmark daily Sharpe
    bench_daily_sr = benchmark_sharpe_ann / math.sqrt(252.0)

    # 1. Probabilistic Sharpe Ratio (PSR) vs hurdle benchmark
    psr, _ = compute_probabilistic_sharpe_ratio(
        observed_sr_daily=daily_sr,
        benchmark_sr_daily=bench_daily_sr,
        n_sessions=n_sessions,
        skewness=skew,
        kurtosis=kurt,
    )

    # 2. Deflated Sharpe Ratio (DSR) accounting for multiple trials
    if trial_sr_variance is None:
        # Default estimation variance based on track record length
        calibrated_trial_var = 1.0 / float(max(2, n_sessions - 1))
    else:
        calibrated_trial_var = max(1e-8, float(trial_sr_variance))

    dsr, exp_max_daily, _ = compute_deflated_sharpe_ratio(
        observed_sr_daily=daily_sr,
        num_trials=num_trials,
        trial_variance=calibrated_trial_var,
        n_sessions=n_sessions,
        skewness=skew,
        kurtosis=kurt,
        benchmark_sr_daily=bench_daily_sr,
    )
    exp_max_ann = exp_max_daily * math.sqrt(252.0)

    # 3. Minimum Track Record Length (MinTRL)
    mintr_sessions = compute_min_track_record_length(
        observed_sr_daily=daily_sr,
        benchmark_sr_daily=bench_daily_sr,
        skewness=skew,
        kurtosis=kurt,
        alpha=0.05,
    )

    is_sig = bool(dsr >= 0.95)
    overfit_prob = float((1.0 - dsr) * 100.0)

    # Overfitting status classification
    if dsr >= 0.95:
        overfit_status = "ROBUST_GENUINE_ALPHA"
    elif dsr >= 0.50:
        overfit_status = "SUSPECT_MULTIPLE_TESTING"
    else:
        overfit_status = "SEVERELY_OVERFIT"

    mintr_desc = f"{mintr_sessions:.0f} sessions" if not math.isinf(mintr_sessions) else "Infinite (SR <= Benchmark)"
    summary = (
        f"Deflated Sharpe Analysis for {clean_ticker} ({n_sessions} sessions, {num_trials} trials): "
        f"Ann SR: {ann_sr:.2f} (Daily: {daily_sr:.4f}, Skew: {skew:.2f}, Kurt: {kurt:.2f}). "
        f"PSR: {psr * 100:.1f}%. E[max SR]: {exp_max_ann:.2f} Ann. DSR: {dsr * 100:.1f}% "
        f"(Overfit Risk: {overfit_prob:.1f}%, Status: {overfit_status}, MinTRL: {mintr_desc})."
    )

    receipt = DeflatedSharpeReceipt(
        ticker=clean_ticker,
        sample_sessions=n_sessions,
        annualized_sharpe_ratio=ann_sr,
        daily_sharpe_ratio=daily_sr,
        skewness=skew,
        kurtosis=kurt,
        benchmark_sharpe_ann=benchmark_sharpe_ann,
        probabilistic_sharpe_ratio=psr,
        num_trials=num_trials,
        trial_sharpe_variance=calibrated_trial_var,
        expected_max_sharpe_ann=exp_max_ann,
        deflated_sharpe_ratio=dsr,
        is_statistically_significant=is_sig,
        overfitting_probability_pct=overfit_prob,
        min_track_record_sessions=mintr_sessions,
        overfitting_status=overfit_status,
        summary=summary,
    )

    return receipt
