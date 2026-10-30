"""Ang, Hodrick, Xing & Zhang (2006, 2009) Cross-Sectional Idiosyncratic Volatility Engine.

Implements the cross-sectional idiosyncratic volatility (IVOL) puzzle and higher-moment
risk decomposition from orthogonalized Fama-French (1993) and Carhart (1997) factor residuals.
Quantifies the Ang et al. (2006) anomaly penalty, idiosyncratic skewness (lottery-demand bias),
idiosyncratic kurtosis (tail-risk clustering), and downside idiosyncratic semivariance.

References:
    Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of
    volatility and expected returns. The Journal of Finance, 61(1), 259-299.
    Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2009). High idiosyncratic
    volatility and low returns: International and further U.S. evidence.
    Journal of Financial Economics, 91(1), 1-23.
    Boyer, B., Mitton, T., & Vorkink, K. (2010). Expected idiosyncratic skewness.
    The Review of Financial Studies, 23(1), 169-202.
    Harvey, C. R., & Siddique, A. (2000). Conditional skewness in asset pricing
    tests. The Journal of Finance, 55(3), 1263-1295.
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
import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

IVOL_SIGNING_KEY = b"nextomni_idiosyncratic_volatility_secret_key_v1"


@dataclass
class IdiosyncraticVolatilityReceipt:
    """Cryptographic audit receipt for Idiosyncratic Volatility & Higher Moments."""

    ticker: str
    benchmark_ticker: str
    sample_sessions: int
    lookback_window: int
    annualized_alpha_pct: float
    total_annualized_vol_pct: float
    idiosyncratic_vol_ann_pct: float  # IVOL (Ang et al., 2006)
    downside_idiosyncratic_vol_ann_pct: float  # IVOL^- (Downside semivariance)
    downside_ivol_ratio: float  # IVOL^- / IVOL
    idiosyncratic_variance_ratio_pct: float  # IVAR / TVAR
    idiosyncratic_skewness: float  # ISKEW (Boyer et al., 2010)
    idiosyncratic_kurtosis: float  # IKURT (Fat-tail excess)
    idiosyncratic_information_ratio: float  # Alpha / IVOL
    ivol_anomaly_regime: str  # LOW_IVOL_ALPHA_PREMIUM, MODERATE, ELEVATED, HIGH_IVOL_DRAG
    ivol_expected_return_haircut_pct: float  # Ang et al. anomaly penalty drag
    r_squared: float
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across IVOL risk metrics."""
        payload = (
            f"{self.ticker}:{self.benchmark_ticker}:{self.sample_sessions}:{self.lookback_window}:"
            f"{self.idiosyncratic_vol_ann_pct:.4f}:{self.downside_idiosyncratic_vol_ann_pct:.4f}:"
            f"{self.idiosyncratic_skewness:.4f}:{self.idiosyncratic_kurtosis:.4f}:"
            f"{self.ivol_expected_return_haircut_pct:.4f}:{self.ivol_anomaly_regime}"
        )
        return hmac.new(
            IVOL_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity of the idiosyncratic volatility receipt."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_idiosyncratic_higher_moments(
    residuals: np.ndarray,
    degrees_of_freedom: int,
) -> Tuple[float, float, float, float, float]:
    """Compute annualized IVOL, downside IVOL, skewness, and kurtosis from factor residuals.

    Args:
        residuals: 1D array of daily factor model residuals eps_t.
        degrees_of_freedom: Denominator degrees of freedom (T - K - 1).

    Returns:
        Tuple of (ivol_ann_pct, downside_ivol_ann_pct, downside_ratio, iskew, ikurt).
    """
    eps = np.asarray(residuals, dtype=float)
    n = len(eps)
    dof = max(1, degrees_of_freedom)

    # 1. Annualized Idiosyncratic Volatility (Ang et al., 2006)
    ss_resid = float(np.sum(eps**2))
    var_idio = ss_resid / float(dof)
    ivol_ann = math.sqrt(max(0.0, var_idio * 252.0)) * 100.0

    # 2. Downside Idiosyncratic Semivariance
    downside_eps = np.minimum(0.0, eps)
    ss_downside = float(np.sum(downside_eps**2))
    var_downside = ss_downside / float(dof)
    downside_ivol_ann = math.sqrt(max(0.0, var_downside * 252.0)) * 100.0

    # Downside Ratio: sqrt(ss_downside / ss_resid)
    downside_ratio = math.sqrt(ss_downside / max(1e-12, ss_resid))

    # Sample standard deviation for moments normalization
    std_eps = math.sqrt(max(1e-12, float(np.mean(eps**2))))

    # 3. Idiosyncratic Skewness (Boyer et al., 2010)
    # ISKEW = E[eps^3] / (E[eps^2])^(3/2)
    m3 = float(np.mean(eps**3))
    iskew = float(m3 / (std_eps**3))

    # 4. Idiosyncratic Kurtosis (Pearson kurtosis: normal = 3.0)
    m4 = float(np.mean(eps**4))
    ikurt = float(m4 / (std_eps**4))

    return ivol_ann, downside_ivol_ann, downside_ratio, iskew, ikurt


def extract_factor_residuals(
    asset_returns: Union[pd.Series, np.ndarray, List[float]],
    market_returns: Union[pd.Series, np.ndarray, List[float]],
    factor_df: Optional[pd.DataFrame] = None,
    risk_free_rate_ann: float = 0.04,
) -> Tuple[np.ndarray, float, float, float, float]:
    """Execute multivariate 4-factor regression and extract residual series.

    Model:
        r_i - r_f = alpha + beta_MKT*(r_m - r_f) + beta_SMB*SMB + beta_HML*HML + beta_UMD*UMD + eps

    Args:
        asset_returns: Asset daily return series.
        market_returns: Benchmark daily return series.
        factor_df: Optional DataFrame with columns [SMB, HML, UMD].
        risk_free_rate_ann: Annualized risk-free rate.

    Returns:
        Tuple of (residuals, alpha_ann_pct, total_vol_ann_pct, r_squared, dof).
    """
    ri = np.asarray(asset_returns, dtype=float)
    rm = np.asarray(market_returns, dtype=float)
    n = len(ri)

    if n != len(rm):
        raise ValueError(f"Length mismatch: asset ({n}) vs market ({len(rm)})")
    if n < 15:
        raise ValueError(f"Insufficient observations ({n}), need at least 15 for IVOL estimation")

    rf_daily = risk_free_rate_ann / 252.0
    y = ri - rf_daily
    mkt_excess = rm - rf_daily

    if factor_df is None or factor_df.empty or len(factor_df) != n:
        factors = construct_synthetic_factor_proxies(rm)
    else:
        factors = factor_df.copy()
        factors["MKT"] = mkt_excess

    # Design matrix X: [Intercept, MKT, SMB, HML, UMD]
    x_intercept = np.ones(n)
    X = np.column_stack([
        x_intercept,
        mkt_excess,
        factors["SMB"].values,
        factors["HML"].values,
        factors["UMD"].values,
    ])
    k_params = X.shape[1]
    dof = n - k_params

    # Robust OLS via pseudo-inverse
    beta_hat = np.linalg.pinv(X) @ y
    y_pred = X @ beta_hat
    residuals = y - y_pred

    alpha_daily = float(beta_hat[0])
    alpha_ann_pct = float(alpha_daily * 252.0 * 100.0)

    # Total return volatility
    total_vol_ann_pct = float(np.std(ri, ddof=1) * math.sqrt(252.0) * 100.0) if n > 1 else 0.0

    # R-squared
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residuals**2))
    r_squared = float(max(0.0, 1.0 - (ss_res / max(1e-12, ss_tot))))

    return residuals, alpha_ann_pct, total_vol_ann_pct, r_squared, dof


def classify_ivol_regime_and_haircut(ivol_ann_pct: float) -> Tuple[str, float]:
    """Classify Ang et al. (2006) IVOL anomaly regime and calculate expected return haircut.

    Stocks in the top quintile/decile of IVOL experience severe negative return drag (-1.06%/month).

    Args:
        ivol_ann_pct: Annualized idiosyncratic volatility in percent.

    Returns:
        Tuple of (ivol_regime_label, expected_return_haircut_pct).
    """
    if ivol_ann_pct < 18.0:
        regime = "LOW_IVOL_ALPHA_PREMIUM"
        haircut = 0.0
    elif ivol_ann_pct < 32.0:
        regime = "MODERATE_IVOL"
        haircut = float(round((ivol_ann_pct - 18.0) * 0.10, 2))
    elif ivol_ann_pct < 50.0:
        regime = "ELEVATED_IVOL"
        haircut = float(round(1.40 + (ivol_ann_pct - 32.0) * 0.25, 2))
    else:
        regime = "HIGH_IVOL_ANOMALY_DRAG"
        # Severe penalty capped at 12% per annum
        haircut = float(min(12.0, round(5.90 + (ivol_ann_pct - 50.0) * 0.35, 2)))

    return regime, haircut


def evaluate_idiosyncratic_volatility_profile(
    ticker: str,
    asset_prices: Union[pd.Series, np.ndarray, List[float]],
    benchmark_ticker: str = "SPY",
    benchmark_prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    lookback_window: int = 60,
    risk_free_rate_ann: float = 0.04,
    db_path: Optional[Path] = None,
) -> IdiosyncraticVolatilityReceipt:
    """Evaluate comprehensive Idiosyncratic Volatility (IVOL) and higher moments profile.

    Args:
        ticker: Asset ticker symbol.
        asset_prices: Historical price series for asset.
        benchmark_ticker: Benchmark ticker symbol (default: 'SPY').
        benchmark_prices: Optional benchmark price series. If None, queried from SQLite.
        lookback_window: Rolling observation window (default: 60).
        risk_free_rate_ann: Annual risk-free rate (default: 4.0%).
        db_path: Optional SQLite database path.

    Returns:
        IdiosyncraticVolatilityReceipt with HMAC-SHA256 signature.
    """
    clean_ticker = validate_ticker(ticker)
    clean_bench = validate_ticker(benchmark_ticker)
    assert_finite_number(lookback_window, "lookback_window")
    assert_finite_number(risk_free_rate_ann, "risk_free_rate_ann")

    if isinstance(asset_prices, pd.Series):
        s_asset = asset_prices.dropna().astype(float)
    else:
        s_asset = pd.Series(asset_prices, dtype=float).dropna()

    if len(s_asset) < 16:
        raise ValueError(f"Insufficient asset price observations ({len(s_asset)}), need at least 16")

    # Load benchmark prices if not provided
    if benchmark_prices is None:
        df_bench = db.get_observations(clean_bench, db_path=db_path)
        if df_bench.empty or len(df_bench) < 16:
            raise ValueError(
                f"Benchmark {clean_bench} has insufficient observations in database. Synchronize {clean_bench} first."
            )
        col = "adj_close" if "adj_close" in df_bench.columns else "close"
        s_bench = df_bench[col].dropna().astype(float)
    elif isinstance(benchmark_prices, pd.Series):
        s_bench = benchmark_prices.dropna().astype(float)
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

    total_samples = len(y_ret)
    if total_samples < 15:
        raise ValueError(f"Insufficient aligned return observations ({total_samples}), need at least 15")

    # Slice lookback window
    window = min(lookback_window, total_samples)
    y_slice = y_ret[-window:]
    m_slice = m_ret[-window:]

    # Extract 4-factor residuals
    residuals, alpha_ann, total_vol_ann, r_sq, dof = extract_factor_residuals(
        asset_returns=y_slice,
        market_returns=m_slice,
        risk_free_rate_ann=risk_free_rate_ann,
    )

    # Compute IVOL and higher moments
    ivol_ann, downside_ivol_ann, downside_ratio, iskew, ikurt = compute_idiosyncratic_higher_moments(
        residuals=residuals,
        degrees_of_freedom=dof,
    )

    # Variance Decomposition
    idio_var = (ivol_ann / 100.0) ** 2
    total_var = (total_vol_ann / 100.0) ** 2
    ivar_ratio_pct = float(min(100.0, (idio_var / max(1e-8, total_var)) * 100.0))

    # Idiosyncratic Information Ratio
    idio_ir = float(alpha_ann / max(1e-4, ivol_ann))

    # Anomaly Regime & Haircut
    regime, haircut = classify_ivol_regime_and_haircut(ivol_ann)

    summary = (
        f"Idiosyncratic Volatility Profile for {clean_ticker} vs {clean_bench} ({window} sessions): "
        f"IVOL: {ivol_ann:.2f}% Ann ({ivar_ratio_pct:.1f}% of Total Vol), "
        f"Downside IVOL: {downside_ivol_ann:.2f}% Ann (Ratio: {downside_ratio:.2f}), "
        f"ISKEW: {iskew:+.2f}, IKURT: {ikurt:.2f}. "
        f"Anomaly Regime: {regime} (Expected Return Haircut: -{haircut:.2f}%/yr)."
    )

    receipt = IdiosyncraticVolatilityReceipt(
        ticker=clean_ticker,
        benchmark_ticker=clean_bench,
        sample_sessions=total_samples,
        lookback_window=window,
        annualized_alpha_pct=alpha_ann,
        total_annualized_vol_pct=total_vol_ann,
        idiosyncratic_vol_ann_pct=ivol_ann,
        downside_idiosyncratic_vol_ann_pct=downside_ivol_ann,
        downside_ivol_ratio=downside_ratio,
        idiosyncratic_variance_ratio_pct=ivar_ratio_pct,
        idiosyncratic_skewness=iskew,
        idiosyncratic_kurtosis=ikurt,
        idiosyncratic_information_ratio=idio_ir,
        ivol_anomaly_regime=regime,
        ivol_expected_return_haircut_pct=haircut,
        r_squared=r_sq,
        summary=summary,
    )

    return receipt
