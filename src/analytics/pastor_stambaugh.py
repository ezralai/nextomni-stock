"""Pástor & Stambaugh (2003) Systematic Liquidity Risk Factor & Liquidity Beta Engine.

Implements the market-wide systematic liquidity risk factor, order flow reversal price impact
parameter (gamma), autoregressive innovation filtering, and two-factor Liquidity Beta (beta_L).
Quantifies the systematic liquidity premium (7.5% per annum spread) and flight-to-safety resilience.

References:
    Pástor, L., & Stambaugh, R. F. (2003). Liquidity risk and expected stock returns.
    Journal of Political Economy, 111(3), 642-685.
    Campbell, J. Y., Grossman, S. J., & Wang, J. (1993). Trading volume and serial
    correlation in stock returns. The Quarterly Journal of Economics, 108(4), 905-939.
    Acharya, V. V., & Pedersen, L. H. (2005). Asset pricing with liquidity risk.
    Journal of Financial Economics, 77(2), 375-410.
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

PASTOR_STAMBAUGH_SIGNING_KEY = b"nextomni_pastor_stambaugh_secret_key_v1"


@dataclass
class PastorStambaughReceipt:
    """Cryptographic audit receipt for Pástor-Stambaugh Systematic Liquidity Risk."""

    ticker: str
    benchmark_ticker: str
    sample_sessions: int
    lookback_window: int
    market_beta: float
    liquidity_beta: float  # beta_L (Sensitivity to systematic liquidity shocks)
    liquidity_beta_t_stat: float
    liquidity_beta_p_value: float
    is_liquidity_beta_significant: bool  # p < 0.05
    annualized_alpha_pct: float
    average_price_impact_gamma: float  # Order flow reversal coefficient gamma * 10^6
    liquidity_risk_regime: str  # HIGH_LIQUIDITY_BETA_PREMIUM, NEUTRAL, LIQUIDITY_HEDGE_DEFENSIVE
    expected_liquidity_premium_pct: float  # Annualized premium (e.g. beta_L * 7.5%)
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
        """Compute deterministic HMAC-SHA256 signature across liquidity risk metrics."""
        payload = (
            f"{self.ticker}:{self.benchmark_ticker}:{self.sample_sessions}:{self.lookback_window}:"
            f"{self.market_beta:.4f}:{self.liquidity_beta:.4f}:{self.liquidity_beta_t_stat:.4f}:"
            f"{self.annualized_alpha_pct:.4f}:{self.average_price_impact_gamma:.4f}:"
            f"{self.expected_liquidity_premium_pct:.4f}:{self.liquidity_risk_regime}"
        )
        return hmac.new(
            PASTOR_STAMBAUGH_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity of the Pástor-Stambaugh liquidity receipt."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_stock_liquidity_gamma(
    returns: np.ndarray,
    market_returns: np.ndarray,
    volumes: np.ndarray,
    prices: np.ndarray,
) -> float:
    """Estimate Campbell-Grossman-Wang / Pástor-Stambaugh order flow reversal gamma.

    Model:
        r_{i,t+1}^e = theta_i + phi_i * r_{i,t} + gamma_i * sign(r_{i,t}^e) * v_{i,t} + eps_{i,t+1}

    where:
        r_{i,t}^e = r_{i,t} - r_{m,t} is excess return over market benchmark.
        v_{i,t} is normalized dollar volume (in millions).
        gamma_i < 0 reflects order-flow price impact reversal (illiquidity).

    Args:
        returns: Daily asset returns (T,).
        market_returns: Daily market returns (T,).
        volumes: Daily share trading volume (T,).
        prices: Daily stock close prices (T,).

    Returns:
        Estimated price impact parameter gamma (scaled by 10^6 for readability).
    """
    n = len(returns)
    if n < 10:
        return 0.0

    r_excess = returns - market_returns
    # Dollar volume in millions: (price * volume) / 1e6
    dollar_vol = (prices * volumes) / 1e6
    mean_vol = max(1e-4, float(np.mean(dollar_vol)))
    # Normalized dollar volume
    v_norm = dollar_vol / mean_vol

    # Predictors at time t: [Intercept, r_{i,t}, sign(r_{i,t}^e) * v_{i,t}]
    signed_flow = np.sign(r_excess[:-1]) * v_norm[:-1]
    y = r_excess[1:]  # r_{i,t+1}^e
    x_ret = returns[:-1]
    x_const = np.ones(n - 1)

    X = np.column_stack([x_const, x_ret, signed_flow])

    try:
        beta = np.linalg.pinv(X) @ y
        gamma = float(beta[2])
    except Exception:
        gamma = 0.0

    return gamma


def extract_liquidity_innovations(
    liquidity_series: np.ndarray,
) -> np.ndarray:
    """Extract unexpected market liquidity innovations via AR(2) filtering.

    Model:
        L_t = c_0 + c_1 * L_{t-1} + c_2 * L_{t-2} + u_t
    where u_t is the unforecasted liquidity innovation shock.

    Args:
        liquidity_series: 1D array of rolling liquidity measures L_t.

    Returns:
        Standardized innovation series u_t^* (mean 0, unit variance).
    """
    L = np.asarray(liquidity_series, dtype=float)
    n = len(L)
    if n < 12:
        # Fallback to simple differencing
        diff = np.diff(L, prepend=L[0])
        std_d = max(1e-6, float(np.std(diff, ddof=1))) if n > 1 else 1.0
        return diff / std_d

    # AR(2) model on L
    y = L[2:]
    X = np.column_stack([np.ones(n - 2), L[1:-1], L[:-2]])

    try:
        coef = np.linalg.pinv(X) @ y
        y_pred = X @ coef
        u = y - y_pred
        # Prepend two zeros for shape alignment
        innovations = np.concatenate([[0.0, 0.0], u])
    except Exception:
        innovations = np.diff(L, prepend=L[0])

    std_u = max(1e-6, float(np.std(innovations[2:], ddof=1))) if n > 3 else 1.0
    return innovations / std_u


def compute_liquidity_beta(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    liquidity_innovations: np.ndarray,
    risk_free_rate_ann: float = 0.04,
) -> Tuple[float, float, float, float, bool, float, float]:
    """Execute two-factor OLS regression to estimate Liquidity Beta (beta_L).

    Model:
        r_{i,t} - r_f = alpha + beta_M * (r_{m,t} - r_f) + beta_L * u_t^* + eps_t

    Args:
        asset_returns: Daily asset return series.
        market_returns: Daily market benchmark return series.
        liquidity_innovations: Standardized liquidity innovation series u_t^*.
        risk_free_rate_ann: Annual risk-free rate (default: 4.0%).

    Returns:
        Tuple of:
            (alpha_ann_pct, beta_M, beta_L, t_stat_beta_L, is_sig, p_val_beta_L, r_squared).
    """
    n = len(asset_returns)
    rf_daily = risk_free_rate_ann / 252.0
    y = asset_returns - rf_daily
    mkt_excess = market_returns - rf_daily
    u_norm = liquidity_innovations

    X = np.column_stack([np.ones(n), mkt_excess, u_norm])
    k_params = 3
    dof = max(1, n - k_params)

    # OLS estimation
    beta_hat = np.linalg.pinv(X) @ y
    alpha_daily = float(beta_hat[0])
    beta_M = float(beta_hat[1])
    beta_L = float(beta_hat[2])

    # White HC0 robust covariance matrix
    residuals = y - (X @ beta_hat)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residuals**2))
    r_sq = float(max(0.0, 1.0 - (ss_res / max(1e-12, ss_tot))))

    XtX_inv = np.linalg.pinv(X.T @ X)
    S = np.zeros((k_params, k_params))
    for t in range(n):
        x_t = X[t : t + 1]
        S += (residuals[t] ** 2) * (x_t.T @ x_t)
    hc0_cov = XtX_inv @ S @ XtX_inv

    se_beta_L = math.sqrt(max(1e-12, float(hc0_cov[2, 2])))
    t_stat_beta_L = float(beta_L / se_beta_L)

    # Two-sided Gaussian p-value approximation
    z = abs(t_stat_beta_L)
    p_val_beta_L = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))
    is_sig = p_val_beta_L < 0.05

    alpha_ann_pct = float(alpha_daily * 252.0 * 100.0)

    return alpha_ann_pct, beta_M, beta_L, t_stat_beta_L, is_sig, p_val_beta_L, r_sq


def classify_pastor_stambaugh_regime(
    liquidity_beta: float,
) -> Tuple[str, float]:
    """Classify systematic liquidity risk exposure regime and expected premium.

    Based on Pástor & Stambaugh (2003)'s empirical 7.5% annualized liquidity decile spread.

    Args:
        liquidity_beta: Asset's sensitivity to market liquidity shocks (beta_L).

    Returns:
        Tuple of (liquidity_risk_regime, expected_liquidity_premium_pct).
    """
    expected_premium = float(round(liquidity_beta * 7.50, 2))

    if liquidity_beta > 0.50:
        regime = "HIGH_LIQUIDITY_BETA_PREMIUM"
    elif liquidity_beta < -0.20:
        regime = "LIQUIDITY_HEDGE_DEFENSIVE"
    else:
        regime = "NEUTRAL_LIQUIDITY_PROFILE"

    return regime, expected_premium


def evaluate_pastor_stambaugh_profile(
    ticker: str,
    asset_prices: Union[pd.Series, np.ndarray, List[float]],
    asset_volumes: Union[pd.Series, np.ndarray, List[float]],
    benchmark_ticker: str = "SPY",
    benchmark_prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    benchmark_volumes: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    lookback_window: int = 60,
    risk_free_rate_ann: float = 0.04,
    db_path: Optional[Path] = None,
) -> PastorStambaughReceipt:
    """Evaluate complete Pástor-Stambaugh Systematic Liquidity Risk & Beta Profile.

    Args:
        ticker: Asset ticker symbol.
        asset_prices: Daily closing prices.
        asset_volumes: Daily share trading volume.
        benchmark_ticker: Benchmark ticker symbol (default: 'SPY').
        benchmark_prices: Optional benchmark prices series.
        benchmark_volumes: Optional benchmark volumes series.
        lookback_window: Rolling observation window in sessions (default: 60).
        risk_free_rate_ann: Annual risk-free rate (default: 4.0%).
        db_path: Optional SQLite database path.

    Returns:
        PastorStambaughReceipt with HMAC-SHA256 signature and tamper detection.
    """
    clean_ticker = validate_ticker(ticker)
    clean_bench = validate_ticker(benchmark_ticker)
    assert_finite_number(lookback_window, "lookback_window")
    assert_finite_number(risk_free_rate_ann, "risk_free_rate_ann")

    s_p = pd.Series(asset_prices, dtype=float).dropna()
    s_v = pd.Series(asset_volumes, dtype=float).dropna()
    min_len = min(len(s_p), len(s_v))
    if min_len < 20:
        raise ValueError(f"Insufficient asset price/volume history ({min_len}), need at least 20 sessions")
    s_p = s_p.iloc[-min_len:]
    s_v = s_v.iloc[-min_len:]

    # Load benchmark data if not provided
    if benchmark_prices is None or benchmark_volumes is None:
        df_bench = db.get_observations(clean_bench, db_path=db_path)
        if df_bench.empty or len(df_bench) < 20:
            raise ValueError(f"Insufficient benchmark observations for {clean_bench}. Synchronize {clean_bench} first.")
        p_col = "adj_close" if "adj_close" in df_bench.columns else "close"
        b_p = df_bench[p_col].dropna().astype(float)
        b_v = df_bench["volume"].dropna().astype(float)
    else:
        b_p = pd.Series(benchmark_prices, dtype=float).dropna()
        b_v = pd.Series(benchmark_volumes, dtype=float).dropna()

    aligned_len = min(len(s_p), len(b_p), len(b_v))
    if aligned_len < 20:
        raise ValueError(f"Insufficient aligned observations ({aligned_len}) between {clean_ticker} and {clean_bench}")

    # Align series
    p_asset = s_p.values[-aligned_len:]
    v_asset = s_v.values[-aligned_len:]
    p_mkt = b_p.values[-aligned_len:]
    v_mkt = b_v.values[-aligned_len:]

    r_asset = np.diff(p_asset) / p_asset[:-1]
    r_mkt = np.diff(p_mkt) / p_mkt[:-1]
    v_asset_sl = v_asset[1:]
    p_asset_sl = p_asset[1:]
    v_mkt_sl = v_mkt[1:]
    p_mkt_sl = p_mkt[1:]

    total_sessions = len(r_asset)
    window = min(lookback_window, total_sessions)

    # Slice lookback window
    r_a_win = r_asset[-window:]
    r_m_win = r_mkt[-window:]
    v_a_win = v_asset_sl[-window:]
    p_a_win = p_asset_sl[-window:]
    v_m_win = v_mkt_sl[-window:]
    p_m_win = p_mkt_sl[-window:]

    # 1. Compute stock-level order flow reversal gamma
    gamma_stock = compute_stock_liquidity_gamma(
        returns=r_a_win,
        market_returns=r_m_win,
        volumes=v_a_win,
        prices=p_a_win,
    )

    # 2. Compute rolling market liquidity series (Amihud order flow proxy)
    # Market dollar volume in millions
    mkt_dollar_vol = (p_m_win * v_m_win) / 1e6
    safe_mkt_dvol = np.maximum(1e-4, mkt_dollar_vol)
    # Daily market price impact measure L_t = - |r_m| / DollarVol (scaled)
    # Negative because high price movement per dollar traded = low liquidity
    mkt_liquidity = -(np.abs(r_m_win) / safe_mkt_dvol) * 1e3

    # 3. Extract market liquidity innovations via AR(2) filtering
    u_innovations = extract_liquidity_innovations(mkt_liquidity)

    # 4. Compute Liquidity Beta (beta_L)
    alpha_ann, beta_M, beta_L, t_stat, is_sig, p_val, r_sq = compute_liquidity_beta(
        asset_returns=r_a_win,
        market_returns=r_m_win,
        liquidity_innovations=u_innovations,
        risk_free_rate_ann=risk_free_rate_ann,
    )

    # 5. Classify regime & expected liquidity premium
    regime, expected_prem = classify_pastor_stambaugh_regime(beta_L)

    sig_label = "YES (p < 0.05)" if is_sig else "NO (p >= 0.05)"
    summary = (
        f"Pástor-Stambaugh Liquidity Risk Profile for {clean_ticker} vs {clean_bench} ({window} sessions): "
        f"Liquidity Beta (β_L): {beta_L:+.3f} (t={t_stat:+.2f}, Sig: {sig_label}), "
        f"Market Beta (β_M): {beta_M:.2f}, Alpha: {alpha_ann:+.2f}% Ann. "
        f"Order Flow Gamma: {gamma_stock:+.4f}. "
        f"Regime: {regime} (Expected Liquidity Premium: {expected_prem:+.2f}%/yr)."
    )

    receipt = PastorStambaughReceipt(
        ticker=clean_ticker,
        benchmark_ticker=clean_bench,
        sample_sessions=total_sessions,
        lookback_window=window,
        market_beta=beta_M,
        liquidity_beta=beta_L,
        liquidity_beta_t_stat=t_stat,
        liquidity_beta_p_value=p_val,
        is_liquidity_beta_significant=is_sig,
        annualized_alpha_pct=alpha_ann,
        average_price_impact_gamma=gamma_stock,
        liquidity_risk_regime=regime,
        expected_liquidity_premium_pct=expected_prem,
        r_squared=r_sq,
        summary=summary,
    )

    return receipt
