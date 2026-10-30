"""Joel Hasbrouck (1991) Vector Autoregression (VAR) Price Discovery & Information Share Engine.

Implements structural market microstructure Vector Autoregression (VAR):
- Formulates bivariate dynamic transmission system between returns (r_t) and signed trade flow (x_t):
    r_t = sum_{i=1}^p a_i * r_{t-i} + b_0 * x_t + sum_{i=1}^p b_i * x_{t-i} + v_{1,t}
    x_t = sum_{i=1}^p c_i * r_{t-i} + sum_{i=1}^p d_i * x_{t-i} + v_{2,t}
- Decomposes price response to an unexpected trade innovation (v_{2,t}) via Vector Moving Average (VMA):
    1. Contemporaneous impact: C(0) = b_0
    2. Permanent price impact: I(inf) = lim_{h -> inf} C(h) (Adverse Selection Information Content)
    3. Transient microstructure impact: S_0 = b_0 - I(inf) (Inventory friction / Bid-Ask bounce)
- Hasbrouck (1991) Random-Walk Variance Decomposition:
    sigma_w^2 = [I(inf)]^2 * sigma_{v2}^2 + sigma_{v1}^2
    R_w^2 = ( [I(inf)]^2 * sigma_{v2}^2 ) / sigma_w^2
    Quantifies the percentage of permanent fundamental volatility driven by private informed order flow.

References:
    Hasbrouck, J. (1991). Measuring the information content of stock trades.
    The Journal of Finance, 46(1), 179-207.
    Hasbrouck, J. (1991). The summary informativeness of stock trades: An econometric analysis.
    The Review of Financial Studies, 4(3), 571-595.
    Lee, C. M., & Ready, M. J. (1991). Inferring trade direction from intraday data.
    The Journal of Finance, 46(2), 733-746.
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

HASBROUCK_VAR_SIGNING_KEY = b"nextomni_hasbrouck_var_secret_key_v1"


@dataclass
class HasbrouckReceipt:
    """Cryptographic audit receipt for Hasbrouck (1991) Price Discovery & Trade Information Share."""

    ticker: str
    sample_sessions: int
    var_lags: int
    irf_horizon: int
    contemporaneous_impact_bps: float  # b_0 in basis points per 1-sigma order flow
    permanent_impact_bps: float  # I(inf) cumulative permanent price change in bps
    transient_impact_bps: float  # S_0 = b_0 - I(inf) transient bounce/inventory decay in bps
    permanent_to_total_ratio_pct: float  # (I(inf) / b_0) * 100%
    information_share_rw2_pct: float  # Hasbrouck R_w^2: trade-correlated random walk variance share
    trade_innovation_std: float  # sigma_{v2} standard deviation of unexpected order flow
    public_innovation_bps: float  # sigma_{v1} in bps (non-trade public news volatility)
    irf_curve_bps: List[float]  # Cumulative impulse response trajectory over 1..H periods
    market_quality_regime: str  # HIGH_INFORMATIONAL_ASYMMETRY, MODERATE_INFORMED_FLOW, PUBLIC_NEWS_DOMINATED
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across structural VAR metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.var_lags}:{self.irf_horizon}:"
            f"{self.contemporaneous_impact_bps:.4f}:{self.permanent_impact_bps:.4f}:"
            f"{self.transient_impact_bps:.4f}:{self.information_share_rw2_pct:.4f}:"
            f"{self.market_quality_regime}"
        )
        return hmac.new(
            HASBROUCK_VAR_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity and tamper-resistance of the Hasbrouck receipt."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def classify_market_quality_regime(information_share_pct: float) -> str:
    """Classify microstructure information asymmetry tier according to Hasbrouck standards.

    Tiers:
        R_w^2 >= 35.0%: HIGH_INFORMATIONAL_ASYMMETRY (Trades carry heavy private info; high adverse selection)
        15.0% <= R_w^2 < 35.0%: MODERATE_INFORMED_FLOW (Balanced microstructure with active trade price discovery)
        R_w^2 < 15.0%: PUBLIC_NEWS_DOMINATED (Prices driven by macro announcements; trades are mostly liquidity noise)
    """
    if information_share_pct >= 35.0:
        return "HIGH_INFORMATIONAL_ASYMMETRY"
    elif information_share_pct >= 15.0:
        return "MODERATE_INFORMED_FLOW"
    else:
        return "PUBLIC_NEWS_DOMINATED"


def construct_signed_order_flow(
    prices: Union[pd.Series, np.ndarray, List[float]],
    volumes: Union[pd.Series, np.ndarray, List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct Lee-Ready (1991) signed trade order flow proxy and percentage returns.

    Args:
        prices: Daily closing price series.
        volumes: Daily trading volume series.

    Returns:
        Tuple of (returns_array, standardized_signed_order_flow_array).
    """
    s_p = pd.Series(prices, dtype=float).dropna()
    s_v = pd.Series(volumes, dtype=float).dropna()

    if len(s_p) != len(s_v):
        min_len = min(len(s_p), len(s_v))
        s_p = s_p.iloc[-min_len:]
        s_v = s_v.iloc[-min_len:]

    if len(s_p) < 20:
        raise ValueError(f"Insufficient observations ({len(s_p)}), need at least 20")

    p = s_p.values
    v = s_v.values

    # Daily price returns
    ret = np.diff(p) / np.maximum(p[:-1], 1e-8)
    dp = np.diff(p)
    v_eff = v[1:]
    p_eff = p[:-1]

    # Lee-Ready tick rule for trade sign
    signs = np.sign(dp)
    # Zero tick resolution: carry forward previous non-zero sign
    for t in range(1, len(signs)):
        if signs[t] == 0:
            signs[t] = signs[t - 1]
    if signs[0] == 0:
        signs[0] = 1.0

    # Signed dollar volume proxy: sign * sqrt(P * V)
    dollar_vol = p_eff * v_eff
    raw_flow = signs * np.sqrt(np.maximum(1e-4, dollar_vol))

    # Standardize order flow to zero mean, unit variance
    flow_std = float(np.std(raw_flow, ddof=1))
    if flow_std > 1e-8:
        x_norm = (raw_flow - np.mean(raw_flow)) / flow_std
    else:
        x_norm = np.zeros_like(raw_flow)

    return ret, x_norm


def estimate_hasbrouck_var(
    returns: np.ndarray,
    trade_flows: np.ndarray,
    lags: int = 5,
    irf_horizon: int = 20,
) -> Tuple[float, float, float, float, float, float, float, List[float], str]:
    """Estimate Joel Hasbrouck (1991) structural VAR and simulate VMA impulse responses.

    Args:
        returns: 1D array of daily price returns.
        trade_flows: 1D array of standardized signed order flows.
        lags: Number of autoregressive lags p in VAR (default: 5).
        irf_horizon: Horizon H for impulse response function simulation (default: 20).

    Returns:
        Tuple of:
            contemporaneous_impact_bps (b_0 in bps)
            permanent_impact_bps (I(inf) in bps)
            transient_impact_bps (S_0 in bps)
            permanent_ratio_pct (%)
            information_share_rw2_pct (Hasbrouck R_w^2 in %)
            trade_innovation_std (sigma_{v2})
            public_innovation_bps (sigma_{v1} in bps)
            irf_curve_bps (List of cumulative impulse responses)
            market_quality_regime (str)
    """
    assert_finite_number(lags, "lags")
    assert_finite_number(irf_horizon, "irf_horizon")

    p = int(max(1, min(10, lags)))
    h_steps = int(max(5, min(50, irf_horizon)))

    r = np.asarray(returns, dtype=float)
    x = np.asarray(trade_flows, dtype=float)
    n = len(r)

    if n != len(x):
        raise ValueError(f"Length mismatch: returns ({n}) vs trade_flows ({len(x)})")
    if n < p * 4 + 10:
        raise ValueError(f"Insufficient observations ({n}) for {p} lags; need at least {p * 4 + 10}")

    # Construct VAR Design Matrices
    # Equation 1: r_t on [1, r_{t-1..p}, x_t, x_{t-1..p}]
    lagged_r = [r[p - i : n - i] for i in range(1, p + 1)]
    contemp_x = [x[p:n]]
    lagged_x = [x[p - i : n - i] for i in range(1, p + 1)]

    Z_r = np.column_stack([np.ones(n - p)] + lagged_r + contemp_x + lagged_x)
    y_r = r[p:n]

    # Equation 2: x_t on [1, r_{t-1..p}, x_{t-1..p}]
    Z_x = np.column_stack([np.ones(n - p)] + lagged_r + lagged_x)
    y_x = x[p:n]

    # Ridge regularized least-squares for high numerical conditioning
    ridge_alpha = 1e-6
    I_r = np.eye(Z_r.shape[1])
    I_r[0, 0] = 0.0  # Do not penalize constant
    beta_r = np.linalg.solve(Z_r.T @ Z_r + ridge_alpha * I_r, Z_r.T @ y_r)

    I_x = np.eye(Z_x.shape[1])
    I_x[0, 0] = 0.0
    beta_x = np.linalg.solve(Z_x.T @ Z_x + ridge_alpha * I_x, Z_x.T @ y_x)

    # Extract coefficients
    # beta_r: [const, a_1..a_p, b_0, b_1..b_p]
    a_coefs = beta_r[1 : p + 1]
    b0 = float(beta_r[p + 1])
    b_coefs = beta_r[p + 2 : 2 * p + 2]

    # beta_x: [const, c_1..c_p, d_1..d_p]
    c_coefs = beta_x[1 : p + 1]
    d_coefs = beta_x[p + 1 : 2 * p + 1]

    # Innovations
    v1 = y_r - (Z_r @ beta_r)
    v2 = y_x - (Z_x @ beta_x)

    sigma_v1 = float(np.std(v1, ddof=1))
    sigma_v2 = float(np.std(v2, ddof=1))

    # Simulate Impulse Response Function to 1-sigma unexpected trade innovation: v_{2,0} = 1.0
    r_sim = np.zeros(h_steps + p)
    x_sim = np.zeros(h_steps + p)

    # Shock at t = p (step k = 0)
    x_sim[p] = 1.0
    r_sim[p] = b0 * x_sim[p]

    for k in range(1, h_steps):
        t = p + k
        # Dynamic transmission of lagged states
        past_r = r_sim[t - p : t][::-1]
        past_x = x_sim[t - p : t][::-1]

        x_sim[t] = float(np.dot(c_coefs, past_r) + np.dot(d_coefs, past_x))
        r_sim[t] = float(np.dot(a_coefs, past_r) + b0 * x_sim[t] + np.dot(b_coefs, past_x))

    # Cumulative impulse response trajectory in basis points
    cum_irf_raw = np.cumsum(r_sim[p:])
    cum_irf_bps = [float(val * 10000.0) for val in cum_irf_raw]

    # Contemporaneous and permanent impacts
    contemp_bps = float(b0 * 10000.0)
    perm_bps = float(cum_irf_bps[-1])
    transient_bps = float(contemp_bps - perm_bps)

    # Permanent to total impact ratio
    if abs(contemp_bps) > 1e-4:
        perm_ratio_pct = float((perm_bps / contemp_bps) * 100.0)
    else:
        perm_ratio_pct = 100.0

    # Hasbrouck Random-Walk Variance Decomposition:
    # sigma_w^2 = [I(inf)]^2 * sigma_{v2}^2 + sigma_{v1}^2
    perm_raw = cum_irf_raw[-1]
    trade_info_var = (perm_raw**2) * (sigma_v2**2)
    public_info_var = sigma_v1**2
    total_rw_var = trade_info_var + public_info_var

    if total_rw_var > 1e-12:
        rw2_pct = float((trade_info_var / total_rw_var) * 100.0)
    else:
        rw2_pct = 0.0

    rw2_pct = max(0.0, min(100.0, rw2_pct))
    public_bps = float(sigma_v1 * 10000.0)

    regime = classify_market_quality_regime(rw2_pct)

    return (
        contemp_bps,
        perm_bps,
        transient_bps,
        perm_ratio_pct,
        rw2_pct,
        sigma_v2,
        public_bps,
        cum_irf_bps,
        regime,
    )


def evaluate_hasbrouck_var_profile(
    ticker: str,
    asset_prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    asset_volumes: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    var_lags: int = 5,
    irf_horizon: int = 20,
    lookback_window: int = 120,
    db_path: Optional[Path] = None,
) -> HasbrouckReceipt:
    """Evaluate comprehensive Joel Hasbrouck (1991) VAR price discovery and trade information share.

    Args:
        ticker: Asset ticker symbol.
        asset_prices: Optional closing price series.
        asset_volumes: Optional volume series.
        var_lags: Number of autoregressive lags in VAR (default: 5).
        irf_horizon: Horizon for impulse response function curve (default: 20).
        lookback_window: Rolling window in sessions to estimate VAR (default: 120).
        db_path: Optional SQLite database path.

    Returns:
        HasbrouckReceipt with cryptographic HMAC-SHA256 signature and tamper verification.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(var_lags, "var_lags")
    assert_finite_number(irf_horizon, "irf_horizon")
    assert_finite_number(lookback_window, "lookback_window")

    # Load data if not supplied
    if asset_prices is None or asset_volumes is None:
        df_obs = db.get_observations(clean_ticker, db_path=db_path)
        if df_obs.empty or len(df_obs) < 30:
            raise ValueError(
                f"Insufficient observation history for {clean_ticker}. Synchronize ticker first."
            )
        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
        v_col = "volume"
        s_p = df_obs[p_col].dropna().astype(float)
        s_v = df_obs[v_col].dropna().astype(float)
    else:
        s_p = pd.Series(asset_prices, dtype=float).dropna()
        s_v = pd.Series(asset_volumes, dtype=float).dropna()

    total_sessions = min(len(s_p), len(s_v))
    if total_sessions < 30:
        raise ValueError(f"Insufficient aligned sessions ({total_sessions}), need at least 30")

    # Slice lookback window
    window = min(lookback_window, total_sessions)
    p_slice = s_p.iloc[-window:].values
    v_slice = s_v.iloc[-window:].values

    # Construct returns and standardized signed order flow
    rets, flows = construct_signed_order_flow(p_slice, v_slice)

    # Estimate Hasbrouck VAR
    (
        contemp_bps,
        perm_bps,
        transient_bps,
        perm_ratio_pct,
        rw2_pct,
        sig_v2,
        pub_bps,
        irf_curve,
        regime,
    ) = estimate_hasbrouck_var(
        returns=rets,
        trade_flows=flows,
        lags=var_lags,
        irf_horizon=irf_horizon,
    )

    summary = (
        f"{clean_ticker} Hasbrouck (1991) VAR price discovery: Contemporaneous impact {contemp_bps:+.2f} bps, "
        f"Permanent informational impact {perm_bps:+.2f} bps, Transient noise bounce {transient_bps:+.2f} bps. "
        f"Trade information variance share (R_w^2): {rw2_pct:.1f}% of random walk volatility. "
        f"Regime: {regime}."
    )

    return HasbrouckReceipt(
        ticker=clean_ticker,
        sample_sessions=len(rets),
        var_lags=var_lags,
        irf_horizon=irf_horizon,
        contemporaneous_impact_bps=contemp_bps,
        permanent_impact_bps=perm_bps,
        transient_impact_bps=transient_bps,
        permanent_to_total_ratio_pct=perm_ratio_pct,
        information_share_rw2_pct=rw2_pct,
        trade_innovation_std=sig_v2,
        public_innovation_bps=pub_bps,
        irf_curve_bps=irf_curve,
        market_quality_regime=regime,
        summary=summary,
    )
