"""Cross-Sectional Multi-Factor Risk Attribution & Style Exposure Engine for NextOmni Stock.

Implements institutional factor risk decomposition and style attribution:
- Leak-free style factor return generation across local investment universes:
  - Market benchmark (MKT)
  - Momentum (WML - Winners Minus Losers)
  - Low Volatility (VOL - Low Volatility Minus High Volatility)
  - Short-Term Mean Reversion (REV - Short-term Losers Minus Winners)
- Robust OLS factor regression with Newey-West Heteroskedasticity and Autocorrelation
  Consistent (HAC) covariance estimators.
- Euler exact variance decomposition: Total Risk = Systematic Factor Risk + Idiosyncratic Specific Risk.
- Multi-asset portfolio factor risk decomposition and percentage risk budgeting.
- Cryptographic HMAC-SHA256 audit receipts guaranteeing analytical provenance.
"""

from dataclasses import asdict, dataclass
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.portfolio import build_aligned_returns_matrix
from src.data.db import get_all_stocks
from src.security.validators import validate_exchange, validate_ticker


FACTOR_SIGNING_KEY = b"nextomni_factor_attribution_audit_key_v1"


@dataclass
class FactorAttributionReceipt:
    """Cryptographic diagnostic receipt summarizing multi-factor risk attribution."""
    ticker: str
    exchange: str
    sample_sessions: int
    r_squared: float
    adj_r_squared: float
    annualized_alpha_pct: float
    alpha_t_stat: float
    alpha_p_value: float
    factor_betas: Dict[str, float]
    factor_t_stats: Dict[str, float]
    factor_p_values: Dict[str, float]
    factor_risk_contributions_pct: Dict[str, float]
    systemic_risk_share_pct: float
    specific_risk_share_pct: float
    total_volatility_ann_pct: float
    idiosyncratic_volatility_ann_pct: float
    summary: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core fields."""
        payload = (
            f"{self.ticker}:{self.exchange}:{self.sample_sessions}:"
            f"{self.r_squared:.4f}:{self.adj_r_squared:.4f}:{self.annualized_alpha_pct:.4f}:"
            f"{self.systemic_risk_share_pct:.4f}:{self.specific_risk_share_pct:.4f}"
        )
        return hmac.new(FACTOR_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt provenance against tamper modification."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def construct_style_factor_returns(
    returns_df: pd.DataFrame,
    window_mom: int = 21,
    window_vol: int = 21,
    window_rev: int = 5,
) -> pd.DataFrame:
    """Construct leak-free style factor proxy return series from asset universe returns.

    Guarantees strict zero-lookahead:
    Portfolio sorting and ranking at session t are determined strictly using returns
    up to session t-1. The realized return at session t is the cross-sectional
    spread return observed at session t.

    Factors:
    - MKT: Cross-sectional equal-weighted market return at session t.
    - MOM: Long top-half winners minus short bottom-half losers based on returns from t-window_mom to t-1.
    - VOL: Long top-half low-volatility minus short bottom-half high-volatility based on stdev from t-window_vol to t-1.
    - REV: Long top-half short-term losers minus short top-half short-term winners from t-window_rev to t-1.

    Args:
        returns_df: Clean DataFrame of aligned daily returns (rows=sessions, columns=tickers).
        window_mom: Momentum formation lookback window (default: 21 sessions).
        window_vol: Realized volatility formation lookback window (default: 21 sessions).
        window_rev: Short-term reversal lookback window (default: 5 sessions).

    Returns:
        pd.DataFrame with columns ['MKT', 'MOM', 'VOL', 'REV'] indexed by session dates.
    """
    if returns_df.empty or len(returns_df.columns) < 2:
        return pd.DataFrame(columns=["MKT", "MOM", "VOL", "REV"])

    clean_returns = returns_df.copy().dropna(how="all")
    n_sessions, n_assets = clean_returns.shape
    min_warmup = max(window_mom, window_vol, window_rev) + 1

    if n_sessions < min_warmup:
        return pd.DataFrame(columns=["MKT", "MOM", "VOL", "REV"])

    mkt_returns = clean_returns.mean(axis=1)

    mom_list: List[float] = []
    vol_list: List[float] = []
    rev_list: List[float] = []
    valid_indices: List[Any] = []

    for t in range(min_warmup, n_sessions):
        # Strictly look back up to t-1 (zero lookahead)
        hist_mom = clean_returns.iloc[t - window_mom:t]
        cum_ret = (1.0 + hist_mom).prod(axis=0) - 1.0

        hist_vol = clean_returns.iloc[t - window_vol:t]
        std_ret = hist_vol.std(axis=0, ddof=1).replace(0.0, np.nan).fillna(1e-4)

        hist_rev = clean_returns.iloc[t - window_rev:t]
        cum_rev = (1.0 + hist_rev).prod(axis=0) - 1.0

        current_returns = clean_returns.iloc[t]

        # 1. Momentum: Long Winners, Short Losers
        median_mom = cum_ret.median()
        winners = current_returns[cum_ret >= median_mom]
        losers = current_returns[cum_ret < median_mom]
        ret_mom = winners.mean() - losers.mean() if (not winners.empty and not losers.empty) else 0.0

        # 2. Low Volatility: Long Low-Vol, Short High-Vol
        median_vol = std_ret.median()
        low_vol = current_returns[std_ret <= median_vol]
        high_vol = current_returns[std_ret > median_vol]
        ret_vol = low_vol.mean() - high_vol.mean() if (not low_vol.empty and not high_vol.empty) else 0.0

        # 3. Short-term Reversal: Long Past Losers, Short Past Winners
        median_rev = cum_rev.median()
        st_losers = current_returns[cum_rev <= median_rev]
        st_winners = current_returns[cum_rev > median_rev]
        ret_rev = st_losers.mean() - st_winners.mean() if (not st_losers.empty and not st_winners.empty) else 0.0

        mom_list.append(float(ret_mom))
        vol_list.append(float(ret_vol))
        rev_list.append(float(ret_rev))
        valid_indices.append(clean_returns.index[t])

    factor_df = pd.DataFrame(
        {
            "MKT": mkt_returns.loc[valid_indices].values,
            "MOM": mom_list,
            "VOL": vol_list,
            "REV": rev_list,
        },
        index=valid_indices,
    )
    return factor_df.dropna()


def estimate_factor_betas(
    asset_returns: Union[pd.Series, np.ndarray],
    factor_returns: pd.DataFrame,
    max_lags: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate multi-factor beta loadings using OLS with Newey-West HAC covariance.

    Solves:
        r_i,t = alpha_i + sum_k (beta_i,k * F_k,t) + epsilon_i,t

    Newey-West HAC standard errors guard against serial autocorrelation and
    heteroskedasticity in time-series residuals.

    Args:
        asset_returns: Historical return series for the target asset.
        factor_returns: DataFrame of factor return series (columns: factor names).
        max_lags: Optional truncation lag for Newey-West HAC kernel. If None,
                  uses standard Andrews/Stock-Watson rule: floor(4 * (T/100)^(2/9)).

    Returns:
        Dictionary containing beta loadings, t-statistics, p-values, R-squared,
        and Euler exact variance decomposition into factor vs specific risk.
    """
    if isinstance(asset_returns, np.ndarray):
        y_series = pd.Series(asset_returns)
    else:
        y_series = asset_returns.copy()

    # Align indexes
    combined = pd.concat([y_series.rename("Y"), factor_returns], axis=1, join="inner").dropna()
    t_obs = len(combined)
    factors = [col for col in combined.columns if col != "Y"]
    k_factors = len(factors)

    if t_obs < k_factors + 5:
        # Insufficient data fallback
        empty_betas = {f: 0.0 for f in factors}
        return {
            "sample_sessions": t_obs,
            "r_squared": 0.0,
            "adj_r_squared": 0.0,
            "alpha": 0.0,
            "annualized_alpha_pct": 0.0,
            "alpha_t_stat": 0.0,
            "alpha_p_value": 1.0,
            "factor_betas": empty_betas,
            "factor_t_stats": {f: 0.0 for f in factors},
            "factor_p_values": {f: 1.0 for f in factors},
            "factor_risk_contributions_pct": {f: 0.0 for f in factors},
            "systemic_risk_share_pct": 0.0,
            "specific_risk_share_pct": 100.0,
            "total_volatility_ann_pct": 0.0,
            "idiosyncratic_volatility_ann_pct": 0.0,
        }

    y = combined["Y"].values
    f_matrix = combined[factors].values

    # Construct design matrix X = [1, F]
    x_matrix = np.column_stack([np.ones(t_obs), f_matrix])
    k_params = k_factors + 1

    # Solve OLS via regularized normal equations
    xtx = x_matrix.T @ x_matrix
    # Condition guard
    eps_reg = 1e-10 * np.eye(k_params)
    xty = x_matrix.T @ y

    try:
        beta_hat = np.linalg.solve(xtx + eps_reg, xty)
    except np.linalg.LinAlgError:
        beta_hat = np.linalg.lstsq(x_matrix, y, rcond=None)[0]

    y_pred = x_matrix @ beta_hat
    residuals = y - y_pred

    # R-squared and Adjusted R-squared
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r_squared = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot))) if ss_tot > 1e-12 else 0.0
    adj_r_squared = max(0.0, 1.0 - (1.0 - r_squared) * ((t_obs - 1) / max(1, t_obs - k_params)))

    # Newey-West HAC Covariance Matrix
    if max_lags is None:
        l_lag = max(1, int(4.0 * ((t_obs / 100.0) ** (2.0 / 9.0))))
    else:
        l_lag = max(0, min(max_lags, t_obs - 2))

    # S_0 term
    u_x = residuals[:, None] * x_matrix
    s_mat = (u_x.T @ u_x) / t_obs

    # Weighted lag cross-products
    for lag in range(1, l_lag + 1):
        weight = 1.0 - (lag / (l_lag + 1.0))
        gamma_l = (u_x[lag:].T @ u_x[:-lag]) / t_obs
        s_mat += weight * (gamma_l + gamma_l.T)

    xtx_inv = np.linalg.pinv(xtx / t_obs)
    var_beta = (xtx_inv @ s_mat @ xtx_inv) / t_obs
    se_beta = np.sqrt(np.maximum(1e-12, np.diag(var_beta)))

    t_stats = beta_hat / se_beta
    df_resid = max(1, t_obs - k_params)
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=df_resid)

    alpha_val = float(beta_hat[0])
    alpha_ann_pct = float(alpha_val * 252.0 * 100.0)
    alpha_t = float(t_stats[0])
    alpha_p = float(p_values[0])

    betas_dict: Dict[str, float] = {}
    t_stats_dict: Dict[str, float] = {}
    p_values_dict: Dict[str, float] = {}

    for i, factor_name in enumerate(factors):
        betas_dict[factor_name] = round(float(beta_hat[i + 1]), 4)
        t_stats_dict[factor_name] = round(float(t_stats[i + 1]), 3)
        p_values_dict[factor_name] = round(float(p_values[i + 1]), 4)

    # Euler exact variance decomposition: Total Variance = Systematic Factor + Specific Residual
    var_y = float(np.var(y, ddof=1))
    ann_total_vol_pct = float(np.sqrt(var_y * 252.0) * 100.0) if var_y > 0 else 0.0

    factor_cov = np.cov(f_matrix, rowvar=False) if k_factors > 1 else np.array([[float(np.var(f_matrix, ddof=1))]])
    factor_betas_vec = beta_hat[1:]
    systemic_var = float(factor_betas_vec.T @ factor_cov @ factor_betas_vec) if var_y > 0 else 0.0
    systemic_var = max(0.0, min(systemic_var, var_y))
    specific_var = max(0.0, var_y - systemic_var)

    ann_idio_vol_pct = float(np.sqrt(specific_var * 252.0) * 100.0) if specific_var > 0 else 0.0

    systemic_risk_share_pct = round(float((systemic_var / var_y) * 100.0), 2) if var_y > 0 else 0.0
    specific_risk_share_pct = round(max(0.0, 100.0 - systemic_risk_share_pct), 2)

    # Individual factor percentage risk contribution: beta_k * Cov(F_k, Y) / Var(Y)
    factor_risk_pcts: Dict[str, float] = {}
    for i, factor_name in enumerate(factors):
        cov_fy = float(np.cov(f_matrix[:, i], y, ddof=1)[0, 1]) if var_y > 0 else 0.0
        pct_contrib = (beta_hat[i + 1] * cov_fy / var_y) * 100.0 if var_y > 0 else 0.0
        factor_risk_pcts[factor_name] = round(float(pct_contrib), 2)

    return {
        "sample_sessions": t_obs,
        "r_squared": round(float(r_squared), 4),
        "adj_r_squared": round(float(adj_r_squared), 4),
        "alpha": round(alpha_val, 6),
        "annualized_alpha_pct": round(alpha_ann_pct, 2),
        "alpha_t_stat": round(alpha_t, 3),
        "alpha_p_value": round(alpha_p, 4),
        "factor_betas": betas_dict,
        "factor_t_stats": t_stats_dict,
        "factor_p_values": p_values_dict,
        "factor_risk_contributions_pct": factor_risk_pcts,
        "systemic_risk_share_pct": systemic_risk_share_pct,
        "specific_risk_share_pct": specific_risk_share_pct,
        "total_volatility_ann_pct": round(ann_total_vol_pct, 2),
        "idiosyncratic_volatility_ann_pct": round(ann_idio_vol_pct, 2),
    }


def decompose_portfolio_factor_risk(
    weights: Dict[str, float],
    asset_betas: Dict[str, Dict[str, float]],
    factor_returns: pd.DataFrame,
    returns_matrix: pd.DataFrame,
) -> Dict[str, Any]:
    """Decompose multi-asset portfolio risk into Systematic Factor Risk and Specific Risk.

    Portfolio beta vector:
        beta_p = sum_i w_i * beta_i

    Systematic portfolio variance:
        sigma_p,sys^2 = beta_p^T * Sigma_F * beta_p

    Total portfolio variance:
        sigma_p^2 = w^T * Sigma_assets * w

    Specific portfolio variance:
        sigma_p,spec^2 = max(0, sigma_p^2 - sigma_p,sys^2)

    Args:
        weights: Dictionary mapping ticker to normalized portfolio weight.
        asset_betas: Dictionary mapping ticker to its factor betas dictionary.
        factor_returns: Clean DataFrame of factor returns.
        returns_matrix: Clean DataFrame of individual asset returns.

    Returns:
        Dictionary with portfolio betas, systematic and specific volatility, and risk shares.
    """
    tickers = [t for t in weights.keys() if t in returns_matrix.columns and t in asset_betas]
    if not tickers or factor_returns.empty:
        return {
            "portfolio_betas": {},
            "portfolio_ann_vol_pct": 0.0,
            "systemic_ann_vol_pct": 0.0,
            "specific_ann_vol_pct": 0.0,
            "systemic_risk_share_pct": 0.0,
            "specific_risk_share_pct": 100.0,
        }

    raw_w = np.array([weights[t] for t in tickers], dtype=float)
    sum_w = float(np.sum(raw_w))
    w = raw_w / sum_w if sum_w > 0 else np.ones(len(tickers)) / len(tickers)

    factors = list(factor_returns.columns)
    k_factors = len(factors)

    # Construct beta matrix B: (n_assets x k_factors)
    b_mat = np.zeros((len(tickers), k_factors))
    for i, t in enumerate(tickers):
        for j, f in enumerate(factors):
            b_mat[i, j] = asset_betas[t].get(f, 0.0)

    # Portfolio factor betas
    beta_p = b_mat.T @ w
    portfolio_betas_dict = {factors[j]: round(float(beta_p[j]), 4) for j in range(k_factors)}

    # Factor covariance
    f_cov = factor_returns.cov().values
    sys_var_daily = float(beta_p.T @ f_cov @ beta_p)

    # Asset covariance
    asset_cov = returns_matrix[tickers].cov().values
    total_var_daily = float(w.T @ asset_cov @ w)

    total_var_daily = max(1e-10, total_var_daily)
    sys_var_daily = max(0.0, min(sys_var_daily, total_var_daily))
    spec_var_daily = max(0.0, total_var_daily - sys_var_daily)

    sys_share_pct = round(float((sys_var_daily / total_var_daily) * 100.0), 2)
    spec_share_pct = round(max(0.0, 100.0 - sys_share_pct), 2)

    total_vol_ann = float(np.sqrt(total_var_daily * 252.0) * 100.0)
    sys_vol_ann = float(np.sqrt(sys_var_daily * 252.0) * 100.0)
    spec_vol_ann = float(np.sqrt(spec_var_daily * 252.0) * 100.0)

    return {
        "portfolio_betas": portfolio_betas_dict,
        "portfolio_ann_vol_pct": round(total_vol_ann, 2),
        "systemic_ann_vol_pct": round(sys_vol_ann, 2),
        "specific_ann_vol_pct": round(spec_vol_ann, 2),
        "systemic_risk_share_pct": sys_share_pct,
        "specific_risk_share_pct": spec_share_pct,
    }


def evaluate_factor_attribution(
    ticker: str,
    exchange: str = "XNYS",
    db_path: Optional[Path] = None,
    min_sessions: int = 60,
) -> FactorAttributionReceipt:
    """Run full factor risk attribution for a specified stock within its exchange universe.

    Args:
        ticker: Target stock symbol.
        exchange: Market identifier code (MIC).
        db_path: Optional custom path to SQLite database.
        min_sessions: Minimum aligned sessions required.

    Returns:
        FactorAttributionReceipt with complete style exposures and audit digest.
    """
    clean_ticker = validate_ticker(ticker)
    clean_exchange = validate_exchange(exchange)

    # Retrieve all tickers for the exchange
    stocks = get_all_stocks(db_path=db_path)
    exch_stocks = [s["ticker"] for s in stocks if s["exchange"] == clean_exchange]

    if clean_ticker not in exch_stocks:
        exch_stocks.append(clean_ticker)

    returns_matrix = build_aligned_returns_matrix(
        tickers=exch_stocks,
        exchange=clean_exchange,
        db_path=db_path,
        min_sessions=min_sessions,
    )

    if returns_matrix.empty or clean_ticker not in returns_matrix.columns:
        return FactorAttributionReceipt(
            ticker=clean_ticker,
            exchange=clean_exchange,
            sample_sessions=0,
            r_squared=0.0,
            adj_r_squared=0.0,
            annualized_alpha_pct=0.0,
            alpha_t_stat=0.0,
            alpha_p_value=1.0,
            factor_betas={},
            factor_t_stats={},
            factor_p_values={},
            factor_risk_contributions_pct={},
            systemic_risk_share_pct=0.0,
            specific_risk_share_pct=100.0,
            total_volatility_ann_pct=0.0,
            idiosyncratic_volatility_ann_pct=0.0,
            summary=f"Insufficient aligned session returns for {clean_ticker} on {clean_exchange}.",
        )

    factor_df = construct_style_factor_returns(returns_matrix)
    if factor_df.empty:
        return FactorAttributionReceipt(
            ticker=clean_ticker,
            exchange=clean_exchange,
            sample_sessions=len(returns_matrix),
            r_squared=0.0,
            adj_r_squared=0.0,
            annualized_alpha_pct=0.0,
            alpha_t_stat=0.0,
            alpha_p_value=1.0,
            factor_betas={},
            factor_t_stats={},
            factor_p_values={},
            factor_risk_contributions_pct={},
            systemic_risk_share_pct=0.0,
            specific_risk_share_pct=100.0,
            total_volatility_ann_pct=0.0,
            idiosyncratic_volatility_ann_pct=0.0,
            summary=f"Unable to construct style factors with {len(returns_matrix.columns)} tickers on {clean_exchange}.",
        )

    res = estimate_factor_betas(returns_matrix[clean_ticker], factor_df)

    mkt_beta = res["factor_betas"].get("MKT", 1.0)
    sys_share = res["systemic_risk_share_pct"]
    spec_share = res["specific_risk_share_pct"]
    r2_pct = res["r_squared"] * 100.0

    summary = (
        f"{clean_ticker} on {clean_exchange} exhibits {sys_share:.1f}% systematic factor risk "
        f"(R²={r2_pct:.1f}%, Market Beta={mkt_beta:.2f}) and {spec_share:.1f}% idiosyncratic specific risk "
        f"across {res['sample_sessions']} sessions."
    )

    return FactorAttributionReceipt(
        ticker=clean_ticker,
        exchange=clean_exchange,
        sample_sessions=res["sample_sessions"],
        r_squared=res["r_squared"],
        adj_r_squared=res["adj_r_squared"],
        annualized_alpha_pct=res["annualized_alpha_pct"],
        alpha_t_stat=res["alpha_t_stat"],
        alpha_p_value=res["alpha_p_value"],
        factor_betas=res["factor_betas"],
        factor_t_stats=res["factor_t_stats"],
        factor_p_values=res["factor_p_values"],
        factor_risk_contributions_pct=res["factor_risk_contributions_pct"],
        systemic_risk_share_pct=sys_share,
        specific_risk_share_pct=spec_share,
        total_volatility_ann_pct=res["total_volatility_ann_pct"],
        idiosyncratic_volatility_ann_pct=res["idiosyncratic_volatility_ann_pct"],
        summary=summary,
    )
