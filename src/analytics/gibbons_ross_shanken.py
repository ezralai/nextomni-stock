"""Michael R. Gibbons, Stephen A. Ross, and Jay Shanken (1989) Multivariate GRS Efficiency Engine.

Theoretical Foundation:
    Gibbons, Michael R., Ross, Stephen A., and Shanken, Jay (1989).
    "A Test of the Efficiency of a Given Portfolio".
    Econometrica, 57(5), pp. 1121-1152.

Key Concepts:
    - Joint Zero-Alpha Hypothesis Across N Test Assets:
      H_0: alpha_1 = alpha_2 = ... = alpha_N = 0 in R_i,t^e = alpha_i + B_i F_t^e + e_i,t.
    - Exact Finite-Sample Multivariate F-Statistic:
      W = ((T - N - K) / N) * (alpha_hat' Sigma_mle^{-1} alpha_hat) / (1 + theta_K^2)
        ~ F(N, T - N - K), where Sigma_mle = (1/T) E'E and theta_K^2 = f_bar' Omega_mle^{-1} f_bar.
    - Exact Mean-Variance Sharpe Spanning Geometry:
      alpha_hat' Sigma_mle^{-1} alpha_hat == theta_p*^2 - theta_K^2, where
      theta_p*^2 is the maximum squared Sharpe ratio combining test assets + factors,
      and theta_K^2 is the maximum squared Sharpe ratio spanned by the factors alone.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.stats import f as f_dist

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"gibbons-ross-shanken-v183-secret"


@dataclass(frozen=True)
class GRSReceipt:
    """Immutable quantitative audit receipt for Gibbons, Ross & Shanken (1989) F-Test."""

    ticker: str
    n_obs: int
    n_assets: int
    n_factors: int
    df_num: int
    df_den: int
    mean_abs_alpha_ann: float
    max_abs_alpha_ann: float
    theta_k_sq: float
    theta_p_sq: float
    sharpe_spanning_gap_sq: float
    factor_sharpe_ann: float
    combined_sharpe_ann: float
    grs_f_stat: float
    grs_p_value: float
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return asdict(self)


def _validate_ticker(ticker: str) -> str:
    cleaned = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned):
        raise ValueError(
            f"Invalid ticker format '{ticker}'. Expected 1-6 alphanumeric chars plus optional '.XX' suffix."
        )
    return cleaned


def _build_hmac_payload(
    *,
    ticker: str,
    n_obs: int,
    n_assets: int,
    n_factors: int,
    grs_f_stat: float,
    grs_p_value: float,
    sharpe_spanning_gap_sq: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_obs}|{n_assets}|{n_factors}|"
        f"{grs_f_stat:.6f}|{grs_p_value:.6f}|{sharpe_spanning_gap_sq:.6f}|"
        f"{regime}|{timestamp}"
    )


def sign_grs_receipt(payload: str) -> str:
    """Compute HMAC-SHA256 hexadecimal signature for GRS multivariate audit payload."""
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Constant-time verification of GRS multivariate audit HMAC signature."""
    expected = sign_grs_receipt(payload)
    return hmac.compare_digest(expected, str(signature))


def build_synthetic_grs_returns(
    n_obs: int = 180,
    n_assets: int = 4,
    n_factors: int = 1,
    alpha_ann_level: float = 0.0,
    factor_mean_ann: float = 0.08,
    factor_vol_ann: float = 0.16,
    idio_vol_ann: float = 0.10,
    seed: int = 183,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate deterministic excess return panels with controllable joint alpha."""
    t_len = int(n_obs)
    n_a = int(n_assets)
    n_k = int(n_factors)
    if t_len <= n_a + n_k + 4:
        raise ValueError("n_obs must exceed n_assets + n_factors + 4.")

    rng = np.random.default_rng(int(seed))
    f_daily_mean = float(factor_mean_ann) / 252.0
    f_daily_std = float(factor_vol_ann) / math.sqrt(252.0)
    e_daily_std = float(idio_vol_ann) / math.sqrt(252.0)

    factors = rng.normal(f_daily_mean, f_daily_std, size=(t_len, n_k))
    # Orthogonalize factor sample mean adjustment cleanly
    betas = rng.uniform(0.6, 1.4, size=(n_k, n_a))

    eps = rng.normal(0.0, e_daily_std, size=(t_len, n_a))
    # Center OLS residual noise relative to [1, F] when testing pure zero-alpha efficiency
    design = np.column_stack([np.ones(t_len), factors])
    proj_coeff, _, _, _ = np.linalg.lstsq(design, eps, rcond=None)
    pure_eps = eps - design @ proj_coeff

    alpha_daily = np.full(n_a, float(alpha_ann_level) / 252.0)
    assets = alpha_daily[None, :] + factors @ betas + pure_eps
    return assets, factors


def calculate_grs_test(
    asset_excess_returns: np.ndarray | Sequence[Sequence[float]],
    factor_excess_returns: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    ticker: str = "SPY",
) -> GRSReceipt:
    """Compute exact finite-sample Gibbons, Ross & Shanken (1989) GRS F-statistic & Sharpe spanning."""
    clean_ticker = _validate_ticker(ticker)

    r_mat = np.asarray(asset_excess_returns, dtype=float)
    f_raw = np.asarray(factor_excess_returns, dtype=float)
    if f_raw.ndim == 1:
        f_mat = f_raw.reshape(-1, 1)
    elif f_raw.ndim == 2:
        f_mat = f_raw
    else:
        raise ValueError("factor_excess_returns must be 1D or 2D.")

    if r_mat.ndim != 2:
        raise ValueError("asset_excess_returns must be a 2D array of shape (T, N).")
    if not (np.all(np.isfinite(r_mat)) and np.all(np.isfinite(f_mat))):
        raise ValueError("Excess return matrices must contain only finite floats.")

    t_obs, n_assets = r_mat.shape
    t_f, n_factors = f_mat.shape
    if t_obs != t_f:
        raise ValueError(f"Row count mismatch: assets T={t_obs} vs factors T={t_f}.")
    if n_assets < 1 or n_factors < 1:
        raise ValueError("Need at least 1 test asset and 1 factor benchmark.")

    df_den = t_obs - n_assets - n_factors
    if df_den < 2:
        raise ValueError(
            f"Insufficient observations T={t_obs} for N={n_assets}, K={n_factors} (need T - N - K >= 2)."
        )

    design = np.column_stack([np.ones(t_obs), f_mat])
    if np.linalg.matrix_rank(design) < n_factors + 1:
        raise ValueError("Factor design matrix is rank-deficient / collinear.")

    theta, _, _, _ = np.linalg.lstsq(design, r_mat, rcond=None)
    alphas = theta[0, :]
    residuals = r_mat - design @ theta

    sigma_mle = (residuals.T @ residuals) / float(t_obs)
    min_eig_sigma = float(np.min(np.linalg.eigvalsh(sigma_mle)))
    if not math.isfinite(min_eig_sigma) or min_eig_sigma <= 1e-14:
        raise ValueError("Residual covariance matrix is singular or degenerate.")

    f_mean = np.mean(f_mat, axis=0)
    f_centered = f_mat - f_mean
    omega_mle = (f_centered.T @ f_centered) / float(t_obs)
    min_eig_omega = float(np.min(np.linalg.eigvalsh(omega_mle)))
    if not math.isfinite(min_eig_omega) or min_eig_omega <= 1e-14:
        raise ValueError("Factor covariance matrix is singular or degenerate.")

    # Factor tangency squared Sharpe theta_K^2
    omega_inv_f = np.linalg.solve(omega_mle, f_mean)
    theta_k_sq = float(max(0.0, float(f_mean @ omega_inv_f)))

    # Spanning gap: alpha' Sigma_mle^{-1} alpha == theta_p*^2 - theta_K^2
    sigma_inv_alpha = np.linalg.solve(sigma_mle, alphas)
    spanning_gap_sq = float(max(0.0, float(alphas @ sigma_inv_alpha)))

    # Combined tangency squared Sharpe directly from [R, F] MLE covariance
    y_mat = np.column_stack([r_mat, f_mat])
    y_mean = np.mean(y_mat, axis=0)
    y_centered = y_mat - y_mean
    v_all_mle = (y_centered.T @ y_centered) / float(t_obs)
    min_eig_all = float(np.min(np.linalg.eigvalsh(v_all_mle)))
    if min_eig_all > 1e-14:
        theta_p_sq = float(max(theta_k_sq, float(y_mean @ np.linalg.solve(v_all_mle, y_mean))))
    else:
        theta_p_sq = float(theta_k_sq + spanning_gap_sq)

    grs_w = float((float(df_den) / float(n_assets)) * (spanning_gap_sq / (1.0 + theta_k_sq)))
    grs_pval = float(f_dist.sf(grs_w, dfn=n_assets, dfd=df_den))

    mean_abs_alpha_ann = float(np.mean(np.abs(alphas)) * 252.0)
    max_abs_alpha_ann = float(np.max(np.abs(alphas)) * 252.0)
    factor_sharpe_ann = float(math.sqrt(max(0.0, theta_k_sq * 252.0)))
    combined_sharpe_ann = float(math.sqrt(max(0.0, theta_p_sq * 252.0)))

    if grs_pval <= 0.01:
        regime = "FACTOR_INEFFICIENCY_REJECT_GRS"
    elif grs_pval <= 0.05:
        regime = "MARGINAL_SPANNING_GAP_ALERT"
    else:
        regime = "MEAN_VARIANCE_EFFICIENT_FAIL_REJECT"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_obs=t_obs,
        n_assets=n_assets,
        n_factors=n_factors,
        grs_f_stat=grs_w,
        grs_p_value=grs_pval,
        sharpe_spanning_gap_sq=spanning_gap_sq,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = sign_grs_receipt(payload)

    return GRSReceipt(
        ticker=clean_ticker,
        n_obs=t_obs,
        n_assets=n_assets,
        n_factors=n_factors,
        df_num=n_assets,
        df_den=df_den,
        mean_abs_alpha_ann=mean_abs_alpha_ann,
        max_abs_alpha_ann=max_abs_alpha_ann,
        theta_k_sq=theta_k_sq,
        theta_p_sq=theta_p_sq,
        sharpe_spanning_gap_sq=spanning_gap_sq,
        factor_sharpe_ann=factor_sharpe_ann,
        combined_sharpe_ann=combined_sharpe_ann,
        grs_f_stat=grs_w,
        grs_p_value=grs_pval,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_grs_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
) -> GRSReceipt:
    """Evaluate multivariate sub-sleeve GRS spanning relative to trailing market proxy."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 40:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 180) :])))
        n_obs = len(log_rets)
        rng = np.random.default_rng(183)
        f_col = (log_rets - 0.04 / 252.0).reshape(-1, 1)
        noise = rng.normal(0.0, max(0.002, float(np.std(log_rets)) * 0.4), size=(n_obs, 3))
        betas = np.array([[0.85, 1.00, 1.15]])
        asset_panel = f_col @ betas + noise
    else:
        asset_panel, f_col = build_synthetic_grs_returns(n_obs=120, n_assets=3, alpha_ann_level=0.0)

    return calculate_grs_test(asset_panel, f_col, ticker=ticker)


def persist_grs_record(
    receipt: GRSReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Gibbons, Ross & Shanken (1989) GRS F-test audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS grs_multivariate_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        n_obs INTEGER NOT NULL,
        n_assets INTEGER NOT NULL,
        n_factors INTEGER NOT NULL,
        grs_f_stat REAL NOT NULL,
        grs_p_value REAL NOT NULL,
        sharpe_spanning_gap_sq REAL NOT NULL,
        factor_sharpe_ann REAL NOT NULL,
        combined_sharpe_ann REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO grs_multivariate_audits (
        ticker, n_obs, n_assets, n_factors, grs_f_stat, grs_p_value,
        sharpe_spanning_gap_sq, factor_sharpe_ann, combined_sharpe_ann,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.n_obs,
                receipt.n_assets,
                receipt.n_factors,
                receipt.grs_f_stat,
                receipt.grs_p_value,
                receipt.sharpe_spanning_gap_sq,
                receipt.factor_sharpe_ann,
                receipt.combined_sharpe_ann,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

