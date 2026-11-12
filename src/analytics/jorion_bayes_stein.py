"""Philippe Jorion (1986) Bayes-Stein Shrinkage & Predictive Covariance Portfolio Engine.

Theoretical Foundation:
    Jorion, Philippe (1986).
    "Bayes-Stein Estimation for Portfolio Analysis".
    Journal of Financial and Quantitative Analysis, 21(3), pp. 279-292.

Key Concepts:
    - Empirical Bayes Shrinkage Toward Global Minimum Variance (GMV) Grand Mean:
      Given N assets over T historical observations with sample mean vector y_bar
      and sample covariance Sigma_hat, the GMV anchor is w_min = Sigma_hat^(-1) 1 / (1^T Sigma_hat^(-1) 1),
      mu_0 = w_min^T y_bar. The analytical James-Stein shrinkage intensity is:
      w_hat = (N + 2) / ((N + 2) + T * d^2),   d^2 = (y_bar - mu_0 1)^T Sigma_hat^(-1) (y_bar - mu_0 1).
      Posterior shrunk expected returns obey mu_BS = (1 - w_hat) * y_bar + w_hat * mu_0 * 1.
    - Bayes-Stein Predictive Covariance Matrix:
      Incorporates parameter estimation risk via precision hyperparameter lambda_hat = (N + 2)/d^2:
      Sigma_BS = Sigma_hat * (1 + 1/(T + lambda_hat)) + lambda_hat / (T*(T + 1 + lambda_hat)) * (1 1^T)/(1^T Sigma_hat^(-1) 1),
      stabilizing extreme Markowitz tangency portfolio weights out-of-sample.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"jorion-bayes-stein-v175-secret"


@dataclass(frozen=True)
class JorionBayesSteinReceipt:
    """Immutable quantitative audit receipt for Jorion (1986) Bayes-Stein Portfolio."""

    ticker: str
    n_assets: int
    sample_size_t: int
    risk_free_rate_r: float
    gmv_grand_mean_mu0: float
    mahalanobis_d2: float
    shrinkage_intensity_w: float
    lambda_hyperparam: float
    mean_dispersion_sample: float
    mean_dispersion_bs: float
    cov_inflation_ratio: float
    mle_gross_leverage: float
    bs_gross_leverage: float
    leverage_reduction_pct: float
    gmv_weights: List[float]
    sample_means: List[float]
    bayes_stein_means: List[float]
    mle_tangency_weights: List[float]
    bs_tangency_weights: List[float]
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return asdict(self)


def generate_hmac_signature(payload: str, secret: bytes = HMAC_SECRET) -> str:
    """Generate SHA-256 HMAC signature for cryptographic receipt integrity."""
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(
    payload: str,
    signature: str,
    secret: bytes = HMAC_SECRET,
) -> bool:
    """Constant-time verification of HMAC-SHA256 receipt signature."""
    expected = generate_hmac_signature(payload, secret=secret)
    return hmac.compare_digest(expected, signature)


def _compute_tangency_weights(
    mu_vec: np.ndarray,
    inv_sigma: np.ndarray,
    rf: float,
    gmv_fallback: np.ndarray,
) -> np.ndarray:
    """Compute normalized tangency portfolio weights with positive-slope GMV guard."""
    excess = mu_vec - rf
    raw = inv_sigma @ excess
    net = float(np.sum(raw))
    if abs(net) <= 1e-8 or net < 0.0:
        return gmv_fallback.copy()
    return raw / net


def classify_bayes_stein_regime(
    shrinkage_w: float,
    sample_size_t: int,
    n_assets: int,
) -> str:
    """Classify empirical-Bayes shrinkage and estimation uncertainty regime."""
    if sample_size_t < max(12, 3 * n_assets):
        return "SHORT_WINDOW_ESTIMATION_RISK"
    if shrinkage_w >= 0.70:
        return "HEAVY_PRIOR_GMV_DOMINANCE"
    if shrinkage_w <= 0.25:
        return "HIGH_SIGNAL_SAMPLE_TRUST"
    return "BALANCED_EMPIRICAL_BAYES"


def _build_hmac_payload(
    ticker: str,
    n_assets: int,
    sample_size_t: int,
    gmv_mu0: float,
    mahal_d2: float,
    shrinkage_w: float,
    bs_gross_leverage: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{n_assets}:{sample_size_t}:{gmv_mu0:.6f}:{mahal_d2:.6f}:"
        f"{shrinkage_w:.6f}:{bs_gross_leverage:.6f}:{regime}:{timestamp}"
    )


def build_synthetic_asset_moments(
    n_assets: int = 4,
    mean_dispersion: float = 0.06,
    base_vol: float = 0.20,
    correlation: float = 0.35,
    base_mean: float = 0.09,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct well-conditioned N-asset sample means and equicorrelated covariance."""
    n = int(n_assets)
    if not (2 <= n <= 50):
        raise ValueError("Number of assets N must be in [2, 50].")
    if not (0.0 < base_vol <= 2.0):
        raise ValueError("Base volatility must be in (0.0, 2.0].")
    if not (-0.15 <= correlation <= 0.95):
        raise ValueError("Equicorrelation must be in [-0.15, 0.95].")

    offsets = np.linspace(-0.5, 0.5, n) * float(mean_dispersion)
    means = float(base_mean) + offsets

    vols = float(base_vol) * np.linspace(0.85, 1.15, n)
    corr_mat = np.full((n, n), float(correlation), dtype=float)
    np.fill_diagonal(corr_mat, 1.0)
    min_eig = float(np.min(np.linalg.eigvalsh(corr_mat)))
    if min_eig <= 1e-6:
        corr_mat += (1e-4 - min_eig) * np.eye(n)
    cov = np.outer(vols, vols) * corr_mat
    return means, cov


def calculate_jorion_bayes_stein(
    sample_means: List[float] | np.ndarray,
    sample_cov: List[List[float]] | np.ndarray,
    sample_size_t: int,
    risk_free_rate_r: float = 0.04,
    ticker: str = "PORT",
) -> JorionBayesSteinReceipt:
    """Compute Philippe Jorion (1986) Bayes-Stein shrunk returns, covariance, and weights."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    mu_bar = np.asarray(sample_means, dtype=float)
    cov_raw = np.asarray(sample_cov, dtype=float)
    t_obs = int(sample_size_t)
    rf = float(risk_free_rate_r)

    n = len(mu_bar)
    if mu_bar.ndim != 1 or not (2 <= n <= 100):
        raise ValueError("Sample means must be a 1D vector with N in [2, 100] assets.")
    if cov_raw.shape != (n, n):
        raise ValueError(f"Covariance matrix shape must match ({n}, {n}).")
    if not (n + 3 <= t_obs <= 50000):
        raise ValueError(f"Sample size T must be in [{n + 3}, 50000] (> N + 2).")
    if not (-0.05 <= rf <= 0.40):
        raise ValueError("Risk-free rate r must be in [-0.05, 0.40].")

    cov = 0.5 * (cov_raw + cov_raw.T)
    min_eig = float(np.min(np.linalg.eigvalsh(cov)))
    if min_eig < -1e-6:
        raise ValueError("Sample covariance matrix must be positive semidefinite.")
    if min_eig < 1e-8:
        cov = cov + (1e-6 - min_eig) * np.eye(n)

    inv_cov = np.linalg.inv(cov)
    ones = np.ones(n, dtype=float)
    denom_ones = float(ones @ inv_cov @ ones)
    if denom_ones <= 1e-12:
        raise ValueError("Degenerate precision matrix sum.")

    gmv_w = (inv_cov @ ones) / denom_ones
    gmv_mu0 = float(gmv_w @ mu_bar)
    diff = mu_bar - gmv_mu0 * ones
    mahal_d2 = float(max(0.0, diff @ inv_cov @ diff))

    shrinkage_w = float((n + 2) / ((n + 2) + t_obs * mahal_d2))
    bs_means = (1.0 - shrinkage_w) * mu_bar + shrinkage_w * gmv_mu0 * ones

    if mahal_d2 > 1e-12:
        lam_hat = float((n + 2) / mahal_d2)
        c1 = 1.0 + 1.0 / (t_obs + lam_hat)
        c2 = lam_hat / (t_obs * (t_obs + 1.0 + lam_hat))
    else:
        lam_hat = 1e9
        c1 = 1.0 + 1.0 / max(1, t_obs)
        c2 = 1.0 / float(t_obs)

    bs_cov = c1 * cov + c2 * (np.outer(ones, ones) / denom_ones)
    inv_bs_cov = np.linalg.inv(bs_cov)

    w_mle = _compute_tangency_weights(mu_bar, inv_cov, rf, gmv_w)
    w_bs = _compute_tangency_weights(bs_means, inv_bs_cov, rf, gmv_w)

    mle_lev = float(np.sum(np.abs(w_mle)))
    bs_lev = float(np.sum(np.abs(w_bs)))
    lev_red = max(0.0, 1.0 - (bs_lev / max(1e-12, mle_lev)))

    disp_sample = float(np.std(mu_bar))
    disp_bs = float(np.std(bs_means))
    cov_ratio = float(np.trace(bs_cov) / max(1e-12, np.trace(cov)))

    regime = classify_bayes_stein_regime(shrinkage_w=shrinkage_w, sample_size_t=t_obs, n_assets=n)
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        n_assets=n,
        sample_size_t=t_obs,
        gmv_mu0=gmv_mu0,
        mahal_d2=mahal_d2,
        shrinkage_w=shrinkage_w,
        bs_gross_leverage=bs_lev,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return JorionBayesSteinReceipt(
        ticker=cleaned_ticker,
        n_assets=n,
        sample_size_t=t_obs,
        risk_free_rate_r=rf,
        gmv_grand_mean_mu0=gmv_mu0,
        mahalanobis_d2=mahal_d2,
        shrinkage_intensity_w=shrinkage_w,
        lambda_hyperparam=lam_hat,
        mean_dispersion_sample=disp_sample,
        mean_dispersion_bs=disp_bs,
        cov_inflation_ratio=cov_ratio,
        mle_gross_leverage=mle_lev,
        bs_gross_leverage=bs_lev,
        leverage_reduction_pct=lev_red,
        gmv_weights=[float(x) for x in gmv_w],
        sample_means=[float(x) for x in mu_bar],
        bayes_stein_means=[float(x) for x in bs_means],
        mle_tangency_weights=[float(x) for x in w_mle],
        bs_tangency_weights=[float(x) for x in w_bs],
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_bayes_stein_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SPY",
    n_assets: int = 4,
    sample_size_t: int = 60,
    mean_dispersion: float = 0.06,
) -> JorionBayesSteinReceipt:
    """Calibrate multi-sleeve asset covariance scale from trailing close prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 120) :])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.08, 0.80))
        ann_mean = float(np.clip(np.mean(log_rets) * 252.0, 0.04, 0.25))
    else:
        ann_vol = 0.20
        ann_mean = 0.09

    means, cov = build_synthetic_asset_moments(
        n_assets=int(n_assets),
        mean_dispersion=float(mean_dispersion),
        base_vol=ann_vol,
        correlation=0.35,
        base_mean=ann_mean,
    )
    return calculate_jorion_bayes_stein(
        sample_means=means,
        sample_cov=cov,
        sample_size_t=int(sample_size_t),
        risk_free_rate_r=0.04,
        ticker=ticker,
    )


def persist_jorion_bayes_stein_record(
    receipt: JorionBayesSteinReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Jorion (1986) Bayes-Stein audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS jorion_bayes_stein_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        n_assets INTEGER NOT NULL,
        sample_size_t INTEGER NOT NULL,
        gmv_grand_mean_mu0 REAL NOT NULL,
        mahalanobis_d2 REAL NOT NULL,
        shrinkage_intensity_w REAL NOT NULL,
        cov_inflation_ratio REAL NOT NULL,
        mle_gross_leverage REAL NOT NULL,
        bs_gross_leverage REAL NOT NULL,
        leverage_reduction_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO jorion_bayes_stein_audits (
        ticker, n_assets, sample_size_t, gmv_grand_mean_mu0, mahalanobis_d2,
        shrinkage_intensity_w, cov_inflation_ratio, mle_gross_leverage,
        bs_gross_leverage, leverage_reduction_pct, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.n_assets,
                receipt.sample_size_t,
                receipt.gmv_grand_mean_mu0,
                receipt.mahalanobis_d2,
                receipt.shrinkage_intensity_w,
                receipt.cov_inflation_ratio,
                receipt.mle_gross_leverage,
                receipt.bs_gross_leverage,
                receipt.leverage_reduction_pct,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

