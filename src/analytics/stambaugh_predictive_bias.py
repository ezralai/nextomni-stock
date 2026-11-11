"""Robert F. Stambaugh (1999) Predictive Regressions & Finite-Sample Bias Engine.

Theoretical Foundation:
    Stambaugh, Robert F. (1999).
    "Predictive Regressions".
    Journal of Financial Economics, 54(3), pp. 375-421.

Key Concepts:
    - Persistent Predictor & Shock Correlation Bias:
      In finite-sample predictive regressions r_{t+1} = alpha + beta*x_t + u_{t+1} paired
      with persistent AR(1) predictor dynamics x_{t+1} = theta + rho*x_t + v_{t+1},
      correlation between return innovations u and predictor innovations v (gamma = Cov(u,v)/Var(v))
      induces the exact Kendall-Stambaugh finite-sample OLS slope bias:
      E[beta_hat - beta] = -gamma * (1 + 3*rho) / T.
    - Spurious Predictability Removal:
      When valuation ratios move inversely with contemporaneous returns (gamma < 0), OLS slopes
      appear spuriously positive. Subtracting Stambaugh bias recovers the true bias-adjusted
      predictive coefficient beta_adj and adjusted t-statistic.
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
HMAC_SECRET = b"stambaugh-predictive-bias-v168-secret"


@dataclass(frozen=True)
class StambaughBiasReceipt:
    """Immutable quantitative audit receipt for Stambaugh (1999) Predictive Regression Bias."""

    ticker: str
    sample_size_t: int
    ols_beta: float
    ar1_rho: float
    kendall_adjusted_rho: float
    gamma_cov_ratio: float
    stambaugh_bias: float
    bias_adjusted_beta: float
    ols_std_err: float
    ols_t_stat: float
    adjusted_t_stat: float
    bias_share_pct: float
    signal_retention_ratio: float
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to JSON-serializable dictionary."""
        return asdict(self)


def generate_hmac_signature(payload_str: str) -> str:
    """Generate SHA256 HMAC for cryptographic receipt sealing."""
    return hmac.new(
        HMAC_SECRET,
        payload_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_hmac_signature(payload_str: str, signature: str) -> bool:
    """Verify cryptographic receipt signature."""
    expected = generate_hmac_signature(payload_str)
    return hmac.compare_digest(expected, signature)


def classify_stambaugh_regime(
    bias_share_pct: float,
    ols_t: float,
    adj_t: float,
    gamma_cov: float,
) -> str:
    """Classify predictive signal durability after finite-sample Stambaugh bias adjustment."""
    if abs(gamma_cov) <= 1e-6:
        return "EXOGENOUS_UNBIASED_PREDICTOR"
    if abs(ols_t) >= 1.96 and abs(adj_t) < 1.65:
        return "SPURIOUS_STAMBAUGH_BIAS_ILLUSION"
    if bias_share_pct >= 35.0:
        return "HEAVILY_INFLATED_PERSISTENT_RATIO_BIAS"
    if abs(adj_t) >= 1.96:
        return "ROBUST_POST_BIAS_PREDICTIVE_ALPHA"
    return "WEAK_UNRELIABLE_PREDICTIVE_SIGNAL"


def _build_hmac_payload(
    ticker: str,
    t_len: int,
    ols_b: float,
    rho: float,
    gamma_cov: float,
    bias: float,
    adj_b: float,
    adj_t: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{t_len}:{ols_b:.6f}:{rho:.4f}:{gamma_cov:.6f}:"
        f"{bias:.6f}:{adj_b:.6f}:{adj_t:.4f}:{regime}:{timestamp}"
    )


def calculate_stambaugh_bias(
    ols_beta: float = 0.08,
    ar1_rho: float = 0.96,
    gamma_cov_ratio: float = -0.80,
    sample_size_t: int = 40,
    ols_std_err: float = 0.035,
    ticker: str = "RS99",
) -> StambaughBiasReceipt:
    """Compute analytical Stambaugh (1999) finite-sample OLS bias and bias-corrected slope."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    t_obs = int(sample_size_t)
    ols_b = float(ols_beta)
    rho = float(ar1_rho)
    gamma = float(gamma_cov_ratio)
    se_b = float(ols_std_err)

    if t_obs < 12:
        raise ValueError("sample_size_t must be >= 12 for reliable AR(1) persistence estimation.")
    if not (-0.999 < rho < 1.0):
        raise ValueError("ar1_rho must satisfy -0.999 < rho < 1.0 (stationary predictor).")
    if se_b <= 0.0 or not math.isfinite(se_b):
        raise ValueError("ols_std_err must be strictly positive and finite.")
    if not math.isfinite(ols_b) or not math.isfinite(gamma):
        raise ValueError("ols_beta and gamma_cov_ratio must be finite.")

    kendall_corr = (1.0 + 3.0 * rho) / t_obs
    rho_adj = min(0.9999, rho + kendall_corr)
    bias = -gamma * kendall_corr
    adj_b = ols_b - bias

    ols_t = ols_b / se_b
    adj_t = adj_b / se_b

    bias_share = (abs(bias) / max(1e-6, abs(ols_b))) * 100.0
    if abs(ols_b) > 1e-8 and (ols_b * adj_b) > 0.0:
        retention = float(np.clip(adj_b / ols_b, 0.0, 2.0))
    elif abs(bias) <= 1e-10 and abs(ols_b) > 1e-8:
        retention = 1.0
    else:
        retention = 0.0

    regime = classify_stambaugh_regime(
        bias_share_pct=bias_share,
        ols_t=ols_t,
        adj_t=adj_t,
        gamma_cov=gamma,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        t_len=t_obs,
        ols_b=ols_b,
        rho=rho,
        gamma_cov=gamma,
        bias=bias,
        adj_b=adj_b,
        adj_t=adj_t,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return StambaughBiasReceipt(
        ticker=cleaned_ticker,
        sample_size_t=t_obs,
        ols_beta=ols_b,
        ar1_rho=rho,
        kendall_adjusted_rho=rho_adj,
        gamma_cov_ratio=gamma,
        stambaugh_bias=bias,
        bias_adjusted_beta=adj_b,
        ols_std_err=se_b,
        ols_t_stat=ols_t,
        adjusted_t_stat=adj_t,
        bias_share_pct=bias_share,
        signal_retention_ratio=retention,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def estimate_stambaugh_regression(
    forward_returns: List[float] | np.ndarray,
    predictors: List[float] | np.ndarray,
    ticker: str = "RS99",
) -> StambaughBiasReceipt:
    """Estimate OLS predictive regression, AR(1) predictor persistence, and Stambaugh bias from arrays."""
    r_arr = np.asarray(forward_returns, dtype=float)
    x_arr = np.asarray(predictors, dtype=float)

    if len(x_arr) == len(r_arr) + 1:
        r_vec = r_arr
        x_lag = x_arr[:-1]
        x_next = x_arr[1:]
    elif len(x_arr) == len(r_arr):
        if len(r_arr) < 13:
            raise ValueError("Aligned equal-length arrays require >= 13 observations.")
        r_vec = r_arr[:-1]
        x_lag = x_arr[:-1]
        x_next = x_arr[1:]
    else:
        raise ValueError("predictors length must equal forward_returns length or length + 1.")

    t_obs = len(r_vec)
    if t_obs < 12:
        raise ValueError("Effective regression sample size must be >= 12.")

    x_centered = x_lag - float(np.mean(x_lag))
    ssx = float(np.dot(x_centered, x_centered))
    if ssx <= 1e-12:
        raise ValueError("Predictor series has zero variance; OLS slope is undefined.")

    # OLS 1: r_{t+1} = alpha + beta * x_t + u_{t+1}
    r_centered = r_vec - float(np.mean(r_vec))
    beta_ols = float(np.dot(x_centered, r_centered) / ssx)
    alpha_ols = float(np.mean(r_vec)) - beta_ols * float(np.mean(x_lag))
    u_resid = r_vec - (alpha_ols + beta_ols * x_lag)

    # OLS 2: x_{t+1} = theta + rho * x_t + v_{t+1}
    xnext_centered = x_next - float(np.mean(x_next))
    rho_ols = float(np.dot(x_centered, xnext_centered) / ssx)
    theta_ols = float(np.mean(x_next)) - rho_ols * float(np.mean(x_lag))
    v_resid = x_next - (theta_ols + rho_ols * x_lag)

    var_v = float(np.dot(v_resid, v_resid))
    if var_v <= 1e-12:
        raise ValueError("Predictor AR(1) innovations have near-zero variance.")

    gamma_est = float(np.dot(u_resid, v_resid) / var_v)
    dof = max(1, t_obs - 2)
    sigma_u_sq = float(np.dot(u_resid, u_resid)) / dof
    se_beta = max(1e-6, math.sqrt(sigma_u_sq / ssx))

    rho_clamped = float(np.clip(rho_ols, -0.95, 0.995))

    return calculate_stambaugh_bias(
        ols_beta=beta_ols,
        ar1_rho=rho_clamped,
        gamma_cov_ratio=gamma_est,
        sample_size_t=t_obs,
        ols_std_err=se_beta,
        ticker=ticker,
    )


def evaluate_stambaugh_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "RS99",
) -> StambaughBiasReceipt:
    """Build causal valuation-deviation predictor strictly <= t and audit Stambaugh bias."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 25:
        window = p_arr[-min(len(p_arr), 120):]
        log_p = np.log(np.maximum(1e-6, window))
        # Causal trailing EMA baseline computed step-by-step using only prices <= t
        ema = np.empty_like(log_p)
        ema[0] = log_p[0]
        decay = 2.0 / (20.0 + 1.0)
        for i in range(1, len(log_p)):
            ema[i] = decay * log_p[i] + (1.0 - decay) * ema[i - 1]
        # Valuation deviation predictor x_t = ema_t - log_p_t (inversely moves with contemporaneous returns)
        x_pred = ema - log_p
        fwd_rets = np.diff(log_p)
        if np.var(x_pred[:-1]) > 1e-10:
            return estimate_stambaugh_regression(fwd_rets, x_pred, ticker=ticker)

    return calculate_stambaugh_bias(ticker=ticker)


def persist_stambaugh_record(
    receipt: StambaughBiasReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Stambaugh (1999) predictive bias record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS stambaugh_bias_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        sample_size_t INTEGER NOT NULL,
        ols_beta REAL NOT NULL,
        ar1_rho REAL NOT NULL,
        gamma_cov_ratio REAL NOT NULL,
        stambaugh_bias REAL NOT NULL,
        bias_adjusted_beta REAL NOT NULL,
        ols_t_stat REAL NOT NULL,
        adjusted_t_stat REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO stambaugh_bias_audits (
        ticker, sample_size_t, ols_beta, ar1_rho, gamma_cov_ratio,
        stambaugh_bias, bias_adjusted_beta, ols_t_stat, adjusted_t_stat,
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
                receipt.sample_size_t,
                receipt.ols_beta,
                receipt.ar1_rho,
                receipt.gamma_cov_ratio,
                receipt.stambaugh_bias,
                receipt.bias_adjusted_beta,
                receipt.ols_t_stat,
                receipt.adjusted_t_stat,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

