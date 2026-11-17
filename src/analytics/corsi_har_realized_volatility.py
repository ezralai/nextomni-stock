"""
Fulvio Corsi (2009) Heterogeneous Autoregressive Realized Volatility (HAR-RV) Cascade Engine.

References:
    Corsi, F. (2009).
    A Simple Approximate Long-Memory Model of Realized Volatility.
    Journal of Financial Econometrics, 7(2), 174–196.

    Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007).
    Roughing It Up: Including Jump Components in the Measurement, Modeling,
    and Forecasting of Return Volatility.
    The Review of Economics and Statistics, 89(4), 701–720.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import hmac
import math
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\_\-]{1,10}$")
_DEFAULT_HMAC_SECRET = b"nextomni-corsi-har-rv-2009-secret"


def _get_hmac_secret() -> bytes:
    env_secret = os.getenv("NEXTOMNI_HMAC_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")
    return _DEFAULT_HMAC_SECRET


def _compute_hmac(payload: str) -> str:
    return hmac.new(_get_hmac_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_receipt_payload(receipt: CorsiHARReceipt) -> str:
    """Canonical string representation for cryptographic HMAC signing."""
    return (
        f"{receipt.ticker}:{receipt.sample_sessions}:{receipt.beta_daily:.4f}:"
        f"{receipt.beta_weekly:.4f}:{receipt.beta_monthly:.4f}:{receipt.r_squared:.4f}:"
        f"{receipt.forecast_1d_ann_vol:.2f}:{receipt.regime}:{receipt.timestamp}"
    )


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature against system secret."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


@dataclasses.dataclass(frozen=True)
class CorsiHARReceipt:
    ticker: str
    sample_sessions: int
    beta_daily: float
    beta_weekly: float
    beta_monthly: float
    intercept: float
    r_squared: float
    persistence: float
    half_life_sessions: float
    forecast_1d_ann_vol: float
    forecast_5d_ann_vol: float
    forecast_22d_ann_vol: float
    historical_mean_ann_vol: float
    regime: str
    derivation_mode: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _empty_receipt(ticker: str, error_msg: str = "INVALID_INPUT") -> CorsiHARReceipt:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = f"{ticker}:0:0.0000:0.0000:0.0000:0.0000:0.00:ERROR_{error_msg}:{ts}"
    return CorsiHARReceipt(
        ticker=ticker,
        sample_sessions=0,
        beta_daily=0.35,
        beta_weekly=0.35,
        beta_monthly=0.25,
        intercept=0.0,
        r_squared=0.0,
        persistence=0.95,
        half_life_sessions=13.5,
        forecast_1d_ann_vol=0.0,
        forecast_5d_ann_vol=0.0,
        forecast_22d_ann_vol=0.0,
        historical_mean_ann_vol=0.0,
        regime=f"ERROR_{error_msg}",
        derivation_mode="FAILED",
        hmac_signature=_compute_hmac(raw),
        timestamp=ts,
    )


def calculate_corsi_har_rv(
    daily_rv_series: Sequence[float],
    ticker: str = "SPY",
    use_log: bool = True,
    ridge_lambda: float = 1e-4,
    derivation_mode: str = "PARAMETRIC_CALIBRATION",
) -> CorsiHARReceipt:
    """
    Fit Fulvio Corsi (2009) Heterogeneous Autoregressive Realized Volatility (HAR-RV) model.

    Args:
        daily_rv_series: Daily realized volatility values (standard deviation or annualized vol)
        ticker: Symbol string
        use_log: Whether to fit Log-HAR model log(RV) to guarantee positive volatility
        ridge_lambda: Tikhonov ridge regularization parameter to mitigate collinearity
        derivation_mode: Analytical calibration tag

    Returns:
        CorsiHARReceipt with cascade coefficients, persistence, forecasts, and HMAC signature.
    """
    clean_ticker = (ticker or "SPY").strip().upper()
    if not TICKER_REGEX.match(clean_ticker):
        clean_ticker = "SPY"

    rv_arr = np.asarray(daily_rv_series, dtype=np.float64)
    # Ensure all values are positive
    rv_arr = np.maximum(1e-5, rv_arr)
    n = len(rv_arr)

    # Minimum history required to construct at least 3 observation rows: 22 + 3 = 25
    if n < 25:
        return _empty_receipt(clean_ticker, "INSUFFICIENT_HISTORY")

    # Construct heterogeneous rolling cascade components:
    # Daily: RV_t^(d) = RV_t
    # Weekly: RV_t^(w) = mean(RV_{t-4..t})
    # Monthly: RV_t^(m) = mean(RV_{t-21..t})
    rv_daily = rv_arr
    rv_weekly = np.zeros(n, dtype=np.float64)
    rv_monthly = np.zeros(n, dtype=np.float64)

    for t in range(n):
        w_start = max(0, t - 4)
        m_start = max(0, t - 21)
        rv_weekly[t] = np.mean(rv_arr[w_start : t + 1])
        rv_monthly[t] = np.mean(rv_arr[m_start : t + 1])

    # Target: RV_{t+1} from t = 21 to n - 2
    # Regressors: [1, RV_t^(d), RV_t^(w), RV_t^(m)]
    start_idx = 21
    num_samples = n - 1 - start_idx
    if num_samples < 3:
        return _empty_receipt(clean_ticker, "INSUFFICIENT_SAMPLES")

    if use_log:
        x_d = np.log(rv_daily[start_idx : n - 1])
        x_w = np.log(rv_weekly[start_idx : n - 1])
        x_m = np.log(rv_monthly[start_idx : n - 1])
        y = np.log(rv_daily[start_idx + 1 : n])
    else:
        x_d = rv_daily[start_idx : n - 1]
        x_w = rv_weekly[start_idx : n - 1]
        x_m = rv_monthly[start_idx : n - 1]
        y = rv_daily[start_idx + 1 : n]

    ones = np.ones(num_samples, dtype=np.float64)
    X = np.column_stack([ones, x_d, x_w, x_m])

    # Regularized Ridge OLS
    reg = max(1e-7, float(ridge_lambda))
    xtx = X.T @ X + reg * np.eye(4)
    xty = X.T @ y
    beta = np.linalg.solve(xtx, xty)

    c_hat = float(beta[0])
    b_d = float(beta[1])
    b_w = float(beta[2])
    b_m = float(beta[3])

    # Invariance & stability bounds on cascade weights
    persistence = b_d + b_w + b_m
    y_pred = X @ beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r_squared = float(np.clip(1.0 - (ss_res / max(1e-12, ss_tot)), 0.0, 1.0))

    # Shock half-life: tau = ln(0.5) / ln(persistence)
    clamped_pers = float(np.clip(persistence, 0.05, 0.999))
    half_life = float(round(math.log(0.5) / math.log(clamped_pers), 1)) if clamped_pers < 1.0 else 100.0

    # Multi-horizon Out-of-Sample Forecasts starting from latest observation t = n - 1
    latest_rv = rv_daily[-1]
    latest_w = rv_weekly[-1]
    latest_m = rv_monthly[-1]

    if use_log:
        log_latest_d = math.log(latest_rv)
        log_latest_w = math.log(latest_w)
        log_latest_m = math.log(latest_m)

        # 1-day ahead
        log_f_1d = c_hat + b_d * log_latest_d + b_w * log_latest_w + b_m * log_latest_m
        f_1d = math.exp(float(np.clip(log_f_1d, -10.0, 5.0)))

        # 5-day ahead (unconditional pull)
        log_uncond = c_hat / max(0.01, 1.0 - persistence)
        f_5d = math.exp(float(np.clip(0.6 * log_f_1d + 0.4 * log_uncond, -10.0, 5.0)))

        # 22-day ahead
        f_22d = math.exp(float(np.clip(0.3 * log_f_1d + 0.7 * log_uncond, -10.0, 5.0)))
    else:
        f_1d = float(max(1e-4, c_hat + b_d * latest_rv + b_w * latest_w + b_m * latest_m))
        uncond = c_hat / max(0.01, 1.0 - persistence)
        f_5d = float(max(1e-4, 0.6 * f_1d + 0.4 * uncond))
        f_22d = float(max(1e-4, 0.3 * f_1d + 0.7 * uncond))

    # Convert to annualized percentage volatility (if input was daily std dev)
    # Check if input looks like daily (~0.01) or already annualized (~0.20)
    mean_val = float(np.mean(rv_arr))
    ann_factor = math.sqrt(252.0) * 100.0 if mean_val < 0.10 else 100.0 if mean_val < 1.0 else 1.0

    f_1d_ann = float(round(f_1d * ann_factor, 2))
    f_5d_ann = float(round(f_5d * ann_factor, 2))
    f_22d_ann = float(round(f_22d * ann_factor, 2))
    hist_mean_ann = float(round(mean_val * ann_factor, 2))

    # Horizon Dominance Regime
    if b_d > max(b_w, b_m) and b_d > 0.40:
        regime = "DAILY_NOISE_TRADER_DOMINATED"
    elif b_w > max(b_d, b_m) and b_w > 0.40:
        regime = "WEEKLY_SWING_TRADER_DOMINATED"
    elif b_m > max(b_d, b_w) and b_m > 0.40:
        regime = "MONTHLY_INSTITUTIONAL_DOMINATED"
    else:
        regime = "BALANCED_MULTI_HORIZON_CASCADE"

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r_bd = float(round(b_d, 4))
    r_bw = float(round(b_w, 4))
    r_bm = float(round(b_m, 4))
    r_c = float(round(c_hat, 4))
    r_r2 = float(round(r_squared, 4))
    r_pers = float(round(persistence, 4))

    raw_payload = (
        f"{clean_ticker}:{n}:{r_bd:.4f}:{r_bw:.4f}:{r_bm:.4f}:"
        f"{r_r2:.4f}:{f_1d_ann:.2f}:{regime}:{ts}"
    )
    sig_hmac = _compute_hmac(raw_payload)

    return CorsiHARReceipt(
        ticker=clean_ticker,
        sample_sessions=n,
        beta_daily=r_bd,
        beta_weekly=r_bw,
        beta_monthly=r_bm,
        intercept=r_c,
        r_squared=r_r2,
        persistence=r_pers,
        half_life_sessions=half_life,
        forecast_1d_ann_vol=f_1d_ann,
        forecast_5d_ann_vol=f_5d_ann,
        forecast_22d_ann_vol=f_22d_ann,
        historical_mean_ann_vol=hist_mean_ann,
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig_hmac,
        timestamp=ts,
    )


def evaluate_corsi_har_from_prices(
    prices: Sequence[float],
    ticker: str = "SPY",
    use_log: bool = True,
) -> CorsiHARReceipt:
    """
    Evaluate Corsi HAR-RV model from daily price series by extracting daily realized return volatility.
    """
    p_arr = np.asarray(prices, dtype=np.float64)
    if len(p_arr) < 26:
        return _empty_receipt(ticker, "INSUFFICIENT_PRICES")

    # Compute daily absolute/squared log returns as realized volatility proxy
    log_rets = np.diff(np.log(p_arr))
    # Daily RV proxy: absolute return with small floor
    daily_rv = np.maximum(1e-4, np.abs(log_rets))

    return calculate_corsi_har_rv(
        daily_rv_series=daily_rv,
        ticker=ticker,
        use_log=use_log,
        derivation_mode="EMPIRICAL_CLOSE_TO_CLOSE_RV",
    )


def persist_as_record(
    receipt: CorsiHARReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Corsi HAR-RV receipt into SQLite audit ledger with foreign key protection."""
    if receipt.derivation_mode == "FAILED":
        return False

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corsi_har_rv_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    sample_sessions INTEGER NOT NULL,
                    beta_daily REAL NOT NULL,
                    beta_weekly REAL NOT NULL,
                    beta_monthly REAL NOT NULL,
                    intercept REAL NOT NULL,
                    r_squared REAL NOT NULL,
                    persistence REAL NOT NULL,
                    half_life_sessions REAL NOT NULL,
                    forecast_1d_ann_vol REAL NOT NULL,
                    forecast_5d_ann_vol REAL NOT NULL,
                    forecast_22d_ann_vol REAL NOT NULL,
                    regime TEXT NOT NULL,
                    derivation_mode TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_corsi_har_ticker_created
                ON corsi_har_rv_records (ticker, created_at);
                """
            )
            conn.execute(
                """
                INSERT INTO corsi_har_rv_records (
                    ticker, sample_sessions, beta_daily, beta_weekly,
                    beta_monthly, intercept, r_squared, persistence,
                    half_life_sessions, forecast_1d_ann_vol, forecast_5d_ann_vol,
                    forecast_22d_ann_vol, regime, derivation_mode,
                    hmac_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    receipt.ticker,
                    receipt.sample_sessions,
                    receipt.beta_daily,
                    receipt.beta_weekly,
                    receipt.beta_monthly,
                    receipt.intercept,
                    receipt.r_squared,
                    receipt.persistence,
                    receipt.half_life_sessions,
                    receipt.forecast_1d_ann_vol,
                    receipt.forecast_5d_ann_vol,
                    receipt.forecast_22d_ann_vol,
                    receipt.regime,
                    receipt.derivation_mode,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
        return True
    except Exception:
        return False
    finally:
        conn.close()

