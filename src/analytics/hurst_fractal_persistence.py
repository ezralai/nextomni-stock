"""Harold Edwin Hurst (1951) & Edgar E. Peters (1994) Rescaled Range (R/S) Fractal Engine.

Theoretical Foundation:
    Hurst, Harold Edwin (1951).
    "Long-Term Storage Capacity of Reservoirs".
    Transactions of the American Society of Civil Engineers, 116, pp. 770-799.

    Anis, A. A., and Lloyd, E. H. (1976).
    "The Expected Value of the Adjusted Rescaled Hurst Range of Independent
    Normal Summands". Biometrika, 63(1), pp. 111-116.

    Peters, Edgar E. (1994).
    "Fractal Market Analysis: Applying Chaos Theory to Investment and Economics".
    John Wiley & Sons.

Key Concepts:
    - Rescaled Range (R/S) Scaling Law:
      (R/S)_n ~ c * n^H across non-overlapping block sizes n, where H in (0, 1)
      is the Hurst exponent estimated via log-log OLS slope.
    - Anis-Lloyd (1976) Bias Correction:
      Subtracts finite-sample random-walk slope H_RW from classical H to yield
      unbiased H_adj = 0.50 + (H - H_RW).
    - Fractal Market Dimension & Joseph/Noah Persistence:
      Fractal dimension D = 2 - H_adj (D = 1.50 for Brownian motion) and
      implied fractional autocorrelation C = 2^(2 * H_adj - 1) - 1.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"hurst-fractal-persistence-v185-secret"


@dataclass(frozen=True)
class HurstFractalReceipt:
    """Immutable quantitative audit receipt for Hurst R/S fractal persistence."""

    ticker: str
    n_returns: int
    n_windows: int
    hurst_classical: float
    hurst_random_walk_bias: float
    hurst_corrected: float
    fractal_dimension: float
    implied_correlation: float
    scaling_r2: float
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
    n_returns: int,
    hurst_classical: float,
    hurst_corrected: float,
    fractal_dimension: float,
    scaling_r2: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_returns}|{hurst_classical:.6f}|{hurst_corrected:.6f}|"
        f"{fractal_dimension:.6f}|{scaling_r2:.6f}|{regime}|{timestamp}"
    )


def sign_hurst_receipt(payload: str) -> str:
    """Compute HMAC-SHA256 hexadecimal signature for Hurst fractal audit payload."""
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Constant-time verification of Hurst fractal audit HMAC signature."""
    expected = sign_hurst_receipt(payload)
    return hmac.compare_digest(expected, str(signature))


def _anis_lloyd_expected_rs(n: int) -> float:
    """Exact Anis & Lloyd (1976) expected R/S of n i.i.d. standard normal summands."""
    if n < 2:
        return 1.0
    r_vals = np.arange(1, n, dtype=float)
    sum_term = float(np.sum(np.sqrt((float(n) - r_vals) / r_vals)))
    prefactor = (float(n) - 0.5) / float(n)
    if n <= 340:
        gamma_ratio = math.exp(math.lgamma(0.5 * (n - 1)) - math.lgamma(0.5 * n)) / math.sqrt(math.pi)
    else:
        gamma_ratio = 1.0 / math.sqrt(0.5 * float(n) * math.pi)
    return float(prefactor * gamma_ratio * sum_term)


def build_synthetic_fractional_prices(
    hurst_target: float = 0.50,
    n_obs: int = 320,
    vol_daily: float = 0.012,
    seed: int = 185,
) -> np.ndarray:
    """Generate exact fractional Brownian price path via Cholesky of fGn autocovariance."""
    h_val = float(hurst_target)
    if not math.isfinite(h_val) or h_val <= 0.05 or h_val >= 0.95:
        raise ValueError(f"hurst_target must be finite within (0.05, 0.95), got {hurst_target}.")
    t_len = int(n_obs)
    if t_len < 32 or t_len > 1500:
        raise ValueError("n_obs must be between 32 and 1500.")

    two_h = 2.0 * h_val
    lags = np.arange(t_len, dtype=float)
    gamma = 0.5 * (
        np.abs(lags + 1.0) ** two_h - 2.0 * (np.abs(lags) ** two_h) + np.abs(lags - 1.0) ** two_h
    )
    idx = np.arange(t_len)
    cov = gamma[np.abs(idx[:, None] - idx[None, :])]
    # Tiny jitter ensures strict positive-definiteness near extreme H
    cov = cov + 1e-12 * np.eye(t_len)
    chol = np.linalg.cholesky(cov)

    rng = np.random.default_rng(int(seed))
    z = rng.normal(0.0, 1.0, size=t_len)
    fgn = chol @ z * float(vol_daily)
    prices = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(fgn))))
    return prices


def calculate_hurst_exponent(
    prices: Sequence[float] | np.ndarray,
    min_window: int = 8,
    ticker: str = "SPY",
) -> HurstFractalReceipt:
    """Compute classical R/S Hurst exponent, Anis-Lloyd bias correction, and fractal dimension."""
    clean_ticker = _validate_ticker(ticker)
    w_min = int(min_window)
    if w_min < 4 or w_min > 64:
        raise ValueError(f"min_window must be within [4, 64], got {min_window}.")

    p_arr = np.asarray(prices, dtype=float)
    if p_arr.ndim != 1 or not np.all(np.isfinite(p_arr)) or np.any(p_arr <= 0.0):
        raise ValueError("prices must be a 1D sequence of finite strictly positive floats.")
    if len(p_arr) < max(4 * w_min + 1, 33):
        raise ValueError(
            f"Need at least {max(4 * w_min + 1, 33)} prices for min_window={w_min}, got {len(p_arr)}."
        )

    rets = np.diff(np.log(p_arr))
    n_ret = len(rets)
    if float(np.std(rets, ddof=1)) <= 1e-12:
        raise ValueError("Degenerate zero-variance price series; cannot estimate Hurst exponent.")

    max_w = max(w_min + 2, n_ret // 4) if n_ret >= 64 else max(w_min + 1, n_ret // 2)
    raw_grid = np.unique(np.round(np.geomspace(w_min, max_w, num=10)).astype(int))
    windows = [int(w) for w in raw_grid if w_min <= int(w) <= max_w]

    log_n: List[float] = []
    log_rs: List[float] = []
    log_ers: List[float] = []

    for w in windows:
        k_blocks = n_ret // w
        if k_blocks < 2:
            continue
        rs_blocks: List[float] = []
        for b in range(k_blocks):
            seg = rets[b * w : (b + 1) * w]
            seg_std = float(np.std(seg, ddof=1))
            if seg_std <= 1e-12:
                continue
            dev = np.cumsum(seg - float(np.mean(seg)))
            rng_val = float(np.max(dev) - np.min(dev))
            rs_blocks.append(rng_val / seg_std)
        if rs_blocks:
            mean_rs = float(np.mean(rs_blocks))
            if mean_rs > 0.0:
                log_n.append(math.log(float(w)))
                log_rs.append(math.log(mean_rs))
                log_ers.append(math.log(_anis_lloyd_expected_rs(w)))

    if len(log_n) < 3:
        raise ValueError("Insufficient non-degenerate scaling windows to estimate Hurst exponent.")

    x = np.asarray(log_n, dtype=float)
    y = np.asarray(log_rs, dtype=float)
    y_rw = np.asarray(log_ers, dtype=float)

    slope_classical, intercept = np.polyfit(x, y, 1)
    slope_rw, _ = np.polyfit(x, y_rw, 1)

    y_hat = slope_classical * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-12))))

    h_classical = float(max(0.01, min(0.99, slope_classical)))
    h_rw = float(slope_rw)
    h_corr = float(max(0.01, min(0.99, 0.50 + (h_classical - h_rw))))
    fractal_dim = float(2.0 - h_corr)
    implied_corr = float((2.0 ** (2.0 * h_corr - 1.0)) - 1.0)

    if h_corr > 0.56:
        regime = "PERSISTENT_FRACTAL_TREND"
    elif h_corr < 0.44:
        regime = "ANTI_PERSISTENT_MEAN_REVERSION"
    else:
        regime = "BROWNIAN_RANDOM_WALK"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_returns=n_ret,
        hurst_classical=h_classical,
        hurst_corrected=h_corr,
        fractal_dimension=fractal_dim,
        scaling_r2=r2,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = sign_hurst_receipt(payload)

    return HurstFractalReceipt(
        ticker=clean_ticker,
        n_returns=n_ret,
        n_windows=len(log_n),
        hurst_classical=h_classical,
        hurst_random_walk_bias=h_rw,
        hurst_corrected=h_corr,
        fractal_dimension=fractal_dim,
        implied_correlation=implied_corr,
        scaling_r2=r2,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def persist_hurst_record(
    receipt: HurstFractalReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Hurst (1951) & Peters (1994) fractal persistence audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS hurst_fractal_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        n_returns INTEGER NOT NULL,
        n_windows INTEGER NOT NULL,
        hurst_classical REAL NOT NULL,
        hurst_corrected REAL NOT NULL,
        fractal_dimension REAL NOT NULL,
        implied_correlation REAL NOT NULL,
        scaling_r2 REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO hurst_fractal_audits (
        ticker, n_returns, n_windows, hurst_classical, hurst_corrected,
        fractal_dimension, implied_correlation, scaling_r2, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.n_returns,
                receipt.n_windows,
                receipt.hurst_classical,
                receipt.hurst_corrected,
                receipt.fractal_dimension,
                receipt.implied_correlation,
                receipt.scaling_r2,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True
