"""Andrew W. Lo & A. Craig MacKinlay (1988) Heteroskedasticity-Robust Variance Ratio Engine.

Theoretical Foundation:
    Lo, Andrew W., and MacKinlay, A. Craig (1988).
    "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple
    Specification Test".
    The Review of Financial Studies, 1(1), pp. 41-66.

Key Concepts:
    - Overlapping Unbiased Variance Estimators:
      For 1-period log returns r_t = ln(P_t / P_{t-1}), t = 1..N,
      sigma_a^2 = sum((r_t - mu)^2) / (N - 1) and
      sigma_c^2(q) = sum((X_t - X_{t-q} - q * mu)^2) / m with finite-sample
      overlapping denominator m = q * (N - q + 1) * (1 - q / N).
    - Variance Ratio Specification:
      VR(q) = sigma_c^2(q) / sigma_a^2, where VR(q) = 1 under the Random Walk
      hypothesis, VR(q) > 1 under positive serial correlation (momentum),
      and VR(q) < 1 under negative serial correlation (mean reversion).
    - Homoskedastic z(q) & Heteroskedasticity-Consistent z*(q):
      z*(q) = (VR(q) - 1) / sqrt(phi*(q)) accounts for ARCH/GARCH volatility
      clustering via cross-products of lagged squared return innovations delta(j).
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
HMAC_SECRET = b"lo-mackinlay-variance-ratio-v182-secret"


@dataclass(frozen=True)
class VarianceRatioReceipt:
    """Immutable quantitative audit receipt for Lo & MacKinlay (1988) Variance Ratio."""

    ticker: str
    n_returns: int
    lag_q: int
    mean_return_period: float
    var_1period: float
    var_qperiod_scaled: float
    variance_ratio: float
    excess_variance_ratio: float
    phi_homoskedastic: float
    phi_heteroskedastic: float
    z_stat_homoskedastic: float
    z_stat_heteroskedastic: float
    p_value_homoskedastic: float
    p_value_heteroskedastic: float
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
    lag_q: int,
    variance_ratio: float,
    z_stat_homoskedastic: float,
    z_stat_heteroskedastic: float,
    p_value_heteroskedastic: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_returns}|{lag_q}|{variance_ratio:.6f}|"
        f"{z_stat_homoskedastic:.6f}|{z_stat_heteroskedastic:.6f}|"
        f"{p_value_heteroskedastic:.6f}|{regime}|{timestamp}"
    )


def sign_variance_ratio_receipt(payload: str) -> str:
    """Compute HMAC-SHA256 hexadecimal signature for Variance Ratio audit payload."""
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Constant-time verification of Variance Ratio audit HMAC signature."""
    expected = sign_variance_ratio_receipt(payload)
    return hmac.compare_digest(expected, str(signature))


def _normal_two_sided_pvalue(z_val: float) -> float:
    return float(math.erfc(abs(float(z_val)) / math.sqrt(2.0)))


def build_synthetic_vr_prices(
    rho: float = 0.0,
    n_obs: int = 260,
    vol_daily: float = 0.012,
    drift_daily: float = 0.0003,
    seed: int = 182,
    heteroskedastic: bool = False,
) -> np.ndarray:
    """Generate deterministic AR(1) or heteroskedastic price path for VR testing."""
    rng = np.random.default_rng(int(seed))
    eps = rng.normal(0.0, 1.0, size=int(n_obs))
    eps = eps - float(np.mean(eps))

    rets = np.zeros(int(n_obs), dtype=float)
    sigma_t = float(vol_daily)
    for t in range(int(n_obs)):
        if heteroskedastic:
            # Periodic high/low volatility regimes generating ARCH-like clustering
            vol_mult = 2.4 if (t // 15) % 2 == 1 else 0.45
            shock = float(eps[t]) * sigma_t * vol_mult
        else:
            shock = float(eps[t]) * sigma_t
        prev_dev = rets[t - 1] - drift_daily if t > 0 else 0.0
        rets[t] = drift_daily + float(rho) * prev_dev + shock

    prices = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(rets))))
    return prices


def calculate_lo_mackinlay_variance_ratio(
    prices: Sequence[float] | np.ndarray,
    lag_q: int = 5,
    ticker: str = "SPY",
) -> VarianceRatioReceipt:
    """Compute unbiased overlapping Lo & MacKinlay (1988) VR(q), z(q), and z*(q)."""
    clean_ticker = _validate_ticker(ticker)
    q = int(lag_q)
    if q < 2 or q > 126:
        raise ValueError(f"lag_q must be an integer within [2, 126], got {lag_q}.")

    p_arr = np.asarray(prices, dtype=float)
    if p_arr.ndim != 1 or not np.all(np.isfinite(p_arr)) or np.any(p_arr <= 0.0):
        raise ValueError("prices must be a 1D sequence of finite strictly positive floats.")

    min_prices = max(2 * q + 2, 12)
    if len(p_arr) < min_prices:
        raise ValueError(
            f"Insufficient price observations ({len(p_arr)}). Need at least {min_prices} for lag_q={q}."
        )

    log_p = np.log(p_arr)
    rets = np.diff(log_p)
    n_obs = len(rets)

    mu_hat = float((log_p[-1] - log_p[0]) / n_obs)
    centered = rets - mu_hat
    ss_total = float(np.sum(centered * centered))
    var_1p = ss_total / float(n_obs - 1)
    if not math.isfinite(var_1p) or var_1p <= 1e-12:
        raise ValueError("Degenerate zero-variance price series; cannot compute variance ratio.")

    q_rets = log_p[q:] - log_p[:-q]
    q_dev = q_rets - float(q) * mu_hat
    m_denom = float(q) * float(n_obs - q + 1) * (1.0 - float(q) / float(n_obs))
    var_qp = float(np.sum(q_dev * q_dev)) / m_denom

    vr_q = float(var_qp / var_1p)
    excess_vr = float(vr_q - 1.0)

    # Homoskedastic asymptotic variance phi(q) (Eq. 16)
    phi_hom = float(2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n_obs))
    phi_hom = max(phi_hom, 1e-12)

    # Heteroskedasticity-consistent asymptotic variance phi*(q) (Eqs. 17-19)
    sq_centered = centered * centered
    denom_sq = max(ss_total * ss_total, 1e-24)
    phi_het = 0.0
    for j in range(1, q):
        weight_j = (2.0 * float(q - j) / float(q)) ** 2
        cross_sq = float(np.sum(sq_centered[j:] * sq_centered[:-j]))
        delta_j_over_n = cross_sq / denom_sq
        phi_het += weight_j * delta_j_over_n
    phi_het = max(float(phi_het), 1e-12)

    z_hom = float(excess_vr / math.sqrt(phi_hom))
    z_het = float(excess_vr / math.sqrt(phi_het))
    p_hom = _normal_two_sided_pvalue(z_hom)
    p_het = _normal_two_sided_pvalue(z_het)

    if z_het >= 1.96 and vr_q > 1.0:
        regime = "MOMENTUM_PERSISTENCE_REJECT_RW"
    elif z_het <= -1.96 and vr_q < 1.0:
        regime = "MEAN_REVERSION_REJECT_RW"
    else:
        regime = "RANDOM_WALK_MARTINGALE_CONSISTENT"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_returns=n_obs,
        lag_q=q,
        variance_ratio=vr_q,
        z_stat_homoskedastic=z_hom,
        z_stat_heteroskedastic=z_het,
        p_value_heteroskedastic=p_het,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = sign_variance_ratio_receipt(payload)

    return VarianceRatioReceipt(
        ticker=clean_ticker,
        n_returns=n_obs,
        lag_q=q,
        mean_return_period=mu_hat,
        var_1period=var_1p,
        var_qperiod_scaled=var_qp,
        variance_ratio=vr_q,
        excess_variance_ratio=excess_vr,
        phi_homoskedastic=phi_hom,
        phi_heteroskedastic=phi_het,
        z_stat_homoskedastic=z_hom,
        z_stat_heteroskedastic=z_het,
        p_value_homoskedastic=p_hom,
        p_value_heteroskedastic=p_het,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_multi_horizon_variance_ratio(
    prices: Sequence[float] | np.ndarray,
    lags: Sequence[int] = (2, 5, 10, 21),
    ticker: str = "SPY",
) -> List[VarianceRatioReceipt]:
    """Evaluate Lo & MacKinlay Variance Ratio across multiple feasible lag horizons."""
    p_arr = np.asarray(prices, dtype=float)
    results: List[VarianceRatioReceipt] = []
    for q in lags:
        if len(p_arr) >= max(2 * int(q) + 2, 12):
            results.append(
                calculate_lo_mackinlay_variance_ratio(p_arr, lag_q=int(q), ticker=ticker)
            )
    if not results:
        raise ValueError("Price series too short for any requested Variance Ratio lag horizon.")
    return results


def persist_variance_ratio_record(
    receipt: VarianceRatioReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Lo & MacKinlay (1988) Variance Ratio audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS lo_mackinlay_vr_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        n_returns INTEGER NOT NULL,
        lag_q INTEGER NOT NULL,
        variance_ratio REAL NOT NULL,
        z_stat_homoskedastic REAL NOT NULL,
        z_stat_heteroskedastic REAL NOT NULL,
        p_value_heteroskedastic REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO lo_mackinlay_vr_audits (
        ticker, n_returns, lag_q, variance_ratio, z_stat_homoskedastic,
        z_stat_heteroskedastic, p_value_heteroskedastic, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.n_returns,
                receipt.lag_q,
                receipt.variance_ratio,
                receipt.z_stat_homoskedastic,
                receipt.z_stat_heteroskedastic,
                receipt.p_value_heteroskedastic,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

