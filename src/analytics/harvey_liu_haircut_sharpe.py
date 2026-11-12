"""Harvey & Liu (2015) / HLZ (2016) Multiple Testing & Haircut Sharpe Ratio Engine.

Implements:
- Multiple-testing p-value adjustments across M backtested strategy trials:
    1. Bonferroni (1936) Family-Wise Error Rate: p_bonf = min(1, M * p_raw)
    2. Holm (1979) step-down FWER: step-down monotonic adjustment
    3. Benjamini-Hochberg (1995) False Discovery Rate (BH): (M / r) * p_raw
    4. Benjamini-Hochberg-Yekutieli (2001) dependent FDR (BHY): (M * c(M) / r) * p_raw
- Analytical inversion of adjusted p-values back to Haircut Sharpe Ratios:
      SR_adj = sign(SR_obs) * Phi^{-1}(1 - p_adj / 2) / sqrt(T / ann_factor)
- Percentage Sharpe haircut & Harvey-Liu-Zhu (2016) significance regime classification
- Strict ticker validation, HMAC-SHA256 audit sealing, and parameterized SQLite logging
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import norm

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-hlz-2016-audit-key"


@dataclass(frozen=True)
class HaircutSharpeReceipt:
    """Immutable receipt for Harvey & Liu (2015) multiple-testing Haircut Sharpe."""

    ticker: str
    n_obs: int
    n_trials: int
    rank: int
    sharpe_observed: float
    t_stat_raw: float
    p_value_raw: float
    p_value_bonferroni: float
    p_value_holm: float
    p_value_bh: float
    p_value_bhy: float
    sharpe_bonferroni: float
    sharpe_holm: float
    sharpe_bh: float
    sharpe_bhy: float
    haircut_pct_bhy: float
    regime: str
    hmac_signature: str
    timestamp: str


def _validate_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not clean or not TICKER_REGEX.match(clean):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return clean


def _build_hmac_payload(
    *,
    ticker: str,
    n_obs: int,
    n_trials: int,
    sharpe_observed: float,
    sharpe_bh: float,
    sharpe_bhy: float,
    haircut_pct_bhy: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_obs}|{n_trials}|{sharpe_observed:.8f}|"
        f"{sharpe_bh:.8f}|{sharpe_bhy:.8f}|{haircut_pct_bhy:.6f}|"
        f"{regime}|{timestamp}"
    )


def sign_haircut_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_haircut_receipt(payload)
    return hmac.compare_digest(expected, signature)


def _invert_p_to_sharpe(p_adj: float, years: float, sign_sr: float) -> float:
    if p_adj >= 1.0 - 1e-12 or sign_sr == 0.0:
        return 0.0
    p_clamped = float(np.clip(p_adj, 1e-15, 1.0 - 1e-12))
    z_adj = float(norm.isf(0.5 * p_clamped))
    return float(sign_sr * z_adj / math.sqrt(years))


def calculate_haircut_sharpe(
    sharpe_observed: float,
    n_obs: int = 1260,
    n_trials: int = 100,
    rank: int = 1,
    ann_factor: float = 252.0,
    ticker: str = "SPY",
) -> HaircutSharpeReceipt:
    """Compute Harvey & Liu (2015) multiple-testing adjusted Haircut Sharpe Ratios."""
    clean_ticker = _validate_ticker(ticker)
    sr_obs = float(sharpe_observed)
    if not math.isfinite(sr_obs):
        raise ValueError("sharpe_observed must be finite")
    if n_obs < 20:
        raise ValueError("n_obs must be >= 20")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if rank < 1 or rank > n_trials:
        raise ValueError("rank must satisfy 1 <= rank <= n_trials")
    if ann_factor <= 0.0 or not math.isfinite(ann_factor):
        raise ValueError("ann_factor must be finite and > 0")

    years = float(n_obs) / float(ann_factor)
    t_raw = float(sr_obs * math.sqrt(years))
    p_raw = float(np.clip(2.0 * norm.sf(abs(t_raw)), 1e-15, 1.0))

    # Harmonic dependence factor c(M) = sum_{j=1}^M 1/j
    c_m = float(np.sum(1.0 / np.arange(1, n_trials + 1, dtype=float)))

    p_bonf = float(min(1.0, n_trials * p_raw))
    p_holm = float(min(1.0, (n_trials - rank + 1) * p_raw))
    p_bh = float(min(1.0, (n_trials / float(rank)) * p_raw))
    p_bhy = float(min(1.0, (n_trials * c_m / float(rank)) * p_raw))

    sign_sr = 1.0 if sr_obs > 0.0 else (-1.0 if sr_obs < 0.0 else 0.0)
    sr_bonf = _invert_p_to_sharpe(p_bonf, years, sign_sr)
    sr_holm = _invert_p_to_sharpe(p_holm, years, sign_sr)
    sr_bh = _invert_p_to_sharpe(p_bh, years, sign_sr)
    sr_bhy = _invert_p_to_sharpe(p_bhy, years, sign_sr)

    if abs(sr_obs) > 1e-12:
        haircut_pct = float(np.clip((1.0 - abs(sr_bhy) / abs(sr_obs)) * 100.0, 0.0, 100.0))
    else:
        haircut_pct = 0.0

    if sr_obs <= 0.0 or p_bhy >= 0.10:
        regime = "INSIGNIFICANT_NOISE"
    elif p_bhy < 0.01 and haircut_pct < 45.0:
        regime = "ROBUST_MULTIPLE_TEST_ALPHA"
    elif p_bhy < 0.05:
        regime = "MARGINAL_BHY_SURVIVOR"
    else:
        regime = "DATA_MINED_HAIRCUT"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_obs=int(n_obs),
        n_trials=int(n_trials),
        sharpe_observed=sr_obs,
        sharpe_bh=sr_bh,
        sharpe_bhy=sr_bhy,
        haircut_pct_bhy=haircut_pct,
        regime=regime,
        timestamp=ts,
    )
    return HaircutSharpeReceipt(
        ticker=clean_ticker,
        n_obs=int(n_obs),
        n_trials=int(n_trials),
        rank=int(rank),
        sharpe_observed=sr_obs,
        t_stat_raw=t_raw,
        p_value_raw=p_raw,
        p_value_bonferroni=p_bonf,
        p_value_holm=p_holm,
        p_value_bh=p_bh,
        p_value_bhy=p_bhy,
        sharpe_bonferroni=sr_bonf,
        sharpe_holm=sr_holm,
        sharpe_bh=sr_bh,
        sharpe_bhy=sr_bhy,
        haircut_pct_bhy=haircut_pct,
        regime=regime,
        hmac_signature=sign_haircut_receipt(payload),
        timestamp=ts,
    )


def evaluate_haircut_sharpe_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    n_trials: int = 50,
) -> HaircutSharpeReceipt:
    """Compute empirical annualized Sharpe from price series and apply HLZ haircut."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price sessions for Haircut Sharpe Ratio")

    log_ret = np.diff(np.log(arr))
    std_r = float(np.std(log_ret, ddof=1))
    sr_obs = float((np.mean(log_ret) / std_r) * math.sqrt(252.0)) if std_r > 1e-12 else 0.0
    return calculate_haircut_sharpe(
        sharpe_observed=sr_obs,
        n_obs=len(log_ret),
        n_trials=n_trials,
        rank=1,
        ann_factor=252.0,
        ticker=ticker,
    )


def persist_haircut_sharpe_record(
    receipt: HaircutSharpeReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Harvey-Liu Haircut Sharpe Ratio receipt to SQLite."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harvey_liu_haircut_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                n_obs INTEGER NOT NULL,
                n_trials INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                sharpe_observed REAL NOT NULL,
                t_stat_raw REAL NOT NULL,
                p_value_raw REAL NOT NULL,
                p_value_bonferroni REAL NOT NULL,
                p_value_holm REAL NOT NULL,
                p_value_bh REAL NOT NULL,
                p_value_bhy REAL NOT NULL,
                sharpe_bonferroni REAL NOT NULL,
                sharpe_holm REAL NOT NULL,
                sharpe_bh REAL NOT NULL,
                sharpe_bhy REAL NOT NULL,
                haircut_pct_bhy REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO harvey_liu_haircut_records (
                ticker, n_obs, n_trials, rank, sharpe_observed, t_stat_raw,
                p_value_raw, p_value_bonferroni, p_value_holm, p_value_bh, p_value_bhy,
                sharpe_bonferroni, sharpe_holm, sharpe_bh, sharpe_bhy,
                haircut_pct_bhy, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.n_obs,
                receipt.n_trials,
                receipt.rank,
                receipt.sharpe_observed,
                receipt.t_stat_raw,
                receipt.p_value_raw,
                receipt.p_value_bonferroni,
                receipt.p_value_holm,
                receipt.p_value_bh,
                receipt.p_value_bhy,
                receipt.sharpe_bonferroni,
                receipt.sharpe_holm,
                receipt.sharpe_bh,
                receipt.sharpe_bhy,
                receipt.haircut_pct_bhy,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

