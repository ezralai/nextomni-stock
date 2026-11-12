"""Franco Modigliani & Leah Modigliani (1997) M^2 Risk-Adjusted Performance Engine.

Theoretical Foundation:
    Modigliani, Franco, and Modigliani, Leah (1997).
    "Risk-Adjusted Performance".
    Journal of Portfolio Management, 23(2), pp. 45-54.

    Treynor, Jack L., and Black, Fischer (1973).
    "How to Use Security Analysis to Improve Portfolio Selection".
    Journal of Business, 46(1), pp. 66-86.

Key Concepts:
    - Volatility-Matched M^2 Equivalence:
      Scales portfolio p to match benchmark market volatility sigma_m via risk-matching
      leverage L* = sigma_m / sigma_p combined with (1 - L*) in risk-free bills:
      M^2 = L* * r_p + (1 - L*) * r_f == r_f + SR_p * sigma_m,
      expressing risk-adjusted outperformance directly in return space (M^2 - r_m).
    - Squared Sharpe & Appraisal Ratio Decomposition:
      Separating total variance sigma_p^2 = beta^2 * sigma_m^2 + sigma_e^2 yields
      Jensen's alpha = (r_p - r_f) - beta*(r_m - r_f), Appraisal Ratio AR = alpha / sigma_e,
      and combined active-benchmark maximum Sharpe SR_combined = sqrt(SR_m^2 + AR^2).
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
HMAC_SECRET = b"modigliani-m2-rap-v176-secret"


@dataclass(frozen=True)
class ModiglianiM2Receipt:
    """Immutable quantitative audit receipt for Modigliani-Modigliani M^2 RAP."""

    ticker: str
    portfolio_return_rp: float
    portfolio_vol_p: float
    benchmark_return_rm: float
    benchmark_vol_m: float
    risk_free_rate_r: float
    portfolio_beta: float
    portfolio_sharpe: float
    benchmark_sharpe: float
    leverage_ratio: float
    cash_or_margin_weight: float
    m2_return: float
    m2_alpha_spread: float
    m2_alpha_bps: float
    jensen_alpha: float
    idiosyncratic_vol: float
    appraisal_ratio: float
    treynor_ratio: float
    treynor_black_sharpe: float
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


def classify_modigliani_m2_regime(
    sharpe_p: float,
    m2_alpha_spread: float,
    leverage_ratio: float,
) -> str:
    """Classify leverage-normalized risk-adjusted performance regime."""
    if sharpe_p <= 0.0:
        return "NEGATIVE_SHARPE_UNDERPERFORMER"
    if m2_alpha_spread > 0.0 and leverage_ratio > 1.0:
        return "LEVERED_SHARPE_OUTPERFORMER"
    if m2_alpha_spread > 0.0:
        return "UNLEVERED_QUALITY_COMPOUNDER"
    return "HIGH_VOL_DILUTED_LAGGARD"


def _build_hmac_payload(
    ticker: str,
    rp: float,
    vol_p: float,
    rm: float,
    vol_m: float,
    rf: float,
    beta: float,
    m2_return: float,
    m2_alpha_bps: float,
    appraisal_ratio: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{rp:.6f}:{vol_p:.6f}:{rm:.6f}:{vol_m:.6f}:{rf:.6f}:{beta:.4f}:"
        f"{m2_return:.6f}:{m2_alpha_bps:.2f}:{appraisal_ratio:.6f}:{regime}:{timestamp}"
    )


def calculate_modigliani_m2(
    portfolio_return_rp: float,
    portfolio_vol_p: float,
    benchmark_return_rm: float = 0.10,
    benchmark_vol_m: float = 0.16,
    risk_free_rate_r: float = 0.04,
    portfolio_beta: float = 1.0,
    ticker: str = "SPY",
) -> ModiglianiM2Receipt:
    """Compute Modigliani & Modigliani (1997) M^2, leverage ratio, and Treynor-Black Sharpe."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    rp = float(portfolio_return_rp)
    vol_p = float(portfolio_vol_p)
    rm = float(benchmark_return_rm)
    vol_m = float(benchmark_vol_m)
    rf = float(risk_free_rate_r)
    beta = float(portfolio_beta)

    if not (-0.95 <= rp <= 5.0) or not (-0.95 <= rm <= 5.0):
        raise ValueError("Annualized returns must be in [-0.95, 5.0].")
    if not (0.0 < vol_p <= 5.0) or not (0.0 < vol_m <= 5.0):
        raise ValueError("Annualized volatilities must be in (0.0, 5.0].")
    if not (-0.05 <= rf <= 0.40):
        raise ValueError("Risk-free rate r must be in [-0.05, 0.40].")
    if not (-5.0 <= beta <= 5.0):
        raise ValueError("Portfolio beta must be in [-5.0, 5.0].")
    if abs(beta) * vol_m > vol_p + 1e-12:
        raise ValueError(
            "Absolute beta * benchmark_vol cannot exceed portfolio volatility (residual variance >= 0)."
        )

    sharpe_p = (rp - rf) / vol_p
    sharpe_m = (rm - rf) / vol_m

    lev_ratio = vol_m / vol_p
    cash_margin = 1.0 - lev_ratio
    m2_ret = lev_ratio * rp + cash_margin * rf
    m2_spread = m2_ret - rm
    m2_bps = m2_spread * 10000.0

    jensen_a = (rp - rf) - beta * (rm - rf)
    res_var = max(0.0, (vol_p * vol_p) - ((beta * vol_m) ** 2))
    idio_vol = math.sqrt(res_var)
    app_ratio = (jensen_a / idio_vol) if idio_vol > 1e-9 else 0.0
    treynor = ((rp - rf) / beta) if abs(beta) > 1e-6 else 0.0
    tb_sharpe = math.sqrt((sharpe_m * sharpe_m) + (app_ratio * app_ratio))

    regime = classify_modigliani_m2_regime(
        sharpe_p=sharpe_p,
        m2_alpha_spread=m2_spread,
        leverage_ratio=lev_ratio,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        rp=rp,
        vol_p=vol_p,
        rm=rm,
        vol_m=vol_m,
        rf=rf,
        beta=beta,
        m2_return=m2_ret,
        m2_alpha_bps=m2_bps,
        appraisal_ratio=app_ratio,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return ModiglianiM2Receipt(
        ticker=cleaned_ticker,
        portfolio_return_rp=rp,
        portfolio_vol_p=vol_p,
        benchmark_return_rm=rm,
        benchmark_vol_m=vol_m,
        risk_free_rate_r=rf,
        portfolio_beta=beta,
        portfolio_sharpe=sharpe_p,
        benchmark_sharpe=sharpe_m,
        leverage_ratio=lev_ratio,
        cash_or_margin_weight=cash_margin,
        m2_return=m2_ret,
        m2_alpha_spread=m2_spread,
        m2_alpha_bps=m2_bps,
        jensen_alpha=jensen_a,
        idiosyncratic_vol=idio_vol,
        appraisal_ratio=app_ratio,
        treynor_ratio=treynor,
        treynor_black_sharpe=tb_sharpe,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_modigliani_m2_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SPY",
    benchmark_return_rm: float = 0.10,
    benchmark_vol_m: float = 0.16,
    risk_free_rate_r: float = 0.04,
) -> ModiglianiM2Receipt:
    """Calibrate trailing annualized return, volatility, and bounded beta from close prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 120) :])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.06, 1.20))
        ann_ret = float(np.clip(np.mean(log_rets) * 252.0, -0.60, 1.50))
    else:
        ann_vol = 0.20
        ann_ret = 0.11

    vol_m = float(np.clip(benchmark_vol_m, 0.05, 1.00))
    max_feasible_beta = 0.85 * (ann_vol / vol_m)
    beta_est = float(min(1.05, max_feasible_beta))

    return calculate_modigliani_m2(
        portfolio_return_rp=ann_ret,
        portfolio_vol_p=ann_vol,
        benchmark_return_rm=float(benchmark_return_rm),
        benchmark_vol_m=vol_m,
        risk_free_rate_r=float(risk_free_rate_r),
        portfolio_beta=beta_est,
        ticker=ticker,
    )


def persist_modigliani_m2_record(
    receipt: ModiglianiM2Receipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Modigliani & Modigliani (1997) M^2 audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS modigliani_m2_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        portfolio_return_rp REAL NOT NULL,
        portfolio_vol_p REAL NOT NULL,
        benchmark_return_rm REAL NOT NULL,
        benchmark_vol_m REAL NOT NULL,
        leverage_ratio REAL NOT NULL,
        m2_return REAL NOT NULL,
        m2_alpha_bps REAL NOT NULL,
        appraisal_ratio REAL NOT NULL,
        treynor_black_sharpe REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO modigliani_m2_audits (
        ticker, portfolio_return_rp, portfolio_vol_p, benchmark_return_rm,
        benchmark_vol_m, leverage_ratio, m2_return, m2_alpha_bps,
        appraisal_ratio, treynor_black_sharpe, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.portfolio_return_rp,
                receipt.portfolio_vol_p,
                receipt.benchmark_return_rm,
                receipt.benchmark_vol_m,
                receipt.leverage_ratio,
                receipt.m2_return,
                receipt.m2_alpha_bps,
                receipt.appraisal_ratio,
                receipt.treynor_black_sharpe,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

