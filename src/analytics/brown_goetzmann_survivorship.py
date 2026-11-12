"""Brown, Goetzmann, Ibbotson & Ross (1992) Survivorship Bias & Truncated Attrition Engine.

Theoretical Foundation:
    Brown, Stephen J., Goetzmann, William N., Ibbotson, Roger G., and Ross, Stephen A. (1992).
    "Survivorship Bias in Performance Studies".
    Review of Financial Studies, 5(4), pp. 553-580.

Key Concepts:
    - Inverse Mills Ratio Mean Uplift Under Left-Truncation Attrition:
      For universe returns R ~ N(mu_true, sigma_true^2) subjected to liquidation cutoff c
      with standardized z-score alpha = (c - mu_true) / sigma_true and survival rate
      p_surv = 1 - N(alpha) = N(-alpha), surviving returns exhibit upward bias:
      E[R | R >= c] = mu_true + sigma_true * lambda(alpha),   lambda(alpha) = n(alpha) / p_surv.
    - Conditional Risk Attenuation & Double Sharpe Distortion:
      Truncation compresses cross-sectional variance via delta = lambda*(lambda - alpha) in (0, 1):
      Var(R | R >= c) = sigma_true^2 * (1 - delta) < sigma_true^2,
      simultaneously inflating apparent mean and deflating apparent risk to create
      spurious outperformance and artificial performance persistence among survivors.
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
HMAC_SECRET = b"brown-goetzmann-survivorship-v178-secret"


@dataclass(frozen=True)
class BrownGoetzmannReceipt:
    """Immutable quantitative audit receipt for BGIR (1992) Survivorship Bias."""

    ticker: str
    true_universe_mean: float
    true_universe_vol: float
    liquidation_cutoff_c: float
    horizon_years: float
    risk_free_rate_r: float
    truncation_z_alpha: float
    annual_survival_prob: float
    multiyear_survival_prob: float
    cumulative_attrition_rate: float
    inverse_mills_ratio: float
    survivorship_return_bias: float
    survivorship_bias_bps: float
    observed_survivor_mean: float
    variance_attenuation_factor: float
    observed_survivor_vol: float
    true_universe_sharpe: float
    apparent_survivor_sharpe: float
    sharpe_bias_delta: float
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


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function N(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function n(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def classify_survivorship_regime(survivorship_bias_bps: float) -> str:
    """Classify survivorship attrition and truncation distortion severity."""
    if survivorship_bias_bps >= 400.0:
        return "SEVERE_TRUNCATION_MIRAGE"
    if survivorship_bias_bps >= 150.0:
        return "MODERATE_DELISTING_BIAS"
    if survivorship_bias_bps >= 25.0:
        return "MILD_ATTRITION_DRIFT"
    return "UNTRUNCATED_UNIVERSE_BASELINE"


def _build_hmac_payload(
    ticker: str,
    mu_true: float,
    vol_true: float,
    cutoff_c: float,
    horizon: float,
    surv_prob: float,
    bias_bps: float,
    obs_vol: float,
    sharpe_delta: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{mu_true:.6f}:{vol_true:.6f}:{cutoff_c:.6f}:{horizon:.2f}:"
        f"{surv_prob:.6f}:{bias_bps:.2f}:{obs_vol:.6f}:{sharpe_delta:.6f}:"
        f"{regime}:{timestamp}"
    )


def calculate_survivorship_bias(
    true_universe_mean: float,
    true_universe_vol: float,
    liquidation_cutoff_c: float = -0.10,
    horizon_years: float = 5.0,
    risk_free_rate_r: float = 0.04,
    ticker: str = "HFRI",
) -> BrownGoetzmannReceipt:
    """Compute Brown, Goetzmann, Ibbotson & Ross (1992) truncated survival bias metrics."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    mu_true = float(true_universe_mean)
    sig_true = float(true_universe_vol)
    c_cut = float(liquidation_cutoff_c)
    horizon = float(horizon_years)
    rf = float(risk_free_rate_r)

    if not (-0.50 <= mu_true <= 1.00):
        raise ValueError("True universe mean return must be in [-0.50, 1.00].")
    if not (0.0 < sig_true <= 2.00):
        raise ValueError("True universe volatility must be in (0.0, 2.00].")
    if not (-2.00 <= c_cut <= 1.50):
        raise ValueError("Liquidation cutoff must be in [-2.00, 1.50].")
    if not (1.0 <= horizon <= 30.0):
        raise ValueError("Attrition horizon must be in [1.0, 30.0] years.")
    if not (-0.05 <= rf <= 0.40):
        raise ValueError("Risk-free rate r must be in [-0.05, 0.40].")

    alpha_z = (c_cut - mu_true) / sig_true
    if not (-6.0 <= alpha_z <= 4.0):
        raise ValueError(
            "Standardized truncation z-score (cutoff - mu)/sigma must lie in [-6.0, 4.0]."
        )

    surv_prob = _norm_cdf(-alpha_z)
    multi_surv = surv_prob ** horizon
    cum_attrition = 1.0 - multi_surv

    inv_mills = _norm_pdf(alpha_z) / surv_prob
    bias_ret = sig_true * inv_mills
    bias_bps = bias_ret * 10000.0
    obs_mean = mu_true + bias_ret

    delta_trunc = inv_mills * (inv_mills - alpha_z)
    var_atten = max(1e-8, min(1.0, 1.0 - delta_trunc))
    obs_vol = sig_true * math.sqrt(var_atten)

    true_sr = (mu_true - rf) / sig_true
    obs_sr = (obs_mean - rf) / obs_vol
    sr_delta = obs_sr - true_sr

    regime = classify_survivorship_regime(bias_bps)
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        mu_true=mu_true,
        vol_true=sig_true,
        cutoff_c=c_cut,
        horizon=horizon,
        surv_prob=surv_prob,
        bias_bps=bias_bps,
        obs_vol=obs_vol,
        sharpe_delta=sr_delta,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return BrownGoetzmannReceipt(
        ticker=cleaned_ticker,
        true_universe_mean=mu_true,
        true_universe_vol=sig_true,
        liquidation_cutoff_c=c_cut,
        horizon_years=horizon,
        risk_free_rate_r=rf,
        truncation_z_alpha=alpha_z,
        annual_survival_prob=surv_prob,
        multiyear_survival_prob=multi_surv,
        cumulative_attrition_rate=cum_attrition,
        inverse_mills_ratio=inv_mills,
        survivorship_return_bias=bias_ret,
        survivorship_bias_bps=bias_bps,
        observed_survivor_mean=obs_mean,
        variance_attenuation_factor=var_atten,
        observed_survivor_vol=obs_vol,
        true_universe_sharpe=true_sr,
        apparent_survivor_sharpe=obs_sr,
        sharpe_bias_delta=sr_delta,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_survivorship_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SPY",
    liquidation_cutoff_c: float = -0.10,
    horizon_years: float = 5.0,
) -> BrownGoetzmannReceipt:
    """Calibrate trailing return mean and volatility into BGIR survivorship truncation audit."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 120) :])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.08, 1.00))
        ann_mean = float(np.clip(np.mean(log_rets) * 252.0, -0.25, 0.50))
    else:
        ann_vol = 0.18
        ann_mean = 0.08

    cutoff = float(np.clip(liquidation_cutoff_c, ann_mean - 5.5 * ann_vol, ann_mean + 3.5 * ann_vol))

    return calculate_survivorship_bias(
        true_universe_mean=ann_mean,
        true_universe_vol=ann_vol,
        liquidation_cutoff_c=cutoff,
        horizon_years=float(horizon_years),
        risk_free_rate_r=0.04,
        ticker=ticker,
    )


def persist_brown_goetzmann_record(
    receipt: BrownGoetzmannReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist BGIR (1992) survivorship bias audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS brown_goetzmann_survivorship_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        true_universe_mean REAL NOT NULL,
        true_universe_vol REAL NOT NULL,
        liquidation_cutoff_c REAL NOT NULL,
        annual_survival_prob REAL NOT NULL,
        survivorship_bias_bps REAL NOT NULL,
        observed_survivor_mean REAL NOT NULL,
        observed_survivor_vol REAL NOT NULL,
        sharpe_bias_delta REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO brown_goetzmann_survivorship_audits (
        ticker, true_universe_mean, true_universe_vol, liquidation_cutoff_c,
        annual_survival_prob, survivorship_bias_bps, observed_survivor_mean,
        observed_survivor_vol, sharpe_bias_delta, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.true_universe_mean,
                receipt.true_universe_vol,
                receipt.liquidation_cutoff_c,
                receipt.annual_survival_prob,
                receipt.survivorship_bias_bps,
                receipt.observed_survivor_mean,
                receipt.observed_survivor_vol,
                receipt.sharpe_bias_delta,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

