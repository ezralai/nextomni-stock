"""John L. Kelly Jr. (1956) & Edward O. Thorp (1997) Fractional Kelly Log-Wealth Growth Engine.

Theoretical Foundation:
    Kelly, John L., Jr. (1956).
    "A New Interpretation of Information Rate".
    Bell System Technical Journal, 35(4), pp. 917-926.

    Thorp, Edward O. (1997 / 2006).
    "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market".
    10th International Conference on Gambling and Risk Taking / Handbook of Asset
    and Liability Management, Vol. 1, pp. 385-428.

Key Concepts:
    - Continuous-Time GBM Optimal Growth Fraction:
      f* = (mu - r_f) / sigma^2, maximizing asymptotic geometric compounding rate
      g(f) = r_f + f * (mu - r_f) - 0.5 * f^2 * sigma^2.
    - Half-Kelly Variance-Growth Tradeoff (k = f / f* = 0.5):
      Achieves 2*k - k^2 = 75.0% of peak excess log-wealth growth with only
      k^2 = 25.0% of Full-Kelly portfolio variance (a 75.0% variance reduction).
    - Double-Kelly Ruin Boundary (k >= 2.0):
      Excess growth rate drops to <= 0 at f >= 2*f*, causing volatility drag to
      destroy positive arithmetic expectancy and yield asymptotic stagnation or ruin.
    - Drawdown Exceedance Probability:
      For k in (0, 2) with mu > r_f, P(Drawdown >= d) = (1 - d)^(2/k - 1).
      Reducing from Full Kelly (k=1) to Half Kelly (k=0.5) drops the 50%
      drawdown probability from 50.0% to 12.5% (a 4x reduction).
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
HMAC_SECRET = b"kelly-growth-criterion-v181-secret"


@dataclass(frozen=True)
class KellyGrowthReceipt:
    """Immutable quantitative audit receipt for Fractional Kelly log-wealth sizing."""

    ticker: str
    expected_return_annual: float
    volatility_annual: float
    risk_free_annual: float
    sharpe_ratio: float
    kelly_multiplier: float
    raw_kelly_fraction: float
    full_kelly_fraction: float
    applied_fraction: float
    half_kelly_fraction: float
    double_kelly_ruin_fraction: float
    full_kelly_growth_ann: float
    applied_growth_ann: float
    applied_excess_growth_ann: float
    half_kelly_growth_ann: float
    growth_efficiency_ratio: float
    variance_reduction_ratio: float
    drawdown_threshold: float
    drawdown_exceedance_prob: float
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


def _validate_finite(name: str, value: float, low: float, high: float) -> float:
    val = float(value)
    if not math.isfinite(val) or val < low or val > high:
        raise ValueError(f"{name} must be finite within [{low}, {high}], got {value}.")
    return val


def _build_hmac_payload(
    *,
    ticker: str,
    expected_return_annual: float,
    volatility_annual: float,
    kelly_multiplier: float,
    full_kelly_fraction: float,
    applied_fraction: float,
    applied_growth_ann: float,
    drawdown_exceedance_prob: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{expected_return_annual:.6f}|{volatility_annual:.6f}|"
        f"{kelly_multiplier:.6f}|{full_kelly_fraction:.6f}|{applied_fraction:.6f}|"
        f"{applied_growth_ann:.6f}|{drawdown_exceedance_prob:.6f}|{regime}|{timestamp}"
    )


def sign_kelly_growth_receipt(payload: str) -> str:
    """Compute HMAC-SHA256 hexadecimal signature for Fractional Kelly audit payload."""
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Constant-time verification of Fractional Kelly audit HMAC signature."""
    expected = sign_kelly_growth_receipt(payload)
    return hmac.compare_digest(expected, str(signature))


def calculate_binomial_kelly(win_prob: float, payoff_odds: float) -> float:
    """Compute discrete Kelly (1956) bet fraction f* = (p*b - (1 - p)) / b."""
    p = _validate_finite("win_prob", win_prob, 1e-6, 1.0 - 1e-6)
    b = _validate_finite("payoff_odds", payoff_odds, 1e-6, 1000.0)
    return float((p * b - (1.0 - p)) / b)


def calculate_kelly_growth(
    expected_return_annual: float,
    volatility_annual: float,
    risk_free_annual: float = 0.04,
    kelly_fraction: float = 0.50,
    drawdown_threshold: float = 0.50,
    ticker: str = "SPY",
) -> KellyGrowthReceipt:
    """Compute continuous-time Kelly (1956) & Thorp (1997) optimal growth and drawdown metrics."""
    clean_ticker = _validate_ticker(ticker)
    mu = _validate_finite("expected_return_annual", expected_return_annual, -1.0, 3.0)
    sigma = _validate_finite("volatility_annual", volatility_annual, 1e-4, 3.0)
    rf = _validate_finite("risk_free_annual", risk_free_annual, -0.10, 0.50)
    k = _validate_finite("kelly_fraction", kelly_fraction, 0.0, 4.0)
    dd_level = _validate_finite("drawdown_threshold", drawdown_threshold, 0.05, 0.95)

    excess_mu = mu - rf
    sigma2 = sigma * sigma
    sharpe = excess_mu / sigma
    raw_f = excess_mu / sigma2

    if excess_mu <= 0.0:
        full_f = 0.0
        applied_f = 0.0
        half_f = 0.0
        double_f = 0.0
        g_full = rf
        g_applied_excess = 0.0
        g_applied = rf
        g_half = rf
        efficiency = 0.0
        var_reduction = 1.0
        dd_prob = 0.0
        regime = "NEGATIVE_EDGE_ZERO_ALLOCATION"
    else:
        full_f = raw_f
        applied_f = k * full_f
        half_f = 0.5 * full_f
        double_f = 2.0 * full_f

        g_full_excess = 0.5 * (sharpe ** 2)
        g_full = rf + g_full_excess

        g_applied_excess = applied_f * excess_mu - 0.5 * (applied_f ** 2) * sigma2
        g_applied = rf + g_applied_excess
        g_half = rf + 0.75 * g_full_excess

        efficiency = float(2.0 * k - (k ** 2))
        var_reduction = float(max(0.0, 1.0 - (k ** 2))) if k <= 1.0 else float(1.0 - (k ** 2))

        if k == 0.0:
            dd_prob = 0.0
            regime = "CONSERVATIVE_LOW_FRACTION"
        elif k >= 2.0:
            dd_prob = 1.0
            regime = "OVERBETTING_VOLATILITY_RUIN"
        else:
            dd_prob = float((1.0 - dd_level) ** ((2.0 / k) - 1.0))
            if k > 1.05:
                regime = "SUBOPTIMAL_OVER_KELLY_DRAG"
            elif k > 0.75:
                regime = "FULL_KELLY_MAX_GROWTH"
            elif k >= 0.25:
                regime = "HALF_KELLY_COMPOUNDING_SWEET_SPOT"
            else:
                regime = "CONSERVATIVE_LOW_FRACTION"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        expected_return_annual=mu,
        volatility_annual=sigma,
        kelly_multiplier=k,
        full_kelly_fraction=full_f,
        applied_fraction=applied_f,
        applied_growth_ann=g_applied,
        drawdown_exceedance_prob=dd_prob,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = sign_kelly_growth_receipt(payload)

    return KellyGrowthReceipt(
        ticker=clean_ticker,
        expected_return_annual=mu,
        volatility_annual=sigma,
        risk_free_annual=rf,
        sharpe_ratio=sharpe,
        kelly_multiplier=k,
        raw_kelly_fraction=raw_f,
        full_kelly_fraction=full_f,
        applied_fraction=applied_f,
        half_kelly_fraction=half_f,
        double_kelly_ruin_fraction=double_f,
        full_kelly_growth_ann=g_full,
        applied_growth_ann=g_applied,
        applied_excess_growth_ann=g_applied_excess,
        half_kelly_growth_ann=g_half,
        growth_efficiency_ratio=efficiency,
        variance_reduction_ratio=var_reduction,
        drawdown_threshold=dd_level,
        drawdown_exceedance_prob=dd_prob,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_kelly_growth_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SPY",
    risk_free_annual: float = 0.04,
    kelly_fraction: float = 0.50,
    drawdown_threshold: float = 0.50,
) -> KellyGrowthReceipt:
    """Calibrate expected return & volatility from trailing log returns via Ito relation."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 20:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 252) :])))
        daily_var = float(np.var(log_rets, ddof=1)) if len(log_rets) > 1 else 1e-4
        vol_ann = max(1e-4, math.sqrt(daily_var * 252.0))
        mu_ann = float(np.mean(log_rets) * 252.0 + 0.5 * daily_var * 252.0)
        mu_ann = max(-0.99, min(2.99, mu_ann))
        vol_ann = min(2.99, vol_ann)
    else:
        mu_ann = 0.12
        vol_ann = 0.20

    return calculate_kelly_growth(
        expected_return_annual=mu_ann,
        volatility_annual=vol_ann,
        risk_free_annual=float(risk_free_annual),
        kelly_fraction=float(kelly_fraction),
        drawdown_threshold=float(drawdown_threshold),
        ticker=ticker,
    )


def persist_kelly_growth_record(
    receipt: KellyGrowthReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Fractional Kelly growth & drawdown audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS kelly_growth_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        expected_return_annual REAL NOT NULL,
        volatility_annual REAL NOT NULL,
        kelly_multiplier REAL NOT NULL,
        full_kelly_fraction REAL NOT NULL,
        applied_fraction REAL NOT NULL,
        full_kelly_growth_ann REAL NOT NULL,
        applied_growth_ann REAL NOT NULL,
        drawdown_exceedance_prob REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO kelly_growth_audits (
        ticker, expected_return_annual, volatility_annual, kelly_multiplier,
        full_kelly_fraction, applied_fraction, full_kelly_growth_ann,
        applied_growth_ann, drawdown_exceedance_prob, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.expected_return_annual,
                receipt.volatility_annual,
                receipt.kelly_multiplier,
                receipt.full_kelly_fraction,
                receipt.applied_fraction,
                receipt.full_kelly_growth_ann,
                receipt.applied_growth_ann,
                receipt.drawdown_exceedance_prob,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

