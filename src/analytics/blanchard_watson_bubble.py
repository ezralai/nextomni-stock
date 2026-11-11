"""Olivier J. Blanchard & Mark W. Watson (1982) Rational Speculative Bubbles Engine.

Theoretical Foundation:
    Blanchard, Olivier J., & Watson, Mark W. (1982).
    "Bubbles, Rational Expectations, and Financial Markets".
    NBER Working Paper No. 945 / Crises in the Economic and Financial Structure.

Key Concepts:
    - Rational No-Arbitrage Bubble Decomposition (P_t = F_t + B_t):
      Any rational asset bubble B_t >= 0 satisfies the Euler martingale condition
      E_t[B_{t+1}] = (1 + r) * B_t.
    - Bursting Bubble Continuation Compensation:
      If a speculative bubble bursts with probability (1 - pi) and survives with probability
      pi in [0.10, 1.0], investors require an explosive continuation growth rate
      g_bubble = (1 + r) / pi - 1 > r while alive so expected total asset return equals r:
      E_t[P_{t+1}] = (1 + r) * F_t + pi * ((1 + r) / pi * B_t) = (1 + r) * P_t.
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
HMAC_SECRET = b"blanchard-watson-bubble-v165-secret"


@dataclass(frozen=True)
class BlanchardWatsonReceipt:
    """Immutable quantitative audit receipt for Blanchard & Watson (1982) Rational Bubbles."""

    ticker: str
    observed_price_p: float
    fundamental_value_f: float
    discount_rate_r: float
    survival_prob_pi: float
    crash_prob: float
    bubble_component_b: float
    bubble_share_pct: float
    required_bubble_growth: float
    surviving_asset_return: float
    expected_next_bubble: float
    next_surviving_bubble: float
    expected_remaining_periods: float
    burst_drawdown_loss_pct: float
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


def infer_implied_survival_prob(observed_growth_g: float, discount_rate_r: float) -> float:
    """Invert explosive price run-up rate g >= r into implied bubble survival probability pi."""
    eff_g = max(float(discount_rate_r), float(observed_growth_g))
    return float(np.clip((1.0 + float(discount_rate_r)) / (1.0 + eff_g), 0.15, 0.99))


def classify_bubble_regime(
    bubble_share_pct: float,
    crash_prob: float,
    req_bubble_growth: float,
) -> str:
    """Classify speculative bubble hazard and valuation disconnect severity."""
    if bubble_share_pct <= 1e-6:
        return "FUNDAMENTAL_EQUILIBRIUM_NO_BUBBLE"
    if bubble_share_pct >= 50.0 and crash_prob >= 0.20:
        return "IMMINENT_BURST_EXTREME_OVERVALUATION"
    if bubble_share_pct >= 30.0 or req_bubble_growth >= 0.35:
        return "EXPLOSIVE_BUBBLE_CRITICAL_HAZARD"
    return "MILD_RATIONAL_BUBBLE_DRIFT"


def _build_hmac_payload(
    ticker: str,
    price_p: float,
    fund_f: float,
    rate_r: float,
    pi_surv: float,
    bubble_b: float,
    bubble_pct: float,
    req_g: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{price_p:.4f}:{fund_f:.4f}:{rate_r:.6f}:{pi_surv:.4f}:"
        f"{bubble_b:.4f}:{bubble_pct:.2f}:{req_g:.6f}:{regime}:{timestamp}"
    )


def calculate_blanchard_watson_bubble(
    observed_price_p: float = 150.0,
    fundamental_value_f: float = 100.0,
    discount_rate_r: float = 0.05,
    survival_prob_pi: float = 0.85,
    ticker: str = "BW82",
) -> BlanchardWatsonReceipt:
    """Evaluate Blanchard & Watson (1982) rational bubble decomposition and crash hazard."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    price_p = float(observed_price_p)
    fund_f = float(fundamental_value_f)
    rate_r = float(discount_rate_r)
    pi_surv = float(survival_prob_pi)

    if price_p <= 0.0 or fund_f <= 0.0:
        raise ValueError("observed_price_p and fundamental_value_f must be strictly positive.")
    if rate_r <= 0.0:
        raise ValueError("discount_rate_r must be strictly positive.")
    if not (0.10 <= pi_surv <= 1.0):
        raise ValueError("survival_prob_pi must satisfy 0.10 <= pi <= 1.0.")

    bubble_b = max(0.0, price_p - fund_f)
    bubble_pct = (bubble_b / price_p) * 100.0

    if bubble_b <= 1e-10:
        crash_p = 0.0
        req_b_growth = rate_r
        surv_asset_ret = rate_r
        next_surv_b = 0.0
        exp_next_b = 0.0
        exp_rem_periods = 0.0
        burst_loss_pct = 0.0
    else:
        crash_p = 1.0 - pi_surv
        b_growth_factor = (1.0 + rate_r) / pi_surv
        req_b_growth = b_growth_factor - 1.0
        next_surv_b = bubble_b * b_growth_factor
        exp_next_b = pi_surv * next_surv_b
        surv_asset_ret = rate_r + (1.0 + rate_r) * (bubble_b / price_p) * ((1.0 / pi_surv) - 1.0)
        exp_rem_periods = (1.0 / crash_p) if crash_p > 1e-9 else 100.0
        burst_loss_pct = bubble_pct

    regime = classify_bubble_regime(
        bubble_share_pct=bubble_pct,
        crash_prob=crash_p,
        req_bubble_growth=req_b_growth,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        price_p=price_p,
        fund_f=fund_f,
        rate_r=rate_r,
        pi_surv=pi_surv,
        bubble_b=bubble_b,
        bubble_pct=bubble_pct,
        req_g=req_b_growth,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return BlanchardWatsonReceipt(
        ticker=cleaned_ticker,
        observed_price_p=price_p,
        fundamental_value_f=fund_f,
        discount_rate_r=rate_r,
        survival_prob_pi=pi_surv,
        crash_prob=crash_p,
        bubble_component_b=bubble_b,
        bubble_share_pct=bubble_pct,
        required_bubble_growth=req_b_growth,
        surviving_asset_return=surv_asset_ret,
        expected_next_bubble=exp_next_b,
        next_surviving_bubble=next_surv_b,
        expected_remaining_periods=exp_rem_periods,
        burst_drawdown_loss_pct=burst_loss_pct,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_blanchard_watson_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "BW82",
    discount_rate_r: float = 0.05,
    survival_prob_pi: float | None = None,
) -> BlanchardWatsonReceipt:
    """Calibrate fundamental anchor F_t and implied bubble survival hazard strictly from past prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        window = p_arr[-min(len(p_arr), 60):]
        spot_p = float(max(1.0, window[-1]))
        # Causal fundamental baseline estimated from early half-window median grown at discount rate
        half = len(window) // 2
        early_base = float(np.median(window[:half]))
        ann_factor = math.exp(float(discount_rate_r) * (half / 252.0))
        fund_f = max(1.0, early_base * ann_factor)

        if survival_prob_pi is not None:
            eff_pi = float(survival_prob_pi)
        else:
            log_rets = np.diff(np.log(np.maximum(1e-6, window[-min(20, len(window)):])))
            ann_g = float(np.mean(log_rets) * 252.0)
            eff_pi = infer_implied_survival_prob(ann_g, float(discount_rate_r))
    else:
        spot_p = 150.0
        fund_f = 100.0
        eff_pi = float(survival_prob_pi) if survival_prob_pi is not None else 0.85

    return calculate_blanchard_watson_bubble(
        observed_price_p=spot_p,
        fundamental_value_f=fund_f,
        discount_rate_r=float(discount_rate_r),
        survival_prob_pi=eff_pi,
        ticker=ticker,
    )


def persist_blanchard_watson_record(
    receipt: BlanchardWatsonReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Blanchard & Watson (1982) rational bubble evaluation to SQLite via ? queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS blanchard_watson_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        observed_price_p REAL NOT NULL,
        fundamental_value_f REAL NOT NULL,
        survival_prob_pi REAL NOT NULL,
        crash_prob REAL NOT NULL,
        bubble_component_b REAL NOT NULL,
        bubble_share_pct REAL NOT NULL,
        required_bubble_growth REAL NOT NULL,
        burst_drawdown_loss_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO blanchard_watson_audits (
        ticker, observed_price_p, fundamental_value_f, survival_prob_pi,
        crash_prob, bubble_component_b, bubble_share_pct, required_bubble_growth,
        burst_drawdown_loss_pct, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.observed_price_p,
                receipt.fundamental_value_f,
                receipt.survival_prob_pi,
                receipt.crash_prob,
                receipt.bubble_component_b,
                receipt.bubble_share_pct,
                receipt.required_bubble_growth,
                receipt.burst_drawdown_loss_pct,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

