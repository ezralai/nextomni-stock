"""Fischer Black (1976) Forward, Commodity & Futures Option Pricing Engine.

Theoretical Foundation:
    Black, Fischer (1976).
    "The Pricing of Commodity Contracts".
    Journal of Financial Economics, 3(1-2), pp. 167-179.

Key Concepts:
    - Driftless Futures Numeraire Pricing:
      Since futures contracts have zero initial entry cost, risk-neutral futures price F
      follows a driftless martingale (mu_F = 0). Discounting expected payoff at r gives:
      C = exp(-r*T) * [F*N(d1) - K*N(d2)],
      P = exp(-r*T) * [K*N(-d2) - F*N(-d1)],
      where d1, d2 = (ln(F/K) +/- 0.5*sigma^2*T) / (sigma*sqrt(T)).
    - Futures Put-Call Parity & Non-Positive Rho:
      C - P == exp(-r*T) * (F - K) identically, while holding forward F fixed produces
      strictly non-positive interest-rate rho = -T * Option_Value for BOTH Calls and Puts,
      alongside spot-futures basis convenience yield inference from S_0.
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
ALLOWED_OPTION_TYPES = ("CALL", "PUT")
HMAC_SECRET = b"black76-futures-options-v170-secret"


@dataclass(frozen=True)
class Black76Receipt:
    """Immutable quantitative audit receipt for Fischer Black (1976) Futures Option."""

    ticker: str
    option_type: str                   # "CALL" or "PUT"
    futures_price_f: float
    strike_k: float
    maturity_years: float
    risk_free_rate_r: float
    volatility: float
    spot_reference_s0: float
    implied_net_carry_rate: float
    discount_factor: float
    option_price: float
    call_price: float
    put_price: float
    discounted_intrinsic: float
    time_value_premium: float
    futures_delta: float
    futures_gamma: float
    vega: float
    theta_annual: float
    discount_rho: float
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


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _compute_implied_carry_rate(spot_s0: float, futures_f: float, maturity: float) -> float:
    """Compute continuously compounded implied cost-of-carry rate from spot and futures."""
    return float(math.log(futures_f / spot_s0) / maturity)


def classify_black76_regime(
    futures_f: float,
    strike_k: float,
    option_type: str,
    vol: float,
    carry_rate: float,
    rate_r: float,
) -> str:
    """Classify forward option moneyness, volatility regime, and commodity curve state."""
    backwardation = carry_rate < (rate_r - 0.01)
    ratio = (futures_f / strike_k) if option_type == "CALL" else (strike_k / futures_f)

    if vol >= 0.45 and abs(ratio - 1.0) <= 0.08:
        return "HIGH_VOL_ATM_FUTURES_GAMMA"
    if ratio >= 1.12:
        return "DEEP_ITM_FORWARD_OPTION"
    if ratio <= 0.88:
        return "OTM_CONVEX_TAIL_OPTIONALITY"
    if backwardation:
        return "CONVENIENCE_YIELD_BACKWARDATION_PARITY"
    return "BALANCED_FUTURES_CARRY_PARITY"


def _build_hmac_payload(
    ticker: str,
    opt_type: str,
    fut_f: float,
    strike_k: float,
    maturity: float,
    rate_r: float,
    vol: float,
    opt_price: float,
    delta_f: float,
    rho_disc: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{opt_type}:{fut_f:.4f}:{strike_k:.4f}:{maturity:.4f}:"
        f"{rate_r:.6f}:{vol:.4f}:{opt_price:.6f}:{delta_f:.6f}:{rho_disc:.6f}:"
        f"{regime}:{timestamp}"
    )


def calculate_black76_option(
    futures_price_f: float = 100.0,
    strike_k: float = 100.0,
    maturity_years: float = 1.0,
    risk_free_rate_r: float = 0.05,
    volatility: float = 0.25,
    option_type: str = "CALL",
    spot_reference_s0: float | None = None,
    ticker: str = "CL",
) -> Black76Receipt:
    """Price European futures/forward/commodity option under Fischer Black (1976)."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    opt_type = str(option_type).strip().upper()
    if opt_type not in ALLOWED_OPTION_TYPES:
        raise ValueError(f"Unsupported option_type: {option_type!r}. Allowed: {ALLOWED_OPTION_TYPES}")

    fut_f = float(futures_price_f)
    strike = float(strike_k)
    maturity = float(maturity_years)
    rate_r = float(risk_free_rate_r)
    vol = float(volatility)

    if fut_f <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("futures_price_f, strike_k, and maturity_years must be strictly positive.")
    if vol < 0.0:
        raise ValueError("volatility must be non-negative.")

    disc = math.exp(-rate_r * maturity)
    if spot_reference_s0 is not None:
        spot_s0 = float(spot_reference_s0)
        if spot_s0 <= 0.0:
            raise ValueError("spot_reference_s0 must be strictly positive when supplied.")
    else:
        spot_s0 = fut_f * disc

    carry_rate = _compute_implied_carry_rate(spot_s0, fut_f, maturity)

    call_int = max(0.0, disc * (fut_f - strike))
    put_int = max(0.0, disc * (strike - fut_f))
    vol_rt = vol * math.sqrt(maturity)

    if vol_rt < 1e-10:
        call_val = call_int
        put_val = put_int
        opt_val = call_val if opt_type == "CALL" else put_val
        if opt_type == "CALL":
            delta_f = disc if fut_f > strike else 0.0
        else:
            delta_f = -disc if strike > fut_f else 0.0
        gamma_f = 0.0
        vega_val = 0.0
        theta_val = rate_r * opt_val
    else:
        d1 = (math.log(fut_f / strike) + 0.5 * vol * vol * maturity) / vol_rt
        d2 = d1 - vol_rt
        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)
        pdf_d1 = _norm_pdf(d1)

        call_val = max(call_int, disc * (fut_f * nd1 - strike * nd2))
        put_val = max(put_int, disc * (strike * _norm_cdf(-d2) - fut_f * _norm_cdf(-d1)))
        opt_val = call_val if opt_type == "CALL" else put_val

        delta_f = (disc * nd1) if opt_type == "CALL" else (-disc * _norm_cdf(-d1))
        gamma_f = (disc * pdf_d1) / (fut_f * vol_rt)
        vega_val = disc * fut_f * pdf_d1 * math.sqrt(maturity)
        theta_val = -(disc * fut_f * pdf_d1 * vol) / (2.0 * math.sqrt(maturity)) + rate_r * opt_val

    rho_disc = -maturity * opt_val
    disc_int = call_int if opt_type == "CALL" else put_int
    time_prem = max(0.0, opt_val - disc_int)

    regime = classify_black76_regime(
        futures_f=fut_f,
        strike_k=strike,
        option_type=opt_type,
        vol=vol,
        carry_rate=carry_rate,
        rate_r=rate_r,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        opt_type=opt_type,
        fut_f=fut_f,
        strike_k=strike,
        maturity=maturity,
        rate_r=rate_r,
        vol=vol,
        opt_price=opt_val,
        delta_f=delta_f,
        rho_disc=rho_disc,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return Black76Receipt(
        ticker=cleaned_ticker,
        option_type=opt_type,
        futures_price_f=fut_f,
        strike_k=strike,
        maturity_years=maturity,
        risk_free_rate_r=rate_r,
        volatility=vol,
        spot_reference_s0=spot_s0,
        implied_net_carry_rate=carry_rate,
        discount_factor=disc,
        option_price=opt_val,
        call_price=call_val,
        put_price=put_val,
        discounted_intrinsic=disc_int,
        time_value_premium=time_prem,
        futures_delta=delta_f,
        futures_gamma=gamma_f,
        vega=vega_val,
        theta_annual=theta_val,
        discount_rho=rho_disc,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_black76_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "CL",
    strike_ratio: float = 1.0,
    maturity_years: float = 0.5,
    risk_free_rate_r: float = 0.05,
    option_type: str = "CALL",
) -> Black76Receipt:
    """Calibrate spot/futures proxy and realized annualized volatility from trailing prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        spot_s0 = float(max(1.0, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.10, 1.20))
    else:
        spot_s0 = 100.0
        ann_vol = 0.25

    fut_f = spot_s0 * math.exp(float(risk_free_rate_r) * float(maturity_years))
    strike_k = fut_f * float(np.clip(strike_ratio, 0.50, 1.80))

    return calculate_black76_option(
        futures_price_f=fut_f,
        strike_k=strike_k,
        maturity_years=float(maturity_years),
        risk_free_rate_r=float(risk_free_rate_r),
        volatility=ann_vol,
        option_type=option_type,
        spot_reference_s0=spot_s0,
        ticker=ticker,
    )


def persist_black76_record(
    receipt: Black76Receipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Black (1976) futures option audit record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS black76_futures_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        option_type TEXT NOT NULL,
        futures_price_f REAL NOT NULL,
        strike_k REAL NOT NULL,
        maturity_years REAL NOT NULL,
        volatility REAL NOT NULL,
        option_price REAL NOT NULL,
        futures_delta REAL NOT NULL,
        futures_gamma REAL NOT NULL,
        vega REAL NOT NULL,
        discount_rho REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO black76_futures_audits (
        ticker, option_type, futures_price_f, strike_k, maturity_years,
        volatility, option_price, futures_delta, futures_gamma, vega,
        discount_rho, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.option_type,
                receipt.futures_price_f,
                receipt.strike_k,
                receipt.maturity_years,
                receipt.volatility,
                receipt.option_price,
                receipt.futures_delta,
                receipt.futures_gamma,
                receipt.vega,
                receipt.discount_rho,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

