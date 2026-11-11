"""John C. Hull & Alan D. White (1990) No-Arbitrage Term Structure & Bond Option Engine.

Theoretical Foundation:
    Hull, John C., & White, Alan D. (1990).
    "Pricing Interest-Rate-Derivative Securities".
    The Review of Financial Studies, 3(4), pp. 573-592.

Key Concepts:
    - Exact Initial Term-Structure Fit:
      Fits time-dependent drift theta(t) in dr_t = (theta(t) - a*r_t)dt + sigma*dW_t
      to match initial market discount factors P(0, T) and P(0, S) identically.
    - Closed-Form Zero-Coupon Bond Options (Jamshidian / Hull-White):
      sigma_p = sigma * sqrt((1 - exp(-2*a*T)) / (2*a)) * B(T, S),
      ZBC = P(0, S)*N(h) - K*P(0, T)*N(h - sigma_p),
      ZBP = K*P(0, T)*N(-h + sigma_p) - P(0, S)*N(-h),
      satisfying exact Bond Put-Call Parity ZBC - ZBP == P(0, S) - K*P(0, T).
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
HMAC_SECRET = b"hull-white-bond-options-v155-secret"


@dataclass(frozen=True)
class HullWhiteReceipt:
    """Immutable quantitative audit receipt for Hull & White (1990) No-Arbitrage Bond Options."""

    ticker: str
    option_type: str                   # "CALL" or "PUT"
    short_rate_r0: float
    long_yield_y_inf: float
    mean_reversion_a: float
    rate_volatility_sigma: float
    option_maturity_t: float
    bond_maturity_s: float
    strike_k: float
    discount_p_0_t: float
    discount_p_0_s: float
    forward_bond_price: float
    forward_yield_rate: float
    bond_price_vol_sigma_p: float
    option_price: float
    bond_call_price: float
    bond_put_price: float
    forward_intrinsic_value: float
    time_value_premium: float
    spot_delta: float
    curve_fit_error_bps: float         # Exact 0.0 under theta(t) no-arbitrage calibration
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


def _b_factor(a: float, tau: float) -> float:
    """Hull-White duration factor B(T, S) = (1 - exp(-a*tau)) / a with small-a numerical guard."""
    x = a * tau
    if x < 1e-8:
        return tau * (1.0 - 0.5 * x + (x * x) / 6.0)
    return -math.expm1(-x) / a


def _market_discount_factor(r0: float, y_inf: float, beta: float, u: float) -> float:
    """Smooth parametric market zero-coupon discount factor P(0, u) = exp(-u * y(u))."""
    bx = beta * u
    phi_u = 1.0 - 0.5 * bx if bx < 1e-8 else (-math.expm1(-bx) / bx)
    spot_yield = y_inf + (r0 - y_inf) * phi_u
    return math.exp(-u * spot_yield)


def classify_hull_white_regime(
    forward_bond_price: float,
    strike_k: float,
    sigma_p: float,
    r0: float,
    y_inf: float,
) -> str:
    """Classify rate-derivative moneyness and term-structure environment."""
    slope = y_inf - r0
    ratio = forward_bond_price / strike_k
    if sigma_p >= 0.08 and abs(ratio - 1.0) <= 0.10:
        return "HIGH_RATE_CONVEXITY_ATM_GAMMA"
    if ratio >= 1.10:
        return "IN_THE_MONEY_BOND_CALL_ZONE" if slope >= 0.0 else "INVERTED_CURVE_HIGH_FORWARD_BOND"
    if ratio <= 0.90:
        return "HIGH_YIELD_STRIKE_BOND_PUT_DOMINANT"
    if slope < -0.005:
        return "INVERTED_TERM_STRUCTURE_DISCOUNTING"
    return "BALANCED_NO_ARBITRAGE_CURVE"


def _build_hmac_payload(
    ticker: str,
    opt_type: str,
    r0: float,
    y_inf: float,
    a: float,
    sigma: float,
    t_opt: float,
    s_bond: float,
    strike: float,
    opt_price: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{opt_type}:{r0:.6f}:{y_inf:.6f}:{a:.4f}:{sigma:.4f}:"
        f"{t_opt:.4f}:{s_bond:.4f}:{strike:.4f}:{opt_price:.6f}:{regime}:{timestamp}"
    )


def calculate_hull_white_bond_option(
    short_rate_r0: float = 0.035,
    long_yield_y_inf: float = 0.050,
    curve_decay_beta: float = 0.40,
    mean_reversion_a: float = 0.15,
    rate_volatility_sigma: float = 0.015,
    option_maturity_t: float = 2.0,
    bond_maturity_s: float = 10.0,
    strike_k: float = 0.65,
    option_type: str = "CALL",
    ticker: str = "HW90",
) -> HullWhiteReceipt:
    """Price zero-coupon bond Call/Put under exact Hull & White (1990) no-arbitrage curve fitting."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    opt_type = str(option_type).strip().upper()
    if opt_type not in ALLOWED_OPTION_TYPES:
        raise ValueError(f"Unsupported option_type: {option_type!r}. Allowed: {ALLOWED_OPTION_TYPES}")

    r0 = float(short_rate_r0)
    y_inf = float(long_yield_y_inf)
    beta = float(curve_decay_beta)
    a = float(mean_reversion_a)
    sigma = float(rate_volatility_sigma)
    t_opt = float(option_maturity_t)
    s_bond = float(bond_maturity_s)
    k_strike = float(strike_k)

    if beta <= 0.0 or a <= 0.0:
        raise ValueError("curve_decay_beta and mean_reversion_a must be strictly positive.")
    if sigma < 0.0:
        raise ValueError("rate_volatility_sigma must be non-negative.")
    if t_opt <= 0.0 or s_bond <= t_opt:
        raise ValueError("Requires 0 < option_maturity_t < bond_maturity_s.")
    if k_strike <= 0.0:
        raise ValueError("strike_k must be strictly positive.")

    p0_t = _market_discount_factor(r0, y_inf, beta, t_opt)
    p0_s = _market_discount_factor(r0, y_inf, beta, s_bond)
    fwd_bond = p0_s / p0_t
    fwd_yield = -math.log(max(1e-12, fwd_bond)) / (s_bond - t_opt)

    b_ts = _b_factor(a, s_bond - t_opt)
    x_2a = 2.0 * a * t_opt
    var_factor = t_opt * (1.0 - 0.5 * x_2a) if x_2a < 1e-8 else (-math.expm1(-x_2a) / (2.0 * a))
    sigma_p = sigma * math.sqrt(max(0.0, var_factor)) * b_ts

    call_intrinsic = max(0.0, p0_s - k_strike * p0_t)
    put_intrinsic = max(0.0, k_strike * p0_t - p0_s)

    if sigma_p < 1e-10:
        call_val = call_intrinsic
        put_val = put_intrinsic
        if opt_type == "CALL":
            delta = 1.0 if p0_s > k_strike * p0_t else 0.0
        else:
            delta = -1.0 if k_strike * p0_t > p0_s else 0.0
    else:
        h_arg = (math.log(p0_s / (k_strike * p0_t)) / sigma_p) + 0.5 * sigma_p
        call_val = max(call_intrinsic, p0_s * _norm_cdf(h_arg) - k_strike * p0_t * _norm_cdf(h_arg - sigma_p))
        put_val = max(put_intrinsic, k_strike * p0_t * _norm_cdf(-h_arg + sigma_p) - p0_s * _norm_cdf(-h_arg))
        delta = _norm_cdf(h_arg) if opt_type == "CALL" else (_norm_cdf(h_arg) - 1.0)

    opt_val = call_val if opt_type == "CALL" else put_val
    intrinsic_val = call_intrinsic if opt_type == "CALL" else put_intrinsic
    time_prem = max(0.0, opt_val - intrinsic_val)

    regime = classify_hull_white_regime(
        forward_bond_price=fwd_bond,
        strike_k=k_strike,
        sigma_p=sigma_p,
        r0=r0,
        y_inf=y_inf,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        opt_type=opt_type,
        r0=r0,
        y_inf=y_inf,
        a=a,
        sigma=sigma,
        t_opt=t_opt,
        s_bond=s_bond,
        strike=k_strike,
        opt_price=opt_val,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return HullWhiteReceipt(
        ticker=cleaned_ticker,
        option_type=opt_type,
        short_rate_r0=r0,
        long_yield_y_inf=y_inf,
        mean_reversion_a=a,
        rate_volatility_sigma=sigma,
        option_maturity_t=t_opt,
        bond_maturity_s=s_bond,
        strike_k=k_strike,
        discount_p_0_t=p0_t,
        discount_p_0_s=p0_s,
        forward_bond_price=fwd_bond,
        forward_yield_rate=fwd_yield,
        bond_price_vol_sigma_p=sigma_p,
        option_price=opt_val,
        bond_call_price=call_val,
        bond_put_price=put_val,
        forward_intrinsic_value=intrinsic_val,
        time_value_premium=time_prem,
        spot_delta=delta,
        curve_fit_error_bps=0.0,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_hull_white_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "HW90",
    short_rate_r0: float = 0.035,
    long_yield_y_inf: float = 0.050,
    option_maturity_t: float = 2.0,
    bond_maturity_s: float = 10.0,
    option_type: str = "CALL",
) -> HullWhiteReceipt:
    """Calibrate normal rate volatility from trailing market returns and price ATM forward bond option."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        ann_vol = float(np.std(log_rets) * math.sqrt(252.0))
        rate_vol = float(np.clip(ann_vol * 0.06, 0.005, 0.040))
    else:
        rate_vol = 0.015

    # Set near-ATM strike equal to forward bond price rounded cleanly
    p0_t = _market_discount_factor(short_rate_r0, long_yield_y_inf, 0.40, option_maturity_t)
    p0_s = _market_discount_factor(short_rate_r0, long_yield_y_inf, 0.40, bond_maturity_s)
    atm_strike = p0_s / p0_t

    return calculate_hull_white_bond_option(
        short_rate_r0=short_rate_r0,
        long_yield_y_inf=long_yield_y_inf,
        rate_volatility_sigma=rate_vol,
        option_maturity_t=option_maturity_t,
        bond_maturity_s=bond_maturity_s,
        strike_k=atm_strike,
        option_type=option_type,
        ticker=ticker,
    )


def persist_hull_white_record(
    receipt: HullWhiteReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Hull & White (1990) bond option evaluation to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS hull_white_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        option_type TEXT NOT NULL,
        short_rate_r0 REAL NOT NULL,
        long_yield_y_inf REAL NOT NULL,
        mean_reversion_a REAL NOT NULL,
        rate_volatility_sigma REAL NOT NULL,
        option_maturity_t REAL NOT NULL,
        bond_maturity_s REAL NOT NULL,
        strike_k REAL NOT NULL,
        forward_bond_price REAL NOT NULL,
        bond_price_vol_sigma_p REAL NOT NULL,
        option_price REAL NOT NULL,
        bond_call_price REAL NOT NULL,
        bond_put_price REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO hull_white_audits (
        ticker, option_type, short_rate_r0, long_yield_y_inf, mean_reversion_a,
        rate_volatility_sigma, option_maturity_t, bond_maturity_s, strike_k,
        forward_bond_price, bond_price_vol_sigma_p, option_price,
        bond_call_price, bond_put_price, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.option_type,
                receipt.short_rate_r0,
                receipt.long_yield_y_inf,
                receipt.mean_reversion_a,
                receipt.rate_volatility_sigma,
                receipt.option_maturity_t,
                receipt.bond_maturity_s,
                receipt.strike_k,
                receipt.forward_bond_price,
                receipt.bond_price_vol_sigma_p,
                receipt.option_price,
                receipt.bond_call_price,
                receipt.bond_put_price,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

