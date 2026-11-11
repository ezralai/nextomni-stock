"""Avinash K. Dixit (1989) Entry-Exit Hysteresis Band & Switching Real-Option Engine.

Theoretical Foundation:
    Dixit, Avinash K. (1989).
    "Entry and Exit Decisions under Uncertainty".
    Journal of Political Economics / Journal of Political Economy, 97(3), pp. 620-638.

Key Concepts:
    - Coupled Two-Sided Hysteresis Free Boundaries (P_H > C + r*k, P_L < C - r*l):
      An idle firm pays sunk entry cost k to enter production only at P_t >= P_H, while an
      active firm absorbs operating losses (P_t < C) down to abandonment threshold P_L before
      paying exit cost l. Between P_L < P_t < P_H lies the Hysteresis Inaction Band.
    - Value-Matching & Smooth-Pasting Solvers:
      Reduces the 4x4 coupled power-law system V_0(P) = B*P^beta1 and
      V_1(P) = A*P^(-alpha) + P/delta - C/r to an exact bounded scalar equation in z = P_L/P_H in (0, 1)
      with guaranteed Intermediate-Value sign reversal whenever k + l > 0 and l < C/r.
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
from scipy.optimize import brentq

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
ALLOWED_INITIAL_STATES = ("IDLE", "ACTIVE")
HMAC_SECRET = b"dixit-hysteresis-entry-exit-v159-secret"


@dataclass(frozen=True)
class DixitHysteresisReceipt:
    """Immutable quantitative audit receipt for Dixit (1989) Entry-Exit Hysteresis."""

    ticker: str
    current_price_p: float
    operating_cost_c: float
    entry_cost_k: float
    exit_cost_l: float
    discount_rate_r: float
    drift_shortfall_delta: float
    price_volatility_sigma: float
    current_state: str                   # "IDLE" or "ACTIVE"
    root_beta1: float
    root_alpha: float
    entry_trigger_ph: float
    exit_trigger_pl: float
    marshallian_entry_trigger: float
    marshallian_exit_trigger: float
    hysteresis_band_width: float
    option_hysteresis_premium: float
    idle_value_v0: float
    active_value_v1: float
    value_matching_residual: float
    recommended_action: str
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


def _solve_dixit_roots(r: float, delta: float, sigma: float) -> tuple[float, float]:
    """Solve characteristic roots beta_1 > 1 and alpha > 0 of 0.5*sigma^2*b*(b-1) + (r-delta)*b - r = 0."""
    if sigma < 1e-4:
        b1 = r / (r - delta) if r > delta else 1e4
        al = r / (delta - r) if delta > r else 1e4
        return float(b1), float(al)

    sig2 = sigma * sigma
    b_term = (r - delta) / sig2 - 0.5
    disc = math.sqrt(b_term * b_term + (2.0 * r) / sig2)
    beta1 = max(1.0 + 1e-8, 0.5 - (r - delta) / sig2 + disc)
    alpha = max(1e-8, (r - delta) / sig2 - 0.5 + disc)
    return float(beta1), float(alpha)


def classify_dixit_regime(
    p_val: float,
    p_h: float,
    p_l: float,
    cost_c: float,
    band_width: float,
    state: str,
) -> tuple[str, str]:
    """Determine optimal switching decision and hysteresis regime."""
    if band_width <= 1e-6:
        action = "ENTER_OR_PRODUCE" if p_val >= cost_c else "STAY_IDLE_OR_EXIT"
        return action, "FRICTIONLESS_MARSHALLIAN_THRESHOLD"
    if p_val >= p_h:
        action = "ENTER_PRODUCTION_NOW" if state == "IDLE" else "CONTINUE_PROFITABLE_PRODUCTION"
        return action, "ABOVE_ENTRY_TRIGGER_ACTIVE_EXPANSION"
    if p_val <= p_l:
        action = "STAY_IDLE" if state == "IDLE" else "ABANDON_EXIT_PRODUCTION_NOW"
        return action, "BELOW_EXIT_TRIGGER_ABANDONMENT"
    if state == "ACTIVE" and p_val < cost_c:
        return "ABSORB_OPERATING_LOSS_WAIT_IN_HYSTERESIS", "LOSS_ABSORBING_HYSTERESIS_PERSISTENCE"
    return "HOLD_STATE_INACTION_BAND", "INSIDE_HYSTERESIS_INACTION_BAND"


def _build_hmac_payload(
    ticker: str,
    p_val: float,
    c_op: float,
    k_ent: float,
    l_ex: float,
    sigma: float,
    p_h: float,
    p_l: float,
    v0: float,
    v1: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{p_val:.4f}:{c_op:.4f}:{k_ent:.4f}:{l_ex:.4f}:{sigma:.4f}:"
        f"{p_h:.4f}:{p_l:.4f}:{v0:.4f}:{v1:.4f}:{regime}:{timestamp}"
    )


def calculate_dixit_hysteresis(
    current_price_p: float = 1.10,
    operating_cost_c: float = 1.00,
    entry_cost_k: float = 2.00,
    exit_cost_l: float = 1.50,
    discount_rate_r: float = 0.05,
    drift_shortfall_delta: float = 0.05,
    price_volatility_sigma: float = 0.25,
    current_state: str = "ACTIVE",
    ticker: str = "DX89",
) -> DixitHysteresisReceipt:
    """Evaluate Dixit (1989) two-sided entry/exit hysteresis triggers and switching option values."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    state = str(current_state).strip().upper()
    if state not in ALLOWED_INITIAL_STATES:
        raise ValueError(f"Unsupported current_state: {current_state!r}. Allowed: {ALLOWED_INITIAL_STATES}")

    p_val = float(current_price_p)
    c_op = float(operating_cost_c)
    k_ent = float(entry_cost_k)
    l_ex = float(exit_cost_l)
    r_rate = float(discount_rate_r)
    delta_y = float(drift_shortfall_delta)
    sigma = float(price_volatility_sigma)

    if p_val <= 0.0 or c_op <= 0.0 or r_rate <= 0.0 or delta_y <= 0.0:
        raise ValueError("current_price_p, operating_cost_c, discount_rate_r, and drift_shortfall_delta must be strictly positive.")
    if k_ent < 0.0 or l_ex < 0.0 or sigma < 0.0:
        raise ValueError("entry_cost_k, exit_cost_l, and price_volatility_sigma must be non-negative.")

    beta1, alpha = _solve_dixit_roots(r_rate, delta_y, sigma)

    marsh_entry = (c_op / r_rate + k_ent) * delta_y
    marsh_exit = max(0.0, (c_op / r_rate - l_ex) * delta_y)

    if sigma < 1e-4:
        p_h = marsh_entry
        p_l = marsh_exit
        b_coef = 0.0
        a_coef = 0.0
        residual = 0.0
    elif (k_ent + l_ex) <= 1e-10:
        p_h = c_op * (delta_y / r_rate)
        p_l = p_h
        b_coef = 0.0
        a_coef = 0.0
        residual = 0.0
    elif l_ex >= (c_op / r_rate):
        # Exit cost exceeds perpetuity PV of operating cost C/r -> exit never optimal (P_L = 0)
        p_l = 0.0
        p_h = (beta1 / (beta1 - 1.0)) * delta_y * (c_op / r_rate + k_ent)
        b_coef = ((p_h / delta_y) - (c_op / r_rate + k_ent)) / (p_h ** beta1)
        a_coef = 0.0
        residual = abs(b_coef * (p_h ** beta1) - ((p_h / delta_y) - (c_op / r_rate + k_ent)))
    else:
        c_over_r = c_op / r_rate
        k_term = c_over_r + k_ent
        l_term = c_over_r - l_ex

        def _ph_1(z: float) -> float:
            zb = math.exp(beta1 * math.log(z))
            return ((alpha * delta_y) / (alpha + 1.0)) * ((l_term - k_term * zb) / (z - zb))

        def _ph_2(z: float) -> float:
            za = math.exp(alpha * math.log(z))
            za1 = z * za
            return ((beta1 * delta_y) / (beta1 - 1.0)) * ((k_term - l_term * za) / (1.0 - za1))

        def _residual_z(z: float) -> float:
            return _ph_2(z) - _ph_1(z)

        z_star = float(brentq(_residual_z, 1e-7, 1.0 - 1e-7, xtol=1e-11, maxiter=200))
        p_h = float(_ph_2(z_star))
        p_l = float(z_star * p_h)

        # Solve x_H = B * P_H^beta1 and y_H = A * P_H^(-alpha) cleanly without huge power overflow
        x_h = ((alpha + 1.0) * (p_h / delta_y) - alpha * k_term) / (alpha + beta1)
        y_h = (beta1 * k_term - (beta1 - 1.0) * (p_h / delta_y)) / (alpha + beta1)
        b_coef = float(x_h / math.exp(beta1 * math.log(p_h)))
        a_coef = float(y_h * math.exp(alpha * math.log(p_h)))

        # Verify value-matching residuals at P_H and P_L
        x_l = x_h * math.exp(beta1 * math.log(z_star))
        y_l = y_h * math.exp(-alpha * math.log(z_star))
        res_h = abs((x_h - y_h) - (p_h / delta_y - k_term))
        res_l = abs((y_l + p_l / delta_y - c_over_r) - (x_l - l_ex))
        residual = float(max(res_h, res_l))

    # Evaluate idle V_0(P) and active V_1(P) safely across boundary segments
    if p_h > p_l and sigma >= 1e-4 and (k_ent + l_ex) > 1e-10:
        v1_at_p = (a_coef * math.exp(-alpha * math.log(p_val)) if p_val > 0 else 0.0) + (p_val / delta_y - c_op / r_rate)
        v0_at_p = b_coef * math.exp(beta1 * math.log(p_val))
        v0_val = float(v0_at_p if p_val < p_h else max(0.0, v1_at_p - k_ent))
        v1_val = float(v1_at_p if p_val > p_l else max(0.0, v0_at_p - l_ex))
    elif sigma < 1e-4:
        v0_val = float(max(0.0, (p_val / delta_y) - (c_op / r_rate + k_ent)))
        v1_val = float(max(v0_val - l_ex, (p_val / delta_y) - (c_op / r_rate)))
    else:
        v0_val = float(max(0.0, (p_val / delta_y) - (c_op / r_rate)))
        v1_val = v0_val

    band_width = max(0.0, p_h - p_l)
    static_band = max(0.0, marsh_entry - marsh_exit)
    opt_hyst_prem = max(0.0, band_width - static_band)

    action, regime = classify_dixit_regime(
        p_val=p_val,
        p_h=p_h,
        p_l=p_l,
        cost_c=c_op,
        band_width=band_width,
        state=state,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        p_val=p_val,
        c_op=c_op,
        k_ent=k_ent,
        l_ex=l_ex,
        sigma=sigma,
        p_h=p_h,
        p_l=p_l,
        v0=v0_val,
        v1=v1_val,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return DixitHysteresisReceipt(
        ticker=cleaned_ticker,
        current_price_p=p_val,
        operating_cost_c=c_op,
        entry_cost_k=k_ent,
        exit_cost_l=l_ex,
        discount_rate_r=r_rate,
        drift_shortfall_delta=delta_y,
        price_volatility_sigma=sigma,
        current_state=state,
        root_beta1=beta1,
        root_alpha=alpha,
        entry_trigger_ph=p_h,
        exit_trigger_pl=p_l,
        marshallian_entry_trigger=marsh_entry,
        marshallian_exit_trigger=marsh_exit,
        hysteresis_band_width=band_width,
        option_hysteresis_premium=opt_hyst_prem,
        idle_value_v0=v0_val,
        active_value_v1=v1_val,
        value_matching_residual=residual,
        recommended_action=action,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_dixit_hysteresis_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "DX89",
    entry_cost_k: float = 2.0,
    exit_cost_l: float = 1.5,
    current_state: str = "ACTIVE",
) -> DixitHysteresisReceipt:
    """Calibrate normalized price ratio around trailing mean and volatility from price series."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        window = p_arr[-min(len(p_arr), 60):]
        # Normalize price vs trailing 60-day median cost anchor around C = 1.0
        c_anchor = 1.0
        norm_p = float(np.clip(window[-1] / max(1e-4, float(np.mean(window))), 0.30, 3.00))
        log_rets = np.diff(np.log(np.maximum(1e-6, window)))
        sigma = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.10, 0.90))
    else:
        c_anchor = 1.0
        norm_p = 1.10
        sigma = 0.25

    return calculate_dixit_hysteresis(
        current_price_p=norm_p,
        operating_cost_c=c_anchor,
        entry_cost_k=float(entry_cost_k),
        exit_cost_l=float(exit_cost_l),
        price_volatility_sigma=sigma,
        current_state=current_state,
        ticker=ticker,
    )


def persist_dixit_hysteresis_record(
    receipt: DixitHysteresisReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Dixit (1989) entry-exit hysteresis record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS dixit_hysteresis_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        current_price_p REAL NOT NULL,
        operating_cost_c REAL NOT NULL,
        entry_cost_k REAL NOT NULL,
        exit_cost_l REAL NOT NULL,
        price_volatility_sigma REAL NOT NULL,
        entry_trigger_ph REAL NOT NULL,
        exit_trigger_pl REAL NOT NULL,
        hysteresis_band_width REAL NOT NULL,
        recommended_action TEXT NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO dixit_hysteresis_audits (
        ticker, current_price_p, operating_cost_c, entry_cost_k, exit_cost_l,
        price_volatility_sigma, entry_trigger_ph, exit_trigger_pl,
        hysteresis_band_width, recommended_action, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.current_price_p,
                receipt.operating_cost_c,
                receipt.entry_cost_k,
                receipt.exit_cost_l,
                receipt.price_volatility_sigma,
                receipt.entry_trigger_ph,
                receipt.exit_trigger_pl,
                receipt.hysteresis_band_width,
                receipt.recommended_action,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

