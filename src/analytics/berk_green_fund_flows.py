"""Jonathan B. Berk & Richard C. Green (2004) Rational Fund Flows & Alpha Capacity Engine.

Theoretical Foundation:
    Berk, Jonathan B., & Green, Richard C. (2004).
    "Mutual Fund Flows and Performance in Rational Markets".
    Journal of Political Economy, 112(6), pp. 1269-1295.

Key Concepts:
    - Bayesian Skill Updating & Rational Investor Flows:
      Investors update expected manager gross skill phi_hat = (gamma*phi_0 + omega*R_t)/(gamma + omega)
      after observing realized active performance R_t, driving net inflows until scale dilution
      exhausts positive net alpha.
    - Decreasing Returns to Scale Equilibrium (C(q) = a * q^2):
      Average market-impact cost a * q clears competitive capital at
      q* = max(0, (phi_hat - f) / a), yielding zero expected future net alpha at q = q*,
      explaining why skilled managers earn large fee rents f * q* (maximized at f* = phi_hat/2)
      while exhibiting zero net-alpha persistence.
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
HMAC_SECRET = b"berk-green-fund-flows-v163-secret"


@dataclass(frozen=True)
class BerkGreenReceipt:
    """Immutable quantitative audit receipt for Berk & Green (2004) Fund Flow Equilibrium."""

    ticker: str
    prior_skill_phi0: float
    prior_skill_vol: float
    realized_alpha_r: float
    return_noise_vol: float
    management_fee_f: float
    cost_convexity_a: float
    current_aum_q: float
    signal_weight: float
    posterior_skill_phi_hat: float
    equilibrium_aum_q_star: float
    expected_flow_delta_q: float
    flow_pct_of_aum: float
    current_net_alpha: float
    equilibrium_net_alpha: float
    capacity_utilization_pct: float
    equilibrium_fee_revenue: float
    optimal_monopoly_fee_f_star: float
    max_manager_fee_rent: float
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


def classify_berk_green_regime(
    posterior_skill: float,
    fee: float,
    q_curr: float,
    q_star: float,
    cur_net_alpha: float,
) -> str:
    """Classify active fund capacity and rational flow equilibrium regime."""
    if posterior_skill <= fee:
        return "SUB_FEE_SKILL_EXIT_TO_INDEX"
    if cur_net_alpha > 0.005 and q_star > q_curr:
        return "UNDER_CAPACITY_POSITIVE_ALPHA_INFLOWS"
    if q_curr > q_star * 1.15:
        return "OVER_CAPACITY_NEGATIVE_ALPHA_OUTFLOWS"
    return "COMPETITIVE_ZERO_ALPHA_EQUILIBRIUM"


def _build_hmac_payload(
    ticker: str,
    phi0: float,
    realized_r: float,
    fee: float,
    cost_a: float,
    q_curr: float,
    phi_hat: float,
    q_star: float,
    cur_net_alpha: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{phi0:.6f}:{realized_r:.6f}:{fee:.6f}:{cost_a:.6f}:"
        f"{q_curr:.4f}:{phi_hat:.6f}:{q_star:.4f}:{cur_net_alpha:.6f}:"
        f"{regime}:{timestamp}"
    )


def calculate_berk_green_equilibrium(
    prior_skill_phi0: float = 0.03,
    prior_skill_vol: float = 0.04,
    realized_alpha_r: float = 0.06,
    return_noise_vol: float = 0.08,
    management_fee_f: float = 0.010,
    cost_convexity_a: float = 0.005,
    current_aum_q: float = 3.0,
    ticker: str = "BG04",
) -> BerkGreenReceipt:
    """Evaluate Berk & Green (2004) Bayesian skill updating and rational AUM equilibrium."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    phi0 = float(prior_skill_phi0)
    sig_phi = float(prior_skill_vol)
    r_real = float(realized_alpha_r)
    sig_r = float(return_noise_vol)
    fee = float(management_fee_f)
    cost_a = float(cost_convexity_a)
    q_curr = float(current_aum_q)

    if sig_phi <= 0.0 or sig_r <= 0.0:
        raise ValueError("prior_skill_vol and return_noise_vol must be strictly positive.")
    if cost_a <= 0.0:
        raise ValueError("cost_convexity_a must be strictly positive.")
    if fee < 0.0 or q_curr < 0.0:
        raise ValueError("management_fee_f and current_aum_q must be non-negative.")

    gamma_prec = 1.0 / (sig_phi * sig_phi)
    omega_prec = 1.0 / (sig_r * sig_r)
    sig_wt = omega_prec / (gamma_prec + omega_prec)
    phi_hat = (1.0 - sig_wt) * phi0 + sig_wt * r_real

    q_star = max(0.0, (phi_hat - fee) / cost_a)
    delta_q = q_star - q_curr
    flow_pct = (delta_q / max(1e-6, q_curr)) * 100.0 if q_curr > 1e-6 else (100.0 if q_star > 0 else 0.0)

    cur_net = phi_hat - cost_a * q_curr - fee
    eq_net = max(0.0, phi_hat - cost_a * q_star - fee) if q_star > 0.0 else (phi_hat - fee)

    if q_star > 1e-8:
        util_pct = (q_curr / q_star) * 100.0
    else:
        util_pct = 100.0 if q_curr > 0.0 else 0.0

    eq_fee_rev = fee * q_star
    f_star = max(0.0, 0.5 * phi_hat)
    max_rent = (phi_hat * phi_hat) / (4.0 * cost_a) if phi_hat > 0.0 else 0.0

    regime = classify_berk_green_regime(
        posterior_skill=phi_hat,
        fee=fee,
        q_curr=q_curr,
        q_star=q_star,
        cur_net_alpha=cur_net,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        phi0=phi0,
        realized_r=r_real,
        fee=fee,
        cost_a=cost_a,
        q_curr=q_curr,
        phi_hat=phi_hat,
        q_star=q_star,
        cur_net_alpha=cur_net,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return BerkGreenReceipt(
        ticker=cleaned_ticker,
        prior_skill_phi0=phi0,
        prior_skill_vol=sig_phi,
        realized_alpha_r=r_real,
        return_noise_vol=sig_r,
        management_fee_f=fee,
        cost_convexity_a=cost_a,
        current_aum_q=q_curr,
        signal_weight=sig_wt,
        posterior_skill_phi_hat=phi_hat,
        equilibrium_aum_q_star=q_star,
        expected_flow_delta_q=delta_q,
        flow_pct_of_aum=flow_pct,
        current_net_alpha=cur_net,
        equilibrium_net_alpha=eq_net,
        capacity_utilization_pct=util_pct,
        equilibrium_fee_revenue=eq_fee_rev,
        optimal_monopoly_fee_f_star=f_star,
        max_manager_fee_rent=max_rent,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_berk_green_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "BG04",
    management_fee_f: float = 0.010,
    cost_convexity_a: float = 0.005,
    current_aum_q: float = 3.0,
) -> BerkGreenReceipt:
    """Calibrate realized active performance and noise dispersion from trailing market prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        window = p_arr[-min(len(p_arr), 60):]
        log_rets = np.diff(np.log(np.maximum(1e-6, window)))
        ann_ret = float(np.clip(np.mean(log_rets) * 252.0 - 0.05, -0.15, 0.25))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.05, 0.60))
    else:
        ann_ret = 0.06
        ann_vol = 0.08

    return calculate_berk_green_equilibrium(
        realized_alpha_r=ann_ret,
        return_noise_vol=ann_vol,
        management_fee_f=float(management_fee_f),
        cost_convexity_a=float(cost_convexity_a),
        current_aum_q=float(current_aum_q),
        ticker=ticker,
    )


def persist_berk_green_record(
    receipt: BerkGreenReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Berk & Green (2004) rational fund flow record to SQLite via ? queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS berk_green_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prior_skill_phi0 REAL NOT NULL,
        realized_alpha_r REAL NOT NULL,
        management_fee_f REAL NOT NULL,
        cost_convexity_a REAL NOT NULL,
        current_aum_q REAL NOT NULL,
        posterior_skill_phi_hat REAL NOT NULL,
        equilibrium_aum_q_star REAL NOT NULL,
        expected_flow_delta_q REAL NOT NULL,
        current_net_alpha REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO berk_green_audits (
        ticker, prior_skill_phi0, realized_alpha_r, management_fee_f,
        cost_convexity_a, current_aum_q, posterior_skill_phi_hat,
        equilibrium_aum_q_star, expected_flow_delta_q, current_net_alpha,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.prior_skill_phi0,
                receipt.realized_alpha_r,
                receipt.management_fee_f,
                receipt.cost_convexity_a,
                receipt.current_aum_q,
                receipt.posterior_skill_phi_hat,
                receipt.equilibrium_aum_q_star,
                receipt.expected_flow_delta_q,
                receipt.current_net_alpha,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

