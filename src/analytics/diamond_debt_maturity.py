"""Douglas W. Diamond (1991) Debt Maturity Structure & Rollover Liquidity Risk Engine.

Theoretical Foundation:
    Diamond, Douglas W. (1991).
    "Debt Maturity Structure and Liquidity Risk".
    The Quarterly Journal of Economics, 106(3), pp. 709-737.

Key Concepts:
    - Liquidity Risk vs. Rating-Upgrade Capture:
      Short-term debt exposes borrowers to date-1 refinancing panic and inefficient
      liquidation (cost q_G * (X - L + C)) when noisy interim bad news drops posterior
      creditworthiness below lender thresholds, yet captures bad-type liquidation screening
      ((1 - f)/f * q_B * L). Long-term debt insulates against rollover risk when f*X >= I.
    - Non-Monotonic Equilibrium Maturity Spectrum:
      (1) Top-rated firms (f_d * X >= I) borrow Short-Term cleanly without panic;
      (2) Intermediate firms select Long-Term debt to avoid costly rollover liquidation;
      (3) Lower-rated firms (f * X < I <= ST capacity) rely on Monitored Short-Term debt.
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
HMAC_SECRET = b"diamond-debt-maturity-v164-secret"


@dataclass(frozen=True)
class DiamondMaturityReceipt:
    """Immutable quantitative audit receipt for Diamond (1991) Debt Maturity Equilibrium."""

    ticker: str
    prior_good_prob_f: float
    investment_i: float
    continuation_payoff_x: float
    liquidation_value_l: float
    control_rent_c: float
    good_bad_news_prob_qg: float
    bad_bad_news_prob_qb: float
    downgrade_posterior_fd: float
    rollover_panic_triggered: bool
    short_term_feasible: bool
    long_term_feasible: bool
    short_term_face_rst: float
    long_term_face_rlt: float
    short_term_utility_ust: float
    long_term_utility_ult: float
    rollover_liquidation_cost: float
    upgrade_screening_gain: float
    short_vs_long_net_advantage: float
    recommended_maturity: str
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


def classify_diamond_maturity_regime(
    st_ok: bool,
    lt_ok: bool,
    panic: bool,
    ust: float,
    ult: float,
) -> tuple[str, str]:
    """Determine optimal debt maturity choice across Diamond (1991) non-monotonic spectrum."""
    if not st_ok and not lt_ok:
        return "CREDIT_RATIONED", "UNFUNDABLE_BOND_CREDIT_RATIONING"
    if st_ok and not lt_ok:
        return "SHORT_TERM", "MONITORED_SHORT_TERM_HIGH_RISK"
    if not panic and st_ok:
        return "SHORT_TERM", "SHORT_TERM_HIGH_GRADE"
    if ust >= ult and st_ok:
        return "SHORT_TERM", "SHORT_TERM_SCREENING_DOMINANT"
    return "LONG_TERM", "LONG_TERM_ROLLOVER_SHIELD"


def _build_hmac_payload(
    ticker: str,
    f_prob: float,
    inv_i: float,
    payoff_x: float,
    liq_l: float,
    rent_c: float,
    fd: float,
    net_adv: float,
    rec_mat: str,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{f_prob:.4f}:{inv_i:.4f}:{payoff_x:.4f}:{liq_l:.4f}:"
        f"{rent_c:.4f}:{fd:.4f}:{net_adv:.6f}:{rec_mat}:{regime}:{timestamp}"
    )


def calculate_diamond_debt_maturity(
    prior_good_prob_f: float = 0.72,
    investment_i: float = 1.00,
    continuation_payoff_x: float = 1.60,
    liquidation_value_l: float = 0.70,
    control_rent_c: float = 0.50,
    good_bad_news_prob_qg: float = 0.25,
    bad_bad_news_prob_qb: float = 0.80,
    ticker: str = "DM91",
) -> DiamondMaturityReceipt:
    """Evaluate Diamond (1991) Short-Term vs. Long-Term debt maturity choice."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    f_prob = float(prior_good_prob_f)
    inv_i = float(investment_i)
    payoff_x = float(continuation_payoff_x)
    liq_l = float(liquidation_value_l)
    rent_c = float(control_rent_c)
    q_g = float(good_bad_news_prob_qg)
    q_b = float(bad_bad_news_prob_qb)

    if not (0.0 < f_prob < 1.0):
        raise ValueError("prior_good_prob_f must satisfy 0.0 < f < 1.0.")
    if inv_i <= 0.0 or liq_l <= 0.0 or payoff_x <= liq_l:
        raise ValueError("Requires 0 < liquidation_value_l < continuation_payoff_x and investment_i > 0.")
    if rent_c < 0.0:
        raise ValueError("control_rent_c must be non-negative.")
    if not (0.0 <= q_g < 1.0) or not (0.0 < q_b <= 1.0) or q_g > q_b:
        raise ValueError("Requires 0 <= q_g <= q_b <= 1 with q_g < 1 and q_b > 0.")

    # Bayesian posterior rating following interim bad news realization
    p_bad_news = f_prob * q_g + (1.0 - f_prob) * q_b
    fd = (f_prob * q_g) / p_bad_news if p_bad_news > 0.0 else f_prob
    fu = (f_prob * (1.0 - q_g)) / max(1e-12, 1.0 - p_bad_news)

    # Creditors force liquidation on bad news when downgraded continuation fails I
    panic = (fd * payoff_x) < inv_i

    lt_feasible = (f_prob * payoff_x) >= inv_i
    r_lt = (inv_i / f_prob) if lt_feasible else payoff_x
    u_lt = (payoff_x - r_lt + rent_c) if lt_feasible else -1.0

    if panic:
        max_st_pledge = p_bad_news * liq_l + f_prob * (1.0 - q_g) * payoff_x
        st_feasible = max_st_pledge >= inv_i
        denom_st = max(1e-9, f_prob * (1.0 - q_g))
        r_st = max(0.0, (inv_i - p_bad_news * liq_l) / denom_st) if st_feasible else payoff_x
        u_st = ((1.0 - q_g) * (payoff_x + rent_c - r_st)) if st_feasible else -1.0
        rollover_cost = q_g * (payoff_x - liq_l + rent_c)
        upgrade_gain = ((1.0 - f_prob) / f_prob) * q_b * liq_l
    else:
        # Both bad-news and good-news states roll over without inefficient liquidation
        st_feasible = lt_feasible or (p_bad_news * min(payoff_x, inv_i / max(1e-9, fd)) >= 0 and (f_prob * payoff_x) >= inv_i)
        denom_safe = max(1e-9, f_prob)
        r_st = inv_i / denom_safe if st_feasible else payoff_x
        u_st = ((payoff_x - r_st + rent_c) + 0.02 * (fu - fd) * inv_i) if st_feasible else -1.0
        rollover_cost = 0.0
        upgrade_gain = 0.02 * (fu - fd) * inv_i

    net_adv = u_st - u_lt if (st_feasible and lt_feasible) else (u_st if st_feasible else u_lt)

    rec_mat, regime = classify_diamond_maturity_regime(
        st_ok=st_feasible,
        lt_ok=lt_feasible,
        panic=panic,
        ust=u_st,
        ult=u_lt,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        f_prob=f_prob,
        inv_i=inv_i,
        payoff_x=payoff_x,
        liq_l=liq_l,
        rent_c=rent_c,
        fd=fd,
        net_adv=net_adv,
        rec_mat=rec_mat,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return DiamondMaturityReceipt(
        ticker=cleaned_ticker,
        prior_good_prob_f=f_prob,
        investment_i=inv_i,
        continuation_payoff_x=payoff_x,
        liquidation_value_l=liq_l,
        control_rent_c=rent_c,
        good_bad_news_prob_qg=q_g,
        bad_bad_news_prob_qb=q_b,
        downgrade_posterior_fd=fd,
        rollover_panic_triggered=panic,
        short_term_feasible=st_feasible,
        long_term_feasible=lt_feasible,
        short_term_face_rst=r_st,
        long_term_face_rlt=r_lt,
        short_term_utility_ust=u_st,
        long_term_utility_ult=u_lt,
        rollover_liquidation_cost=rollover_cost,
        upgrade_screening_gain=upgrade_gain,
        short_vs_long_net_advantage=net_adv,
        recommended_maturity=rec_mat,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_diamond_maturity_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "DM91",
    prior_good_prob_f: float = 0.72,
    control_rent_c: float = 0.50,
) -> DiamondMaturityReceipt:
    """Calibrate continuation payoff and liquidation recovery proxy from market volatility."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        window = p_arr[-min(len(p_arr), 60):]
        log_rets = np.diff(np.log(np.maximum(1e-6, window)))
        ann_vol = float(np.std(log_rets) * math.sqrt(252.0))
        q_g = float(np.clip(0.10 + 0.50 * ann_vol, 0.08, 0.45))
        liq_l = float(np.clip(0.82 - 0.45 * ann_vol, 0.45, 0.85))
    else:
        q_g = 0.25
        liq_l = 0.70

    return calculate_diamond_debt_maturity(
        prior_good_prob_f=float(prior_good_prob_f),
        liquidation_value_l=liq_l,
        control_rent_c=float(control_rent_c),
        good_bad_news_prob_qg=q_g,
        ticker=ticker,
    )


def persist_diamond_maturity_record(
    receipt: DiamondMaturityReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Diamond (1991) debt maturity evaluation to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS diamond_maturity_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prior_good_prob_f REAL NOT NULL,
        liquidation_value_l REAL NOT NULL,
        control_rent_c REAL NOT NULL,
        downgrade_posterior_fd REAL NOT NULL,
        rollover_panic_triggered INTEGER NOT NULL,
        rollover_liquidation_cost REAL NOT NULL,
        short_vs_long_net_advantage REAL NOT NULL,
        recommended_maturity TEXT NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO diamond_maturity_audits (
        ticker, prior_good_prob_f, liquidation_value_l, control_rent_c,
        downgrade_posterior_fd, rollover_panic_triggered, rollover_liquidation_cost,
        short_vs_long_net_advantage, recommended_maturity, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.prior_good_prob_f,
                receipt.liquidation_value_l,
                receipt.control_rent_c,
                receipt.downgrade_posterior_fd,
                int(receipt.rollover_panic_triggered),
                receipt.rollover_liquidation_cost,
                receipt.short_vs_long_net_advantage,
                receipt.recommended_maturity,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

