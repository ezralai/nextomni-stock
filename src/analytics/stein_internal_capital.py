"""Jeremy C. Stein (1997) Internal Capital Markets & Winner-Picking Engine.

Theoretical Foundation:
    Stein, Jeremy C. (1997).
    "Internal Capital Markets and the Competition for Corporate Resources".
    The Journal of Finance, 52(1), pp. 111-133.

Key Concepts:
    - Scarce Credit Allocation & Marginal Productivity Equalization:
      Divisions i in {1, 2} earn concave net project value V_i(K_i) = theta_i * sqrt(K_i) - K_i
      with first-best capacity K_i^FB = theta_i^2 / 4. Under constrained aggregate budget C = c_1 + c_2,
      headquarters (HQ) winner-picking equates marginal returns f_1'(K_1*) = f_2'(K_2*) at
      K_i* = K_total * theta_i^2 / (theta_1^2 + theta_2^2), generating a Conglomerate Reallocation
      Premium above focused standalone firms.
    - Corporate Socialism & Conglomerate Discount:
      When internal influence/lobbying (agency_subsidy_lambda in [0, 1]) diverts capital toward
      weaker prospects, allocative efficiency drops below standalone focus, producing a
      Conglomerate Diversification Discount.
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
HMAC_SECRET = b"stein-internal-capital-v167-secret"


@dataclass(frozen=True)
class SteinInternalCapitalReceipt:
    """Immutable quantitative audit receipt for Stein (1997) Internal Capital Markets."""

    ticker: str
    div1_productivity_theta1: float
    div2_productivity_theta2: float
    div1_standalone_budget_c1: float
    div2_standalone_budget_c2: float
    agency_subsidy_lambda: float
    total_available_capital: float
    first_best_capacity_sum: float
    budget_constrained: bool
    winner_pick_k1: float
    winner_pick_k2: float
    actual_alloc_k1: float
    actual_alloc_k2: float
    marginal_product_div1: float
    marginal_product_div2: float
    standalone_total_value: float
    winner_pick_total_value: float
    actual_conglomerate_value: float
    gross_winner_picking_surplus: float
    agency_socialism_loss: float
    net_conglomerate_premium: float
    conglomerate_premium_pct: float
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


def _division_npv(theta: float, k_val: float) -> float:
    """Concave divisional net value f(K) = theta * sqrt(K) - K."""
    return float(theta * math.sqrt(max(0.0, k_val)) - k_val)


def _division_marginal_product(theta: float, k_val: float) -> float:
    """Marginal net product f'(K) = theta / (2 * sqrt(K)) - 1."""
    if k_val <= 1e-12:
        return 1e6
    return float((theta / (2.0 * math.sqrt(k_val))) - 1.0)


def classify_stein_regime(
    constrained: bool,
    net_premium_pct: float,
    agency_loss: float,
    winner_surplus: float,
) -> str:
    """Classify conglomerate capital reallocation vs. cross-subsidization regime."""
    if not constrained and net_premium_pct >= -1e-4 and abs(winner_surplus) <= 1e-4:
        return "UNCONSTRAINED_FIRST_BEST_PARITY"
    if net_premium_pct <= -1.0:
        return "CORPORATE_SOCIALISM_CONGLOMERATE_DISCOUNT"
    if net_premium_pct >= 2.0 and winner_surplus > agency_loss:
        return "ACTIVE_WINNER_PICKING_CONGLOMERATE_PREMIUM"
    if agency_loss > winner_surplus:
        return "CROSS_SUBSIDIZATION_VALUE_DRAG"
    return "BALANCED_INTERNAL_CAPITAL_ALLOCATION"


def _build_hmac_payload(
    ticker: str,
    theta1: float,
    theta2: float,
    c1: float,
    c2: float,
    lam: float,
    wp_val: float,
    act_val: float,
    net_prem_pct: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{theta1:.4f}:{theta2:.4f}:{c1:.4f}:{c2:.4f}:{lam:.4f}:"
        f"{wp_val:.6f}:{act_val:.6f}:{net_prem_pct:.4f}:{regime}:{timestamp}"
    )


def calculate_stein_internal_capital(
    div1_productivity_theta1: float = 8.0,
    div2_productivity_theta2: float = 4.0,
    div1_standalone_budget_c1: float = 5.0,
    div2_standalone_budget_c2: float = 5.0,
    agency_subsidy_lambda: float = 0.20,
    ticker: str = "ST97",
) -> SteinInternalCapitalReceipt:
    """Evaluate Stein (1997) internal capital winner-picking vs. corporate socialism."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    th1 = float(div1_productivity_theta1)
    th2 = float(div2_productivity_theta2)
    c1 = float(div1_standalone_budget_c1)
    c2 = float(div2_standalone_budget_c2)
    lam = float(agency_subsidy_lambda)

    if th1 <= 0.0 or th2 <= 0.0:
        raise ValueError("div1_productivity_theta1 and div2_productivity_theta2 must be positive.")
    if c1 < 0.0 or c2 < 0.0 or (c1 + c2) <= 0.0:
        raise ValueError("Standalone budgets must be non-negative and total capital positive.")
    if not (0.0 <= lam <= 1.0):
        raise ValueError("agency_subsidy_lambda must satisfy 0.0 <= lambda <= 1.0.")

    fb1 = 0.25 * (th1 * th1)
    fb2 = 0.25 * (th2 * th2)
    fb_sum = fb1 + fb2
    tot_c = c1 + c2

    # Standalone firms invest up to min(c_i, fb_i)
    k1_stand = min(c1, fb1)
    k2_stand = min(c2, fb2)
    v_stand = _division_npv(th1, k1_stand) + _division_npv(th2, k2_stand)

    constrained = (c1 < fb1) or (c2 < fb2)
    k_eff_total = min(tot_c, fb_sum)

    denom_sq = th1 * th1 + th2 * th2
    wp_k1 = k_eff_total * ((th1 * th1) / denom_sq)
    wp_k2 = k_eff_total * ((th2 * th2) / denom_sq)
    v_wp = _division_npv(th1, wp_k1) + _division_npv(th2, wp_k2)

    # Agency lobbying shifts capital from optimal winner-picking toward weak division inversion
    act_k1 = (1.0 - lam) * wp_k1 + lam * wp_k2
    act_k2 = (1.0 - lam) * wp_k2 + lam * wp_k1
    v_act = _division_npv(th1, act_k1) + _division_npv(th2, act_k2)

    mp1 = _division_marginal_product(th1, wp_k1)
    mp2 = _division_marginal_product(th2, wp_k2)

    gross_surplus = max(0.0, v_wp - v_stand)
    agency_loss = max(0.0, v_wp - v_act)
    net_prem = v_act - v_stand
    prem_pct = (net_prem / max(1e-6, abs(v_stand))) * 100.0

    regime = classify_stein_regime(
        constrained=constrained,
        net_premium_pct=prem_pct,
        agency_loss=agency_loss,
        winner_surplus=gross_surplus,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        theta1=th1,
        theta2=th2,
        c1=c1,
        c2=c2,
        lam=lam,
        wp_val=v_wp,
        act_val=v_act,
        net_prem_pct=prem_pct,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return SteinInternalCapitalReceipt(
        ticker=cleaned_ticker,
        div1_productivity_theta1=th1,
        div2_productivity_theta2=th2,
        div1_standalone_budget_c1=c1,
        div2_standalone_budget_c2=c2,
        agency_subsidy_lambda=lam,
        total_available_capital=k_eff_total,
        first_best_capacity_sum=fb_sum,
        budget_constrained=constrained,
        winner_pick_k1=wp_k1,
        winner_pick_k2=wp_k2,
        actual_alloc_k1=act_k1,
        actual_alloc_k2=act_k2,
        marginal_product_div1=mp1,
        marginal_product_div2=mp2,
        standalone_total_value=v_stand,
        winner_pick_total_value=v_wp,
        actual_conglomerate_value=v_act,
        gross_winner_picking_surplus=gross_surplus,
        agency_socialism_loss=agency_loss,
        net_conglomerate_premium=net_prem,
        conglomerate_premium_pct=prem_pct,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_stein_internal_capital_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "ST97",
    agency_subsidy_lambda: float = 0.20,
    div1_productivity_theta1: float | None = None,
) -> SteinInternalCapitalReceipt:
    """Calibrate divisional return dispersion proxy from trailing asset volatility."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        window = p_arr[-min(len(p_arr), 60):]
        log_rets = np.diff(np.log(np.maximum(1e-6, window)))
        ann_vol = float(np.std(log_rets) * math.sqrt(252.0))
        theta1 = float(div1_productivity_theta1) if div1_productivity_theta1 is not None else float(np.clip(6.0 + 8.0 * ann_vol, 6.0, 12.0))
    else:
        theta1 = float(div1_productivity_theta1) if div1_productivity_theta1 is not None else 8.0

    return calculate_stein_internal_capital(
        div1_productivity_theta1=theta1,
        div2_productivity_theta2=4.0,
        div1_standalone_budget_c1=5.0,
        div2_standalone_budget_c2=5.0,
        agency_subsidy_lambda=float(agency_subsidy_lambda),
        ticker=ticker,
    )


def persist_stein_internal_capital_record(
    receipt: SteinInternalCapitalReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Stein (1997) internal capital evaluation to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS stein_internal_capital_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        div1_productivity_theta1 REAL NOT NULL,
        div2_productivity_theta2 REAL NOT NULL,
        agency_subsidy_lambda REAL NOT NULL,
        winner_pick_k1 REAL NOT NULL,
        winner_pick_k2 REAL NOT NULL,
        standalone_total_value REAL NOT NULL,
        actual_conglomerate_value REAL NOT NULL,
        net_conglomerate_premium REAL NOT NULL,
        conglomerate_premium_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO stein_internal_capital_audits (
        ticker, div1_productivity_theta1, div2_productivity_theta2, agency_subsidy_lambda,
        winner_pick_k1, winner_pick_k2, standalone_total_value, actual_conglomerate_value,
        net_conglomerate_premium, conglomerate_premium_pct, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.div1_productivity_theta1,
                receipt.div2_productivity_theta2,
                receipt.agency_subsidy_lambda,
                receipt.winner_pick_k1,
                receipt.winner_pick_k2,
                receipt.standalone_total_value,
                receipt.actual_conglomerate_value,
                receipt.net_conglomerate_premium,
                receipt.conglomerate_premium_pct,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

