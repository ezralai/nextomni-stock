"""Robert McDonald & Daniel Siegel (1986) Value of Waiting & Irreversible Investment Engine.

Theoretical Foundation:
    McDonald, Robert, & Siegel, Daniel (1986).
    "The Value of Waiting to Invest".
    The Quarterly Journal of Economics, 101(4), pp. 707-727.

Key Concepts:
    - Perpetual Irreversible Capital Timing Option:
      Project value V follows geometric Brownian motion with risk-neutral drift (r - delta)
      and volatility sigma. Fundamental quadratic root beta_1 > 1 (guaranteed for delta > 0)
      yields optimal hysteresis investment hurdle:
      V* = (beta_1 / (beta_1 - 1)) * I > I.
    - Value-Matching, Smooth-Pasting & Waiting Premium:
      Option value F(V) = (V* - I)*(V / V*)^beta_1 for V < V* satisfies F(V*) = V* - I
      and F'(V*) = 1 identically. Even when naive NPV = V - I > 0, committing inside
      I < V < V* forfeits the Pure Value of Waiting = F(V) - max(0, V - I).
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
HMAC_SECRET = b"mcdonald-siegel-waiting-v157-secret"


@dataclass(frozen=True)
class McDonaldSiegelReceipt:
    """Immutable quantitative audit receipt for McDonald & Siegel (1986) real-option timing."""

    ticker: str
    project_value_v: float
    investment_cost_i: float
    risk_free_rate_r: float
    dividend_shortfall_delta: float
    cashflow_volatility_sigma: float
    characteristic_root_beta1: float
    hysteresis_hurdle_multiple: float
    optimal_trigger_v_star: float
    trigger_coverage_ratio: float
    naive_static_npv: float
    real_option_value_f: float
    pure_value_of_waiting: float
    waiting_option_delta: float
    invest_immediately: bool
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


def _solve_characteristic_beta1(r: float, delta: float, sigma: float) -> float:
    """Solve positive root beta_1 > 1 of 0.5*sigma^2*b*(b-1) + (r-delta)*b - r = 0."""
    if sigma < 1e-6:
        if r > delta:
            return r / (r - delta)
        return 1e6

    sig2 = sigma * sigma
    b_term = (r - delta) / sig2 - 0.5
    disc = math.sqrt(b_term * b_term + (2.0 * r) / sig2)
    return float(max(1.0 + 1e-9, 0.5 - (r - delta) / sig2 + disc))


def classify_mcdonald_siegel_regime(
    v_val: float,
    i_cost: float,
    v_star: float,
    hurdle_mult: float,
) -> str:
    """Classify corporate irreversible capex timing and hysteresis regime."""
    if v_val >= v_star:
        return "OPTIMAL_CAPEX_TRIGGER_REACHED_INVEST_NOW"
    if v_val > i_cost:
        return "NAIVE_NPV_TRAP_WAIT_FOR_HYSTERESIS_HURDLE"
    if hurdle_mult >= 2.25:
        return "HIGH_UNCERTAINTY_DEEP_OPTION_DEFERRAL"
    return "OUT_OF_THE_MONEY_PROJECT_OPTION"


def _build_hmac_payload(
    ticker: str,
    v_val: float,
    i_cost: float,
    r_rate: float,
    delta_y: float,
    sigma: float,
    v_star: float,
    opt_val: float,
    wait_val: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{v_val:.4f}:{i_cost:.4f}:{r_rate:.6f}:{delta_y:.6f}:"
        f"{sigma:.4f}:{v_star:.4f}:{opt_val:.6f}:{wait_val:.6f}:{regime}:{timestamp}"
    )


def calculate_mcdonald_siegel_waiting(
    project_value_v: float = 130.0,
    investment_cost_i: float = 100.0,
    risk_free_rate_r: float = 0.05,
    dividend_shortfall_delta: float = 0.04,
    cashflow_volatility_sigma: float = 0.28,
    ticker: str = "MS86",
) -> McDonaldSiegelReceipt:
    """Evaluate McDonald & Siegel (1986) optimal waiting trigger and real option value."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    v_val = float(project_value_v)
    i_cost = float(investment_cost_i)
    r_rate = float(risk_free_rate_r)
    delta_y = float(dividend_shortfall_delta)
    sigma = float(cashflow_volatility_sigma)

    if v_val <= 0.0 or i_cost <= 0.0:
        raise ValueError("project_value_v and investment_cost_i must be strictly positive.")
    if r_rate <= 0.0 or delta_y <= 0.0:
        raise ValueError("risk_free_rate_r and dividend_shortfall_delta must be strictly positive.")
    if sigma < 0.0:
        raise ValueError("cashflow_volatility_sigma must be non-negative.")

    beta1 = _solve_characteristic_beta1(r_rate, delta_y, sigma)
    hurdle_mult = beta1 / (beta1 - 1.0)
    v_star = hurdle_mult * i_cost
    cov_ratio = v_val / v_star

    naive_npv = v_val - i_cost
    invest_now = v_val >= v_star

    if invest_now:
        opt_val = naive_npv
        opt_delta = 1.0
    else:
        ratio_pow = math.exp(beta1 * math.log(v_val / v_star))
        opt_val = (v_star - i_cost) * ratio_pow
        opt_delta = (beta1 * (v_star - i_cost) / v_val) * ratio_pow

    waiting_prem = max(0.0, opt_val - max(0.0, naive_npv))
    regime = classify_mcdonald_siegel_regime(
        v_val=v_val,
        i_cost=i_cost,
        v_star=v_star,
        hurdle_mult=hurdle_mult,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        v_val=v_val,
        i_cost=i_cost,
        r_rate=r_rate,
        delta_y=delta_y,
        sigma=sigma,
        v_star=v_star,
        opt_val=opt_val,
        wait_val=waiting_prem,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return McDonaldSiegelReceipt(
        ticker=cleaned_ticker,
        project_value_v=v_val,
        investment_cost_i=i_cost,
        risk_free_rate_r=r_rate,
        dividend_shortfall_delta=delta_y,
        cashflow_volatility_sigma=sigma,
        characteristic_root_beta1=beta1,
        hysteresis_hurdle_multiple=hurdle_mult,
        optimal_trigger_v_star=v_star,
        trigger_coverage_ratio=cov_ratio,
        naive_static_npv=naive_npv,
        real_option_value_f=opt_val,
        pure_value_of_waiting=waiting_prem,
        waiting_option_delta=opt_delta,
        invest_immediately=invest_now,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_mcdonald_siegel_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "MS86",
    cost_multiple: float = 0.75,
    dividend_shortfall_delta: float = 0.04,
) -> McDonaldSiegelReceipt:
    """Calibrate project value and cashflow volatility proxy from trailing market closes."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        v_val = float(max(10.0, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        sigma = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.10, 1.00))
    else:
        v_val = 130.0
        sigma = 0.28

    i_cost = v_val * float(np.clip(cost_multiple, 0.20, 1.50))
    return calculate_mcdonald_siegel_waiting(
        project_value_v=v_val,
        investment_cost_i=i_cost,
        dividend_shortfall_delta=float(dividend_shortfall_delta),
        cashflow_volatility_sigma=sigma,
        ticker=ticker,
    )


def persist_mcdonald_siegel_record(
    receipt: McDonaldSiegelReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist McDonald & Siegel (1986) real-option timing evaluation to SQLite via ? queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS mcdonald_siegel_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        project_value_v REAL NOT NULL,
        investment_cost_i REAL NOT NULL,
        cashflow_volatility_sigma REAL NOT NULL,
        characteristic_root_beta1 REAL NOT NULL,
        hysteresis_hurdle_multiple REAL NOT NULL,
        optimal_trigger_v_star REAL NOT NULL,
        real_option_value_f REAL NOT NULL,
        pure_value_of_waiting REAL NOT NULL,
        invest_immediately INTEGER NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO mcdonald_siegel_audits (
        ticker, project_value_v, investment_cost_i, cashflow_volatility_sigma,
        characteristic_root_beta1, hysteresis_hurdle_multiple, optimal_trigger_v_star,
        real_option_value_f, pure_value_of_waiting, invest_immediately,
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
                receipt.project_value_v,
                receipt.investment_cost_i,
                receipt.cashflow_volatility_sigma,
                receipt.characteristic_root_beta1,
                receipt.hysteresis_hurdle_multiple,
                receipt.optimal_trigger_v_star,
                receipt.real_option_value_f,
                receipt.pure_value_of_waiting,
                int(receipt.invest_immediately),
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

