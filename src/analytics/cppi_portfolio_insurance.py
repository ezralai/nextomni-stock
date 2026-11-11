"""Fischer Black & Robert W. Jones (1987) / Black & Perold (1992) CPPI & TIPP Engine.

Theoretical Foundation:
    1. Black, Fischer, & Jones, Robert W. (1987).
       "Simplifying Portfolio Insurance".
       The Journal of Portfolio Management, 14(1), pp. 48-54.

    2. Black, Fischer, & Perold, André F. (1992).
       "Theory of Constant Proportion Portfolio Insurance".
       Journal of Economic Dynamics and Control, 16(3-4), pp. 403-426.

Key Concepts:
    - Convex Cushion Allocation:
      Risky asset exposure E_t = min(L_max * W_t, m * max(0, W_t - F_t)) scales dynamically
      with cushion C_t = W_t - F_t above guaranteed floor F_t, locking principal in a safe
      reserve B_t = W_t - E_t.
    - CPPI vs. TIPP & Overnight Gap-Risk Tolerance:
      CPPI compounds the initial floor at the risk-free rate (F_t = F_0 * exp(r * t)), while
      TIPP (Time-Invariant Portfolio Protection) ratchets F_t upward with portfolio highs.
      Single-period unlevered crash tolerance equals 1 / m; jumps exceeding -1 / m trigger
      overnight gap-risk floor breaches or cash lock (C_t = 0).
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
ALLOWED_STRATEGIES = ("CPPI", "TIPP")
HMAC_SECRET = b"cppi-portfolio-insurance-v161-secret"


@dataclass(frozen=True)
class CPPIPortfolioReceipt:
    """Immutable quantitative audit receipt for Black & Jones (1987) CPPI/TIPP Insurance."""

    ticker: str
    strategy: str                      # "CPPI" or "TIPP"
    portfolio_value_w: float
    floor_value_f: float
    cushion_c: float
    multiplier_m: float
    max_leverage_cap: float
    risk_free_rate_r: float
    risky_exposure_e: float
    risky_weight_pct: float
    safe_cash_reserve_b: float
    gap_crash_tolerance_pct: float
    cash_locked: bool
    gap_risk_breach: bool
    realized_drawdown_pct: float
    uninsured_drawdown_pct: float
    drawdown_reduction_pct: float
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


def classify_cppi_regime(
    cash_locked: bool,
    gap_breach: bool,
    risky_weight_pct: float,
) -> str:
    """Classify CPPI/TIPP portfolio insurance protection and participation regime."""
    if gap_breach:
        return "GAP_RISK_FLOOR_BREACHED"
    if cash_locked or risky_weight_pct <= 1e-4:
        return "CASH_LOCKED_FLOOR_PRESERVATION"
    if risky_weight_pct >= 65.0:
        return "CONVEX_UPSIDE_PARTICIPATION"
    return "CAUTIOUS_CUSHION_BUFFER"


def _build_hmac_payload(
    ticker: str,
    strat: str,
    wealth: float,
    floor_val: float,
    mult: float,
    exp_e: float,
    wt_pct: float,
    cash_locked: bool,
    gap_breach: bool,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{strat}:{wealth:.4f}:{floor_val:.4f}:{mult:.2f}:"
        f"{exp_e:.4f}:{wt_pct:.2f}:{int(cash_locked)}:{int(gap_breach)}:{regime}:{timestamp}"
    )


def calculate_cppi_allocation(
    portfolio_value_w: float = 100.0,
    floor_value_f: float = 85.0,
    multiplier_m: float = 4.0,
    max_leverage_cap: float = 1.0,
    risk_free_rate_r: float = 0.04,
    strategy: str = "CPPI",
    ticker: str = "CPPI",
    realized_drawdown_pct: float = 0.0,
    uninsured_drawdown_pct: float = 0.0,
) -> CPPIPortfolioReceipt:
    """Compute instantaneous Black & Jones (1987) CPPI / TIPP cushion allocation."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    strat = str(strategy).strip().upper()
    if strat not in ALLOWED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy!r}. Allowed: {ALLOWED_STRATEGIES}")

    wealth = float(portfolio_value_w)
    floor_val = float(floor_value_f)
    mult = float(multiplier_m)
    lev_cap = float(max_leverage_cap)
    rf = float(risk_free_rate_r)

    if wealth <= 0.0 or floor_val < 0.0:
        raise ValueError("portfolio_value_w must be strictly positive and floor_value_f non-negative.")
    if mult < 1.0:
        raise ValueError("multiplier_m must be >= 1.0.")
    if not (1.0 <= lev_cap <= 2.0):
        raise ValueError("max_leverage_cap must be in [1.0, 2.0].")
    if rf < 0.0:
        raise ValueError("risk_free_rate_r must be non-negative.")

    gap_breach = wealth < (floor_val - 1e-8)
    cushion = max(0.0, wealth - floor_val)
    cash_locked = cushion <= 1e-8

    exp_e = min(lev_cap * wealth, mult * cushion)
    wt_pct = (exp_e / wealth) * 100.0
    reserve_b = wealth - exp_e
    gap_tol_pct = (1.0 / mult) * 100.0

    dd_red = max(0.0, float(uninsured_drawdown_pct) - float(realized_drawdown_pct))
    regime = classify_cppi_regime(
        cash_locked=cash_locked,
        gap_breach=gap_breach,
        risky_weight_pct=wt_pct,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        strat=strat,
        wealth=wealth,
        floor_val=floor_val,
        mult=mult,
        exp_e=exp_e,
        wt_pct=wt_pct,
        cash_locked=cash_locked,
        gap_breach=gap_breach,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return CPPIPortfolioReceipt(
        ticker=cleaned_ticker,
        strategy=strat,
        portfolio_value_w=wealth,
        floor_value_f=floor_val,
        cushion_c=cushion,
        multiplier_m=mult,
        max_leverage_cap=lev_cap,
        risk_free_rate_r=rf,
        risky_exposure_e=exp_e,
        risky_weight_pct=wt_pct,
        safe_cash_reserve_b=reserve_b,
        gap_crash_tolerance_pct=gap_tol_pct,
        cash_locked=cash_locked,
        gap_risk_breach=gap_breach,
        realized_drawdown_pct=float(realized_drawdown_pct),
        uninsured_drawdown_pct=float(uninsured_drawdown_pct),
        drawdown_reduction_pct=dd_red,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_cppi_path(
    prices: List[float] | np.ndarray,
    ticker: str = "CPPI",
    initial_wealth: float = 100.0,
    floor_pct: float = 0.85,
    multiplier_m: float = 4.0,
    max_leverage_cap: float = 1.0,
    risk_free_rate_r: float = 0.04,
    strategy: str = "CPPI",
) -> CPPIPortfolioReceipt:
    """Run strict zero-lookahead walk-forward CPPI/TIPP floor-protection simulation over prices."""
    strat = str(strategy).strip().upper()
    if strat not in ALLOWED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy!r}. Allowed: {ALLOWED_STRATEGIES}")
    if not (0.50 <= float(floor_pct) <= 0.99):
        raise ValueError("floor_pct must satisfy 0.50 <= floor_pct <= 0.99.")
    if float(initial_wealth) <= 0.0:
        raise ValueError("initial_wealth must be strictly positive.")
    if float(multiplier_m) < 1.0 or not (1.0 <= float(max_leverage_cap) <= 2.0):
        raise ValueError("multiplier_m must be >= 1.0 and max_leverage_cap in [1.0, 2.0].")

    p_arr = np.asarray(prices, dtype=float)
    w_init = float(initial_wealth)
    f_curr = w_init * float(floor_pct)

    if len(p_arr) < 2:
        return calculate_cppi_allocation(
            portfolio_value_w=w_init,
            floor_value_f=f_curr,
            multiplier_m=multiplier_m,
            max_leverage_cap=max_leverage_cap,
            risk_free_rate_r=risk_free_rate_r,
            strategy=strat,
            ticker=ticker,
        )

    dt = 1.0 / 252.0
    rf_step = math.exp(float(risk_free_rate_r) * dt) - 1.0

    w_curr = w_init
    w_peak = w_curr
    max_dd_cppi = 0.0
    p_peak = float(p_arr[0])
    max_dd_asset = 0.0
    ever_breached = False

    for t in range(len(p_arr) - 1):
        # Compute pre-return weight strictly prior to observing return r_{t+1} (Zero Leakage)
        cushion_t = max(0.0, w_curr - f_curr)
        risky_e = min(float(max_leverage_cap) * w_curr, float(multiplier_m) * cushion_t)
        weight_e = risky_e / max(1e-10, w_curr)
        weight_b = 1.0 - weight_e

        p_prev = max(1e-8, float(p_arr[t]))
        p_next = max(1e-8, float(p_arr[t + 1]))
        r_asset = (p_next - p_prev) / p_prev

        w_curr = max(1e-6, w_curr * (1.0 + weight_e * r_asset + weight_b * rf_step))
        f_curr = f_curr * (1.0 + rf_step)
        if strat == "TIPP":
            f_curr = max(f_curr, float(floor_pct) * w_curr)

        if w_curr < f_curr - 1e-8:
            ever_breached = True

        w_peak = max(w_peak, w_curr)
        max_dd_cppi = max(max_dd_cppi, ((w_peak - w_curr) / w_peak) * 100.0)

        p_peak = max(p_peak, p_next)
        max_dd_asset = max(max_dd_asset, ((p_peak - p_next) / p_peak) * 100.0)

    rec = calculate_cppi_allocation(
        portfolio_value_w=w_curr,
        floor_value_f=f_curr,
        multiplier_m=multiplier_m,
        max_leverage_cap=max_leverage_cap,
        risk_free_rate_r=risk_free_rate_r,
        strategy=strat,
        ticker=ticker,
        realized_drawdown_pct=max_dd_cppi,
        uninsured_drawdown_pct=max_dd_asset,
    )
    if ever_breached and not rec.gap_risk_breach:
        # Preserve historical floor breach audit signal if gap occurred mid-path
        now_iso = rec.timestamp
        sig_payload = _build_hmac_payload(
            ticker=rec.ticker,
            strat=rec.strategy,
            wealth=rec.portfolio_value_w,
            floor_val=rec.floor_value_f,
            mult=rec.multiplier_m,
            exp_e=rec.risky_exposure_e,
            wt_pct=rec.risky_weight_pct,
            cash_locked=rec.cash_locked,
            gap_breach=True,
            regime="GAP_RISK_FLOOR_BREACHED",
            timestamp=now_iso,
        )
        return CPPIPortfolioReceipt(
            ticker=rec.ticker,
            strategy=rec.strategy,
            portfolio_value_w=rec.portfolio_value_w,
            floor_value_f=rec.floor_value_f,
            cushion_c=rec.cushion_c,
            multiplier_m=rec.multiplier_m,
            max_leverage_cap=rec.max_leverage_cap,
            risk_free_rate_r=rec.risk_free_rate_r,
            risky_exposure_e=rec.risky_exposure_e,
            risky_weight_pct=rec.risky_weight_pct,
            safe_cash_reserve_b=rec.safe_cash_reserve_b,
            gap_crash_tolerance_pct=rec.gap_crash_tolerance_pct,
            cash_locked=rec.cash_locked,
            gap_risk_breach=True,
            realized_drawdown_pct=rec.realized_drawdown_pct,
            uninsured_drawdown_pct=rec.uninsured_drawdown_pct,
            drawdown_reduction_pct=rec.drawdown_reduction_pct,
            regime="GAP_RISK_FLOOR_BREACHED",
            hmac_signature=generate_hmac_signature(sig_payload),
            timestamp=now_iso,
        )
    return rec


def persist_cppi_insurance_record(
    receipt: CPPIPortfolioReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist CPPI/TIPP portfolio insurance audit record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS cppi_insurance_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        strategy TEXT NOT NULL,
        portfolio_value_w REAL NOT NULL,
        floor_value_f REAL NOT NULL,
        cushion_c REAL NOT NULL,
        multiplier_m REAL NOT NULL,
        risky_exposure_e REAL NOT NULL,
        risky_weight_pct REAL NOT NULL,
        cash_locked INTEGER NOT NULL,
        gap_risk_breach INTEGER NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO cppi_insurance_audits (
        ticker, strategy, portfolio_value_w, floor_value_f, cushion_c,
        multiplier_m, risky_exposure_e, risky_weight_pct, cash_locked,
        gap_risk_breach, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.strategy,
                receipt.portfolio_value_w,
                receipt.floor_value_f,
                receipt.cushion_c,
                receipt.multiplier_m,
                receipt.risky_exposure_e,
                receipt.risky_weight_pct,
                int(receipt.cash_locked),
                int(receipt.gap_risk_breach),
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

