"""Brinson, Hood & Beebower (1986) & Brinson & Fachler (1985) Return Attribution Engine.

Theoretical Foundation:
    Brinson, Gary P., Hood, L. Randolph, and Beebower, Gilbert L. (1986).
    "Determinants of Portfolio Performance".
    Financial Analysts Journal, 42(4), pp. 39-44.

    Brinson, Gary P., and Fachler, Nimrod (1985).
    "Measuring Non-U.S. Equity Portfolio Performance".
    The Journal of Portfolio Management, 11(3), pp. 73-76.

Key Concepts:
    - Exact Arithmetic Active-Return Decomposition (R_p - R_b):
      Active return across N asset classes or equity sectors decomposes without residual:
        R_p - R_b = sum_i (A_i + S_i + I_i)
    - Brinson-Fachler (1985) Excess Allocation Effect:
      A_i = (w_{p,i} - w_{b,i}) * (r_{b,i} - R_b), accurately crediting positive
      allocation skill to overweighting outperforming sectors even in down markets.
    - Pure Selection & Interaction Effects:
      S_i = w_{b,i} * (r_{p,i} - r_{b,i}), I_i = (w_{p,i} - w_{b,i}) * (r_{p,i} - r_{b,i}),
      with Total Selection (BHB combined) equal to S_i + I_i.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"brinson-attribution-v187-secret"


@dataclass(frozen=True)
class BrinsonAttributionReceipt:
    """Immutable quantitative audit receipt for Brinson-Hood-Beebower / Fachler attribution."""

    ticker: str
    n_sectors: int
    portfolio_return: float
    benchmark_return: float
    active_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_selection_effect: float
    top_driver_sector: str
    sector_allocation_effects: List[float]
    sector_selection_effects: List[float]
    sector_interaction_effects: List[float]
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


def _validate_weights(name: str, weights: np.ndarray) -> None:
    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError(f"{name} must be a 1D vector with at least 2 sleeves.")
    if not np.all(np.isfinite(weights)) or np.any(weights < -1e-9):
        raise ValueError(f"{name} must contain finite non-negative sleeve weights.")
    total = float(np.sum(weights))
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"{name} weights must sum to 1.0 (+/- 1e-4), got {total:.6f}.")


def _build_hmac_payload(
    *,
    ticker: str,
    n_sectors: int,
    portfolio_return: float,
    benchmark_return: float,
    active_return: float,
    allocation_effect: float,
    selection_effect: float,
    interaction_effect: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_sectors}|{portfolio_return:.6f}|{benchmark_return:.6f}|"
        f"{active_return:.6f}|{allocation_effect:.6f}|{selection_effect:.6f}|"
        f"{interaction_effect:.6f}|{regime}|{timestamp}"
    )


def sign_brinson_receipt(payload: str) -> str:
    """Compute HMAC-SHA256 hexadecimal signature for Brinson attribution payload."""
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Constant-time verification of Brinson attribution HMAC signature."""
    expected = sign_brinson_receipt(payload)
    return hmac.compare_digest(expected, str(signature))


def calculate_brinson_attribution(
    portfolio_weights: Sequence[float] | np.ndarray,
    benchmark_weights: Sequence[float] | np.ndarray,
    portfolio_returns: Sequence[float] | np.ndarray,
    benchmark_returns: Sequence[float] | np.ndarray,
    sector_names: Optional[Sequence[str]] = None,
    ticker: str = "SPY",
) -> BrinsonAttributionReceipt:
    """Compute exact Brinson-Hood-Beebower (1986) & Brinson-Fachler (1985) attribution."""
    clean_ticker = _validate_ticker(ticker)

    wp = np.asarray(portfolio_weights, dtype=float)
    wb = np.asarray(benchmark_weights, dtype=float)
    rp = np.asarray(portfolio_returns, dtype=float)
    rb = np.asarray(benchmark_returns, dtype=float)

    _validate_weights("portfolio_weights", wp)
    _validate_weights("benchmark_weights", wb)

    n_sec = len(wp)
    if len(wb) != n_sec or rp.shape != (n_sec,) or rb.shape != (n_sec,):
        raise ValueError("All weight and return vectors must share identical length N >= 2.")
    if not (np.all(np.isfinite(rp)) and np.all(np.isfinite(rb))):
        raise ValueError("Sleeve returns must contain only finite floats.")
    if np.any(rp < -1.0) or np.any(rb < -1.0):
        raise ValueError("Sleeve returns cannot be less than -100% (-1.0).")

    if sector_names is None:
        names = [f"Sleeve_{i + 1}" for i in range(n_sec)]
    else:
        if len(sector_names) != n_sec:
            raise ValueError("sector_names length must match number of sleeves.")
        names = [str(s).strip() or f"Sleeve_{i + 1}" for i, s in enumerate(sector_names)]

    wp_norm = wp / float(np.sum(wp))
    wb_norm = wb / float(np.sum(wb))

    port_ret = float(np.dot(wp_norm, rp))
    bench_ret = float(np.dot(wb_norm, rb))
    active_ret = float(port_ret - bench_ret)

    # Brinson-Fachler (1985) Allocation + BHB (1986) Selection & Interaction
    alloc_vec = (wp_norm - wb_norm) * (rb - bench_ret)
    select_vec = wb_norm * (rp - rb)
    interact_vec = (wp_norm - wb_norm) * (rp - rb)

    alloc_tot = float(np.sum(alloc_vec))
    select_tot = float(np.sum(select_vec))
    interact_tot = float(np.sum(interact_vec))
    total_select = float(select_tot + interact_tot)

    sector_total_active = alloc_vec + select_vec + interact_vec
    top_idx = int(np.argmax(np.abs(sector_total_active)))
    top_sector = names[top_idx]

    if abs(active_ret) < 1e-4:
        regime = "MATCHED_BENCHMARK_TRACKING"
    elif active_ret < 0.0:
        regime = "ACTIVE_UNDERPERFORMANCE_DRAG"
    elif total_select >= alloc_tot:
        regime = "SELECTION_DRIVEN_ALPHA"
    else:
        regime = "ALLOCATION_DRIVEN_ALPHA"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_sectors=n_sec,
        portfolio_return=port_ret,
        benchmark_return=bench_ret,
        active_return=active_ret,
        allocation_effect=alloc_tot,
        selection_effect=select_tot,
        interaction_effect=interact_tot,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = sign_brinson_receipt(payload)

    return BrinsonAttributionReceipt(
        ticker=clean_ticker,
        n_sectors=n_sec,
        portfolio_return=port_ret,
        benchmark_return=bench_ret,
        active_return=active_ret,
        allocation_effect=alloc_tot,
        selection_effect=select_tot,
        interaction_effect=interact_tot,
        total_selection_effect=total_select,
        top_driver_sector=top_sector,
        sector_allocation_effects=[float(x) for x in alloc_vec],
        sector_selection_effects=[float(x) for x in select_vec],
        sector_interaction_effects=[float(x) for x in interact_vec],
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_brinson_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
) -> BrinsonAttributionReceipt:
    """Evaluate stylized 4-sleeve active tilt attribution around trailing asset return."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 21:
        base_ret = float((p_arr[-1] / max(1e-6, p_arr[-min(len(p_arr), 63)])) - 1.0)
        base_ret = max(-0.80, min(2.00, base_ret))
    else:
        base_ret = 0.08

    wb = [0.35, 0.30, 0.20, 0.15]
    wp = [0.45, 0.25, 0.20, 0.10]
    rb = [base_ret * 1.15, base_ret * 0.95, base_ret * 0.85, base_ret * 1.05]
    rp = [rb[0] + 0.015, rb[1] + 0.005, rb[2] - 0.005, rb[3] + 0.010]
    sectors = ["Tech_Growth", "Core_Equity", "Defensive_Val", "Quality_Tilt"]
    return calculate_brinson_attribution(wp, wb, rp, rb, sector_names=sectors, ticker=ticker)


def persist_brinson_record(
    receipt: BrinsonAttributionReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Brinson-Hood-Beebower / Brinson-Fachler audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS brinson_attribution_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        n_sectors INTEGER NOT NULL,
        portfolio_return REAL NOT NULL,
        benchmark_return REAL NOT NULL,
        active_return REAL NOT NULL,
        allocation_effect REAL NOT NULL,
        selection_effect REAL NOT NULL,
        interaction_effect REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO brinson_attribution_audits (
        ticker, n_sectors, portfolio_return, benchmark_return, active_return,
        allocation_effect, selection_effect, interaction_effect, regime,
        hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.n_sectors,
                receipt.portfolio_return,
                receipt.benchmark_return,
                receipt.active_return,
                receipt.allocation_effect,
                receipt.selection_effect,
                receipt.interaction_effect,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

