"""Ho (1992) Key-Rate Duration & Fisher-Weil (1971) Liability Immunization Engine.

Implements:
- Piecewise-linear tent key-rate shift profiles across canonical tenors [1, 2, 5, 10, 30]
  satisfying exact partition-of-unity sum_k h_k(t) == 1.0 for all maturities t >= 0
- Central-difference Key-Rate Durations KRD_k whose sum equals effective duration D_eff
- Fisher & Weil (1971) duration & value-matched hedging optimization across a bond ladder
- Asset vs. liability convexity surplus verification (C_asset > C_liab) and
  non-parallel twist exposure norm
- Strict ticker validation, HMAC-SHA256 audit sealing, and parameterized SQLite logging
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-krd-1992-audit-key"
CANONICAL_KEY_TENORS = np.array([1.0, 2.0, 5.0, 10.0, 30.0], dtype=float)


@dataclass(frozen=True)
class KeyRateImmunizationReceipt:
    """Immutable receipt for Ho (1992) Key-Rate Duration & Fisher-Weil immunization."""

    ticker: str
    liability_maturity: float
    liability_pv: float
    liability_duration: float
    liability_convexity: float
    asset_duration: float
    asset_convexity: float
    convexity_surplus: float
    krd_sum: float
    partition_error: float
    krd_twist_l2_gap: float
    krd_1y: float
    krd_2y: float
    krd_5y: float
    krd_10y: float
    krd_30y: float
    regime: str
    hmac_signature: str
    timestamp: str


def _validate_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not clean or not TICKER_REGEX.match(clean):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return clean


def _build_hmac_payload(
    *,
    ticker: str,
    liability_maturity: float,
    liability_pv: float,
    liability_duration: float,
    asset_convexity: float,
    convexity_surplus: float,
    krd_twist_l2_gap: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{liability_maturity:.4f}|{liability_pv:.6f}|"
        f"{liability_duration:.6f}|{asset_convexity:.6f}|{convexity_surplus:.6f}|"
        f"{krd_twist_l2_gap:.6f}|{regime}|{timestamp}"
    )


def sign_krd_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_krd_receipt(payload)
    return hmac.compare_digest(expected, signature)


def key_rate_tent_weights(
    cashflow_times: Sequence[float] | np.ndarray,
    key_tenors: Sequence[float] | np.ndarray = CANONICAL_KEY_TENORS,
) -> np.ndarray:
    """Compute (K, M) piecewise-linear Ho (1992) tent weights summing to 1.0."""
    t = np.asarray(cashflow_times, dtype=float)
    tenors = np.asarray(key_tenors, dtype=float)
    k_len = len(tenors)
    w = np.zeros((k_len, len(t)), dtype=float)

    for idx in range(k_len):
        tk = tenors[idx]
        if idx == 0:
            t_next = tenors[1]
            vals = np.where(
                t <= tk,
                1.0,
                np.where(t < t_next, (t_next - t) / (t_next - tk), 0.0),
            )
        elif idx == k_len - 1:
            t_prev = tenors[idx - 1]
            vals = np.where(
                t >= tk,
                1.0,
                np.where(t > t_prev, (t - t_prev) / (tk - t_prev), 0.0),
            )
        else:
            t_prev = tenors[idx - 1]
            t_next = tenors[idx + 1]
            rise = np.where((t > t_prev) & (t <= tk), (t - t_prev) / (tk - t_prev), 0.0)
            fall = np.where((t > tk) & (t < t_next), (t_next - t) / (t_next - tk), 0.0)
            vals = rise + fall
        w[idx] = vals
    return w


def _price_cashflows(
    times: np.ndarray,
    cashflows: np.ndarray,
    zero_rates: np.ndarray,
) -> float:
    return float(np.sum(cashflows * np.exp(-zero_rates * times)))


def compute_krd_and_moments(
    times: np.ndarray,
    cashflows: np.ndarray,
    zero_rates: np.ndarray,
    dy: float = 1e-4,
) -> tuple[float, float, float, np.ndarray]:
    """Compute PV, effective duration, convexity, and 5-tenor Ho (1992) KRD vector."""
    pv0 = _price_cashflows(times, cashflows, zero_rates)
    if pv0 <= 0.0:
        raise ValueError("Bond or liability PV must be strictly positive")

    pv_up = _price_cashflows(times, cashflows, zero_rates + dy)
    pv_dn = _price_cashflows(times, cashflows, zero_rates - dy)
    eff_dur = float(-(pv_up - pv_dn) / (2.0 * dy * pv0))
    conv = float((pv_up + pv_dn - 2.0 * pv0) / (dy * dy * pv0))

    tent = key_rate_tent_weights(times, CANONICAL_KEY_TENORS)
    krd = np.zeros(len(CANONICAL_KEY_TENORS), dtype=float)
    for k in range(len(CANONICAL_KEY_TENORS)):
        shift = dy * tent[k]
        p_u = _price_cashflows(times, cashflows, zero_rates + shift)
        p_d = _price_cashflows(times, cashflows, zero_rates - shift)
        krd[k] = float(-(p_u - p_d) / (2.0 * dy * pv0))
    return pv0, eff_dur, conv, krd


def _bond_cashflow_schedule(maturity: float, coupon_rate: float = 0.045, face: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    n_periods = int(round(maturity))
    times = np.arange(1, n_periods + 1, dtype=float)
    cfs = np.full(n_periods, face * coupon_rate, dtype=float)
    cfs[-1] += face
    return times, cfs


def solve_key_rate_immunization(
    liability_maturity: float = 10.0,
    liability_amount: float = 1000.0,
    base_rate: float = 0.042,
    slope_bps: float = 80.0,
    mode: str = "barbell",
    ticker: str = "ALM_FUND",
) -> KeyRateImmunizationReceipt:
    """Solve Ho (1992) KRD & Fisher-Weil (1971) duration/convexity immunization."""
    clean_ticker = _validate_ticker(ticker)
    if not (2.0 <= liability_maturity <= 25.0):
        raise ValueError("liability_maturity must be in [2.0, 25.0] years")
    if liability_amount <= 0.0:
        raise ValueError("liability_amount must be > 0")
    if base_rate <= 0.0:
        raise ValueError("base_rate must be > 0")

    def _curve_at(t: np.ndarray) -> np.ndarray:
        return base_rate + (slope_bps / 10000.0) * (1.0 - np.exp(-t / 8.0))

    # Zero-coupon bullet liability at T = liability_maturity
    liab_t = np.array([float(liability_maturity)], dtype=float)
    liab_cf = np.array([float(liability_amount)], dtype=float)
    liab_pv, liab_dur, liab_conv, liab_krd = compute_krd_and_moments(
        liab_t, liab_cf, _curve_at(liab_t)
    )

    clean_mode = str(mode).strip().lower()
    if clean_mode == "barbell":
        # 2Y + 30Y barbell spans liability duration with strictly higher convexity
        bond_mats = [2.0, 30.0]
        coupons = [0.02, 0.03]
    elif clean_mode == "ladder":
        bond_mats = [2.0, 5.0, 10.0, 30.0]
        coupons = [0.04, 0.045, 0.045, 0.045]
    elif clean_mode == "bullet":
        # Short/mid bonds that match duration only via leverage/twist mismatch
        bond_mats = [5.0, 15.0]
        coupons = [0.06, 0.06]
    else:
        raise ValueError(f"Unsupported immunization mode: {mode!r}")

    b_durs = []
    b_convs = []
    b_krds = []
    for mat, cpn in zip(bond_mats, coupons):
        bt, bcf = _bond_cashflow_schedule(mat, coupon_rate=cpn)
        _, bd, bc, bk = compute_krd_and_moments(bt, bcf, _curve_at(bt))
        b_durs.append(bd)
        b_convs.append(bc)
        b_krds.append(bk)

    d_vec = np.asarray(b_durs, dtype=float)
    c_vec = np.asarray(b_convs, dtype=float)
    k_mat = np.asarray(b_krds, dtype=float)  # (M, 5)
    m_bonds = len(bond_mats)

    def _obj(w: np.ndarray) -> float:
        asset_krd = k_mat.T @ w
        return float(np.sum((asset_krd - liab_krd) ** 2))

    cons = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        {"type": "eq", "fun": lambda w: float(np.dot(w, d_vec) - liab_dur)},
    ]
    bounds = [(0.0, 1.0) for _ in range(m_bonds)] if (np.min(d_vec) <= liab_dur <= np.max(d_vec)) else [(-1.0, 2.0) for _ in range(m_bonds)]
    w0 = np.full(m_bonds, 1.0 / m_bonds, dtype=float)
    res = minimize(_obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 200})
    w_opt = np.asarray(res.x, dtype=float)

    asset_dur = float(np.dot(w_opt, d_vec))
    asset_conv = float(np.dot(w_opt, c_vec))
    conv_surplus = float(asset_conv - liab_conv)
    asset_krd = k_mat.T @ w_opt
    krd_sum = float(np.sum(asset_krd))
    partition_err = float(abs(krd_sum - asset_dur))
    twist_gap = float(np.linalg.norm(asset_krd - liab_krd))

    if conv_surplus > 0.0 and twist_gap <= 5.0:
        regime = "IMMUNIZED_CONVEXITY_SURPLUS"
    elif conv_surplus > 0.0:
        regime = "KEY_RATE_TWIST_EXPOSED"
    else:
        regime = "DURATION_MATCHED_CONCAVE_RISK"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        liability_maturity=float(liability_maturity),
        liability_pv=liab_pv,
        liability_duration=liab_dur,
        asset_convexity=asset_conv,
        convexity_surplus=conv_surplus,
        krd_twist_l2_gap=twist_gap,
        regime=regime,
        timestamp=ts,
    )
    return KeyRateImmunizationReceipt(
        ticker=clean_ticker,
        liability_maturity=float(liability_maturity),
        liability_pv=liab_pv,
        liability_duration=liab_dur,
        liability_convexity=liab_conv,
        asset_duration=asset_dur,
        asset_convexity=asset_conv,
        convexity_surplus=conv_surplus,
        krd_sum=krd_sum,
        partition_error=partition_err,
        krd_twist_l2_gap=twist_gap,
        krd_1y=float(asset_krd[0]),
        krd_2y=float(asset_krd[1]),
        krd_5y=float(asset_krd[2]),
        krd_10y=float(asset_krd[3]),
        krd_30y=float(asset_krd[4]),
        regime=regime,
        hmac_signature=sign_krd_receipt(payload),
        timestamp=ts,
    )


def evaluate_krd_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    liability_maturity: float = 10.0,
) -> KeyRateImmunizationReceipt:
    """Derive a price-implied yield environment and evaluate KRD immunization for Tab 2."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price sessions to evaluate KRD immunization")
    trend = float((arr[-1] / arr[0]) - 1.0)
    slope_bps = float(np.clip(trend * 180.0, -80.0, 160.0))
    return solve_key_rate_immunization(
        liability_maturity=liability_maturity,
        liability_amount=1000.0,
        base_rate=0.042,
        slope_bps=slope_bps,
        mode="ladder",
        ticker=ticker,
    )


def persist_krd_record(
    receipt: KeyRateImmunizationReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Ho (1992) Key-Rate Duration & Fisher-Weil immunization receipt to SQLite."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS key_rate_immunization_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                liability_maturity REAL NOT NULL,
                liability_pv REAL NOT NULL,
                liability_duration REAL NOT NULL,
                liability_convexity REAL NOT NULL,
                asset_duration REAL NOT NULL,
                asset_convexity REAL NOT NULL,
                convexity_surplus REAL NOT NULL,
                krd_sum REAL NOT NULL,
                partition_error REAL NOT NULL,
                krd_twist_l2_gap REAL NOT NULL,
                krd_1y REAL NOT NULL,
                krd_2y REAL NOT NULL,
                krd_5y REAL NOT NULL,
                krd_10y REAL NOT NULL,
                krd_30y REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO key_rate_immunization_records (
                ticker, liability_maturity, liability_pv, liability_duration,
                liability_convexity, asset_duration, asset_convexity, convexity_surplus,
                krd_sum, partition_error, krd_twist_l2_gap,
                krd_1y, krd_2y, krd_5y, krd_10y, krd_30y,
                regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.liability_maturity,
                receipt.liability_pv,
                receipt.liability_duration,
                receipt.liability_convexity,
                receipt.asset_duration,
                receipt.asset_convexity,
                receipt.convexity_surplus,
                receipt.krd_sum,
                receipt.partition_error,
                receipt.krd_twist_l2_gap,
                receipt.krd_1y,
                receipt.krd_2y,
                receipt.krd_5y,
                receipt.krd_10y,
                receipt.krd_30y,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

