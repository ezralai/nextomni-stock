"""Milton Harris & Artur Raviv (1990) Capital Structure and the Informational Role of Debt Engine.

Theoretical Foundation:
    Harris, Milton, & Raviv, Artur (1990).
    "Capital Structure and the Informational Role of Debt".
    The Journal of Finance, 45(2), pp. 321-349.

Key Concepts:
    - Auditing & Liquidation Discipline via Default:
      Management prefers continuing operations privately even when asset liquidation value L
      exceeds continuation cash flows. Promised debt D forces an informative default
      investigation (audit cost k) whenever realized earnings Y < D, discovering true firm
      value and executing efficient liquidation whenever Y < L.
    - Analytical Tradeoff & Optimal Debt D*:
      Over Uniform[y_min, y_max], marginal firm value is dV/dD = ((L - D) - k)/(y_max - y_min)
      for D <= L and -k/(y_max - y_min) for D > L, producing exact interior optimal debt
      D* = clip(L - k, y_min, min(L, y_max)) that balances liquidation discipline against
      expected bankruptcy investigation costs k * PD.
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
HMAC_SECRET = b"harris-raviv-informational-debt-v162-secret"


@dataclass(frozen=True)
class HarrisRavivDebtReceipt:
    """Immutable quantitative audit receipt for Harris & Raviv (1990) Informational Debt."""

    ticker: str
    y_min: float
    y_max: float
    liquidation_value_l: float
    investigation_cost_k: float
    promised_debt_d: float
    unlevered_value: float
    firm_value_v_d: float
    optimal_debt_d_star: float
    optimal_firm_value_v_star: float
    default_audit_prob: float
    liquidation_prob: float
    gross_liquidation_gain: float
    expected_investigation_cost: float
    net_discipline_value: float
    optimal_discipline_value: float
    over_leverage_waste: float
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


def _evaluate_firm_value_components(
    d_face: float,
    y_min: float,
    y_max: float,
    liq_l: float,
    inv_k: float,
) -> tuple[float, float, float, float, float]:
    """Evaluate exact closed-form V(D) components under Uniform[y_min, y_max]."""
    span = y_max - y_min
    d_clamped = float(np.clip(d_face, y_min, y_max))

    default_prob = (d_clamped - y_min) / span
    liq_upper = min(d_clamped, liq_l)
    liq_prob = max(0.0, liq_upper - y_min) / span

    if liq_upper > y_min:
        delta_u = liq_upper - y_min
        gross_gain = ((liq_l - y_min) * delta_u - 0.5 * (delta_u ** 2)) / span
    else:
        gross_gain = 0.0

    exp_inv = inv_k * default_prob
    net_disc = gross_gain - exp_inv
    return default_prob, liq_prob, gross_gain, exp_inv, net_disc


def classify_harris_raviv_regime(
    d_val: float,
    d_star: float,
    liq_l: float,
    y_min: float,
    inv_k: float,
    over_waste: float,
) -> str:
    """Classify informational debt discipline and investigation cost regime."""
    if liq_l <= y_min + inv_k:
        return "ALL_EQUITY_ZERO_DEFAULT_PROHIBITIVE_AUDIT_COST"
    if d_val > d_star and over_waste >= 0.25:
        return "EXCESS_DEBT_OVER_INVESTIGATION_WASTE"
    if d_val < d_star - 1e-6:
        return "UNDER_LEVERAGED_CONTINUATION_BIAS"
    if abs(d_val - d_star) <= 1e-4:
        return "OPTIMAL_INFORMATIONAL_DEBT_DISCIPLINE"
    return "NEAR_OPTIMAL_AUDIT_DISCIPLINE"


def _build_hmac_payload(
    ticker: str,
    y_min: float,
    y_max: float,
    liq_l: float,
    inv_k: float,
    d_val: float,
    v_d: float,
    d_star: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{y_min:.4f}:{y_max:.4f}:{liq_l:.4f}:{inv_k:.4f}:"
        f"{d_val:.4f}:{v_d:.6f}:{d_star:.4f}:{regime}:{timestamp}"
    )


def calculate_harris_raviv_debt(
    y_min: float = 40.0,
    y_max: float = 160.0,
    liquidation_value_l: float = 95.0,
    investigation_cost_k: float = 8.0,
    promised_debt_d: float | None = None,
    ticker: str = "HR90",
) -> HarrisRavivDebtReceipt:
    """Evaluate Harris & Raviv (1990) informational role of debt and optimal debt level D*."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    y0 = float(y_min)
    y1 = float(y_max)
    liq_l = float(liquidation_value_l)
    inv_k = float(investigation_cost_k)

    if y0 < 0.0 or y1 <= y0:
        raise ValueError("Requires 0 <= y_min < y_max for cash-flow support.")
    if liq_l < 0.0 or inv_k < 0.0:
        raise ValueError("liquidation_value_l and investigation_cost_k must be non-negative.")

    unlevered_val = 0.5 * (y0 + y1)
    upper_eff_l = min(y1, liq_l)
    d_star = float(np.clip(liq_l - inv_k, y0, upper_eff_l))

    eff_d = float(promised_debt_d) if promised_debt_d is not None else d_star
    if eff_d < 0.0:
        raise ValueError("promised_debt_d must be non-negative.")

    dp, lp, gg, eic, net_disc = _evaluate_firm_value_components(eff_d, y0, y1, liq_l, inv_k)
    _, _, _, _, opt_disc = _evaluate_firm_value_components(d_star, y0, y1, liq_l, inv_k)

    v_d = unlevered_val + net_disc
    v_star = unlevered_val + opt_disc
    waste = max(0.0, v_star - v_d)

    regime = classify_harris_raviv_regime(
        d_val=eff_d,
        d_star=d_star,
        liq_l=liq_l,
        y_min=y0,
        inv_k=inv_k,
        over_waste=waste,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        y_min=y0,
        y_max=y1,
        liq_l=liq_l,
        inv_k=inv_k,
        d_val=eff_d,
        v_d=v_d,
        d_star=d_star,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return HarrisRavivDebtReceipt(
        ticker=cleaned_ticker,
        y_min=y0,
        y_max=y1,
        liquidation_value_l=liq_l,
        investigation_cost_k=inv_k,
        promised_debt_d=eff_d,
        unlevered_value=unlevered_val,
        firm_value_v_d=v_d,
        optimal_debt_d_star=d_star,
        optimal_firm_value_v_star=v_star,
        default_audit_prob=dp,
        liquidation_prob=lp,
        gross_liquidation_gain=gg,
        expected_investigation_cost=eic,
        net_discipline_value=net_disc,
        optimal_discipline_value=opt_disc,
        over_leverage_waste=waste,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_harris_raviv_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "HR90",
    liquidation_ratio: float = 0.85,
    investigation_cost_k: float = 8.0,
) -> HarrisRavivDebtReceipt:
    """Calibrate cash-flow uncertainty span [y_min, y_max] and liquidation anchor from prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        spot = float(max(20.0, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.12, 0.80))
        half_span = spot * min(0.75, 1.5 * ann_vol)
        y0 = max(5.0, spot - half_span)
        y1 = spot + half_span
        liq_l = spot * float(np.clip(liquidation_ratio, 0.25, 1.25))
    else:
        y0, y1, liq_l = 40.0, 160.0, 95.0

    return calculate_harris_raviv_debt(
        y_min=y0,
        y_max=y1,
        liquidation_value_l=liq_l,
        investigation_cost_k=float(investigation_cost_k),
        ticker=ticker,
    )


def persist_harris_raviv_record(
    receipt: HarrisRavivDebtReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Harris & Raviv (1990) informational debt evaluation to SQLite via ? queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS harris_raviv_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        liquidation_value_l REAL NOT NULL,
        investigation_cost_k REAL NOT NULL,
        promised_debt_d REAL NOT NULL,
        firm_value_v_d REAL NOT NULL,
        optimal_debt_d_star REAL NOT NULL,
        default_audit_prob REAL NOT NULL,
        net_discipline_value REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO harris_raviv_audits (
        ticker, liquidation_value_l, investigation_cost_k, promised_debt_d,
        firm_value_v_d, optimal_debt_d_star, default_audit_prob,
        net_discipline_value, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.liquidation_value_l,
                receipt.investigation_cost_k,
                receipt.promised_debt_d,
                receipt.firm_value_v_d,
                receipt.optimal_debt_d_star,
                receipt.default_audit_prob,
                receipt.net_discipline_value,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

