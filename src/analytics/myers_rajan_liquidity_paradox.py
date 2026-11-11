"""Stewart C. Myers & Raghuram G. Rajan (1998) Paradox of Liquidity Engine.

Theoretical Foundation:
    Myers, Stewart C., & Rajan, Raghuram G. (1998).
    "The Paradox of Liquidity".
    The Quarterly Journal of Economics, 113(3), pp. 733-771.

Key Concepts:
    - Creditor Liquidation Capacity vs. Insider Transformation Ceiling:
      Higher asset liquidity alpha in (0, 1] expands external collateral recovery
      D_liq(alpha) = alpha * c_1, yet makes it easier for managers to transform assets
      into untraceable private substitutes yielding alpha * (1 - lambda_trans) * c_1.
      Incentive compatibility caps long-term borrowing at:
      D_trans(alpha) = max(0, (1 - theta) * d_2 - alpha * (1 - lambda_trans) * c_1).
    - Non-Monotonic Humped Debt Capacity:
      Binding borrowing capacity min(D_liq, D_trans) increases for alpha < alpha*
      and decreases in the Paradox Zone (alpha > alpha*), creating demand for
      short-term bank monitoring to neutralize transformation risk.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"myers-rajan-liquidity-paradox-v154-secret"


@dataclass(frozen=True)
class MyersRajanReceipt:
    """Immutable quantitative audit receipt for Myers & Rajan (1998) Paradox of Liquidity."""

    ticker: str
    invested_assets_c1: float
    continuation_value_d2: float
    asset_liquidity_alpha: float
    transformation_friction_lambda: float
    insider_rent_share_theta: float
    liquidation_debt_capacity: float
    transformation_debt_ceiling: float
    binding_debt_capacity: float
    optimal_liquidity_alpha_star: float
    peak_debt_capacity: float
    paradox_capacity_loss: float
    paradox_zone_active: bool
    bank_monitoring_requirement_pct: float
    leverage_ratio_pct: float
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


def classify_myers_rajan_regime(
    alpha: float,
    alpha_star: float,
    binding_debt: float,
    paradox_loss: float,
    liquidation_debt: float,
) -> str:
    """Classify corporate borrowing capacity and liquidity-paradox governance regime."""
    if binding_debt <= 1e-6:
        return "TRANSFORMATION_DEFAULT_ZERO_DEBT_CAPACITY"
    loss_ratio = paradox_loss / max(1e-6, liquidation_debt)
    if alpha > alpha_star and loss_ratio >= 0.30:
        return "SEVERE_LIQUIDITY_PARADOX_SHORT_DEBT_REQUIRED"
    if alpha > alpha_star:
        return "TRANSFORMATION_CONSTRAINED_PARADOX_ZONE"
    if abs(alpha - alpha_star) <= 0.03:
        return "OPTIMAL_INTERIOR_LIQUIDITY_FRONTIER"
    return "COLLATERAL_LIMITED_ILLIQUID_ZONE"


def _build_hmac_payload(
    ticker: str,
    c1: float,
    d2: float,
    alpha: float,
    lam: float,
    theta: float,
    binding_debt: float,
    alpha_star: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{c1:.4f}:{d2:.4f}:{alpha:.4f}:{lam:.4f}:{theta:.4f}:"
        f"{binding_debt:.4f}:{alpha_star:.4f}:{regime}:{timestamp}"
    )


def calculate_myers_rajan_paradox(
    invested_assets_c1: float = 100.0,
    continuation_value_d2: float = 125.0,
    asset_liquidity_alpha: float = 0.75,
    transformation_friction_lambda: float = 0.25,
    insider_rent_share_theta: float = 0.0,
    ticker: str = "MR98",
) -> MyersRajanReceipt:
    """Evaluate Myers & Rajan (1998) liquidation vs. transformation debt capacity."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    c1 = float(invested_assets_c1)
    d2 = float(continuation_value_d2)
    alpha = float(asset_liquidity_alpha)
    lam = float(transformation_friction_lambda)
    theta = float(insider_rent_share_theta)

    if c1 <= 0.0 or d2 <= 0.0:
        raise ValueError("invested_assets_c1 and continuation_value_d2 must be strictly positive.")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("asset_liquidity_alpha must satisfy 0.0 < alpha <= 1.0.")
    if not (0.0 <= lam <= 1.0):
        raise ValueError("transformation_friction_lambda must satisfy 0.0 <= lambda <= 1.0.")
    if not (0.0 <= theta < 1.0):
        raise ValueError("insider_rent_share_theta must satisfy 0.0 <= theta < 1.0.")

    pledgeable_d2 = (1.0 - theta) * d2
    d_liq = alpha * c1
    d_trans = max(0.0, pledgeable_d2 - alpha * (1.0 - lam) * c1)
    binding_debt = min(d_liq, d_trans)

    alpha_star_raw = pledgeable_d2 / ((2.0 - lam) * c1)
    alpha_star = float(min(1.0, alpha_star_raw))
    peak_debt = alpha_star * c1

    paradox_active = alpha > alpha_star + 1e-12
    paradox_loss = max(0.0, d_liq - binding_debt)
    bank_req_pct = (paradox_loss / max(1e-6, d_liq)) * 100.0
    lev_pct = (binding_debt / c1) * 100.0

    regime = classify_myers_rajan_regime(
        alpha=alpha,
        alpha_star=alpha_star,
        binding_debt=binding_debt,
        paradox_loss=paradox_loss,
        liquidation_debt=d_liq,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        c1=c1,
        d2=d2,
        alpha=alpha,
        lam=lam,
        theta=theta,
        binding_debt=binding_debt,
        alpha_star=alpha_star,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return MyersRajanReceipt(
        ticker=cleaned_ticker,
        invested_assets_c1=c1,
        continuation_value_d2=d2,
        asset_liquidity_alpha=alpha,
        transformation_friction_lambda=lam,
        insider_rent_share_theta=theta,
        liquidation_debt_capacity=d_liq,
        transformation_debt_ceiling=d_trans,
        binding_debt_capacity=binding_debt,
        optimal_liquidity_alpha_star=alpha_star,
        peak_debt_capacity=peak_debt,
        paradox_capacity_loss=paradox_loss,
        paradox_zone_active=paradox_active,
        bank_monitoring_requirement_pct=bank_req_pct,
        leverage_ratio_pct=lev_pct,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_myers_rajan_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "MR98",
    asset_liquidity_alpha: float = 0.75,
    transformation_friction_lambda: float = 0.25,
) -> MyersRajanReceipt:
    """Calibrate going-concern continuation multiple from trailing market dynamics."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        c1 = float(max(10.0, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        drift_bonus = float(np.clip(np.mean(log_rets) * 252.0, -0.15, 0.60))
        d2 = c1 * float(np.clip(1.20 + 0.5 * drift_bonus, 0.90, 1.80))
    else:
        c1 = 100.0
        d2 = 125.0

    return calculate_myers_rajan_paradox(
        invested_assets_c1=c1,
        continuation_value_d2=d2,
        asset_liquidity_alpha=float(asset_liquidity_alpha),
        transformation_friction_lambda=float(transformation_friction_lambda),
        ticker=ticker,
    )


def persist_myers_rajan_record(
    receipt: MyersRajanReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Myers & Rajan (1998) liquidity paradox record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS myers_rajan_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        invested_assets_c1 REAL NOT NULL,
        continuation_value_d2 REAL NOT NULL,
        asset_liquidity_alpha REAL NOT NULL,
        transformation_friction_lambda REAL NOT NULL,
        liquidation_debt_capacity REAL NOT NULL,
        transformation_debt_ceiling REAL NOT NULL,
        binding_debt_capacity REAL NOT NULL,
        optimal_liquidity_alpha_star REAL NOT NULL,
        paradox_capacity_loss REAL NOT NULL,
        paradox_zone_active INTEGER NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO myers_rajan_audits (
        ticker, invested_assets_c1, continuation_value_d2, asset_liquidity_alpha,
        transformation_friction_lambda, liquidation_debt_capacity, transformation_debt_ceiling,
        binding_debt_capacity, optimal_liquidity_alpha_star, paradox_capacity_loss,
        paradox_zone_active, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.invested_assets_c1,
                receipt.continuation_value_d2,
                receipt.asset_liquidity_alpha,
                receipt.transformation_friction_lambda,
                receipt.liquidation_debt_capacity,
                receipt.transformation_debt_ceiling,
                receipt.binding_debt_capacity,
                receipt.optimal_liquidity_alpha_star,
                receipt.paradox_capacity_loss,
                int(receipt.paradox_zone_active),
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

