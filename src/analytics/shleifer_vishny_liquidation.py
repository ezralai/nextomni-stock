"""Andrei Shleifer & Robert W. Vishny (1992) Liquidation Values & Debt Capacity Engine.

Theoretical Foundation:
    Shleifer, Andrei, & Vishny, Robert W. (1992).
    "Liquidation Values and Debt Capacity: A Market Equilibrium Approach".
    The Journal of Finance, 47(4), pp. 1343-1366.

Key Concepts:
    - Idiosyncratic vs. Industry-Correlated Asset Liquidation:
      Healthy industry peers bid first-best specialist value V_1 during idiosyncratic
      firm distress. When an industry-wide shock hits, peers themselves face leverage
      constraints (low peer_debt_capacity_slack), forcing asset sales to lower-valuation
      generalist outsiders V_2 = V_1 * (1 - delta_spec).
    - Pro-Cyclical Collateral Debt Capacity:
      Higher asset specificity delta_spec and industry shock correlation rho_ind compress
      liquidation recovery P_liq and reduce sustainable ex-ante corporate debt capacity.
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
HMAC_SECRET = b"shleifer-vishny-liquidation-v158-secret"


@dataclass(frozen=True)
class ShleiferVishnyLiquidationReceipt:
    """Immutable quantitative audit receipt for Shleifer & Vishny (1992) liquidation equilibrium."""

    ticker: str
    specialist_asset_value_v1: float
    asset_specificity_discount: float
    peer_debt_capacity_slack: float
    industry_correlation_rho: float
    downturn_prob_q: float
    outside_buyer_value_v2: float
    idiosyncratic_liquidation_price: float
    downturn_liquidation_price: float
    expected_distress_liquidation_price: float
    risky_debt_capacity: float
    collateral_capacity_loss: float
    fire_sale_haircut_pct: float
    debt_capacity_haircut_pct: float
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


def classify_liquidation_regime(
    fire_sale_haircut_pct: float,
    delta_spec: float,
    peer_slack: float,
    rho_ind: float,
) -> str:
    """Classify asset redeployability and industry fire-sale collateral regime."""
    if delta_spec <= 0.05 or peer_slack >= 0.95:
        return "GENERALIST_LIQUID_COLLATERAL_FULL_CAPACITY"
    if fire_sale_haircut_pct >= 25.0 or (rho_ind >= 0.70 and peer_slack <= 0.25 and delta_spec >= 0.35):
        return "SYSTEMIC_INDUSTRY_FIRE_SALE_DISCOUNT"
    if rho_ind <= 0.25:
        return "IDIOSYNCRATIC_PEER_ABSORPTION_PROTECTED"
    if fire_sale_haircut_pct >= 10.0:
        return "PROCYCLICAL_PEER_ILLIQUIDITY_DRAG"
    return "MODERATE_SPECIFICITY_COLLATERAL_REGIME"


def _build_hmac_payload(
    ticker: str,
    v1: float,
    delta_spec: float,
    peer_slack: float,
    rho_ind: float,
    q_prob: float,
    p_liq: float,
    debt_cap: float,
    haircut_pct: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{v1:.4f}:{delta_spec:.4f}:{peer_slack:.4f}:{rho_ind:.4f}:"
        f"{q_prob:.4f}:{p_liq:.4f}:{debt_cap:.4f}:{haircut_pct:.2f}:{regime}:{timestamp}"
    )


def calculate_shleifer_vishny_liquidation(
    specialist_asset_value_v1: float = 100.0,
    asset_specificity_discount: float = 0.40,
    peer_debt_capacity_slack: float = 0.35,
    industry_correlation_rho: float = 0.65,
    downturn_prob_q: float = 0.30,
    ticker: str = "SV92",
) -> ShleiferVishnyLiquidationReceipt:
    """Evaluate Shleifer & Vishny (1992) collateral liquidation price and optimal debt capacity."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    v1 = float(specialist_asset_value_v1)
    delta_spec = float(asset_specificity_discount)
    slack = float(peer_debt_capacity_slack)
    rho = float(industry_correlation_rho)
    q_prob = float(downturn_prob_q)

    if v1 <= 0.0:
        raise ValueError("specialist_asset_value_v1 must be strictly positive.")
    if not (0.0 <= delta_spec < 1.0):
        raise ValueError("asset_specificity_discount must satisfy 0.0 <= delta_spec < 1.0.")
    if not (0.0 <= slack <= 1.0):
        raise ValueError("peer_debt_capacity_slack must satisfy 0.0 <= slack <= 1.0.")
    if not (0.0 <= rho <= 1.0):
        raise ValueError("industry_correlation_rho must satisfy 0.0 <= rho <= 1.0.")
    if not (0.0 < q_prob < 1.0):
        raise ValueError("downturn_prob_q must satisfy 0.0 < q_prob < 1.0.")

    v2 = v1 * (1.0 - delta_spec)
    p_idio = v1
    p_corr = slack * v1 + (1.0 - slack) * v2
    p_liq = (1.0 - rho) * p_idio + rho * p_corr

    risky_debt_cap = (1.0 - q_prob) * v1 + q_prob * p_liq
    coll_loss = max(0.0, v1 - p_liq)
    fire_haircut_pct = (coll_loss / v1) * 100.0
    debt_haircut_pct = (max(0.0, v1 - risky_debt_cap) / v1) * 100.0

    regime = classify_liquidation_regime(
        fire_sale_haircut_pct=fire_haircut_pct,
        delta_spec=delta_spec,
        peer_slack=slack,
        rho_ind=rho,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        v1=v1,
        delta_spec=delta_spec,
        peer_slack=slack,
        rho_ind=rho,
        q_prob=q_prob,
        p_liq=p_liq,
        debt_cap=risky_debt_cap,
        haircut_pct=fire_haircut_pct,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return ShleiferVishnyLiquidationReceipt(
        ticker=cleaned_ticker,
        specialist_asset_value_v1=v1,
        asset_specificity_discount=delta_spec,
        peer_debt_capacity_slack=slack,
        industry_correlation_rho=rho,
        downturn_prob_q=q_prob,
        outside_buyer_value_v2=v2,
        idiosyncratic_liquidation_price=p_idio,
        downturn_liquidation_price=p_corr,
        expected_distress_liquidation_price=p_liq,
        risky_debt_capacity=risky_debt_cap,
        collateral_capacity_loss=coll_loss,
        fire_sale_haircut_pct=fire_haircut_pct,
        debt_capacity_haircut_pct=debt_haircut_pct,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_shleifer_vishny_liquidation_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SV92",
    asset_specificity_discount: float = 0.40,
    industry_correlation_rho: float = 0.65,
) -> ShleiferVishnyLiquidationReceipt:
    """Calibrate peer balance-sheet absorption slack from trailing drawdown and volatility stress."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        v1 = float(max(10.0, p_arr[-1]))
        window = p_arr[-min(len(p_arr), 60):]
        log_rets = np.diff(np.log(np.maximum(1e-6, window)))
        ann_vol = float(np.std(log_rets) * math.sqrt(252.0))
        running_max = np.maximum.accumulate(window)
        max_dd = float(np.max((running_max - window) / np.maximum(1e-6, running_max)))
        # Higher volatility/drawdown erodes peer industry absorption capacity
        peer_slack = float(np.clip(0.80 - 0.8 * max_dd - 0.6 * ann_vol, 0.05, 0.90))
    else:
        v1 = 100.0
        peer_slack = 0.35

    return calculate_shleifer_vishny_liquidation(
        specialist_asset_value_v1=v1,
        asset_specificity_discount=float(asset_specificity_discount),
        peer_debt_capacity_slack=peer_slack,
        industry_correlation_rho=float(industry_correlation_rho),
        ticker=ticker,
    )


def persist_shleifer_vishny_liquidation_record(
    receipt: ShleiferVishnyLiquidationReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Shleifer & Vishny (1992) liquidation record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS shleifer_vishny_liquidation_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        specialist_asset_value_v1 REAL NOT NULL,
        asset_specificity_discount REAL NOT NULL,
        peer_debt_capacity_slack REAL NOT NULL,
        industry_correlation_rho REAL NOT NULL,
        downturn_liquidation_price REAL NOT NULL,
        expected_distress_liquidation_price REAL NOT NULL,
        risky_debt_capacity REAL NOT NULL,
        fire_sale_haircut_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO shleifer_vishny_liquidation_audits (
        ticker, specialist_asset_value_v1, asset_specificity_discount,
        peer_debt_capacity_slack, industry_correlation_rho, downturn_liquidation_price,
        expected_distress_liquidation_price, risky_debt_capacity, fire_sale_haircut_pct,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.specialist_asset_value_v1,
                receipt.asset_specificity_discount,
                receipt.peer_debt_capacity_slack,
                receipt.industry_correlation_rho,
                receipt.downturn_liquidation_price,
                receipt.expected_distress_liquidation_price,
                receipt.risky_debt_capacity,
                receipt.fire_sale_haircut_pct,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

