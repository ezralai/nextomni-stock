"""Andrei Shleifer & Robert W. Vishny (1997) Performance-Based Arbitrage & Limits of Arbitrage Engine.

Theoretical Foundation:
    Shleifer, Andrei, & Vishny, Robert W. (1997). "The Limits of Arbitrage".
    Journal of Finance, 52(1), pp. 35-55.
    Foundational to modern behavioral finance, specialized capital constraints,
    hedge fund redemption risk, and fire-sale spirals.

Key Concepts:
    1. Performance-Based Arbitrage (PBA):
       Real-world arbitrage is conducted by specialized professional money managers using
       other people's money. Outside investors cannot distinguish between managerial
       incompetence and temporarily widening mispricing, withdrawing funds after interim losses:
           F_1 = F_0 * (1 + a * R_1)
       where a >= 0 is investor flow-performance sensitivity.

    2. Interim Widening & Forced Liquidation:
       When an underpriced asset's price temporarily drops further at t=1 (P_1 < P_0 < V),
       the arbitrageur incurs mark-to-market losses. Outside redemptions exceed cash reserves,
       forcing the arbitrageur to liquidate positions at distressed fire-sale prices precisely
       when expected future returns are highest.

    3. Equilibrium Limits of Arbitrage:
       Anticipating t=1 redemption and liquidation risk, rational risk-averse arbitrageurs
       voluntarily restrict initial trade size at t=0, leaving substantial equilibrium
       mispricing uncorrected:
           Delta_PBA = V - P_0 > 0
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"shleifer-vishny-arbitrage-secret-v132"


@dataclass(frozen=True)
class ShleiferVishnyReceipt:
    """Immutable quantitative audit receipt for Shleifer-Vishny (1997) limits of arbitrage equilibrium."""

    ticker: str
    fundamental_value: float          # V
    initial_price: float              # P_0
    adverse_shock_price: float        # P_1
    initial_mispricing_dollar: float  # V - P_0
    initial_mispricing_pct: float     # (V - P_0) / V
    flow_sensitivity: float           # a
    adverse_shock_prob: float         # q
    initial_capital: float            # F_0
    optimal_initial_position: float   # x_0*
    capital_deployed: float           # x_0* * P_0
    cash_buffer: float                # C_0 = F_0 - x_0* * P_0
    interim_loss_pct: float           # R_1 = x_0*(P_1 - P_0) / F_0
    redemption_amount: float          # F_0 - F_1
    forced_liquidation_units: float   # Delta x_1
    forced_liquidation_pct: float     # Delta x_1 / x_0*
    residual_mispricing_ratio: float  # Delta_PBA / V
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


def calculate_shleifer_vishny_equilibrium(
    fundamental_value: float = 100.0,
    initial_price: float = 90.0,
    adverse_shock_price: float = 80.0,
    flow_sensitivity: float = 1.20,
    adverse_shock_prob: float = 0.30,
    initial_capital: float = 1000.0,
    ticker: str = "ARB",
) -> ShleiferVishnyReceipt:
    """Calculate the Shleifer & Vishny (1997) Performance-Based Arbitrage equilibrium.

    Args:
        fundamental_value: Terminal liquidation value V (> 0).
        initial_price: Initial market price P_0 (0 < P_0 <= V).
        adverse_shock_price: Depressed interim price P_1 (0 < P_1 <= P_0).
        flow_sensitivity: Investor redemption sensitivity a (>= 0).
        adverse_shock_prob: Probability q of interim widening shock in [0, 1].
        initial_capital: Arbitrageur initial equity capital F_0 (> 0).
        ticker: Asset ticker symbol.

    Returns:
        ShleiferVishnyReceipt with optimal position, liquidation volume, and residual mispricing.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "ARB"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    v = max(1e-4, float(fundamental_value))
    p0 = float(np.clip(initial_price, 1e-4, v))
    p1 = float(np.clip(adverse_shock_price, 1e-4, p0))
    a = max(0.0, float(flow_sensitivity))
    q = float(np.clip(adverse_shock_prob, 0.0, 1.0))
    f0 = max(1.0, float(initial_capital))

    # Initial mispricing
    initial_mispricing = v - p0
    initial_mispricing_pct = initial_mispricing / v

    # Max possible positions if 100% capital deployed
    max_position = f0 / p0

    # In Shleifer-Vishny (1997), the arbitrageur anticipates that if an adverse shock occurs:
    # Loss per share = (p0 - p1)
    # Total loss = x0 * (p0 - p1)
    # Outside investor redemptions = a * x0 * (p0 - p1)
    # Available cash buffer = F_0 - x0 * p0
    # To avoid fire-sale liquidation or insolvency, the safe allocation requires:
    # a * x0 * (p0 - p1) <= F_0 - x0 * p0
    # => x0 * [p0 + a * (p0 - p1)] <= F_0
    # => x_safe = F_0 / [p0 + a * (p0 - p1)]
    shock_depth = p0 - p1
    effective_capital_requirement = p0 + a * shock_depth

    if a == 0.0 or shock_depth == 0.0:
        # Frictionless / proprietary arbitrage: full deployment
        x0_star = max_position
    else:
        # Optimal position balances upside (v - p0) against risk of distress liquidation
        # Under risk aversion and probability of shock q:
        x_safe = f0 / effective_capital_requirement
        # If expected recovery is very high relative to shock, position is intermediate between x_safe and max_position
        risk_weight = q * a
        x0_star = float(np.clip(
            max_position * (1.0 / (1.0 + risk_weight * (shock_depth / p0))),
            0.0,
            max_position
        ))

    capital_deployed = x0_star * p0
    cash_buffer = max(0.0, f0 - capital_deployed)

    # Interim adverse shock consequences at t=1:
    interim_loss = x0_star * (p1 - p0)
    interim_loss_pct = interim_loss / f0

    # Redemption demand:
    redemption_amount = max(0.0, a * abs(interim_loss))

    # Forced liquidation:
    # If redemptions exceed cash buffer, forced liquidation at price p1:
    unmet_redemptions = max(0.0, redemption_amount - cash_buffer)
    forced_liquidation_units = float(np.clip(unmet_redemptions / p1, 0.0, x0_star))
    forced_liquidation_pct = forced_liquidation_units / max(1e-6, x0_star)

    # Residual mispricing ratio:
    residual_mispricing = initial_mispricing_pct

    regime = classify_shleifer_vishny_regime(
        forced_liquidation_pct=forced_liquidation_pct,
        residual_mispricing_pct=initial_mispricing_pct,
        flow_sensitivity=a,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    signature_payload = (
        f"{cleaned_ticker}:{v:.4f}:{p0:.4f}:{p1:.4f}:{a:.4f}:{q:.4f}:"
        f"{x0_star:.4f}:{forced_liquidation_pct:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(signature_payload)

    return ShleiferVishnyReceipt(
        ticker=cleaned_ticker,
        fundamental_value=v,
        initial_price=p0,
        adverse_shock_price=p1,
        initial_mispricing_dollar=initial_mispricing,
        initial_mispricing_pct=initial_mispricing_pct,
        flow_sensitivity=a,
        adverse_shock_prob=q,
        initial_capital=f0,
        optimal_initial_position=x0_star,
        capital_deployed=capital_deployed,
        cash_buffer=cash_buffer,
        interim_loss_pct=interim_loss_pct,
        redemption_amount=redemption_amount,
        forced_liquidation_units=forced_liquidation_units,
        forced_liquidation_pct=forced_liquidation_pct,
        residual_mispricing_ratio=residual_mispricing,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def classify_shleifer_vishny_regime(
    forced_liquidation_pct: float,
    residual_mispricing_pct: float,
    flow_sensitivity: float,
) -> str:
    """Classify the Shleifer-Vishny limits of arbitrage regime."""
    if flow_sensitivity < 0.10:
        return "FRICTIONLESS_PROPRIETARY_ARBITRAGE"
    if forced_liquidation_pct >= 0.15:
        return "PERFORMANCE_BASED_FIRE_SALE"
    if residual_mispricing_pct >= 0.10:
        return "CAPITAL_CONSTRAINED_ARBITRAGE"
    return "WELL_CAPITALIZED_CONVERGENCE"


def evaluate_shleifer_vishny_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "ASSET",
    flow_sensitivity: float = 1.20,
    initial_capital: float = 1000.0,
) -> ShleiferVishnyReceipt:
    """Empirically evaluate the Shleifer-Vishny profile from historical price history."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) < 10:
        return calculate_shleifer_vishny_equilibrium(
            fundamental_value=100.0,
            initial_price=90.0,
            adverse_shock_price=80.0,
            flow_sensitivity=flow_sensitivity,
            initial_capital=initial_capital,
            ticker=ticker,
        )

    # Estimate fundamental value proxy from peak / moving average
    current_price = float(p_arr[-1])
    p_max = float(np.max(p_arr[-60:] if len(p_arr) >= 60 else p_arr))
    fundamental_val = max(current_price, p_max * 1.05)

    # Estimate adverse shock depth from historical downside volatility
    returns = np.diff(p_arr) / p_arr[:-1]
    downside_vol = float(np.std(returns[returns < 0])) if np.any(returns < 0) else 0.02
    adverse_price = max(1.0, current_price * (1.0 - downside_vol * 2.0))

    return calculate_shleifer_vishny_equilibrium(
        fundamental_value=fundamental_val,
        initial_price=current_price,
        adverse_shock_price=adverse_price,
        flow_sensitivity=flow_sensitivity,
        adverse_shock_prob=0.35,
        initial_capital=initial_capital,
        ticker=ticker,
    )


def persist_shleifer_vishny_record(
    receipt: ShleiferVishnyReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Shleifer-Vishny audit record to SQLite database using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS shleifer_vishny_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        fundamental_value REAL NOT NULL,
        initial_price REAL NOT NULL,
        adverse_shock_price REAL NOT NULL,
        initial_mispricing_dollar REAL NOT NULL,
        initial_mispricing_pct REAL NOT NULL,
        flow_sensitivity REAL NOT NULL,
        adverse_shock_prob REAL NOT NULL,
        initial_capital REAL NOT NULL,
        optimal_initial_position REAL NOT NULL,
        capital_deployed REAL NOT NULL,
        cash_buffer REAL NOT NULL,
        interim_loss_pct REAL NOT NULL,
        redemption_amount REAL NOT NULL,
        forced_liquidation_units REAL NOT NULL,
        forced_liquidation_pct REAL NOT NULL,
        residual_mispricing_ratio REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sv_ticker_time ON shleifer_vishny_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO shleifer_vishny_audits (
        ticker, fundamental_value, initial_price, adverse_shock_price,
        initial_mispricing_dollar, initial_mispricing_pct, flow_sensitivity,
        adverse_shock_prob, initial_capital, optimal_initial_position,
        capital_deployed, cash_buffer, interim_loss_pct, redemption_amount,
        forced_liquidation_units, forced_liquidation_pct, residual_mispricing_ratio,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.fundamental_value,
                    receipt.initial_price,
                    receipt.adverse_shock_price,
                    receipt.initial_mispricing_dollar,
                    receipt.initial_mispricing_pct,
                    receipt.flow_sensitivity,
                    receipt.adverse_shock_prob,
                    receipt.initial_capital,
                    receipt.optimal_initial_position,
                    receipt.capital_deployed,
                    receipt.cash_buffer,
                    receipt.interim_loss_pct,
                    receipt.redemption_amount,
                    receipt.forced_liquidation_units,
                    receipt.forced_liquidation_pct,
                    receipt.residual_mispricing_ratio,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False
