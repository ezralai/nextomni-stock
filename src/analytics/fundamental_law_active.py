"""Richard C. Grinold (1989) & Clarke, de Silva & Thorley (2002) Fundamental Law of Active Management.

Theoretical Foundation:
    Grinold, Richard C. (1989).
    "The Fundamental Law of Active Management".
    Journal of Portfolio Management, 15(3), pp. 30-37.

    Clarke, Roger, de Silva, Harindra, and Thorley, Steven (2002).
    "Portfolio Constraints and the Fundamental Law of Active Management".
    Financial Analysts Journal, 58(5), pp. 48-66.

Key Concepts:
    - Unconstrained & Constrained Information Ratio:
      IR_unconstrained = IC * sqrt(BR),   IR_constrained = TC * IC * sqrt(BR),
      where IC in [-1, 1] is manager signal-return skill per independent decision,
      BR >= 1 is annual decision breadth, and TC in [0, 1] is the Transfer Coefficient
      measuring portfolio-constraint alignment between realized and unconstrained weights.
    - Quadratic Breadth Destruction & Alpha Drag:
      Because IR = IC * sqrt(TC^2 * BR), constraints shrink effective breadth quadratically:
      BR_effective = TC^2 * BR (e.g., TC = 0.50 destroys 75% of effective breadth).
      Ex-ante constrained alpha obeys alpha_active = IR_constrained * omega_TE.
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
HMAC_SECRET = b"fundamental-law-grinold-kahn-v174-secret"


@dataclass(frozen=True)
class FundamentalLawReceipt:
    """Immutable quantitative audit receipt for Grinold-Kahn Fundamental Law."""

    ticker: str
    information_coefficient_ic: float
    breadth_br: float
    transfer_coefficient_tc: float
    tracking_error_te: float
    unconstrained_ir: float
    constrained_ir: float
    effective_breadth: float
    breadth_loss_pct: float
    unconstrained_alpha: float
    constrained_alpha: float
    constraint_alpha_drag: float
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return asdict(self)


def generate_hmac_signature(payload: str, secret: bytes = HMAC_SECRET) -> str:
    """Generate SHA-256 HMAC signature for cryptographic receipt integrity."""
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(
    payload: str,
    signature: str,
    secret: bytes = HMAC_SECRET,
) -> bool:
    """Constant-time verification of HMAC-SHA256 receipt signature."""
    expected = generate_hmac_signature(payload, secret=secret)
    return hmac.compare_digest(expected, signature)


def compute_transfer_coefficient(
    unconstrained_weights: List[float] | np.ndarray,
    constrained_weights: List[float] | np.ndarray,
    covariance: np.ndarray | None = None,
) -> float:
    """Compute Clarke-de Silva-Thorley (2002) portfolio risk-weighted transfer correlation."""
    w_u = np.asarray(unconstrained_weights, dtype=float)
    w_c = np.asarray(constrained_weights, dtype=float)
    if w_u.ndim != 1 or w_c.ndim != 1 or len(w_u) != len(w_c) or len(w_u) < 2:
        raise ValueError("Active weight vectors must be matching 1D arrays of length >= 2.")

    n = len(w_u)
    cov = np.eye(n, dtype=float) if covariance is None else np.asarray(covariance, dtype=float)
    if cov.shape != (n, n):
        raise ValueError(f"Covariance matrix shape must be ({n}, {n}).")

    num = float(w_c @ cov @ w_u)
    var_c = float(w_c @ cov @ w_c)
    var_u = float(w_u @ cov @ w_u)
    if var_c <= 1e-14 or var_u <= 1e-14:
        raise ValueError("Active weight variance must be strictly positive for TC calculation.")

    tc = num / math.sqrt(var_c * var_u)
    return float(max(-1.0, min(1.0, tc)))


def classify_fundamental_law_regime(
    ic: float,
    tc: float,
    constrained_ir: float,
    breadth_loss_pct: float,
) -> str:
    """Classify active portfolio management regime."""
    if ic <= 0.0:
        return "ZERO_SKILL_NOISE_CHURNER"
    if tc < 0.45 or breadth_loss_pct >= 0.80:
        return "CONSTRAINT_STARVED_LONG_ONLY"
    if constrained_ir >= 0.75 and tc >= 0.65:
        return "ELITE_TRANSFER_ALPHA_ENGINE"
    return "BREADTH_SCALED_CORE_STRATEGY"


def _build_hmac_payload(
    ticker: str,
    ic: float,
    breadth: float,
    tc: float,
    te: float,
    constrained_ir: float,
    effective_breadth: float,
    constrained_alpha: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{ic:.6f}:{breadth:.4f}:{tc:.6f}:{te:.6f}:"
        f"{constrained_ir:.6f}:{effective_breadth:.4f}:{constrained_alpha:.6f}:"
        f"{regime}:{timestamp}"
    )


def calculate_fundamental_law(
    information_coefficient_ic: float,
    breadth_br: float = 250.0,
    transfer_coefficient_tc: float = 0.70,
    tracking_error_te: float = 0.04,
    ticker: str = "SPY",
) -> FundamentalLawReceipt:
    """Evaluate Grinold (1989) & Clarke-de Silva-Thorley (2002) active skill metrics."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    ic = float(information_coefficient_ic)
    br = float(breadth_br)
    tc = float(transfer_coefficient_tc)
    te = float(tracking_error_te)

    if not (-1.0 <= ic <= 1.0):
        raise ValueError("Information Coefficient (IC) must be in [-1.0, 1.0].")
    if not (1.0 <= br <= 10000.0):
        raise ValueError("Decision breadth (BR) must be in [1.0, 10000.0].")
    if not (0.0 <= tc <= 1.0):
        raise ValueError("Transfer Coefficient (TC) must be in [0.0, 1.0].")
    if not (0.0 < te <= 1.0):
        raise ValueError("Tracking error (TE) must be in (0.0, 1.0].")

    sqrt_br = math.sqrt(br)
    unconstrained_ir = ic * sqrt_br
    constrained_ir = tc * unconstrained_ir
    eff_breadth = (tc * tc) * br
    loss_pct = 1.0 - (tc * tc)

    unconstrained_alpha = unconstrained_ir * te
    constrained_alpha = constrained_ir * te
    alpha_drag = unconstrained_alpha - constrained_alpha

    regime = classify_fundamental_law_regime(
        ic=ic,
        tc=tc,
        constrained_ir=constrained_ir,
        breadth_loss_pct=loss_pct,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        ic=ic,
        breadth=br,
        tc=tc,
        te=te,
        constrained_ir=constrained_ir,
        effective_breadth=eff_breadth,
        constrained_alpha=constrained_alpha,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return FundamentalLawReceipt(
        ticker=cleaned_ticker,
        information_coefficient_ic=ic,
        breadth_br=br,
        transfer_coefficient_tc=tc,
        tracking_error_te=te,
        unconstrained_ir=unconstrained_ir,
        constrained_ir=constrained_ir,
        effective_breadth=eff_breadth,
        breadth_loss_pct=loss_pct,
        unconstrained_alpha=unconstrained_alpha,
        constrained_alpha=constrained_alpha,
        constraint_alpha_drag=alpha_drag,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_fundamental_law_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SPY",
    breadth_br: float = 250.0,
    transfer_coefficient_tc: float = 0.70,
) -> FundamentalLawReceipt:
    """Calibrate trailing signal persistence IC and tracking error scale from close prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 65) :])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.05, 0.80))
        te_est = float(np.clip(0.35 * ann_vol, 0.01, 0.25))
        if len(log_rets) >= 10 and np.std(log_rets[:-1]) > 1e-12 and np.std(log_rets[1:]) > 1e-12:
            raw_corr = float(np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1])
            ic_est = float(np.clip(0.03 + 0.15 * max(0.0, raw_corr), -0.30, 0.30))
        else:
            ic_est = 0.05
    else:
        te_est = 0.04
        ic_est = 0.05

    return calculate_fundamental_law(
        information_coefficient_ic=ic_est,
        breadth_br=float(breadth_br),
        transfer_coefficient_tc=float(transfer_coefficient_tc),
        tracking_error_te=te_est,
        ticker=ticker,
    )


def persist_fundamental_law_record(
    receipt: FundamentalLawReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Fundamental Law active management audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS fundamental_law_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        information_coefficient_ic REAL NOT NULL,
        breadth_br REAL NOT NULL,
        transfer_coefficient_tc REAL NOT NULL,
        tracking_error_te REAL NOT NULL,
        unconstrained_ir REAL NOT NULL,
        constrained_ir REAL NOT NULL,
        effective_breadth REAL NOT NULL,
        constrained_alpha REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO fundamental_law_audits (
        ticker, information_coefficient_ic, breadth_br, transfer_coefficient_tc,
        tracking_error_te, unconstrained_ir, constrained_ir, effective_breadth,
        constrained_alpha, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.information_coefficient_ic,
                receipt.breadth_br,
                receipt.transfer_coefficient_tc,
                receipt.tracking_error_te,
                receipt.unconstrained_ir,
                receipt.constrained_ir,
                receipt.effective_breadth,
                receipt.constrained_alpha,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

