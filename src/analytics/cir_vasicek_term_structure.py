"""Cox-Ingersoll-Ross (1985) & Vasicek (1977) Affine Term Structure & Bond Pricing Engine.

Theoretical Foundation:
    1. Cox, John C., Ingersoll, Jonathan E., & Ross, Stephen A. (1985). "A Theory of the
       Term Structure of Interest Rates". Econometrica, 53(2), pp. 385-407.
       Derives general-equilibrium square-root mean-reverting short-rate dynamics guaranteeing
       strictly positive yields under the Feller condition (2 * kappa * theta > sigma^2).

    2. Vasicek, Oldrich A. (1977). "An Equilibrium Characterization of the Term Structure".
       Journal of Financial Economics, 5(2), pp. 177-188.
       Ornstein-Uhlenbeck Gaussian short-rate benchmark pricing zero-coupon bonds via
       exact affine exponential factors.

Key Concepts:
    - Affine Bond Pricing:
      P(0, T) = A(T) * exp(-B(T) * r_0), with continuously compounded spot yield
      y(T) = (B(T) * r_0 - ln A(T)) / T and effective duration B(T).
    - Yield Curve Regimes:
      Normal upward-sloping (theta > r_0), recession-warning inverted (r_0 > theta),
      and flat equilibrium curves, with exact long-run asymptotic yields y_inf.
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
ALLOWED_MODELS = ("CIR", "VASICEK")
HMAC_SECRET = b"cir-vasicek-term-structure-v149-secret"


@dataclass(frozen=True)
class AffineTermStructureReceipt:
    """Immutable quantitative audit receipt for CIR (1985) & Vasicek (1977) term structures."""

    ticker: str
    model_type: str                     # "CIR" or "VASICEK"
    short_rate_r0: float                # Current instantaneous rate r_0
    long_run_mean_theta: float          # Risk-neutral mean attractor theta
    mean_reversion_kappa: float         # Mean reversion speed kappa
    rate_volatility_sigma: float        # Volatility parameter sigma
    feller_ratio: float                 # 2 * kappa * theta / sigma^2 (CIR positivity ratio)
    feller_condition_satisfied: bool
    asymptotic_long_yield: float        # y_inf
    bond_price_1y: float
    bond_price_2y: float
    bond_price_5y: float
    bond_price_10y: float
    bond_price_30y: float
    yield_3m: float
    yield_2y: float
    yield_5y: float
    yield_10y: float
    yield_30y: float
    term_slope_10y_2y_bps: float        # (y_10Y - y_2Y) * 10000
    duration_10y: float                 # Affine sensitivity B(10)
    convexity_adjustment_bps: float     # Theta - asymptotic_long_yield (Jensen's effect)
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


def _cir_bond_price(r0: float, tau: float, kappa: float, theta: float, sigma: float) -> tuple[float, float, float]:
    """Compute overflow-safe exact CIR (1985) zero-coupon price P(0, tau), yield y(tau), and duration B(tau)."""
    if tau <= 1e-8:
        return 1.0, r0, 0.0
    if sigma <= 1e-6:
        b_tau = (1.0 - math.exp(-kappa * tau)) / kappa
        ln_a = -theta * (tau - b_tau)
    else:
        gamma = math.sqrt(kappa ** 2 + 2.0 * (sigma ** 2))
        exp_neg_gt = math.exp(max(-700.0, -gamma * tau))
        denom = (gamma + kappa) + (gamma - kappa) * exp_neg_gt
        b_tau = (2.0 * (1.0 - exp_neg_gt)) / denom
        ln_a = (2.0 * kappa * theta / (sigma ** 2)) * (
            math.log(2.0 * gamma) + 0.5 * (kappa - gamma) * tau - math.log(denom)
        )
    ln_p = ln_a - b_tau * r0
    price = math.exp(ln_p)
    spot_yield = -ln_p / tau
    return price, spot_yield, b_tau


def _vasicek_bond_price(r0: float, tau: float, kappa: float, theta: float, sigma: float) -> tuple[float, float, float]:
    """Compute exact Vasicek (1977) zero-coupon price P(0, tau), yield y(tau), and duration B(tau)."""
    if tau <= 1e-8:
        return 1.0, r0, 0.0
    b_tau = (1.0 - math.exp(-kappa * tau)) / kappa
    y_inf = theta - (sigma ** 2) / (2.0 * (kappa ** 2))
    ln_a = y_inf * (b_tau - tau) - ((sigma ** 2) / (4.0 * kappa)) * (b_tau ** 2)
    ln_p = ln_a - b_tau * r0
    price = math.exp(ln_p)
    spot_yield = -ln_p / tau
    return price, spot_yield, b_tau


def classify_term_structure_regime(
    term_slope_bps: float,
    model_type: str,
    feller_ok: bool,
) -> str:
    """Classify the sovereign yield curve & short-rate stability regime."""
    if model_type == "CIR" and not feller_ok:
        return "CIR_FELLER_BOUNDARY_ZERO_RATE_RISK"
    if term_slope_bps <= -25.0:
        return "INVERTED_YIELD_CURVE_RECESSION_SIGNAL"
    if term_slope_bps >= 100.0:
        return "STEEP_EXPANSIONARY_TERM_STRUCTURE"
    if abs(term_slope_bps) < 20.0:
        return "FLAT_TRANSITION_TERM_STRUCTURE"
    return "NORMAL_UPWARD_SLOPING_YIELD_CURVE"


def calculate_affine_term_structure(
    short_rate_r0: float = 0.035,
    long_run_mean_theta: float = 0.050,
    mean_reversion_kappa: float = 0.35,
    rate_volatility_sigma: float = 0.04,
    model_type: str = "CIR",
    ticker: str = "BOND",
) -> AffineTermStructureReceipt:
    """Solve closed-form affine zero-coupon bond prices and spot yield curves under CIR or Vasicek.

    Args:
        short_rate_r0: Current instantaneous short interest rate r_0.
        long_run_mean_theta: Long-run risk-neutral interest rate attractor theta (> 0).
        mean_reversion_kappa: Speed of short-rate mean reversion kappa (> 0).
        rate_volatility_sigma: Short-rate volatility parameter sigma (>= 0).
        model_type: Affine short-rate specification ("CIR" or "VASICEK").
        ticker: Benchmark fixed-income ticker symbol.

    Returns:
        AffineTermStructureReceipt with exact bond prices, spot yields, 10Y-2Y slope, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "BOND"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    m_type = model_type.strip().upper() if model_type else "CIR"
    if m_type not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model_type '{model_type}'. Allowed: {ALLOWED_MODELS}")

    r0 = float(short_rate_r0)
    theta = float(long_run_mean_theta)
    kappa = float(mean_reversion_kappa)
    sigma = max(0.0, float(rate_volatility_sigma))

    if not (math.isfinite(r0) and math.isfinite(theta) and math.isfinite(kappa)):
        raise ValueError("Rate parameters must be finite numbers.")
    if kappa <= 0.0:
        raise ValueError(f"mean_reversion_kappa must be positive, got: {mean_reversion_kappa}")
    if m_type == "CIR" and (r0 < 0.0 or theta <= 0.0):
        raise ValueError("CIR model requires non-negative r0 and positive theta.")

    feller_ratio = (2.0 * kappa * theta) / max(1e-10, sigma ** 2) if sigma > 1e-5 else 100.0
    feller_ok = (2.0 * kappa * theta) > (sigma ** 2)

    pricer = _cir_bond_price if m_type == "CIR" else _vasicek_bond_price

    _, y_3m, _ = pricer(r0, 0.25, kappa, theta, sigma)
    p_1y, _, _ = pricer(r0, 1.0, kappa, theta, sigma)
    p_2y, y_2y, _ = pricer(r0, 2.0, kappa, theta, sigma)
    p_5y, y_5y, _ = pricer(r0, 5.0, kappa, theta, sigma)
    p_10y, y_10y, dur_10y = pricer(r0, 10.0, kappa, theta, sigma)
    p_30y, y_30y, _ = pricer(r0, 30.0, kappa, theta, sigma)

    if m_type == "CIR":
        gamma = math.sqrt(kappa ** 2 + 2.0 * (sigma ** 2))
        y_inf = (2.0 * kappa * theta) / (kappa + gamma)
    else:
        y_inf = theta - (sigma ** 2) / (2.0 * (kappa ** 2))

    slope_bps = (y_10y - y_2y) * 10000.0
    convexity_bps = (theta - y_inf) * 10000.0

    regime = classify_term_structure_regime(
        term_slope_bps=slope_bps,
        model_type=m_type,
        feller_ok=feller_ok,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{m_type}:{r0:.6f}:{theta:.6f}:{kappa:.4f}:{sigma:.4f}:"
        f"{p_10y:.6f}:{y_10y:.6f}:{slope_bps:.2f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return AffineTermStructureReceipt(
        ticker=cleaned_ticker,
        model_type=m_type,
        short_rate_r0=r0,
        long_run_mean_theta=theta,
        mean_reversion_kappa=kappa,
        rate_volatility_sigma=sigma,
        feller_ratio=feller_ratio,
        feller_condition_satisfied=feller_ok,
        asymptotic_long_yield=y_inf,
        bond_price_1y=p_1y,
        bond_price_2y=p_2y,
        bond_price_5y=p_5y,
        bond_price_10y=p_10y,
        bond_price_30y=p_30y,
        yield_3m=y_3m,
        yield_2y=y_2y,
        yield_5y=y_5y,
        yield_10y=y_10y,
        yield_30y=y_30y,
        term_slope_10y_2y_bps=slope_bps,
        duration_10y=dur_10y,
        convexity_adjustment_bps=convexity_bps,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_term_structure_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "BOND",
    short_rate_r0: float = 0.035,
    long_run_mean_theta: float = 0.050,
    model_type: str = "CIR",
) -> AffineTermStructureReceipt:
    """Calibrate short-rate term structure equilibrium profile from asset series and rate parameters."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        ann_vol = float(np.std(log_rets) * math.sqrt(252.0))
        rate_vol = float(np.clip(ann_vol * 0.15, 0.02, 0.12))
    else:
        rate_vol = 0.04

    return calculate_affine_term_structure(
        short_rate_r0=short_rate_r0,
        long_run_mean_theta=long_run_mean_theta,
        mean_reversion_kappa=0.35,
        rate_volatility_sigma=rate_vol,
        model_type=model_type,
        ticker=ticker,
    )


def persist_term_structure_record(
    receipt: AffineTermStructureReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist CIR/Vasicek term structure record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS term_structure_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        model_type TEXT NOT NULL,
        short_rate_r0 REAL NOT NULL,
        long_run_mean_theta REAL NOT NULL,
        mean_reversion_kappa REAL NOT NULL,
        rate_volatility_sigma REAL NOT NULL,
        feller_ratio REAL NOT NULL,
        feller_condition_satisfied INTEGER NOT NULL,
        asymptotic_long_yield REAL NOT NULL,
        bond_price_1y REAL NOT NULL,
        bond_price_2y REAL NOT NULL,
        bond_price_5y REAL NOT NULL,
        bond_price_10y REAL NOT NULL,
        bond_price_30y REAL NOT NULL,
        yield_3m REAL NOT NULL,
        yield_2y REAL NOT NULL,
        yield_5y REAL NOT NULL,
        yield_10y REAL NOT NULL,
        yield_30y REAL NOT NULL,
        term_slope_10y_2y_bps REAL NOT NULL,
        duration_10y REAL NOT NULL,
        convexity_adjustment_bps REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ts_ticker_time ON term_structure_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO term_structure_audits (
        ticker, model_type, short_rate_r0, long_run_mean_theta, mean_reversion_kappa,
        rate_volatility_sigma, feller_ratio, feller_condition_satisfied,
        asymptotic_long_yield, bond_price_1y, bond_price_2y, bond_price_5y,
        bond_price_10y, bond_price_30y, yield_3m, yield_2y, yield_5y,
        yield_10y, yield_30y, term_slope_10y_2y_bps, duration_10y,
        convexity_adjustment_bps, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.model_type,
                    receipt.short_rate_r0,
                    receipt.long_run_mean_theta,
                    receipt.mean_reversion_kappa,
                    receipt.rate_volatility_sigma,
                    receipt.feller_ratio,
                    1 if receipt.feller_condition_satisfied else 0,
                    receipt.asymptotic_long_yield,
                    receipt.bond_price_1y,
                    receipt.bond_price_2y,
                    receipt.bond_price_5y,
                    receipt.bond_price_10y,
                    receipt.bond_price_30y,
                    receipt.yield_3m,
                    receipt.yield_2y,
                    receipt.yield_5y,
                    receipt.yield_10y,
                    receipt.yield_30y,
                    receipt.term_slope_10y_2y_bps,
                    receipt.duration_10y,
                    receipt.convexity_adjustment_bps,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

