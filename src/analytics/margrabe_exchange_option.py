"""William Margrabe (1978) Exchange Option & Relative-Performance Switching Engine.

Theoretical Foundation:
    Margrabe, William (1978). "The Value of an Option to Exchange One Asset for Another".
    The Journal of Finance, 33(1), pp. 177-186.

Key Concepts:
    - Numeraire-Invariant Exchange Option:
      Values the option to exchange Asset 2 for Asset 1 at maturity T,
      V(S_1, S_2, T) = max(S_{1,T} - S_{2,T}, 0), yielding exact closed-form price
      S_1*exp(-q_1*T)*N(d_1) - S_2*exp(-q_2*T)*N(d_2) with composite spread volatility
      sigma_rel = sqrt(sigma_1^2 + sigma_2^2 - 2*rho*sigma_1*sigma_2).
    - Interest-Rate Invariance & Euler Homogeneity:
      Independent of risk-free rate r, linear homogeneous of degree one so
      V = S_1 * delta_1 + S_2 * delta_2 exactly holds across all non-degenerate regimes.
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
HMAC_SECRET = b"margrabe-exchange-option-v151-secret"


@dataclass(frozen=True)
class MargrabeExchangeReceipt:
    """Immutable quantitative audit receipt for Margrabe (1978) Exchange Option."""

    asset1_ticker: str
    asset2_ticker: str
    spot_asset1: float
    spot_asset2: float
    vol_asset1: float
    vol_asset2: float
    div_yield_asset1: float
    div_yield_asset2: float
    correlation_rho: float
    maturity_years: float
    composite_volatility: float
    exchange_option_value: float
    intrinsic_forward_bound: float
    time_switching_value: float
    normalized_switch_premium_pct: float
    delta_asset1: float
    delta_asset2: float
    relative_vega: float
    correlation_sensitivity: float
    outperformance_prob: float
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


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def classify_exchange_option_regime(
    spot_ratio: float,
    composite_vol: float,
    rho: float,
) -> str:
    """Classify pair switching regime based on moneyness ratio, dispersion, and correlation."""
    if composite_vol < 0.05 or rho >= 0.90:
        return "TIGHT_COINTEGRATED_SPREAD"
    if spot_ratio >= 1.15:
        return "ASSET1_DOMINANT_EXCHANGE"
    if spot_ratio <= 0.85:
        return "OUT_OF_THE_MONEY_SWITCH"
    if rho <= 0.25 and composite_vol >= 0.30:
        return "HIGH_DISPERSION_SWITCH_OPTIONALITY"
    return "BALANCED_RELATIVE_PERFORMANCE"


def _build_hmac_payload(
    ticker1: str,
    ticker2: str,
    s1: float,
    s2: float,
    vol1: float,
    vol2: float,
    rho: float,
    maturity: float,
    value: float,
    comp_vol: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker1}:{ticker2}:{s1:.4f}:{s2:.4f}:{vol1:.4f}:{vol2:.4f}:"
        f"{rho:.4f}:{maturity:.4f}:{value:.6f}:{comp_vol:.6f}:{regime}:{timestamp}"
    )


def calculate_margrabe_exchange_option(
    spot_asset1: float = 100.0,
    spot_asset2: float = 100.0,
    vol_asset1: float = 0.25,
    vol_asset2: float = 0.20,
    div_yield_asset1: float = 0.01,
    div_yield_asset2: float = 0.015,
    correlation_rho: float = 0.50,
    maturity_years: float = 1.0,
    asset1_ticker: str = "ASSET1",
    asset2_ticker: str = "SPY",
) -> MargrabeExchangeReceipt:
    """Price a European option to exchange Asset 2 for Asset 1 via Margrabe (1978)."""
    cleaned_t1 = str(asset1_ticker).strip().upper()
    cleaned_t2 = str(asset2_ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_t1):
        raise ValueError(f"Invalid asset1_ticker symbol: {asset1_ticker!r}")
    if not TICKER_REGEX.match(cleaned_t2):
        raise ValueError(f"Invalid asset2_ticker symbol: {asset2_ticker!r}")

    s1 = float(spot_asset1)
    s2 = float(spot_asset2)
    v1 = float(vol_asset1)
    v2 = float(vol_asset2)
    q1 = float(div_yield_asset1)
    q2 = float(div_yield_asset2)
    rho = float(correlation_rho)
    t_mat = float(maturity_years)

    if s1 <= 0.0 or s2 <= 0.0:
        raise ValueError("spot_asset1 and spot_asset2 must be strictly positive.")
    if v1 < 0.0 or v2 < 0.0:
        raise ValueError("vol_asset1 and vol_asset2 must be non-negative.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("correlation_rho must satisfy -1.0 <= rho <= 1.0.")
    if t_mat <= 0.0:
        raise ValueError("maturity_years must be strictly positive.")

    comp_var = max(0.0, v1 * v1 + v2 * v2 - 2.0 * rho * v1 * v2)
    comp_vol = math.sqrt(comp_var)

    f1 = s1 * math.exp(-q1 * t_mat)
    f2 = s2 * math.exp(-q2 * t_mat)
    intrinsic_bound = max(0.0, f1 - f2)

    sqrt_t = math.sqrt(t_mat)
    vol_rt = comp_vol * sqrt_t

    if vol_rt < 1e-10:
        option_val = intrinsic_bound
        if f1 > f2:
            d_a1 = math.exp(-q1 * t_mat)
            d_a2 = -math.exp(-q2 * t_mat)
            out_prob = 1.0
        elif f1 < f2:
            d_a1 = 0.0
            d_a2 = 0.0
            out_prob = 0.0
        else:
            d_a1 = 0.5 * math.exp(-q1 * t_mat)
            d_a2 = -0.5 * math.exp(-q2 * t_mat)
            out_prob = 0.5
        rel_vega = 0.0
        corr_sens = 0.0
    else:
        d1 = (math.log(s1 / s2) + (q2 - q1 + 0.5 * comp_var) * t_mat) / vol_rt
        d2 = d1 - vol_rt
        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)

        option_val = max(intrinsic_bound, f1 * nd1 - f2 * nd2)
        d_a1 = math.exp(-q1 * t_mat) * nd1
        d_a2 = -math.exp(-q2 * t_mat) * nd2
        rel_vega = f1 * _norm_pdf(d1) * sqrt_t
        corr_sens = -rel_vega * ((v1 * v2) / comp_vol)
        out_prob = nd2

    time_val = max(0.0, option_val - intrinsic_bound)
    norm_prem_pct = (option_val / max(s1, s2)) * 100.0

    regime = classify_exchange_option_regime(
        spot_ratio=f1 / f2,
        composite_vol=comp_vol,
        rho=rho,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker1=cleaned_t1,
        ticker2=cleaned_t2,
        s1=s1,
        s2=s2,
        vol1=v1,
        vol2=v2,
        rho=rho,
        maturity=t_mat,
        value=option_val,
        comp_vol=comp_vol,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return MargrabeExchangeReceipt(
        asset1_ticker=cleaned_t1,
        asset2_ticker=cleaned_t2,
        spot_asset1=s1,
        spot_asset2=s2,
        vol_asset1=v1,
        vol_asset2=v2,
        div_yield_asset1=q1,
        div_yield_asset2=q2,
        correlation_rho=rho,
        maturity_years=t_mat,
        composite_volatility=comp_vol,
        exchange_option_value=option_val,
        intrinsic_forward_bound=intrinsic_bound,
        time_switching_value=time_val,
        normalized_switch_premium_pct=norm_prem_pct,
        delta_asset1=d_a1,
        delta_asset2=d_a2,
        relative_vega=rel_vega,
        correlation_sensitivity=corr_sens,
        outperformance_prob=out_prob,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_margrabe_exchange_profile(
    prices: List[float] | np.ndarray,
    asset1_ticker: str = "ASSET1",
    asset2_ticker: str = "SPY",
    benchmark_vol: float = 0.18,
    correlation_rho: float = 0.55,
    maturity_years: float = 1.0,
) -> MargrabeExchangeReceipt:
    """Calibrate Asset 1 spot/volatility from price series and price parity exchange option."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        s1 = float(max(1e-4, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        v1 = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.08, 1.50))
    else:
        s1 = 100.0
        v1 = 0.25

    return calculate_margrabe_exchange_option(
        spot_asset1=s1,
        spot_asset2=s1,
        vol_asset1=v1,
        vol_asset2=float(benchmark_vol),
        correlation_rho=float(correlation_rho),
        maturity_years=float(maturity_years),
        asset1_ticker=asset1_ticker,
        asset2_ticker=asset2_ticker,
    )


def persist_margrabe_exchange_record(
    receipt: MargrabeExchangeReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Margrabe (1978) exchange option audit record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS margrabe_exchange_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset1_ticker TEXT NOT NULL,
        asset2_ticker TEXT NOT NULL,
        spot_asset1 REAL NOT NULL,
        spot_asset2 REAL NOT NULL,
        vol_asset1 REAL NOT NULL,
        vol_asset2 REAL NOT NULL,
        correlation_rho REAL NOT NULL,
        maturity_years REAL NOT NULL,
        composite_volatility REAL NOT NULL,
        exchange_option_value REAL NOT NULL,
        delta_asset1 REAL NOT NULL,
        delta_asset2 REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO margrabe_exchange_audits (
        asset1_ticker, asset2_ticker, spot_asset1, spot_asset2,
        vol_asset1, vol_asset2, correlation_rho, maturity_years,
        composite_volatility, exchange_option_value, delta_asset1, delta_asset2,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.asset1_ticker,
                receipt.asset2_ticker,
                receipt.spot_asset1,
                receipt.spot_asset2,
                receipt.vol_asset1,
                receipt.vol_asset2,
                receipt.correlation_rho,
                receipt.maturity_years,
                receipt.composite_volatility,
                receipt.exchange_option_value,
                receipt.delta_asset1,
                receipt.delta_asset2,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

