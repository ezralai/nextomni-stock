"""Michael J. Brennan (1998) Delegated Institutional Agency & Two-Beta Benchmark Asset Pricing Engine.

Theoretical Foundation:
    Brennan, Michael J. (1998). "Agency and Asset Pricing".
    Journal of Financial Economics, 48(3), pp. 249-278.
    Foundational to modern institutional asset pricing, tracking error constraints,
    and index-inclusion premium modeling.

Key Concepts:
    1. Delegated Portfolio Management:
       Rather than individual investors directly managing assets, institutional fund managers
       control a substantial fraction omega_inst in [0, 1] of aggregate wealth.

    2. Benchmark Tracking Error Penalty:
       Institutional managers are compensated based on performance relative to an exogenous
       benchmark portfolio B (e.g. S&P 500). Their objective function penalizes both absolute
       portfolio variance and tracking error variance:
           U(R_p) = E[R_p] - 0.5 * gamma_abs * Var(R_p) - 0.5 * gamma_rel * Var(R_p - R_B)

    3. Two-Beta Agency General Equilibrium:
       In market clearing equilibrium, expected returns follow a two-beta structure:
           E[R_i] - R_f = lambda_M * beta_{i, M} + lambda_agency * (beta_{i, M} - beta_{i, B})
       where:
           - beta_{i, M} is standard market beta
           - beta_{i, B} is benchmark beta
           - lambda_agency = omega_inst * [gamma_rel / (gamma_abs + gamma_rel)] * Var(R_B) >= 0

    4. Asset Pricing Implications:
       - Benchmark Congestion: Assets with high benchmark sensitivity (beta_{i, B} > beta_{i, M})
         experience elevated institutional demand, driving prices up and depressing expected returns.
       - Out-of-Benchmark Alpha: Assets neglected by the benchmark (beta_{i, B} < beta_{i, M})
         require an agency compensation premium (alpha_agency > 0) to entice unconstrained capital.
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
HMAC_SECRET = b"brennan-agency-pricing-secret-v131"


@dataclass(frozen=True)
class BrennanAgencyReceipt:
    """Immutable quantitative audit receipt for Brennan (1998) agency asset pricing equilibrium."""

    ticker: str
    beta_market: float                 # beta_{i, M}
    beta_benchmark: float              # beta_{i, B}
    institutional_share: float         # omega_inst
    risk_aversion_abs: float           # gamma_abs
    tracking_aversion_rel: float       # gamma_rel
    market_risk_premium: float         # lambda_M
    benchmark_variance: float          # Var(R_B)
    agency_risk_intensity: float       # lambda_agency
    risk_free_rate: float              # R_f
    expected_return_capm: float        # Standard CAPM: R_f + beta_{i,M} * lambda_M
    expected_return_agency: float      # Brennan: R_f + beta_{i,M}*lambda_M + lambda_agency*(beta_{i,M} - beta_{i,B})
    agency_distortion_spread: float    # Brennan - CAPM
    agency_alpha: float                # -agency_distortion_spread
    benchmark_overweight_ratio: float  # beta_{i, B} / beta_{i, M}
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


def calculate_brennan_agency_pricing(
    beta_market: float = 1.15,
    beta_benchmark: float = 1.30,
    institutional_share: float = 0.70,
    risk_aversion_abs: float = 2.50,
    tracking_aversion_rel: float = 5.00,
    market_risk_premium: float = 0.06,
    benchmark_variance: float = 0.04,
    risk_free_rate: float = 0.04,
    ticker: str = "BENCH",
) -> BrennanAgencyReceipt:
    """Calculate the Brennan (1998) two-beta agency asset pricing equilibrium.

    Args:
        beta_market: Sensitivity to total market portfolio beta_{i, M}.
        beta_benchmark: Sensitivity to institutional benchmark portfolio beta_{i, B}.
        institutional_share: Fraction of total wealth managed by institutions omega_inst in [0, 1].
        risk_aversion_abs: Absolute risk aversion gamma_abs (> 0).
        tracking_aversion_rel: Relative benchmark tracking error penalty gamma_rel (>= 0).
        market_risk_premium: Market equity risk premium lambda_M (e.g. 0.06).
        benchmark_variance: Benchmark return variance Var(R_B) (> 0).
        risk_free_rate: Risk-free rate R_f (e.g. 0.04).
        ticker: Asset ticker symbol.

    Returns:
        BrennanAgencyReceipt with agency-adjusted expected returns and alpha spreads.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "BENCH"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    bm = float(beta_market)
    bb = float(beta_benchmark)
    omega = float(np.clip(institutional_share, 0.0, 1.0))
    g_abs = max(1e-4, float(risk_aversion_abs))
    g_rel = max(0.0, float(tracking_aversion_rel))
    lambda_m = float(market_risk_premium)
    var_b = max(1e-6, float(benchmark_variance))
    rf = float(risk_free_rate)

    # Agency risk intensity:
    # lambda_agency = omega_inst * [gamma_rel / (gamma_abs + gamma_rel)] * Var(R_B)
    total_aversion = g_abs + g_rel
    aversion_ratio = g_rel / total_aversion if total_aversion > 0 else 0.0
    lambda_agency = omega * aversion_ratio * var_b

    # Expected Returns
    # Standard CAPM:
    er_capm = rf + bm * lambda_m

    # Brennan Agency Two-Beta Model:
    # E[R_i] = R_f + beta_{i,M} * lambda_M + lambda_agency * (beta_{i,M} - beta_{i,B})
    # If beta_{i,B} > beta_{i,M}: benchmark darling -> high institutional demand -> lower expected return
    agency_spread = lambda_agency * (bm - bb)
    er_agency = er_capm + agency_spread
    agency_alpha = -agency_spread  # positive alpha when out-of-benchmark / neglected

    # Ratio
    overweight_ratio = bb / max(1e-4, abs(bm)) if bm != 0 else 1.0

    regime = classify_brennan_regime(
        benchmark_beta=bb,
        market_beta=bm,
        agency_spread=agency_spread,
        tracking_aversion=g_rel,
        institutional_share=omega,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    signature_payload = (
        f"{cleaned_ticker}:{bm:.4f}:{bb:.4f}:{omega:.4f}:{lambda_agency:.6f}:"
        f"{er_agency:.6f}:{agency_spread:.6f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(signature_payload)

    return BrennanAgencyReceipt(
        ticker=cleaned_ticker,
        beta_market=bm,
        beta_benchmark=bb,
        institutional_share=omega,
        risk_aversion_abs=g_abs,
        tracking_aversion_rel=g_rel,
        market_risk_premium=lambda_m,
        benchmark_variance=var_b,
        agency_risk_intensity=lambda_agency,
        risk_free_rate=rf,
        expected_return_capm=er_capm,
        expected_return_agency=er_agency,
        agency_distortion_spread=agency_spread,
        agency_alpha=agency_alpha,
        benchmark_overweight_ratio=overweight_ratio,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def classify_brennan_regime(
    benchmark_beta: float,
    market_beta: float,
    agency_spread: float,
    tracking_aversion: float,
    institutional_share: float,
) -> str:
    """Classify the Brennan delegated institutional agency regime."""
    if tracking_aversion < 0.10 or institutional_share < 0.05:
        return "UNCONSTRAINED_CAPM_EQUILIBRIUM"
    if agency_spread <= -0.005 and benchmark_beta > market_beta * 1.15:
        # Expected return depressed by at least 50 bps due to institutional benchmark crowding
        return "BENCHMARK_CONGESTION_OVERPRICING"
    if agency_spread >= 0.005 and benchmark_beta < market_beta * 0.85:
        # Expected return elevated due to benchmark neglect / out-of-benchmark alpha
        return "OUT_OF_BENCHMARK_ALPHA_PREMIUM"
    return "BALANCED_INSTITUTIONAL_EXPOSURE"


def evaluate_brennan_agency_profile(
    asset_prices: List[float] | np.ndarray,
    market_prices: Optional[List[float] | np.ndarray] = None,
    benchmark_prices: Optional[List[float] | np.ndarray] = None,
    ticker: str = "ASSET",
    institutional_share: float = 0.70,
    tracking_aversion: float = 5.00,
    risk_free_rate: float = 0.04,
) -> BrennanAgencyReceipt:
    """Empirically evaluate the Brennan agency asset pricing profile from historical time series."""
    p_arr = np.asarray(asset_prices, dtype=float)
    if len(p_arr) < 15:
        return calculate_brennan_agency_pricing(
            beta_market=1.0,
            beta_benchmark=1.0,
            institutional_share=institutional_share,
            tracking_aversion_rel=tracking_aversion,
            risk_free_rate=risk_free_rate,
            ticker=ticker,
        )

    r_asset = np.diff(p_arr) / p_arr[:-1]

    # Setup or generate market/benchmark series if not supplied
    if market_prices is not None and len(market_prices) == len(p_arr):
        m_arr = np.asarray(market_prices, dtype=float)
        r_market = np.diff(m_arr) / m_arr[:-1]
    else:
        # Default synthetic market proxy
        np.random.seed(len(p_arr))
        r_market = r_asset * 0.70 + 0.01 * np.random.randn(len(r_asset))

    if benchmark_prices is not None and len(benchmark_prices) == len(p_arr):
        b_arr = np.asarray(benchmark_prices, dtype=float)
        r_bench = np.diff(b_arr) / b_arr[:-1]
    else:
        # Default benchmark proxy (e.g. S&P 500 closely aligned with market)
        r_bench = r_market * 1.05 + 0.005 * np.random.randn(len(r_market))

    # Regress asset returns on market and benchmark
    var_m = max(1e-6, float(np.var(r_market)))
    var_b = max(1e-6, float(np.var(r_bench)))
    cov_im = float(np.cov(r_asset, r_market)[0, 1])
    cov_ib = float(np.cov(r_asset, r_bench)[0, 1])

    beta_m = float(cov_im / var_m)
    beta_b = float(cov_ib / var_b)

    return calculate_brennan_agency_pricing(
        beta_market=beta_m,
        beta_benchmark=beta_b,
        institutional_share=institutional_share,
        risk_aversion_abs=2.50,
        tracking_aversion_rel=tracking_aversion,
        market_risk_premium=0.06,
        benchmark_variance=var_b * 252.0,  # Annualized variance
        risk_free_rate=risk_free_rate,
        ticker=ticker,
    )


def persist_brennan_agency_record(
    receipt: BrennanAgencyReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Brennan agency record to SQLite database using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS brennan_agency_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        beta_market REAL NOT NULL,
        beta_benchmark REAL NOT NULL,
        institutional_share REAL NOT NULL,
        risk_aversion_abs REAL NOT NULL,
        tracking_aversion_rel REAL NOT NULL,
        market_risk_premium REAL NOT NULL,
        benchmark_variance REAL NOT NULL,
        agency_risk_intensity REAL NOT NULL,
        risk_free_rate REAL NOT NULL,
        expected_return_capm REAL NOT NULL,
        expected_return_agency REAL NOT NULL,
        agency_distortion_spread REAL NOT NULL,
        agency_alpha REAL NOT NULL,
        benchmark_overweight_ratio REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_brennan_ticker_time ON brennan_agency_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO brennan_agency_audits (
        ticker, beta_market, beta_benchmark, institutional_share,
        risk_aversion_abs, tracking_aversion_rel, market_risk_premium,
        benchmark_variance, agency_risk_intensity, risk_free_rate,
        expected_return_capm, expected_return_agency, agency_distortion_spread,
        agency_alpha, benchmark_overweight_ratio, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.beta_market,
                    receipt.beta_benchmark,
                    receipt.institutional_share,
                    receipt.risk_aversion_abs,
                    receipt.tracking_aversion_rel,
                    receipt.market_risk_premium,
                    receipt.benchmark_variance,
                    receipt.agency_risk_intensity,
                    receipt.risk_free_rate,
                    receipt.expected_return_capm,
                    receipt.expected_return_agency,
                    receipt.agency_distortion_spread,
                    receipt.agency_alpha,
                    receipt.benchmark_overweight_ratio,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False
