"""Robert E. Lucas Jr. (1978) Pure Exchange Tree Equilibrium & Stochastic Discount Factor (SDF) Engine.

Theoretical Foundation:
    Lucas, Robert E., Jr. (1978). "Asset Prices in an Exchange Economy".
    Econometrica, 46(6), pp. 1429-1445 (1995 Nobel Memorial Prize in Economic Sciences).
    Also foundational to Mehra & Prescott (1985) "The Equity Premium: A Puzzle" and
    Hansen & Jagannathan (1991) SDF volatility bounds.

Key Concepts:
    1. Lucas Tree Economy:
       A pure exchange economy where identical infinitely-lived consumers maximize expected
       discounted CRRA utility over an exogenous stochastic dividend stream (fruit) D_t:
           U(C) = (C^(1-gamma) - 1) / (1 - gamma)
       In equilibrium, market clearing requires C_t = D_t.

    2. Stochastic Discount Factor (Pricing Kernel):
       M_{t+1} = beta * (D_{t+1} / D_t)^(-gamma) = beta * g_{t+1}^(-gamma)
       where beta in (0, 1) is the subjective discount factor and gamma > 0 is relative risk aversion.

    3. Closed-Form Price-Dividend Ratio:
       For log-normal dividend growth ln(g) ~ N(mu, sigma^2):
           E[g^(1-gamma)] = exp((1-gamma)*mu + 0.5*(1-gamma)^2 * sigma^2)
           kappa = beta * E[g^(1-gamma)]
           V = P_t / D_t = kappa / (1 - kappa)   (subject to transversality kappa < 1.0)

    4. Asset Pricing Implications:
       - Gross risk-free rate: R_f = 1 / (beta * E[g^(-gamma)]) = beta^(-1) * exp(gamma*mu - 0.5*gamma^2*sigma^2)
       - Expected gross equity return: E[R] = (V + 1) / V * E[g] = beta^(-1) * exp(gamma*mu + 0.5*(2*gamma - gamma^2)*sigma^2)
       - Theoretical Equity Premium: EP = E[R] - R_f
       - Hansen-Jagannathan Sharpe Bound: sigma(M) / E[M] = sqrt(exp(gamma^2 * sigma^2) - 1)
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
HMAC_SECRET = b"lucas-tree-1978-exchange-secret-v1"


@dataclass(frozen=True)
class LucasTreeReceipt:
    """Immutable quantitative audit receipt for Lucas (1978) tree asset pricing equilibrium."""

    ticker: str
    current_dividend: float
    subjective_discount_factor: float  # beta
    relative_risk_aversion: float      # gamma
    dividend_growth_mean: float        # mu
    dividend_growth_vol: float         # sigma
    transversality_ratio: float        # kappa = beta * E[g^(1-gamma)]
    transversality_satisfied: bool
    price_dividend_ratio: float        # V = P / D
    equilibrium_price: float           # P = V * D
    expected_dividend_growth: float    # E[g]
    risk_free_gross_rate: float        # R_f
    expected_equity_gross_return: float  # E[R]
    equity_premium: float              # E[R] - R_f
    hansen_jagannathan_bound: float    # sigma(M) / E[M]
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


def calculate_lucas_tree_equilibrium(
    current_dividend: float = 2.0,
    subjective_discount_factor: float = 0.96,
    relative_risk_aversion: float = 2.5,
    dividend_growth_mean: float = 0.02,
    dividend_growth_vol: float = 0.04,
    ticker: str = "TREE",
) -> LucasTreeReceipt:
    """Calculate the exact closed-form Lucas (1978) pure exchange asset pricing equilibrium.

    Args:
        current_dividend: Current period dividend payout D_t (> 0).
        subjective_discount_factor: Consumer time discount factor beta in (0, 1).
        relative_risk_aversion: CRRA curvature parameter gamma (> 0).
        dividend_growth_mean: Expected log dividend growth mu.
        dividend_growth_vol: Volatility of log dividend growth sigma (> 0).
        ticker: Asset ticker symbol.

    Returns:
        LucasTreeReceipt with analytical equilibrium valuations and Hansen-Jagannathan bounds.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "TREE"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    d_0 = max(1e-4, float(current_dividend))
    beta = float(np.clip(subjective_discount_factor, 0.01, 0.9999))
    gamma = max(1e-4, float(relative_risk_aversion))
    mu = float(dividend_growth_mean)
    sigma = max(1e-4, float(dividend_growth_vol))

    # Expected gross dividend growth E[g] = exp(mu + 0.5 * sigma^2)
    eg = math.exp(mu + 0.5 * (sigma**2))

    # Transversality kernel: E[g^(1-gamma)] = exp((1-gamma)*mu + 0.5 * (1-gamma)^2 * sigma^2)
    log_eg_one_minus_gamma = (1.0 - gamma) * mu + 0.5 * ((1.0 - gamma) ** 2) * (sigma**2)
    eg_one_minus_gamma = math.exp(log_eg_one_minus_gamma)
    kappa = beta * eg_one_minus_gamma

    transversality_satisfied = kappa < 1.0

    if transversality_satisfied:
        price_dividend_ratio = kappa / (1.0 - kappa)
    else:
        # Divergence / bubble regime: clamp to finite ceiling
        price_dividend_ratio = 1000.0

    equilibrium_price = price_dividend_ratio * d_0

    # Risk-free gross rate: R_f = 1 / (beta * E[g^(-gamma)])
    # E[g^(-gamma)] = exp(-gamma * mu + 0.5 * gamma^2 * sigma^2)
    eg_minus_gamma = math.exp(-gamma * mu + 0.5 * (gamma**2) * (sigma**2))
    r_f = 1.0 / (beta * eg_minus_gamma)

    # Expected gross equity return: E[R] = ((V + 1) / V) * E[g]
    if price_dividend_ratio > 0:
        expected_equity_gross_return = ((price_dividend_ratio + 1.0) / price_dividend_ratio) * eg
    else:
        expected_equity_gross_return = eg

    equity_premium = expected_equity_gross_return - r_f

    # Hansen-Jagannathan SDF Sharpe bound: sigma(M) / E[M] = sqrt(exp(gamma^2 * sigma^2) - 1)
    hj_var = math.exp((gamma**2) * (sigma**2)) - 1.0
    hansen_jagannathan_bound = math.sqrt(max(0.0, hj_var))

    regime = classify_lucas_regime(
        transversality_satisfied=transversality_satisfied,
        equity_premium=equity_premium,
        relative_risk_aversion=gamma,
        hansen_jagannathan_bound=hansen_jagannathan_bound,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    signature_payload = (
        f"{cleaned_ticker}:{beta:.6f}:{gamma:.6f}:{mu:.6f}:{sigma:.6f}:"
        f"{price_dividend_ratio:.4f}:{equity_premium:.6f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(signature_payload)

    return LucasTreeReceipt(
        ticker=cleaned_ticker,
        current_dividend=d_0,
        subjective_discount_factor=beta,
        relative_risk_aversion=gamma,
        dividend_growth_mean=mu,
        dividend_growth_vol=sigma,
        transversality_ratio=kappa,
        transversality_satisfied=transversality_satisfied,
        price_dividend_ratio=price_dividend_ratio,
        equilibrium_price=equilibrium_price,
        expected_dividend_growth=eg,
        risk_free_gross_rate=r_f,
        expected_equity_gross_return=expected_equity_gross_return,
        equity_premium=equity_premium,
        hansen_jagannathan_bound=hansen_jagannathan_bound,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def classify_lucas_regime(
    transversality_satisfied: bool,
    equity_premium: float,
    relative_risk_aversion: float,
    hansen_jagannathan_bound: float,
) -> str:
    """Classify the Lucas exchange economy regime."""
    if not transversality_satisfied:
        return "TRANSVERSALITY_VIOLATION_BUBBLE"
    if relative_risk_aversion >= 8.0:
        return "HIGH_RISK_AVERSION_COMPRESSION"
    if relative_risk_aversion <= 4.0 and equity_premium < 0.015:
        # Standard Mehra-Prescott Equity Premium Puzzle: low theoretical premium despite moderate risk aversion
        return "EQUITY_PREMIUM_PUZZLE"
    if hansen_jagannathan_bound >= 0.40:
        return "VOLATILE_PRICING_KERNEL"
    return "LUCAS_STEADY_STATE_EQUILIBRIUM"


def evaluate_lucas_tree_profile(
    prices: List[float] | np.ndarray,
    dividends: Optional[List[float] | np.ndarray] = None,
    ticker: str = "EQUITY",
    beta: float = 0.96,
    gamma: float = 2.5,
) -> LucasTreeReceipt:
    """Empirically evaluate the Lucas tree asset pricing profile from historical price/dividend series."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) < 5:
        return calculate_lucas_tree_equilibrium(
            current_dividend=2.0,
            subjective_discount_factor=beta,
            relative_risk_aversion=gamma,
            dividend_growth_mean=0.02,
            dividend_growth_vol=0.04,
            ticker=ticker,
        )

    if dividends is not None and len(dividends) >= 5:
        d_arr = np.asarray(dividends, dtype=float)
        # Calculate log dividend growth
        d_growth = np.log(np.maximum(1e-4, d_arr[1:] / np.maximum(1e-4, d_arr[:-1])))
        mu = float(np.mean(d_growth))
        sigma = max(0.01, float(np.std(d_growth)))
        current_div = float(d_arr[-1])
    else:
        # Proxy dividend from price returns and historical yield benchmark (~2.5%)
        returns = np.diff(p_arr) / p_arr[:-1]
        log_ret = np.log(np.maximum(0.01, 1.0 + returns))
        mu = float(np.mean(log_ret)) * 0.40  # Dividend growth typically fraction of price return
        sigma = max(0.01, float(np.std(log_ret)) * 0.50)
        current_div = float(p_arr[-1]) * 0.025

    return calculate_lucas_tree_equilibrium(
        current_dividend=current_div,
        subjective_discount_factor=beta,
        relative_risk_aversion=gamma,
        dividend_growth_mean=mu,
        dividend_growth_vol=sigma,
        ticker=ticker,
    )


def persist_lucas_tree_record(
    receipt: LucasTreeReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Lucas tree equilibrium record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS lucas_tree_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        current_dividend REAL NOT NULL,
        subjective_discount_factor REAL NOT NULL,
        relative_risk_aversion REAL NOT NULL,
        dividend_growth_mean REAL NOT NULL,
        dividend_growth_vol REAL NOT NULL,
        transversality_ratio REAL NOT NULL,
        transversality_satisfied INTEGER NOT NULL,
        price_dividend_ratio REAL NOT NULL,
        equilibrium_price REAL NOT NULL,
        expected_dividend_growth REAL NOT NULL,
        risk_free_gross_rate REAL NOT NULL,
        expected_equity_gross_return REAL NOT NULL,
        equity_premium REAL NOT NULL,
        hansen_jagannathan_bound REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_lucas_ticker_time ON lucas_tree_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO lucas_tree_audits (
        ticker, current_dividend, subjective_discount_factor, relative_risk_aversion,
        dividend_growth_mean, dividend_growth_vol, transversality_ratio,
        transversality_satisfied, price_dividend_ratio, equilibrium_price,
        expected_dividend_growth, risk_free_gross_rate, expected_equity_gross_return,
        equity_premium, hansen_jagannathan_bound, regime, hmac_signature, timestamp
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
                    receipt.current_dividend,
                    receipt.subjective_discount_factor,
                    receipt.relative_risk_aversion,
                    receipt.dividend_growth_mean,
                    receipt.dividend_growth_vol,
                    receipt.transversality_ratio,
                    1 if receipt.transversality_satisfied else 0,
                    receipt.price_dividend_ratio,
                    receipt.equilibrium_price,
                    receipt.expected_dividend_growth,
                    receipt.risk_free_gross_rate,
                    receipt.expected_equity_gross_return,
                    receipt.equity_premium,
                    receipt.hansen_jagannathan_bound,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False
