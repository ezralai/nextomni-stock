"""Cox, Ross & Rubinstein (1979) Binomial Option Pricing & Early-Exercise Engine.

Theoretical Foundation:
    Cox, John C., Ross, Stephen A., & Rubinstein, Mark (1979).
    "Option Pricing: A Simplified Approach". Journal of Financial Economics, 7(3), pp. 229-263.

Key Concepts:
    - Recombining Multiplicative Lattice:
      Up/down factors u = exp(sigma * sqrt(dt)), d = 1 / u, risk-neutral probability
      p = (exp((r - q) * dt) - d) / (u - d) with no-arbitrage condition d < exp((r - q)*dt) < u.
    - Early-Exercise Decomposition:
      Backward induction computes both American (optimal stopping at each node) and
      European values, isolating the Early-Exercise Premium (V_American - V_European),
      optimal early-exercise boundary count, lattice Greeks (Delta, Gamma, Theta), and
      continuous-time Black-Scholes convergence error.
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
ALLOWED_OPTION_TYPES = ("CALL", "PUT")
ALLOWED_EXERCISE_STYLES = ("AMERICAN", "EUROPEAN")
HMAC_SECRET = b"crr-binomial-options-v150-secret"


@dataclass(frozen=True)
class BinomialOptionReceipt:
    """Immutable quantitative audit receipt for CRR (1979) Binomial Option Valuation."""

    ticker: str
    option_type: str                   # "CALL" or "PUT"
    exercise_style: str                # "AMERICAN" or "EUROPEAN"
    spot_price: float
    strike_price: float
    maturity_years: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    steps: int
    up_factor_u: float
    down_factor_d: float
    risk_neutral_prob_p: float
    option_price: float                # Valued under chosen exercise_style
    european_price: float
    american_price: float
    early_exercise_premium: float      # max(0.0, american_price - european_price)
    early_exercise_nodes: int          # Count of tree nodes with optimal early exercise
    black_scholes_benchmark: float
    discretization_error_vs_bsm: float
    delta: float
    gamma: float
    theta_annual: float
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


def _black_scholes_european(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div_yield: float,
    vol: float,
    option_type: str,
) -> float:
    """Continuous-time Black-Scholes-Merton European option price benchmark."""
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc_r = math.exp(-rate * maturity)
    disc_q = math.exp(-div_yield * maturity)

    if option_type == "CALL":
        return float(spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2))
    return float(strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1))


def classify_binomial_option_regime(
    spot: float,
    strike: float,
    early_exercise_premium: float,
    american_price: float,
    option_type: str,
    exercise_style: str,
) -> str:
    """Classify option moneyness and early-exercise regime."""
    prem_ratio = early_exercise_premium / max(1e-6, american_price)
    if exercise_style == "AMERICAN" and prem_ratio >= 0.03:
        return "HIGH_EARLY_EXERCISE_PREMIUM_AMERICAN"
    if option_type == "CALL":
        moneyness = spot / strike
    else:
        moneyness = strike / spot

    if moneyness >= 1.08:
        return "DEEP_IN_THE_MONEY"
    if moneyness <= 0.92:
        return "OUT_OF_THE_MONEY"
    if exercise_style == "AMERICAN" and prem_ratio > 0.002:
        return "ACTIVE_EARLY_EXERCISE_BOUNDARY"
    return "NEAR_ATM_EUROPEAN_PARITY"


def _build_hmac_payload(
    ticker: str,
    opt_type: str,
    style: str,
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    steps: int,
    option_price: float,
    early_prem: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC signing."""
    return (
        f"{ticker}:{opt_type}:{style}:{spot:.4f}:{strike:.4f}:"
        f"{maturity:.4f}:{vol:.4f}:{steps}:{option_price:.6f}:"
        f"{early_prem:.6f}:{regime}:{timestamp}"
    )


def calculate_crr_binomial_option(
    spot_price: float = 100.0,
    strike_price: float = 100.0,
    maturity_years: float = 1.0,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    volatility: float = 0.25,
    steps: int = 150,
    option_type: str = "PUT",
    exercise_style: str = "AMERICAN",
    ticker: str = "OPT",
) -> BinomialOptionReceipt:
    """Price an American or European option using Cox-Ross-Rubinstein (1979) backward induction."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    opt_type = str(option_type).strip().upper()
    if opt_type not in ALLOWED_OPTION_TYPES:
        raise ValueError(f"Unsupported option_type: {option_type!r}. Allowed: {ALLOWED_OPTION_TYPES}")

    style = str(exercise_style).strip().upper()
    if style not in ALLOWED_EXERCISE_STYLES:
        raise ValueError(f"Unsupported exercise_style: {exercise_style!r}. Allowed: {ALLOWED_EXERCISE_STYLES}")

    spot = float(spot_price)
    strike = float(strike_price)
    maturity = float(maturity_years)
    rate = float(risk_free_rate)
    div = float(dividend_yield)
    vol = float(volatility)
    n_steps = int(steps)

    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot_price and strike_price must be strictly positive.")
    if maturity <= 0.0:
        raise ValueError("maturity_years must be strictly positive.")
    if vol <= 0.0:
        raise ValueError("volatility must be strictly positive.")
    if not (10 <= n_steps <= 1000):
        raise ValueError("steps must be between 10 and 1000.")

    dt = maturity / n_steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((rate - div) * dt)
    discount = math.exp(-rate * dt)

    if not (d < growth < u):
        raise ValueError(
            f"No-arbitrage lattice condition violated: d={d:.6f} < exp((r-q)dt)={growth:.6f} < u={u:.6f}. "
            "Increase steps or volatility."
        )

    p = (growth - d) / (u - d)

    # Terminal payoff at step N
    j_terminal = np.arange(n_steps + 1, dtype=float)
    s_terminal = spot * (u ** j_terminal) * (d ** (n_steps - j_terminal))
    if opt_type == "CALL":
        euro_vals = np.maximum(s_terminal - strike, 0.0)
    else:
        euro_vals = np.maximum(strike - s_terminal, 0.0)
    amer_vals = euro_vals.copy()

    early_nodes = 0
    step1_vals: np.ndarray | None = None
    step2_vals: np.ndarray | None = None

    for i in range(n_steps - 1, -1, -1):
        euro_vals = discount * (p * euro_vals[1:] + (1.0 - p) * euro_vals[:-1])
        amer_cont = discount * (p * amer_vals[1:] + (1.0 - p) * amer_vals[:-1])

        j_idx = np.arange(i + 1, dtype=float)
        s_level = spot * (u ** j_idx) * (d ** (i - j_idx))
        if opt_type == "CALL":
            intrinsic = np.maximum(s_level - strike, 0.0)
        else:
            intrinsic = np.maximum(strike - s_level, 0.0)

        exercised = intrinsic > (amer_cont + 1e-10)
        early_nodes += int(np.sum(exercised))
        amer_vals = np.maximum(amer_cont, intrinsic)

        selected_slice = amer_vals if style == "AMERICAN" else euro_vals
        if i == 2:
            step2_vals = selected_slice.copy()
        elif i == 1:
            step1_vals = selected_slice.copy()

    euro_price = float(euro_vals[0])
    amer_price = float(amer_vals[0])
    early_prem = max(0.0, amer_price - euro_price)
    if early_prem < 1e-10:
        early_prem = 0.0

    selected_price = amer_price if style == "AMERICAN" else euro_price

    # Lattice finite-difference Greeks
    assert step1_vals is not None and step2_vals is not None
    s1_up, s1_dn = spot * u, spot * d
    delta = float((step1_vals[1] - step1_vals[0]) / (s1_up - s1_dn))

    s2_uu, s2_mid, s2_dd = spot * (u ** 2), spot, spot * (d ** 2)
    delta_up = (step2_vals[2] - step2_vals[1]) / (s2_uu - s2_mid)
    delta_dn = (step2_vals[1] - step2_vals[0]) / (s2_mid - s2_dd)
    gamma = float((delta_up - delta_dn) / (0.5 * (s2_uu - s2_dd)))
    theta = float((step2_vals[1] - selected_price) / (2.0 * dt))

    bsm_price = _black_scholes_european(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        div_yield=div,
        vol=vol,
        option_type=opt_type,
    )
    disc_error = abs(euro_price - bsm_price)

    regime = classify_binomial_option_regime(
        spot=spot,
        strike=strike,
        early_exercise_premium=early_prem,
        american_price=amer_price,
        option_type=opt_type,
        exercise_style=style,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        opt_type=opt_type,
        style=style,
        spot=spot,
        strike=strike,
        maturity=maturity,
        vol=vol,
        steps=n_steps,
        option_price=selected_price,
        early_prem=early_prem,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return BinomialOptionReceipt(
        ticker=cleaned_ticker,
        option_type=opt_type,
        exercise_style=style,
        spot_price=spot,
        strike_price=strike,
        maturity_years=maturity,
        risk_free_rate=rate,
        dividend_yield=div,
        volatility=vol,
        steps=n_steps,
        up_factor_u=u,
        down_factor_d=d,
        risk_neutral_prob_p=p,
        option_price=selected_price,
        european_price=euro_price,
        american_price=amer_price,
        early_exercise_premium=early_prem,
        early_exercise_nodes=early_nodes,
        black_scholes_benchmark=bsm_price,
        discretization_error_vs_bsm=disc_error,
        delta=delta,
        gamma=gamma,
        theta_annual=theta,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_binomial_option_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "OPT",
    strike_price: float | None = None,
    maturity_years: float = 0.5,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.015,
    steps: int = 120,
    option_type: str = "PUT",
    exercise_style: str = "AMERICAN",
) -> BinomialOptionReceipt:
    """Calibrate spot and annualized realized volatility from price series and run CRR (1979) lattice."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        spot = float(max(1e-4, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.08, 1.50))
    else:
        spot = 100.0
        ann_vol = 0.25

    eff_strike = float(strike_price) if strike_price is not None else spot
    return calculate_crr_binomial_option(
        spot_price=spot,
        strike_price=eff_strike,
        maturity_years=maturity_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=ann_vol,
        steps=steps,
        option_type=option_type,
        exercise_style=exercise_style,
        ticker=ticker,
    )


def persist_binomial_option_record(
    receipt: BinomialOptionReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist CRR (1979) binomial option valuation record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS binomial_option_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        option_type TEXT NOT NULL,
        exercise_style TEXT NOT NULL,
        spot_price REAL NOT NULL,
        strike_price REAL NOT NULL,
        maturity_years REAL NOT NULL,
        risk_free_rate REAL NOT NULL,
        dividend_yield REAL NOT NULL,
        volatility REAL NOT NULL,
        steps INTEGER NOT NULL,
        option_price REAL NOT NULL,
        european_price REAL NOT NULL,
        american_price REAL NOT NULL,
        early_exercise_premium REAL NOT NULL,
        delta REAL NOT NULL,
        gamma REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO binomial_option_audits (
        ticker, option_type, exercise_style, spot_price, strike_price,
        maturity_years, risk_free_rate, dividend_yield, volatility, steps,
        option_price, european_price, american_price, early_exercise_premium,
        delta, gamma, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.option_type,
                receipt.exercise_style,
                receipt.spot_price,
                receipt.strike_price,
                receipt.maturity_years,
                receipt.risk_free_rate,
                receipt.dividend_yield,
                receipt.volatility,
                receipt.steps,
                receipt.option_price,
                receipt.european_price,
                receipt.american_price,
                receipt.early_exercise_premium,
                receipt.delta,
                receipt.gamma,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

