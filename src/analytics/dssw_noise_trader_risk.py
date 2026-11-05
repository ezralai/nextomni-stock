"""J. Bradford De Long, Andrei Shleifer, Lawrence H. Summers & Robert J. Waldmann (1990) Noise Trader Risk & Limits of Arbitrage Engine.

Theoretical Foundation:
    De Long, J. B., Shleifer, A., Summers, L. H., & Waldmann, R. J. (1990).
    "Noise Trader Risk in Financial Markets".
    Journal of Political Economy, 98(4), 703-738.
    
    Shleifer, A., & Vishny, R. W. (1997). "The Limits of Arbitrage".
    Journal of Finance, 52(1), 35-55.
    
    Barberis, N., & Thaler, R. (2003). "A Survey of Behavioral Finance".
    Handbook of the Economics of Finance, 1, 1053-1128.

Key Insights:
    - Overturning the Friedman (1953) Hypothesis: Milton Friedman asserted that rational
      arbitrageurs always stabilize prices by trading against irrational noise traders,
      driving them out of business. DSSW (1990) proved this is false when arbitrageurs
      have finite horizons.
    - Resale Price Risk: Unpredictable future changes in noise trader sentiment create
      systematic risk that rational arbitrageurs cannot hedge. If sentiment becomes even
      more extreme tomorrow, arbitrageurs who bet against mispricing today face catastrophic
      interim mark-to-market losses.
    - Create-Space Effect: Because noise traders introduce exogenous volatility into asset
      prices (sentiment variance \\sigma_S^2), they lower fundamental equilibrium prices and
      increase the equilibrium expected return.
    - Noise Traders Can Outperform: When average sentiment is bullish (\\mu > 0), noise traders
      hold a disproportionate share of the risky asset, earning higher expected returns
      than rational arbitrageurs by bearing the very risk that they create.
    - DSSW General Equilibrium Price Equation:
        P_t = P^* + \\frac{\\mu}{r} + \\frac{S_t - \\mu}{1 + r} - \\frac{2 \\gamma \\sigma_S^2}{r (1 + r)^2}
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker, SecurityValidationError

HMAC_SECRET = b"nextomni-dssw-noise-v127-secret"

DSSW_REGIMES = (
    "NOISE_TRADER_SENTIMENT_EUPHORIA",
    "NOISE_TRADER_PANIC_DEPRESSION",
    "ARBITRAGE_RISK_DOMINATED",
    "RATIONAL_FUNDAMENTAL_EQUILIBRIUM",
)

DEFAULT_RISK_FREE_RATE = 0.0425         # 4.25% annual risk-free / dividend discount rate (r)
DEFAULT_MEAN_SENTIMENT = 0.0500         # 5.00% average bullish sentiment bias (\mu)
DEFAULT_SENTIMENT_SHOCK = 0.0800        # 8.00% active transitory sentiment realization (S_t)
DEFAULT_SENTIMENT_VARIANCE = 0.0150     # 0.0150 sentiment variance (\sigma_S^2)
DEFAULT_RISK_AVERSION = 1.2500          # Arbitrageur absolute risk aversion (\gamma)


@dataclass(frozen=True)
class DSSWReceipt:
    """Immutable audit receipt for DSSW (1990) noise trader risk and equilibrium evaluation."""

    ticker: str
    fundamental_price: float              # Intrinsic fundamental present value P^*
    dssw_market_price: float              # Resulting DSSW market equilibrium price P_t
    mispricing_dollar_wedge: float        # P_t - P^* in dollars
    mispricing_percentage_pct: float      # (P_t - P^*) / P^* in %
    mean_sentiment_mu: float              # Average sentiment bias \mu
    active_sentiment_shock: float         # Current period sentiment shock S_t
    sentiment_variance_sigma_sq: float    # Volatility/variance of noise trader sentiment \sigma_S^2
    risk_aversion_gamma: float            # Arbitrageur risk aversion coefficient \gamma
    risk_free_rate_pct: float             # Risk-free / discount rate r in %
    persistent_bias_premium_dollar: float # \mu / r in dollars
    transitory_shock_wedge_dollar: float  # (S_t - \mu) / (1 + r) in dollars
    create_space_discount_dollar: float   # 2 * \gamma * \sigma_S^2 / [r * (1 + r)^2] in dollars
    expected_return_differential_pct: float # E[R_N] - E[R_A] (Noise trader return minus Arbitrageur) in %
    is_noise_trader_outperforming: bool   # Whether noise traders earn higher expected return
    regime: str
    is_sentiment_euphoria: bool
    is_panic_depression: bool
    is_arbitrage_risk_dominated: bool
    is_fundamental_equilibrium: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_dssw_audit_digest(
    ticker: str,
    fundamental_price: float,
    dssw_market_price: float,
    mispricing_percentage_pct: float,
    mean_sentiment_mu: float,
    active_sentiment_shock: float,
    sentiment_variance_sigma_sq: float,
    create_space_discount_dollar: float,
    expected_return_differential_pct: float,
    regime: str,
    calculated_at: str,
) -> str:
    """Generate tamper-evident HMAC-SHA256 signature for DSSW noise trader evaluation."""
    payload = (
        f"{ticker}|{fundamental_price:.4f}|{dssw_market_price:.4f}|{mispricing_percentage_pct:.4f}|"
        f"{mean_sentiment_mu:.4f}|{active_sentiment_shock:.4f}|{sentiment_variance_sigma_sq:.4f}|"
        f"{create_space_discount_dollar:.4f}|{expected_return_differential_pct:.4f}|"
        f"{regime}|{calculated_at}"
    )
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_dssw_audit_digest(receipt: DSSWReceipt) -> bool:
    """Verify cryptographic authenticity and tamper-resistance of a DSSWReceipt."""
    expected = generate_dssw_audit_digest(
        ticker=receipt.ticker,
        fundamental_price=receipt.fundamental_price,
        dssw_market_price=receipt.dssw_market_price,
        mispricing_percentage_pct=receipt.mispricing_percentage_pct,
        mean_sentiment_mu=receipt.mean_sentiment_mu,
        active_sentiment_shock=receipt.active_sentiment_shock,
        sentiment_variance_sigma_sq=receipt.sentiment_variance_sigma_sq,
        create_space_discount_dollar=receipt.create_space_discount_dollar,
        expected_return_differential_pct=receipt.expected_return_differential_pct,
        regime=receipt.regime,
        calculated_at=receipt.calculated_at,
    )
    return hmac.compare_digest(receipt.audit_digest, expected)


def classify_dssw_regime(
    mispricing_pct: float,
    create_space_discount: float,
    fundamental_price: float,
    sentiment_shock: float,
) -> str:
    """Classify asset into one of four DSSW noise trader risk regimes."""
    if mispricing_pct > 15.0 and sentiment_shock > 0.0:
        return "NOISE_TRADER_SENTIMENT_EUPHORIA"
    if mispricing_pct < -15.0 and sentiment_shock < 0.0:
        return "NOISE_TRADER_PANIC_DEPRESSION"
    if create_space_discount > (0.10 * fundamental_price):
        return "ARBITRAGE_RISK_DOMINATED"
    return "RATIONAL_FUNDAMENTAL_EQUILIBRIUM"


def calculate_dssw_equilibrium(
    fundamental_price: float = 100.0,
    mean_sentiment: float = DEFAULT_MEAN_SENTIMENT,
    sentiment_shock: float = DEFAULT_SENTIMENT_SHOCK,
    sentiment_variance: float = DEFAULT_SENTIMENT_VARIANCE,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ticker: str = "TICKER",
) -> DSSWReceipt:
    """Solve closed-form overlapping generations equilibrium pricing equation (DSSW 1990).
    
    Args:
        fundamental_price: Fundamental intrinsic value P^* (> 0).
        mean_sentiment: Average bullish sentiment bias \\mu.
        sentiment_shock: Current period sentiment realization S_t.
        sentiment_variance: Unpredictable sentiment variance \\sigma_S^2 (>= 0).
        risk_aversion: Arbitrageur absolute risk aversion coefficient \\gamma (> 0).
        risk_free_rate: Risk-free interest rate r (> 0).
        ticker: Normalized asset ticker symbol.
        
    Returns:
        DSSWReceipt: Complete audited quantitative DSSW equilibrium evaluation.
    """
    assert_finite_number(fundamental_price, "fundamental_price")
    assert_finite_number(mean_sentiment, "mean_sentiment")
    assert_finite_number(sentiment_shock, "sentiment_shock")
    assert_finite_number(sentiment_variance, "sentiment_variance")
    assert_finite_number(risk_aversion, "risk_aversion")
    assert_finite_number(risk_free_rate, "risk_free_rate")

    if fundamental_price <= 0.0:
        raise ValueError(f"Fundamental price must be positive, got {fundamental_price}")
    if sentiment_variance < 0.0:
        raise ValueError(f"Sentiment variance must be non-negative, got {sentiment_variance}")
    if risk_aversion <= 0.0:
        raise ValueError(f"Risk aversion must be positive, got {risk_aversion}")
    if risk_free_rate <= 0.0:
        raise ValueError(f"Risk-free rate must be positive, got {risk_free_rate}")

    p_star = float(fundamental_price)
    mu = float(mean_sentiment)
    st = float(sentiment_shock)
    sigma_sq = float(sentiment_variance)
    gamma = float(risk_aversion)
    r = float(risk_free_rate)

    # 1. Persistent sentiment bias premium (scaled by fundamental price unit)
    persistent_bias = (mu / r) * (p_star * 0.10)

    # 2. Transitory sentiment shock wedge
    transitory_wedge = ((st - mu) / (1.0 + r)) * (p_star * 0.50)

    # 3. Create-space noise trader risk discount: 2 * \gamma * \sigma_S^2 / [r * (1 + r)^2]
    denominator = r * math.pow(1.0 + r, 2)
    create_space_discount = ((2.0 * gamma * sigma_sq) / max(1e-6, denominator)) * (p_star * 0.10)

    # Theoretical DSSW market price with floor protection (P_t >= 0.05 * P^*)
    raw_price = p_star + persistent_bias + transitory_wedge - create_space_discount
    p_market = float(max(0.05 * p_star, raw_price))

    dollar_wedge = float(p_market - p_star)
    mispricing_pct = float((dollar_wedge / p_star) * 100.0)

    # Expected return differential: E[R_N] - E[R_A] = \mu - [2 * \gamma * \sigma_S^2 / (1 + r)^2]
    ret_diff = mu - ((2.0 * gamma * sigma_sq) / math.pow(1.0 + r, 2))
    ret_diff_pct = float(ret_diff * 100.0)
    outperforming = bool(ret_diff_pct > 0.0)

    regime = classify_dssw_regime(
        mispricing_pct=mispricing_pct,
        create_space_discount=create_space_discount,
        fundamental_price=p_star,
        sentiment_shock=st,
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    # Round metrics for audit consistency
    p_star_round = round(p_star, 4)
    p_market_round = round(p_market, 4)
    dollar_wedge_round = round(dollar_wedge, 4)
    mispricing_pct_round = round(mispricing_pct, 2)
    mu_round = round(mu, 4)
    st_round = round(st, 4)
    sigma_sq_round = round(sigma_sq, 4)
    gamma_round = round(gamma, 4)
    rf_round = round(r * 100.0, 2)
    bias_round = round(persistent_bias, 4)
    transitory_round = round(transitory_wedge, 4)
    create_space_round = round(create_space_discount, 4)
    ret_diff_round = round(ret_diff_pct, 2)

    digest = generate_dssw_audit_digest(
        ticker=ticker,
        fundamental_price=p_star_round,
        dssw_market_price=p_market_round,
        mispricing_percentage_pct=mispricing_pct_round,
        mean_sentiment_mu=mu_round,
        active_sentiment_shock=st_round,
        sentiment_variance_sigma_sq=sigma_sq_round,
        create_space_discount_dollar=create_space_round,
        expected_return_differential_pct=ret_diff_round,
        regime=regime,
        calculated_at=now_iso,
    )

    summary = (
        f"DSSW (1990) Noise Trader Assessment: Regime={regime} | "
        f"Fundamental=${p_star_round:.2f} -> Market Price=${p_market_round:.2f} "
        f"(Mispricing={mispricing_pct_round:+.1f}%) | Active Shock S_t={st_round:+.2f} | "
        f"Create-Space Discount=${create_space_round:.2f} | "
        f"Return Differential (Noise - Arbs)={ret_diff_round:+.2f}%"
    )

    return DSSWReceipt(
        ticker=ticker,
        fundamental_price=p_star_round,
        dssw_market_price=p_market_round,
        mispricing_dollar_wedge=dollar_wedge_round,
        mispricing_percentage_pct=mispricing_pct_round,
        mean_sentiment_mu=mu_round,
        active_sentiment_shock=st_round,
        sentiment_variance_sigma_sq=sigma_sq_round,
        risk_aversion_gamma=gamma_round,
        risk_free_rate_pct=rf_round,
        persistent_bias_premium_dollar=bias_round,
        transitory_shock_wedge_dollar=transitory_round,
        create_space_discount_dollar=create_space_round,
        expected_return_differential_pct=ret_diff_round,
        is_noise_trader_outperforming=outperforming,
        regime=regime,
        is_sentiment_euphoria=(regime == "NOISE_TRADER_SENTIMENT_EUPHORIA"),
        is_panic_depression=(regime == "NOISE_TRADER_PANIC_DEPRESSION"),
        is_arbitrage_risk_dominated=(regime == "ARBITRAGE_RISK_DOMINATED"),
        is_fundamental_equilibrium=(regime == "RATIONAL_FUNDAMENTAL_EQUILIBRIUM"),
        summary=summary,
        audit_digest=digest,
        calculated_at=now_iso,
    )


def evaluate_dssw_profile(
    ticker: str,
    prices: Optional[List[float]] = None,
    mean_sentiment_override: Optional[float] = None,
    sentiment_shock_override: Optional[float] = None,
    sentiment_variance_override: Optional[float] = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    db_path: Optional[Union[str, Path]] = None,
) -> DSSWReceipt:
    """Evaluate DSSW profile using historical price observations from database."""
    valid_ticker = validate_ticker(ticker)

    if prices is None or len(prices) < 20:
        try:
            obs = db.get_observations(valid_ticker, db_path=db_path)
            if obs is not None and not obs.empty and "close" in obs.columns:
                closes = [
                    float(x)
                    for x in obs["close"].tolist()
                    if math.isfinite(float(x)) and float(x) > 0
                ]
                if closes and len(closes) >= 15:
                    prices = closes
        except Exception:
            pass

    if prices is None or len(prices) < 15:
        p_star = 100.0
        mu = DEFAULT_MEAN_SENTIMENT if mean_sentiment_override is None else mean_sentiment_override
        st = DEFAULT_SENTIMENT_SHOCK if sentiment_shock_override is None else sentiment_shock_override
        sigma_sq = DEFAULT_SENTIMENT_VARIANCE if sentiment_variance_override is None else sentiment_variance_override
    else:
        p_star = float(np.mean(prices[-20:]))
        latest_price = float(prices[-1])

        # Infer sentiment shock from deviation of latest price from 20-day moving average
        inferred_shock = float(np.clip((latest_price - p_star) / max(1.0, p_star), -0.30, 0.30))
        st = sentiment_shock_override if sentiment_shock_override is not None else inferred_shock

        # Infer sentiment variance from return volatility
        rets = np.diff(np.log(np.array(prices)))
        daily_var = float(np.var(rets)) if len(rets) > 1 else 0.0004
        inferred_sigma_sq = float(np.clip(daily_var * 252.0 * 0.25, 0.001, 0.10))
        sigma_sq = sentiment_variance_override if sentiment_variance_override is not None else inferred_sigma_sq

        # Infer mean bias from 60-day price drift
        drift = float((latest_price - float(prices[0])) / float(prices[0])) if len(prices) >= 2 else 0.05
        inferred_mu = float(np.clip(drift * 0.20, -0.20, 0.20))
        mu = mean_sentiment_override if mean_sentiment_override is not None else inferred_mu

    return calculate_dssw_equilibrium(
        fundamental_price=p_star,
        mean_sentiment=mu,
        sentiment_shock=st,
        sentiment_variance=sigma_sq,
        risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate,
        ticker=valid_ticker,
    )


def persist_dssw_record(
    receipt: DSSWReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Store DSSW evaluation record in SQLite database table."""
    conn = db.get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dssw_noise_trader_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                fundamental_price REAL NOT NULL,
                dssw_market_price REAL NOT NULL,
                mispricing_dollar_wedge REAL NOT NULL,
                mispricing_percentage_pct REAL NOT NULL,
                mean_sentiment_mu REAL NOT NULL,
                active_sentiment_shock REAL NOT NULL,
                sentiment_variance_sigma_sq REAL NOT NULL,
                risk_aversion_gamma REAL NOT NULL,
                risk_free_rate_pct REAL NOT NULL,
                persistent_bias_premium_dollar REAL NOT NULL,
                transitory_shock_wedge_dollar REAL NOT NULL,
                create_space_discount_dollar REAL NOT NULL,
                expected_return_differential_pct REAL NOT NULL,
                is_noise_trader_outperforming INTEGER NOT NULL,
                regime TEXT NOT NULL,
                summary TEXT NOT NULL,
                audit_digest TEXT NOT NULL,
                calculated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO dssw_noise_trader_records (
                ticker, fundamental_price, dssw_market_price, mispricing_dollar_wedge,
                mispricing_percentage_pct, mean_sentiment_mu, active_sentiment_shock,
                sentiment_variance_sigma_sq, risk_aversion_gamma, risk_free_rate_pct,
                persistent_bias_premium_dollar, transitory_shock_wedge_dollar,
                create_space_discount_dollar, expected_return_differential_pct,
                is_noise_trader_outperforming, regime, summary, audit_digest, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.fundamental_price,
                receipt.dssw_market_price,
                receipt.mispricing_dollar_wedge,
                receipt.mispricing_percentage_pct,
                receipt.mean_sentiment_mu,
                receipt.active_sentiment_shock,
                receipt.sentiment_variance_sigma_sq,
                receipt.risk_aversion_gamma,
                receipt.risk_free_rate_pct,
                receipt.persistent_bias_premium_dollar,
                receipt.transitory_shock_wedge_dollar,
                receipt.create_space_discount_dollar,
                receipt.expected_return_differential_pct,
                int(receipt.is_noise_trader_outperforming),
                receipt.regime,
                receipt.summary,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
