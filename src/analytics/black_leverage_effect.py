"""Fischer Black (1976) & Campbell-Hentschel (1992) Financial Leverage Effect & Volatility Asymmetry Engine.

Theoretical Foundation:
    Black, F. (1976). "Studies of Stock Price Volatility Changes".
    Proceedings of the 1976 Meetings of the American Finance Association, 177-181.
    
    Campbell, J. Y., & Hentschel, L. (1992). "No News is Good News: An Asymmetric Model of Changing Volatility in Stock Returns".
    Journal of Financial Economics, 31(3), 281-318.
    
    Christie, A. A. (1982). "The Stochastic Behavior of Common Stock Variances: Value, Leverage and Interest Rate Effects".
    Journal of Financial Economics, 10(4), 407-432.

Key Insights:
    - The Financial Leverage Effect (Black 1976): As market capitalization drops, fixed financial debt D
      becomes a larger fraction of total firm enterprise value (D/E rises). This mechanical shift in capital
      structure amplifies equity beta and equity volatility:
        \\sigma_E(P) = \\sigma_A \\cdot (1 + D / P)
      Where \\sigma_A is intrinsic unlevered asset volatility and P is stock price.
    - Negative Volatility-Price Derivative:
        d\\sigma_E / dP = -\\sigma_A \\cdot (D / P^2) < 0
      Stock price declines directly induce mechanical volatility increases.
    - Volatility Feedback Hypothesis (Campbell & Hentschel 1992): If volatility is priced, an anticipation
      of higher volatility increases required future returns, triggering an immediate drop in current stock price,
      generating downside return asymmetry and negative skewness.
    - Volatility Asymmetry Ratio (VAR):
        VAR = \\sigma(r_t | r_t < 0) / \\sigma(r_t | r_t > 0)
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

HMAC_SECRET = b"nextomni-black-leverage-v128-secret"

BLACK_LEVERAGE_REGIMES = (
    "ACUTE_LEVERAGE_AMPLIFICATION",
    "VOLATILITY_FEEDBACK_CASCADE",
    "UNLEVERED_INSULATION",
    "MILD_ASYMMETRY_STABILITY",
)

DEFAULT_DEBT_PER_SHARE = 25.0           # $25.00 debt per share (D)
DEFAULT_ASSET_VOLATILITY = 0.1800       # 18.00% annual unlevered asset volatility (\sigma_A)
DEFAULT_FEEDBACK_ELASTICITY = 0.2500    # Volatility feedback discount elasticity (\theta)


@dataclass(frozen=True)
class BlackLeverageReceipt:
    """Immutable audit receipt for Black (1976) & Campbell-Hentschel (1992) leverage effect evaluation."""

    ticker: str
    market_price: float                   # Current stock market price P
    debt_per_share: float                 # Financial debt per share D
    debt_to_equity_ratio: float           # D / P ratio
    asset_volatility_pct: float           # Unlevered asset volatility \sigma_A in %
    equity_volatility_pct: float          # Leverage-amplified equity volatility \sigma_E(P) in %
    leverage_volatility_multiplier: float # \sigma_E / \sigma_A = 1 + D/P
    volatility_price_sensitivity: float   # d\sigma_E / dP (negative)
    realized_upside_volatility_pct: float # \sigma^+ in %
    realized_downside_volatility_pct: float # \sigma^- in %
    volatility_asymmetry_ratio: float     # VAR = \sigma^- / \sigma^+
    volatility_feedback_discount_dollar: float # Price discount from volatility feedback in $
    volatility_feedback_discount_pct: float # Feedback discount in % of P
    feedback_adjusted_price: float        # P - \Delta P_{feedback} in $
    regime: str
    is_acute_leverage_amplification: bool
    is_volatility_feedback_cascade: bool
    is_unlevered_insulation: bool
    is_mild_asymmetry_stability: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_black_leverage_audit_digest(
    ticker: str,
    market_price: float,
    debt_per_share: float,
    debt_to_equity_ratio: float,
    asset_volatility_pct: float,
    equity_volatility_pct: float,
    volatility_asymmetry_ratio: float,
    volatility_feedback_discount_pct: float,
    feedback_adjusted_price: float,
    regime: str,
    calculated_at: str,
) -> str:
    """Generate tamper-evident HMAC-SHA256 signature for Black leverage evaluation."""
    payload = (
        f"{ticker}|{market_price:.4f}|{debt_per_share:.4f}|{debt_to_equity_ratio:.4f}|"
        f"{asset_volatility_pct:.4f}|{equity_volatility_pct:.4f}|{volatility_asymmetry_ratio:.4f}|"
        f"{volatility_feedback_discount_pct:.4f}|{feedback_adjusted_price:.4f}|"
        f"{regime}|{calculated_at}"
    )
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_black_leverage_audit_digest(receipt: BlackLeverageReceipt) -> bool:
    """Verify cryptographic authenticity and tamper-resistance of a BlackLeverageReceipt."""
    expected = generate_black_leverage_audit_digest(
        ticker=receipt.ticker,
        market_price=receipt.market_price,
        debt_per_share=receipt.debt_per_share,
        debt_to_equity_ratio=receipt.debt_to_equity_ratio,
        asset_volatility_pct=receipt.asset_volatility_pct,
        equity_volatility_pct=receipt.equity_volatility_pct,
        volatility_asymmetry_ratio=receipt.volatility_asymmetry_ratio,
        volatility_feedback_discount_pct=receipt.volatility_feedback_discount_pct,
        feedback_adjusted_price=receipt.feedback_adjusted_price,
        regime=receipt.regime,
        calculated_at=receipt.calculated_at,
    )
    return hmac.compare_digest(receipt.audit_digest, expected)


def classify_leverage_effect_regime(
    de_ratio: float,
    vol_multiplier: float,
    var: float,
    feedback_pct: float,
) -> str:
    """Classify the asset into one of four Black-Campbell-Hentschel regimes."""
    if de_ratio > 1.50 and vol_multiplier > 2.50:
        return "ACUTE_LEVERAGE_AMPLIFICATION"
    if var > 1.35 and feedback_pct > 8.0:
        return "VOLATILITY_FEEDBACK_CASCADE"
    if de_ratio <= 0.15:
        return "UNLEVERED_INSULATION"
    return "MILD_ASYMMETRY_STABILITY"


def calculate_black_leverage_effect(
    market_price: float = 100.0,
    debt_per_share: float = DEFAULT_DEBT_PER_SHARE,
    asset_volatility: float = DEFAULT_ASSET_VOLATILITY,
    feedback_elasticity: float = DEFAULT_FEEDBACK_ELASTICITY,
    returns: Optional[Union[List[float], np.ndarray]] = None,
    ticker: str = "TICKER",
) -> BlackLeverageReceipt:
    """Solve the Black (1976) leverage effect and Campbell-Hentschel (1992) feedback model.
    
    Args:
        market_price: Current equity share price P (> 0).
        debt_per_share: Debt per share D (>= 0).
        asset_volatility: Unlevered asset volatility \\sigma_A (> 0).
        feedback_elasticity: Feedback discount parameter \\theta (> 0).
        returns: Optional historical returns to calculate empirical asymmetry (VAR).
        ticker: Normalized asset ticker symbol.
        
    Returns:
        BlackLeverageReceipt: Complete audited quantitative leverage effect evaluation.
    """
    assert_finite_number(market_price, "market_price")
    assert_finite_number(debt_per_share, "debt_per_share")
    assert_finite_number(asset_volatility, "asset_volatility")
    assert_finite_number(feedback_elasticity, "feedback_elasticity")

    if market_price <= 0.0:
        raise ValueError(f"Market price must be positive, got {market_price}")
    if debt_per_share < 0.0:
        raise ValueError(f"Debt per share must be non-negative, got {debt_per_share}")
    if asset_volatility <= 0.0:
        raise ValueError(f"Asset volatility must be positive, got {asset_volatility}")
    if feedback_elasticity <= 0.0:
        raise ValueError(f"Feedback elasticity must be positive, got {feedback_elasticity}")

    p = float(market_price)
    d = float(debt_per_share)
    sigma_a = float(asset_volatility)
    theta = float(feedback_elasticity)

    # 1. Debt-to-Equity ratio
    de_ratio = float(d / p)

    # 2. Leverage-amplified equity volatility (Black 1976): \sigma_E(P) = \sigma_A * (1 + D/P)
    # Capped at 5.0 * \sigma_A to guard against distress singularities
    raw_vol_multiplier = 1.0 + de_ratio
    vol_multiplier = float(min(5.0, raw_vol_multiplier))
    sigma_e = float(sigma_a * vol_multiplier)

    # 3. Volatility price sensitivity: d\sigma_E / dP = -\sigma_A * (D / P^2)
    vol_sensitivity = float(-sigma_a * (d / math.pow(p, 2)))

    # 4. Realized upside vs downside volatility from empirical returns
    if returns is not None and len(returns) >= 10:
        rets = np.asarray(returns, dtype=float)
        downside_rets = rets[rets < 0.0]
        upside_rets = rets[rets > 0.0]

        sigma_down = float(np.std(downside_rets) * math.sqrt(252.0)) if len(downside_rets) >= 2 else sigma_e * 1.15
        sigma_up = float(np.std(upside_rets) * math.sqrt(252.0)) if len(upside_rets) >= 2 else sigma_e * 0.85
        var = float(sigma_down / max(1e-4, sigma_up))
    else:
        # Theoretical fallback proportional to leverage
        var = float(1.0 + min(1.5, de_ratio * 0.50))
        sigma_down = float(sigma_e * math.sqrt(var))
        sigma_up = float(sigma_e / math.sqrt(var))

    # 5. Volatility feedback price discount (Campbell & Hentschel 1992)
    # \Delta P = P * min(0.50, \theta * (\sigma_E - \sigma_A))
    vol_excess = max(0.0, sigma_e - sigma_a)
    feedback_discount_pct = float(min(0.50, theta * vol_excess))
    feedback_discount_dollar = float(p * feedback_discount_pct)
    feedback_adjusted_price = float(max(0.10 * p, p - feedback_discount_dollar))

    regime = classify_leverage_effect_regime(
        de_ratio=de_ratio,
        vol_multiplier=vol_multiplier,
        var=var,
        feedback_pct=feedback_discount_pct * 100.0,
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    # Round metrics for audit consistency
    p_round = round(p, 4)
    d_round = round(d, 4)
    de_round = round(de_ratio, 4)
    sigma_a_round = round(sigma_a * 100.0, 2)
    sigma_e_round = round(sigma_e * 100.0, 2)
    vol_mult_round = round(vol_multiplier, 4)
    vol_sens_round = round(vol_sensitivity, 6)
    sigma_up_round = round(sigma_up * 100.0, 2)
    sigma_down_round = round(sigma_down * 100.0, 2)
    var_round = round(var, 4)
    feedback_dollar_round = round(feedback_discount_dollar, 4)
    feedback_pct_round = round(feedback_discount_pct * 100.0, 2)
    adj_price_round = round(feedback_adjusted_price, 4)

    digest = generate_black_leverage_audit_digest(
        ticker=ticker,
        market_price=p_round,
        debt_per_share=d_round,
        debt_to_equity_ratio=de_round,
        asset_volatility_pct=sigma_a_round,
        equity_volatility_pct=sigma_e_round,
        volatility_asymmetry_ratio=var_round,
        volatility_feedback_discount_pct=feedback_pct_round,
        feedback_adjusted_price=adj_price_round,
        regime=regime,
        calculated_at=now_iso,
    )

    summary = (
        f"Black (1976) Leverage Effect Assessment: Regime={regime} | "
        f"Price=${p_round:.2f} | Debt/Share=${d_round:.2f} (D/E={de_round:.2f}x) | "
        f"Asset Vol={sigma_a_round:.1f}% -> Equity Vol={sigma_e_round:.1f}% ({vol_mult_round:.2f}x) | "
        f"Volatility Asymmetry Ratio (VAR)={var_round:.2f}x | "
        f"Volatility Feedback Discount=-${feedback_dollar_round:.2f} (-{feedback_pct_round:.1f}%)"
    )

    return BlackLeverageReceipt(
        ticker=ticker,
        market_price=p_round,
        debt_per_share=d_round,
        debt_to_equity_ratio=de_round,
        asset_volatility_pct=sigma_a_round,
        equity_volatility_pct=sigma_e_round,
        leverage_volatility_multiplier=vol_mult_round,
        volatility_price_sensitivity=vol_sens_round,
        realized_upside_volatility_pct=sigma_up_round,
        realized_downside_volatility_pct=sigma_down_round,
        volatility_asymmetry_ratio=var_round,
        volatility_feedback_discount_dollar=feedback_dollar_round,
        volatility_feedback_discount_pct=feedback_pct_round,
        feedback_adjusted_price=adj_price_round,
        regime=regime,
        is_acute_leverage_amplification=(regime == "ACUTE_LEVERAGE_AMPLIFICATION"),
        is_volatility_feedback_cascade=(regime == "VOLATILITY_FEEDBACK_CASCADE"),
        is_unlevered_insulation=(regime == "UNLEVERED_INSULATION"),
        is_mild_asymmetry_stability=(regime == "MILD_ASYMMETRY_STABILITY"),
        summary=summary,
        audit_digest=digest,
        calculated_at=now_iso,
    )


def evaluate_black_leverage_profile(
    ticker: str,
    prices: Optional[List[float]] = None,
    debt_per_share_override: Optional[float] = None,
    asset_volatility_override: Optional[float] = None,
    feedback_elasticity: float = DEFAULT_FEEDBACK_ELASTICITY,
    db_path: Optional[Union[str, Path]] = None,
) -> BlackLeverageReceipt:
    """Evaluate Black leverage effect profile using historical price observations from database."""
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
        latest_price = 100.0
        d = DEFAULT_DEBT_PER_SHARE if debt_per_share_override is None else debt_per_share_override
        sigma_a = DEFAULT_ASSET_VOLATILITY if asset_volatility_override is None else asset_volatility_override
        rets = None
    else:
        latest_price = float(prices[-1])
        rets = list(np.diff(np.log(np.array(prices))))
        daily_vol = float(np.std(rets)) if len(rets) > 1 else 0.012
        ann_equity_vol = float(daily_vol * math.sqrt(252.0))

        # Inferred debt per share from price drawdown if not explicitly supplied
        max_p = float(np.max(prices[-60:])) if len(prices) >= 60 else latest_price
        drawdown = max(0.0, (max_p - latest_price) / max(1.0, max_p))
        inferred_debt = float(latest_price * (0.20 + drawdown * 0.80))
        d = debt_per_share_override if debt_per_share_override is not None else inferred_debt

        # Infer asset volatility: \sigma_A = \sigma_E / (1 + D/P)
        inferred_sigma_a = float(ann_equity_vol / max(1.0, 1.0 + (d / latest_price)))
        sigma_a = asset_volatility_override if asset_volatility_override is not None else max(0.05, inferred_sigma_a)

    return calculate_black_leverage_effect(
        market_price=latest_price,
        debt_per_share=d,
        asset_volatility=sigma_a,
        feedback_elasticity=feedback_elasticity,
        returns=rets,
        ticker=valid_ticker,
    )


def persist_black_leverage_record(
    receipt: BlackLeverageReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Store Black leverage effect evaluation record in SQLite database table."""
    conn = db.get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS black_leverage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                market_price REAL NOT NULL,
                debt_per_share REAL NOT NULL,
                debt_to_equity_ratio REAL NOT NULL,
                asset_volatility_pct REAL NOT NULL,
                equity_volatility_pct REAL NOT NULL,
                leverage_volatility_multiplier REAL NOT NULL,
                volatility_price_sensitivity REAL NOT NULL,
                realized_upside_volatility_pct REAL NOT NULL,
                realized_downside_volatility_pct REAL NOT NULL,
                volatility_asymmetry_ratio REAL NOT NULL,
                volatility_feedback_discount_dollar REAL NOT NULL,
                volatility_feedback_discount_pct REAL NOT NULL,
                feedback_adjusted_price REAL NOT NULL,
                regime TEXT NOT NULL,
                summary TEXT NOT NULL,
                audit_digest TEXT NOT NULL,
                calculated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO black_leverage_records (
                ticker, market_price, debt_per_share, debt_to_equity_ratio,
                asset_volatility_pct, equity_volatility_pct, leverage_volatility_multiplier,
                volatility_price_sensitivity, realized_upside_volatility_pct,
                realized_downside_volatility_pct, volatility_asymmetry_ratio,
                volatility_feedback_discount_dollar, volatility_feedback_discount_pct,
                feedback_adjusted_price, regime, summary, audit_digest, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.market_price,
                receipt.debt_per_share,
                receipt.debt_to_equity_ratio,
                receipt.asset_volatility_pct,
                receipt.equity_volatility_pct,
                receipt.leverage_volatility_multiplier,
                receipt.volatility_price_sensitivity,
                receipt.realized_upside_volatility_pct,
                receipt.realized_downside_volatility_pct,
                receipt.volatility_asymmetry_ratio,
                receipt.volatility_feedback_discount_dollar,
                receipt.volatility_feedback_discount_pct,
                receipt.feedback_adjusted_price,
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
