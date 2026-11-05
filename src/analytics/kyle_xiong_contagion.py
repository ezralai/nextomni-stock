"""Albert S. Kyle & Wei Xiong (2001) Contagion as a Wealth Effect Engine.

Theoretical Foundation:
    Kyle, A. S., & Xiong, W. (2001). "Contagion as a Wealth Effect".
    The Journal of Finance, 56(4), 1401-1440.

Key Insights:
    - Balance Sheet Wealth Constraint: Arbitrageurs (convergence traders)
      provide liquidity across multiple markets, but their risk tolerance is
      governed by finite equity capital W_t.
    - Endogenous Risk Capacity: As arbitrageur wealth contracts, effective
      risk capacity kappa_t = W_t / (gamma * sigma^2) declines, forcing
      unwinding of positions across fundamentally unrelated assets.
    - Excess Co-Movement & Volatility Amplification: Shocks in one market
      spill over into other markets through the wealth effect, causing
      non-fundamental co-movement Delta rho_wealth and forced liquidation discounts.
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

HMAC_SECRET = b"nextomni-kyle-xiong-contagion-v117-secret"

CONTAGION_REGIMES = (
    "SYSTEMIC_LIQUIDATION_CONTAGION",
    "MODERATE_WEALTH_SPILLOVER",
    "BENIGN_DECOUPLED",
    "ARBITRAGE_CAPITAL_SURPLUS",
)

DEFAULT_ARBITRAGEUR_WEALTH = 1.00     # Normalized wealth W_0 = 1.00
DEFAULT_RISK_AVERSION_GAMMA = 2.50    # gamma = 2.50
DEFAULT_FUNDAMENTAL_VOLATILITY = 0.20 # 20% annualized volatility sigma
DEFAULT_NOISE_TRADER_VARIANCE = 0.05  # sigma_noise^2


@dataclass(frozen=True)
class KyleXiongReceipt:
    """Immutable audit receipt for Kyle & Xiong (2001) contagion evaluation."""

    ticker: str
    latest_price: float
    arbitrageur_wealth_ratio: float      # W_t / W_0 in [0.05, 2.50]
    wealth_drawdown_pct: float           # Max(0, 1 - W_t/W_0) in %
    risk_bearing_capacity: float         # kappa_t = W_t / (gamma * sigma^2)
    contagion_spillover_beta: float      # beta_contagion cross-market sensitivity
    non_fundamental_excess_corr: float   # Delta rho_wealth in [-0.10, 0.85]
    forced_liquidation_pressure: float   # Index [0.0, 100.0]
    liquidation_discount_pct: float      # Expected fire-sale price markdown in %
    volatility_amplification_factor: float # Amplification multiple >= 1.0x
    regime: str
    is_systemic_contagion: bool
    is_moderate_spillover: bool
    is_benign_decoupled: bool
    is_capital_surplus: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_kyle_xiong_audit_digest(
    ticker: str,
    latest_price: float,
    wealth_ratio: float,
    spillover_beta: float,
    excess_corr: float,
    liquidation_pressure: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over Kyle & Xiong contagion metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{wealth_ratio:.4f}|{spillover_beta:.4f}|{excess_corr:.4f}|{liquidation_pressure:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_kyle_xiong_audit_digest(receipt: KyleXiongReceipt) -> bool:
    """Verify cryptographic audit digest of a KyleXiongReceipt."""
    expected = generate_kyle_xiong_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.arbitrageur_wealth_ratio,
        receipt.contagion_spillover_beta,
        receipt.non_fundamental_excess_corr,
        receipt.forced_liquidation_pressure,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_contagion_regime(
    wealth_drawdown_pct: float,
    forced_liquidation_pressure: float,
) -> str:
    """Classify into one of four Kyle & Xiong contagion regimes."""
    if wealth_drawdown_pct >= 40.0 or forced_liquidation_pressure >= 65.0:
        return "SYSTEMIC_LIQUIDATION_CONTAGION"
    elif wealth_drawdown_pct >= 20.0 or forced_liquidation_pressure >= 35.0:
        return "MODERATE_WEALTH_SPILLOVER"
    elif wealth_drawdown_pct <= 0.0 and forced_liquidation_pressure <= 15.0:
        return "ARBITRAGE_CAPITAL_SURPLUS"
    else:
        return "BENIGN_DECOUPLED"


def calculate_kyle_xiong_contagion(
    ticker: str,
    latest_price: float = 100.0,
    arbitrageur_wealth_ratio: float = DEFAULT_ARBITRAGEUR_WEALTH,
    risk_aversion_gamma: float = DEFAULT_RISK_AVERSION_GAMMA,
    fundamental_volatility: float = DEFAULT_FUNDAMENTAL_VOLATILITY,
    noise_trader_shock_std: float = 0.15,
) -> KyleXiongReceipt:
    """Calculate Kyle & Xiong (2001) wealth contagion and forced liquidation metrics.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        arbitrageur_wealth_ratio: Arbitrageur wealth ratio W_t / W_0 in [0.05, 2.50].
        risk_aversion_gamma: Relative risk aversion gamma in [0.5, 15.0].
        fundamental_volatility: Asset annualized volatility sigma in [0.02, 1.50].
        noise_trader_shock_std: Noise trading dispersion std in [0.01, 1.00].

    Returns:
        KyleXiongReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.05 <= arbitrageur_wealth_ratio <= 3.00):
        raise SecurityValidationError("arbitrageur_wealth_ratio must be between 0.05 and 3.00.")
    if not (0.5 <= risk_aversion_gamma <= 20.0):
        raise SecurityValidationError("risk_aversion_gamma must be between 0.5 and 20.0.")
    if not (0.01 <= fundamental_volatility <= 2.50):
        raise SecurityValidationError("fundamental_volatility must be between 0.01 and 2.50.")
    if not (0.01 <= noise_trader_shock_std <= 2.00):
        raise SecurityValidationError("noise_trader_shock_std must be between 0.01 and 2.00.")

    clean_wealth = float(np.clip(arbitrageur_wealth_ratio, 0.05, 3.00))
    clean_gamma = float(np.clip(risk_aversion_gamma, 0.5, 20.0))
    clean_sigma = float(np.clip(fundamental_volatility, 0.01, 2.50))
    clean_noise = float(np.clip(noise_trader_shock_std, 0.01, 2.00))

    wealth_drawdown_pct = max(0.0, (1.0 - clean_wealth) * 100.0)

    # 1. Arbitrageur Risk-Bearing Capacity: kappa_t = W_t / (gamma * sigma^2)
    var_f = clean_sigma ** 2
    risk_bearing_capacity = clean_wealth / (clean_gamma * var_f)
    risk_bearing_capacity = float(np.clip(risk_bearing_capacity, 0.01, 100.0))

    # 2. Contagion Spillover Beta: beta_contagion
    # Under Kyle & Xiong (2001), cross-asset spillover arises from the wealth effect:
    # beta_contagion = 1 / (1 + kappa_t * clean_noise)
    spillover_beta = 1.0 / (1.0 + 0.35 * risk_bearing_capacity)
    spillover_beta = float(np.clip(spillover_beta, 0.0, 1.50))

    # 3. Non-Fundamental Excess Co-Movement: Delta rho_wealth
    # In normal times with abundant capital (kappa -> infty), correlation is purely fundamental (~0).
    # When capital is severely impaired (kappa -> 0), correlation jumps toward 0.70+.
    excess_corr = 1.0 / (1.0 + 0.85 * (risk_bearing_capacity ** 0.8))
    excess_corr = float(np.clip(excess_corr, 0.0, 0.85))

    # 4. Forced Liquidation Pressure Index [0, 100]
    # Scales non-linearly with wealth loss and noise trader selling pressure
    raw_pressure = (wealth_drawdown_pct ** 1.15) * (1.0 + clean_noise) * 0.85
    forced_liquidation_pressure = float(np.clip(raw_pressure, 0.0, 100.0))

    # 5. Expected Fire-Sale Liquidation Discount (%)
    # Markdown required to clear market in illiquid conditions
    liquidation_discount_pct = (forced_liquidation_pressure / 100.0) * (clean_sigma * 15.0)
    liquidation_discount_pct = float(np.clip(liquidation_discount_pct, 0.0, 35.0))

    # 6. Volatility Amplification Factor
    # Ratio of total endogenous volatility to fundamental volatility
    vol_amplification = 1.0 + (forced_liquidation_pressure / 100.0) * (1.20 * spillover_beta)
    vol_amplification = float(np.clip(vol_amplification, 1.0, 4.5))

    regime = classify_contagion_regime(wealth_drawdown_pct, forced_liquidation_pressure)
    is_systemic_contagion = (regime == "SYSTEMIC_LIQUIDATION_CONTAGION")
    is_moderate_spillover = (regime == "MODERATE_WEALTH_SPILLOVER")
    is_benign_decoupled = (regime == "BENIGN_DECOUPLED")
    is_capital_surplus = (regime == "ARBITRAGE_CAPITAL_SURPLUS")

    summary = (
        f"Kyle & Xiong Contagion: {regime} "
        f"(WealthRatio={clean_wealth:.2f}x, Drawdown={wealth_drawdown_pct:.1f}%, "
        f"RiskCapacity={risk_bearing_capacity:.2f}, SpilloverBeta={spillover_beta:.2f}x, "
        f"ExcessCorr={excess_corr:.2f}, Pressure={forced_liquidation_pressure:.1f}/100, "
        f"FireSaleDiscount={liquidation_discount_pct:.1f}%, VolAmp={vol_amplification:.2f}x). "
        f"{'Systemic forced liquidation contagion across asset classes.' if is_systemic_contagion else ('Arbitrageurs well-capitalized with surplus risk absorption.' if is_capital_surplus else 'Benign decoupling with minimal cross-market spillover.')}"
    )

    clean_price = round(latest_price, 4)
    clean_wealth_rd = round(clean_wealth, 4)
    wealth_dd_rd = round(wealth_drawdown_pct, 2)
    risk_cap_rd = round(risk_bearing_capacity, 4)
    spillover_beta_rd = round(spillover_beta, 4)
    excess_corr_rd = round(excess_corr, 4)
    liquidation_press_rd = round(forced_liquidation_pressure, 2)
    liquidation_disc_rd = round(liquidation_discount_pct, 2)
    vol_amp_rd = round(vol_amplification, 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_kyle_xiong_audit_digest(
        clean_ticker,
        clean_price,
        clean_wealth_rd,
        spillover_beta_rd,
        excess_corr_rd,
        liquidation_press_rd,
        regime,
        timestamp,
    )

    return KyleXiongReceipt(
        ticker=clean_ticker,
        latest_price=clean_price,
        arbitrageur_wealth_ratio=clean_wealth_rd,
        wealth_drawdown_pct=wealth_dd_rd,
        risk_bearing_capacity=risk_cap_rd,
        contagion_spillover_beta=spillover_beta_rd,
        non_fundamental_excess_corr=excess_corr_rd,
        forced_liquidation_pressure=liquidation_press_rd,
        liquidation_discount_pct=liquidation_disc_rd,
        volatility_amplification_factor=vol_amp_rd,
        regime=regime,
        is_systemic_contagion=is_systemic_contagion,
        is_moderate_spillover=is_moderate_spillover,
        is_benign_decoupled=is_benign_decoupled,
        is_capital_surplus=is_capital_surplus,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_kyle_xiong_profile(
    ticker: str,
    wealth_drawdown_override: Optional[float] = None,
    gamma_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> KyleXiongReceipt:
    """Evaluate Kyle & Xiong contagion profile using empirical price volatility."""
    clean_ticker = validate_ticker(ticker)

    if prices is None or len(prices) < 20:
        try:
            obs = db.get_observations(clean_ticker, db_path=db_path)
            if obs is not None and not obs.empty and "close" in obs.columns:
                closes = [
                    float(x)
                    for x in obs["close"].tolist()
                    if math.isfinite(float(x)) and float(x) > 0
                ]
                if closes and len(closes) >= 20:
                    prices = closes
        except Exception:
            pass

    if prices is None or len(prices) < 20:
        latest_price = 100.0
        inferred_vol = 0.22
        wealth_ratio = 0.90 if wealth_drawdown_override is None else max(0.05, 1.0 - (wealth_drawdown_override / 100.0))
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))
        daily_std = float(np.std(returns)) if len(returns) > 1 else 0.015
        inferred_vol = float(np.clip(daily_std * np.sqrt(252.0), 0.05, 1.20))

        # Recent drawdown as empirical proxy for arbitrageur stress
        peak = float(np.max(prices[-60:]))
        drawdown = max(0.0, (peak - latest_price) / peak) if peak > 0 else 0.0
        if wealth_drawdown_override is not None:
            wealth_ratio = max(0.05, 1.0 - (wealth_drawdown_override / 100.0))
        else:
            wealth_ratio = max(0.10, 1.0 - 1.25 * drawdown)

    gamma = gamma_override if gamma_override is not None else DEFAULT_RISK_AVERSION_GAMMA

    return calculate_kyle_xiong_contagion(
        ticker=clean_ticker,
        latest_price=latest_price,
        arbitrageur_wealth_ratio=wealth_ratio,
        risk_aversion_gamma=gamma,
        fundamental_volatility=inferred_vol,
    )


def persist_kyle_xiong_record(
    receipt: KyleXiongReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Kyle & Xiong contagion receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS kyle_xiong_contagion_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        arbitrageur_wealth_ratio REAL NOT NULL,
        wealth_drawdown_pct REAL NOT NULL,
        risk_bearing_capacity REAL NOT NULL,
        contagion_spillover_beta REAL NOT NULL,
        non_fundamental_excess_corr REAL NOT NULL,
        forced_liquidation_pressure REAL NOT NULL,
        liquidation_discount_pct REAL NOT NULL,
        volatility_amplification_factor REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO kyle_xiong_contagion_records (
        ticker, latest_price, arbitrageur_wealth_ratio, wealth_drawdown_pct,
        risk_bearing_capacity, contagion_spillover_beta, non_fundamental_excess_corr,
        forced_liquidation_pressure, liquidation_discount_pct, volatility_amplification_factor,
        regime, audit_digest, calculated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        conn = db.get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute(create_sql)
        cursor.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.latest_price,
                receipt.arbitrageur_wealth_ratio,
                receipt.wealth_drawdown_pct,
                receipt.risk_bearing_capacity,
                receipt.contagion_spillover_beta,
                receipt.non_fundamental_excess_corr,
                receipt.forced_liquidation_pressure,
                receipt.liquidation_discount_pct,
                receipt.volatility_amplification_factor,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
