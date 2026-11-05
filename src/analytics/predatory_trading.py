"""Markus K. Brunnermeier & Lasse Heje Pedersen (2005) Predatory Trading Engine.

Theoretical Foundation:
    Brunnermeier, M. K., & Pedersen, L. H. (2005). "Predatory Trading".
    The Journal of Finance, 60(4), 1825-1863.

Key Insights:
    - Strategic Liquidation Front-Running: When a large distressed trader is
      forced to liquidate an asset position (e.g., margin calls or fund redemptions),
      other strategic traders (predators) have an incentive to sell alongside
      or ahead of the distressed trader rather than providing liquidity.
    - Predatory Price Overshoot: Predation temporarily pushes the price
      below the post-liquidation fundamental level, creating a temporary price
      trough before a sharp rebound.
    - Liquidity Evaporation & Slippage Inflation: Predatory selling drains
      market depth precisely when the distressed entity requires liquidity,
      multiplying total execution slippage and triggering further liquidations.
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

HMAC_SECRET = b"nextomni-predatory-trading-v118-secret"

PREDATORY_REGIMES = (
    "AGGRESSIVE_PREDATORY_RUN",
    "COORDINATED_FRONT_RUNNING",
    "COMPETITIVE_ABSORPTION",
    "BENIGN_LIQUIDITY_PROVISION",
)

DEFAULT_DISTRESSED_SIZE_PCT = 3.50       # 3.5% of Average Daily Volume (ADV)
DEFAULT_PREDATOR_CONCENTRATION = 0.50    # 50% predator share of strategic order flow
DEFAULT_MARKET_DEPTH_FACTOR = 1.00       # Normal market depth lambda_inv = 1.0x
DEFAULT_PRICE_IMPACT_COEFF = 0.0250      # Kyle's lambda coefficient


@dataclass(frozen=True)
class PredatoryTradingReceipt:
    """Immutable audit receipt for Brunnermeier & Pedersen (2005) predatory trading evaluation."""

    ticker: str
    latest_price: float
    distressed_liquidation_size_pct_adv: float  # Q_distressed as % of ADV [0.1, 25.0]
    predator_concentration_ratio: float         # Predator order flow share [0.05, 0.95]
    market_depth_multiplier: float             # Market depth parameter [0.20, 5.00]
    predatory_shorting_volume_pct_adv: float   # Q_predator as % of ADV
    price_overshoot_ratio: float               # (P_0 - P_trough) / (P_0 - P_rebound) >= 1.0x
    predatory_liquidity_drain_pct: float       # % of available book depth consumed by predators
    distressed_slippage_inflation_pct: float   # Execution cost inflation suffered by distressed trader
    expected_rebound_potential_pct: float      # Price recovery from trough in %
    regime: str
    is_aggressive_predatory_run: bool
    is_coordinated_front_running: bool
    is_competitive_absorption: bool
    is_benign_liquidity: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_predatory_audit_digest(
    ticker: str,
    latest_price: float,
    distressed_size_pct: float,
    predator_ratio: float,
    price_overshoot: float,
    liquidity_drain_pct: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over predatory trading metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{distressed_size_pct:.4f}|{predator_ratio:.4f}|{price_overshoot:.4f}|{liquidity_drain_pct:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_predatory_audit_digest(receipt: PredatoryTradingReceipt) -> bool:
    """Verify cryptographic audit digest of a PredatoryTradingReceipt."""
    expected = generate_predatory_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.distressed_liquidation_size_pct_adv,
        receipt.predator_concentration_ratio,
        receipt.price_overshoot_ratio,
        receipt.predatory_liquidity_drain_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_predatory_regime(
    distressed_size_pct: float,
    predator_ratio: float,
    price_overshoot: float,
) -> str:
    """Classify into one of four Brunnermeier & Pedersen (2005) predatory regimes."""
    if (distressed_size_pct >= 4.0 and predator_ratio >= 0.55) or (distressed_size_pct >= 3.0 and price_overshoot >= 1.45):
        return "AGGRESSIVE_PREDATORY_RUN"
    elif (distressed_size_pct >= 2.0 and predator_ratio >= 0.35) or (distressed_size_pct >= 1.5 and price_overshoot >= 1.25):
        return "COORDINATED_FRONT_RUNNING"
    elif predator_ratio <= 0.20 and distressed_size_pct >= 1.2:
        return "COMPETITIVE_ABSORPTION"
    else:
        return "BENIGN_LIQUIDITY_PROVISION"


def calculate_predatory_trading(
    ticker: str,
    latest_price: float = 100.0,
    distressed_liquidation_size_pct_adv: float = DEFAULT_DISTRESSED_SIZE_PCT,
    predator_concentration_ratio: float = DEFAULT_PREDATOR_CONCENTRATION,
    market_depth_multiplier: float = DEFAULT_MARKET_DEPTH_FACTOR,
    price_impact_lambda: float = DEFAULT_PRICE_IMPACT_COEFF,
) -> PredatoryTradingReceipt:
    """Calculate Brunnermeier & Pedersen (2005) predatory trading and price overshoot metrics.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        distressed_liquidation_size_pct_adv: Distressed liquidation size as % of ADV in [0.1, 25.0].
        predator_concentration_ratio: Strategic predator concentration ratio in [0.05, 0.95].
        market_depth_multiplier: Market depth capacity multiplier in [0.20, 5.00].
        price_impact_lambda: Price impact Kyle lambda coefficient in [0.001, 0.20].

    Returns:
        PredatoryTradingReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.10 <= distressed_liquidation_size_pct_adv <= 30.0):
        raise SecurityValidationError("distressed_liquidation_size_pct_adv must be between 0.10 and 30.0.")
    if not (0.05 <= predator_concentration_ratio <= 0.95):
        raise SecurityValidationError("predator_concentration_ratio must be between 0.05 and 0.95.")
    if not (0.10 <= market_depth_multiplier <= 10.0):
        raise SecurityValidationError("market_depth_multiplier must be between 0.10 and 10.0.")
    if not (0.001 <= price_impact_lambda <= 0.50):
        raise SecurityValidationError("price_impact_lambda must be between 0.001 and 0.50.")

    clean_size = float(np.clip(distressed_liquidation_size_pct_adv, 0.10, 30.0))
    clean_predator = float(np.clip(predator_concentration_ratio, 0.05, 0.95))
    clean_depth = float(np.clip(market_depth_multiplier, 0.10, 10.0))
    clean_lambda = float(np.clip(price_impact_lambda, 0.001, 0.50))

    # Effective illiquidity slope adjusted for depth
    effective_lambda = clean_lambda / clean_depth

    # 1. Predatory Shorting Volume: Q_predator = Q_distressed * predator_ratio * front_run_multiplier
    # Under Brunnermeier-Pedersen, predators front-run proportionally to the size of the prey
    predatory_short_pct = clean_size * clean_predator * 0.85
    predatory_short_pct = float(np.clip(predatory_short_pct, 0.05, 20.0))

    # Total order flow pressure during predatory run
    total_selling_pressure = clean_size + predatory_short_pct

    # 2. Predatory Price Overshoot Ratio: (P_0 - P_trough) / (P_0 - P_rebound)
    # Fundamental permanent impact is driven solely by Q_distressed
    # Temporary trough impact is driven by (Q_distressed + Q_predator)
    permanent_impact_pct = clean_size * effective_lambda * 100.0
    trough_impact_pct = total_selling_pressure * effective_lambda * 100.0

    if permanent_impact_pct > 0.001:
        price_overshoot_ratio = trough_impact_pct / permanent_impact_pct
    else:
        price_overshoot_ratio = 1.00
    price_overshoot_ratio = float(np.clip(price_overshoot_ratio, 1.00, 3.50))

    # 3. Predatory Liquidity Drain (% of book depth consumed by predators)
    raw_drain = (predatory_short_pct / (clean_depth * 4.0)) * 100.0
    liquidity_drain_pct = float(np.clip(raw_drain, 0.0, 95.0))

    # 4. Distressed Slippage Inflation (% execution cost increase for distressed trader)
    # Caused by the predator pushing the price down before the distressed trader finishes
    slippage_inflation_pct = (price_overshoot_ratio - 1.0) * 85.0
    slippage_inflation_pct = float(np.clip(slippage_inflation_pct, 0.0, 150.0))

    # 5. Expected Post-Liquidation Rebound Potential (%)
    rebound_pct = trough_impact_pct - permanent_impact_pct
    rebound_pct = float(np.clip(rebound_pct, 0.0, 25.0))

    regime = classify_predatory_regime(clean_size, clean_predator, price_overshoot_ratio)
    is_aggressive_predatory_run = (regime == "AGGRESSIVE_PREDATORY_RUN")
    is_coordinated_front_running = (regime == "COORDINATED_FRONT_RUNNING")
    is_competitive_absorption = (regime == "COMPETITIVE_ABSORPTION")
    is_benign_liquidity = (regime == "BENIGN_LIQUIDITY_PROVISION")

    summary = (
        f"Brunnermeier-Pedersen Predatory Trading: {regime} "
        f"(DistressedSize={clean_size:.1f}% ADV, PredatorShare={clean_predator:.1%}, "
        f"PredatoryShort={predatory_short_pct:.1f}% ADV, Overshoot={price_overshoot_ratio:.2f}x, "
        f"LiquidityDrain={liquidity_drain_pct:.1f}%, SlippageInflation={slippage_inflation_pct:+.1f}%, "
        f"ReboundPotential={rebound_pct:.1f}%). "
        f"{'Severe predatory run front-running distressed liquidation.' if is_aggressive_predatory_run else ('Coordinated strategic front-running detected.' if is_coordinated_front_running else 'Benign or competitive liquidity absorption.')}"
    )

    clean_price = round(latest_price, 4)
    clean_size_rd = round(clean_size, 2)
    clean_predator_rd = round(clean_predator, 4)
    clean_depth_rd = round(clean_depth, 2)
    predatory_short_rd = round(predatory_short_pct, 2)
    overshoot_rd = round(price_overshoot_ratio, 2)
    liquidity_drain_rd = round(liquidity_drain_pct, 2)
    slippage_infl_rd = round(slippage_inflation_pct, 2)
    rebound_rd = round(rebound_pct, 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_predatory_audit_digest(
        clean_ticker,
        clean_price,
        clean_size_rd,
        clean_predator_rd,
        overshoot_rd,
        liquidity_drain_rd,
        regime,
        timestamp,
    )

    return PredatoryTradingReceipt(
        ticker=clean_ticker,
        latest_price=clean_price,
        distressed_liquidation_size_pct_adv=clean_size_rd,
        predator_concentration_ratio=clean_predator_rd,
        market_depth_multiplier=clean_depth_rd,
        predatory_shorting_volume_pct_adv=predatory_short_rd,
        price_overshoot_ratio=overshoot_rd,
        predatory_liquidity_drain_pct=liquidity_drain_rd,
        distressed_slippage_inflation_pct=slippage_infl_rd,
        expected_rebound_potential_pct=rebound_rd,
        regime=regime,
        is_aggressive_predatory_run=is_aggressive_predatory_run,
        is_coordinated_front_running=is_coordinated_front_running,
        is_competitive_absorption=is_competitive_absorption,
        is_benign_liquidity=is_benign_liquidity,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_predatory_trading_profile(
    ticker: str,
    distressed_size_override: Optional[float] = None,
    predator_ratio_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> PredatoryTradingReceipt:
    """Evaluate predatory trading profile from historical price and volume dynamics."""
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
                if "volume" in obs.columns:
                    vols = [
                        float(x)
                        for x in obs["volume"].tolist()
                        if math.isfinite(float(x)) and float(x) >= 0
                    ]
                    if vols and len(vols) >= 20:
                        volumes = vols
        except Exception:
            pass

    if prices is None or len(prices) < 20:
        latest_price = 110.0
        distressed_size = 3.20 if distressed_size_override is None else distressed_size_override
        predator_ratio = 0.45 if predator_ratio_override is None else predator_ratio_override
        depth_mult = 1.00
    else:
        latest_price = float(prices[-1])
        # Volatility as proxy for inverse market depth
        returns = np.diff(np.log(np.array(prices)))
        daily_vol = float(np.std(returns)) if len(returns) > 1 else 0.015
        depth_mult = float(np.clip(0.015 / max(daily_vol, 0.005), 0.35, 2.50))

        # Recent price trend & volume surge as empirical footprint of distressed selling
        if distressed_size_override is not None:
            distressed_size = distressed_size_override
        else:
            recent_drawdown = max(0.0, (float(np.max(prices[-20:])) - latest_price) / float(np.max(prices[-20:])))
            distressed_size = float(np.clip(1.5 + recent_drawdown * 15.0, 0.5, 12.0))

        predator_ratio = predator_ratio_override if predator_ratio_override is not None else DEFAULT_PREDATOR_CONCENTRATION

    return calculate_predatory_trading(
        ticker=clean_ticker,
        latest_price=latest_price,
        distressed_liquidation_size_pct_adv=distressed_size,
        predator_concentration_ratio=predator_ratio,
        market_depth_multiplier=depth_mult,
    )


def persist_predatory_trading_record(
    receipt: PredatoryTradingReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist predatory trading receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS predatory_trading_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        distressed_liquidation_size_pct_adv REAL NOT NULL,
        predator_concentration_ratio REAL NOT NULL,
        market_depth_multiplier REAL NOT NULL,
        predatory_shorting_volume_pct_adv REAL NOT NULL,
        price_overshoot_ratio REAL NOT NULL,
        predatory_liquidity_drain_pct REAL NOT NULL,
        distressed_slippage_inflation_pct REAL NOT NULL,
        expected_rebound_potential_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO predatory_trading_records (
        ticker, latest_price, distressed_liquidation_size_pct_adv,
        predator_concentration_ratio, market_depth_multiplier,
        predatory_shorting_volume_pct_adv, price_overshoot_ratio,
        predatory_liquidity_drain_pct, distressed_slippage_inflation_pct,
        expected_rebound_potential_pct, regime, audit_digest, calculated_at
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
                receipt.distressed_liquidation_size_pct_adv,
                receipt.predator_concentration_ratio,
                receipt.market_depth_multiplier,
                receipt.predatory_shorting_volume_pct_adv,
                receipt.price_overshoot_ratio,
                receipt.predatory_liquidity_drain_pct,
                receipt.distressed_slippage_inflation_pct,
                receipt.expected_rebound_potential_pct,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
