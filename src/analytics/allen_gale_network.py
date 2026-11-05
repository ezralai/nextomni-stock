"""Franklin Allen & Douglas Gale (2000) Financial Contagion & Network Cascades Engine.

Theoretical Foundation:
    Allen, F., & Gale, D. (2000). "Financial Contagion".
    Journal of Political Economy, 108(1), 1-33.
    Eisenberg, L., & Noe, T. H. (2001). "Systemic Risk in Financial Systems".
    Management Science, 47(2), 236-249.

Key Insights:
    - Network Architecture Determines Fragility: Financial fragility is driven
      by the topology of cross-holdings and liquidity commitments.
    - Complete vs. Incomplete Networks:
        * In a complete network, each node shares mutual exposures with all other nodes.
          Localized shocks are dispersed globally and absorbed with minimal distress.
        * In an incomplete network (e.g., circular, sequential, or disconnected clusters),
          liquidity cannot flow freely to the point of strain, causing localized
          defaults to cascade sequentially into systemic collapse.
    - Cascade Multiplier:
        Measures the amplification of a localized liquidity deficit into
        systemic fire-sales across the network.
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

HMAC_SECRET = b"nextomni-allen-gale-network-v122-secret"

NETWORK_REGIMES = (
    "SYSTEMIC_CASCADE_COLLAPSE",
    "INCOMPLETE_FRAGILE_NETWORK",
    "RESILIENT_PARTIAL_SHARING",
    "COMPLETE_NETWORK_FORTRESS",
)

DEFAULT_NETWORK_COMPLETENESS = 0.60     # 60% network completeness gamma_net
DEFAULT_LOCAL_LIQUIDITY_SHOCK = 0.35    # 35% localized liquidity demand shock theta
DEFAULT_AGGREGATE_LIQUIDITY = 1.00      # Normalized aggregate liquidity buffer L_agg


@dataclass(frozen=True)
class AllenGaleNetworkReceipt:
    """Immutable audit receipt for Allen & Gale (2000) network contagion evaluation."""

    ticker: str
    latest_price: float
    network_completeness_ratio: float      # gamma_net in [0.05, 1.00]
    local_liquidity_shock_ratio: float     # theta_shock in [0.01, 1.00]
    aggregate_liquidity_buffer: float      # L_agg normalized buffer
    cross_holding_absorption_capacity: float # B_buffer = gamma_net * L_agg
    unabsorbed_liquidity_deficit: float    # Max(0, theta - B_buffer)
    contagion_cascade_multiplier: float    # M_cascade >= 1.0x
    systemic_contagion_index: float        # Index [0.0, 100.0]
    network_fire_sale_markdown_pct: float  # Expected price markdown in %
    systemic_hurdle_surcharge_pct: float   # Required return premium in % / year
    regime: str
    is_systemic_cascade: bool
    is_incomplete_fragile: bool
    is_resilient_sharing: bool
    is_complete_fortress: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_allen_gale_audit_digest(
    ticker: str,
    latest_price: float,
    completeness: float,
    shock: float,
    multiplier: float,
    contagion_index: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over Allen & Gale network contagion metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{completeness:.4f}|{shock:.4f}|{multiplier:.4f}|{contagion_index:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_allen_gale_audit_digest(receipt: AllenGaleNetworkReceipt) -> bool:
    """Verify cryptographic audit digest of an AllenGaleNetworkReceipt."""
    expected = generate_allen_gale_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.network_completeness_ratio,
        receipt.local_liquidity_shock_ratio,
        receipt.contagion_cascade_multiplier,
        receipt.systemic_contagion_index,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_network_regime(
    completeness: float,
    unabsorbed_deficit: float,
    contagion_index: float,
) -> str:
    """Classify into one of four Allen & Gale (2000) network contagion regimes."""
    if (completeness <= 0.35 and unabsorbed_deficit > 0.05) or contagion_index >= 65.0:
        return "SYSTEMIC_CASCADE_COLLAPSE"
    elif (completeness <= 0.65 and unabsorbed_deficit > 0.0) or contagion_index >= 35.0:
        return "INCOMPLETE_FRAGILE_NETWORK"
    elif completeness >= 0.88 or unabsorbed_deficit <= 0.001:
        return "COMPLETE_NETWORK_FORTRESS"
    else:
        return "RESILIENT_PARTIAL_SHARING"


def calculate_allen_gale_network(
    ticker: str,
    latest_price: float = 100.0,
    network_completeness_ratio: float = DEFAULT_NETWORK_COMPLETENESS,
    local_liquidity_shock_ratio: float = DEFAULT_LOCAL_LIQUIDITY_SHOCK,
    aggregate_liquidity_buffer: float = DEFAULT_AGGREGATE_LIQUIDITY,
    asset_volatility: float = 0.22,
) -> AllenGaleNetworkReceipt:
    """Calculate Allen & Gale (2000) financial network contagion and cascade metrics.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        network_completeness_ratio: Completeness degree gamma_net in [0.05, 1.00].
        local_liquidity_shock_ratio: Localized liquidity shock theta in [0.01, 1.00].
        aggregate_liquidity_buffer: Normalized aggregate network liquidity L_agg in [0.10, 5.00].
        asset_volatility: Annualized asset volatility sigma in [0.02, 2.00].

    Returns:
        AllenGaleNetworkReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.05 <= network_completeness_ratio <= 1.00):
        raise SecurityValidationError("network_completeness_ratio must be between 0.05 and 1.00.")
    if not (0.01 <= local_liquidity_shock_ratio <= 1.50):
        raise SecurityValidationError("local_liquidity_shock_ratio must be between 0.01 and 1.50.")
    if not (0.05 <= aggregate_liquidity_buffer <= 10.0):
        raise SecurityValidationError("aggregate_liquidity_buffer must be between 0.05 and 10.0.")
    if not (0.01 <= asset_volatility <= 3.00):
        raise SecurityValidationError("asset_volatility must be between 0.01 and 3.00.")

    clean_gamma = float(np.clip(network_completeness_ratio, 0.05, 1.00))
    clean_theta = float(np.clip(local_liquidity_shock_ratio, 0.01, 1.50))
    clean_l_agg = float(np.clip(aggregate_liquidity_buffer, 0.05, 10.0))
    clean_vol = float(np.clip(asset_volatility, 0.01, 3.00))

    # 1. Cross-Holding Absorption Capacity: B_buffer = gamma_net * L_agg
    absorption_capacity = clean_gamma * clean_l_agg

    # 2. Unabsorbed Liquidity Deficit: Max(0, theta_shock - B_buffer)
    unabsorbed_deficit = max(0.0, clean_theta - absorption_capacity)

    # 3. Contagion Cascade Multiplier: M_cascade >= 1.0
    # In complete network (gamma -> 1.0), multiplier is strictly 1.0x (zero amplification).
    # In incomplete sparse network (gamma -> 0), unabsorbed deficit cascades along links.
    if unabsorbed_deficit > 0.0:
        incomplete_penalty = (1.0 - clean_gamma) / max(clean_gamma, 0.15)
        raw_multiplier = 1.0 + incomplete_penalty * ((unabsorbed_deficit / max(absorption_capacity, 0.05)) ** 0.85) * 0.80
    else:
        raw_multiplier = 1.00
    cascade_multiplier = float(np.clip(raw_multiplier, 1.00, 10.00))

    # 4. Systemic Contagion Probability Index [0.0, 100.0]
    raw_index = (1.0 - clean_gamma) * 55.0 + (unabsorbed_deficit / max(clean_theta, 0.05)) * 45.0
    contagion_index = float(np.clip(raw_index, 0.0, 100.0))

    # 5. Network Fire-Sale Price Markdown (%)
    raw_markdown = (cascade_multiplier - 1.0) * clean_vol * 12.0
    fire_sale_markdown_pct = float(np.clip(raw_markdown, 0.0, 45.0))

    # 6. Systemic Hurdle Surcharge (% / year)
    surcharge_pct = (contagion_index / 100.0) * clean_vol * 4.50
    surcharge_pct = float(np.clip(surcharge_pct, 0.0, 25.0))

    regime = classify_network_regime(clean_gamma, unabsorbed_deficit, contagion_index)
    is_cascade = (regime == "SYSTEMIC_CASCADE_COLLAPSE")
    is_fragile = (regime == "INCOMPLETE_FRAGILE_NETWORK")
    is_resilient = (regime == "RESILIENT_PARTIAL_SHARING")
    is_fortress = (regime == "COMPLETE_NETWORK_FORTRESS")

    summary = (
        f"Allen-Gale Network Contagion: {regime} "
        f"(Completeness={clean_gamma:.1%}, Shock={clean_theta:.1%}, Capacity={absorption_capacity:.2f}, "
        f"Deficit={unabsorbed_deficit:.2f}, CascadeMult={cascade_multiplier:.2f}x, "
        f"ContagionIndex={contagion_index:.1f}/100, FireSaleMarkdown={fire_sale_markdown_pct:.1f}%, "
        f"Surcharge=+{surcharge_pct:.2f}%). "
        f"{'Systemic cascading defaults across fragile incomplete network.' if is_cascade else ('Complete network fortress: localized shock fully absorbed with zero cascade.' if is_fortress else 'Moderate network fragility with partial cross-holding risk.')}"
    )

    clean_p_rd = round(latest_price, 4)
    gamma_rd = round(clean_gamma, 4)
    theta_rd = round(clean_theta, 4)
    l_agg_rd = round(clean_l_agg, 4)
    capacity_rd = round(absorption_capacity, 4)
    deficit_rd = round(unabsorbed_deficit, 4)
    mult_rd = round(cascade_multiplier, 2)
    index_rd = round(contagion_index, 1)
    markdown_rd = round(fire_sale_markdown_pct, 2)
    surch_rd = round(surcharge_pct, 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_allen_gale_audit_digest(
        clean_ticker,
        clean_p_rd,
        gamma_rd,
        theta_rd,
        mult_rd,
        index_rd,
        regime,
        timestamp,
    )

    return AllenGaleNetworkReceipt(
        ticker=clean_ticker,
        latest_price=clean_p_rd,
        network_completeness_ratio=gamma_rd,
        local_liquidity_shock_ratio=theta_rd,
        aggregate_liquidity_buffer=l_agg_rd,
        cross_holding_absorption_capacity=capacity_rd,
        unabsorbed_liquidity_deficit=deficit_rd,
        contagion_cascade_multiplier=mult_rd,
        systemic_contagion_index=index_rd,
        network_fire_sale_markdown_pct=markdown_rd,
        systemic_hurdle_surcharge_pct=surch_rd,
        regime=regime,
        is_systemic_cascade=is_cascade,
        is_incomplete_fragile=is_fragile,
        is_resilient_sharing=is_resilient,
        is_complete_fortress=is_fortress,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_allen_gale_network_profile(
    ticker: str,
    network_completeness_override: Optional[float] = None,
    liquidity_shock_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> AllenGaleNetworkReceipt:
    """Evaluate Allen & Gale network profile using historical market volatility and correlation."""
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
        completeness = 0.55 if network_completeness_override is None else network_completeness_override
        shock = 0.35 if liquidity_shock_override is None else liquidity_shock_override
        vol = 0.22
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))
        daily_vol = float(np.std(returns)) if len(returns) > 1 else 0.015
        vol = float(np.clip(daily_vol * np.sqrt(252.0), 0.05, 1.50))

        # Infer network completeness and shock from recent return tail risk
        recent_max_drop = abs(min(0.0, float(np.min(returns[-30:])))) if len(returns) >= 30 else 0.02
        inferred_shock = float(np.clip(recent_max_drop * 10.0, 0.10, 0.85))
        shock = liquidity_shock_override if liquidity_shock_override is not None else inferred_shock

        inferred_completeness = float(np.clip(1.0 - (vol / 0.60) * 0.5, 0.20, 0.95))
        completeness = network_completeness_override if network_completeness_override is not None else inferred_completeness

    return calculate_allen_gale_network(
        ticker=clean_ticker,
        latest_price=latest_price,
        network_completeness_ratio=completeness,
        local_liquidity_shock_ratio=shock,
        asset_volatility=vol,
    )


def persist_allen_gale_network_record(
    receipt: AllenGaleNetworkReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Allen & Gale network receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS allen_gale_network_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        network_completeness_ratio REAL NOT NULL,
        local_liquidity_shock_ratio REAL NOT NULL,
        aggregate_liquidity_buffer REAL NOT NULL,
        cross_holding_absorption_capacity REAL NOT NULL,
        unabsorbed_liquidity_deficit REAL NOT NULL,
        contagion_cascade_multiplier REAL NOT NULL,
        systemic_contagion_index REAL NOT NULL,
        network_fire_sale_markdown_pct REAL NOT NULL,
        systemic_hurdle_surcharge_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO allen_gale_network_records (
        ticker, latest_price, network_completeness_ratio,
        local_liquidity_shock_ratio, aggregate_liquidity_buffer,
        cross_holding_absorption_capacity, unabsorbed_liquidity_deficit,
        contagion_cascade_multiplier, systemic_contagion_index,
        network_fire_sale_markdown_pct, systemic_hurdle_surcharge_pct,
        regime, audit_digest, calculated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                receipt.network_completeness_ratio,
                receipt.local_liquidity_shock_ratio,
                receipt.aggregate_liquidity_buffer,
                receipt.cross_holding_absorption_capacity,
                receipt.unabsorbed_liquidity_deficit,
                receipt.contagion_cascade_multiplier,
                receipt.systemic_contagion_index,
                receipt.network_fire_sale_markdown_pct,
                receipt.systemic_hurdle_surcharge_pct,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
