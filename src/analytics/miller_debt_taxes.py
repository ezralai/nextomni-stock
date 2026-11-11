"""Merton H. Miller (1977) Debt and Taxes Personal Clientele Equilibrium Engine.

Theoretical Foundation:
    Miller, Merton H. (1977). "Debt and Taxes". The Journal of Finance, 32(2), pp. 261-275.
    (1976 American Finance Association Presidential Address).
    Generalizes Modigliani & Miller (1963) to incorporate differential personal income taxes
    on corporate equity (tau_ps) and debt interest (tau_pd), explaining why aggregate corporate
    leverage achieves macro-equilibrium without unbounded tax-driven debt issuance.

Key Concepts:
    1. General Miller (1977) Net Tax Gain from Leverage:
           G_L = [ 1 - ((1 - tau_c) * (1 - tau_ps)) / (1 - tau_pd) ] * B_D
       When personal equity and bond taxes match (tau_ps = tau_pd), this collapses identically
       to Modigliani & Miller (1963): G_L = tau_c * B_D.

    2. Investor Tax-Bracket Clienteles & Break-Even Marginal Tax Rate (tau_pd*):
       Low-bracket investors (tau_pd < tau_pd*) capture positive after-tax gains from holding
       corporate bonds; high-bracket investors (tau_pd > tau_pd*) strictly prefer equity returns.
           tau_pd* = 1 - (1 - tau_c) * (1 - tau_ps)

    3. Aggregate Miller Equilibrium & Taxable Bond Yield Parity:
       Market clearing corporate bond yield rises to r_d* = r_0 / (1 - tau_pd*), where at
       the marginal tax bracket tau_pd*, incremental debt financing yields zero net tax advantage.
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
HMAC_SECRET = b"miller-debt-taxes-v147-secret"


@dataclass(frozen=True)
class MillerTaxEquilibriumReceipt:
    """Immutable quantitative audit receipt for Miller (1977) personal clientele tax equilibrium."""

    ticker: str
    unlevered_firm_value: float          # V_U
    debt_market_value: float             # B_D
    corporate_tax_rate: float            # tau_c
    personal_equity_tax_rate: float      # tau_ps
    personal_debt_tax_rate: float        # tau_pd
    tax_free_rate: float                 # r_0
    miller_gain_factor: float            # phi_Miller = 1 - (1-tau_c)(1-tau_ps)/(1-tau_pd)
    miller_leverage_gain: float          # G_L = phi_Miller * B_D
    mm63_corporate_shield: float         # tau_c * B_D
    personal_tax_erosion: float          # mm63 - G_L
    levered_firm_value: float            # V_U + G_L
    breakeven_debt_tax_rate: float       # tau_pd* = 1 - (1-tau_c)(1-tau_ps)
    market_clearing_bond_yield: float    # r_0 / (1 - tau_pd*)
    tax_clientele_spread_bps: float      # (r_d* - r_0) * 10000
    investor_clientele: str
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


def classify_miller_tax_regime(
    miller_factor: float,
    tau_pd: float,
    breakeven_tau: float,
    debt_value: float,
) -> tuple[str, str]:
    """Classify investor clientele and Miller (1977) leverage equilibrium regime."""
    if debt_value <= 1e-8:
        return "ALL_EQUITY_CLIENTELE", "UNLEVERED_ZERO_TAX_SHIELD"
    if abs(tau_pd - breakeven_tau) <= 0.01:
        return "MARGINAL_EQUILIBRIUM_CLIENTELE", "MILLER_MACRO_INDIFFERENCE_EQUILIBRIUM"
    if miller_factor < -0.005:
        return "HIGH_BRACKET_EQUITY_CLIENTELE", "TAX_PENALTY_NEGATIVE_LEVERAGE_GAIN"
    if miller_factor >= 0.15:
        return "TAX_EXEMPT_BOND_CLIENTELE", "STRONG_AFTER_TAX_DEBT_ADVANTAGE"
    return "MODERATE_BOND_CLIENTELE", "PARTIAL_PERSONAL_TAX_OFFSET"


def calculate_miller_debt_taxes(
    unlevered_firm_value: float = 100.0,
    debt_market_value: float = 40.0,
    corporate_tax_rate: float = 0.25,
    personal_equity_tax_rate: float = 0.12,
    personal_debt_tax_rate: float = 0.30,
    tax_free_rate: float = 0.035,
    ticker: str = "TAX",
) -> MillerTaxEquilibriumReceipt:
    """Compute Miller (1977) net tax gain from leverage, clientele brackets, and yield parity.

    Args:
        unlevered_firm_value: Unlevered enterprise value V_U (> 0).
        debt_market_value: Corporate debt market value B_D (>= 0).
        corporate_tax_rate: Statutory corporate tax rate tau_c in [0.0, 0.60].
        personal_equity_tax_rate: Effective personal tax rate on equity income tau_ps in [0.0, 0.50].
        personal_debt_tax_rate: Investor personal tax rate on bond interest tau_pd in [0.0, 0.70].
        tax_free_rate: Benchmark tax-exempt municipal/treasury yield r_0 (> 0).
        ticker: Security ticker symbol.

    Returns:
        MillerTaxEquilibriumReceipt detailing leverage tax gain, break-even bracket, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "TAX"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    vu = float(unlevered_firm_value)
    if not math.isfinite(vu) or vu <= 0.0:
        raise ValueError(f"unlevered_firm_value must be finite and positive, got: {unlevered_firm_value}")

    bd = max(0.0, float(debt_market_value))
    tau_c = float(np.clip(corporate_tax_rate, 0.0, 0.60))
    tau_ps = float(np.clip(personal_equity_tax_rate, 0.0, 0.50))
    tau_pd = float(np.clip(personal_debt_tax_rate, 0.0, 0.70))
    r0 = max(1e-4, float(tax_free_rate))

    # Miller (1977) Eq. (2): phi = 1 - [(1 - tau_c)(1 - tau_ps) / (1 - tau_pd)]
    miller_factor = 1.0 - (((1.0 - tau_c) * (1.0 - tau_ps)) / (1.0 - tau_pd))
    g_l = miller_factor * bd
    mm63_shield = tau_c * bd
    personal_erosion = mm63_shield - g_l
    vl = max(0.01, vu + g_l)

    # Macro equilibrium marginal tax rate tau_pd* = 1 - (1 - tau_c)(1 - tau_ps)
    breakeven_tau = 1.0 - (1.0 - tau_c) * (1.0 - tau_ps)
    clearing_yield = r0 / max(1e-4, 1.0 - breakeven_tau)
    spread_bps = (clearing_yield - r0) * 10000.0

    clientele, regime = classify_miller_tax_regime(
        miller_factor=miller_factor,
        tau_pd=tau_pd,
        breakeven_tau=breakeven_tau,
        debt_value=bd,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{vu:.4f}:{bd:.4f}:{tau_c:.4f}:{tau_ps:.4f}:{tau_pd:.4f}:"
        f"{miller_factor:.6f}:{g_l:.4f}:{breakeven_tau:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return MillerTaxEquilibriumReceipt(
        ticker=cleaned_ticker,
        unlevered_firm_value=vu,
        debt_market_value=bd,
        corporate_tax_rate=tau_c,
        personal_equity_tax_rate=tau_ps,
        personal_debt_tax_rate=tau_pd,
        tax_free_rate=r0,
        miller_gain_factor=miller_factor,
        miller_leverage_gain=g_l,
        mm63_corporate_shield=mm63_shield,
        personal_tax_erosion=personal_erosion,
        levered_firm_value=vl,
        breakeven_debt_tax_rate=breakeven_tau,
        market_clearing_bond_yield=clearing_yield,
        tax_clientele_spread_bps=spread_bps,
        investor_clientele=clientele,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_miller_tax_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "TAX",
    corporate_tax_rate: float = 0.25,
    personal_equity_tax_rate: float = 0.12,
    personal_debt_tax_rate: float = 0.30,
    debt_ratio: float = 0.40,
) -> MillerTaxEquilibriumReceipt:
    """Empirically evaluate Miller (1977) personal tax clientele equilibrium from equity prices."""
    p_arr = np.asarray(prices, dtype=float)
    spot = float(p_arr[-1]) if len(p_arr) >= 5 else 100.0
    debt_val = spot * max(0.0, float(debt_ratio))

    return calculate_miller_debt_taxes(
        unlevered_firm_value=spot,
        debt_market_value=debt_val,
        corporate_tax_rate=corporate_tax_rate,
        personal_equity_tax_rate=personal_equity_tax_rate,
        personal_debt_tax_rate=personal_debt_tax_rate,
        tax_free_rate=0.035,
        ticker=ticker,
    )


def persist_miller_tax_record(
    receipt: MillerTaxEquilibriumReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Miller (1977) tax clientele equilibrium record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS miller_tax_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        unlevered_firm_value REAL NOT NULL,
        debt_market_value REAL NOT NULL,
        corporate_tax_rate REAL NOT NULL,
        personal_equity_tax_rate REAL NOT NULL,
        personal_debt_tax_rate REAL NOT NULL,
        tax_free_rate REAL NOT NULL,
        miller_gain_factor REAL NOT NULL,
        miller_leverage_gain REAL NOT NULL,
        mm63_corporate_shield REAL NOT NULL,
        personal_tax_erosion REAL NOT NULL,
        levered_firm_value REAL NOT NULL,
        breakeven_debt_tax_rate REAL NOT NULL,
        market_clearing_bond_yield REAL NOT NULL,
        tax_clientele_spread_bps REAL NOT NULL,
        investor_clientele TEXT NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_miller_tax_ticker_time ON miller_tax_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO miller_tax_audits (
        ticker, unlevered_firm_value, debt_market_value, corporate_tax_rate,
        personal_equity_tax_rate, personal_debt_tax_rate, tax_free_rate,
        miller_gain_factor, miller_leverage_gain, mm63_corporate_shield,
        personal_tax_erosion, levered_firm_value, breakeven_debt_tax_rate,
        market_clearing_bond_yield, tax_clientele_spread_bps, investor_clientele,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.unlevered_firm_value,
                    receipt.debt_market_value,
                    receipt.corporate_tax_rate,
                    receipt.personal_equity_tax_rate,
                    receipt.personal_debt_tax_rate,
                    receipt.tax_free_rate,
                    receipt.miller_gain_factor,
                    receipt.miller_leverage_gain,
                    receipt.mm63_corporate_shield,
                    receipt.personal_tax_erosion,
                    receipt.levered_firm_value,
                    receipt.breakeven_debt_tax_rate,
                    receipt.market_clearing_bond_yield,
                    receipt.tax_clientele_spread_bps,
                    receipt.investor_clientele,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

