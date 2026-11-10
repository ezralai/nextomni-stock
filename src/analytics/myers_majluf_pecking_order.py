"""Stewart C. Myers & Nicholas S. Majluf (1984) Pecking Order & Equity Adverse Selection Engine.

Theoretical Foundation:
    Myers, Stewart C., & Majluf, Nicholas S. (1984). "Corporate Financing and Investment
    Decisions when Firms Have Information That Investors Do Not Have".
    Journal of Financial Economics, 13(2), pp. 187-221.
    The most cited foundational paper in corporate finance, capital structure,
    and equity offering dilution.

Key Concepts:
    1. Pecking Order Theory:
       Firms finance investments hierarchically due to information asymmetry costs:
           Internal Cash Reserves > Safe Debt > Risky External Equity

    2. Equity Issuance Lemons Markdown:
       Insiders possess superior private information regarding true assets-in-place A.
       Because managers acting for existing shareholders only issue equity if it is
       fairly priced or overvalued (A <= P), an equity offering announcement signals
       negative news, triggering an immediate market price markdown:
           Delta P_issuance < 0

    3. The Underinvestment Trap:
       If assets-in-place are undervalued by the market (A > P), issuing equity dilutes
       old shareholders. If the dilution transfer exceeds the project's net present value:
           NPV < [(I - C) / (P + NPV)] * (A - P)
       the firm rationally passes up a valuable positive-NPV project (Underinvestment).

    4. Value of Financial Slack:
       Internal cash reserves C insulate the firm from costly equity dilution, preserving
       valuable growth opportunities without adverse selection penalties.
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
HMAC_SECRET = b"myers-majluf-pecking-order-v133-secret"


@dataclass(frozen=True)
class MyersMajlufReceipt:
    """Immutable quantitative audit receipt for Myers-Majluf (1984) pecking order equilibrium."""

    ticker: str
    assets_in_place_true: float      # A
    assets_market_price: float       # P = E[A]
    project_outlay: float            # I
    project_payoff: float            # B
    project_npv: float               # B - I
    cash_slack: float                # C
    external_financing_needed: float # max(0, I - C)
    information_asymmetry_spread: float  # sigma_A
    undervaluation_wedge: float      # max(0, A - P)
    dilution_transfer: float         # [(I - C) / (P + NPV)] * (A - P)
    net_old_shareholder_gain: float  # NPV - Dilution Transfer
    underinvestment_triggered: bool  # True if NPV > 0 but passed up
    equity_lemons_discount_pct: float# Announcement drop %
    equity_lemons_discount_dollar: float # Drop in $
    financing_source: str            # "INTERNAL_CASH", "SAFE_DEBT", "EXTERNAL_EQUITY"
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


def calculate_myers_majluf_equilibrium(
    assets_in_place_true: float = 120.0,
    assets_market_price: float = 100.0,
    project_outlay: float = 30.0,
    project_payoff: float = 35.0,
    cash_slack: float = 10.0,
    information_asymmetry_spread: float = 0.15,
    ticker: str = "CORP",
) -> MyersMajlufReceipt:
    """Calculate Myers & Majluf (1984) financing equilibrium, dilution wedge, and underinvestment.

    Args:
        assets_in_place_true: True intrinsic value of assets in place A (> 0).
        assets_market_price: Market prior valuation of assets in place P (> 0).
        project_outlay: Required capital expenditure I (> 0).
        project_payoff: Expected gross terminal project cash flow B (> 0).
        cash_slack: Available internal cash reserves / safe debt capacity C (>= 0).
        information_asymmetry_spread: Info spread between insiders and market sigma_A (>= 0).
        ticker: Asset ticker symbol.

    Returns:
        MyersMajlufReceipt detailing financing hierarchy, dilution, and underinvestment status.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "CORP"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    a = max(1.0, float(assets_in_place_true))
    p = max(1.0, float(assets_market_price))
    i_outlay = max(1.0, float(project_outlay))
    b_payoff = max(0.0, float(project_payoff))
    c_slack = max(0.0, float(cash_slack))
    sigma_a = max(0.0, float(information_asymmetry_spread))

    npv = b_payoff - i_outlay

    # Pecking order hierarchy:
    # 1. Internal cash slack covers outlay (C >= I): No external financing needed
    if c_slack >= i_outlay:
        external_financing = 0.0
        financing_source = "INTERNAL_CASH"
        dilution_transfer = 0.0
        net_gain = npv
        underinvestment = False if npv >= 0 else True
        lemons_discount_pct = 0.0
        lemons_discount_dollar = 0.0
    else:
        # 2. External equity financing needed (I - C > 0):
        external_financing = i_outlay - c_slack
        financing_source = "EXTERNAL_EQUITY"

        # Undervaluation wedge A - P
        undervaluation = max(0.0, a - p)

        # In Myers-Majluf (1984), issuing equity gives new investors a fraction f of post-project firm
        # Dilution transfer from old to new shareholders = [ (I - C) / (P + max(0, NPV)) ] * (A - P)
        post_val_base = p + max(0.0, npv)
        dilution_ratio = external_financing / post_val_base if post_val_base > 0 else 0.0
        dilution_transfer = dilution_ratio * undervaluation

        net_gain = npv - dilution_transfer

        # Myers-Majluf Underinvestment Condition:
        # Project is positive-NPV (NPV > 0), but net gain to old shareholders is negative (net_gain < 0)
        # Old shareholders pass up the project because dilution loss > project NPV!
        if npv > 0.0 and net_gain < 0.0:
            underinvestment = True
        else:
            underinvestment = False if npv >= 0.0 else True

        # Market Lemons Discount upon equity issuance announcement:
        # Market infers overvaluation risk proportional to information asymmetry
        lemons_discount_pct = float(np.clip(
            (sigma_a * 0.50 + 0.02) * (external_financing / (p + external_financing)),
            0.0,
            0.15
        ))
        lemons_discount_dollar = p * lemons_discount_pct

    regime = classify_myers_majluf_regime(
        underinvestment=underinvestment,
        npv=npv,
        lemons_discount_pct=lemons_discount_pct,
        financing_source=financing_source,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    signature_payload = (
        f"{cleaned_ticker}:{a:.4f}:{p:.4f}:{i_outlay:.4f}:{b_payoff:.4f}:"
        f"{c_slack:.4f}:{dilution_transfer:.4f}:{net_gain:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(signature_payload)

    return MyersMajlufReceipt(
        ticker=cleaned_ticker,
        assets_in_place_true=a,
        assets_market_price=p,
        project_outlay=i_outlay,
        project_payoff=b_payoff,
        project_npv=npv,
        cash_slack=c_slack,
        external_financing_needed=external_financing,
        information_asymmetry_spread=sigma_a,
        undervaluation_wedge=max(0.0, a - p),
        dilution_transfer=dilution_transfer,
        net_old_shareholder_gain=net_gain,
        underinvestment_triggered=underinvestment,
        equity_lemons_discount_pct=lemons_discount_pct,
        equity_lemons_discount_dollar=lemons_discount_dollar,
        financing_source=financing_source,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def classify_myers_majluf_regime(
    underinvestment: bool,
    npv: float,
    lemons_discount_pct: float,
    financing_source: str,
) -> str:
    """Classify the Myers-Majluf corporate financing regime."""
    if financing_source == "INTERNAL_CASH":
        return "PECKING_ORDER_INTERNAL_FINANCED"
    if underinvestment and npv > 0:
        return "UNDERINVESTMENT_TRAP"
    if lemons_discount_pct >= 0.04:
        return "ADVERSE_SELECTION_LEMONS_DISCOUNT"
    if lemons_discount_pct < 0.01:
        return "SYMMETRIC_INFORMATION_FIRST_BEST"
    return "EQUITY_FINANCED_DILUTION_STABLE"


def evaluate_myers_majluf_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "CORP",
    cash_slack: Optional[float] = None,
    capex_outlay: Optional[float] = None,
    info_spread: float = 0.15,
) -> MyersMajlufReceipt:
    """Empirically evaluate the Myers-Majluf profile from equity price history and balance sheet proxies."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) < 10:
        return calculate_myers_majluf_equilibrium(
            assets_in_place_true=115.0,
            assets_market_price=100.0,
            project_outlay=25.0,
            project_payoff=30.0,
            cash_slack=10.0 if cash_slack is None else cash_slack,
            information_asymmetry_spread=info_spread,
            ticker=ticker,
        )

    current_price = float(p_arr[-1])
    # Fundamental proxy: highest rolling median valuation
    p_high = float(np.max(p_arr[-60:] if len(p_arr) >= 60 else p_arr))
    true_assets = max(current_price * 1.05, p_high)

    # Capex proxy: ~20% of share price
    outlay = current_price * 0.20 if capex_outlay is None else float(capex_outlay)
    payoff = outlay * 1.25  # +25% return on capital (positive NPV)

    # Cash slack proxy: ~10% of share price
    slack = current_price * 0.10 if cash_slack is None else float(cash_slack)

    return calculate_myers_majluf_equilibrium(
        assets_in_place_true=true_assets,
        assets_market_price=current_price,
        project_outlay=outlay,
        project_payoff=payoff,
        cash_slack=slack,
        information_asymmetry_spread=info_spread,
        ticker=ticker,
    )


def persist_myers_majluf_record(
    receipt: MyersMajlufReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Myers-Majluf record to SQLite database using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS myers_majluf_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        assets_in_place_true REAL NOT NULL,
        assets_market_price REAL NOT NULL,
        project_outlay REAL NOT NULL,
        project_payoff REAL NOT NULL,
        project_npv REAL NOT NULL,
        cash_slack REAL NOT NULL,
        external_financing_needed REAL NOT NULL,
        information_asymmetry_spread REAL NOT NULL,
        undervaluation_wedge REAL NOT NULL,
        dilution_transfer REAL NOT NULL,
        net_old_shareholder_gain REAL NOT NULL,
        underinvestment_triggered INTEGER NOT NULL,
        equity_lemons_discount_pct REAL NOT NULL,
        equity_lemons_discount_dollar REAL NOT NULL,
        financing_source TEXT NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_mm_ticker_time ON myers_majluf_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO myers_majluf_audits (
        ticker, assets_in_place_true, assets_market_price, project_outlay,
        project_payoff, project_npv, cash_slack, external_financing_needed,
        information_asymmetry_spread, undervaluation_wedge, dilution_transfer,
        net_old_shareholder_gain, underinvestment_triggered, equity_lemons_discount_pct,
        equity_lemons_discount_dollar, financing_source, regime, hmac_signature, timestamp
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
                    receipt.assets_in_place_true,
                    receipt.assets_market_price,
                    receipt.project_outlay,
                    receipt.project_payoff,
                    receipt.project_npv,
                    receipt.cash_slack,
                    receipt.external_financing_needed,
                    receipt.information_asymmetry_spread,
                    receipt.undervaluation_wedge,
                    receipt.dilution_transfer,
                    receipt.net_old_shareholder_gain,
                    1 if receipt.underinvestment_triggered else 0,
                    receipt.equity_lemons_discount_pct,
                    receipt.equity_lemons_discount_dollar,
                    receipt.financing_source,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False
