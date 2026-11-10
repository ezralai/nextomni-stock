"""Stewart C. Myers (1977) Debt Overhang, Real Growth Options & Underinvestment Engine.

Theoretical Foundation:
    Myers, Stewart C. (1977). "Determinants of Corporate Borrowing". Journal of Financial
    Economics, 5(2), pp. 147-175.
    The foundational corporate finance theory of Debt Overhang, proving that senior risky
    debt maturing after an investment horizon captures positive-NPV growth payoffs and forces
    levered equity holders to forfeit valuable real growth options (underinvestment).

Key Concepts:
    1. First-Best Growth Option Valuation (V_growth^FB):
       An all-equity firm exercises a discretionary real growth option G(s) with outlay I
       whenever G(s) > I (positive NPV states).

    2. Levered Equity Exercise Rule & Debt Overhang (Myers 1977):
       Shareholders bear 100% of outlay I while senior debt of face D absorbs initial
       upside Delta_debt(s) = min(D, A(s) + G(s)) - min(D, A(s)). Shareholders exercise iff:
           max(0, A(s) + G(s) - D) - I > max(0, A(s) - D)
       In states where G(s) > I yet G(s) - I <= Delta_debt(s), shareholders rationally pass up
       the positive-NPV project, destroying enterprise value.

    3. Deadweight Overhang Loss & Firm Value Conservation:
       L_overhang = e^{-rT} E[(G(s) - I) * 1_{overhang}(s)] = V_firm^FB - V_firm^levered >= 0
       Levered Firm Value V_firm^levered = E_0 + D_0 = V_firm^FB - L_overhang.
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
HMAC_SECRET = b"myers-debt-overhang-v140-secret"

# Precompute 64-point Gauss-Hermite quadrature for exact normal expectations
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(64)
_STD_NORM_NODES = np.sqrt(2.0) * _GH_NODES
_STD_NORM_PROBS = _GH_WEIGHTS / np.sqrt(np.pi)


@dataclass(frozen=True)
class MyersOverhangReceipt:
    """Immutable quantitative audit receipt for Myers (1977) debt overhang equilibrium."""

    ticker: str
    assets_in_place_pv: float          # A_0
    growth_option_gross_pv: float      # G_0
    investment_outlay: float           # I
    debt_face_value: float             # D
    asset_volatility: float            # sigma_A
    growth_volatility: float           # sigma_G
    horizon_years: float               # T
    risk_free_rate: float              # r
    first_best_growth_value: float     # V_growth^FB
    first_best_firm_value: float       # V_firm^FB
    levered_growth_captured: float     # Actual growth value exercised
    levered_equity_value: float        # E_0
    levered_debt_value: float          # D_0
    levered_firm_value: float          # E_0 + D_0
    overhang_agency_loss: float        # L_overhang = V_firm^FB - V_firm^levered
    overhang_severity_pct: float       # L_overhang / V_growth^FB
    overhang_probability: float        # P(G > I and rejected by equity)
    debt_overhang_triggered: bool      # True if L_overhang > 1e-6
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


def classify_overhang_regime(
    debt_overhang_triggered: bool,
    overhang_severity_pct: float,
    debt_face_value: float,
) -> str:
    """Classify the Myers (1977) debt overhang and underinvestment regime."""
    if debt_face_value <= 1e-8 or not debt_overhang_triggered:
        return "OVERHANG_FREE_FIRST_BEST_INVESTMENT"
    if overhang_severity_pct >= 0.40:
        return "SEVERE_DEBT_OVERHANG_GROWTH_PARALYSIS"
    if overhang_severity_pct >= 0.15:
        return "MODERATE_OVERHANG_UNDERINVESTMENT"
    return "MILD_DEBT_SUBSIDY_FRICTION"


def calculate_myers_debt_overhang(
    assets_in_place_pv: float = 100.0,
    growth_option_gross_pv: float = 60.0,
    investment_outlay: float = 40.0,
    debt_face_value: float = 95.0,
    asset_volatility: float = 0.25,
    growth_volatility: float = 0.35,
    horizon_years: float = 1.0,
    risk_free_rate: float = 0.04,
    ticker: str = "GROWTH",
) -> MyersOverhangReceipt:
    """Compute Myers (1977) debt overhang loss, growth option forfeiture, and claim values.

    Args:
        assets_in_place_pv: Current risk-neutral present value of existing assets A_0 (> 0).
        growth_option_gross_pv: Expected gross PV of growth project G_0 (>= 0).
        investment_outlay: Capital outlay I required at maturity to exercise growth option (>= 0).
        debt_face_value: Senior debt face value D due at maturity (>= 0).
        asset_volatility: Annual volatility sigma_A of existing assets in place (> 0).
        growth_volatility: Annual volatility sigma_G of growth option cash flows (> 0).
        horizon_years: Option exercise & debt maturity horizon T in years (> 0).
        risk_free_rate: Continuously compounded risk-free rate r.
        ticker: Security ticker symbol.

    Returns:
        MyersOverhangReceipt quantifying debt overhang loss, captured growth value, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "GROWTH"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    a0 = float(assets_in_place_pv)
    if not math.isfinite(a0) or a0 <= 0.0:
        raise ValueError(f"assets_in_place_pv must be finite and positive, got: {assets_in_place_pv}")

    g0 = max(0.0, float(growth_option_gross_pv))
    i_outlay = max(0.0, float(investment_outlay))
    d_face = max(0.0, float(debt_face_value))
    sig_a = max(0.02, float(asset_volatility))
    sig_g = max(0.02, float(growth_volatility))
    t = max(0.05, float(horizon_years))
    r = float(risk_free_rate)
    df = math.exp(-r * t)
    sqrt_t = math.sqrt(t)

    # Terminal state arrays over Gauss-Hermite quadrature nodes (E[e^{-rT} A_T] = A_0, E[e^{-rT} G_T] = G_0)
    a_t = a0 * np.exp((r - 0.5 * sig_a ** 2) * t + sig_a * sqrt_t * _STD_NORM_NODES)
    g_t = g0 * np.exp((r - 0.5 * sig_g ** 2) * t + sig_g * sqrt_t * _STD_NORM_NODES)

    # First-best all-equity rule: exercise growth option whenever G_T > I
    fb_invest = g_t > i_outlay
    fb_growth_payoff = np.where(fb_invest, g_t - i_outlay, 0.0)
    fb_growth_val = float(df * np.sum(_STD_NORM_PROBS * fb_growth_payoff))
    fb_firm_val = float(df * np.sum(_STD_NORM_PROBS * (a_t + fb_growth_payoff)))

    # Levered equity rule: exercise iff equity gain exceeds zero-investment equity payoff
    eq_if_invest = np.maximum(0.0, a_t + g_t - d_face) - i_outlay
    eq_if_pass = np.maximum(0.0, a_t - d_face)
    lev_invest = eq_if_invest > eq_if_pass

    eq_payoff = np.where(lev_invest, eq_if_invest, eq_if_pass)
    debt_if_invest = np.minimum(d_face, a_t + g_t)
    debt_if_pass = np.minimum(d_face, a_t)
    debt_payoff = np.where(lev_invest, debt_if_invest, debt_if_pass)

    lev_equity_val = float(df * np.sum(_STD_NORM_PROBS * eq_payoff))
    lev_debt_val = float(df * np.sum(_STD_NORM_PROBS * debt_payoff))
    lev_firm_val = lev_equity_val + lev_debt_val

    lev_growth_payoff = np.where(lev_invest, g_t - i_outlay, 0.0)
    lev_growth_val = float(df * np.sum(_STD_NORM_PROBS * lev_growth_payoff))

    overhang_states = fb_invest & (~lev_invest)
    overhang_prob = float(np.sum(_STD_NORM_PROBS * overhang_states))
    overhang_loss = max(0.0, fb_firm_val - lev_firm_val)
    overhang_triggered = overhang_loss > 1e-5
    overhang_severity = float(np.clip(overhang_loss / fb_growth_val, 0.0, 1.0)) if fb_growth_val > 1e-8 else 0.0

    regime = classify_overhang_regime(
        debt_overhang_triggered=overhang_triggered,
        overhang_severity_pct=overhang_severity,
        debt_face_value=d_face,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{a0:.4f}:{g0:.4f}:{i_outlay:.4f}:{d_face:.4f}:"
        f"{sig_a:.4f}:{sig_g:.4f}:{fb_firm_val:.4f}:{lev_firm_val:.4f}:"
        f"{overhang_loss:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return MyersOverhangReceipt(
        ticker=cleaned_ticker,
        assets_in_place_pv=a0,
        growth_option_gross_pv=g0,
        investment_outlay=i_outlay,
        debt_face_value=d_face,
        asset_volatility=sig_a,
        growth_volatility=sig_g,
        horizon_years=t,
        risk_free_rate=r,
        first_best_growth_value=fb_growth_val,
        first_best_firm_value=fb_firm_val,
        levered_growth_captured=lev_growth_val,
        levered_equity_value=lev_equity_val,
        levered_debt_value=lev_debt_val,
        levered_firm_value=lev_firm_val,
        overhang_agency_loss=overhang_loss,
        overhang_severity_pct=overhang_severity,
        overhang_probability=overhang_prob,
        debt_overhang_triggered=overhang_triggered,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_myers_overhang_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "GROWTH",
    debt_to_assets_ratio: float = 0.85,
    capex_to_assets_ratio: float = 0.35,
    growth_multiple: float = 1.45,
) -> MyersOverhangReceipt:
    """Empirically calibrate Myers (1977) debt overhang profile from historical equity prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 10:
        spot = float(p_arr[-1])
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        sig_a = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.12, 0.70))
    else:
        spot = 100.0
        sig_a = 0.25

    a0 = spot
    outlay = a0 * max(0.05, float(capex_to_assets_ratio))
    g0 = outlay * max(0.80, float(growth_multiple))
    d_face = a0 * max(0.0, float(debt_to_assets_ratio))

    return calculate_myers_debt_overhang(
        assets_in_place_pv=a0,
        growth_option_gross_pv=g0,
        investment_outlay=outlay,
        debt_face_value=d_face,
        asset_volatility=sig_a,
        growth_volatility=min(0.85, sig_a * 1.35),
        horizon_years=1.0,
        risk_free_rate=0.04,
        ticker=ticker,
    )


def persist_myers_overhang_record(
    receipt: MyersOverhangReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Myers (1977) debt overhang record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS myers_overhang_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        assets_in_place_pv REAL NOT NULL,
        growth_option_gross_pv REAL NOT NULL,
        investment_outlay REAL NOT NULL,
        debt_face_value REAL NOT NULL,
        asset_volatility REAL NOT NULL,
        growth_volatility REAL NOT NULL,
        horizon_years REAL NOT NULL,
        risk_free_rate REAL NOT NULL,
        first_best_growth_value REAL NOT NULL,
        first_best_firm_value REAL NOT NULL,
        levered_growth_captured REAL NOT NULL,
        levered_equity_value REAL NOT NULL,
        levered_debt_value REAL NOT NULL,
        levered_firm_value REAL NOT NULL,
        overhang_agency_loss REAL NOT NULL,
        overhang_severity_pct REAL NOT NULL,
        overhang_probability REAL NOT NULL,
        debt_overhang_triggered INTEGER NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_mo_ticker_time ON myers_overhang_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO myers_overhang_audits (
        ticker, assets_in_place_pv, growth_option_gross_pv, investment_outlay,
        debt_face_value, asset_volatility, growth_volatility, horizon_years,
        risk_free_rate, first_best_growth_value, first_best_firm_value,
        levered_growth_captured, levered_equity_value, levered_debt_value,
        levered_firm_value, overhang_agency_loss, overhang_severity_pct,
        overhang_probability, debt_overhang_triggered, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.assets_in_place_pv,
                    receipt.growth_option_gross_pv,
                    receipt.investment_outlay,
                    receipt.debt_face_value,
                    receipt.asset_volatility,
                    receipt.growth_volatility,
                    receipt.horizon_years,
                    receipt.risk_free_rate,
                    receipt.first_best_growth_value,
                    receipt.first_best_firm_value,
                    receipt.levered_growth_captured,
                    receipt.levered_equity_value,
                    receipt.levered_debt_value,
                    receipt.levered_firm_value,
                    receipt.overhang_agency_loss,
                    receipt.overhang_severity_pct,
                    receipt.overhang_probability,
                    1 if receipt.debt_overhang_triggered else 0,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

