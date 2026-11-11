"""Oliver Kim & Robert E. Verrecchia (1991, 1994) Differential Precision & Announcement Liquidity Engine.

Theoretical Foundation:
    1. Kim, Oliver, & Verrecchia, Robert E. (1991).
       "Trading Volume and Price Reactions to Public Announcements".
       Journal of Accounting Research, 29(2), pp. 302-321.
       Demonstrates that expected announcement volume is proportional to the product
       of differential private information precision and absolute price change |Delta P|.

    2. Kim, Oliver, & Verrecchia, Robert E. (1994).
       "Market Liquidity and Volume Around Earnings Announcements".
       Journal of Accounting and Economics, 17(1-2), pp. 41-67.
       Models how public announcements enable superior private information processors to
       generate new informed order flow, simultaneously spiking volume and market-maker
       adverse-selection illiquidity lambda_KV.
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
HMAC_SECRET = b"kim-verrecchia-liquidity-v153-secret"


@dataclass(frozen=True)
class KimVerrecchiaReceipt:
    """Immutable quantitative audit receipt for Kim & Verrecchia (1991, 1994) disclosure equilibrium."""

    ticker: str
    prior_precision_h: float
    public_signal_precision_s: float
    mean_private_precision_p: float
    precision_dispersion_sigma_p: float
    risk_tolerance_gamma: float
    announcement_price_change_abs: float
    noise_liquidity_sigma_u: float
    total_posterior_precision: float
    price_response_alpha: float
    belief_dispersion_ratio: float
    expected_announcement_volume: float
    volume_to_price_change_ratio: float
    announcement_illiquidity_lambda: float
    baseline_illiquidity_lambda: float
    illiquidity_surge_bps: float
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


def classify_kim_verrecchia_regime(
    dispersion_ratio: float,
    illiquidity_lambda: float,
    abs_price_change: float,
) -> str:
    """Classify announcement information asymmetry and volume-liquidity regime."""
    if dispersion_ratio <= 1e-6:
        return "HOMOGENEOUS_PRECISION_NO_EXCESS_VOLUME"
    if illiquidity_lambda >= 0.65 and dispersion_ratio >= 0.18:
        return "CONCENTRATED_PRIVATE_PROCESSOR_ILLIQUIDITY"
    if abs_price_change >= 0.05 and dispersion_ratio >= 0.12:
        return "HIGH_SURPRISE_BELIEF_DISPERSION_SURGE"
    if illiquidity_lambda >= 0.40:
        return "ELEVATED_ADVERSE_SELECTION_DISCLOSURE"
    return "ORDERLY_PUBLIC_PRICE_DISCOVERY"


def _build_hmac_payload(
    ticker: str,
    h_prior: float,
    s_pub: float,
    p_mean: float,
    sigma_p: float,
    abs_dp: float,
    exp_vol: float,
    lambda_kv: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{h_prior:.4f}:{s_pub:.4f}:{p_mean:.4f}:{sigma_p:.4f}:"
        f"{abs_dp:.6f}:{exp_vol:.6f}:{lambda_kv:.6f}:{regime}:{timestamp}"
    )


def calculate_kim_verrecchia_liquidity(
    prior_precision_h: float = 4.0,
    public_signal_precision_s: float = 2.0,
    mean_private_precision_p: float = 3.0,
    precision_dispersion_sigma_p: float = 1.5,
    risk_tolerance_gamma: float = 1.0,
    announcement_price_change_abs: float = 0.04,
    noise_liquidity_sigma_u: float = 0.50,
    ticker: str = "KV91",
) -> KimVerrecchiaReceipt:
    """Evaluate Kim & Verrecchia (1991, 1994) announcement volume and illiquidity equilibrium."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    h_prior = float(prior_precision_h)
    s_pub = float(public_signal_precision_s)
    p_mean = float(mean_private_precision_p)
    sigma_p = float(precision_dispersion_sigma_p)
    gamma = float(risk_tolerance_gamma)
    abs_dp = float(announcement_price_change_abs)
    sigma_u = float(noise_liquidity_sigma_u)

    if h_prior <= 0.0:
        raise ValueError("prior_precision_h must be strictly positive.")
    if s_pub < 0.0 or p_mean < 0.0 or sigma_p < 0.0:
        raise ValueError("Signal precisions and dispersion must be non-negative.")
    if gamma <= 0.0 or sigma_u <= 0.0:
        raise ValueError("risk_tolerance_gamma and noise_liquidity_sigma_u must be strictly positive.")
    if abs_dp < 0.0:
        raise ValueError("announcement_price_change_abs must be non-negative.")

    total_prec = h_prior + p_mean + s_pub
    price_alpha = s_pub / total_prec
    disp_ratio = sigma_p / total_prec

    # KV (1991) folded-normal trading volume driven by differential precision & |Delta P|
    slope_kv = math.sqrt(2.0 / math.pi) * gamma * disp_ratio
    expected_vol = slope_kv * abs_dp
    vol_to_dp = slope_kv if abs_dp >= 0.0 else 0.0

    # KV (1994) announcement-induced information asymmetry illiquidity lambda_KV
    # Pre-disclosure baseline reflects static private asymmetry p_mean without announcement catalysis
    baseline_lambda = math.sqrt(p_mean) / ((h_prior + p_mean) * sigma_u) if p_mean > 0.0 else 0.0
    if s_pub == 0.0 or (p_mean + sigma_p) == 0.0:
        ann_lambda = baseline_lambda
    else:
        event_asymmetry = math.sqrt(s_pub * (p_mean + sigma_p)) / (total_prec * sigma_u)
        ann_lambda = baseline_lambda + event_asymmetry

    surge_bps = max(0.0, (ann_lambda - baseline_lambda) * 10000.0)

    regime = classify_kim_verrecchia_regime(
        dispersion_ratio=disp_ratio,
        illiquidity_lambda=ann_lambda,
        abs_price_change=abs_dp,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        h_prior=h_prior,
        s_pub=s_pub,
        p_mean=p_mean,
        sigma_p=sigma_p,
        abs_dp=abs_dp,
        exp_vol=expected_vol,
        lambda_kv=ann_lambda,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return KimVerrecchiaReceipt(
        ticker=cleaned_ticker,
        prior_precision_h=h_prior,
        public_signal_precision_s=s_pub,
        mean_private_precision_p=p_mean,
        precision_dispersion_sigma_p=sigma_p,
        risk_tolerance_gamma=gamma,
        announcement_price_change_abs=abs_dp,
        noise_liquidity_sigma_u=sigma_u,
        total_posterior_precision=total_prec,
        price_response_alpha=price_alpha,
        belief_dispersion_ratio=disp_ratio,
        expected_announcement_volume=expected_vol,
        volume_to_price_change_ratio=vol_to_dp,
        announcement_illiquidity_lambda=ann_lambda,
        baseline_illiquidity_lambda=baseline_lambda,
        illiquidity_surge_bps=surge_bps,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_kim_verrecchia_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "KV91",
    public_signal_precision_s: float = 2.0,
    precision_dispersion_sigma_p: float = 1.5,
) -> KimVerrecchiaReceipt:
    """Calibrate announcement surprise magnitude and prior precision from trailing market prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        ret_std = float(np.std(log_rets))
        ann_vol = ret_std * math.sqrt(252.0)
        abs_dp = float(np.clip(abs(log_rets[-1]) if len(log_rets) > 0 else ret_std, 0.002, 0.25))
        # Higher trailing volatility maps to lower prior belief precision h
        prior_h = float(np.clip(1.0 / max(0.05, ann_vol), 1.0, 15.0))
    else:
        abs_dp = 0.04
        prior_h = 4.0

    return calculate_kim_verrecchia_liquidity(
        prior_precision_h=prior_h,
        public_signal_precision_s=float(public_signal_precision_s),
        mean_private_precision_p=3.0,
        precision_dispersion_sigma_p=float(precision_dispersion_sigma_p),
        announcement_price_change_abs=abs_dp,
        ticker=ticker,
    )


def persist_kim_verrecchia_record(
    receipt: KimVerrecchiaReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Kim & Verrecchia (1991, 1994) audit record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS kim_verrecchia_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prior_precision_h REAL NOT NULL,
        public_signal_precision_s REAL NOT NULL,
        precision_dispersion_sigma_p REAL NOT NULL,
        announcement_price_change_abs REAL NOT NULL,
        price_response_alpha REAL NOT NULL,
        expected_announcement_volume REAL NOT NULL,
        volume_to_price_change_ratio REAL NOT NULL,
        announcement_illiquidity_lambda REAL NOT NULL,
        illiquidity_surge_bps REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO kim_verrecchia_audits (
        ticker, prior_precision_h, public_signal_precision_s, precision_dispersion_sigma_p,
        announcement_price_change_abs, price_response_alpha, expected_announcement_volume,
        volume_to_price_change_ratio, announcement_illiquidity_lambda, illiquidity_surge_bps,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.prior_precision_h,
                receipt.public_signal_precision_s,
                receipt.precision_dispersion_sigma_p,
                receipt.announcement_price_change_abs,
                receipt.price_response_alpha,
                receipt.expected_announcement_volume,
                receipt.volume_to_price_change_ratio,
                receipt.announcement_illiquidity_lambda,
                receipt.illiquidity_surge_bps,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

