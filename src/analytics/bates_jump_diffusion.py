r"""David S. Bates (1996) Stochastic Volatility Jump-Diffusion (SVJ) Option Pricing Engine.

Implements David S. Bates (1996), "Jumps and Stochastic Volatility: Exchange Rate Processes
Implicit in Deutsche Mark Options", The Review of Financial Studies, 9(1), pp. 69-107.

- Unifies Heston (1993) continuous mean-reverting stochastic variance with Merton (1976)
  log-normal jump-diffusion under risk-neutral measure Q:
      dS_t / S_{t-} = (r - q - lambda * k) dt + sqrt(V_t) dW_t^S + (J - 1) dN_t
      dV_t = kappa * (theta - V_t) dt + sigma_v * sqrt(V_t) dW_t^V
      d<W^S, W^V>_t = rho * dt
  where:
      N_t is Poisson with intensity lambda >= 0.
      ln(J) ~ Normal(mu_J, sigma_J^2), with k = E[J - 1] = exp(mu_J + 0.5 * sigma_J^2) - 1.
      Feller condition for variance strictly positive: 2 * kappa * theta > sigma_v^2.
- Closed-form factorized characteristic function:
      phi_Bates(u) = phi_Heston(u; V0, kappa, theta, sigma_v, rho) * phi_Jump(u; lambda, mu_J, sigma_J)
  using the stable Albrecher et al. (2007) / Lord-Kahl continuous branch formulation.
- Semi-analytical European option pricing via Gil-Pelaez Fourier inversion with Gauss-Legendre quadrature.
- Decomposes option value into continuous stochastic volatility vs jump risk contributions.
- Strict ticker input validation, HMAC-SHA256 audit sealing, and parameterized SQLite logging.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-bates-1996-audit-key"


@dataclass(frozen=True)
class BatesReceipt:
    """Immutable receipt for David S. Bates (1996) SVJ option pricing."""

    ticker: str
    spot_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    dividend_yield: float
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    lambda_jump: float
    mu_jump: float
    sigma_jump: float
    bates_call_price: float
    bates_put_price: float
    heston_benchmark_call: float
    jump_contribution: float
    feller_ratio: float
    feller_satisfied: bool
    regime: str
    hmac_signature: str
    timestamp: str


def _validate_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not clean or not TICKER_REGEX.match(clean):
        raise ValueError(
            f"Invalid ticker: '{ticker}'. Must be alphanumeric with optional ._- and max 20 chars."
        )
    return clean


def _build_hmac_payload(
    *,
    ticker: str,
    spot_price: float,
    strike_price: float,
    time_to_maturity: float,
    bates_call_price: float,
    bates_put_price: float,
    heston_benchmark_call: float,
    jump_contribution: float,
    feller_ratio: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{spot_price:.4f}|{strike_price:.4f}|{time_to_maturity:.4f}|"
        f"{bates_call_price:.8f}|{bates_put_price:.8f}|{heston_benchmark_call:.8f}|"
        f"{jump_contribution:.8f}|{feller_ratio:.4f}|{regime}|{timestamp}"
    )


def sign_bates_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_bates_receipt(payload)
    return hmac.compare_digest(expected, signature)


def _bates_characteristic_function(
    u: complex,
    *,
    t: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    lambda_jump: float,
    mu_jump: float,
    sigma_jump: float,
) -> complex:
    """Evaluate stable Bates (1996) characteristic function phi(u) = E[exp(i * u * ln(S_T / S_0))]."""
    # 1. Heston component (Albrecher / Lord-Kahl stable continuous branch formulation)
    xi = sigma_v
    i_u = 1j * u
    d = cmath.sqrt((kappa - i_u * rho * xi) ** 2 + (xi ** 2) * (u ** 2 + i_u))
    g = (kappa - i_u * rho * xi - d) / (kappa - i_u * rho * xi + d)

    exp_minus_dt = cmath.exp(-d * t)
    c_term = (r - q) * i_u * t + (kappa * theta / (xi ** 2)) * (
        (kappa - i_u * rho * xi - d) * t - 2.0 * cmath.log((1.0 - g * exp_minus_dt) / (1.0 - g))
    )
    d_term = ((kappa - i_u * rho * xi - d) / (xi ** 2)) * ((1.0 - exp_minus_dt) / (1.0 - g * exp_minus_dt))
    phi_heston = cmath.exp(c_term + d_term * v0)

    # 2. Merton log-normal jump component
    if lambda_jump > 1e-12:
        k_mean = math.exp(mu_jump + 0.5 * (sigma_jump ** 2)) - 1.0
        jump_cf = cmath.exp(i_u * mu_jump - 0.5 * (sigma_jump ** 2) * (u ** 2)) - 1.0
        drift_comp = -i_u * lambda_jump * k_mean * t
        phi_jump = cmath.exp(lambda_jump * t * jump_cf + drift_comp)
    else:
        phi_jump = 1.0 + 0.0j

    return phi_heston * phi_jump


def calculate_bates_option_price(
    *,
    spot_price: float = 100.0,
    strike_price: float = 100.0,
    time_to_maturity: float = 0.50,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.30,
    rho: float = -0.50,
    lambda_jump: float = 0.50,
    mu_jump: float = -0.05,
    sigma_jump: float = 0.15,
    ticker: str = "SPY",
    n_quad: int = 128,
    u_max: float = 150.0,
) -> BatesReceipt:
    """Price European call and put options under David S. Bates (1996) SVJ model."""
    clean_ticker = _validate_ticker(ticker)
    s0 = float(spot_price)
    k_strike = float(strike_price)
    t = float(time_to_maturity)
    r = float(risk_free_rate)
    q = float(dividend_yield)

    if s0 <= 0.0 or k_strike <= 0.0:
        raise ValueError(f"Spot and strike must be positive, got S={s0}, K={k_strike}")
    if t <= 0.0:
        raise ValueError(f"Time to maturity must be positive, got {t}")
    if v0 <= 0.0 or theta <= 0.0:
        raise ValueError(f"Initial variance v0 and theta must be positive, got v0={v0}, theta={theta}")
    if kappa <= 0.0:
        raise ValueError(f"Mean-reversion kappa must be positive, got {kappa}")
    if sigma_v <= 0.0:
        raise ValueError(f"Vol-of-vol sigma_v must be positive, got {sigma_v}")
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"Correlation rho must be strictly in (-1, 1), got {rho}")
    if lambda_jump < 0.0:
        raise ValueError(f"Jump intensity lambda must be non-negative, got {lambda_jump}")
    if sigma_jump <= 0.0:
        raise ValueError(f"Jump volatility sigma_jump must be positive, got {sigma_jump}")

    feller_ratio = float((2.0 * kappa * theta) / (sigma_v ** 2))
    feller_satisfied = feller_ratio > 1.0

    # Gauss-Legendre quadrature nodes on [0, u_max]
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    u_vals = 0.5 * u_max * (nodes + 1.0)
    w_vals = 0.5 * u_max * weights

    log_k_over_s = math.log(k_strike / s0)
    exp_minus_rt = math.exp(-r * t)
    exp_minus_qt = math.exp(-q * t)

    int1 = 0.0
    int2 = 0.0

    for u_val, w_val in zip(u_vals, w_vals):
        u = float(u_val)
        w = float(w_val)
        phase = cmath.exp(-1j * u * log_k_over_s)

        # phi(u) for Pi2
        phi2 = _bates_characteristic_function(
            complex(u, 0.0),
            t=t,
            r=r,
            q=q,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            lambda_jump=lambda_jump,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
        )
        integrand2 = ((phi2 * phase) / (1j * u)).real
        int2 += w * integrand2

        # phi(u - i) for Pi1
        phi1 = _bates_characteristic_function(
            complex(u, -1.0),
            t=t,
            r=r,
            q=q,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            lambda_jump=lambda_jump,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
        )
        # Note: phi(-i) = exp((r - q) * T), so integrand factor includes exp(-(r - q) * T)
        integrand1 = ((math.exp(-(r - q) * t) * phi1 * phase) / (1j * u)).real
        int1 += w * integrand1

    pi1 = float(np.clip(0.5 + (1.0 / math.pi) * int1, 0.0, 1.0))
    pi2 = float(np.clip(0.5 + (1.0 / math.pi) * int2, 0.0, 1.0))

    bates_call = float(max(0.0, s0 * exp_minus_qt * pi1 - k_strike * exp_minus_rt * pi2))
    # Put price via Put-Call Parity: P = C - S0 * exp(-q*T) + K * exp(-r*T)
    bates_put = float(max(0.0, bates_call - s0 * exp_minus_qt + k_strike * exp_minus_rt))

    # Benchmark: Pure Heston call price (lambda_jump = 0)
    int1_h = 0.0
    int2_h = 0.0
    for u_val, w_val in zip(u_vals, w_vals):
        u = float(u_val)
        w = float(w_val)
        phase = cmath.exp(-1j * u * log_k_over_s)

        phi2_h = _bates_characteristic_function(
            complex(u, 0.0),
            t=t,
            r=r,
            q=q,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            lambda_jump=0.0,
            mu_jump=0.0,
            sigma_jump=sigma_jump,
        )
        int2_h += w * ((phi2_h * phase) / (1j * u)).real

        phi1_h = _bates_characteristic_function(
            complex(u, -1.0),
            t=t,
            r=r,
            q=q,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            lambda_jump=0.0,
            mu_jump=0.0,
            sigma_jump=sigma_jump,
        )
        int1_h += w * ((math.exp(-(r - q) * t) * phi1_h * phase) / (1j * u)).real

    pi1_h = float(np.clip(0.5 + (1.0 / math.pi) * int1_h, 0.0, 1.0))
    pi2_h = float(np.clip(0.5 + (1.0 / math.pi) * int2_h, 0.0, 1.0))
    heston_call = float(max(0.0, s0 * exp_minus_qt * pi1_h - k_strike * exp_minus_rt * pi2_h))

    jump_contribution = float(bates_call - heston_call)

    if not feller_satisfied:
        regime = "FELLER_VIOLATION_SUB_CRITICAL"
    elif lambda_jump >= 0.40 and mu_jump < -0.02:
        regime = "ASYMMETRIC_CRASH_JUMP_DOMINANT"
    elif abs(jump_contribution) < 0.05:
        regime = "DIFFUSION_DOMINANT_HESTON_REGIME"
    else:
        regime = "BALANCED_STOCHASTIC_VOL_JUMP"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        spot_price=s0,
        strike_price=k_strike,
        time_to_maturity=t,
        bates_call_price=bates_call,
        bates_put_price=bates_put,
        heston_benchmark_call=heston_call,
        jump_contribution=jump_contribution,
        feller_ratio=feller_ratio,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_bates_receipt(payload)

    return BatesReceipt(
        ticker=clean_ticker,
        spot_price=s0,
        strike_price=k_strike,
        time_to_maturity=t,
        risk_free_rate=r,
        dividend_yield=q,
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        lambda_jump=lambda_jump,
        mu_jump=mu_jump,
        sigma_jump=sigma_jump,
        bates_call_price=bates_call,
        bates_put_price=bates_put,
        heston_benchmark_call=heston_call,
        jump_contribution=jump_contribution,
        feller_ratio=feller_ratio,
        feller_satisfied=feller_satisfied,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_bates_profile_from_prices(
    prices: Sequence[float] | np.ndarray,
    *,
    ticker: str = "SPY",
    time_to_maturity: float = 0.50,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> BatesReceipt:
    """Infer Bates SVJ parameters from empirical price history and return an audit receipt."""
    clean_ticker = _validate_ticker(ticker)
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 25:
        raise ValueError(f"At least 25 price observations required, got {len(arr)}")

    spot = float(arr[-1])
    log_returns = np.diff(np.log(arr))
    realized_vol = float(np.std(log_returns, ddof=1) * math.sqrt(252))
    v0_est = max(0.005, realized_vol ** 2)

    # Leverage correlation rho proxy from skewness
    m3 = float(np.mean((log_returns - np.mean(log_returns)) ** 3))
    s3 = float(np.std(log_returns, ddof=1) ** 3)
    sample_skew = (m3 / s3) if s3 > 1e-12 else 0.0
    rho_est = float(np.clip(sample_skew * 0.40, -0.80, 0.80))

    # Jump detection proxy (excess returns beyond 2.5 std)
    z_scores = (log_returns - np.mean(log_returns)) / (np.std(log_returns, ddof=1) + 1e-12)
    jump_mask = np.abs(z_scores) > 2.5
    n_jumps = int(np.sum(jump_mask))
    lambda_est = float(min(2.5, (n_jumps / len(log_returns)) * 252))

    mu_j_est = float(np.mean(log_returns[jump_mask])) if n_jumps > 0 else -0.05
    sigma_j_est = float(np.std(log_returns[jump_mask])) if n_jumps > 1 else 0.12

    return calculate_bates_option_price(
        spot_price=spot,
        strike_price=spot,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        v0=v0_est,
        kappa=2.5,
        theta=v0_est,
        sigma_v=0.35,
        rho=rho_est,
        lambda_jump=max(0.1, lambda_est),
        mu_jump=mu_j_est,
        sigma_jump=max(0.05, sigma_j_est),
        ticker=clean_ticker,
    )


def persist_bates_record(receipt: BatesReceipt, db_path: str = "data/nextomni_stock.db") -> bool:
    """Persist Bates SVJ receipt into SQLite store under parameterized query."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bates_jump_diffusion_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                spot_price REAL NOT NULL,
                strike_price REAL NOT NULL,
                time_to_maturity REAL NOT NULL,
                risk_free_rate REAL NOT NULL,
                dividend_yield REAL NOT NULL,
                v0 REAL NOT NULL,
                kappa REAL NOT NULL,
                theta REAL NOT NULL,
                sigma_v REAL NOT NULL,
                rho REAL NOT NULL,
                lambda_jump REAL NOT NULL,
                mu_jump REAL NOT NULL,
                sigma_jump REAL NOT NULL,
                bates_call_price REAL NOT NULL,
                bates_put_price REAL NOT NULL,
                heston_benchmark_call REAL NOT NULL,
                jump_contribution REAL NOT NULL,
                feller_ratio REAL NOT NULL,
                feller_satisfied INTEGER NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO bates_jump_diffusion_records (
                ticker, spot_price, strike_price, time_to_maturity, risk_free_rate,
                dividend_yield, v0, kappa, theta, sigma_v, rho, lambda_jump,
                mu_jump, sigma_jump, bates_call_price, bates_put_price,
                heston_benchmark_call, jump_contribution, feller_ratio,
                feller_satisfied, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.spot_price,
                receipt.strike_price,
                receipt.time_to_maturity,
                receipt.risk_free_rate,
                receipt.dividend_yield,
                receipt.v0,
                receipt.kappa,
                receipt.theta,
                receipt.sigma_v,
                receipt.rho,
                receipt.lambda_jump,
                receipt.mu_jump,
                receipt.sigma_jump,
                receipt.bates_call_price,
                receipt.bates_put_price,
                receipt.heston_benchmark_call,
                receipt.jump_contribution,
                receipt.feller_ratio,
                int(receipt.feller_satisfied),
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

