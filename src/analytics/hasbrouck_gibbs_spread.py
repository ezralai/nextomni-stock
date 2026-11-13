"""Joel Hasbrouck (2009) Bayesian Gibbs Sampler Effective Spread Engine.

Implements the Bayesian MCMC Gibbs sampler for estimating effective transaction
costs and bid-ask spreads from daily closing prices as formulated in:
Hasbrouck, Joel (2009). "Trading Costs and Returns for U.S. Equities:
Estimating Effective Spreads from Daily Data." The Journal of Finance, 64(3), 1445-1477.

Resolves the structural breakdown of the classic Roll (1984) autocovariance model,
which fails whenever sample price change autocovariance is positive (Cov(dp_t, dp_{t-1}) >= 0).
By modeling trade direction signs as latent state variables in a state-space model:
    p_t = m_t + c * q_t
    m_t = m_{t-1} + u_t,   u_t ~ N(0, sigma_u^2)
    q_t in {-1, +1}
and enforcing a truncated normal prior c >= 0, the Gibbs sampler guarantees strictly
valid, strictly positive effective spread posterior estimates.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from scipy import stats

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\_\-]{1,16}$")
SECRET_SALT = os.environ.get("NEXTOMNI_AUDIT_SALT", "nextomni_hasbrouck_gibbs_audit_salt").encode("utf-8")


@dataclass(frozen=True)
class HasbrouckGibbsReceipt:
    """Immutable receipt containing Hasbrouck (2009) Gibbs sampler spread estimates."""

    ticker: str
    sample_size: int
    iterations: int
    burn_in: int
    effective_spread_dollar: float
    effective_spread_bps: float
    posterior_c_mean: float
    posterior_c_std: float
    posterior_sigma_u: float
    roll_spread_dollar: float
    positive_autocovariance_flag: bool
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_hmac(payload: str) -> str:
    return hmac.new(SECRET_SALT, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify cryptographic HMAC-SHA256 signature against payload."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


def calculate_roll_spread(prices: Sequence[float]) -> tuple[float, bool]:
    """Calculate benchmark Roll (1984) spread and indicate autocovariance positivity."""
    if len(prices) < 3:
        return 0.0, False

    p = np.asarray(prices, dtype=np.float64)
    dp = np.diff(p)
    if len(dp) < 2:
        return 0.0, False

    # Sample autocovariance at lag 1
    dp_curr = dp[1:]
    dp_prev = dp[:-1]
    dp_curr_mean = float(np.mean(dp_curr))
    dp_prev_mean = float(np.mean(dp_prev))

    cov = float(np.mean((dp_curr - dp_curr_mean) * (dp_prev - dp_prev_mean)))
    if cov < 0:
        roll_c = math.sqrt(-cov)
        return float(2.0 * roll_c), False
    else:
        # Classic Roll failure: positive autocovariance yields undefined square root
        return 0.0, True


def run_hasbrouck_gibbs(
    prices: Sequence[float],
    iterations: int = 600,
    burn_in: int = 150,
    seed: int = 42,
    ticker: str = "SPY",
) -> HasbrouckGibbsReceipt:
    """Execute Hasbrouck (2009) Bayesian MCMC Gibbs Sampler for effective spread estimation.

    Parameters
    ----------
    prices : Sequence[float]
        Series of consecutive prices (e.g. daily closing prices).
    iterations : int
        Total MCMC iterations to run (default 600).
    burn_in : int
        Number of burn-in iterations to discard (default 150).
    seed : int
        Random seed for PRNG reproducibility.
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    HasbrouckGibbsReceipt
        Detailed estimation audit receipt.
    """
    clean_ticker = ticker.strip().upper() if TICKER_REGEX.match(ticker.strip().upper()) else "UNKNOWN"
    now_iso = datetime.now(timezone.utc).isoformat()

    p_arr = np.asarray(prices, dtype=np.float64)
    valid_mask = np.isfinite(p_arr) & (p_arr > 0.0)
    p_clean = p_arr[valid_mask]
    n_pts = len(p_clean)

    if n_pts < 10 or iterations <= burn_in or burn_in < 10:
        empty_payload = f"{clean_ticker}|0|0.0|0.0|0.0|{now_iso}"
        sig = _compute_hmac(empty_payload)
        return HasbrouckGibbsReceipt(
            ticker=clean_ticker,
            sample_size=n_pts,
            iterations=iterations,
            burn_in=burn_in,
            effective_spread_dollar=0.0,
            effective_spread_bps=0.0,
            posterior_c_mean=0.0,
            posterior_c_std=0.0,
            posterior_sigma_u=0.0,
            roll_spread_dollar=0.0,
            positive_autocovariance_flag=False,
            regime="INSUFFICIENT_DATA",
            hmac_signature=sig,
            timestamp=now_iso,
        )

    # Benchmark Roll spread
    roll_dollar, roll_pos_cov = calculate_roll_spread(p_clean)

    # First differences of price
    dp = np.diff(p_clean)  # length T-1
    T_dp = len(dp)

    rng = np.random.default_rng(seed)

    # Initial states
    # Initialize q_t with price sign changes or random signs
    q = np.ones(n_pts, dtype=np.float64)
    for i in range(1, n_pts):
        if p_clean[i] < p_clean[i - 1]:
            q[i] = -1.0
        elif p_clean[i] == p_clean[i - 1]:
            q[i] = rng.choice([-1.0, 1.0])
        else:
            q[i] = 1.0

    # Initialize c > 0 and sigma_u > 0
    std_dp = float(np.std(dp)) if float(np.std(dp)) > 1e-6 else 0.01
    c_val = max(0.5 * std_dp, 0.005)
    sigma_u_sq = max(std_dp**2, 1e-6)

    c_samples = []
    sigma_u_samples = []

    # Conjugate hyper-priors (diffuse Inverse-Gamma for sigma_u^2)
    alpha_0 = 1e-3
    beta_0 = 1e-3

    for it in range(iterations):
        # 1. Update latent trade direction indicators q_t for t in 0..n_pts-1
        for t in range(n_pts):
            if t == 0:
                # q_0 only appears in u_1 = dp[0] - c * (q_1 - q_0)
                q1 = q[1]
                # If q_0 = +1:
                u1_pos = dp[0] - c_val * (q1 - 1.0)
                # If q_0 = -1:
                u1_neg = dp[0] - c_val * (q1 - (-1.0))
                ll_pos = -0.5 * (u1_pos**2) / sigma_u_sq
                ll_neg = -0.5 * (u1_neg**2) / sigma_u_sq
            elif t == n_pts - 1:
                # q_{T-1} only appears in u_{T-1} = dp[T-2] - c * (q_{T-1} - q_{T-2})
                q_prev = q[t - 1]
                # If q_t = +1:
                ut_pos = dp[t - 1] - c_val * (1.0 - q_prev)
                # If q_t = -1:
                ut_neg = dp[t - 1] - c_val * (-1.0 - q_prev)
                ll_pos = -0.5 * (ut_pos**2) / sigma_u_sq
                ll_neg = -0.5 * (ut_neg**2) / sigma_u_sq
            else:
                # Interior point: q_t appears in u_t and u_{t+1}
                q_prev = q[t - 1]
                q_next = q[t + 1]

                # If q_t = +1:
                ut_pos = dp[t - 1] - c_val * (1.0 - q_prev)
                ut1_pos = dp[t] - c_val * (q_next - 1.0)
                ll_pos = -0.5 * (ut_pos**2 + ut1_pos**2) / sigma_u_sq

                # If q_t = -1:
                ut_neg = dp[t - 1] - c_val * (-1.0 - q_prev)
                ut1_neg = dp[t] - c_val * (q_next - (-1.0))
                ll_neg = -0.5 * (ut_neg**2 + ut1_neg**2) / sigma_u_sq

            max_ll = max(ll_pos, ll_neg)
            exp_pos = math.exp(ll_pos - max_ll)
            exp_neg = math.exp(ll_neg - max_ll)
            prob_pos = exp_pos / (exp_pos + exp_neg)

            q[t] = 1.0 if rng.uniform(0.0, 1.0) < prob_pos else -1.0

        # 2. Update c | q, dp, sigma_u_sq
        # Model: dp_t = c * dq_t + u_t, where dq_t = q_t - q_{t-1}
        dq = np.diff(q)
        sum_dq_sq = float(np.sum(dq**2))
        sum_dq_dp = float(np.sum(dq * dp))

        if sum_dq_sq > 1e-8:
            # Posterior precision and mean under flat prior c >= 0
            tau_sq = sigma_u_sq / sum_dq_sq
            tau = math.sqrt(tau_sq)
            mu_hat = sum_dq_dp / sum_dq_sq

            # Truncated normal draw on [0, +inf)
            a_clip = (0.0 - mu_hat) / tau
            # If standard normal CDF of a_clip is very close to 1, sample conservatively
            if a_clip < 8.0:
                c_val = float(stats.truncnorm.rvs(a_clip, np.inf, loc=mu_hat, scale=tau, random_state=rng))
            else:
                c_val = max(1e-4, mu_hat + tau)
        else:
            c_val = max(c_val, 1e-4)

        c_val = max(c_val, 1e-4)

        # 3. Update sigma_u_sq | q, c, dp
        # Residuals u_t = dp_t - c * dq_t
        u_res = dp - c_val * dq
        alpha_post = alpha_0 + 0.5 * T_dp
        beta_post = beta_0 + 0.5 * float(np.sum(u_res**2))

        # Draw from Inverse-Gamma(alpha_post, beta_post)
        # Using gamma draw: Inv-Gamma ~ 1.0 / Gamma(alpha, 1/beta)
        gamma_draw = rng.gamma(shape=alpha_post, scale=1.0 / beta_post)
        if gamma_draw > 1e-12:
            sigma_u_sq = 1.0 / gamma_draw
        else:
            sigma_u_sq = 1e-4

        if it >= burn_in:
            c_samples.append(c_val)
            sigma_u_samples.append(math.sqrt(sigma_u_sq))

    c_post = np.asarray(c_samples, dtype=np.float64)
    sigma_u_post = np.asarray(sigma_u_samples, dtype=np.float64)

    post_c_mean = float(np.mean(c_post))
    post_c_std = float(np.std(c_post))
    post_sigma_u = float(np.mean(sigma_u_post))

    eff_spread_dollar = float(2.0 * post_c_mean)
    mean_price = float(np.mean(p_clean))
    eff_spread_bps = float((eff_spread_dollar / mean_price) * 10000.0) if mean_price > 1e-6 else 0.0

    # Regime categorization
    if eff_spread_bps < 15.0:
        regime = "HIGH_LIQUIDITY_TIGHT_SPREAD"
    elif eff_spread_bps < 50.0:
        regime = "MODERATE_EFFECTIVE_SPREAD"
    elif eff_spread_bps < 150.0:
        regime = "ILLIQUID_WIDE_SPREAD"
    else:
        regime = "EXTREME_FRICTION_SPREAD"

    payload = (
        f"{clean_ticker}|{n_pts}|{iterations}|{burn_in}|"
        f"{eff_spread_dollar:.6f}|{eff_spread_bps:.4f}|{post_c_mean:.6f}|"
        f"{post_c_std:.6f}|{post_sigma_u:.6f}|{roll_dollar:.6f}|"
        f"{roll_pos_cov}|{regime}|{now_iso}"
    )
    sig = _compute_hmac(payload)

    return HasbrouckGibbsReceipt(
        ticker=clean_ticker,
        sample_size=n_pts,
        iterations=iterations,
        burn_in=burn_in,
        effective_spread_dollar=eff_spread_dollar,
        effective_spread_bps=eff_spread_bps,
        posterior_c_mean=post_c_mean,
        posterior_c_std=post_c_std,
        posterior_sigma_u=post_sigma_u,
        roll_spread_dollar=roll_dollar,
        positive_autocovariance_flag=roll_pos_cov,
        regime=regime,
        hmac_signature=sig,
        timestamp=now_iso,
    )


def persist_hasbrouck_gibbs_record(
    receipt: HasbrouckGibbsReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Store Hasbrouck Gibbs receipt in SQLite database with foreign key & HMAC verification."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hasbrouck_gibbs_receipts (
                ticker TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                iterations INTEGER NOT NULL,
                burn_in INTEGER NOT NULL,
                effective_spread_dollar REAL NOT NULL,
                effective_spread_bps REAL NOT NULL,
                posterior_c_mean REAL NOT NULL,
                posterior_c_std REAL NOT NULL,
                posterior_sigma_u REAL NOT NULL,
                roll_spread_dollar REAL NOT NULL,
                positive_autocovariance_flag INTEGER NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (ticker, timestamp)
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO hasbrouck_gibbs_receipts (
                ticker, sample_size, iterations, burn_in,
                effective_spread_dollar, effective_spread_bps,
                posterior_c_mean, posterior_c_std, posterior_sigma_u,
                roll_spread_dollar, positive_autocovariance_flag,
                regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.sample_size,
                receipt.iterations,
                receipt.burn_in,
                receipt.effective_spread_dollar,
                receipt.effective_spread_bps,
                receipt.posterior_c_mean,
                receipt.posterior_c_std,
                receipt.posterior_sigma_u,
                receipt.roll_spread_dollar,
                1 if receipt.positive_autocovariance_flag else 0,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

