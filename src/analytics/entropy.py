"""Market Information Entropy & Ordinal Complexity Engine for NextOmni Stock.

Implements Bandt & Pompe (2002) Permutation Entropy, Richman & Moorman (2000)
Sample Entropy (SampEn), and market predictability regime diagnostics to distinguish
structured alpha states from entropy-dominated random walks.

References:
    Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure
    for time series. Physical Review Letters, 88(17), 174102.
    Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using
    approximate entropy and sample entropy. Am J Physiol Heart Circ Physiol, 278(6), H2039-H2049.
"""

import hashlib
import hmac
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

ENTROPY_SIGNING_KEY = b"nextomni_entropy_secret_audit_key_v1"


@dataclass
class MarketEntropyReceipt:
    """Cryptographic audit receipt for market information entropy profile."""

    ticker: str
    sample_sessions: int
    window: int
    embedding_dim: int
    permutation_entropy: float
    sample_entropy: float
    predictability_index_pct: float
    entropy_regime: str
    mean_permutation_entropy: float
    entropy_history: List[Dict[str, Any]]
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core complexity metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.window}:{self.embedding_dim}:"
            f"{self.permutation_entropy:.4f}:{self.sample_entropy:.4f}:"
            f"{self.predictability_index_pct:.2f}:{self.entropy_regime}"
        )
        return hmac.new(ENTROPY_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt provenance against unauthorized record modification."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def calculate_permutation_entropy(
    series: Union[np.ndarray, List[float], pd.Series],
    m: int = 3,
    tau: int = 1,
) -> float:
    """Calculate Bandt & Pompe (2002) Normalized Permutation Entropy H_P.

    Mathematical Formulation:
        Vectors: x_i = [s_i, s_{i + tau}, ..., s_{i + (m - 1) * tau}]
        Ordinal pattern: pi(x_i) in S_m
        Permutation Shannon Entropy: S[P] = - sum_{j=1}^{m!} p(pi_j) * ln(p(pi_j))
        Normalized Permutation Entropy: H_P = S[P] / ln(m!) in [0.0, 1.0]

    Args:
        series: 1D continuous sequence.
        m: Embedding dimension (default: 3).
        tau: Time delay lag (default: 1).

    Returns:
        Normalized Permutation Entropy in [0.0, 1.0].
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)

    # Required length check: need at least m points
    total_vectors = n - (m - 1) * tau
    if total_vectors <= 0 or m < 2:
        return 1.0

    # Extract delay embedded vectors and map to ordinal ranking permutations
    patterns: Dict[Tuple[int, ...], int] = {}
    for i in range(total_vectors):
        vec = s[i : i + m * tau : tau]
        # Lexicographic tie-breaker via rankdata
        ranks = tuple(stats.rankdata(vec, method="ordinal").astype(int))
        patterns[ranks] = patterns.get(ranks, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / float(total_vectors)

    # Shannon entropy
    entropy = -float(np.sum(probs * np.log(probs)))

    # Normalization by maximum possible entropy ln(m!)
    max_entropy = math.log(math.factorial(m))
    norm_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    return norm_entropy


def calculate_sample_entropy(
    series: Union[np.ndarray, List[float], pd.Series],
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """Calculate Richman & Moorman (2000) Sample Entropy (SampEn).

    Mathematical Formulation:
        Tolerance: r = r_factor * std(series)
        B^m(r): probability that two template vectors of length m match within r
        A^{m+1}(r): probability that they match for length m+1
        SampEn(m, r) = -ln(A^{m+1}(r) / B^m(r))

    Args:
        series: 1D time-series data.
        m: Template length (default: 2).
        r_factor: Tolerance scaling factor against standard deviation (default: 0.2).

    Returns:
        Sample Entropy value >= 0.0.
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < m + 10:
        return 0.0

    sigma = float(np.std(s, ddof=1))
    if sigma <= 1e-12:
        return 0.0

    r = float(r_factor * sigma)

    def _count_matches(dim: int) -> int:
        num_vecs = n - dim + 1
        if num_vecs <= 1:
            return 0
        # Build 2D matrix of shape (num_vecs, dim)
        vecs = np.array([s[i : i + dim] for i in range(num_vecs)])
        matches = 0
        for i in range(num_vecs):
            # Chebyshev distance: max absolute coordinate difference
            dists = np.max(np.abs(vecs[i + 1 :] - vecs[i]), axis=1)
            matches += int(np.sum(dists <= r))
        return matches

    b_matches = _count_matches(m)
    a_matches = _count_matches(m + 1)

    if b_matches == 0 or a_matches == 0:
        return 0.0

    samp_en = -math.log(float(a_matches) / float(b_matches))
    return float(max(0.0, samp_en))


def evaluate_market_entropy_profile(
    ticker: str,
    prices: Union[pd.Series, np.ndarray, List[float]],
    window: int = 60,
    m: int = 3,
) -> MarketEntropyReceipt:
    """Evaluate comprehensive market information entropy and complexity diagnostics.

    Args:
        ticker: Asset ticker symbol.
        prices: Historical price series.
        window: Rolling observation window (default: 60 sessions).
        m: Permutation embedding dimension (default: 3).

    Returns:
        Cryptographically signed MarketEntropyReceipt.
    """
    if isinstance(prices, pd.Series):
        s = prices.dropna().astype(float)
        dates = [str(d) for d in s.index]
    else:
        s = pd.Series(prices).dropna().astype(float)
        dates = [f"t-{i}" for i in range(len(s))]

    n_sessions = len(s)
    min_required = max(30, m * 2)

    if n_sessions < min_required:
        return MarketEntropyReceipt(
            ticker=ticker.upper().strip(),
            sample_sessions=n_sessions,
            window=window,
            embedding_dim=m,
            permutation_entropy=1.0,
            sample_entropy=0.0,
            predictability_index_pct=0.0,
            entropy_regime="INSUFFICIENT_DATA",
            mean_permutation_entropy=1.0,
            entropy_history=[],
            summary=f"Insufficient history ({n_sessions} sessions, minimum {min_required} required) for entropy diagnostics.",
        )

    # Compute trailing rolling permutation entropy
    s_vals = s.values
    win_size = min(window, n_sessions)
    step = 1

    history_records: List[Dict[str, Any]] = []
    pe_values: List[float] = []

    # Sample rolling evaluations across the series
    start_idx = max(0, n_sessions - 60)
    for idx in range(start_idx, n_sessions):
        chunk = s_vals[max(0, idx - win_size + 1) : idx + 1]
        if len(chunk) >= 15:
            pe = calculate_permutation_entropy(chunk, m=m)
            pi = (1.0 - pe) * 100.0
            pe_values.append(pe)
            history_records.append({
                "session_date": dates[idx],
                "price": round(float(s_vals[idx]), 2),
                "permutation_entropy": round(pe, 4),
                "predictability_index": round(pi, 2),
            })

    # Latest window metrics
    latest_window_chunk = s_vals[-win_size:]
    latest_pe = calculate_permutation_entropy(latest_window_chunk, m=m)
    latest_sampen = calculate_sample_entropy(latest_window_chunk, m=2)
    latest_pi = (1.0 - latest_pe) * 100.0

    mean_pe = float(np.mean(pe_values)) if pe_values else latest_pe

    # Complexity & Predictability Regime Categorization
    if latest_pi >= 25.0:
        regime = "HIGH_PREDICTABILITY_ORDER"
        regime_desc = (
            f"Strong structural order (Predictability Index = {latest_pi:.1f}%), "
            "indicating high momentum persistence where directional signals have strong statistical edge"
        )
    elif latest_pi >= 10.0:
        regime = "MODERATE_STRUCTURE"
        regime_desc = (
            f"Moderate market complexity (Predictability Index = {latest_pi:.1f}%), "
            "typical liquid equilibrium with mixed noise and trend signals"
        )
    else:
        regime = "HIGH_ENTROPY_RANDOM_WALK"
        regime_desc = (
            f"High entropy regime (H_P = {latest_pe:.3f}, Predictability Index = {latest_pi:.1f}%), "
            "price trajectory behaves near pure white noise / efficient random walk (advising reduced position sizing)"
        )

    summary = (
        f"{ticker}: Normalized Permutation Entropy is {latest_pe:.3f} and Sample Entropy is {latest_sampen:.3f}. "
        f"{regime_desc}."
    )

    return MarketEntropyReceipt(
        ticker=ticker.upper().strip(),
        sample_sessions=n_sessions,
        window=win_size,
        embedding_dim=m,
        permutation_entropy=round(latest_pe, 4),
        sample_entropy=round(latest_sampen, 4),
        predictability_index_pct=round(latest_pi, 2),
        entropy_regime=regime,
        mean_permutation_entropy=round(mean_pe, 4),
        entropy_history=history_records,
        summary=summary,
    )
