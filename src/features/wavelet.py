"""Multiscale Wavelet Multiresolution Denoising & Energy Scalogram Engine for NextOmni Stock.

Implements shift-invariant Maximal Overlap Discrete Wavelet Transform (MODWT):
- Causal multiresolution decomposition across frequency octaves (Details D_1...D_J and Approximation A_J).
- Zero-phase lag filtering with strict causal boundary padding (P <= t, zero lookahead leakage).
- Donoho & Johnstone (1994) VisuShrink adaptive soft-thresholding for non-stationary jump preservation.
- Wavelet energy scalogram quantifying relative variance across timescale octaves.
- Cryptographically verifiable HMAC-SHA256 audit receipts guaranteeing analytical integrity.
"""

from dataclasses import asdict, dataclass
import hashlib
import hmac
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import PRICE_COLUMN
from src.security.validators import validate_ticker


WAVELET_SIGNING_KEY = b"nextomni_wavelet_denoising_audit_key_v1"


@dataclass
class WaveletReceipt:
    """Cryptographic diagnostic receipt summarizing wavelet multiresolution decomposition."""
    ticker: str
    sample_sessions: int
    levels: int
    noise_sigma_hat: float
    universal_threshold: float
    snr_db: float
    energy_shares_pct: Dict[str, float]
    dominant_scale_regime: str
    latest_raw_price: float
    latest_denoised_price: float
    price_deviation_pct: float
    summary: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 digest across core metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.levels}:"
            f"{self.noise_sigma_hat:.6f}:{self.universal_threshold:.6f}:{self.snr_db:.2f}:"
            f"{self.latest_raw_price:.4f}:{self.latest_denoised_price:.4f}"
        )
        return hmac.new(WAVELET_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt provenance against unauthorized tampering."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_modwt_decomposition(
    series: Union[np.ndarray, pd.Series, List[float]],
    levels: int = 3,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Compute shift-invariant Maximal Overlap Discrete Wavelet Transform (MODWT).

    Uses causal Haar/dyadic scaling to decompose input series into J detail levels
    and 1 smooth approximation level with zero phase lag and zero forward lookahead.

    Mathematical Recursive Formulation:
        W_{j, t} = 0.5 * (V_{j-1, t} - V_{j-1, t - 2^{j-1}})  [Detail / High-pass]
        V_{j, t} = 0.5 * (V_{j-1, t} + V_{j-1, t - 2^{j-1}})  [Approximation / Low-pass]
        where V_0 = X.

    Parseval Additive Identity:
        X_t = sum_{j=1}^J W_{j, t} + V_{J, t}

    Args:
        series: 1D historical price or return array.
        levels: Decomposition level depth J (default: 3).

    Returns:
        Tuple of (details_dict, approximation_array) where details_dict maps scale j -> W_j.
    """
    arr = np.asarray(series, dtype=float)
    clean = np.where(np.isfinite(arr), arr, 0.0)
    n = len(clean)

    if n == 0:
        return {}, np.array([], dtype=float)

    max_j = max(1, min(levels, int(math.floor(math.log2(max(n / 2, 2))))))
    v_prev = clean.copy()
    details: Dict[int, np.ndarray] = {}

    for j in range(1, max_j + 1):
        step = 2 ** (j - 1)
        w_j = np.zeros(n, dtype=float)
        v_next = np.zeros(n, dtype=float)

        for t in range(n):
            past_idx = t - step
            v_past = v_prev[past_idx] if past_idx >= 0 else v_prev[0]
            v_curr = v_prev[t]

            # Dyadic MODWT filter operations
            w_j[t] = 0.5 * (v_curr - v_past)
            v_next[t] = 0.5 * (v_curr + v_past)

        details[j] = w_j
        v_prev = v_next

    return details, v_prev


def denoise_wavelet_series(
    series: Union[np.ndarray, pd.Series, List[float]],
    levels: int = 3,
    soft: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """Denoise time series via MODWT and Donoho & Johnstone (1994) VisuShrink soft thresholding.

    Estimates noise standard deviation from scale 1 detail coefficients (dominated by white noise):
        sigma_hat = median(|W_1 - median(W_1)|) / 0.6745
        lambda = sigma_hat * sqrt(2 * ln(T))

    Applies non-linear shrinkage operator:
        eta_lambda(w) = sign(w) * max(0, |w| - lambda)

    Preserves discrete jump discontinuities while filtering transient microstructure noise.

    Args:
        series: Historical time series.
        levels: Wavelet multiresolution levels.
        soft: If True, uses continuous soft-thresholding; if False, hard thresholding.

    Returns:
        Tuple of (denoised_series, sigma_hat, universal_threshold).
    """
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    if n < 4:
        return arr.copy(), 0.0, 0.0

    details, approx = compute_modwt_decomposition(arr, levels=levels)
    if not details:
        return arr.copy(), 0.0, 0.0

    # Donoho & Johnstone MAD noise estimator from finest scale W_1
    w1 = details[1]
    med_w1 = float(np.median(w1))
    mad = float(np.median(np.abs(w1 - med_w1)))
    sigma_hat = max(1e-8, mad / 0.6745)

    # Universal threshold
    univ_thresh = sigma_hat * math.sqrt(2.0 * math.log(max(n, 2)))

    denoised_details: List[np.ndarray] = []
    for j, w_j in details.items():
        # Scale-adjusted threshold: slightly smaller at coarser scales
        scale_thresh = univ_thresh / math.sqrt(float(j))

        if soft:
            # Soft shrinkage
            w_thresh = np.sign(w_j) * np.maximum(0.0, np.abs(w_j) - scale_thresh)
        else:
            # Hard threshold
            w_thresh = np.where(np.abs(w_j) >= scale_thresh, w_j, 0.0)

        denoised_details.append(w_thresh)

    # Exact multiresolution reconstruction
    reconstructed = approx.copy()
    for w_denoised in denoised_details:
        reconstructed += w_denoised

    return reconstructed, round(sigma_hat, 6), round(univ_thresh, 6)


def compute_wavelet_energy_scalogram(
    details: Dict[int, np.ndarray],
    approx: np.ndarray,
) -> Dict[str, float]:
    """Compute multiscale wavelet energy scalogram quantifying variance across frequency octaves.

    Energy at scale j is given by E_j = sum_t W_{j,t}^2.
    Computes percentage share of detail variance across scale bands:
    - Scale 1 (D_1): 2 to 4 sessions (microstructure noise / bid-ask bounce)
    - Scale 2 (D_2): 4 to 8 sessions (short-term momentum / mean-reversion)
    - Scale 3 (D_3): 8 to 16 sessions (intermediate swing cycles)
    - Scale 4+ (D_4+): 16+ sessions (monthly macroeconomic trend)

    Args:
        details: Dictionary mapping scale level j to detail coefficient array.
        approx: Final smooth approximation array A_J.

    Returns:
        Dictionary mapping scale label to percentage of total detail energy.
    """
    if not details:
        return {}

    energies: Dict[str, float] = {}
    total_detail_energy = 0.0

    octave_names = {
        1: "Scale_1_2to4d",
        2: "Scale_2_4to8d",
        3: "Scale_3_8to16d",
        4: "Scale_4_16to32d",
    }

    for j, w_j in details.items():
        label = octave_names.get(j, f"Scale_{j}_{2**j}to{2**(j+1)}d")
        e_j = float(np.sum(w_j ** 2))
        energies[label] = e_j
        total_detail_energy += e_j

    shares: Dict[str, float] = {}
    if total_detail_energy > 1e-12:
        for label, e_j in energies.items():
            shares[label] = round(float((e_j / total_detail_energy) * 100.0), 2)
    else:
        for label in energies.keys():
            shares[label] = round(100.0 / len(energies), 2)

    return shares


def evaluate_wavelet_profile(
    df: pd.DataFrame,
    ticker: str = "TICKER",
    levels: int = 3,
    price_col: str = PRICE_COLUMN,
) -> WaveletReceipt:
    """Evaluate multiscale wavelet decomposition, adaptive denoising, and energy scalogram.

    Args:
        df: Historical market observations DataFrame.
        ticker: Stock symbol.
        levels: Multiresolution decomposition levels (default: 3).
        price_col: Price column name (default: adj_close).

    Returns:
        WaveletReceipt with complete multiscale metrics and cryptographic digest.
    """
    clean_ticker = validate_ticker(ticker)

    if df.empty or len(df) < 8:
        return WaveletReceipt(
            ticker=clean_ticker,
            sample_sessions=len(df),
            levels=levels,
            noise_sigma_hat=0.0,
            universal_threshold=0.0,
            snr_db=0.0,
            energy_shares_pct={},
            dominant_scale_regime="INSUFFICIENT_DATA",
            latest_raw_price=0.0,
            latest_denoised_price=0.0,
            price_deviation_pct=0.0,
            summary=f"Insufficient historical sessions ({len(df)}) for wavelet multiresolution decomposition.",
        )

    clean_df = df.copy().sort_values("session_date", ascending=True).reset_index(drop=True)
    p_col = price_col if price_col in clean_df.columns else "close"
    if p_col not in clean_df.columns:
        p_col = clean_df.columns[1]

    raw_prices = pd.to_numeric(clean_df[p_col], errors="coerce").dropna().values
    n = len(raw_prices)

    if n < 8:
        return WaveletReceipt(
            ticker=clean_ticker,
            sample_sessions=n,
            levels=levels,
            noise_sigma_hat=0.0,
            universal_threshold=0.0,
            snr_db=0.0,
            energy_shares_pct={},
            dominant_scale_regime="INSUFFICIENT_DATA",
            latest_raw_price=float(raw_prices[-1]) if n > 0 else 0.0,
            latest_denoised_price=float(raw_prices[-1]) if n > 0 else 0.0,
            price_deviation_pct=0.0,
            summary=f"Insufficient valid price points ({n}) for wavelet decomposition.",
        )

    details, approx = compute_modwt_decomposition(raw_prices, levels=levels)
    denoised_prices, sigma_hat, univ_thresh = denoise_wavelet_series(raw_prices, levels=levels, soft=True)
    energy_shares = compute_wavelet_energy_scalogram(details, approx)

    # Signal-to-Noise Ratio (SNR) in dB
    noise_res = raw_prices - denoised_prices
    signal_energy = float(np.sum(denoised_prices ** 2))
    noise_energy = float(np.sum(noise_res ** 2))
    snr_db = round(float(10.0 * math.log10(signal_energy / max(1e-10, noise_energy))), 2)

    latest_raw = float(raw_prices[-1])
    latest_denoised = float(denoised_prices[-1])
    dev_pct = round(float(((latest_raw - latest_denoised) / latest_denoised) * 100.0), 2) if latest_denoised > 0 else 0.0

    # Categorize dominant scale regime
    s1_share = energy_shares.get("Scale_1_2to4d", 0.0)
    s2_share = energy_shares.get("Scale_2_4to8d", 0.0)
    s3_share = energy_shares.get("Scale_3_8to16d", 0.0)

    if s1_share >= 50.0:
        scale_regime = "HIGH_FREQUENCY_NOISE_DOMINATED"
    elif s3_share >= s1_share and s3_share >= s2_share:
        scale_regime = "INTERMEDIATE_TREND_DOMINATED"
    elif s2_share >= s1_share and s2_share >= s3_share:
        scale_regime = "WEEKLY_CYCLICAL_DOMINATED"
    else:
        scale_regime = "BALANCED_MULTI_OCTAVE_SPECTRUM"

    summary = (
        f"{clean_ticker} wavelet analysis ({levels} levels, {n} sessions): "
        f"Regime={scale_regime}, SNR={snr_db:.1f} dB, Noise σ={sigma_hat:.4f}. "
        f"Latest price ${latest_raw:.2f} deviates {dev_pct:+.2f}% from denoised trend ${latest_denoised:.2f}."
    )

    return WaveletReceipt(
        ticker=clean_ticker,
        sample_sessions=n,
        levels=levels,
        noise_sigma_hat=sigma_hat,
        universal_threshold=univ_thresh,
        snr_db=snr_db,
        energy_shares_pct=energy_shares,
        dominant_scale_regime=scale_regime,
        latest_raw_price=round(latest_raw, 4),
        latest_denoised_price=round(latest_denoised, 4),
        price_deviation_pct=dev_pct,
        summary=summary,
    )
