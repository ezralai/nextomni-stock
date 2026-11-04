"""Brad M. Barber & Terrance Odean (2008) / Zhi Da, Joseph Engelberg & Pengjie Gao (2011)
Salience, Abnormal Retail Attention Spike (ASVI) & Buying Pressure Reversal Engine.

Implements the attention-induced retail buying pressure and post-spike mean-reversion framework:
- All That Glitters Attention Hypothesis (Barber & Odean, 2008):
    Retail investors face a massive search problem when buying (choosing from thousands of stocks)
    but not when selling (selling only what they already own, rarely shorting). Consequently,
    attention-grabbing events (abnormal trading volume, extreme 1-day returns, media headlines)
    induce asymmetric net retail buying pressure, driving temporary overpricing.
- Abnormal Search Volume Index (Da, Engelberg & Gao, 2011):
    ASVI_t = ln(SVI_t) - ln(Median(SVI_{t-1 ... t-8}))
    Spikes in retail search attention predict short-term (1-2 week) positive price pressure
    followed by complete long-horizon return reversal as retail euphoria subsides.
- Regimes:
    - UNDER_THE_RADAR_STEADY_COMPOUNDER: Low attention salience (BASS_t <= 0.35 and ASVI_t <= 0.10) (+Alpha)
    - NORMAL_ATTENTION_EQUILIBRIUM: Balanced media/search visibility (0.35 < BASS_t < 0.95)
    - TRANSIENT_SVI_PRICE_PRESSURE_WATCH: Elevated retail attention spike (0.95 <= BASS_t < 1.55 or ASVI_t >= 0.45)
    - GLITTER_ATTENTION_BUYING_CLIMAX_TRAP: Extreme retail FOMO buying climax (BASS_t >= 1.55 or ARRO_t >= 1.85) (-Alpha trap)

References:
    Barber, B. M., & Odean, T. (2008). All that glitters: The effect of attention and news on the
    buying behavior of individual and institutional investors. The Review of Financial Studies, 21(2), 785-818.
    Da, Z., Engelberg, J., & Gao, P. (2011). In search of attention.
    The Journal of Finance, 66(5), 1461-1499.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

ATTENTION_SALIENCE_SIGNING_KEY = b"nextomni_attention_salience_secret_v1"


@dataclass
class AttentionSalienceReceipt:
    """Cryptographic audit receipt for Barber & Odean (2008) & Da et al. (2011) Attention Salience Profile."""

    ticker: str
    latest_price: float
    # Attention & Retail Flow Inputs
    abnormal_svi_log_change: float  # Da-Engelberg-Gao ASVI_t in [-3.0, 5.0]
    abnormal_volume_ratio: float  # Vol_t / Mean(Vol_20) (> 0.0)
    extreme_day_return: float  # Most salient recent daily return in [-0.95, 5.0]
    retail_buy_imbalance: float  # Net retail order buy imbalance in [-1.0, 1.0]
    # Derived Barber-Odean & Da-Engelberg-Gao Metrics
    barber_odean_salience_score: float  # Composite attention salience BASS_t in [0.0, 3.0]
    retail_reversal_overhang_index: float  # Asymmetric buying climax overhang ARRO_t in [-1.5, 3.5]
    short_term_pressure_bps: float  # Projected 1-to-2 week transient price pressure (bps)
    expected_reversal_alpha_pct: float  # Projected annualized post-attention net alpha (%)
    # Flags & Regime
    is_under_the_radar_compounder: bool  # True if BASS_t <= 0.35 and ASVI_t <= 0.10
    is_glitter_buying_climax_trap: bool  # True if BASS_t >= 1.55 or ARRO_t >= 1.85
    regime: str
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core attention salience metrics."""
        payload = (
            f"{self.ticker}:{self.latest_price:.2f}:{self.abnormal_svi_log_change:.4f}:"
            f"{self.abnormal_volume_ratio:.4f}:{self.extreme_day_return:.4f}:"
            f"{self.retail_buy_imbalance:.4f}:{self.barber_odean_salience_score:.4f}:"
            f"{self.retail_reversal_overhang_index:.4f}:{self.short_term_pressure_bps:.2f}:"
            f"{self.expected_reversal_alpha_pct:.2f}:{int(self.is_under_the_radar_compounder)}:"
            f"{int(self.is_glitter_buying_climax_trap)}:{self.regime}"
        )
        return hmac.new(
            ATTENTION_SALIENCE_SIGNING_KEY,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper resistance."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary format for JSON serialization."""
        return asdict(self)


def classify_attention_regime(
    salience_score: float,
    asvi_log_change: float,
    reversal_overhang: float,
) -> str:
    """Classify asset into Barber & Odean (2008) / Da, Engelberg & Gao (2011) Attention Regimes."""
    if salience_score >= 1.55 or reversal_overhang >= 1.85:
        return "GLITTER_ATTENTION_BUYING_CLIMAX_TRAP"
    if salience_score >= 0.95 or asvi_log_change >= 0.45:
        return "TRANSIENT_SVI_PRICE_PRESSURE_WATCH"
    if salience_score <= 0.35 and asvi_log_change <= 0.10:
        return "UNDER_THE_RADAR_STEADY_COMPOUNDER"
    return "NORMAL_ATTENTION_EQUILIBRIUM"


def calculate_attention_salience(
    ticker: str,
    latest_price: float,
    abnormal_svi_log_change: float = 0.04,
    abnormal_volume_ratio: float = 1.05,
    extreme_day_return: float = 0.012,
    retail_buy_imbalance: float = 0.05,
) -> AttentionSalienceReceipt:
    """Calculate Barber & Odean (2008) Salience Score and Da, Engelberg & Gao (2011) ASVI Reversal Overhang.

    Args:
        ticker: Validated stock ticker symbol.
        latest_price: Latest closing stock price ($).
        abnormal_svi_log_change: Log abnormal Search Volume Index ASVI_t in [-3.0, 5.0].
        abnormal_volume_ratio: Current volume divided by 20-day trailing average volume (> 0.0).
        extreme_day_return: Most salient recent 1-day return in [-0.95, 5.0].
        retail_buy_imbalance: Net retail order imbalance in [-1.0, 1.0].

    Returns:
        AttentionSalienceReceipt with BASS_t, ARRO_t, transient pressure, and expected reversal alpha.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(latest_price, "latest_price")
    assert_finite_number(abnormal_svi_log_change, "abnormal_svi_log_change")
    assert_finite_number(abnormal_volume_ratio, "abnormal_volume_ratio")
    assert_finite_number(extreme_day_return, "extreme_day_return")
    assert_finite_number(retail_buy_imbalance, "retail_buy_imbalance")

    if latest_price <= 0.0:
        raise ValueError("latest_price must be strictly positive.")
    if not (-3.0 <= abnormal_svi_log_change <= 5.0):
        raise ValueError("abnormal_svi_log_change must be in [-3.0, 5.0].")
    if abnormal_volume_ratio <= 0.0:
        raise ValueError("abnormal_volume_ratio must be strictly positive.")
    if not (-0.95 <= extreme_day_return <= 5.0):
        raise ValueError("extreme_day_return must be in [-0.95, 5.0].")
    if not (-1.0 <= retail_buy_imbalance <= 1.0):
        raise ValueError("retail_buy_imbalance must be in [-1.0, 1.0].")

    asvi = float(abnormal_svi_log_change)
    vol_ratio = float(abnormal_volume_ratio)
    ext_ret = float(extreme_day_return)
    ret_imb = float(retail_buy_imbalance)

    # 1. Barber & Odean (2008) Three-Pillar Attention Salience Score BASS_t in [0.0, 3.0]
    vol_comp = min(max(vol_ratio - 1.0, 0.0) / 1.5, 2.0)
    ret_comp = min(abs(ext_ret) / 0.04, 2.0)
    svi_comp = min(max(asvi, 0.0) / 0.80, 2.5)
    raw_bass = 0.35 * vol_comp + 0.35 * ret_comp + 0.30 * svi_comp
    bass = float(np.clip(raw_bass, 0.0, 3.0))

    # 2. Asymmetric Retail Buying Pressure & Reversal Overhang Index ARRO_t in [-1.5, 3.5]
    pos_retail = max(ret_imb, 0.0)
    raw_arro = bass * (1.0 + 1.4 * pos_retail) * 1.15
    arro = float(np.clip(raw_arro, -1.5, 3.5))

    # 3. Da, Engelberg & Gao (2011) Transient 1-2 Week Price Pressure (bps) vs. Long-Horizon Reversal Alpha
    transient_bps = float(np.clip(32.0 * max(asvi, 0.0) + 18.0 * pos_retail * bass, 0.0, 180.0))

    regime = classify_attention_regime(bass, asvi, arro)
    is_compounder = bool(regime == "UNDER_THE_RADAR_STEADY_COMPOUNDER")
    is_climax_trap = bool(regime == "GLITTER_ATTENTION_BUYING_CLIMAX_TRAP")

    expected_alpha = float(
        np.clip(3.6 - 4.8 * arro - 1.5 * max(asvi, 0.0), -12.5, 4.5)
    )

    if regime == "UNDER_THE_RADAR_STEADY_COMPOUNDER":
        summary = (
            f"Under-the-Radar Steady Compounder (BASS={bass:.2f}, ASVI={asvi:+.2f}, VolRatio={vol_ratio:.2f}x). "
            f"Absence of retail hype and media salience prevents Barber-Odean buying-pressure overvaluation "
            f"({expected_alpha:+.1f}%/yr)."
        )
    elif regime == "NORMAL_ATTENTION_EQUILIBRIUM":
        summary = (
            f"Normal Attention Equilibrium (BASS={bass:.2f}, ASVI={asvi:+.2f}, Overhang={arro:.2f}). "
            f"Balanced investor visibility with {expected_alpha:+.1f}%/yr expected alpha."
        )
    elif regime == "TRANSIENT_SVI_PRICE_PRESSURE_WATCH":
        summary = (
            f"Transient SVI Price Pressure Watch (BASS={bass:.2f}, ASVI={asvi:+.2f}, "
            f"ShortTermBoost=+{transient_bps:.0f}bps). Rising retail search volume creates short-lived "
            f"upward pressure ahead of mean reversion ({expected_alpha:+.1f}%/yr)."
        )
    else:
        summary = (
            f"Glitter Attention Buying Climax Trap (BASS={bass:.2f}, ASVI={asvi:+.2f}, Overhang={arro:.2f}). "
            f"Extreme Barber-Odean retail attention fever and one-sided buying pressure predict severe "
            f"long-horizon reversal ({expected_alpha:+.1f}%/yr)."
        )

    return AttentionSalienceReceipt(
        ticker=clean_ticker,
        latest_price=round(float(latest_price), 2),
        abnormal_svi_log_change=round(asvi, 4),
        abnormal_volume_ratio=round(vol_ratio, 4),
        extreme_day_return=round(ext_ret, 4),
        retail_buy_imbalance=round(ret_imb, 4),
        barber_odean_salience_score=round(bass, 4),
        retail_reversal_overhang_index=round(arro, 4),
        short_term_pressure_bps=round(transient_bps, 2),
        expected_reversal_alpha_pct=round(expected_alpha, 2),
        is_under_the_radar_compounder=is_compounder,
        is_glitter_buying_climax_trap=is_climax_trap,
        regime=regime,
        summary=summary,
    )


def evaluate_attention_salience_profile(
    ticker: str,
    db_path: Optional[Union[str, Path]] = None,
    abnormal_svi_log_change: Optional[float] = None,
    abnormal_volume_ratio: Optional[float] = None,
    extreme_day_return: Optional[float] = None,
    retail_buy_imbalance: Optional[float] = None,
) -> AttentionSalienceReceipt:
    """Evaluate Barber & Odean (2008) & Da et al. (2011) Attention Salience profile from SQLite OHLCV and overrides."""
    clean_ticker = validate_ticker(ticker)

    latest_price = 100.0
    inferred_asvi = 0.04
    inferred_vol_ratio = 1.05
    inferred_ext_ret = 0.012
    inferred_ret_imb = 0.05

    try:
        obs = db.get_observations(clean_ticker, db_path=db_path)
        if obs is not None and not obs.empty and "close" in obs.columns:
            closes = [
                float(x)
                for x in obs["close"].tolist()
                if math.isfinite(float(x)) and float(x) > 0
            ]
            if closes:
                latest_price = closes[-1]
            if len(closes) >= 20:
                arr_c = np.asarray(closes[-20:], dtype=float)
                rets = np.diff(arr_c) / arr_c[:-1]
                if len(rets) > 0:
                    max_idx = int(np.argmax(np.abs(rets[-5:])))
                    inferred_ext_ret = float(
                        np.clip(rets[-5:][max_idx], -0.80, 2.00)
                    )
                if "volume" in obs.columns:
                    vols = [
                        max(float(v), 1.0)
                        for v in obs["volume"].tolist()[-20:]
                    ]
                    mean_v = float(np.mean(vols[:-1])) if len(vols) > 1 else vols[-1]
                    inferred_vol_ratio = float(
                        np.clip(vols[-1] / max(mean_v, 1.0), 0.20, 8.0)
                    )
                    inferred_asvi = float(
                        np.clip(math.log(max(inferred_vol_ratio, 0.25)), -1.5, 2.5)
                    )
    except Exception:
        latest_price = 100.0

    asvi_val = (
        float(abnormal_svi_log_change)
        if abnormal_svi_log_change is not None
        else inferred_asvi
    )
    vol_val = (
        float(abnormal_volume_ratio)
        if abnormal_volume_ratio is not None
        else inferred_vol_ratio
    )
    ext_val = (
        float(extreme_day_return)
        if extreme_day_return is not None
        else inferred_ext_ret
    )
    imb_val = (
        float(retail_buy_imbalance)
        if retail_buy_imbalance is not None
        else inferred_ret_imb
    )

    return calculate_attention_salience(
        ticker=clean_ticker,
        latest_price=latest_price,
        abnormal_svi_log_change=asvi_val,
        abnormal_volume_ratio=vol_val,
        extreme_day_return=ext_val,
        retail_buy_imbalance=imb_val,
    )

