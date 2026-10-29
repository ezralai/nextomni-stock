"""Minimum Redundancy Maximum Relevance (mRMR) Feature Selection Engine for NextOmni Stock.

Implements Peng, Long & Ding's (2005) mRMR feature selection algorithm to identify
the most predictive, non-redundant quantitative signals, suppressing multicollinearity
and out-of-sample forecast variance before training multi-horizon models.

References:
    Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual
    information: criteria of max-dependency, max-relevance, and min-redundancy.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), 1226-1238.
    Ding, C., & Peng, H. (2005). Minimum redundancy feature selection from microarray
    gene expression data. Journal of Bioinformatics and Computational Biology, 3(2), 185-205.
"""

import hashlib
import hmac
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from src.data.db import get_observations
from src.features.builder import ALL_FEATURE_COLUMNS, build_feature_matrix
from src.security.validators import assert_finite_number, validate_ticker

MRMR_SIGNING_KEY = b"nextomni_mrmr_feature_selection_key_v1"


@dataclass
class MRMRStep:
    """Diagnostic record for a single greedy selection step."""

    step: int
    feature: str
    relevance: float
    redundancy: float
    mrmr_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MRMRReceipt:
    """Cryptographic audit receipt for mRMR feature selection."""

    ticker: str
    horizon_sessions: int
    sample_sessions: int
    criterion: str  # 'MID' (Difference) or 'MIQ' (Quotient)
    num_candidates: int
    num_selected: int
    selected_features: List[str]
    relevance_scores: Dict[str, float]
    redundancy_scores: Dict[str, float]
    mrmr_scores: Dict[str, float]
    selection_history: List[Dict[str, Any]]
    mean_candidate_correlation: float
    mean_selected_correlation: float
    redundancy_reduction_pct: float
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core selection metrics."""
        payload = (
            f"{self.ticker}:{self.horizon_sessions}:{self.sample_sessions}:{self.criterion}:"
            f"{','.join(self.selected_features)}:{self.mean_selected_correlation:.4f}:"
            f"{self.redundancy_reduction_pct:.4f}"
        )
        return hmac.new(MRMR_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt authenticity and cryptographic tamper status."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_mrmr_features(
    X: pd.DataFrame,
    y: np.ndarray,
    n_features_to_select: int = 5,
    criterion: str = "MID",
    ticker: str = "GENERIC",
    horizon_sessions: int = 1,
    random_state: int = 42,
) -> MRMRReceipt:
    """Perform greedy forward Minimum Redundancy Maximum Relevance feature selection.

    Args:
        X: DataFrame of candidate feature columns.
        y: Target array (percentage return target r_{t+k}).
        n_features_to_select: Target number of features K to select.
        criterion: 'MID' (Mutual Information Difference) or 'MIQ' (Mutual Information Quotient).
        ticker: Asset ticker symbol for receipt provenance.
        horizon_sessions: Forward horizon for receipt provenance.
        random_state: Seed for mutual information estimation reproducibility.

    Returns:
        MRMRReceipt detailing ranking, relevance, redundancy, and cryptographic proof.
    """
    if X.empty or len(X) < 10:
        raise ValueError("Insufficient feature rows to execute mRMR selection.")

    candidate_cols = list(X.columns)
    num_candidates = len(candidate_cols)
    if num_candidates == 0:
        raise ValueError("Candidate feature matrix has no columns.")

    crit_norm = criterion.upper().strip()
    if crit_norm not in ("MID", "MIQ"):
        raise ValueError(f"Invalid mRMR criterion: {criterion}. Must be 'MID' or 'MIQ'.")

    k_target = max(1, min(n_features_to_select, num_candidates))
    sample_size = len(X)

    # 1. Compute Mutual Information Relevance: I(f_i; y)
    X_vals = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
    y_vals = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    raw_mi = mutual_info_regression(X_vals, y_vals, random_state=random_state)
    # Ensure non-negative
    raw_mi = np.maximum(raw_mi, 0.0)
    relevance_map = {candidate_cols[i]: float(raw_mi[i]) for i in range(num_candidates)}

    # 2. Pairwise Absolute Correlation Matrix as Redundancy Proxy: |corr(f_i, f_j)|
    corr_df = X.corr(method="pearson").abs().fillna(0.0)
    np.fill_diagonal(corr_df.values, 1.0)

    # Calculate baseline mean candidate correlation (upper triangle off-diagonal)
    if num_candidates > 1:
        triu_indices = np.triu_indices(num_candidates, k=1)
        mean_candidate_corr = float(np.mean(corr_df.values[triu_indices]))
    else:
        mean_candidate_corr = 0.0

    # 3. Greedy Forward Selection
    selected: List[str] = []
    unselected = list(candidate_cols)
    history: List[Dict[str, Any]] = []
    redundancy_map: Dict[str, float] = {}
    mrmr_score_map: Dict[str, float] = {}

    # Step 1: Pick feature with highest relevance I(f; y)
    best_first = max(unselected, key=lambda f: relevance_map[f])
    selected.append(best_first)
    unselected.remove(best_first)
    redundancy_map[best_first] = 0.0
    mrmr_score_map[best_first] = relevance_map[best_first]

    history.append(
        MRMRStep(
            step=1,
            feature=best_first,
            relevance=round(relevance_map[best_first], 4),
            redundancy=0.0,
            mrmr_score=round(relevance_map[best_first], 4),
        ).to_dict()
    )

    # Steps 2 to K
    while len(selected) < k_target and unselected:
        step_idx = len(selected) + 1
        best_feat = None
        best_score = -float("inf")
        best_redundancy = 0.0

        for candidate in unselected:
            # Average redundancy with currently selected set: 1/|S| * sum_{s in S} |corr(candidate, s)|
            mean_red = float(np.mean([corr_df.loc[candidate, s] for s in selected]))
            rel = relevance_map[candidate]

            if crit_norm == "MID":
                score = rel - mean_red
            else:  # MIQ
                score = rel / (mean_red + 1e-4)

            if score > best_score:
                best_score = score
                best_feat = candidate
                best_redundancy = mean_red

        if best_feat is None:
            best_feat = unselected[0]
            best_score = 0.0
            best_redundancy = 0.0

        selected.append(best_feat)
        unselected.remove(best_feat)
        redundancy_map[best_feat] = round(best_redundancy, 4)
        mrmr_score_map[best_feat] = round(best_score, 4)

        history.append(
            MRMRStep(
                step=step_idx,
                feature=best_feat,
                relevance=round(relevance_map[best_feat], 4),
                redundancy=round(best_redundancy, 4),
                mrmr_score=round(best_score, 4),
            ).to_dict()
        )

    # 4. Measure Redundancy Reduction in Selected Set
    if len(selected) > 1:
        sel_corr_mat = corr_df.loc[selected, selected].values
        sel_triu = np.triu_indices(len(selected), k=1)
        mean_selected_corr = float(np.mean(sel_corr_mat[sel_triu]))
    else:
        mean_selected_corr = 0.0

    if mean_candidate_corr > 1e-6:
        red_reduction_pct = max(
            0.0,
            round(
                ((mean_candidate_corr - mean_selected_corr) / mean_candidate_corr) * 100.0,
                2,
            ),
        )
    else:
        red_reduction_pct = 0.0

    summary = (
        f"mRMR Feature Selection ({crit_norm}) for {ticker} (Horizon: {horizon_sessions}D, "
        f"{sample_size} sessions): Selected {len(selected)}/{num_candidates} signals: "
        f"[{', '.join(selected)}]. Average collinearity reduced from "
        f"{mean_candidate_corr:.3f} to {mean_selected_corr:.3f} (-{red_reduction_pct:.1f}% redundancy)."
    )

    receipt = MRMRReceipt(
        ticker=ticker,
        horizon_sessions=horizon_sessions,
        sample_sessions=sample_size,
        criterion=crit_norm,
        num_candidates=num_candidates,
        num_selected=len(selected),
        selected_features=selected,
        relevance_scores={f: round(relevance_map[f], 4) for f in selected},
        redundancy_scores=redundancy_map,
        mrmr_scores=mrmr_score_map,
        selection_history=history,
        mean_candidate_correlation=round(mean_candidate_corr, 4),
        mean_selected_correlation=round(mean_selected_corr, 4),
        redundancy_reduction_pct=red_reduction_pct,
        summary=summary,
    )
    return receipt


def evaluate_mrmr_profile(
    ticker: str,
    horizon_sessions: int = 1,
    n_features: int = 5,
    criterion: str = "MID",
    db_path: Optional[Path] = None,
) -> MRMRReceipt:
    """High-level orchestrator: loads data, builds feature matrix, and executes mRMR selection."""
    valid_ticker = validate_ticker(ticker.strip().upper())
    horizon = int(horizon_sessions)
    if horizon not in (1, 5, 21):
        raise ValueError(f"Invalid horizon_sessions: {horizon}. Must be 1, 5, or 21.")

    df_obs = get_observations(valid_ticker, db_path=db_path)
    if df_obs.empty or len(df_obs) < 35:
        raise ValueError(
            f"Insufficient observation history for {valid_ticker} (found {len(df_obs)} sessions). "
            "At least 35 sessions are required for feature matrix generation."
        )

    # Build feature matrix with forward target
    feat_df, cols = build_feature_matrix(
        df_obs,
        include_target=True,
        horizon_sessions=horizon,
        include_microstructure=True,
    )

    feat_df = feat_df.dropna().reset_index(drop=True)
    if feat_df.empty or len(feat_df) < 15:
        raise ValueError("Insufficient non-null feature rows after alignment.")

    # Extract candidate features
    candidate_cols = [c for c in cols if c in feat_df.columns]
    X = feat_df[candidate_cols]

    # Target: expected percentage return r_{t+k} = (P_{t+k} - P_t) / P_t
    p_t = feat_df["close"].values
    p_tk = feat_df["target_price"].values
    y = (p_tk - p_t) / np.maximum(p_t, 1e-8)

    return select_mrmr_features(
        X=X,
        y=y,
        n_features_to_select=n_features,
        criterion=criterion,
        ticker=valid_ticker,
        horizon_sessions=horizon,
    )
