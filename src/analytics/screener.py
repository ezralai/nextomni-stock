"""Cross-sectional alpha screener and multi-asset ranking engine for NextOmni Stock.

Ranks the synchronized asset universe across expected returns, directional win probability,
Cornish-Fisher downside tail risk, and risk-adjusted Sharpe ratios.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.analytics.probability import calculate_directional_probability
from src.analytics.risk import calculate_var_cornish_fisher
from src.config import PRICE_COLUMN
from src.data.db import get_all_stocks, get_observations
from src.features.builder import build_feature_matrix
from src.models.base_model import BaseForecaster
from src.models.ensemble import ConvexEnsembleForecaster
from src.models.gb_model import GradientBoostingForecaster
from src.models.linear_model import LinearRegressionForecaster
from src.security.validators import validate_exchange, validate_model_family


@dataclass
class ScreenerRecord:
    """Individual asset ranking metric record."""
    rank: int
    ticker: str
    exchange: str
    company_name: str
    currency: str
    latest_price: float
    expected_return_pct: float
    directional_win_probability: float
    conviction_label: str
    cornish_fisher_var_pct: float
    predicted_sharpe: float
    horizon_sessions: int
    model_family: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def screen_asset_universe(
    db_path: Optional[Path] = None,
    exchange: Optional[str] = None,
    horizon_sessions: int = 5,
    model_family: str = "ensemble",
    min_observations: int = 40,
) -> List[ScreenerRecord]:
    """Evaluate and rank all active assets in the database cross-sectionally.

    Args:
        db_path: Optional SQLite database path.
        exchange: Optional MIC filter ('XNYS', 'XKLS').
        horizon_sessions: Forecasting horizon (1, 5, or 21 sessions).
        model_family: Forecaster algorithm ('ensemble', 'linear', 'gradient_boosting').
        min_observations: Minimum historical sessions required to evaluate asset.

    Returns:
        List of ScreenerRecord objects sorted by expected_return_pct descending.
    """
    valid_exchange = validate_exchange(exchange) if exchange else None
    valid_family = validate_model_family(model_family)

    stocks_df = get_all_stocks(exchange=valid_exchange, db_path=db_path)
    if stocks_df.empty:
        return []

    records: List[Dict[str, Any]] = []

    for _, row in stocks_df.iterrows():
        ticker = str(row["ticker"])
        exc = str(row["exchange"])
        comp_name = str(row.get("company_name", ticker))
        curr = str(row.get("currency", "USD"))

        obs_df = get_observations(ticker, db_path=db_path)
        if len(obs_df) < min_observations:
            continue

        sorted_obs = obs_df.sort_values("session_date", ascending=True).reset_index(drop=True)
        latest_price = float(sorted_obs.iloc[-1][PRICE_COLUMN])

        # Build feature matrix
        feat_df, feat_cols = build_feature_matrix(
            sorted_obs, include_target=True, horizon_sessions=horizon_sessions
        )
        valid_df = feat_df.dropna(subset=feat_cols).copy().reset_index(drop=True)
        train_data = valid_df.dropna(subset=["target_price"]).copy().reset_index(drop=True)

        if len(train_data) < 25:
            continue

        # Instantiate model family
        if valid_family == "linear":
            model: BaseForecaster = LinearRegressionForecaster(alpha=1.0)
        elif valid_family == "gradient_boosting":
            model = GradientBoostingForecaster(max_depth=2, max_iter=30, random_state=42)
        else:
            model = ConvexEnsembleForecaster()

        # Fit model
        X_tr = train_data[feat_cols]
        y_tr = train_data["target_price"]
        p_tr = train_data[PRICE_COLUMN]
        model.fit(X_tr, y_tr, p_tr, fit_quantiles=False)

        # Predict next horizon price
        latest_features = valid_df.iloc[-1:][feat_cols]
        pred_price = float(model.predict(latest_features, latest_price)[0])
        exp_ret = (pred_price - latest_price) / latest_price
        exp_ret_pct = round(exp_ret * 100.0, 2)

        # Directional Win Probability
        res_std = getattr(model, "residual_std_", 0.02)
        win_prob, conviction = calculate_directional_probability(
            expected_return=exp_ret,
            residual_std=res_std if res_std > 0 else 0.02,
        )

        # 95% Cornish-Fisher VaR downside risk
        trailing_returns = sorted_obs[PRICE_COLUMN].pct_change().dropna().tail(60).tolist()
        cf_var_pct = round(
            float(calculate_var_cornish_fisher(trailing_returns, confidence_level=0.95)) * 100.0, 2
        )

        # Predicted Sharpe / Conviction-to-Risk Ratio
        safe_std = max(res_std, 1e-4)
        pred_sharpe = round(float(exp_ret / safe_std), 3)

        records.append({
            "ticker": ticker,
            "exchange": exc,
            "company_name": comp_name,
            "currency": curr,
            "latest_price": round(latest_price, 2),
            "expected_return_pct": exp_ret_pct,
            "directional_win_probability": win_prob,
            "conviction_label": conviction,
            "cornish_fisher_var_pct": cf_var_pct,
            "predicted_sharpe": pred_sharpe,
            "horizon_sessions": horizon_sessions,
            "model_family": valid_family,
        })

    # Sort descending by expected return
    records.sort(key=lambda x: x["expected_return_pct"], reverse=True)

    ranked_results: List[ScreenerRecord] = []
    for rank_idx, r in enumerate(records, start=1):
        ranked_results.append(
            ScreenerRecord(
                rank=rank_idx,
                ticker=r["ticker"],
                exchange=r["exchange"],
                company_name=r["company_name"],
                currency=r["currency"],
                latest_price=r["latest_price"],
                expected_return_pct=r["expected_return_pct"],
                directional_win_probability=r["directional_win_probability"],
                conviction_label=r["conviction_label"],
                cornish_fisher_var_pct=r["cornish_fisher_var_pct"],
                predicted_sharpe=r["predicted_sharpe"],
                horizon_sessions=r["horizon_sessions"],
                model_family=r["model_family"],
            )
        )

    return ranked_results
