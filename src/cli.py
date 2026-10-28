"""Command-line interface for NextOmni Stock (nextomni-stock).

Allows syncing historical stock data, inspecting database state,
running quantitative analytics, training multi-horizon models,
generating forecast predictions, and monitoring live market feeds.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

from src.config import INITIAL_TICKERS, SUPPORTED_EXCHANGES
from src.analytics.probability import calculate_directional_probability
from src.data.db import get_connection, get_observations, init_db, prune_raw_payloads, vacuum_database
from src.data.pipeline import DataPipeline
from src.security.validators import (
    SUPPORTED_MODEL_FAMILIES,
    assert_safe_model_path,
    sanitize_path_component,
    validate_exchange,
    validate_model_family,
    validate_session_date,
    validate_ticker,
)


def sync_command(args: argparse.Namespace) -> None:
    """Sync historical data for a given ticker or all default initial tickers."""
    pipeline = DataPipeline()

    if args.all:
        for exchange, tickers in INITIAL_TICKERS.items():
            for ticker in tickers:
                print(f"Syncing default ticker {ticker} ({exchange})...")
                res = pipeline.sync_stock(
                    ticker=ticker,
                    exchange=exchange,
                    start_date=args.start,
                    end_date=args.end,
                )
                print(
                    f"  -> {ticker}: Synced {res['sessions_synced']} sessions. "
                    f"Missing calendar sessions: {res['missing_calendar_sessions_count']}"
                )
        return

    if not args.ticker or not args.exchange:
        print("Error: --ticker and --exchange are required unless --all is specified.")
        sys.exit(1)

    ticker = validate_ticker(args.ticker)
    exchange = validate_exchange(args.exchange)
    start_date = validate_session_date(args.start)
    end_date = validate_session_date(args.end)

    print(f"Syncing {ticker} on {exchange} from {start_date} to {end_date}...")
    result = pipeline.sync_stock(
        ticker=ticker,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        company_name=args.name,
    )
    print("\nSync Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


def status_command(args: argparse.Namespace) -> None:
    """Display current database summary: stocks, observation counts, and date ranges."""
    init_db()
    conn = get_connection()

    stocks = conn.execute("SELECT * FROM stocks;").fetchall()
    if not stocks:
        print("No stocks registered in database yet. Run 'sync' to import data.")
        conn.close()
        return

    print("\nNextOmni Stock Database Status:")
    print(f"{'Ticker':<10} {'Exchange':<8} {'Currency':<8} {'Sessions':<10} {'Earliest':<12} {'Latest':<12}")
    print("-" * 65)

    for s in stocks:
        ticker = s["ticker"]
        stats = conn.execute(
            """
            SELECT COUNT(*) as cnt, MIN(session_date) as min_date, MAX(session_date) as max_date
            FROM market_observations
            WHERE ticker = ?;
            """,
            (ticker,),
        ).fetchone()

        cnt = stats["cnt"] if stats else 0
        min_date = stats["min_date"] if stats and stats["min_date"] else "N/A"
        max_date = stats["max_date"] if stats and stats["max_date"] else "N/A"

        print(
            f"{ticker:<10} {s['exchange']:<8} {s['currency']:<8} {cnt:<10} {min_date:<12} {max_date:<12}"
        )
    conn.close()


def analyze_command(args: argparse.Namespace) -> None:
    """Run exploratory data analysis and export interactive visual report."""
    from src.analytics.eda import export_eda_report

    ticker = validate_ticker(args.ticker)
    print(f"\nRunning Exploratory Data Analysis for {ticker}...")
    try:
        stats, report_path = export_eda_report(ticker=ticker)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

    print("\n--- Summary Statistics ---")
    print(f"  Ticker:                 {stats.ticker} ({stats.exchange})")
    print(f"  Period:                 {stats.start_date} to {stats.end_date} ({stats.total_sessions} sessions)")
    print(f"  Price Range:            {stats.min_price} to {stats.max_price} (Mean: {stats.mean_price})")
    print(f"  First -> Latest Close:  {stats.first_price} -> {stats.last_price}")
    print(f"  Cumulative Return:      {stats.cumulative_return_pct}%")
    print(f"  Annualized Return:      {stats.annualized_return_pct}%")
    print(f"  Annualized Volatility:  {stats.annualized_volatility_pct}%")
    print(f"  Max Drawdown:           {stats.max_drawdown_pct}%")
    print(f"  Return Skewness:        {stats.skewness}")
    print(f"  Return Kurtosis:        {stats.kurtosis}")
    print(f"  Average Daily Volume:   {int(stats.avg_daily_volume):,}")

    try:
        from src.analytics.regime import detect_market_regime
        df_obs = get_observations(ticker)
        if len(df_obs) >= 20:
            regime_rec = detect_market_regime(df_obs)
            print("\n--- Market Regime Diagnostic ---")
            print(f"  Quantitative Regime:    {regime_rec.regime} [{regime_rec.regime_label}]")
            print(f"  Macro Sentiment Score:  {regime_rec.sentiment_score:+.3f} (Range [-1.0, +1.0])")
            print(f"  Volatility State:       {regime_rec.volatility_regime} ({regime_rec.rolling_vol_annualized:.1f}% Ann, {regime_rec.vol_percentile_1y:.1f}% 1Y Percentile)")
            print(f"  Trend State:            {regime_rec.trend_regime} (SMA50: {regime_rec.sma_50:.2f}" + (f", SMA200: {regime_rec.sma_200:.2f})" if regime_rec.sma_200 else ")"))
    except Exception:
        pass

    print(f"\nInteractive visual report generated at: {report_path.resolve()}\n")


def forecast_baseline_command(args: argparse.Namespace) -> None:
    """Generate persistence baseline forecast for next trading session and evaluate test performance."""
    from src.data.db import get_observations
    from src.models.baseline import PersistenceBaselineModel

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = validate_exchange(args.exchange) if args.exchange else df["exchange"].iloc[0]
    model = PersistenceBaselineModel()

    # 1. Evaluate historical performance
    eval_result = model.evaluate(df, horizon_sessions=1, train_ratio=0.8)
    metrics = eval_result["metrics"]

    # 2. Generate future forecast
    future = model.generate_future_forecast(ticker, exchange, horizon_sessions=1)

    print(f"\n=======================================================")
    print(f"  NextOmni Stock: Next-Session Persistence Baseline")
    print(f"=======================================================")
    print(f"  Ticker:                 {future['ticker']} ({future['exchange']})")
    print(f"  Forecast Horizon:       {future['forecast_horizon']} (Next Trading Session)")
    print(f"  Forecast Origin Date:   {future['forecast_origin_date']}")
    print(f"  Origin Observed Price:  {future['forecast_origin_price']:.4f} {future['currency']}")
    print(f"  Target Session Date:    {future['target_session_date']}")
    print(f"  Predicted Price:        {future['predicted_price']:.4f} {future['currency']}")
    print(f"  Model Name / Version:   {future['model_name']} (v{future['model_version']})")
    print(f"  Forecast Status:        {future['status']}")

    print(f"\n--- Out-of-Sample Historical Test Performance (Horizon = 1 Session) ---")
    print(f"  Test Period:            {eval_result['test_period_start']} to {eval_result['test_period_end']} ({eval_result['test_sessions_count']} sessions)")
    print(f"  Mean Absolute Error:    {metrics['mae']} {future['currency']}")
    print(f"  Root Mean Squared Error:{metrics['rmse']} {future['currency']}")
    print(f"  Mean Absolute Pct Error:{metrics['mape_pct']}%")
    print(f"  Directional Accuracy:   {metrics['directional_accuracy_pct']}%")
    print(f"\nEducational Context:\n  {eval_result['educational_note']}\n")


def train_linear_command(args: argparse.Namespace) -> None:
    """Execute walk-forward validation and train linear regression model."""
    from src.data.db import get_observations
    from src.evaluation.validation import WalkForwardValidator
    from src.features.builder import build_feature_matrix
    from src.models.linear_model import LinearRegressionForecaster

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    currency = "USD" if df["exchange"].iloc[0] == "XNYS" else "MYR"
    validator = WalkForwardValidator(min_train_sessions=args.min_train, step_size=1)

    print(f"\nRunning Chronological Walk-Forward Validation for {ticker}...")
    result = validator.evaluate(df, horizon_sessions=1, alpha=args.alpha)

    m_mod = result["model_metrics"]
    m_base = result["baseline_metrics"]

    print("\n" + "=" * 75)
    print(f"  NextOmni Stock: Linear Regression vs Persistence Baseline (Walk-Forward)")
    print("=" * 75)
    print(f"  Ticker:               {result['ticker']} ({result['exchange']})")
    print(f"  Evaluation Window:    {result['evaluation_period_start']} to {result['evaluation_period_end']}")
    print(f"  Walk-Forward Steps:   {result['walk_forward_steps']} chronological out-of-sample steps")
    print("-" * 75)
    print(f"{'Metric':<25} {'Linear Model':<20} {'Persistence Baseline':<20} {'Improvement':<12}")
    print("-" * 75)
    mod_mape_str = f"{m_mod['mape_pct']:.2f}%"
    base_mape_str = f"{m_base['mape_pct']:.2f}%"
    mod_da_str = f"{m_mod['directional_accuracy_pct']:.2f}%"
    base_da_str = f"{m_base['directional_accuracy_pct']:.2f}%"

    print(f"{'MAE (' + currency + ')':<25} {m_mod['mae']:<20.4f} {m_base['mae']:<20.4f} {result['mae_improvement_pct']:>+8.2f}%")
    print(f"{'RMSE (' + currency + ')':<25} {m_mod['rmse']:<20.4f} {m_base['rmse']:<20.4f} {result['rmse_improvement_pct']:>+8.2f}%")
    print(f"{'MAPE (%)':<25} {mod_mape_str:<20} {base_mape_str:<20}")
    print(f"{'Directional Acc (%)':<25} {mod_da_str:<20} {base_da_str:<20}")
    print("-" * 75)

    print("\nFeature Coefficients (Interpretable Model Drivers):")
    for feat, coef in result["linear_model_coefficients"].items():
        print(f"  {feat:<22}: {coef:>+10.6f}")

    # Persist model artifact
    models_dir = Path("models").resolve()
    models_dir.mkdir(exist_ok=True)
    safe_name = f"linear_{sanitize_path_component(ticker)}.joblib"
    model_path = assert_safe_model_path(models_dir / safe_name, models_dir)

    # Train final production model on full feature set
    feature_df, feat_cols = build_feature_matrix(df, include_target=True, horizon_sessions=1)
    trainable = feature_df.dropna(subset=["target_price"])
    final_model = LinearRegressionForecaster(alpha=args.alpha)
    final_model.fit(
        X=trainable[feat_cols],
        y_target_price=trainable["target_price"],
        p_origin=trainable["adj_close"],
    )
    final_model.save(model_path)
    print(f"\nModel artifact saved to: {model_path.resolve()}\n")


def train_all_horizons_command(args: argparse.Namespace) -> None:
    """Train direct models for 1-session, 5-session, and 21-session horizons with walk-forward validation."""
    from src.data.db import get_observations
    from src.models.multi_horizon import MultiHorizonForecaster

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = validate_exchange(args.exchange) if args.exchange else df["exchange"].iloc[0]
    model_family = validate_model_family(getattr(args, "model", "linear"))

    print(f"\n==========================================================================")
    print(f"  NextOmni Stock: Multi-Horizon Direct Model Training ({ticker}) [{model_family}]")
    print(f"==========================================================================")
    print("Training dedicated direct models for:")
    print("  • 1 Session  (P_{t+1})  - Next Trading Day")
    print("  • 5 Sessions (P_{t+5})  - Next Trading Week (1 Calendar Week)")
    print("  • 21 Sessions (P_{t+21}) - Next Trading Month (~1 Calendar Month)")
    print("\nRunning Walk-Forward Out-of-Sample Validation across all horizons...")

    forecaster = MultiHorizonForecaster(model_family=model_family, alpha=args.alpha)
    validation_results = forecaster.evaluate_walk_forward_all(df, min_train_sessions=args.min_train)

    print("\n" + "=" * 114)
    print(f"  Multi-Horizon Out-of-Sample Performance Comparison ({ticker}) [{model_family}]")
    print("=" * 114)
    print(f"{'Horizon':<15} {'Steps':<8} {'Model MAE':<12} {'Base MAE':<12} {'Improvement':<13} {'Dir Acc (%)':<13} {'Info Ratio':<12} {'Max Error':<12}")
    print("-" * 114)

    for h_key, res in validation_results.items():
        m_m = res["model_metrics"]
        b_m = res["baseline_metrics"]
        mae_imp = f"{res['mae_improvement_pct']:>+6.2f}%"
        da = f"{m_m['directional_accuracy_pct']:.2f}%"
        ir = f"{m_m.get('information_ratio', 0.0):.3f}"
        me = f"{m_m.get('max_error', 0.0):.4f}"
        print(
            f"{h_key:<15} {res['walk_forward_steps']:<8} "
            f"{m_m['mae']:<12.4f} {b_m['mae']:<12.4f} {mae_imp:<13} {da:<13} {ir:<12} {me:<12}"
        )
    print("-" * 114)

    # Train final models on full feature sets and save
    forecaster.fit(df)
    models_dir = Path("models").resolve()
    save_path = forecaster.save_models(models_dir, ticker)
    print(f"\nMulti-horizon models saved to: {save_path.resolve()}")

    # Generate current future predictions
    future_forecasts = forecaster.predict_all_horizons(df, ticker, exchange)
    print("\nCurrent Multi-Horizon Forecasts & 80% Confidence Bands (from latest observed session):")
    print(f"{'Horizon':<15} {'Target Date':<14} {'Origin Price':<14} {'Predicted Price':<16} {'80% Confidence Band':<24} {'Exp Return':<12}")
    print("-" * 98)
    for h_key, fc in future_forecasts.items():
        interval_str = f"[{fc['lower_bound']:.2f} – {fc['upper_bound']:.2f}]"
        print(
            f"{h_key:<15} {fc['target_session_date']:<14} "
            f"{fc['origin_price']:<14.4f} {fc['predicted_price']:<16.4f} {interval_str:<24} {fc['expected_return_pct']:>+6.2f}%"
        )
    print("-" * 98 + "\n")


def predict_command(args: argparse.Namespace) -> None:
    """Generate and record a live forecast prediction into SQLite."""
    import exchange_calendars as xcals
    from src.data.db import get_observations
    from src.evaluation.tracker import PredictionTracker
    from src.features.builder import build_feature_matrix
    from src.models.base_model import BaseForecaster
    from src.models.baseline import PersistenceBaselineModel
    from src.models.linear_model import LinearRegressionForecaster
    from src.models.multi_horizon import MultiHorizonForecaster

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = validate_exchange(args.exchange) if args.exchange else df["exchange"].iloc[0]
    horizon = int(args.horizon)
    tracker = PredictionTracker()

    latest_obs = df.iloc[-1]
    origin_session = str(latest_obs["session_date"])
    origin_price = float(latest_obs["adj_close"])

    # Resolve target session date using exchange calendar
    cal = xcals.get_calendar(exchange)
    origin_ts = pd.Timestamp(origin_session)
    sessions_ahead = cal.sessions_in_range(origin_ts, origin_ts + pd.Timedelta(days=horizon * 7))
    try:
        idx = list(sessions_ahead).index(origin_ts)
        target_ts = sessions_ahead[idx + horizon]
    except Exception:
        target_ts = cal.next_session(origin_ts)
    target_date_str = target_ts.strftime("%Y-%m-%d")

    safe_ticker = sanitize_path_component(ticker)
    models_dir = Path("models").resolve()
    multi_dir = models_dir / f"multi_horizon_{safe_ticker}"
    linear_file = models_dir / f"linear_{safe_ticker}.joblib"

    lower_bound = None
    upper_bound = None
    confidence_level = 0.80

    if args.model == "baseline":
        model_name = "PersistenceBaseline"
        model_version = "1.0.0"
        pred_price = origin_price
        repro_ref = "persistence_t"
    elif multi_dir.exists():
        multi_forecaster = MultiHorizonForecaster.load_models(models_dir, ticker)
        horizon_map = {1: "1_SESSION", 5: "5_SESSIONS", 21: "21_SESSIONS"}
        h_key = horizon_map.get(horizon, "1_SESSION")
        forecaster = multi_forecaster.models[h_key]
        feature_df, feat_cols = build_feature_matrix(df, include_target=False)
        latest_features = feature_df.iloc[-1][feat_cols].to_frame().T
        pred_arr, low_arr, up_arr = forecaster.predict_interval(
            latest_features, p_origin=origin_price, confidence_level=confidence_level
        )
        pred_price = float(pred_arr[0])
        lower_bound = float(low_arr[0])
        upper_bound = float(up_arr[0])
        model_name = f"{forecaster.model_name}_Direct_{h_key}"
        model_version = forecaster.model_version
        repro_ref = str(multi_dir / f"{h_key}.joblib")
    elif linear_file.exists():
        forecaster = LinearRegressionForecaster.load(linear_file)
        feature_df, feat_cols = build_feature_matrix(df, include_target=False)
        latest_features = feature_df.iloc[-1][feat_cols].to_frame().T
        pred_arr, low_arr, up_arr = forecaster.predict_interval(
            latest_features, p_origin=origin_price, confidence_level=confidence_level
        )
        pred_price = float(pred_arr[0])
        lower_bound = float(low_arr[0])
        upper_bound = float(up_arr[0])
        model_name = forecaster.model_name
        model_version = forecaster.model_version
        repro_ref = str(linear_file)
    else:
        print(f"Model artifact not found for {ticker}. Run 'train-all-horizons --ticker {ticker}' first.")
        sys.exit(1)

    record = tracker.record_prediction(
        ticker=ticker,
        exchange=exchange,
        forecast_origin_session=origin_session,
        origin_price=origin_price,
        horizon_sessions=horizon,
        target_session_date=target_date_str,
        predicted_price=pred_price,
        model_name=model_name,
        model_version=model_version,
        reproducibility_ref=repro_ref,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level if lower_bound is not None else None,
    )

    print("\n" + "=" * 65)
    print("  NextOmni Stock: Forecast Prediction Registered")
    print("=" * 65)
    print(f"  Prediction ID:          {record['prediction_id']}")
    print(f"  Ticker:                 {record['ticker']} ({record['exchange']})")
    print(f"  Forecast Horizon:       {record['horizon_type']}")
    print(f"  Origin Session Date:    {record['forecast_origin_session']}")
    print(f"  Origin Observed Price:  {record['origin_price']:.4f} {record['currency']}")
    print(f"  Target Session Date:    {record['target_session_date']}")
    print(f"  Predicted Price:        {record['predicted_price']:.4f} {record['currency']}")
    if record.get("lower_bound") is not None and record.get("upper_bound") is not None:
        print(f"  80% Confidence Band:    [{record['lower_bound']:.4f} – {record['upper_bound']:.4f}] {record['currency']}")
    exp_ret_pct = ((pred_price - origin_price) / origin_price) * 100.0
    res_std = getattr(forecaster, "residual_std_", 0.02)
    win_prob, conviction_label = calculate_directional_probability(
        expected_return=exp_ret_pct / 100.0,
        residual_std=res_std if res_std > 0 else 0.02,
    )
    print(f"  Expected Return:        {exp_ret_pct:>+6.2f}%")
    print(f"  Directional Win Prob:   {win_prob * 100:.1f}% [{conviction_label}]")
    print(f"  Model:                  {record['model_name']} (v{record['model_version']})")
    print(f"  Status:                 {record['status']}")

    if getattr(args, "explain", False) and hasattr(forecaster, "explain_forecast"):
        try:
            attributions = forecaster.explain_forecast(latest_features)
            if attributions:
                print("  " + "-" * 61)
                print("  Signal Attribution Breakdown (Additive Return Driver):")
                sorted_attrs = sorted(
                    [(k, v) for k, v in attributions.items() if k != "intercept"],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                intercept_val = attributions.get("intercept", 0.0)
                print(f"    {'[Baseline Intercept]':<25} {intercept_val:>+8.4f} ({intercept_val*100:>+6.2f}%)")
                for feat, val in sorted_attrs:
                    direction = "▲" if val > 0 else ("▼" if val < 0 else " ")
                    print(f"    {feat:<25} {val:>+8.4f} ({val*100:>+6.2f}%) {direction}")
        except Exception:
            pass

    print("=" * 65 + "\n")


def evaluate_pending_command(args: argparse.Namespace) -> None:
    """Trigger idempotent evaluation workflow on pending predictions."""
    from src.evaluation.tracker import PredictionTracker

    tracker = PredictionTracker()
    print("\nRunning Idempotent Prediction Evaluation Workflow...")
    summary = tracker.evaluate_pending_predictions()

    print("\nEvaluation Summary:")
    print(f"  Newly Evaluated:        {summary['evaluated']}")
    print(f"  Awaiting Actual Data:   {summary['awaiting']}")
    print(f"  Pending Future Target:  {summary['pending']}")
    print(f"  Failed:                 {summary['failed']}\n")


def list_predictions_command(args: argparse.Namespace) -> None:
    """Display table of recorded predictions and evaluations."""
    from src.data.db import get_predictions

    ticker = validate_ticker(args.ticker) if args.ticker else None
    df = get_predictions(status=args.status, ticker=ticker)
    if df.empty:
        print("\nNo prediction records found.\n")
        return

    print("\n" + "=" * 115)
    print(f"  NextOmni Stock: Predictions Audit Log ({len(df)} records)")
    print("=" * 115)
    print(
        f"{'Ticker':<8} {'Model':<22} {'Origin Date':<12} {'Target Date':<12} "
        f"{'Pred Price':<11} {'Actual Price':<13} {'Error':<10} {'Dir':<5} {'Status':<10}"
    )
    print("-" * 115)

    for _, r in df.iterrows():
        pred_p = f"{float(r['predicted_price']):.2f}"
        act_p = f"{float(r['actual_price']):.2f}" if pd.notna(r["actual_price"]) else "Pending"
        err = f"{float(r['percentage_error']):.2f}%" if pd.notna(r["percentage_error"]) else "-"
        dir_ok = "✓" if pd.notna(r["direction_correct"]) and bool(r["direction_correct"]) else ("✗" if pd.notna(r["direction_correct"]) else "-")
        status = str(r["status"])

        print(
            f"{r['ticker']:<8} {r['model_name'][:20]:<22} {r['forecast_origin_session']:<12} {r['target_session_date']:<12} "
            f"{pred_p:<11} {act_p:<13} {err:<10} {dir_ok:<5} {status:<10}"
        )
    print("-" * 115 + "\n")

    if getattr(args, "export", None):
        from src.security.audit import generate_audit_manifest

        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.suffix.lower() == ".json":
            df.to_json(export_path, orient="records", indent=2)
        else:
            df.to_csv(export_path, index=False)

        manifest_hash = generate_audit_manifest(df.to_dict("records"))
        manifest_file = export_path.with_suffix(export_path.suffix + ".manifest")
        manifest_file.write_text(f"SHA256_MANIFEST:{manifest_hash}\nRECORDS_COUNT:{len(df)}\n")

        print(f"Exported {len(df)} records to: {export_path}")
        print(f"Cryptographic Manifest:     {manifest_file} ({manifest_hash[:16]}...)\n")



def monitor_command(args: argparse.Namespace) -> None:
    """Monitor real-time quotes with session status and latency badges."""
    from src.data.monitor import MarketMonitorService

    ticker = validate_ticker(args.ticker) if args.ticker else None
    exchange = validate_exchange(args.exchange) if args.exchange else None

    service = MarketMonitorService()
    quotes = service.get_multi_quotes(
        tickers=[ticker] if ticker else None,
        exchange=exchange,
        force_refresh=args.refresh,
    )

    print("\nNextOmni Stock Market Monitor:")
    print(f"{'Ticker':<10} {'Exchange':<8} {'Price':<14} {'Change':<10} {'Change %':<10} {'Status':<14} {'Latency Badge':<16} {'Cached':<8}")
    print("-" * 96)
    for q in quotes:
        chg_str = f"{q.change:+.2f}"
        pct_str = f"{q.pct_change:+.2f}%"
        price_str = f"{q.last_price:.2f} {q.currency}"
        print(f"{q.ticker:<10} {q.exchange:<8} {price_str:<14} {chg_str:<10} {pct_str:<10} {q.session_status:<14} {q.latency_badge:<16} {str(q.cached):<8}")
    print("-" * 96)
    for q in quotes:
        print(f"[{q.ticker} ({q.exchange})]: Local Time: {q.local_time} | Status: {q.next_event}")
        if q.note:
            print(f"  Note: {q.note}")
    print()


def session_status_command(args: argparse.Namespace) -> None:
    """Display real-time operational status of supported exchanges."""
    from src.data.session_status import ExchangeSessionEngine

    exchange = validate_exchange(args.exchange) if args.exchange else None
    exchanges = [exchange] if exchange else list(SUPPORTED_EXCHANGES.keys())

    print("\nNextOmni Stock Exchange Session Status:")
    print(f"{'Exchange':<8} {'Name':<26} {'Status':<14} {'Open?':<8} {'Local Time':<30}")
    print("-" * 90)
    for exc in exchanges:
        status_info = ExchangeSessionEngine.get_status(exc)
        print(f"{status_info.exchange:<8} {status_info.name:<26} {status_info.status:<14} {str(status_info.is_open):<8} {status_info.local_time:<30}")
        if status_info.is_open and status_info.next_close:
            print(f"  -> Active Session: Scheduled Close at {status_info.next_close}")
        elif not status_info.is_open and status_info.next_open:
            print(f"  -> Closed Session: Scheduled Re-open at {status_info.next_open}")
        print(f"  -> Operational Details: {status_info.details}\n")


def daemon_reconcile_command(args: argparse.Namespace) -> None:
    """Run automated session-aware reconciliation daemon."""
    from src.evaluation.daemon import EvaluationDaemon

    daemon = EvaluationDaemon(poll_interval_seconds=args.interval)
    if args.once:
        print("\nExecuting single reconciliation daemon pass...")
        res = daemon.run_once()
        metrics = res["reconciliation_metrics"]
        print(f"Timestamp:              {res['timestamp']}")
        print(f"Newly Evaluated:        {metrics.get('evaluated', 0)}")
        print(f"Awaiting Actual Data:   {metrics.get('awaiting', 0)}")
        print(f"Pending Future Targets: {metrics.get('pending', 0)}")
        print(f"Failed Records:         {metrics.get('failed', 0)}\n")
    else:
        print(f"\nStarting NextOmni Stock Evaluation Daemon (Poll Interval: {args.interval}s)...")
        daemon.run_loop(interval_seconds=args.interval, max_iterations=args.max_iterations)


def verify_audit_command(args: argparse.Namespace) -> None:
    """Audit SQLite cryptographic SHA-256 signatures for tamper detection."""
    from src.data.db import verify_database_integrity

    print("\n" + "=" * 65)
    print("  NextOmni Stock: Cryptographic Audit Trail Verification")
    print("=" * 65)
    report = verify_database_integrity()

    print(f"  Predictions Audited:    {report['predictions_audited']}")
    print(f"  Valid Predictions:      {report['predictions_valid']}")
    print(f"  Evaluations Audited:    {report['evaluations_audited']}")
    print(f"  Valid Evaluations:      {report['evaluations_valid']}")
    print("-" * 65)

    if report["is_intact"]:
        print("  ✓ AUDIT STATUS: 100% INTACT. Zero tampering detected.")
        print("=" * 65 + "\n")
    else:
        print("  ✗ AUDIT STATUS: TAMPERING DETECTED!")
        if report["tampered_predictions"]:
            print(f"    Tampered Prediction IDs: {report['tampered_predictions']}")
        if report["tampered_evaluations"]:
            print(f"    Tampered Evaluation IDs: {report['tampered_evaluations']}")
        print("=" * 65 + "\n")
        sys.exit(1)


def backup_db_command(args: argparse.Namespace) -> None:
    """Execute online atomic SQLite backup snapshot with cryptographic verification."""
    from src.data.db import backup_database

    print("\n" + "=" * 65)
    print("  NextOmni Stock: Online Atomic Database Backup")
    print("=" * 65)
    try:
        dest = args.dest if getattr(args, "dest", None) else None
        saved_path = backup_database(destination_path=dest)
        print(f"  ✓ Snapshot Created & Verified: {saved_path}")
        print("  ✓ Cryptographic Integrity:      100% INTACT")
        print("=" * 65 + "\n")
    except Exception as e:
        print(f"  ✗ Backup Failed: {e}")
        print("=" * 65 + "\n")
        sys.exit(1)


def risk_report_command(args: argparse.Namespace) -> None:
    """Generate comprehensive Value-at-Risk (VaR) and Expected Shortfall (CVaR) report."""
    from src.analytics.risk import compute_multi_horizon_risk_profile
    from src.data.db import get_observations

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    print("\n" + "=" * 116)
    print(f"  NextOmni Stock: Multi-Horizon Quantitative Tail Risk Report ({ticker})")
    print("=" * 116)
    risk_summary = compute_multi_horizon_risk_profile(df, horizons=[1, 5, 21])

    if not risk_summary:
        print(f"Error: Insufficient data to compute risk distribution for {ticker}.")
        sys.exit(1)

    print(
        f"{'Horizon':<15} {'VaR 95% (G)':<14} {'VaR 95% (CF)':<14} {'VaR 99% (CF)':<14} "
        f"{'VaR 95% (Hist)':<16} {'CVaR 95% (Hist)':<17} {'CVaR 99% (Hist)':<17}"
    )
    print("-" * 116)

    for h_key, r in risk_summary.items():
        v95_g = f"{r['var_95_gaussian_pct']:.2f}%"
        v95_cf = f"{r['var_95_cornish_fisher_pct']:.2f}%"
        v99_cf = f"{r['var_99_cornish_fisher_pct']:.2f}%"
        v95_h = f"{r['var_95_historical_pct']:.2f}%"
        c95_h = f"{r['cvar_95_historical_pct']:.2f}%"
        c99_h = f"{r['cvar_99_historical_pct']:.2f}%"
        print(
            f"{h_key:<15} {v95_g:<14} {v95_cf:<14} {v99_cf:<14} "
            f"{v95_h:<16} {c95_h:<17} {c99_h:<17}"
        )
    print("-" * 116)
    latest_price = float(df["adj_close"].iloc[-1])
    h1 = risk_summary.get("1_SESSION", {})
    var95_1d = h1.get("var_95_cornish_fisher_pct", 0.0)
    cvar95_1d = h1.get("cvar_95_historical_pct", 0.0)
    print(f"Latest Settlement Price: {latest_price:.2f}")
    print(f"1-Day Maximum Expected Loss (95% Cornish-Fisher VaR): {var95_1d:.2f}% (~{latest_price * var95_1d / 100:.2f})")
    print(f"1-Day Extreme Tail Shortfall (95% Historical CVaR):     {cvar95_1d:.2f}% (~{latest_price * cvar95_1d / 100:.2f})")

    if getattr(args, "stress_test", False):
        from src.analytics.risk import simulate_macro_stress_scenarios
        stress_results = simulate_macro_stress_scenarios(df)
        if stress_results:
            print("\n  Historical Macroeconomic Stress-Testing Scenarios ($10,000 Portfolio Base):")
            print("  " + "-" * 112)
            print(
                f"  {'Scenario':<36} {'Severity':<14} {'Benchmark':<12} {'Ticker Beta':<14} "
                f"{'Est. Loss %':<14} {'Est. Loss ($)':<15} {'Post-Stress Price'}"
            )
            print("  " + "-" * 112)
            for s in stress_results:
                print(
                    f"  {s['scenario_name']:<36} {s['severity']:<14} {s['benchmark_shock_pct']:>+6.1f}%     "
                    f"{s['ticker_vol_beta']:>8.2f}       {s['estimated_loss_pct']:>8.2f}%     "
                    f"${s['estimated_loss_dollars']:>9.2f}      {s['post_stress_price']:>10.2f}"
                )
            print("  " + "-" * 112)

    print("=" * 116 + "\n")


def vacuum_db_command(args: argparse.Namespace) -> None:
    """Run SQLite database integrity check, foreign key verification, and VACUUM defragmentation."""
    print("\nInitiating SQLite database verification and VACUUM compaction...")
    res = vacuum_database()
    kb_before = res["size_before_bytes"] / 1024.0
    kb_after = res["size_after_bytes"] / 1024.0
    kb_reclaimed = res["bytes_reclaimed"] / 1024.0
    print(f"  Foreign Key Consistency: {res['foreign_keys']}")
    print(f"  B-Tree Page Integrity:   {res['integrity']}")
    print(f"  Storage Before:          {kb_before:.2f} KB")
    print(f"  Storage After:           {kb_after:.2f} KB")
    print(f"  Space Reclaimed:         {kb_reclaimed:.2f} KB")
    print("VACUUM and integrity verification completed successfully.\n")


def prune_db_command(args: argparse.Namespace) -> None:
    """Safely prune expired raw provider payloads older than specified days."""
    days = args.days
    if days < 1:
        print("Error: --days must be greater than or equal to 1.")
        sys.exit(1)

    print(f"Pruning raw provider payloads older than {days} days from database...")
    deleted_count = prune_raw_payloads(retention_days=days)
    print(f"Pruning complete. Removed {deleted_count} expired payload record(s).")


def backtest_command(args: argparse.Namespace) -> None:
    """Run walk-forward strategy backtest and output risk-adjusted performance scorecard."""
    from src.analytics.backtester import run_forecast_backtest

    ticker = validate_ticker(args.ticker)
    model_family = validate_model_family(args.model)
    horizon = args.horizon
    fee_bps = float(args.fee_bps)

    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    sizing_mode = getattr(args, "sizing", "fixed")
    target_vol = float(getattr(args, "target_vol", 0.15))
    max_leverage = float(getattr(args, "max_leverage", 1.0))

    print(f"\nRunning out-of-sample walk-forward strategy backtest for {ticker}...")
    print(f"  Model Family:      {model_family}")
    print(f"  Forecast Horizon:  {horizon} session(s)")
    print(f"  Position Sizing:   {sizing_mode.upper()} (Target Vol: {target_vol:.1%}, Max Leverage: {max_leverage:.1f}x)")
    print(f"  Transaction Fee:   {fee_bps} bps")
    print(f"  Min Train Burn-in: {args.min_train} sessions\n")

    res = run_forecast_backtest(
        df,
        horizon_sessions=horizon,
        fee_bps=fee_bps,
        min_train_sessions=args.min_train,
        model_family=model_family,
        sizing_mode=sizing_mode,
        target_annual_vol=target_vol,
        max_leverage=max_leverage,
    )
    m = res.metrics

    print("=" * 65)
    print("  NextOmni Stock: Out-of-Sample Backtest Scorecard")
    print("=" * 65)
    print(f"  Strategy Net Return:       {m.total_return_pct:>+8.2f}%")
    print(f"  Buy & Hold Benchmark:      {m.benchmark_return_pct:>+8.2f}%")
    print(f"  Annualized Strategy Return:{m.annualized_return_pct:>+8.2f}%")
    print(f"  Annualized Sharpe Ratio:   {m.annualized_sharpe:>8.3f}")
    print(f"  Annualized Sortino Ratio:  {m.annualized_sortino:>8.3f}")
    print(f"  Maximum Drawdown:          {m.max_drawdown_pct:>8.2f}%")
    print(f"  Win Rate (Active Trades):  {m.win_rate_pct:>8.2f}%")
    print(f"  Profit Factor:             {m.profit_factor:>8.2f}")
    print(f"  Total Trades (Turnover):   {m.total_trades:>8d}")
    print(f"  Average Portfolio Exposure:{m.avg_exposure:>8.2f}x")
    print(f"  Total Fee Drag Deducted:   {m.fee_drag_pct:>8.2f}%")
    print("=" * 65 + "\n")


def system_health_command(args: argparse.Namespace) -> None:
    """Run unified security and system health pre-flight audit."""
    import json
    from src.security.health import run_system_health_audit

    receipt = run_system_health_audit()

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 65)
    print("  NextOmni Stock: Unified Security & Health Audit")
    print("=" * 65)
    print(f"  Audit Timestamp:         {receipt.timestamp}")
    print(f"  Database Storage:        {receipt.database_path}")
    print(f"  B-Tree Integrity:        {receipt.database_integrity}")
    print(f"  Foreign Key Checks:      {receipt.foreign_keys_integrity}")
    print(f"  Prediction Audit Trail:  {receipt.prediction_audit} ({receipt.audited_predictions_count} preds, {receipt.audited_evaluations_count} evals)")
    print(f"  Model Signatures:        {receipt.model_signatures} ({receipt.verified_models_count} verified)")
    if receipt.tampered_models:
        print(f"  Tampered Models:         {receipt.tampered_models}")
    print(f"  Secret Masking Hygiene:  {receipt.secret_hygiene}")
    print(f"  Overall Health Status:   {'HEALTHY (PASS)' if receipt.is_healthy else 'ISSUES DETECTED (FAIL)'}")
    print(f"  Summary:                 {receipt.summary}")
    print("=" * 65 + "\n")


def screen_command(args: argparse.Namespace) -> None:
    """Run cross-sectional alpha universe screener and display ranked asset leaderboard."""
    import json
    from src.analytics.screener import screen_asset_universe

    exchange = validate_exchange(args.exchange) if args.exchange else None
    model_family = validate_model_family(args.model)
    horizon = args.horizon
    top_n = int(args.top) if getattr(args, "top", None) else None

    records = screen_asset_universe(
        exchange=exchange,
        horizon_sessions=horizon,
        model_family=model_family,
    )

    if getattr(args, "json", False):
        data = [r.to_dict() for r in records[:top_n]] if records else []
        print(json.dumps(data, indent=2))
        return

    if not records:
        print("\nNo qualified assets found with sufficient history for screening.\n")
        return

    display_records = records[:top_n] if top_n else records
    print("\n" + "=" * 105)
    print(f"  NextOmni Stock: Cross-Sectional Alpha Universe Screener ({len(display_records)} Assets)")
    print(f"  Horizon: {horizon} Session(s) | Model: {model_family.upper()} | Filter: {exchange or 'ALL EXCHANGES'}")
    print("=" * 105)
    print(
        f"{'Rank':<5} {'Ticker':<8} {'Exchange':<8} {'Price':<14} {'Exp Return':<12} "
        f"{'Win Prob':<10} {'Conviction':<16} {'95% CF-VaR':<12} {'Sharpe':<8}"
    )
    print("-" * 105)

    for r in display_records:
        price_str = f"{r.latest_price:.2f} {r.currency}"
        ret_str = f"{r.expected_return_pct:>+6.2f}%"
        prob_str = f"{r.directional_win_probability * 100:.1f}%"
        var_str = f"-{r.cornish_fisher_var_pct:.2f}%"
        print(
            f"{r.rank:<5} {r.ticker:<8} {r.exchange:<8} {price_str:<14} {ret_str:<12} "
            f"{prob_str:<10} {r.conviction_label:<16} {var_str:<12} {r.predicted_sharpe:>+6.2f}"
        )
    print("=" * 105 + "\n")


def ic_drift_command(args: argparse.Namespace) -> None:
    """Evaluate Information Coefficient drift and detect alpha decay on reconciled predictions."""
    import json
    from src.analytics.drift import calculate_information_coefficient_drift
    from src.data.db import get_predictions

    ticker = validate_ticker(args.ticker) if getattr(args, "ticker", None) else None
    horizon = f"{args.horizon}d" if getattr(args, "horizon", None) else None
    window = int(args.window) if getattr(args, "window", None) else 20

    preds_df = get_predictions(
        ticker=ticker,
        horizon_type=horizon,
        status="EVALUATED",
    )

    receipt = calculate_information_coefficient_drift(preds_df, window=window)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print("  NextOmni Stock: Information Coefficient & Alpha Drift Monitor")
    print("=" * 70)
    print(f"  Ticker Filter:       {receipt.ticker or 'ALL TICKERS'}")
    print(f"  Horizon Filter:      {receipt.horizon_type or 'ALL HORIZONS'}")
    print(f"  Sample Size:         {receipt.sample_size} evaluated predictions")
    print(f"  Mean IC (Pearson):   {receipt.mean_ic:>+7.4f}")
    print(f"  Mean Rank IC (Spear):{receipt.mean_rank_ic:>+7.4f}")
    print(f"  Rank IC Std Dev:     {receipt.ic_std:>7.4f}")
    print(f"  Information Ratio:   {receipt.information_ratio:>+7.3f}")
    print(f"  IC t-Statistic:      {receipt.t_statistic:>+7.3f}")
    print(f"  Trailing Rank IC:    {receipt.trailing_rank_ic:>+7.4f}")
    print(f"  Alpha Status:        {receipt.decay_status}")
    if receipt.decay_warning:
        print(f"  Alert Warning:       {receipt.decay_warning}")
def portfolio_command(args: argparse.Namespace) -> None:
    """Optimize multi-asset alpha portfolio using Ledoit-Wolf shrinkage covariance."""
    import json
    from src.analytics.portfolio import optimize_alpha_portfolio
    from src.analytics.screener import screen_asset_universe

    exchange = validate_exchange(args.exchange) if args.exchange else None
    horizon = args.horizon
    mode = args.mode
    max_weight = float(args.max_weight)

    records = screen_asset_universe(
        exchange=exchange,
        horizon_sessions=horizon,
        model_family="ensemble",
    )

    if not records:
        print("\nNo qualified assets found with sufficient history for portfolio optimization.\n")
        return

    receipt = optimize_alpha_portfolio(
        screener_records=records,
        mode=mode,
        max_weight=max_weight,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"  NextOmni Stock: Alpha Portfolio Optimizer ({receipt.mode.upper()})")
    print(f"  Horizon: {receipt.horizon_sessions} Sessions | Exchange: {exchange or 'ALL EXCHANGES'}")
    print("=" * 80)
    print(f"  Expected Annual Return:  {receipt.expected_annual_return_pct:>+7.2f}%")
    print(f"  Annualized Volatility:   {receipt.annualized_volatility_pct:>7.2f}%")
    print(f"  Sharpe Ratio (Rf=4%):    {receipt.sharpe_ratio:>+7.3f}")
    print(f"  Ledoit-Wolf Shrinkage:   {receipt.shrinkage_intensity:>7.4f}")
    print(f"  Effective Asset Count:   {receipt.effective_number_of_assets:>7.2f} / {len(receipt.tickers)}")
    print("-" * 80)
    print(f"  {'Ticker':<10} {'Weight %':<12} {'Risk Contribution %':<22}")
    print("-" * 80)
    for ticker, w in receipt.weights.items():
        rc = receipt.risk_contributions_pct.get(ticker, 0.0)
        print(f"  {ticker:<10} {w * 100:>8.2f}%    {rc:>12.2f}%")
    print("=" * 80 + "\n")


def calibration_report_command(args: argparse.Namespace) -> None:
    """Evaluate prediction interval empirical coverage and Winkler sharpness scores."""
    import json
    from src.analytics.calibration import evaluate_interval_calibration
    from src.data.db import get_predictions

    ticker = validate_ticker(args.ticker) if getattr(args, "ticker", None) else None
    horizon = f"{args.horizon}d" if getattr(args, "horizon", None) else None
    confidence = float(args.confidence) if getattr(args, "confidence", None) else 0.80

    preds_df = get_predictions(
        ticker=ticker,
        horizon_type=horizon,
        status="EVALUATED",
    )

    receipt = evaluate_interval_calibration(preds_df, confidence_level=confidence)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print("  NextOmni Stock: Prediction Interval Calibration & Winkler Report")
    print("=" * 70)
    print(f"  Ticker Filter:           {receipt.ticker or 'ALL TICKERS'}")
    print(f"  Horizon Filter:          {receipt.horizon_type or 'ALL HORIZONS'}")
    print(f"  Evaluated Samples:       {receipt.sample_size}")
    print(f"  Nominal Coverage Target: {receipt.nominal_confidence_level * 100:.1f}%")
    print(f"  Empirical Coverage:      {receipt.empirical_coverage_pct:>6.2f}%")
    print(f"  Coverage Error:          {receipt.coverage_error_pct:>6.2f}%")
    print(f"  Under-Breach Rate:       {receipt.under_breach_pct:>6.2f}%")
    print(f"  Over-Breach Rate:        {receipt.over_breach_pct:>6.2f}%")
    print(f"  Mean Interval Width:     {receipt.mean_interval_width:>6.2f}")
    print(f"  Mean Winkler Score:      {receipt.mean_winkler_score:>6.2f} (lower is sharper)")
    print(f"  Calibration Status:      {receipt.calibration_status}")
    print("=" * 70 + "\n")


def lead_lag_command(args: argparse.Namespace) -> None:
    """Evaluate cross-market lead-lag Granger causality between two equities."""
    import json
    from src.analytics.causality import evaluate_granger_causality
    from src.config import PRICE_COLUMN
    from src.data.db import get_observations

    source_ticker = validate_ticker(args.source)
    target_ticker = validate_ticker(args.target)
    max_lag = int(args.max_lag) if getattr(args, "max_lag", None) else 5

    df_src = get_observations(source_ticker)
    df_tgt = get_observations(target_ticker)

    if df_src.empty or df_tgt.empty:
        print(f"\nError: Insufficient data for {source_ticker} or {target_ticker}. Please synchronize both first.\n")
        return

    clean_src = df_src.sort_values("session_date").drop_duplicates(subset=["session_date"])
    clean_tgt = df_tgt.sort_values("session_date").drop_duplicates(subset=["session_date"])

    ret_src = clean_src[PRICE_COLUMN].astype(float).pct_change().dropna()
    ret_src.index = clean_src["session_date"].iloc[1:]

    ret_tgt = clean_tgt[PRICE_COLUMN].astype(float).pct_change().dropna()
    ret_tgt.index = clean_tgt["session_date"].iloc[1:]

    receipt = evaluate_granger_causality(
        source_series=ret_src,
        target_series=ret_tgt,
        max_lag=max_lag,
        source_ticker=source_ticker,
        target_ticker=target_ticker,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print("  NextOmni Stock: Cross-Market Lead-Lag Granger Causality Test")
    print("=" * 70)
    print(f"  Source (Leading Hypothesis): {receipt.source_ticker}")
    print(f"  Target (Lagging Hypothesis): {receipt.target_ticker}")
    print(f"  Aligned Sample Size:         {receipt.sample_size} sessions")
    print(f"  Optimal Evaluated Lag:       {receipt.optimal_lag} session(s)")
    print(f"  Fisher-Snedecor F-Statistic: {receipt.f_statistic:>7.4f}")
    print(f"  Granger Causality p-Value:   {receipt.p_value:>7.4f}")
    print(f"  Unrestricted R-Squared:      {receipt.r_squared_unrestricted:>7.4f}")
    print(f"  Restricted (AR) R-Squared:   {receipt.r_squared_restricted:>7.4f}")
    print(f"  Causality Verdict:           {'CAUSAL (PASS, p < 0.05)' if receipt.is_causal else 'NOT SIGNIFICANT (FAIL)'}")
    print(f"  Summary:                     {receipt.summary}")
    print("=" * 70 + "\n")


def cointegrate_command(args: argparse.Namespace) -> None:
    """Evaluate Engle-Granger cointegration and mean-reversion spread dynamics between two assets."""
    import json
    from src.analytics.causality import evaluate_engle_granger_cointegration
    from src.config import PRICE_COLUMN
    from src.data.db import get_observations

    ticker1 = validate_ticker(args.ticker1)
    ticker2 = validate_ticker(args.ticker2)

    df1 = get_observations(ticker1)
    df2 = get_observations(ticker2)

    if df1.empty or df2.empty:
        print(f"\nError: Insufficient data for {ticker1} or {ticker2}. Please synchronize both first.\n")
        return

    clean_1 = df1.sort_values("session_date").drop_duplicates(subset=["session_date"])
    clean_2 = df2.sort_values("session_date").drop_duplicates(subset=["session_date"])

    p1 = clean_1.set_index("session_date")[PRICE_COLUMN].astype(float)
    p2 = clean_2.set_index("session_date")[PRICE_COLUMN].astype(float)

    receipt = evaluate_engle_granger_cointegration(p1, p2, ticker_1=ticker1, ticker_2=ticker2)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Engle-Granger Cointegration & Mean-Reversion Engine")
    print("=" * 75)
    print(f"  Asset 1 (Dependent):       {receipt.ticker_1}")
    print(f"  Asset 2 (Independent):     {receipt.ticker_2}")
    print(f"  Aligned Sample Size:       {receipt.sample_size} sessions")
    print(f"  Hedge Ratio (Beta):        {receipt.hedge_ratio_beta:>7.4f}")
    print(f"  Intercept (Alpha):         {receipt.intercept_alpha:>7.4f}")
    print(f"  ADF Spread t-Statistic:    {receipt.adf_statistic:>7.4f}")
    print(f"  Engle-Granger 5% Critical: {receipt.critical_values['5%']:>7.2f}")
    print(f"  p-Value (Approx):          {receipt.p_value_approx:>7.4f}")
    print(f"  Cointegration Status:      {'COINTEGRATED (PASS)' if receipt.is_cointegrated else 'NOT COINTEGRATED (FAIL)'}")
    print(f"  Mean-Reversion Half-Life:  {receipt.half_life_sessions:.1f} sessions")
    print(f"  Current Spread Value:      {receipt.current_spread:>+7.4f}")
    print(f"  Current Spread Z-Score:    {receipt.current_spread_zscore:>+7.2f} sigma")
    print(f"  Summary:                   {receipt.summary}")
    print("=" * 75 + "\n")


def liquidity_report_command(args: argparse.Namespace) -> None:
    """Evaluate market microstructure liquidity and price impact diagnostics."""
    import json
    from src.data.db import get_observations
    from src.features.microstructure import evaluate_liquidity_profile

    ticker = validate_ticker(args.ticker)
    window = int(args.window) if getattr(args, "window", None) else 21

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observations found for {ticker}. Please synchronize data first.\n")
        return

    receipt = evaluate_liquidity_profile(df_obs, ticker=ticker, window=window)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print("  NextOmni Stock: Market Microstructure & Liquidity Profile")
    print("=" * 70)
    print(f"  Ticker Symbol:             {receipt.ticker}")
    print(f"  Sample Size:               {receipt.sample_size} sessions")
    print(f"  Latest Corwin-Schultz Sprd:{receipt.latest_corwin_schultz_spread_pct:>6.2f}%")
    print(f"  Mean Corwin-Schultz Spread:{receipt.mean_corwin_schultz_spread_pct:>6.2f}%")
    print(f"  Latest Amihud Illiquidity: {receipt.latest_amihud_illiq:>7.4f}")
    print(f"  Mean Amihud Illiquidity:   {receipt.mean_amihud_illiq:>7.4f}")
    print(f"  Mean Daily Turnover:       ${receipt.mean_dollar_volume:>13,.0f}")
    print(f"  Liquidity Regime:          {receipt.liquidity_regime}")
    print(f"  Summary:                   {receipt.summary}")
    print("=" * 70 + "\n")


def volatility_report_command(args: argparse.Namespace) -> None:
    """Evaluate multi-estimator realized volatility and extreme value diagnostics."""
    import json
    from src.data.db import get_observations
    from src.analytics.volatility import evaluate_volatility_profile

    ticker = validate_ticker(args.ticker)
    window = int(args.window) if getattr(args, "window", None) else 20

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observations found for {ticker}. Please synchronize data first.\n")
        return

    receipt = evaluate_volatility_profile(df_obs, ticker=ticker, window=window)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Multi-Estimator Realized Volatility Diagnostics")
    print("=" * 75)
    print(f"  Ticker Symbol:             {receipt.ticker}")
    print(f"  Sample Size:               {receipt.sample_sessions} sessions")
    print(f"  Estimation Window:         {receipt.window} sessions")
    print("-" * 75)
    print(f"  Close-to-Close (Standard): {receipt.close_to_close_ann_pct:>6.2f}% ann.  (Efficiency:  1.0x baseline)")
    print(f"  Parkinson (1980 High-Low): {receipt.parkinson_ann_pct:>6.2f}% ann.  (Efficiency: ~5.2x baseline)")
    print(f"  Garman-Klass (1980 OHLC):  {receipt.garman_klass_ann_pct:>6.2f}% ann.  (Efficiency: ~7.4x baseline)")
    print(f"  Rogers-Satchell (1991):    {receipt.rogers_satchell_ann_pct:>6.2f}% ann.  (Drift-independent)")
    print(f"  Yang-Zhang (2000 Optimal): {receipt.yang_zhang_ann_pct:>6.2f}% ann.  (Efficiency: ~14.0x baseline)")
    print(f"  RiskMetrics EWMA (λ=0.94): {receipt.ewma_ann_pct:>6.2f}% ann.  (Exponential clustering)")
    print("-" * 75)
    print(f"  Volatility Ratio (YZ / CC):{receipt.volatility_ratio_yz_cc:>7.3f}")
    print(f"  Regime Classification:     {receipt.regime_signal}")
    if receipt.term_structure_yang_zhang:
        ts_str = ", ".join(f"{k}: {v:.1f}%" for k, v in receipt.term_structure_yang_zhang.items())
        print(f"  Term Structure (YZ):       {ts_str}")
    print("=" * 75 + "\n")


def crps_eval_command(args: argparse.Namespace) -> None:
    """Evaluate probabilistic forecasts using proper scoring rules (CRPS, Pinball Loss, Winkler Score)."""
    import json
    from src.data.db import get_predictions
    from src.evaluation.probabilistic import evaluate_probabilistic_forecasts

    ticker = validate_ticker(args.ticker) if getattr(args, "ticker", None) else None
    horizon = int(args.horizon) if getattr(args, "horizon", None) else None
    model_family = getattr(args, "model", None)

    preds_df = get_predictions(ticker=ticker)
    if preds_df.empty:
        msg = f"No predictions recorded in database" + (f" for {ticker}" if ticker else "")
        print(f"\nError: {msg}.\n")
        return

    receipt = evaluate_probabilistic_forecasts(
        preds_df,
        horizon_sessions=horizon,
        model_family=model_family,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Probabilistic Forecast Quality & CRPS Scorecard")
    print("=" * 75)
    if receipt.ticker:
        print(f"  Ticker Filter:             {receipt.ticker}")
    if receipt.horizon_sessions:
        print(f"  Horizon Filter:            {receipt.horizon_sessions} sessions")
    if receipt.model_family:
        print(f"  Model Family Filter:       {receipt.model_family}")
    print(f"  Sample Reconciled Folds:   {receipt.sample_size}")
    print("-" * 75)
    print(f"  Continuous Ranked Prob (CRPS): {receipt.mean_crps:>8.4f}  (Proper continuous scoring rule)")
    print(f"  Mean Absolute Error (MAE):     {receipt.mean_mae:>8.4f}  (Deterministic point baseline)")
    print(f"  Quantile Pinball Loss (10%):   {receipt.mean_pinball_loss_q10:>8.4f}  (Downside tail risk loss)")
    print(f"  Quantile Pinball Loss (50%):   {receipt.mean_pinball_loss_q50:>8.4f}  (Median check loss)")
    print(f"  Quantile Pinball Loss (90%):   {receipt.mean_pinball_loss_q90:>8.4f}  (Upside tail opportunity loss)")
    print(f"  Winkler Sharpness (80% CI):    {receipt.mean_winkler_score_80:>8.4f}  (Width + breach penalty)")
    print(f"  Empirical 80% Coverage:        {receipt.empirical_coverage_80_pct:>7.2f}%")
    print("-" * 75)
    print(f"  Summary: {receipt.summary}")
    print("=" * 75 + "\n")



def main() -> None:

    parser = argparse.ArgumentParser(
        prog="nextomni-stock",
        description="NextOmni Stock Intelligence & Forecasting CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sync parser
    sync_parser = subparsers.add_parser("sync", help="Synchronize historical market data")
    sync_parser.add_argument("--ticker", type=str, help="Stock ticker (e.g. JPM, 1155.KL)")
    sync_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        help="Exchange MIC (XNYS, XKLS)",
    )
    sync_parser.add_argument("--name", type=str, default=None, help="Company name")
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    sync_parser.add_argument(
        "--start", type=str, default=default_start, help="Start date YYYY-MM-DD"
    )
    sync_parser.add_argument(
        "--end", type=str, default=default_end, help="End date YYYY-MM-DD"
    )
    sync_parser.add_argument(
        "--all", action="store_true", help="Sync all default initial tickers"
    )

    # Status parser
    subparsers.add_parser("status", help="Show database status and tracked stocks")

    # Analyze parser
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run exploratory data analysis and export charts"
    )
    analyze_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker to analyze (e.g. JPM, 1155.KL)"
    )

    # Forecast Baseline parser
    baseline_parser = subparsers.add_parser(
        "forecast-baseline", help="Generate next-session persistence forecast & evaluate benchmark"
    )
    baseline_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    baseline_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )

    # Train Linear parser
    linear_parser = subparsers.add_parser(
        "train-linear", help="Train Linear Regression model with walk-forward validation"
    )
    linear_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    linear_parser.add_argument(
        "--min-train", type=int, default=120, help="Minimum initial training sessions"
    )
    linear_parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength alpha"
    )

    # Train All Horizons parser
    multi_parser = subparsers.add_parser(
        "train-all-horizons", help="Train and validate dedicated models for 1, 5, and 21 sessions"
    )
    multi_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    multi_parser.add_argument(
        "--model",
        choices=["linear", "gradient_boosting", "ensemble", "adaptive_ensemble"],
        default="linear",
        help="Forecasting model family (linear, gradient_boosting, ensemble, adaptive_ensemble)",
    )
    multi_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )
    multi_parser.add_argument(
        "--min-train", type=int, default=120, help="Minimum initial training sessions"
    )
    multi_parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength alpha (linear only)"
    )

    # Predict parser
    predict_parser = subparsers.add_parser(
        "predict", help="Generate and store a forecast prediction"
    )
    predict_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    predict_parser.add_argument(
        "--model",
        choices=["linear", "gradient_boosting", "ensemble", "adaptive_ensemble", "baseline"],
        default="linear",
        help="Forecasting model family to use",
    )

    predict_parser.add_argument(
        "--horizon", type=int, default=1, choices=[1, 5, 21], help="Forecast horizon sessions"
    )
    predict_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )
    predict_parser.add_argument(
        "--explain", action="store_true", help="Print local feature signal attribution breakdown"
    )

    # Evaluate pending parser
    subparsers.add_parser(
        "evaluate-pending", help="Reconcile and evaluate pending predictions with actual prices"
    )

    # List predictions parser
    list_parser = subparsers.add_parser(
        "list-predictions", help="Display predictions audit log"
    )
    list_parser.add_argument("--ticker", type=str, default=None, help="Filter by ticker")
    list_parser.add_argument(
        "--status",
        choices=["PENDING", "AWAITING_ACTUAL", "EVALUATED", "FAILED"],
        default=None,
        help="Filter by status",
    )
    list_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export predictions to signed CSV or JSON file with SHA-256 manifest",
    )

    # Monitor parser
    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor real-time quotes, session status, and feed latency"
    )
    monitor_parser.add_argument("--ticker", type=str, default=None, help="Stock ticker (e.g. JPM, 1155.KL)")
    monitor_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )
    monitor_parser.add_argument(
        "--refresh", action="store_true", help="Bypass cache and force live fetch"
    )

    # Session status parser
    session_parser = subparsers.add_parser(
        "session-status", help="Display current exchange operational state and calendar countdown"
    )
    session_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )

    # Daemon reconcile parser
    daemon_parser = subparsers.add_parser(
        "daemon-reconcile", help="Run automated session-aware forecast evaluation daemon"
    )
    daemon_parser.add_argument(
        "--once", action="store_true", help="Run a single reconciliation pass and exit"
    )
    daemon_parser.add_argument(
        "--interval", type=int, default=300, help="Polling interval in seconds (default: 300)"
    )
    daemon_parser.add_argument(
        "--max-iterations", type=int, default=None, help="Maximum daemon cycles before stopping"
    )

    # Verify audit parser
    subparsers.add_parser(
        "verify-audit", help="Verify cryptographic SHA-256 integrity of predictions audit trail"
    )

    # Backup DB parser
    backup_parser = subparsers.add_parser(
        "backup-db", help="Create an atomic verified online SQLite database backup"
    )
    backup_parser.add_argument(
        "--dest", type=str, default=None, help="Target destination file path for backup"
    )

    # Risk report parser
    risk_parser = subparsers.add_parser(
        "risk-report", help="Compute quantitative tail-risk metrics (VaR & CVaR)"
    )
    risk_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    risk_parser.add_argument(
        "--stress-test", action="store_true", help="Simulate historical macroeconomic stress scenarios"
    )

    # Prune DB parser
    prune_parser = subparsers.add_parser(
        "prune-db", help="Safely prune expired raw provider payloads older than N days"
    )
    prune_parser.add_argument(
        "--days", type=int, default=90, help="Retention period in days (default: 90)"
    )

    # Vacuum DB parser
    subparsers.add_parser(
        "vacuum-db", help="Verify B-tree integrity and execute VACUUM storage defragmentation"
    )

    # Backtest parser
    backtest_parser = subparsers.add_parser(
        "backtest", help="Simulate out-of-sample quantitative strategy backtest"
    )
    backtest_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    backtest_parser.add_argument(
        "--model",
        choices=["ensemble", "adaptive_ensemble", "linear", "gradient_boosting"],
        default="ensemble",
        help="Forecasting model family (default: ensemble)",
    )
    backtest_parser.add_argument(
        "--horizon", type=int, default=1, choices=[1, 5, 21], help="Forecast horizon (default: 1)"
    )
    backtest_parser.add_argument(
        "--fee-bps", type=float, default=5.0, help="Transaction fee in basis points (default: 5.0)"
    )
    backtest_parser.add_argument(
        "--min-train", type=int, default=120, help="Initial burn-in training sessions (default: 120)"
    )
    backtest_parser.add_argument(
        "--sizing",
        choices=["fixed", "vol_target", "half_kelly"],
        default="fixed",
        help="Position sizing mechanism (default: fixed)",
    )
    backtest_parser.add_argument(
        "--target-vol",
        type=float,
        default=0.15,
        help="Annualized volatility target for vol_target sizing (default: 0.15)",
    )
    backtest_parser.add_argument(
        "--max-leverage",
        type=float,
        default=1.0,
        help="Maximum portfolio leverage constraint (default: 1.0)",
    )

    # System health parser
    health_parser = subparsers.add_parser(
        "system-health", help="Execute unified security and health pre-flight audit"
    )
    health_parser.add_argument(
        "--json", action="store_true", help="Output audit receipt as JSON"
    )

    # Screener parser
    screen_parser = subparsers.add_parser(
        "screen", help="Scan and rank asset universe cross-sectionally"
    )
    screen_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        default=None,
        help="Optional exchange MIC filter (XNYS, XKLS)",
    )
    screen_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=5,
        help="Forecast horizon in sessions (default: 5)",
    )
    screen_parser.add_argument(
        "--model",
        choices=["ensemble", "adaptive_ensemble", "linear", "gradient_boosting"],
        default="ensemble",
        help="Forecasting model family (default: ensemble)",
    )

    screen_parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit output to top N ranked assets",
    )
    screen_parser.add_argument(
        "--json",
        action="store_true",
        help="Output screener rankings as JSON",
    )

    # IC Drift parser
    ic_drift_parser = subparsers.add_parser(
        "ic-drift", help="Monitor Information Coefficient drift and alpha decay"
    )
    ic_drift_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional ticker symbol to filter evaluated predictions",
    )
    ic_drift_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=None,
        help="Optional forecast horizon filter (1, 5, 21)",
    )
    ic_drift_parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size in sessions (default: 20)",
    )
    ic_drift_parser.add_argument(
        "--json",
        action="store_true",
        help="Output IC drift receipt as JSON",
    )

    # Portfolio parser
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Optimize multi-asset alpha portfolio using Ledoit-Wolf shrinkage"
    )
    portfolio_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        default=None,
        help="Optional exchange MIC filter (XNYS, XKLS)",
    )
    portfolio_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=5,
        help="Forecast horizon in sessions (default: 5)",
    )
    portfolio_parser.add_argument(
        "--mode",
        choices=["max_sharpe", "min_variance", "risk_parity", "equal_weight", "hrp"],
        default="max_sharpe",
        help="Optimization objective mode (default: max_sharpe)",
    )

    portfolio_parser.add_argument(
        "--max-weight",
        type=float,
        default=0.40,
        help="Maximum individual asset allocation weight (default: 0.40)",
    )
    portfolio_parser.add_argument(
        "--json",
        action="store_true",
        help="Output portfolio allocation as JSON",
    )

    # Calibration report parser
    calib_parser = subparsers.add_parser(
        "calibration-report", help="Audit prediction interval empirical coverage and Winkler sharpness"
    )
    calib_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional ticker symbol to filter evaluated predictions",
    )
    calib_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=None,
        help="Optional forecast horizon filter (1, 5, 21)",
    )
    calib_parser.add_argument(
        "--confidence",
        type=float,
        default=0.80,
        help="Nominal target confidence level (default: 0.80)",
    )
    calib_parser.add_argument(
        "--json",
        action="store_true",
        help="Output calibration report as JSON",
    )

    # Lead-lag Granger causality parser
    lead_lag_parser = subparsers.add_parser(
        "lead-lag", help="Evaluate cross-market lead-lag Granger causality"
    )
    lead_lag_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source leading ticker symbol (e.g. JPM)",
    )
    lead_lag_parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target lagging ticker symbol (e.g. 1155.KL)",
    )
    lead_lag_parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Maximum lag orders to test in sessions (default: 5)",
    )
    lead_lag_parser.add_argument(
        "--json",
        action="store_true",
        help="Output lead-lag receipt as JSON",
    )

    # Cointegration parser
    coint_parser = subparsers.add_parser(
        "cointegrate", help="Evaluate Engle-Granger cointegration and mean-reverting spread"
    )
    coint_parser.add_argument(
        "--ticker1",
        type=str,
        required=True,
        help="First ticker symbol (dependent P1)",
    )
    coint_parser.add_argument(
        "--ticker2",
        type=str,
        required=True,
        help="Second ticker symbol (independent P2)",
    )
    coint_parser.add_argument(
        "--json",
        action="store_true",
        help="Output cointegration receipt as JSON",
    )

    # Liquidity report parser
    liq_parser = subparsers.add_parser(
        "liquidity-report", help="Audit market microstructure liquidity and price impact"
    )
    liq_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to audit (e.g. JPM, 1155.KL)",
    )
    liq_parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Rolling observation window in sessions (default: 21)",
    )
    liq_parser.add_argument(
        "--json",
        action="store_true",
        help="Output liquidity receipt as JSON",
    )

    # Volatility report parser
    vol_parser = subparsers.add_parser(
        "volatility-report", help="Multi-estimator realized volatility diagnostics (Yang-Zhang, GK, Parkinson, EWMA)"
    )
    vol_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to audit (e.g. JPM, 1155.KL)",
    )
    vol_parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling observation window in sessions (default: 20)",
    )
    vol_parser.add_argument(
        "--json",
        action="store_true",
        help="Output volatility diagnostics receipt as JSON",
    )

    # CRPS evaluation parser
    crps_parser = subparsers.add_parser(
        "crps-eval", help="Evaluate probabilistic forecasting quality (CRPS, Pinball Loss, Winkler Score)"
    )
    crps_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional stock ticker filter (e.g. JPM, 1155.KL)",
    )
    crps_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=None,
        help="Optional forecast horizon filter (1, 5, 21)",
    )
    crps_parser.add_argument(
        "--model",
        choices=["ensemble", "adaptive_ensemble", "linear", "gradient_boosting"],
        default=None,
        help="Optional model family filter",
    )

    crps_parser.add_argument(
        "--json",
        action="store_true",
        help="Output probabilistic evaluation receipt as JSON",
    )

    args = parser.parse_args()
    if args.command == "sync":
        sync_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    elif args.command == "forecast-baseline":
        forecast_baseline_command(args)
    elif args.command == "train-linear":
        train_linear_command(args)
    elif args.command == "train-all-horizons":
        train_all_horizons_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "evaluate-pending":
        evaluate_pending_command(args)
    elif args.command == "list-predictions":
        list_predictions_command(args)
    elif args.command == "monitor":
        monitor_command(args)
    elif args.command == "session-status":
        session_status_command(args)
    elif args.command == "daemon-reconcile":
        daemon_reconcile_command(args)
    elif args.command == "verify-audit":
        verify_audit_command(args)
    elif args.command == "backup-db":
        backup_db_command(args)
    elif args.command == "risk-report":
        risk_report_command(args)
    elif args.command == "prune-db":
        prune_db_command(args)
    elif args.command == "vacuum-db":
        vacuum_db_command(args)
    elif args.command == "backtest":
        backtest_command(args)
    elif args.command == "system-health":
        system_health_command(args)
    elif args.command == "screen":
        screen_command(args)
    elif args.command == "ic-drift":
        ic_drift_command(args)
    elif args.command == "portfolio":
        portfolio_command(args)
    elif args.command == "calibration-report":
        calibration_report_command(args)
    elif args.command == "lead-lag":
        lead_lag_command(args)
    elif args.command == "cointegrate":
        cointegrate_command(args)
    elif args.command == "liquidity-report":
        liquidity_report_command(args)
    elif args.command == "volatility-report":
        volatility_report_command(args)
    elif args.command == "crps-eval":
        crps_eval_command(args)


if __name__ == "__main__":
    main()



