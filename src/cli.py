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
    from src.analytics.risk import compute_multi_horizon_risk_profile, compute_path_risk_profile
    from src.data.db import get_observations

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    risk_summary = compute_multi_horizon_risk_profile(df, horizons=[1, 5, 21])
    if not risk_summary:
        print(f"Error: Insufficient data to compute risk distribution for {ticker}.")
        sys.exit(1)

    path_rec = compute_path_risk_profile(df, ticker=ticker)

    stress_results = []
    if getattr(args, "stress_test", False):
        from src.analytics.risk import simulate_macro_stress_scenarios
        stress_results = simulate_macro_stress_scenarios(df)

    if getattr(args, "json", False):
        import json
        payload = {
            "ticker": ticker,
            "multi_horizon_var": risk_summary,
            "path_risk_profile": path_rec.to_dict(),
            "stress_scenarios": stress_results if stress_results else None,
        }
        print(json.dumps(payload, indent=2))
        return

    print("\n" + "=" * 116)
    print(f"  NextOmni Stock: Multi-Horizon Quantitative Tail Risk Report ({ticker})")
    print("=" * 116)

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
    print("-" * 116)
    print("  Path-Dependent Cumulative Drawdown & CDaR Risk Profile:")
    print(f"  Current Price / High-Water Mark: {path_rec.current_price:.2f} / {path_rec.high_water_mark:.2f} (Current Drawdown: {path_rec.current_drawdown_pct:.2f}%)")
    print(f"  Historical Maximum Drawdown:     {path_rec.max_drawdown_pct:.2f}%")
    print(f"  Drawdown-at-Risk (95% DaR):      {path_rec.dar_95_pct:.2f}%")
    print(f"  Conditional DaR (95% CDaR):      {path_rec.cdar_95_pct:.2f}%  (Expected tail drawdown depth)")
    print(f"  Conditional DaR (99% CDaR):      {path_rec.cdar_99_pct:.2f}%")
    print(f"  Peter Martin's Ulcer Index:      {path_rec.ulcer_index_pct:.2f}%  (Quadratic depth & duration penalty)")
    print(f"  Underwater Sessions:             Current = {path_rec.current_underwater_sessions} sessions | Historical Max = {path_rec.max_underwater_sessions} sessions")

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
    print(f"  Expected Shortfall (CVaR 95%): {receipt.cvar_95_pct:>6.2f}% daily")
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


def changepoint_detect_command(args: argparse.Namespace) -> None:
    """Detect structural breaks and regime change points using Bayesian Online Change-Point Detection."""
    import json
    from src.data.db import get_observations
    from src.analytics.changepoints import detect_bayesian_changepoints

    ticker = validate_ticker(args.ticker)
    hazard = int(args.hazard) if getattr(args, "hazard", None) else 100

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observations found for {ticker}. Please synchronize data first.\n")
        return

    receipt = detect_bayesian_changepoints(df_obs, ticker=ticker, hazard_lambda=hazard)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Bayesian Online Change-Point & Structural Break Audit")
    print("=" * 75)
    print(f"  Ticker Symbol:             {receipt.ticker}")
    print(f"  Sample Size:               {receipt.sample_sessions} sessions")
    print(f"  Hazard Timescale (λ):      {receipt.hazard_lambda} sessions")
    print("-" * 75)
    print(f"  Current Run-Length:        {receipt.current_run_length} sessions since last break")
    print(f"  Change-Point Probability:  {receipt.current_change_point_prob * 100:>6.2f}%")
    print(f"  Regime Stability Status:   {receipt.regime_stability_status}")
    if receipt.detected_break_dates:
        breaks_str = ", ".join(receipt.detected_break_dates[-5:])
        print(f"  Recent Structural Breaks:  {breaks_str}")
    print("-" * 75)
    print(f"  Summary: {receipt.summary}")
    print("=" * 75 + "\n")


def toxicity_report_command(args: argparse.Namespace) -> None:
    """Audit order flow toxicity and Volume-Synchronized Probability of Toxicity (VPIN)."""
    import json
    from src.data.db import get_observations
    from src.features.toxicity import evaluate_flow_toxicity

    ticker = validate_ticker(args.ticker)
    window = int(args.window) if getattr(args, "window", None) else 20

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observations found for {ticker}. Please synchronize data first.\n")
        return

    receipt = evaluate_flow_toxicity(df_obs, ticker=ticker, window=window)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Order Flow Toxicity & VPIN Microstructure Audit")
    print("=" * 75)
    print(f"  Ticker Symbol:             {receipt.ticker}")
    print(f"  Sample Size:               {receipt.sample_sessions} sessions")
    print(f"  Observation Window:        {receipt.window} sessions")
    print("-" * 75)
    print(f"  Latest VPIN:               {receipt.latest_vpin * 100:>6.2f}%")
    print(f"  Trailing Mean VPIN:        {receipt.mean_vpin * 100:>6.2f}%")
    print(f"  Historical Peak VPIN:      {receipt.max_vpin * 100:>6.2f}%")
    print("-" * 75)
    print(f"  Latest Volume Initiation:  {receipt.latest_buyer_volume_pct:.1f}% Buyer / {receipt.latest_seller_volume_pct:.1f}% Seller")
    print(f"  Toxicity Flow Regime:      {receipt.toxicity_regime}")
    print("-" * 75)
    print(f"  Summary: {receipt.summary}")
    print("=" * 75 + "\n")


def tail_dependence_command(args: argparse.Namespace) -> None:
    """Evaluate multi-asset asymmetric lower-tail dependence and joint crash fragility."""
    import json
    from src.analytics.copula import evaluate_universe_tail_dependence
    from src.security.validators import validate_exchange

    exchange = validate_exchange(args.exchange) if getattr(args, "exchange", None) else None
    threshold = float(args.threshold) if getattr(args, "threshold", None) else 0.05

    receipt = evaluate_universe_tail_dependence(exchange=exchange, threshold=threshold)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Asymmetric Lower-Tail Dependence & Joint Crash Fragility")
    print("=" * 80)
    print(f"  Assets Evaluated:          {len(receipt.tickers)} securities ({', '.join(receipt.tickers)})")
    print(f"  Overlapping Sessions:      {receipt.sample_sessions} sessions")
    print(f"  Tail Quantile Cutoff (q):  {receipt.tail_threshold * 100:.1f}% extreme tail")
    print("-" * 80)
    print(f"  Market Joint Crash Regime: {receipt.market_joint_crash_regime}")
    p1, p2, val = receipt.highest_risk_pair
    print(f"  Highest Co-Crash Pair:     {p1} & {p2} (λ_L = {val * 100:.1f}%)")
    print("-" * 80)
    print("  Systemic Joint Fragility Scores (Mean Off-Diagonal Co-Crash Risk):")
    for t, score in receipt.systemic_fragility_scores.items():
        print(f"    - {t:<10}: {score * 100:>5.1f}%")
    print("-" * 80)
    print("  Lower-Tail Dependence Matrix (λ_L):")
    header = "  " + f"{'':<8}" + "".join(f"{t:>10}" for t in receipt.tickers)
    print(header)
    for t1 in receipt.tickers:
        row_str = "  " + f"{t1:<8}" + "".join(f"{receipt.tail_dependence_matrix.loc[t1, t2] * 100:>9.1f}%" for t2 in receipt.tickers)
        print(row_str)
    print("-" * 80)
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def factor_attribution_command(args: argparse.Namespace) -> None:
    """Evaluate multi-factor risk attribution and style factor exposures."""
    import json
    import math
    from src.analytics.factor_attribution import evaluate_factor_attribution
    from src.security.validators import validate_exchange, validate_ticker

    ticker = validate_ticker(args.ticker)
    exchange = validate_exchange(args.exchange) if getattr(args, "exchange", None) else "XNYS"

    receipt = evaluate_factor_attribution(ticker=ticker, exchange=exchange)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Multi-Factor Risk Attribution & Style Exposure Engine")
    print("=" * 80)
    print(f"  Security / Ticker:         {receipt.ticker} ({receipt.exchange})")
    print(f"  Sample History:            {receipt.sample_sessions} aligned sessions")
    print(f"  Regression Fit (R²):       {receipt.r_squared * 100:.1f}% (Adj R²: {receipt.adj_r_squared * 100:.1f}%)")
    print(f"  Annualized Alpha (α):      {receipt.annualized_alpha_pct:+.2f}% (t={receipt.alpha_t_stat:.2f}, p={receipt.alpha_p_value:.3f})")
    print("-" * 80)
    print("  Euler Variance Decomposition:")
    sys_vol = receipt.total_volatility_ann_pct * math.sqrt(receipt.systemic_risk_share_pct / 100.0) if receipt.total_volatility_ann_pct > 0 else 0.0
    print(f"    - Systematic Factor Risk: {receipt.systemic_risk_share_pct:>5.1f}% (Ann Vol: {sys_vol:.1f}%)")
    print(f"    - Idiosyncratic Specific: {receipt.specific_risk_share_pct:>5.1f}% (Ann Vol: {receipt.idiosyncratic_volatility_ann_pct:.1f}%)")
    print(f"    - Total Asset Volatility: {receipt.total_volatility_ann_pct:>5.1f}% annualized")
    print("-" * 80)
    print("  Style Factor Loadings (Newey-West HAC Robust Estimator):")
    header = f"    {'Factor':<10} {'Beta (β)':>10} {'t-stat':>10} {'p-value':>10} {'% Risk Contrib':>16}"
    print(header)
    print("    " + "-" * 60)
    for f, beta in receipt.factor_betas.items():
        t_stat = receipt.factor_t_stats.get(f, 0.0)
        p_val = receipt.factor_p_values.get(f, 1.0)
        p_pct = receipt.factor_risk_contributions_pct.get(f, 0.0)
        print(f"    {f:<10} {beta:>10.4f} {t_stat:>10.2f} {p_val:>10.3f} {p_pct:>15.1f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):       {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def wavelet_denoise_command(args: argparse.Namespace) -> None:
    """Evaluate multiscale wavelet multiresolution decomposition and adaptive denoising."""
    import json
    from src.data.db import get_observations
    from src.features.wavelet import evaluate_wavelet_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    levels = int(args.levels) if getattr(args, "levels", None) else 3

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    receipt = evaluate_wavelet_profile(df=df_obs, ticker=ticker, levels=levels)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Multiscale Wavelet Denoising & Energy Scalogram Engine")
    print("=" * 80)
    print(f"  Security / Ticker:         {receipt.ticker}")
    print(f"  Sample History:            {receipt.sample_sessions} sessions")
    print(f"  MODWT Decomposition Depth: {receipt.levels} octave levels (Haar shift-invariant)")
    print(f"  Estimated Noise (σ_hat):   ${receipt.noise_sigma_hat:.4f}")
    print(f"  Universal Threshold (λ):   ${receipt.universal_threshold:.4f} (VisuShrink)")
    print(f"  Signal-to-Noise Ratio:     {receipt.snr_db:.2f} dB")
    print("-" * 80)
    print(f"  Latest Settlement Price:   ${receipt.latest_raw_price:.2f}")
    print(f"  Denoised Jump-Clean Trend: ${receipt.latest_denoised_price:.2f}")
    print(f"  Noise Spike Deviation:     {receipt.price_deviation_pct:+.2f}%")
    print(f"  Timescale Energy Regime:   {receipt.dominant_scale_regime}")
    print("-" * 80)
    print("  Multiscale Wavelet Energy Scalogram:")
    for scale_lbl, share_pct in receipt.energy_shares_pct.items():
        print(f"    - {scale_lbl:<20}: {share_pct:>6.2f}% of detail variance")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):       {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def momentum_crash_report_command(args: argparse.Namespace) -> None:
    """Evaluate Barroso & Santa-Clara risk-managed momentum and Daniel & Moskowitz crash hazard."""
    import json
    from src.analytics.risk_managed_momentum import evaluate_momentum_risk_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    target_vol = float(args.target_vol) if getattr(args, "target_vol", None) else 0.12
    max_leverage = float(args.max_leverage) if getattr(args, "max_leverage", None) else 2.0

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    receipt = evaluate_momentum_risk_profile(
        df=df_obs,
        ticker=ticker,
        target_vol=target_vol,
        max_leverage=max_leverage,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Risk-Managed Momentum & Crash-Risk Dampened Trend Engine")
    print("=" * 80)
    print(f"  Security / Ticker:         {receipt.ticker}")
    print(f"  Sample History:            {receipt.sample_sessions} sessions")
    print(f"  Target Volatility (σ*):    {receipt.target_volatility_pct:.1f}% annualized")
    print(f"  Max Leverage Cap:          {receipt.max_leverage:.2f}x")
    print(f"  Current Dynamic Leverage:  {receipt.current_leverage_weight:.2f}x")
    print("-" * 80)
    print(f"  Crash Hazard Probability:  {receipt.current_crash_hazard_pct:.1f}%")
    print(f"  Crash Risk State:          {receipt.crash_risk_regime}")
    print("-" * 80)
    print("  Comparative Strategy Performance (Barroso & Santa-Clara 2015):")
    print(f"    {'Metric':<26} {'Unmanaged Momentum':>20} {'Risk-Managed (RMM)':>20}")
    print("    " + "-" * 68)
    print(f"    {'Annualized Return':<26} {receipt.raw_momentum_ann_return_pct:>+19.2f}% {receipt.rmm_ann_return_pct:>+19.2f}%")
    print(f"    {'Annualized Volatility':<26} {receipt.raw_momentum_volatility_pct:>19.2f}% {receipt.rmm_volatility_pct:>19.2f}%")
    print(f"    {'Sharpe Ratio (Rf=4%)':<26} {receipt.raw_momentum_sharpe:>+19.2f} {receipt.rmm_sharpe:>+19.2f}")
    print(f"    {'Max Drawdown':<26} {receipt.raw_momentum_max_drawdown_pct:>19.2f}% {receipt.rmm_max_drawdown_pct:>19.2f}%")
    print(f"    {'Return Skewness':<26} {receipt.raw_momentum_skewness:>+19.2f} {receipt.rmm_skewness:>+19.2f}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):       {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def fracdiff_command(args: argparse.Namespace) -> None:
    """Evaluate memory-preserving fractional differentiation and minimal stationary order d*."""
    import json
    from src.data.db import get_observations
    from src.features.fracdiff import evaluate_fracdiff_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    threshold = float(args.threshold) if getattr(args, "threshold", None) else 1e-4
    target_d = float(args.d) if getattr(args, "d", None) is not None else None
    auto_d = getattr(args, "auto_d", True) if target_d is None else False

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col].dropna()

    receipt = evaluate_fracdiff_profile(
        ticker=ticker,
        prices=prices,
        threshold=threshold,
        auto_d=auto_d,
        target_d=target_d,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Memory-Preserving Fractional Differentiation Engine (FFD)")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions")
    print(f"  Truncation Threshold (tau):   {receipt.threshold:.1e}")
    print(f"  Filter Window Length (l):     {receipt.window_length} sessions")
    print("-" * 80)
    print(f"  Selected Order (d):           {receipt.selected_d:.3f}")
    print(f"  Optimal Stationary Order (d*):{receipt.optimal_d:.3f}")
    print(f"  ADF Test Statistic:           {receipt.adf_stat:.4f} (5% Critical: -2.863)")
    print(f"  ADF p-value:                  {receipt.adf_pvalue:.4f}")
    print(f"  Stationary (p < 0.05):        {'YES' if receipt.is_stationary else 'NO'}")
    print("-" * 80)
    print("  Memory Preservation vs Integer Differencing:")
    print(f"    Memory Retained Correlation rho(P, P^(d)):      {receipt.retained_memory_corr:>+8.4f} ({receipt.retained_memory_corr * 100:.1f}%)")
    print(f"    Integer Diff Memory Correlation rho(P, Delta P): {receipt.raw_to_diff1_corr:>+8.4f} ({receipt.raw_to_diff1_corr * 100:.1f}%)")
    if receipt.is_stationary:
        print(f"    Alpha Retention Advantage:                     {receipt.retained_memory_corr - receipt.raw_to_diff1_corr:>+8.4f} correlation units preserved!")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print("=" * 80 + "\n")


def kalman_spread_command(args: argparse.Namespace) -> None:
    """Evaluate Online Kalman Filter dynamic hedge ratio and adaptive spread tracking."""
    import json
    from src.analytics.kalman import evaluate_kalman_pairs_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    y_ticker = validate_ticker(args.y_ticker)
    x_ticker = validate_ticker(args.x_ticker)
    delta = float(args.delta) if getattr(args, "delta", None) else 1e-4

    df_y = get_observations(y_ticker)
    df_x = get_observations(x_ticker)

    if df_y.empty or df_x.empty:
        missing = y_ticker if df_y.empty else x_ticker
        print(f"\nError: No observation history found for {missing}. Please synchronize first.\n")
        return

    receipt = evaluate_kalman_pairs_profile(
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        df_y=df_y,
        df_x=df_x,
        delta=delta,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Online Kalman Filter Dynamic Hedge Ratio & Spread Tracker")
    print("=" * 80)
    print(f"  Paired Assets:                Y={receipt.y_ticker} vs X={receipt.x_ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} aligned sessions")
    print(f"  Process Noise (delta):        {receipt.delta_process_noise:.1e}")
    print("-" * 80)
    print(f"  Latest State Intercept (alpha_t):   {receipt.latest_alpha:>+10.4f}")
    print(f"  Latest Dynamic Hedge Ratio (beta_t):{receipt.latest_beta:>10.4f}")
    print(f"  Static Benchmark OLS Beta:          {receipt.ols_static_beta:>10.4f}")
    print(f"  Hedge Ratio Parameter Drift:        {receipt.beta_drift_pct:>+9.2f}% vs static OLS")
    print("-" * 80)
    print(f"  Current Spread (e_t):               {receipt.latest_spread:>+10.4f}")
    print(f"  Adaptive Spread Volatility:         {receipt.latest_spread_std:>10.4f}")
    print(f"  Spread Z-Score:                     {receipt.latest_zscore:>+10.2f}")
    print(f"  Mean-Reversion Half-Life:           {receipt.mean_reversion_half_life:.1f} sessions")
    print(f"  Current Arbitrage Signal:           {receipt.current_signal}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print("=" * 80 + "\n")


def evt_report_command(args: argparse.Namespace) -> None:
    """Evaluate Extreme Value Theory (EVT) Peaks-Over-Threshold tail risk and GPD parameters."""
    import json
    from src.analytics.evt import evaluate_evt_tail_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    threshold_q = float(args.threshold_quantile) if getattr(args, "threshold_quantile", None) else 0.95

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col].dropna()

    receipt = evaluate_evt_tail_profile(
        ticker=ticker,
        prices=prices,
        threshold_quantile=threshold_q,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Extreme Value Theory (EVT) Peaks-Over-Threshold Tail Risk Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions")
    print(f"  Threshold Quantile:           {receipt.threshold_quantile * 100:.1f}% percentile")
    print(f"  Loss Cutoff Threshold (u):    {receipt.threshold_u:.2f}% daily loss")
    print(f"  Tail Exceedances (N_u):       {receipt.exceedances_count} sessions")
    print("-" * 80)
    print("  Generalized Pareto Distribution (GPD) Parameters:")
    print(f"    Shape Parameter (xi / ξ):   {receipt.gpd_shape_xi:>+10.4f} ({receipt.tail_fatness_regime})")
    print(f"    Scale Parameter (beta / β): {receipt.gpd_scale_beta:>10.6f}")
    print("-" * 80)
    print("  Comparative Tail Risk Measures (Daily Loss %):")
    print(f"    {'Risk Model':<30} {'95% VaR':>12} {'99% VaR':>12} {'99% CVaR (ES)':>15}")
    print("    " + "-" * 71)
    print(f"    {'EVT Peaks-Over-Threshold (GPD)':<30} {receipt.var_95_pct:>11.2f}% {receipt.var_99_pct:>11.2f}% {receipt.cvar_99_pct:>14.2f}%")
    print(f"    {'Cornish-Fisher Expansion':<30} {'—':>12} {receipt.cornish_fisher_var_99_pct:>11.2f}% {'—':>15}")
    print(f"    {'Standard Gaussian':<30} {'—':>12} {receipt.gaussian_var_99_pct:>11.2f}% {'—':>15}")
    print("-" * 80)
    print(f"  Gaussian Tail Underestimation: {receipt.tail_risk_underestimation_pct:>+8.1f}% severe under-prediction")
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def entropy_report_command(args: argparse.Namespace) -> None:
    """Evaluate Market Information Entropy, Permutation Complexity, and Predictability Index."""
    import json
    from src.analytics.entropy import evaluate_market_entropy_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(args.window) if getattr(args, "window", None) else 60
    embedding_dim = int(args.embedding_dim) if getattr(args, "embedding_dim", None) else 3

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col].dropna()

    receipt = evaluate_market_entropy_profile(
        ticker=ticker,
        prices=prices,
        window=window,
        m=embedding_dim,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Market Information Entropy & Ordinal Complexity Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions")
    print(f"  Analysis Window:              {receipt.window} sessions")
    print(f"  Embedding Dimension (m):      {receipt.embedding_dim}")
    print("-" * 80)
    print("  Information Entropy & Complexity Measures:")
    print(f"    Permutation Entropy (H_P):    {receipt.permutation_entropy:>10.4f} (0 = Deterministic, 1 = White Noise)")
    print(f"    Sample Entropy (SampEn):      {receipt.sample_entropy:>10.4f}")
    print(f"    Mean Trailing Entropy:        {receipt.mean_permutation_entropy:>10.4f}")
    print(f"    Market Predictability Index:  {receipt.predictability_index_pct:>9.2f}%")
    print("-" * 80)
    print(f"  Information Regime State:     {receipt.entropy_regime}")
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def garch_volatility_command(args: argparse.Namespace) -> None:
    """Evaluate Asymmetric GJR-GARCH(1, 1) volatility forecasting and forward term structure."""
    import json
    from src.analytics.garch import evaluate_garch_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    model_choice = str(getattr(args, "model", "gjr")).lower().strip()
    asymmetric = model_choice != "standard"

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col].dropna()

    receipt = evaluate_garch_profile(
        ticker=ticker,
        prices=prices,
        asymmetric=asymmetric,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Asymmetric GJR-GARCH(1, 1) Volatility Forecasting Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions")
    print(f"  Specification Model:          {receipt.model_type}")
    print("-" * 80)
    print("  QMLE Conditional Variance Parameters:")
    print(f"    Baseline Intercept (omega / ω):       {receipt.omega:.8e}")
    print(f"    Symmetric Shock (alpha / α):          {receipt.alpha:>10.4f}")
    print(f"    Asymmetric Leverage (gamma / γ):      {receipt.gamma_asymmetry:>10.4f}")
    print(f"    Persistence GARCH (beta / β):         {receipt.beta:>10.4f}")
    print(f"    Variance Persistence (phi / φ):       {receipt.persistence_phi:>10.4f}")
    print(f"    Leverage Asymmetry Ratio (α+γ)/α:     {receipt.leverage_asymmetry_ratio:>9.2f}x bad news volatility amplification")
    print("-" * 80)
    print("  Annualized Volatility Levels & Forward Term Structure:")
    print(f"    Current Conditional Volatility:       {receipt.current_conditional_volatility_ann_pct:>9.2f}% annualized")
    print(f"    Unconditional Long-Term Volatility:   {receipt.unconditional_volatility_ann_pct:>9.2f}% annualized")
    print("    Forward Term Structure Projections:")
    for h_label, h_val in receipt.term_structure_forecast_ann_pct.items():
        print(f"      {h_label:<20} {h_val:>10.2f}% annualized")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def resampled_frontier_command(args: argparse.Namespace) -> None:
    """Evaluate Michaud (1998) Resampled Efficient Frontier and compare with Classical Markowitz."""
    import json
    from src.analytics.resampled_portfolio import evaluate_resampled_frontier_profile

    raw_tickers = args.tickers
    if isinstance(raw_tickers, str):
        ticker_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    else:
        ticker_list = [str(t).strip().upper() for t in raw_tickers if str(t).strip()]

    simulations = int(getattr(args, "simulations", 100))
    points = int(getattr(args, "points", 15))
    rf = float(getattr(args, "risk_free_rate", 0.04))

    receipt = evaluate_resampled_frontier_profile(
        tickers=ticker_list,
        num_simulations=simulations,
        num_frontier_points=points,
        risk_free_rate=rf,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Michaud (1998) Resampled Efficiency & Robust Frontier")
    print("=" * 80)
    print(f"  Asset Universe:               {', '.join(receipt.tickers)}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions")
    print(f"  Monte Carlo Simulations:      {receipt.num_simulations} paths")
    print(f"  Frontier Discrete Points:     {receipt.num_frontier_points}")
    print(f"  Shrinkage Intensity:          {receipt.shrinkage_intensity:.4f}")
    print(f"  Risk-Free Rate:               {receipt.risk_free_rate_pct:.2f}%")
    print("-" * 80)
    print("  Maximum Sharpe Ratio Allocations Comparison:")
    print("    Metric                       Classical Markowitz         Michaud Resampled")
    print(f"    Expected Annual Return:      {receipt.classical_max_sharpe.expected_return_ann_pct:>18.2f}%      {receipt.resampled_max_sharpe.expected_return_ann_pct:>18.2f}%")
    print(f"    Annualized Volatility:       {receipt.classical_max_sharpe.volatility_ann_pct:>18.2f}%      {receipt.resampled_max_sharpe.volatility_ann_pct:>18.2f}%")
    print(f"    Sharpe Ratio:                {receipt.classical_max_sharpe.sharpe_ratio:>19.3f}       {receipt.resampled_max_sharpe.sharpe_ratio:>19.3f}")
    print(f"    Effective Assets (N_eff):    {receipt.classical_max_sharpe.effective_assets:>19.2f}       {receipt.resampled_max_sharpe.effective_assets:>19.2f}")
    print(f"    Michaud Diversification Gain: {receipt.michaud_diversification_gain_pct:>+18.1f}%")
    print("-" * 80)
    print("  Asset Weights Comparison at Maximum Sharpe:")
    print("    Ticker            Classical Markowitz         Michaud Resampled         Dispersion (Std)")
    for t in receipt.tickers:
        cw = receipt.classical_max_sharpe.weights.get(t, 0.0) * 100.0
        rw = receipt.resampled_max_sharpe.weights.get(t, 0.0) * 100.0
        sd = receipt.weight_stability_dispersion.get(t, 0.0) * 100.0
        print(f"    {t:<14}    {cw:>16.1f}%         {rw:>16.1f}%         {sd:>15.1f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def mrmr_select_command(args: argparse.Namespace) -> None:
    """Execute Minimum Redundancy Maximum Relevance (mRMR) quantitative feature selection."""
    import json
    from src.features.mrmr import evaluate_mrmr_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    horizon = int(getattr(args, "horizon", 1))
    n_features = int(getattr(args, "n_features", 5))
    criterion = str(getattr(args, "criterion", "MID")).upper()

    receipt = evaluate_mrmr_profile(
        ticker=ticker,
        horizon_sessions=horizon,
        n_features=n_features,
        criterion=criterion,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Minimum Redundancy Maximum Relevance (mRMR) Feature Selection")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Forecast Horizon:             {receipt.horizon_sessions} trading session(s)")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions")
    print(f"  Optimization Criterion:       {receipt.criterion}")
    print(f"  Candidate Signals:            {receipt.num_candidates} features")
    print(f"  Selected Signals:             {receipt.num_selected} features")
    print("-" * 80)
    print("  Greedy Forward Selection History:")
    print("    Step   Selected Feature          Relevance I(f;y)    Redundancy R(f;S)    mRMR Score")
    for step in receipt.selection_history:
        print(
            f"    {step['step']:<6} {step['feature']:<25} {step['relevance']:>16.4f} "
            f"{step['redundancy']:>20.4f} {step['mrmr_score']:>13.4f}"
        )
    print("-" * 80)
    print("  Collinearity & Redundancy Suppression:")
    print(f"    Mean Candidate Collinearity:  {receipt.mean_candidate_correlation:.4f}")
    print(f"    Mean Selected Collinearity:   {receipt.mean_selected_correlation:.4f}")
    print(f"    Multicollinearity Reduction:  {receipt.redundancy_reduction_pct:+.1f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def price_impact_command(args: argparse.Namespace) -> None:
    """Evaluate Kyle's (1985) Informed Trading Lambda and market price impact."""
    import json
    from src.analytics.price_impact import evaluate_price_impact_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 60))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 20:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_price_impact_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        volumes=df_obs["volume"],
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Kyle's (1985) Informed Trading Lambda & Price Impact Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions (Window: {receipt.rolling_window}D)")
    print(f"  Adverse Selection Regime:     {receipt.adverse_selection_regime}")
    print("-" * 80)
    print("  Kyle's Lambda & Market Depth:")
    print(f"    Kyle's Lambda (Percent):      {receipt.kyle_lambda_pct_per_million:.6f}% per $1,000,000 net order flow")
    print(f"    Kyle's Lambda (Dollar/Shares): ${receipt.kyle_lambda_abs_per_k_shares:.6f} per 1,000 shares")
    print(f"    Market Depth (1% Price Move): ${receipt.market_depth_millions:.2f} Million order flow required")
    print(f"    Huber Regression R-Squared:   {receipt.r_squared:.4f}")
    print("-" * 80)
    print("  Trading Volume & Liquidity Baseline:")
    print(f"    21-Day Average Daily Volume:  {receipt.average_daily_volume_shares:,.0f} shares")
    print(f"    Average Daily Dollar Volume:  ${receipt.average_daily_dollar_volume:,.2f}")
    print("-" * 80)
    print("  Square-Root Law Projected Execution Price Impact:")
    print(f"    1% ADV Order Size:            {receipt.sqrt_impact_1pct_adv_bps:>8.2f} basis points (bps)")
    print(f"    5% ADV Order Size:            {receipt.sqrt_impact_5pct_adv_bps:>8.2f} basis points (bps)")
    print(f"    10% ADV Order Size:           {receipt.sqrt_impact_10pct_adv_bps:>8.2f} basis points (bps)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def tail_shape_command(args: argparse.Namespace) -> None:
    """Evaluate robust quantile tail shape, Moors kurtosis, and crash asymmetry."""
    import json
    from src.analytics.tail_shape import evaluate_tail_shape_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 120))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 30:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_tail_shape_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Robust Quantile Tail Shape & Moors Kurtosis Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Window: {receipt.rolling_window}D)")
    print(f"  Tail Shape Regime:            {receipt.tail_shape_regime}")
    print("-" * 80)
    print("  Quantile Fat-Tailedness & Kurtosis:")
    print(f"    Moors Quantile Kurtosis:      {receipt.moors_quantile_kurtosis:.4f} (Normal Benchmark: 1.2330)")
    print(f"    Moors Excess Kurtosis:        {receipt.moors_excess_kurtosis:+.4f}")
    print(f"    Hogg Tail Weight Indicator:   {receipt.hogg_tail_weight:.4f} (Normal Benchmark: 2.5800)")
    print("-" * 80)
    print("  Quantile Asymmetry & Crash Susceptibility:")
    print(f"    Bowley Quantile Skewness:     {receipt.bowley_quantile_skewness:+.4f} ([-1.0, +1.0])")
    print(f"    Tail Asymmetry Ratio (L/R):   {receipt.tail_asymmetry_ratio:.3f}x downside vs upside tail depth")
    print(f"    Left 1% Tail Depth (Loss):    {receipt.left_tail_depth_1pct:.2f}%")
    print(f"    Right 1% Tail Depth (Gain):   {receipt.right_tail_depth_1pct:.2f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def pain_index_command(args: argparse.Namespace) -> None:
    """Evaluate Burke Ratio, Becker Pain Index, and Ulcer Performance Index."""
    import json
    from src.analytics.drawdown_analytics import evaluate_drawdown_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    rf = float(getattr(args, "risk_free_rate", 0.04))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 15:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_drawdown_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        risk_free_rate=rf,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Burke Ratio, Becker Pain Index & Drawdown Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions")
    print(f"  Annualized Return:            {receipt.annualized_return_pct:+.2f}% (Risk-Free: {receipt.risk_free_rate_pct:.2f}%)")
    print(f"  Maximum Drawdown:             {receipt.max_drawdown_pct:.2f}% (Current: {receipt.current_drawdown_pct:.2f}%)")
    print(f"  Under-Water State:            {'ACTIVE DRAWDOWN' if receipt.is_in_drawdown else 'AT HIGH-WATER MARK'}")
    print(f"  Longest Drawdown Duration:    {receipt.max_drawdown_duration_sessions} trading sessions")
    print(f"  Discrete Drawdown Episodes:   {receipt.num_discrete_drawdowns} valleys")
    print("-" * 80)
    print("  Path-Dependent Pain & Drawdown Metrics:")
    print(f"    Becker (2001) Pain Index:     {receipt.pain_index_pct:>8.2f}% average time under water")
    print(f"    Becker Pain Ratio:            {receipt.pain_ratio:>8.3f} excess return per unit average drawdown")
    print(f"    Burke (1994) Ratio:           {receipt.burke_ratio:>8.3f} quadratic penalty on drawdown valleys")
    print(f"    Martin (1989) Ulcer Index:    {receipt.ulcer_index_pct:>8.2f}% RMS drawdown")
    print(f"    Ulcer Performance Index (UPI):{receipt.ulcer_performance_index:>8.3f} Martin ratio")
    print(f"    Calmar Ratio (Ann/MaxDD):     {receipt.calmar_ratio:>8.3f}")
    print(f"    Sterling Ratio (Ann/Top5 DD): {receipt.sterling_ratio:>8.3f}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def markov_regime_command(args: argparse.Namespace) -> None:
    """Evaluate Hamilton (1989) two-state Markov-switching dynamic regime model."""
    import json
    from src.analytics.markov_regime import evaluate_markov_regime_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 120))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 40:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_markov_regime_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hamilton (1989) Two-State Markov-Switching Regime Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Window: {receipt.rolling_window}D)")
    print(f"  Current Inferred Regime:      {receipt.current_regime}")
    print(f"  Crisis State Probability:     {receipt.current_crisis_probability * 100:>6.2f}% (P(S_T = 1))")
    print(f"  Model Log-Likelihood:         {receipt.log_likelihood:>8.2f}")
    print("-" * 80)
    print("  State Characteristics (Annualized Moments):")
    print(f"    State 0 (Low Vol Expansion):  {receipt.state_0_volatility_ann_pct:>6.2f}% Ann Vol, {receipt.state_0_mean_ann_pct:>+6.2f}% Ann Mean")
    print(f"    State 1 (High Vol Crisis):    {receipt.state_1_volatility_ann_pct:>6.2f}% Ann Vol, {receipt.state_1_mean_ann_pct:>+6.2f}% Ann Mean ({receipt.volatility_ratio_state1_to_state0:.1f}x Vol Amplification)")
    print("-" * 80)
    print("  Regime Transition Probabilities & Expected Persistence:")
    print(f"    P(Stay in State 0 | S0):      {receipt.p00_persistence:>6.4f} (Expected Duration: {receipt.expected_low_vol_duration_sessions:.1f} sessions)")
    print(f"    P(Stay in State 1 | S1):      {receipt.p11_persistence:>6.4f} (Expected Duration: {receipt.expected_high_vol_duration_sessions:.1f} sessions)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def rmt_denoise_command(args: argparse.Namespace) -> None:
    """Evaluate Marchenko-Pastur (1967) RMT covariance matrix denoising and detrending."""
    import json
    from src.analytics.rmt_denoiser import evaluate_rmt_denoising_profile

    raw_tickers = getattr(args, "tickers", None)
    ticker_list = None
    if raw_tickers:
        if isinstance(raw_tickers, str):
            ticker_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        else:
            ticker_list = [str(t).strip().upper() for t in raw_tickers if str(t).strip()]

    exchange = getattr(args, "exchange", None)
    lookback = int(getattr(args, "lookback", 252))
    method = getattr(args, "method", "constant_residual")
    detrend = bool(getattr(args, "detrend", False))

    receipt, raw_cov, denoised_cov = evaluate_rmt_denoising_profile(
        tickers=ticker_list,
        exchange=exchange,
        lookback_sessions=lookback,
        method=method,
        detrend=detrend,
    )

    if getattr(args, "json", False):
        out = {
            "receipt": receipt.to_dict(),
            "raw_covariance": raw_cov.to_dict(),
            "denoised_covariance": denoised_cov.to_dict(),
        }
        print(json.dumps(out, indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Marchenko-Pastur (1967) RMT Covariance Denoising Engine")
    print("=" * 80)
    print(f"  Asset Universe:               {', '.join(receipt.tickers)}")
    print(f"  Sample Dimensions:            {receipt.num_assets} assets x {receipt.sample_sessions} sessions (q = {receipt.q_ratio:.2f})")
    print(f"  Denoising Method:             {receipt.method} (Detrend Market Mode: {receipt.is_detrended})")
    print(f"  Fitted Noise Variance:        {receipt.fitted_noise_variance:.4f}")
    print(f"  Marchenko-Pastur Null Band:   [{receipt.noise_floor_lambda_minus:.4f}, {receipt.noise_cutoff_lambda_plus:.4f}] (Lambda+ Cutoff)")
    print("-" * 80)
    print(f"  Signal Eigenvalues (> L+):    {receipt.num_signal_eigenvalues} modes ({receipt.signal_variance_ratio_pct:.1f}% Total Variance)")
    print(f"  Noise Eigenvalues (<= L+):    {receipt.num_noise_eigenvalues} modes (Shrunk / Constant Residual)")
    print("-" * 80)
    print("  Condition Number (Matrix Stability):")
    print(f"    Raw Empirical Matrix:       {receipt.condition_number_raw:>10.2f}")
    print(f"    RMT Denoised Matrix:        {receipt.condition_number_denoised:>10.2f}")
    print(f"    Stability Improvement:      {receipt.condition_number_improvement_ratio:>10.1f}x reduction")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def jump_test_command(args: argparse.Namespace) -> None:
    """Evaluate Barndorff-Nielsen & Shephard (2004, 2006) Bipower Variation and Jump Decomposition."""
    import json
    from src.analytics.jump_diffusion import evaluate_jump_diffusion_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 21))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < window + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_jump_diffusion_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Barndorff-Nielsen & Shephard (2004, 2006) Jump Detection Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lookback Window: {receipt.window}D)")
    print(f"  Jump Regime:                  {receipt.jump_regime}")
    print(f"  Statistically Significant:    {'YES (p < 0.05)' if receipt.has_significant_jump else 'NO (Continuous Brownian Path)'}")
    print("-" * 80)
    print("  Quadratic Variation Volatility Decomposition (Annualized):")
    print(f"    Total Realized Volatility:  {receipt.annualized_total_vol_pct:>6.2f}% Ann (RV = {receipt.realized_variance:.6f})")
    print(f"    Continuous Path Volatility: {receipt.annualized_continuous_vol_pct:>6.2f}% Ann (BV = {receipt.bipower_variation:.6f})")
    print(f"    Jump Component Volatility:  {receipt.annualized_jump_vol_pct:>6.2f}% Ann (JV = {receipt.jump_variation:.6f})")
    print(f"    Jump Variance Proportion:   {receipt.jump_contribution_ratio * 100:>6.2f}% of Total Quadratic Variation")
    print("-" * 80)
    print("  BNS Asymptotic Hypothesis Test Statistics:")
    print(f"    BNS Z-Score:                {receipt.bns_z_statistic:>8.4f} (Standard Normal N(0, 1))")
    print(f"    One-Sided p-value:          {receipt.p_value:>8.4f}")
    print(f"    Tri-power Quarticity (TQ):  {receipt.tripower_quarticity:>12.8f}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def deflated_sharpe_command(args: argparse.Namespace) -> None:
    """Evaluate Bailey & López de Prado (2014) Probabilistic & Deflated Sharpe Ratio."""
    import json
    import math
    from src.evaluation.deflated_sharpe import evaluate_deflated_sharpe_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    trials = int(getattr(args, "trials", 10))
    benchmark_sr = float(getattr(args, "benchmark_sr", 0.0))
    rf = float(getattr(args, "risk_free_rate", 0.04))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 15:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    ret = df_obs[price_col].astype(float).pct_change().dropna()

    receipt = evaluate_deflated_sharpe_profile(
        ticker=ticker,
        returns=ret,
        num_trials=trials,
        benchmark_sharpe_ann=benchmark_sr,
        risk_free_rate_ann=rf,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Bailey & López de Prado (2014) Deflated Sharpe Ratio Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Trials Evaluated: {receipt.num_trials})")
    print(f"  Overfitting Status:           {receipt.overfitting_status}")
    print(f"  Statistically Significant:    {'YES (DSR >= 95%)' if receipt.is_statistically_significant else 'NO (Likely Backtest Overfitting)'}")
    print("-" * 80)
    print("  Performance & Statistical Moments:")
    print(f"    Classical Annualized Sharpe:  {receipt.annualized_sharpe_ratio:>8.2f} (Daily SR: {receipt.daily_sharpe_ratio:.4f})")
    print(f"    Skewness (Crash Asymmetry):   {receipt.skewness:>8.2f} (gamma_3)")
    print(f"    Kurtosis (Tail Heaviness):    {receipt.kurtosis:>8.2f} (gamma_4, Normal = 3.0)")
    print("-" * 80)
    print("  Probabilistic & Deflated Sharpe Ratios:")
    print(f"    Benchmark Hurdle Sharpe:      {receipt.benchmark_sharpe_ann:>8.2f} Ann")
    print(f"    Probabilistic Sharpe (PSR):   {receipt.probabilistic_sharpe_ratio * 100:>7.1f}% (Corrected for Skew & Kurtosis)")
    print(f"    Expected Max SR from Trials:  {receipt.expected_max_sharpe_ann:>8.2f} Ann (Null E[max_N SR] under Data Snooping)")
    print(f"    Deflated Sharpe Ratio (DSR):  {receipt.deflated_sharpe_ratio * 100:>7.1f}% (Multiple Testing & Snooping Adjusted)")
    print(f"    Backtest Overfit Probability: {receipt.overfitting_probability_pct:>7.1f}% (1 - DSR)")
    mintr_str = f"{receipt.min_track_record_sessions:.0f} sessions" if not math.isinf(receipt.min_track_record_sessions) else "Infinite (SR <= Benchmark)"
    print(f"    Minimum Track Record Length:  {mintr_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def effective_spread_command(args: argparse.Namespace) -> None:
    """Evaluate Richard Roll (1984) Effective Bid-Ask Spread and Composite Execution Friction."""
    import json
    import math
    from src.analytics.effective_spread import evaluate_effective_spread_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 21))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < window + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]
    high = df_obs["high"] if "high" in df_obs.columns else None
    low = df_obs["low"] if "low" in df_obs.columns else None
    volumes = df_obs["volume"] if "volume" in df_obs.columns else None

    receipt = evaluate_effective_spread_profile(
        ticker=ticker,
        prices=prices,
        high=high,
        low=low,
        volumes=volumes,
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Richard Roll (1984) Effective Bid-Ask Spread Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lookback Window: {receipt.window}D)")
    print(f"  Mean Price Level:             ${receipt.mean_price:.2f}")
    print(f"  Liquidity Friction Tier:      {receipt.liquidity_tier}")
    print(f"  Autocovariance Regime:        {'Positive (Trending / Momentum)' if receipt.has_positive_autocovariance else 'Negative (Dealer Order Bounce)'}")
    print("-" * 80)
    print("  Implicit Bid-Ask Spread Metrics:")
    print(f"    Roll Dollar Spread:         ${receipt.roll_spread_dollars:.4f}")
    print(f"    Roll Percentage Spread:      {receipt.roll_spread_pct:>6.2f}%")
    print(f"    Signed Continuous Spread:    {receipt.signed_roll_spread_pct:>6.2f}% (Chu, Stivers & Truong)")
    print(f"    Corwin-Schultz Spread:       {receipt.corwin_schultz_spread_pct:>6.2f}% (High-Low 2-Day Range)")
    print(f"    Serial Autocovariance:       {receipt.serial_autocovariance:>10.6f} (gamma_1)")
    print("-" * 80)
    print("  Composite Trading Execution Friction:")
    print(f"    Inferred Round-Trip Cost:    {receipt.composite_round_trip_bps:>6.1f} bps ({receipt.composite_round_trip_bps / 100.0:.2f}%)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def cpcv_validate_command(args: argparse.Namespace) -> None:
    """Evaluate Marcos López de Prado (2018) Combinatorial Purged & Embargoed Cross-Validation."""
    import json
    import math
    from src.evaluation.cpcv import evaluate_cpcv_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    horizon = int(getattr(args, "horizon", 1))
    groups = int(getattr(args, "groups", 6))
    test_groups = int(getattr(args, "test_groups", 2))
    embargo = float(getattr(args, "embargo", 0.01))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 80:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    receipt = evaluate_cpcv_profile(
        ticker=ticker,
        df=df_obs,
        horizon_sessions=horizon,
        num_groups=groups,
        test_groups=test_groups,
        embargo_pct=embargo,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Marcos López de Prado (2018) CPCV & PBO Validation Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Horizon: {receipt.horizon_sessions}D)")
    print(f"  Combinatorial Topology:       N = {receipt.num_groups} Groups, k = {receipt.test_groups} Test Groups")
    print(f"  Folds Evaluated:              C({receipt.num_groups}, {receipt.test_groups}) = {receipt.num_folds} Purged Folds ({receipt.num_paths} Complete Paths)")
    print(f"  Leakage Protection:           Embargo = {receipt.embargo_sessions} sessions, Mean Purged = {receipt.mean_purged_sessions_per_fold:.1f} sessions/fold")
    print(f"  CPCV Overfitting Status:      {receipt.cpcv_status}")
    print("-" * 80)
    print("  Out-of-Sample Performance Benchmarking:")
    print(f"    Model MAE:                  {receipt.mean_model_mae:>8.4f}")
    print(f"    Baseline Persistence MAE:   {receipt.mean_baseline_mae:>8.4f}")
    print(f"    MAE Outperformance:         {receipt.mae_improvement_pct:>+7.2f}%")
    print(f"    Directional Accuracy:       {receipt.directional_accuracy_pct:>7.2f}%")
    print("-" * 80)
    print("  Combinatorial Path Sharpe Distribution & PBO:")
    print(f"    Mean OOS Sharpe Ratio:      {receipt.mean_oos_sharpe_ratio:>8.2f}")
    print(f"    Sharpe Dispersion [Min,Max]:[{receipt.min_oos_sharpe_ratio:.2f}, {receipt.max_oos_sharpe_ratio:.2f}]")
    print(f"    Prob. Backtest Overfit(PBO):{receipt.probability_of_backtest_overfitting_pct:>7.1f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def downside_beta_command(args: argparse.Namespace) -> None:
    """Evaluate Ang, Chen & Xing (2006) Downside Beta and Realized Semivariance."""
    import json
    import math
    from src.analytics.downside_risk import evaluate_downside_risk_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    window = int(getattr(args, "window", 60))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < window + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]

    receipt = evaluate_downside_risk_profile(
        ticker=ticker,
        asset_prices=prices,
        benchmark_ticker=benchmark,
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Ang, Chen & Xing (2006) Downside Beta & Semivariance Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lookback Window: {receipt.lookback_window}D)")
    print(f"  Crash Sensitivity Tier:       {receipt.crash_sensitivity_tier}")
    print("-" * 80)
    print("  Asymmetric Beta Attributions:")
    print(f"    Standard CAPM Beta:         {receipt.capm_beta:>8.2f} (Unconditional Market Co-Movement)")
    print(f"    Downside Beta (Beta^-):     {receipt.downside_beta:>8.2f} (Down-Market Co-Movement: r_m < mean)")
    print(f"    Upside Beta (Beta^+):       {receipt.upside_beta:>8.2f} (Up-Market Co-Movement: r_m > mean)")
    print(f"    Beta Asymmetry Spread:      {receipt.beta_asymmetry_spread:>+8.2f} (Delta Beta = Beta^- - Beta^+)")
    print("-" * 80)
    print("  Realized Semivariance Decomposition (Annualized):")
    print(f"    Total Realized Volatility:  {receipt.annualized_total_vol_pct:>6.2f}% Ann (RV = {receipt.realized_variance:.6f})")
    print(f"    Good Volatility (Upside):   {receipt.annualized_upside_vol_pct:>6.2f}% Ann (RS^+ = {receipt.upside_semivariance:.6f})")
    print(f"    Bad Volatility (Downside):  {receipt.annualized_downside_vol_pct:>6.2f}% Ann (RS^- = {receipt.downside_semivariance:.6f})")
    print(f"    Downside Variance Ratio:    {receipt.downside_variance_ratio_pct:>6.2f}% of Total Variance")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def optimal_execution_command(args: argparse.Namespace) -> None:
    """Evaluate Almgren & Chriss (2000) optimal execution trajectory and market impact."""
    import json
    from src.analytics.optimal_execution import evaluate_optimal_execution_plan
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    shares = float(args.shares)
    days = int(getattr(args, "days", 5))
    lam = float(getattr(args, "risk_aversion", 1e-6))

    receipt = evaluate_optimal_execution_plan(
        ticker=ticker,
        target_shares=shares,
        execution_days=days,
        risk_aversion_lambda=lam,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    action = "Liquidation (Sell)" if receipt.target_shares > 0 else "Acquisition (Buy)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Almgren & Chriss (2000) Optimal Execution Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} ({action})")
    print(f"  Order Size:                   {abs(receipt.target_shares):,.0f} shares (Benchmark S_0: ${receipt.initial_price:.2f})")
    print(f"  Total Market Value:           ${receipt.total_market_value:,.2f}")
    print(f"  Execution Horizon:            {receipt.execution_days} sessions (Urgency Kappa: {receipt.urgency_parameter_kappa:.4f})")
    print(f"  Execution Half-Life:          {receipt.half_life_days:.2f} trading days")
    print(f"  Risk Aversion (Lambda):       {receipt.risk_aversion_lambda:.8f}")
    print("-" * 80)
    print("  Transaction Cost & Market Impact Metrics:")
    print(f"    Expected Total Shortfall:   ${receipt.expected_total_shortfall_dollars:>10.2f} ({receipt.expected_total_shortfall_bps:>6.2f} bps)")
    print(f"    Permanent Impact Cost:      ${receipt.expected_permanent_impact_dollars:>10.2f}")
    print(f"    Temporary Impact Cost:      ${receipt.expected_temporary_impact_dollars:>10.2f}")
    print(f"    Timing Risk (StdDev):       ${receipt.shortfall_std_dev_dollars:>10.2f}")
    print("-" * 80)
    print("  Discrete Execution Trajectory Schedule:")
    print(f"    {'Step':<6} {'Time(D)':<8} {'Start Shares':<14} {'Trade (n_j)':<14} {'Remaining':<14} {'Executed %':<10}")
    for step in receipt.trajectory_schedule:
        print(
            f"    {step['step']:<6} {step['time_day']:<8.1f} {step['inventory_start']:<14,.0f} "
            f"{step['trade_shares']:<14,.0f} {step['inventory_remaining']:<14,.0f} {step['percent_executed_pct']:<10.1f}%"
        )
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def fama_french_command(args: argparse.Namespace) -> None:
    """Evaluate Fama-French (1993) and Carhart (1997) multi-factor risk and alpha attribution."""
    import json
    from src.analytics.fama_french import evaluate_fama_french_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 60))

    df_obs = db.get_observations(ticker)
    if df_obs.empty or len(df_obs) < lookback + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]

    receipt = evaluate_fama_french_profile(
        ticker=ticker,
        asset_prices=prices,
        benchmark_ticker=benchmark,
        lookback_window=lookback,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sig_label = "YES (p < 0.05)" if receipt.is_alpha_significant else "NO (p >= 0.05)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Fama-French & Carhart (1997) Multi-Factor Attribution Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: SPY)")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total History: {receipt.sample_sessions} sessions)")
    print(f"  Adjusted R-Squared:           {receipt.r_squared_adjusted * 100:.2f}% (R^2: {receipt.r_squared * 100:.2f}%)")
    print("-" * 80)
    print("  Alpha & Factor Loadings (White HC0 Robust Standard Errors):")
    print(f"    {'Factor':<22} {'Beta / Loading':<16} {'Std Error':<12} {'t-Stat':<12} {'p-Value':<10} {'Sig (5%)':<8}")
    for f in receipt.factor_loadings:
        sig_str = "YES" if f["is_significant_5pct"] else "no"
        print(
            f"    {f['factor_name']:<22} {f['beta']:>12.4f}    {f['std_error']:>10.4f}  "
            f"{f['t_stat']:>10.2f}  {f['p_value']:>8.4f}  {sig_str:>6}"
        )
    print("-" * 80)
    print("  Jensen's Alpha & Risk Decomposition (Annualized):")
    print(f"    Annualized Alpha:           {receipt.annualized_alpha_pct:>+8.2f}% ({receipt.alpha_daily_bps:>+6.1f} bps/day | t={receipt.alpha_t_stat:+.2f}, Sig: {sig_label})")
    print(f"    Total Return Volatility:    {receipt.total_annualized_vol_pct:>8.2f}% Ann")
    print(f"    Systematic Factor Vol:      {receipt.systematic_annualized_vol_pct:>8.2f}% Ann")
    print(f"    Idiosyncratic (Specific):   {receipt.idiosyncratic_annualized_vol_pct:>8.2f}% Ann ({receipt.idiosyncratic_variance_ratio_pct:.1f}% of Total Variance)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def black_litterman_command(args: argparse.Namespace) -> None:
    """Execute Black-Litterman (1992) Bayesian portfolio optimization & active view blending."""
    import json
    from src.analytics.black_litterman import evaluate_black_litterman_portfolio
    from src.security.validators import validate_ticker

    raw_tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    universe = [validate_ticker(t) for t in raw_tickers]
    if len(universe) < 2:
        print("\nError: Universe must contain at least 2 distinct tickers.\n")
        return

    views_dict = {}
    if getattr(args, "views", None):
        raw_views = args.views.strip()
        if raw_views.startswith("{"):
            try:
                parsed = json.loads(raw_views)
                views_dict = {validate_ticker(k): float(v) for k, v in parsed.items()}
            except Exception as e:
                print(f"\nError: Invalid JSON views format: {e}\n")
                return
        else:
            for item in raw_views.split(","):
                if ":" in item:
                    k, v = item.split(":", 1)
                    views_dict[validate_ticker(k.strip())] = float(v.strip())

    tau = float(getattr(args, "tau", 0.05))
    delta = float(getattr(args, "risk_aversion", 2.5))
    lookback = int(getattr(args, "lookback", 120))
    denoise = not getattr(args, "no_denoise", False)

    try:
        receipt = evaluate_black_litterman_portfolio(
            universe=universe,
            views_dict=views_dict,
            tau=tau,
            risk_aversion_delta=delta,
            lookback_window=lookback,
            denoise_cov=denoise,
        )
    except Exception as ex:
        print(f"\nError: Black-Litterman optimization failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    denoise_status = "ACTIVE (RMT Marchenko-Pastur)" if receipt.is_rmt_denoised else "INACTIVE (Sample Cov)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Black-Litterman (1992) Bayesian Portfolio Engine")
    print("=" * 80)
    print(f"  Asset Universe ({len(receipt.universe)}):        {', '.join(receipt.universe)}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lookback: {receipt.lookback_window}d)")
    print(f"  Parameters:                   tau = {receipt.tau:.3f} | delta = {receipt.risk_aversion_delta:.2f} | Views = {receipt.views_count}")
    print(f"  Covariance Denoising:         {denoise_status}")
    print("-" * 80)
    print("  Bayesian Return & Constrained Allocation Decomposition:")
    print(f"    {'Ticker':<10} {'Market Wgt':<12} {'Prior Ret (Pi)':<16} {'Posterior Ret (μ)':<18} {'BL Optimal Wgt':<16} {'Shift (Δw)':<10}")
    for alloc in receipt.allocations:
        shift_sign = "+" if alloc["weight_shift"] >= 0 else ""
        print(
            f"    {alloc['ticker']:<10} {alloc['market_weight'] * 100:>10.2f}%  "
            f"{alloc['prior_return_ann_pct']:>14.2f}%  "
            f"{alloc['posterior_return_ann_pct']:>16.2f}%  "
            f"{alloc['optimal_weight'] * 100:>14.2f}%  "
            f"{shift_sign}{alloc['weight_shift'] * 100:>8.2f}%"
        )
    print("-" * 80)
    print("  Portfolio Risk & Return Aggregate Performance:")
    print(f"    Expected Annual Return:     {receipt.expected_portfolio_return_ann_pct:>+8.2f}% Ann (Market Prior: {receipt.prior_market_return_ann_pct:>+8.2f}%)")
    print(f"    Portfolio Annualized Vol:   {receipt.expected_portfolio_vol_ann_pct:>8.2f}% Ann")
    print(f"    Portfolio Sharpe Ratio:     {receipt.sharpe_ratio:>+8.2f} (Rf = 4.0%)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def idiosyncratic_vol_command(args: argparse.Namespace) -> None:
    """Evaluate Ang, Hodrick, Xing & Zhang (2006, 2009) Idiosyncratic Volatility & Higher Moments."""
    import json
    from src.analytics.idiosyncratic_volatility import evaluate_idiosyncratic_volatility_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 60))

    df_obs = db.get_observations(ticker)
    if df_obs.empty or len(df_obs) < lookback + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]

    try:
        receipt = evaluate_idiosyncratic_volatility_profile(
            ticker=ticker,
            asset_prices=prices,
            benchmark_ticker=benchmark,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Idiosyncratic volatility evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Ang et al. (2006) Idiosyncratic Volatility & Higher Moments Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total Aligned: {receipt.sample_sessions} sessions)")
    print(f"  Annualized Alpha:             {receipt.annualized_alpha_pct:>+8.2f}% Ann")
    print(f"  Total Return Volatility:      {receipt.total_annualized_vol_pct:>8.2f}% Ann")
    print(f"  R-Squared (Carhart 4-Factor): {receipt.r_squared * 100:>8.2f}%")
    print("-" * 80)
    print("  Idiosyncratic Dispersion & Downside Asymmetry:")
    print(f"    Idiosyncratic Vol (IVOL):   {receipt.idiosyncratic_vol_ann_pct:>8.2f}% Ann ({receipt.idiosyncratic_variance_ratio_pct:.1f}% of Total Variance)")
    print(f"    Downside IVOL (IVOL^-):     {receipt.downside_idiosyncratic_vol_ann_pct:>8.2f}% Ann (Downside Ratio: {receipt.downside_ivol_ratio:.2f})")
    print(f"    Idiosyncratic Skewness:     {receipt.idiosyncratic_skewness:>+8.2f} (Boyer et al. 2010 Lottery Bias)")
    print(f"    Idiosyncratic Kurtosis:     {receipt.idiosyncratic_kurtosis:>8.2f} (Fat-Tail Excess)")
    print(f"    Idiosyncratic Info Ratio:   {receipt.idiosyncratic_information_ratio:>+8.2f} (Alpha / IVOL)")
    print("-" * 80)
    print("  Ang et al. (2006) Cross-Sectional Pricing Anomaly Assessment:")
    print(f"    Anomaly Risk Regime:        {receipt.ivol_anomaly_regime}")
    print(f"    Expected Return Haircut:    -{receipt.ivol_expected_return_haircut_pct:.2f}% / year (Cross-sectional penalty)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def liquidity_beta_command(args: argparse.Namespace) -> None:
    """Evaluate Pástor & Stambaugh (2003) Systematic Liquidity Risk Factor & Liquidity Beta."""
    import json
    from src.analytics.pastor_stambaugh import evaluate_pastor_stambaugh_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 60))

    df_obs = db.get_observations(ticker)
    if df_obs.empty or len(df_obs) < lookback + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]
    volumes = df_obs["volume"]

    try:
        receipt = evaluate_pastor_stambaugh_profile(
            ticker=ticker,
            asset_prices=prices,
            asset_volumes=volumes,
            benchmark_ticker=benchmark,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Pástor-Stambaugh liquidity evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sig_str = "YES (p < 0.05)" if receipt.is_liquidity_beta_significant else "NO (p >= 0.05)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Pástor & Stambaugh (2003) Systematic Liquidity Risk Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total Aligned: {receipt.sample_sessions} sessions)")
    print(f"  Annualized Alpha:             {receipt.annualized_alpha_pct:>+8.2f}% Ann")
    print(f"  Market Beta (β_M):            {receipt.market_beta:>8.2f}")
    print(f"  Model R-Squared:              {receipt.r_squared * 100:>8.2f}%")
    print("-" * 80)
    print("  Systematic Liquidity Sensitivity & Risk Premium:")
    print(f"    Liquidity Beta (β_L):       {receipt.liquidity_beta:>+8.3f} (t={receipt.liquidity_beta_t_stat:>+6.2f}, Sig: {sig_str})")
    print(f"    Order Flow Gamma (γ):       {receipt.average_price_impact_gamma:>+8.4f} (Price Impact Reversal)")
    print(f"    Liquidity Risk Regime:      {receipt.liquidity_risk_regime}")
    print(f"    Expected Liquidity Premium: {receipt.expected_liquidity_premium_pct:>+8.2f}% / year (Cross-sectional premium)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def fama_macbeth_command(args: argparse.Namespace) -> None:
    """Execute Fama & MacBeth (1973) Two-Pass Cross-Sectional Asset Pricing & Factor Premia estimation."""
    import json
    from src.analytics.fama_macbeth import evaluate_fama_macbeth_profile
    from src.security.validators import validate_ticker

    raw_tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    universe = [validate_ticker(t) for t in raw_tickers]
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 120))

    if len(universe) < 4:
        print("\nError: Universe must contain at least 4 distinct tickers for cross-sectional regression.\n")
        return

    try:
        receipt = evaluate_fama_macbeth_profile(
            universe=universe,
            benchmark_ticker=benchmark,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Fama-MacBeth evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    status_str = "REJECTED (p < 0.05)" if receipt.is_model_rejected_5pct else "SUPPORTED (p >= 0.05)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Fama & MacBeth (1973) Two-Pass Asset Pricing Engine")
    print("=" * 80)
    print(f"  Asset Universe ({len(receipt.universe)}):        {', '.join(receipt.universe)}")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total Aligned: {receipt.sample_sessions} sessions)")
    print(f"  Cross-Sectional R-Squared:    {receipt.cross_sectional_r_squared * 100:>8.2f}%")
    print(f"  Pricing Error RMSE:           {receipt.pricing_error_rmse_ann_pct:>8.2f}% Ann (Unexplained Alpha)")
    print(f"  Pricing Model Test:           {status_str}")
    print("-" * 80)
    print("  Cross-Sectional Factor Risk Premia (with Shanken 1992 Errors-in-Variables):")
    print(f"    {'Factor':<12} {'Premium (Ann)':<16} {'t-Stat (FM)':<14} {'t-Stat (Shanken)':<18} {'p-Val':<10} {'Sig (5%)':<8}")
    for p in receipt.factor_premia:
        sig_str = "YES" if p["is_significant_5pct"] else "no"
        print(
            f"    {p['factor_name']:<12} {p['risk_premium_ann_pct']:>12.2f}%  "
            f"{p['t_stat_fm']:>12.2f}  "
            f"{p['t_stat_shanken']:>16.2f}  "
            f"{p['p_value_shanken']:>8.4f}  {sig_str:>6}"
        )
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def higher_moments_command(args: argparse.Namespace) -> None:
    """Evaluate Harvey-Siddique (2000) Co-Skewness & Dittmar (2002) Co-Kurtosis Higher-Moment Asset Pricing."""
    import json
    from src.analytics.higher_moments import evaluate_higher_moments_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 60))

    df_obs = db.get_observations(ticker)
    if df_obs.empty or len(df_obs) < lookback + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]

    try:
        receipt = evaluate_higher_moments_profile(
            ticker=ticker,
            asset_prices=prices,
            benchmark_ticker=benchmark,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Higher-moments asset pricing evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sig_sks = "YES (p < 0.05)" if receipt.is_coskewness_significant else "no (p >= 0.05)"
    sig_ckt = "YES (p < 0.05)" if receipt.is_cokurtosis_significant else "no (p >= 0.05)"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Harvey-Siddique & Dittmar Higher-Moment Asset Pricing Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total Aligned: {receipt.sample_sessions} sessions)")
    print(f"  Market Beta (β_M):            {receipt.market_beta:>8.2f}")
    print(f"  Four-Moment Alpha:            {receipt.annualized_alpha_pct:>+8.2f}% Ann (R²: {receipt.four_moment_r_squared * 100:.1f}%)")
    print("-" * 80)
    print("  Higher-Order Systematic Co-Moments:")
    print(f"    Co-Skewness (β_SKS):        {receipt.coskewness_beta:>+8.3f} (t={receipt.coskewness_t_stat:>+6.2f}, Sig: {sig_sks})")
    print(f"    Co-Kurtosis (β_CKT):        {receipt.cokurtosis_beta:>8.3f} (t={receipt.cokurtosis_t_stat:>+6.2f}, Sig: {sig_ckt})")
    print("-" * 80)
    print("  Higher-Moment Risk Assessment & Expected Return Premia:")
    print(f"    Risk Regime:                {receipt.higher_moment_risk_regime}")
    print(f"    Co-Skewness Premium:        {receipt.expected_coskewness_premium_pct:>+8.2f}% / year (-3.6% * β_SKS)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def merton_credit_command(args: argparse.Namespace) -> None:
    """Evaluate Robert C. Merton (1974) Structural Credit Risk & Distance-to-Default (DD)."""
    import json
    from src.analytics.merton_credit import evaluate_merton_credit_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    debt = float(getattr(args, "debt", 0.0))
    shares = getattr(args, "shares", None)
    if shares is not None:
        shares = float(shares)
    equity_val = getattr(args, "market_cap", None)
    if equity_val is not None:
        equity_val = float(equity_val)
    vol = getattr(args, "volatility", None)
    if vol is not None:
        vol = float(vol)
    rf = float(getattr(args, "risk_free", 0.045))
    horizon = float(getattr(args, "horizon", 1.0))
    lookback = int(getattr(args, "lookback", 60))

    try:
        receipt = evaluate_merton_credit_profile(
            ticker=ticker,
            debt_value=debt,
            shares_outstanding=shares,
            equity_value=equity_val,
            equity_volatility=vol,
            risk_free_rate_ann=rf,
            horizon_years=horizon,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Merton structural credit evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Merton (1974) Structural Credit Risk & Distance-to-Default Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Equity Market Value (E):      ${receipt.equity_value:,.0f} (Vol: {receipt.annualized_equity_volatility * 100:.2f}%)")
    print(f"  Debt Liabilities (D):         ${receipt.debt_value:,.0f} (Horizon T: {receipt.horizon_years:.1f} yr, Rf: {receipt.risk_free_rate_ann * 100:.2f}%)")
    print("-" * 80)
    print("  Structural Implied Asset State (Option Inversion):")
    print(f"    Implied Firm Assets (V_A):  ${receipt.implied_asset_value:,.0f}")
    print(f"    Implied Asset Vol (σ_A):    {receipt.implied_asset_volatility * 100:>8.2f}%")
    print(f"    Leverage Ratio (D / V_A):   {receipt.leverage_ratio_pct:>8.2f}%")
    print("-" * 80)
    print("  Default Probability & Structural Credit Spread:")
    print(f"    Distance-to-Default (DD):   {receipt.distance_to_default:>+8.2f} std devs (Risk-Neutral d2: {receipt.risk_neutral_dd:>+8.2f})")
    print(f"    Default Probability (π):    {receipt.default_probability_pct:>8.4f}%")
    print(f"    Implied Credit Spread:      {receipt.implied_credit_spread_bps:>8.1f} bps")
    print(f"    Credit Distress Regime:     {receipt.distress_regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def hasbrouck_var_command(args: argparse.Namespace) -> None:
    """Evaluate Joel Hasbrouck (1991) Vector Autoregression (VAR) Price Discovery & Information Share."""
    import json
    from src.analytics.hasbrouck_var import evaluate_hasbrouck_var_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    lags = int(getattr(args, "lags", 5))
    horizon = int(getattr(args, "horizon", 20))
    lookback = int(getattr(args, "lookback", 120))

    try:
        receipt = evaluate_hasbrouck_var_profile(
            ticker=ticker,
            var_lags=lags,
            irf_horizon=horizon,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Hasbrouck VAR evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hasbrouck (1991) VAR Price Discovery & Information Share")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lags p={receipt.var_lags}, IRF H={receipt.irf_horizon})")
    print("-" * 80)
    print("  Price Impact Decomposition (Unexpected 1-σ Trade Shock):")
    print(f"    Contemporaneous Impact:     {receipt.contemporaneous_impact_bps:>+8.2f} bps (b0 initial trade response)")
    print(f"    Permanent Impact I(∞):      {receipt.permanent_impact_bps:>+8.2f} bps (Private Information Content)")
    print(f"    Transient Friction S0:      {receipt.transient_impact_bps:>+8.2f} bps (Bid-Ask Bounce / Inventory Decay)")
    print(f"    Permanent Ratio:            {receipt.permanent_to_total_ratio_pct:>8.1f}%")
    print("-" * 80)
    print("  Hasbrouck Random-Walk Variance Decomposition:")
    print(f"    Trade Info Share (R_w²):    {receipt.information_share_rw2_pct:>8.2f}% of random walk volatility")
    print(f"    Public News Volatility:     {receipt.public_innovation_bps:>8.2f} bps (Non-trade macro innovations)")
    print(f"    Market Quality Regime:      {receipt.market_quality_regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def vrp_command(args: argparse.Namespace) -> None:
    """Evaluate Bollerslev, Tauchen & Zhou (2009) Variance Risk Premium (VRP) & Volatility-of-Volatility."""
    import json
    from src.analytics.variance_risk_premium import evaluate_variance_risk_premium_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    iv = getattr(args, "iv", None)
    if iv is not None:
        iv = float(iv)
    horizon = int(getattr(args, "horizon", 21))
    lookback = int(getattr(args, "lookback", 120))

    try:
        receipt = evaluate_variance_risk_premium_profile(
            ticker=ticker,
            implied_volatility=iv,
            horizon_sessions=horizon,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Variance Risk Premium evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sig_str = "YES (p < 0.05)" if receipt.predictive_p_value < 0.05 else f"no (p = {receipt.predictive_p_value:.3f})"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Bollerslev et al. (2009) Variance Risk Premium Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Forecast Horizon:             {receipt.horizon_sessions} sessions (~1 trading month)")
    print(f"  Implied Volatility (IV):      {receipt.implied_volatility_pct:>8.2f}% (Variance: {receipt.implied_variance:.4f})")
    print(f"  Realized Volatility (RV):     {receipt.realized_volatility_pct:>8.2f}% (Variance: {receipt.realized_variance:.4f})")
    print("-" * 80)
    print("  Variance Risk Premium Spread & Volatility of Volatility:")
    print(f"    Variance Risk Premium (VRP):{receipt.variance_risk_premium:>+8.2f} pts ((IV² - RV²) * 100)")
    print(f"    Volatility Spread:          {receipt.volatility_spread_pct_points:>+8.2f} percentage points (IV - RV)")
    print(f"    Volatility-of-Vol (VoV):    {receipt.volatility_of_volatility_pct:>8.2f}% (Annualized VVIX proxy)")
    print(f"    VRP Regime:                 {receipt.vrp_regime}")
    print("-" * 80)
    print(f"  Bollerslev Forward Return Prediction ({receipt.horizon_sessions}-Session Forward Return):")
    print(f"    Predicted Forward Return:   {receipt.predicted_forward_return_pct:>+8.2f}% (R² = {receipt.r_squared_pct:.1f}%)")
    print(f"    Predictive Beta (β_VRP):    {receipt.predictive_beta_vrp:>+8.4f} (t={receipt.predictive_t_stat:>+6.2f}, Sig: {sig_str})")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def max_lottery_command(args: argparse.Namespace) -> None:
    """Evaluate Bali, Cakici & Whitelaw (2011) Extreme Daily Return MAX Effect & Lottery Demand."""
    import json
    from src.analytics.max_lottery import evaluate_max_lottery_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    top_n = int(getattr(args, "top_n", 5))
    lookback = int(getattr(args, "lookback", 60))

    try:
        receipt = evaluate_max_lottery_profile(
            ticker=ticker,
            top_n_days=top_n,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: MAX lottery evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Bali et al. (2011) MAX Effect & Lottery Demand Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Top N Days = {receipt.top_n_days})")
    print(f"  Daily Volatility:             {receipt.daily_volatility_pct:>8.1f}% Ann (Skewness: {receipt.return_skewness:>+6.2f})")
    print("-" * 80)
    print("  Extreme Daily Return Tail Order Statistics:")
    print(f"    Single Highest Return (MAX_1):{receipt.max_1_pct:>+8.2f}%")
    print(f"    Average Top {receipt.top_n_days} Return (MAX_{receipt.top_n_days}):{receipt.max_n_pct:>+8.2f}%")
    print(f"    Average Bottom {receipt.top_n_days} Return (MIN_{receipt.top_n_days}):{receipt.min_n_pct:>+8.2f}%")
    print(f"    Max-to-Min Asymmetry Ratio:   {receipt.max_to_min_ratio:>8.2f}x")
    print("-" * 80)
    print("  Behavioral Lottery Attraction & Expected Performance Penalty:")
    print(f"    Lottery Demand Score (Λ):     {receipt.lottery_demand_score:>8.1f} / 100")
    print(f"    Expected Annualized Alpha Drag:{receipt.expected_alpha_drag_ann_pct:>+8.2f}% / year")
    print(f"    Lottery Demand Regime:        {receipt.lottery_regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def profitability_command(args: argparse.Namespace) -> None:
    """Evaluate Robert Novy-Marx (2013) Gross Profitability Premium & Asness et al. (2019) QMJ."""
    import json
    from src.analytics.gross_profitability import evaluate_gross_profitability_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    rev = getattr(args, "revenue", None)
    if rev is not None:
        rev = float(rev)
    cost = getattr(args, "cogs", None)
    if cost is not None:
        cost = float(cost)
    assets = getattr(args, "assets", None)
    if assets is not None:
        assets = float(assets)
    debt = getattr(args, "debt", None)
    if debt is not None:
        debt = float(debt)

    try:
        receipt = evaluate_gross_profitability_profile(
            ticker=ticker,
            revenue=rev,
            cogs=cost,
            total_assets=assets,
            total_debt=debt,
        )
    except Exception as ex:
        print(f"\nError: Gross profitability evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    trap_str = "YES (Unprofitable/High Leverage)" if receipt.is_value_trap else "NO (Quality Compounder)"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Novy-Marx (2013) Gross Profitability & QMJ Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Total Revenue:                ${receipt.revenue:,.0f} (COGS: ${receipt.cogs:,.0f})")
    print(f"  Gross Profit:                 ${receipt.gross_profit:,.0f} (Margin: {receipt.gross_margin_pct:>5.1f}%)")
    print(f"  Total Book Assets:            ${receipt.total_assets:,.0f} (Debt: ${receipt.total_debt:,.0f})")
    print("-" * 80)
    print("  Novy-Marx Profitability to Capital & Turnover:")
    print(f"    Gross Profitability (GP/A): {receipt.gross_profitability_pct:>8.2f}% (Gross Profit / Assets)")
    print(f"    Asset Turnover (ATO):       {receipt.asset_turnover:>8.2f}x (Revenue / Assets)")
    print(f"    Debt-to-Assets Leverage:    {receipt.debt_to_assets_pct:>8.2f}%")
    print("-" * 80)
    print("  Asness et al. (2019) QMJ Composite Quality & Value-Trap Screening:")
    print(f"    Quality Score (Q):          {receipt.quality_score:>8.1f} / 100")
    print(f"    Value-Trap Risk Hazard:     {trap_str}")
    print(f"    Expected Annualized Alpha:  {receipt.expected_quality_alpha_ann_pct:>+8.2f}% / year")
    print(f"    Quality Factor Regime:      {receipt.quality_regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def dual_momentum_command(args: argparse.Namespace) -> None:
    """Evaluate Jegadeesh & Titman (1993, 2001) / Antonacci (2014) Dual Momentum Profile."""
    import json
    from src.analytics.dual_momentum import evaluate_dual_momentum_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    bench = validate_ticker(getattr(args, "benchmark", "SPY") or "SPY")
    rf_rate = float(getattr(args, "rf_rate", 4.0) or 4.0)

    try:
        receipt = evaluate_dual_momentum_profile(
            ticker=ticker,
            benchmark_ticker=bench,
            rf_rate_pct=rf_rate,
        )
    except Exception as ex:
        print(f"\nError: Dual momentum evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    crash_str = (
        "YES (High Past Winner + Severe 1M Drop)"
        if receipt.has_momentum_crash_risk
        else "NO (Stable Drift)"
    )
    abs_str = (
        "BULLISH (Positive Trend > Rf, >= SMA200)"
        if receipt.is_absolute_trend_positive
        else "BEARISH (Defensive / Cash Hedge)"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Jegadeesh & Titman (1993) / Antonacci Dual Momentum Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sessions Evaluated:           {receipt.sessions_evaluated} trading days")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f} (SMA 200: ${receipt.sma_200:.2f})")
    print("-" * 80)
    print("  Jegadeesh & Titman (1993, 2001) Intermediate Momentum & Microstructure Reversal:")
    print(f"    12-2 Intermediate Momentum: {receipt.momentum_12_2_pct:>+8.2f}% (t-252 to t-21, skipping 1M)")
    print(f"    6-1 Intermediate Momentum:  {receipt.momentum_6_1_pct:>+8.2f}% (t-126 to t-21, skipping 1M)")
    print(f"    1-Month Reversal (R_1,0):   {receipt.reversal_1_0_pct:>+8.2f}% (Microstructure bounce check)")
    print(f"    Unlagged 12M Total Return:  {receipt.total_return_12m_pct:>+8.2f}%")
    print("-" * 80)
    print("  Cross-Sectional Relative Strength & Moskowitz-Grinblatt Decomposition:")
    print(f"    Benchmark 12-2 Momentum:    {receipt.benchmark_12_2_pct:>+8.2f}%")
    print(f"    Relative Momentum Spread:   {receipt.relative_momentum_spread_pct:>+8.2f}% vs {receipt.benchmark_ticker}")
    print(f"    Idiosyncratic Alpha Mom:    {receipt.idiosyncratic_momentum_pct:>+8.2f}% (Market-beta adjusted)")
    print(f"    Realized Market Beta:       {receipt.beta_to_benchmark:>8.2f}")
    print(f"    Volatility-Adjusted Sharpe: {receipt.sharpe_momentum:>8.2f} (Ann Vol: {receipt.annualized_volatility_pct:.1f}%)")
    print("-" * 80)
    print("  Gary Antonacci Dual Momentum & Crash Risk Overlay:")
    print(f"    Absolute Trend Filter:      {abs_str}")
    print(f"    Momentum Crash Risk:        {crash_str}")
    print(f"    Expected Annualized Alpha:  {receipt.expected_momentum_alpha_pct:>+8.2f}% / year")
    print(f"    Momentum Regime:            {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def bab_command(args: argparse.Namespace) -> None:
    """Evaluate Andrea Frazzini & Lasse Heje Pedersen (2014) Betting Against Beta (BAB) Profile."""
    import json
    from src.analytics.betting_against_beta import evaluate_bab_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    bench = validate_ticker(getattr(args, "benchmark", "SPY") or "SPY")
    rf_rate = float(getattr(args, "rf_rate", 4.0) or 4.0)
    erp = float(getattr(args, "market_premium", 6.0) or 6.0)

    try:
        receipt = evaluate_bab_profile(
            ticker=ticker,
            benchmark_ticker=bench,
            rf_rate_pct=rf_rate,
            equity_risk_premium_pct=erp,
        )
    except Exception as ex:
        print(f"\nError: Betting Against Beta evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Frazzini & Pedersen (2014) Betting Against Beta (BAB) Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sessions Evaluated:           {receipt.sessions_evaluated} trading days")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Systematic Market Risk & Vasicek (1973) Bayesian Shrinkage:")
    print(f"    Raw Sample Beta (β_raw):    {receipt.raw_beta:>8.2f} (ρ = {receipt.correlation_to_benchmark:.2f})")
    print(f"    Vasicek Shrunk Beta (β*):   {receipt.shrunk_beta:>8.2f} (0.60·β_raw + 0.40·1.0)")
    print(f"    Asset Realized Volatility:  {receipt.asset_annualized_volatility_pct:>8.1f}% Ann (Bench Vol: {receipt.benchmark_annualized_volatility_pct:.1f}%)")
    print("-" * 80)
    print("  Security Market Line (SML) & Jensen's Alpha:")
    print(f"    Realized Annual Return:     {receipt.realized_annualized_return_pct:>+8.2f}% / year")
    print(f"    CAPM Expected Return:       {receipt.capm_expected_return_pct:>+8.2f}% / year (Rf={rf_rate:.1f}%, ERP={erp:.1f}%)")
    print(f"    Jensen's Alpha (α_i):       {receipt.jensen_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print("  Frazzini-Pedersen Leverage Multiplier & BAB Spread:")
    print(f"    Optimal BAB Multiplier:     {receipt.bab_leverage_multiplier:>8.2f}x (1.0 / β*)")
    print(f"    Expected BAB Factor Alpha:  {receipt.expected_bab_spread_pct:>+8.2f}% / year")
    print(f"    Low-Beta Compounder:        {'YES' if receipt.is_low_beta_compounder else 'NO'}")
    print(f"    High-Beta Overpriced Drag:  {'YES (Overpriced)' if receipt.is_high_beta_overpriced else 'NO'}")
    print(f"    BAB Factor Regime:          {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def reversal_command(args: argparse.Namespace) -> None:
    """Evaluate Narasimhan Jegadeesh (1990) & Bruce Lehmann (1990) Short-Term Reversal Profile."""
    import json
    from src.analytics.short_term_reversal import evaluate_short_term_reversal_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    bench = validate_ticker(getattr(args, "benchmark", "SPY") or "SPY")

    try:
        receipt = evaluate_short_term_reversal_profile(
            ticker=ticker,
            benchmark_ticker=bench,
        )
    except Exception as ex:
        print(f"\nError: Short-term reversal evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    dip_str = (
        "YES (High-Conviction Pullback in Bull Trend)"
        if receipt.is_liquidity_dip_buy
        else "NO"
    )
    knife_str = (
        "YES (Avoid - Falling Knife in Bear Trend)"
        if receipt.is_falling_knife
        else "NO"
    )
    trend_str = (
        "BULLISH (12-2 Mom > 0, Price >= SMA200)"
        if receipt.is_bullish_intermediate_trend
        else "BEARISH (Downtrend)"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Jegadeesh & Lehmann (1990) Short-Term Reversal Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sessions Evaluated:           {receipt.sessions_evaluated} trading days")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Short-Term Weekly Returns & Microstructure Serial Correlation:")
    print(f"    1-Week Return (R_1w):       {receipt.return_1w_pct:>+8.2f}% (t-5 to t)")
    print(f"    2-Week Return (R_2w):       {receipt.return_2w_pct:>+8.2f}% (t-10 to t)")
    print(f"    Benchmark 1-Week Return:    {receipt.benchmark_1w_pct:>+8.2f}%")
    print(f"    Excess 1-Week Return:       {receipt.excess_return_1w_pct:>+8.2f}% vs {receipt.benchmark_ticker}")
    print(f"    Serial Correlation (ρ_1):   {receipt.serial_correlation_rho1:>8.2f} (Daily first-order autocovariance)")
    print(f"    Roll (1984) Spread Proxy:   {receipt.roll_spread_proxy_pct:>8.2f}% (Bid-ask bounce cost)")
    print(f"    Volume Shock Z-Score:       {receipt.volume_shock_zscore:>8.2f}σ (Liquidity selling intensity)")
    print("-" * 80)
    print("  Trend Confluence & Liquidity Dip Filtering:")
    print(f"    12-2 Intermediate Momentum: {receipt.intermediate_momentum_12_2_pct:>+8.2f}%")
    print(f"    Macro Trend Status:         {trend_str}")
    print(f"    Liquidity Dip Buy Signal:   {dip_str}")
    print(f"    Falling Knife Warning:      {knife_str}")
    print(f"    Expected Rebound Alpha:     {receipt.expected_contrarian_alpha_pct:>+8.2f}% / year")
    print(f"    Reversal Regime:            {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def distress_command(args: argparse.Namespace) -> None:
    """Evaluate John Y. Campbell et al. (2008) Financial Distress Risk Hazard Profile."""
    import json
    from src.analytics.distress_risk import evaluate_distress_risk_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    mcap = getattr(args, "market_cap", None)
    if mcap is not None:
        mcap = float(mcap)
    debt = getattr(args, "debt", None)
    if debt is not None:
        debt = float(debt)
    ni = getattr(args, "net_income", None)
    if ni is not None:
        ni = float(ni)
    cash = getattr(args, "cash", None)
    if cash is not None:
        cash = float(cash)

    try:
        receipt = evaluate_distress_risk_profile(
            ticker=ticker,
            market_cap=mcap,
            total_liabilities=debt,
            net_income=ni,
            cash_equivalents=cash,
        )
    except Exception as ex:
        print(f"\nError: Financial distress evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    safe_str = "YES (Impeccable Solvency)" if receipt.is_solvent_safe_haven else "NO"
    trap_str = "YES (High Failure Risk - Avoid Zombie)" if receipt.is_distress_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Campbell, Hilscher & Szilagyi (2008) Distress Risk Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.0f}")
    print(f"  Market-Adjusted Assets (MTA): ${receipt.market_adjusted_assets:,.0f}")
    print("-" * 80)
    print("  Campbell et al. Dynamic Multi-Period Balance Sheet & Market Covariates:")
    print(f"    Total Liabilities / MTA:    {receipt.tlmta_pct:>8.2f}% (Market-implied leverage)")
    print(f"    Net Income / MTA (NIMTA):   {receipt.nimta_pct:>+8.2f}% (Operating capital efficiency)")
    print(f"    Cash Buffer / MTA (CASHMTA):{receipt.cashmta_pct:>8.2f}% (Liquidity cushion)")
    print(f"    Annualized Volatility:      {receipt.annualized_volatility_pct:>8.1f}% Ann (SIGMA)")
    print(f"    Excess Trailing Return:     {receipt.excess_trailing_return_pct:>+8.2f}% (EXRET momentum)")
    print("-" * 80)
    print("  Dynamic Failure Hazard & Financial Distress Puzzle Spread:")
    print(f"    CHS Linear Logit Score (z): {receipt.chs_linear_index:>8.2f}")
    print(f"    12-Month Failure Hazard (P):{receipt.distress_probability_12m_pct:>8.2f}% (Default/Bankruptcy probability)")
    print(f"    Solvent Safe Haven:         {safe_str}")
    print(f"    Distress Risk Trap:         {trap_str}")
    print(f"    Expected Annualized Alpha:  {receipt.expected_distress_alpha_pct:>+8.2f}% / year")
    print(f"    Solvency Factor Regime:     {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def sentiment_command(args: argparse.Namespace) -> None:
    """Evaluate Malcolm Baker & Jeffrey Wurgler (2006) Cross-Sectional Sentiment Profile."""
    import json
    from src.analytics.investor_sentiment import evaluate_investor_sentiment_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    sent_idx = float(getattr(args, "market_sentiment", 0.0) or 0.0)
    margin = getattr(args, "net_margin", None)
    if margin is not None:
        margin = float(margin)
    mcap = getattr(args, "market_cap", None)
    if mcap is not None:
        mcap = float(mcap)

    try:
        receipt = evaluate_investor_sentiment_profile(
            ticker=ticker,
            market_sentiment_index=sent_idx,
            net_profit_margin_pct=margin,
            market_cap=mcap,
        )
    except Exception as ex:
        print(f"\nError: Investor sentiment evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    bubble_str = (
        "YES (Euphoric Sentiment + High Speculation)"
        if receipt.is_speculative_bubble
        else "NO"
    )
    bargain_str = (
        "YES (Panicked Sentiment + Oversold Speculation)"
        if receipt.is_contrarian_bargain
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Baker & Wurgler (2006) Investor Sentiment Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.0f}")
    print("-" * 80)
    print("  Baker & Wurgler Subjectivity & Limits to Arbitrage Covariates:")
    print(f"    Annualized Volatility:      {receipt.annualized_volatility_pct:>8.1f}% Ann (Arbitrage risk)")
    print(f"    Net Profit Margin:          {receipt.net_profit_margin_pct:>+8.1f}% (Cash flow subjectivity)")
    print(f"    MAX_5 Lottery Demand:       {receipt.max_daily_return_5d_pct:>8.2f}% (Retail speculation)")
    print(f"    Firm Speculative Index (S): {receipt.speculative_index:>8.1f} / 100")
    print("-" * 80)
    print("  Cross-Sectional Sentiment Sensitivity & Reversal Dynamics:")
    print(f"    Macro Sentiment Index (I):  {receipt.market_sentiment_index:>+8.2f}σ (Standardized wave)")
    print(f"    Equity Sentiment Beta:      {receipt.sentiment_beta:>+8.2f} (Sensitivity to sentiment)")
    print(f"    Speculative Bubble Risk:    {bubble_str}")
    print(f"    Contrarian Bargain Rebound: {bargain_str}")
    print(f"    Expected Sentiment Alpha:   {receipt.expected_sentiment_alpha_pct:>+8.2f}% / year")
    print(f"    Sentiment Factor Regime:    {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def arbitrage_asymmetry_command(args: argparse.Namespace) -> None:
    """Evaluate Robert F. Stambaugh, Jianfeng Yu & Yu Yuan (2012, 2015) Short-Sale Impediments & Arbitrage Asymmetry."""
    import json
    from src.analytics.arbitrage_asymmetry import evaluate_arbitrage_asymmetry_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    sent_idx = float(getattr(args, "market_sentiment", 0.0) or 0.0)
    borrow_fee = getattr(args, "borrow_fee", None)
    if borrow_fee is not None:
        borrow_fee = float(borrow_fee)
    short_interest = getattr(args, "short_interest", None)
    if short_interest is not None:
        short_interest = float(short_interest)
    dtc = getattr(args, "days_to_cover", None)
    if dtc is not None:
        dtc = float(dtc)

    try:
        receipt = evaluate_arbitrage_asymmetry_profile(
            ticker=ticker,
            market_sentiment_index=sent_idx,
            borrow_fee_annual_pct=borrow_fee,
            short_interest_pct_float=short_interest,
            days_to_cover=dtc,
        )
    except Exception as ex:
        print(f"\nError: Arbitrage asymmetry evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    severe_str = (
        "YES (High Overpricing Hazard)"
        if receipt.is_severe_overpriced
        else "NO"
    )
    squeeze_str = (
        "YES (Acute Short Squeeze Risk)"
        if receipt.is_squeeze_risk
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Stambaugh, Yu & Yuan (2012) Arbitrage Asymmetry Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.0f}")
    print("-" * 80)
    print("  Short-Sale Impediment & Friction Covariates:")
    print(f"    Idiosyncratic Volatility:   {receipt.idiosyncratic_volatility_pct:>8.1f}% Ann (Arbitrage risk)")
    print(f"    Annualized Borrow Fee:      {receipt.borrow_fee_annual_pct:>8.2f}% / yr (Cost to borrow)")
    print(f"    Short Interest (% Float):   {receipt.short_interest_pct_float:>8.1f}% of float")
    print(f"    Days to Cover (DTC):        {receipt.days_to_cover:>8.1f} sessions")
    print(f"    Daily Share Turnover:       {receipt.daily_turnover_pct:>8.2f}% per session")
    print(f"    Short Impediment Index (I): {receipt.short_sale_impediment_index:>8.1f} / 100")
    print("-" * 80)
    print("  Arbitrage Asymmetry & Overpricing Dynamics:")
    print(f"    Macro Sentiment Index (I):  {receipt.market_sentiment_index:>+8.2f}σ (Standardized)")
    print(f"    Overpricing Hazard Score:   {receipt.asymmetric_overpricing_hazard:>8.1f} / 100")
    print(f"    Short Squeeze Vulnerability:{receipt.short_squeeze_vulnerability:>8.1f} / 100")
    print(f"    Severe Overpriced Drag:     {severe_str}")
    print(f"    Short Squeeze Powder Keg:   {squeeze_str}")
    print(f"    Projected Mispricing Drag:  {receipt.expected_asymmetry_alpha_pct:>+8.2f}% / year")
    print(f"    Arbitrage Asymmetry Regime: {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def q_factor_command(args: argparse.Namespace) -> None:
    """Evaluate Kewei Hou, Chen Xue & Lu Zhang (2015, 2020) q-Factor Investment Profile."""
    import json
    from src.analytics.q_factor import evaluate_q_factor_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    roe = getattr(args, "roe", None)
    if roe is not None:
        roe = float(roe)
    inv = getattr(args, "investment_rate", None)
    if inv is None:
        inv = getattr(args, "asset_growth", None)
    if inv is not None:
        inv = float(inv)
    tobin = getattr(args, "tobin_q", None)
    if tobin is not None:
        tobin = float(tobin)

    try:
        receipt = evaluate_q_factor_profile(
            ticker=ticker,
            return_on_equity_pct=roe,
            investment_to_assets_pct=inv,
            tobin_q_proxy=tobin,
        )
    except Exception as ex:
        print(f"\nError: q-Factor evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    compounder_str = (
        "YES (High ROE + Conservative Investment)"
        if receipt.is_disciplined_compounder
        else "NO"
    )
    overinvestor_str = (
        "YES (Low/Negative ROE + Aggressive Asset Expansion)"
        if receipt.is_overinvestor_trap
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hou, Xue & Zhang (2015) q-Factor Investment Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.0f}")
    print("-" * 80)
    print("  Neoclassical Investment & Profitability Covariates:")
    print(f"    Return on Equity (ROE):     {receipt.return_on_equity_pct:>+8.1f}% (Marginal profitability)")
    print(f"    Investment-to-Assets (I/A): {receipt.investment_to_assets_pct:>+8.1f}% Ann (Asset growth rate)")
    print(f"    Tobin's q Proxy:            {receipt.tobin_q_proxy:>8.2f} (Market-to-book assets)")
    print(f"    Operating Cash Flow/Assets: {receipt.cash_flow_to_assets_pct:>+8.1f}%")
    print(f"    Profitability Score (ROE):  {receipt.profitability_score:>8.1f} / 100")
    print(f"    Investment Score (I/A):     {receipt.investment_score:>8.1f} / 100 (Lower = More Conservative)")
    print("-" * 80)
    print("  Hou-Xue-Zhang Factor Dynamics & Alpha Projections:")
    print(f"    Composite q-Alpha Score:    {receipt.q_alpha_score:>8.1f} / 100")
    print(f"    Disciplined Compounder:     {compounder_str}")
    print(f"    Over-Investment Trap:       {overinvestor_str}")
    print(f"    Expected q-Factor Alpha:    {receipt.expected_q_alpha_pct:>+8.2f}% / year")
    print(f"    Investment Factor Regime:   {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def accruals_command(args: argparse.Namespace) -> None:
    """Evaluate Richard G. Sloan (1996) Accrual Anomaly & Earnings Quality Decomposition."""
    import json
    from src.analytics.accrual_anomaly import evaluate_accrual_anomaly_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    ni = getattr(args, "net_income", None)
    if ni is not None:
        ni = float(ni)
    cfo = getattr(args, "operating_cash_flow", None)
    if cfo is None:
        cfo = getattr(args, "cash_flow", None)
    if cfo is not None:
        cfo = float(cfo)
    assets = getattr(args, "total_assets", None)
    if assets is not None:
        assets = float(assets)

    try:
        receipt = evaluate_accrual_anomaly_profile(
            ticker=ticker,
            net_income=ni,
            operating_cash_flow=cfo,
            average_total_assets=assets,
        )
    except Exception as ex:
        print(f"\nError: Accrual anomaly evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    compounder_str = (
        "YES (High Cash Conversion & Persistent Earnings)"
        if receipt.is_cash_quality_compounder
        else "NO"
    )
    trap_str = (
        "YES (Aggressive Accrual Distortion Trap)"
        if receipt.is_accrual_trap
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Richard G. Sloan (1996) Accrual Anomaly Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Average Total Assets:         ${receipt.average_total_assets:,.0f}")
    print("-" * 80)
    print("  Earnings Decomposition & Cash Conversion Covariates:")
    print(f"    Reported Net Income:        ${receipt.net_income:,.0f}")
    print(f"    Operating Cash Flow:        ${receipt.operating_cash_flow:,.0f}")
    print(f"    Total Accounting Accruals:  ${receipt.total_accruals:>+15,.0f} (Net Income - Cash Flow)")
    print(f"    Accruals / Assets (CFA):    {receipt.accrual_ratio_pct:>+8.2f}% (Normalized Accrual Ratio)")
    print(f"    Cash Flow / Net Income:     {receipt.cash_flow_to_net_income:>8.2f}x (Cash Conversion Multiplier)")
    print(f"    Earnings Quality Index:     {receipt.earnings_quality_index:>8.1f} / 100")
    print("-" * 80)
    print("  Sloan Accrual Fixation & Alpha Spread Dynamics:")
    print(f"    Cash Quality Compounder:    {compounder_str}")
    print(f"    Accrual Inflation Trap:     {trap_str}")
    print(f"    Expected Accrual Alpha:     {receipt.expected_accrual_alpha_pct:>+8.2f}% / year")
    print(f"    Earnings Quality Regime:    {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def beneish_command(args: argparse.Namespace) -> None:
    """Evaluate Messod D. Beneish (1999) Forensic Accounting M-Score & Manipulation Risk."""
    import json
    from src.analytics.beneish_mscore import evaluate_beneish_mscore_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    dsri = getattr(args, "dsri", None)
    if dsri is not None:
        dsri = float(dsri)
    gmi = getattr(args, "gmi", None)
    if gmi is not None:
        gmi = float(gmi)
    aqi = getattr(args, "aqi", None)
    if aqi is not None:
        aqi = float(aqi)
    sgi = getattr(args, "sgi", None)
    if sgi is not None:
        sgi = float(sgi)
    depi = getattr(args, "depi", None)
    if depi is not None:
        depi = float(depi)
    sgai = getattr(args, "sgai", None)
    if sgai is not None:
        sgai = float(sgai)
    lvgi = getattr(args, "lvgi", None)
    if lvgi is not None:
        lvgi = float(lvgi)
    tata = getattr(args, "tata", None)
    if tata is not None:
        tata = float(tata)

    try:
        receipt = evaluate_beneish_mscore_profile(
            ticker=ticker,
            dsri=dsri,
            gmi=gmi,
            aqi=aqi,
            sgi=sgi,
            depi=depi,
            sgai=sgai,
            lvgi=lvgi,
            tata=tata,
        )
    except Exception as ex:
        print(f"\nError: Beneish M-Score evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    fraud_str = (
        "YES (HIGH MANIPULATION / FRAUD PROBABILITY)"
        if receipt.is_manipulator_risk
        else "NO"
    )
    clean_str = (
        "YES (PRISTINE UNMANIPULATED FINANCIALS)"
        if receipt.is_clean_profile
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Messod D. Beneish (1999) Forensic Accounting M-Score")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Beneish 8-Variable Forensic Covariates:")
    print(f"    Days' Sales in Receivables: {receipt.dsri:>8.2f} (DSRI: Revenue acceleration)")
    print(f"    Gross Margin Index:         {receipt.gmi:>8.2f} (GMI: Margin deterioration)")
    print(f"    Asset Quality Index:        {receipt.aqi:>8.2f} (AQI: Expense capitalization)")
    print(f"    Sales Growth Index:         {receipt.sgi:>8.2f} (SGI: Deceleration pressure)")
    print(f"    Depreciation Index:         {receipt.depi:>8.2f} (DEPI: Asset life extension)")
    print(f"    SG&A Expense Index:         {receipt.sgai:>8.2f} (SGAI: Operational inefficiency)")
    print(f"    Leverage Index:             {receipt.lvgi:>8.2f} (LVGI: Debt covenant pressure)")
    print(f"    Total Accruals to Assets:   {receipt.tata:>+8.3f} (TATA: Sloan paper profits)")
    print("-" * 80)
    print("  Forensic Model Inferences & Manipulation Probabilities:")
    print(f"    Beneish M-Score:            {receipt.m_score:>+8.2f} (Cutoff: > -1.78)")
    print(f"    Manipulation Probability:   {receipt.manipulation_probability_pct:>8.1f}% (Phi(M))")
    print(f"    Manipulator Red Flag:       {fraud_str}")
    print(f"    Clean Financial Profile:    {clean_str}")
    print(f"    Expected Forensic Alpha:    {receipt.expected_forensic_alpha_pct:>+8.2f}% / year")
    print(f"    Forensic Regime:            {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def fscore_command(args: argparse.Namespace) -> None:
    """Evaluate Joseph D. Piotroski (2000) 9-Point Fundamental Accounting F-Score."""
    import json
    from src.analytics.piotroski_fscore import evaluate_piotroski_fscore_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    ni = getattr(args, "net_income", None)
    if ni is not None:
        ni = float(ni)
    cfo = getattr(args, "operating_cash_flow", None)
    if cfo is None:
        cfo = getattr(args, "cash_flow", None)
    if cfo is not None:
        cfo = float(cfo)
    assets = getattr(args, "total_assets", None)
    if assets is not None:
        assets = float(assets)
    roa = getattr(args, "roa", None)
    if roa is not None:
        roa = float(roa)
    d_roa = getattr(args, "delta_roa", None)
    if d_roa is not None:
        d_roa = float(d_roa)
    d_debt = getattr(args, "delta_debt", None)
    if d_debt is not None:
        d_debt = float(d_debt)
    d_liq = getattr(args, "delta_current_ratio", None)
    if d_liq is not None:
        d_liq = float(d_liq)
    dilut = getattr(args, "shares_dilution", None)
    if dilut is not None:
        dilut = float(dilut)
    d_margin = getattr(args, "delta_gross_margin", None)
    if d_margin is not None:
        d_margin = float(d_margin)
    d_turnover = getattr(args, "delta_asset_turnover", None)
    if d_turnover is not None:
        d_turnover = float(d_turnover)

    try:
        receipt = evaluate_piotroski_fscore_profile(
            ticker=ticker,
            net_income=ni,
            operating_cash_flow=cfo,
            total_assets=assets,
            return_on_assets_pct=roa,
            delta_roa_pct=d_roa,
            delta_long_term_debt=d_debt,
            delta_current_ratio=d_liq,
            shares_dilution_pct=dilut,
            delta_gross_margin_pct=d_margin,
            delta_asset_turnover=d_turnover,
        )
    except Exception as ex:
        print(f"\nError: Piotroski F-Score evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    compounder_str = (
        "YES (HIGH CONVICTION FUNDAMENTAL COMPOUNDER)"
        if receipt.is_strong_compounder
        else "NO"
    )
    trap_str = (
        "YES (HIGH RISK VALUE TRAP / DISTRESS)"
        if receipt.is_value_trap
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Joseph D. Piotroski (2000) Fundamental F-Score Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Aggregate Piotroski F-Score:  {receipt.f_score} / 9 Points ({receipt.regime})")
    print("  Fundamental Subscore Breakdown:")
    print(f"    1. Profitability Signals:   {receipt.profitability_points} / 4 Points")
    print(f"       - Return on Assets > 0:       {'PASS [1]' if receipt.f_roa else 'FAIL [0]'} (ROA: {receipt.return_on_assets_pct:+.1f}%)")
    print(f"       - Cash Flow from Ops > 0:     {'PASS [1]' if receipt.f_cfo else 'FAIL [0]'} (CFO: ${receipt.operating_cash_flow:,.0f})")
    print(f"       - Improving ROA (ΔROA > 0):   {'PASS [1]' if receipt.f_delta_roa else 'FAIL [0]'} (ΔROA: {receipt.delta_roa_pct:+.1f}%)")
    print(f"       - Accrual Quality (CFO > NI): {'PASS [1]' if receipt.f_accrual else 'FAIL [0]'} (Net Income: ${receipt.net_income:,.0f})")
    print(f"    2. Leverage & Liquidity:    {receipt.leverage_liquidity_points} / 3 Points")
    print(f"       - De-leveraging (ΔDebt <= 0): {'PASS [1]' if receipt.f_delta_leverage else 'FAIL [0]'}")
    print(f"       - Expanding Liquidity (ΔCR):  {'PASS [1]' if receipt.f_delta_liquidity else 'FAIL [0]'}")
    print(f"       - Zero Share Dilution:        {'PASS [1]' if receipt.f_no_dilution else 'FAIL [0]'}")
    print(f"    3. Operating Efficiency:    {receipt.operating_efficiency_points} / 2 Points")
    print(f"       - Expanding Margin (ΔMargin): {'PASS [1]' if receipt.f_delta_margin else 'FAIL [0]'}")
    print(f"       - Higher Turnover (ΔTurnover):{'PASS [1]' if receipt.f_delta_turnover else 'FAIL [0]'}")
    print("-" * 80)
    print("  Valuation Health & Alpha Spread Dynamics:")
    print(f"    Strong Compounder Quality:  {compounder_str}")
    print(f"    Vulnerable Value Trap Trap: {trap_str}")
    print(f"    Expected Fundamental Alpha: {receipt.expected_fundamental_alpha_pct:>+8.2f}% / year")
    print(f"    Fundamental Health Regime:  {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def diffusion_command(args: argparse.Namespace) -> None:
    """Evaluate Hong & Stein (1999) / Chordia & Shivakumar (2002) Gradual Information Diffusion."""
    import json
    from src.analytics.gradual_diffusion import evaluate_gradual_diffusion_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    coverage = getattr(args, "analyst_coverage", None)
    if coverage is not None:
        coverage = int(coverage)

    try:
        receipt = evaluate_gradual_diffusion_profile(
            ticker=ticker,
            analyst_coverage=coverage,
        )
    except Exception as ex:
        print(f"\nError: Hong-Stein Diffusion evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    underreaction_str = (
        "YES (HIGH CONVICTION CONTINUATION MOMENTUM)"
        if receipt.is_early_underreaction
        else "NO"
    )
    overreaction_str = (
        "YES (HIGH REVERSAL / CROWDING CRASH RISK)"
        if receipt.is_crowded_overreaction
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hong & Stein (1999) Gradual Information Diffusion Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Cumulative Momentum:          {receipt.cumulative_momentum_pct:>+8.2f}% (Annual Volatility: {receipt.annualized_volatility_pct:.1f}%)")
    print(f"  Daily Turnover Ratio:         {receipt.turnover_ratio_pct:.2f}%")
    print(f"  Information Diffusion (IDV):  {receipt.information_diffusion_velocity:.2f} / 1.00 (Speed of news permeation)")
    print(f"  Trend Maturity Index (TMI):   {receipt.trend_maturity_index:.2f} / 1.00 (Speculative exhaustion)")
    print(f"  Momentum Crowding Risk (MCR): {receipt.momentum_crowding_risk_pct:.1f}%")
    print("-" * 80)
    print("  Behavioral Regime & Alpha Dynamics:")
    print(f"    Behavioral Regime:          {receipt.regime}")
    print(f"    Early Underreaction Phase:  {underreaction_str}")
    print(f"    Crowded Overreaction Peak:  {overreaction_str}")
    print(f"    Expected Hong-Stein Alpha:  {receipt.expected_hong_stein_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def contrarian_command(args: argparse.Namespace) -> None:
    """Evaluate Josef Lakonishok, Andrei Shleifer & Robert W. Vishny (1994) Contrarian Investment."""
    import json
    from src.analytics.contrarian_extrapolation import evaluate_contrarian_extrapolation_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    sg = getattr(args, "sales_growth", None)
    if sg is not None:
        sg = float(sg)
    eg = getattr(args, "earnings_growth", None)
    if eg is not None:
        eg = float(eg)
    pe = getattr(args, "pe", None)
    if pe is not None:
        pe = float(pe)
    pcf = getattr(args, "pcf", None)
    if pcf is not None:
        pcf = float(pcf)
    pb = getattr(args, "pb", None)
    if pb is not None:
        pb = float(pb)

    try:
        receipt = evaluate_contrarian_extrapolation_profile(
            ticker=ticker,
            sales_growth_3yr_pct=sg,
            earnings_growth_3yr_pct=eg,
            price_to_earnings=pe,
            price_to_cash_flow=pcf,
            price_to_book=pb,
        )
    except Exception as ex:
        print(f"\nError: LSV Contrarian evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    contrarian_str = (
        "YES (HIGH CONVICTION CONTRARIAN VALUE BUY)"
        if receipt.is_contrarian_value
        else "NO"
    )
    glamour_str = (
        "YES (VULNERABLE GLAMOUR OVERPRICING TRAP)"
        if receipt.is_glamour_trap
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Lakonishok, Shleifer & Vishny (1994) Contrarian Investment Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Sales Growth (3-Yr Ann):      {receipt.sales_growth_3yr_pct:>+8.2f}%")
    print(f"  Earnings Growth (3-Yr Ann):   {receipt.earnings_growth_3yr_pct:>+8.2f}%")
    print(f"  Trailing 1-Yr Momentum:       {receipt.trailing_momentum_pct:>+8.2f}%")
    print(f"  Valuation Multiples:          P/E: {receipt.price_to_earnings:.1f} | P/CF: {receipt.price_to_cash_flow:.1f} | P/B: {receipt.price_to_book:.1f}")
    print("-" * 80)
    print(f"  Naive Extrapolation (NEI):    {receipt.naive_extrapolation_index:.2f} / 1.00 (Degree of past growth extrapolation)")
    print(f"  Valuation Multiple (VMC):     {receipt.valuation_multiple_composite:.2f} / 1.00 (Normalized multiple composite)")
    print(f"  Expectation Discrepancy (ED): {receipt.expectation_discrepancy:>+8.2f} (VMC - NEI)")
    print("-" * 80)
    print("  Contrarian Regime & Alpha Dynamics:")
    print(f"    Behavioral Regime:          {receipt.regime}")
    print(f"    Contrarian Value Pick:      {contrarian_str}")
    print(f"    Vulnerable Glamour Trap:    {glamour_str}")
    print(f"    Expected LSV Alpha Spread:  {receipt.expected_contrarian_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def dhs_command(args: argparse.Namespace) -> None:
    """Evaluate Daniel, Hirshleifer & Subrahmanyam (1998) Overconfidence & Self-Attribution."""
    import json
    from src.analytics.dhs_overconfidence import evaluate_dhs_overconfidence_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    surprise = getattr(args, "earnings_surprise", None)
    if surprise is not None:
        surprise = float(surprise)
    dispersion = getattr(args, "dispersion", None)
    if dispersion is not None:
        dispersion = float(dispersion)

    try:
        receipt = evaluate_dhs_overconfidence_profile(
            ticker=ticker,
            earnings_surprise_pct=surprise,
            analyst_dispersion_pct=dispersion,
        )
    except Exception as ex:
        print(f"\nError: DHS Overconfidence evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    accel_str = (
        "YES (CONFIRMATORY PUBLIC NEWS AMPLIFICATION)"
        if receipt.is_attribution_accelerating
        else "NO"
    )
    fragile_str = (
        "YES (CONTRADICTORY PUBLIC NEWS CRASH HAZARD)"
        if receipt.is_contradiction_fragile
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Daniel, Hirshleifer & Subrahmanyam (1998) DHS Behavioral Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Trailing Momentum:            {receipt.trailing_momentum_pct:>+8.2f}% (Annual Volatility: {receipt.annualized_volatility_pct:.1f}%)")
    print(f"  Earnings Surprise:            {receipt.earnings_surprise_pct:>+8.2f}% (Analyst Dispersion: {receipt.analyst_dispersion_pct:.1f}%)")
    print(f"  Turnover Ratio:               {receipt.turnover_ratio_pct:.2f}%")
    print("-" * 80)
    print(f"  Overconfidence Intensity:     {receipt.overconfidence_intensity:.2f} / 1.00 (Private signal overestimation)")
    print(f"  Biased Self-Attribution (BAI):{receipt.biased_self_attribution_index:>+8.2f} (Confirmation > 0, Contradiction < 0)")
    print(f"  Reversal Crash Hazard:        {receipt.reversal_crash_hazard_pct:.1f}%")
    print("-" * 80)
    print("  Behavioral Life-Cycle & Alpha Dynamics:")
    print(f"    Life-Cycle Regime:          {receipt.regime}")
    print(f"    Self-Attribution Accelerating: {accel_str}")
    print(f"    Contradiction Fragility:       {fragile_str}")
    print(f"    Expected DHS Alpha Spread:  {receipt.expected_dhs_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def bsv_command(args: argparse.Namespace) -> None:
    """Evaluate Barberis, Shleifer & Vishny (1998) Dual-Heuristic Investor Sentiment."""
    import json
    from src.analytics.bsv_sentiment import evaluate_bsv_sentiment_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    surprise = getattr(args, "earnings_surprise", None)
    if surprise is not None:
        surprise = float(surprise)
    streak = getattr(args, "streak", None)
    if streak is not None:
        streak = int(streak)

    try:
        receipt = evaluate_bsv_sentiment_profile(
            ticker=ticker,
            latest_surprise_pct=surprise,
            consecutive_streak=streak,
        )
    except Exception as ex:
        print(f"\nError: BSV Sentiment evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    pead_str = (
        "YES (HIGH POST-EARNINGS UNDERREACTION DRIFT)"
        if receipt.is_conservatism_drift
        else "NO"
    )
    overreact_str = (
        "YES (SPECULATIVE STREAK OVERREACTION / REVERSAL)"
        if receipt.is_representativeness_peak
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Barberis, Shleifer & Vishny (1998) Dual-Heuristic Sentiment")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Trailing Momentum:            {receipt.trailing_momentum_pct:>+8.2f}% (Annual Volatility: {receipt.annualized_volatility_pct:.1f}%)")
    print(f"  Latest Earnings Surprise:     {receipt.latest_earnings_surprise_pct:>+8.2f}%")
    print(f"  Consecutive Streak Length:    {receipt.consecutive_streak_quarters} quarters")
    print("-" * 80)
    print(f"  Trending State Prob (pi_t):   {receipt.trending_state_probability:.2f} / 1.00 (Subjective belief in persistent trend)")
    print(f"  Conservatism Drift Index:     {receipt.conservatism_drift_index:.2f} / 1.00 (PEAD underreaction magnitude)")
    print(f"  Representativeness Score:     {receipt.representativeness_streak_score:.2f} / 1.00 (Streak over-extrapolation)")
    print("-" * 80)
    print("  Cognitive Bias Regime & Alpha Dynamics:")
    print(f"    Behavioral Regime:          {receipt.regime}")
    print(f"    Conservatism Underreaction: {pead_str}")
    print(f"    Representativeness Peak:    {overreact_str}")
    print(f"    Expected BSV Alpha Spread:  {receipt.expected_bsv_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def innovation_command(args: argparse.Namespace) -> None:
    """Evaluate Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Misvaluation."""
    import json
    from src.analytics.innovation_capital import evaluate_innovation_capital_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    rd = getattr(args, "rd_expense", None)
    if rd is not None:
        rd = float(rd)
    rev = getattr(args, "revenues", None)
    if rev is not None:
        rev = float(rev)
    assets = getattr(args, "total_assets", None)
    if assets is not None:
        assets = float(assets)

    try:
        receipt = evaluate_innovation_capital_profile(
            ticker=ticker,
            current_rd_expense=rd,
            annual_revenues=rev,
            total_assets=assets,
        )
    except Exception as ex:
        print(f"\nError: Innovation Capital evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    innovator_str = (
        "YES (HIGH CONVICTION INTANGIBLE INNOVATION ALPHA)"
        if receipt.is_high_innovation_alpha
        else "NO"
    )
    laggard_str = (
        "YES (HIGH OBSOLESCENCE RISK / R&D DEFICIT)"
        if receipt.is_innovation_laggard
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_capitalization:,.0f}")
    print("-" * 80)
    print(f"  Annual R&D Expenditure:       ${receipt.current_rd_expense:,.0f}")
    print(f"  Capitalized Knowledge Stock:  ${receipt.capitalized_knowledge_stock:,.0f} (Perpetual inventory, delta=20%)")
    print(f"  R&D-to-Market Ratio (RD/M):   {receipt.rd_to_market_pct:.2f}%")
    print(f"  R&D-to-Sales Intensity (RD/S):{receipt.rd_to_sales_pct:.2f}%")
    print(f"  Knowledge-to-Assets Ratio:    {receipt.knowledge_to_assets_pct:.2f}%")
    print("-" * 80)
    print(f"  Innovation Misvaluation Index:{receipt.innovation_misvaluation_index:.2f} / 1.00 (Unpriced intangible patent power)")
    print("  Asset Pricing Regime & Alpha Dynamics:")
    print(f"    Behavioral Regime:          {receipt.regime}")
    print(f"    High Conviction Innovator:  {innovator_str}")
    print(f"    Innovation-Starved Laggard: {laggard_str}")
    print(f"    Expected Innovation Alpha:  {receipt.expected_innovation_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def lifecycle_command(args: argparse.Namespace) -> None:
    """Evaluate Charles M. C. Lee & Bhaskaran Swaminathan (2000) Momentum Life Cycle."""
    import json
    from src.analytics.momentum_lifecycle import evaluate_momentum_lifecycle_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    turnover = getattr(args, "turnover", None)
    if turnover is not None:
        turnover = float(turnover)

    try:
        receipt = evaluate_momentum_lifecycle_profile(
            ticker=ticker,
            turnover_ratio_override=turnover,
        )
    except Exception as ex:
        print(f"\nError: Momentum Life Cycle evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    s1_str = "YES (STAGE 1 NEGLECTED WINNER - HIGH PERSISTENCE)" if receipt.is_neglected_winner else "NO"
    s2_str = "YES (STAGE 2 CROWDED WINNER - EXHAUSTION REVERSAL)" if receipt.is_crowded_winner else "NO"
    s3_str = "YES (STAGE 3 CAPITULATION LOSER - REBOUND BOUNCE)" if receipt.is_capitulation_loser else "NO"
    s4_str = "YES (STAGE 4 NEGLECTED ZOMBIE - PERSISTENT DECAY)" if receipt.is_neglected_loser else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Lee & Swaminathan (2000) Momentum Life-Cycle Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Trailing 12-M Momentum:       {receipt.trailing_momentum_pct:>+8.2f}% (Annual Volatility: {receipt.annualized_volatility_pct:.1f}%)")
    print(f"  Daily Turnover Ratio:         {receipt.daily_turnover_ratio_pct:.2f}% (Shares Transacted Daily)")
    print(f"  Volume-Momentum Score (VMIS): {receipt.volume_momentum_interaction_score:>+8.2f} (Underreaction > 0, Crowding < 0)")
    print("-" * 80)
    print("  Momentum Life-Cycle Stage & Alpha Dynamics:")
    print(f"    Life-Cycle Stage:           {receipt.regime}")
    print(f"    Stage 1 Neglected Winner:   {s1_str}")
    print(f"    Stage 2 Crowded Winner:     {s2_str}")
    print(f"    Stage 3 Capitulation Loser: {s3_str}")
    print(f"    Stage 4 Neglected Zombie:   {s4_str}")
    print(f"    Expected MLC Alpha Spread:  {receipt.expected_mlc_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def altman_command(args: argparse.Namespace) -> None:
    """Evaluate Edward I. Altman (1968, 2000) Z-Score and James A. Ohlson (1980) O-Score Solvency."""
    import json
    from src.analytics.altman_zscore import evaluate_solvency_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    assets = getattr(args, "total_assets", None)
    if assets is not None:
        assets = float(assets)
    liab = getattr(args, "total_liabilities", None)
    if liab is not None:
        liab = float(liab)
    wc = getattr(args, "working_capital", None)
    if wc is not None:
        wc = float(wc)
    re = getattr(args, "retained_earnings", None)
    if re is not None:
        re = float(re)
    ebit = getattr(args, "ebit", None)
    if ebit is not None:
        ebit = float(ebit)
    sales = getattr(args, "sales", None)
    if sales is not None:
        sales = float(sales)
    mcap = getattr(args, "market_cap", None)
    if mcap is not None:
        mcap = float(mcap)
    ca = getattr(args, "current_assets", None)
    if ca is not None:
        ca = float(ca)
    cl = getattr(args, "current_liabilities", None)
    if cl is not None:
        cl = float(cl)
    ni = getattr(args, "net_income", None)
    if ni is not None:
        ni = float(ni)
    ffo = getattr(args, "funds_from_operations", None)
    if ffo is not None:
        ffo = float(ffo)

    try:
        receipt = evaluate_solvency_profile(
            ticker=ticker,
            total_assets=assets,
            total_liabilities=liab,
            working_capital=wc,
            retained_earnings=re,
            ebit=ebit,
            sales=sales,
            market_cap=mcap,
            current_assets=ca,
            current_liabilities=cl,
            net_income=ni,
            funds_from_operations=ffo,
        )
    except Exception as ex:
        print(f"\nError: Altman Z-Score & Ohlson O-Score evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    safe_str = "YES (Z > 2.99 - MINIMAL DEFAULT HAZARD)" if receipt.is_safe_zone else "NO"
    grey_str = "YES (1.81 <= Z <= 2.99 - MONITOR BALANCE SHEET)" if receipt.is_grey_zone else "NO"
    distress_str = "YES (Z < 1.81 - ACUTE INSOLVENCY TRAP HAZARD)" if receipt.is_distress_zone else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Altman (1968, 2000) Z-Score & Ohlson (1980) O-Score Solvency Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Altman Multi-Discriminant Factors:")
    print(f"    X1 (Working Capital / Assets):  {receipt.x1_working_capital_to_assets:>+8.4f}")
    print(f"    X2 (Retained Earnings / Assets):{receipt.x2_retained_earnings_to_assets:>+8.4f}")
    print(f"    X3 (EBIT / Total Assets):       {receipt.x3_ebit_to_assets:>+8.4f}")
    print(f"    X4 (Market Equity / Liab):      {receipt.x4_market_equity_to_liabilities:>8.4f}")
    print(f"    X5 (Sales / Total Assets):      {receipt.x5_sales_to_assets:>8.4f}")
    print("-" * 80)
    print("  Solvency & Bankruptcy Hazard Inferences:")
    print(f"    Altman Manufacturing Z-Score:   {receipt.altman_z_score:>8.2f} (Safe > 2.99, Distress < 1.81)")
    print(f"    Altman Non-Mfg Z''-Score:       {receipt.altman_z_double_prime:>8.2f} (Safe > 2.60, Distress < 1.10)")
    print(f"    Ohlson O-Score Index:           {receipt.ohlson_o_score:>+8.2f}")
    print(f"    Implied 2-Yr Default Prob:      {receipt.default_probability_pct:>8.1f}% (P(Default))")
    print(f"    Safe Zone Classification:       {safe_str}")
    print(f"    Grey Zone Classification:       {grey_str}")
    print(f"    Distress Zone Classification:   {distress_str}")
    print("-" * 80)
    print("  Structural Solvency Regime & Alpha Dynamics:")
    print(f"    Solvency Regime:                {receipt.regime}")
    print(f"    Expected Solvency Alpha Spread: {receipt.expected_solvency_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def vme_command(args: argparse.Namespace) -> None:
    """Evaluate Asness, Moskowitz & Pedersen (2013) 'Value and Momentum Everywhere' (VME)."""
    import json
    from src.analytics.vme_unified import evaluate_vme_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    bm = getattr(args, "book_to_market", None)
    if bm is not None:
        bm = float(bm)
    ep = getattr(args, "earnings_yield", None)
    if ep is not None:
        ep = float(ep)
    corr = getattr(args, "correlation", None)
    if corr is not None:
        corr = float(corr)

    try:
        receipt = evaluate_vme_profile(
            ticker=ticker,
            book_to_market=bm,
            earnings_yield_pct=ep,
            value_momentum_correlation=corr,
        )
    except Exception as ex:
        print(f"\nError: Value and Momentum Everywhere (VME) evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    comp_str = "YES (TWIN-ENGINE VALUE + MOMENTUM COMPOUNDER)" if receipt.is_twin_engine_compounder else "NO"
    trap_str = "YES (EXPENSIVE + CRASHING HEADWIND TRAP)" if receipt.is_dual_headwind_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Asness, Moskowitz & Pedersen (2013) VME Unified Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Value & Momentum Raw Covariates:")
    print(f"    Book-to-Market Ratio (B/M):     {receipt.book_to_market:>8.4f}")
    print(f"    Earnings Yield (E/P):           {receipt.earnings_yield_pct:>8.2f}%")
    print(f"    12-2 Skip-Month Momentum:       {receipt.momentum_12_2_pct:>+8.2f}% (1-M Skipped: {receipt.short_term_1m_pct:+.2f}%)")
    print(f"    Annualized Volatility:          {receipt.annualized_volatility_pct:>8.2f}%")
    print("-" * 80)
    print("  Standardized Style Z-Scores & Negative Covariance Engine:")
    print(f"    Standardized Value Z-Score:     {receipt.value_z_score:>+8.2f}")
    print(f"    Standardized Momentum Z-Score:  {receipt.momentum_z_score:>+8.2f}")
    print(f"    Value-Momentum Correlation:     {receipt.value_momentum_correlation:>+8.2f} (Asness Benchmark ~ -0.52)")
    print(f"    Unified 50/50 VME Combo Score:  {receipt.vme_combo_z_score:>+8.2f}")
    print(f"    Diversification Benefit Ratio:  {receipt.diversification_benefit_ratio:>8.2f}x (sqrt(2 / (1 + rho)))")
    print(f"    Implied VME Portfolio Sharpe:   {receipt.implied_vme_sharpe:>8.2f}")
    print("-" * 80)
    print("  VME Regime & Alpha Dynamics:")
    print(f"    VME Portfolio Regime:           {receipt.regime}")
    print(f"    Twin-Engine Compounder:         {comp_str}")
    print(f"    Dual-Factor Headwind Trap:      {trap_str}")
    print(f"    Expected VME Alpha Spread:      {receipt.expected_vme_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


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
        choices=["linear", "gradient_boosting", "ensemble", "adaptive_ensemble", "stacking"],
        default="linear",
        help="Forecasting model family (linear, gradient_boosting, ensemble, adaptive_ensemble, stacking)",
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
        choices=["linear", "gradient_boosting", "ensemble", "adaptive_ensemble", "stacking", "baseline"],
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
    risk_parser.add_argument(
        "--json", action="store_true", help="Output risk report and path profile as JSON"
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
        choices=["ensemble", "adaptive_ensemble", "stacking", "linear", "gradient_boosting"],
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
        choices=["ensemble", "adaptive_ensemble", "stacking", "linear", "gradient_boosting"],
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
        choices=["max_sharpe", "min_variance", "risk_parity", "equal_weight", "hrp", "cvar_min", "cvar_parity"],
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
        choices=["ensemble", "adaptive_ensemble", "stacking", "linear", "gradient_boosting"],
        default=None,
        help="Optional model family filter",
    )

    # Changepoint detection parser
    changepoint_parser = subparsers.add_parser(
        "changepoint-detect", help="Detect structural breaks and regime shifts using Bayesian Online Change-Point Detection"
    )
    changepoint_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to audit (e.g. JPM, 1155.KL)",
    )
    changepoint_parser.add_argument(
        "--hazard",
        type=int,
        default=100,
        help="Expected timescale between regime shifts in trading sessions (default: 100)",
    )
    changepoint_parser.add_argument(
        "--json",
        action="store_true",
        help="Output change-point receipt as JSON",
    )

    # Toxicity report parser
    toxicity_parser = subparsers.add_parser(
        "toxicity-report", help="Audit order flow toxicity and Volume-Synchronized Probability of Toxicity (VPIN)"
    )
    toxicity_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to audit (e.g. JPM, 1155.KL)",
    )
    toxicity_parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling observation window in sessions (default: 20)",
    )
    toxicity_parser.add_argument(
        "--json",
        action="store_true",
        help="Output toxicity receipt as JSON",
    )

    # Tail dependence parser
    tail_parser = subparsers.add_parser(
        "tail-dependence", help="Evaluate multi-asset lower-tail dependence and joint crash fragility"
    )
    tail_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        default=None,
        help="Optional exchange MIC filter (XNYS, XKLS)",
    )
    tail_parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Lower-tail quantile cutoff q (default: 0.05 for 5% extreme crash tail)",
    )
    tail_parser.add_argument(
        "--json",
        action="store_true",
        help="Output tail dependence receipt as JSON",
    )

    # Factor attribution parser
    factor_parser = subparsers.add_parser(
        "factor-attribution", help="Evaluate multi-factor risk attribution and style exposures"
    )
    factor_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. JPM, MSFT)",
    )
    factor_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        default="XNYS",
        help="Exchange MIC code (default: XNYS)",
    )
    factor_parser.add_argument(
        "--json",
        action="store_true",
        help="Output factor attribution receipt as JSON",
    )

    # Wavelet denoise parser
    wavelet_parser = subparsers.add_parser(
        "wavelet-denoise", help="Evaluate multiscale wavelet decomposition and adaptive denoising"
    )
    wavelet_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, JPM)",
    )
    wavelet_parser.add_argument(
        "--levels",
        type=int,
        default=3,
        help="Multiresolution decomposition levels (default: 3)",
    )
    wavelet_parser.add_argument(
        "--json",
        action="store_true",
        help="Output wavelet receipt as JSON",
    )

    # Momentum crash report parser
    momentum_parser = subparsers.add_parser(
        "momentum-crash-report", help="Evaluate risk-managed momentum and crash hazard diagnostics"
    )
    momentum_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, JPM)",
    )
    momentum_parser.add_argument(
        "--target-vol",
        type=float,
        default=0.12,
        help="Target annualized volatility sigma* (default: 0.12)",
    )
    momentum_parser.add_argument(
        "--max-leverage",
        type=float,
        default=2.0,
        help="Maximum allowable leverage cap (default: 2.0)",
    )
    momentum_parser.add_argument(
        "--json",
        action="store_true",
        help="Output momentum crash receipt as JSON",
    )

    # Subparser: fracdiff
    fracdiff_parser = subparsers.add_parser(
        "fracdiff",
        help="Evaluate memory-preserving fractional differentiation and optimal d*",
    )
    fracdiff_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    fracdiff_parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="FFD weight truncation threshold (default: 1e-4)",
    )
    fracdiff_parser.add_argument(
        "--d",
        type=float,
        default=None,
        help="Specific fractional differentiation order to evaluate (optional)",
    )
    fracdiff_parser.add_argument(
        "--auto-d",
        action="store_true",
        default=True,
        help="Automatically search for minimal stationary order d* (default: True)",
    )
    fracdiff_parser.add_argument(
        "--json",
        action="store_true",
        help="Output fractional differentiation receipt as JSON",
    )

    # Subparser: kalman-spread
    kalman_parser = subparsers.add_parser(
        "kalman-spread",
        help="Evaluate online Kalman Filter dynamic hedge ratio and adaptive spread",
    )
    kalman_parser.add_argument(
        "--y-ticker",
        type=str,
        required=True,
        help="Dependent asset ticker Y (e.g. JPM, MSFT)",
    )
    kalman_parser.add_argument(
        "--x-ticker",
        type=str,
        required=True,
        help="Independent asset ticker X (e.g. BAC, AAPL)",
    )
    kalman_parser.add_argument(
        "--delta",
        type=float,
        default=1e-4,
        help="Process noise factor controlling rate of parameter adaptation (default: 1e-4)",
    )
    kalman_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Kalman spread receipt as JSON",
    )

    # Subparser: evt-report
    evt_parser = subparsers.add_parser(
        "evt-report",
        help="Evaluate Extreme Value Theory (EVT) Peaks-Over-Threshold tail risk and GPD",
    )
    evt_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    evt_parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Loss threshold quantile u (default: 0.95)",
    )
    evt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output EVT tail receipt as JSON",
    )

    # Subparser: entropy-report
    entropy_parser = subparsers.add_parser(
        "entropy-report",
        help="Evaluate market information entropy, permutation complexity, and predictability",
    )
    entropy_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    entropy_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling analysis window in trading sessions (default: 60)",
    )
    entropy_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=3,
        help="Permutation embedding dimension m (default: 3)",
    )
    entropy_parser.add_argument(
        "--json",
        action="store_true",
        help="Output market entropy receipt as JSON",
    )

    # Subparser: garch-volatility
    garch_parser = subparsers.add_parser(
        "garch-volatility",
        help="Evaluate asymmetric GJR-GARCH(1, 1) volatility forecasting and leverage effect",
    )
    garch_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    garch_parser.add_argument(
        "--model",
        type=str,
        default="gjr",
        choices=["gjr", "standard"],
        help="GARCH model specification: 'gjr' (asymmetric) or 'standard' (default: gjr)",
    )
    garch_parser.add_argument(
        "--json",
        action="store_true",
        help="Output GARCH receipt as JSON",
    )

    # Subparser: resampled-frontier
    rf_parser = subparsers.add_parser(
        "resampled-frontier",
        help="Evaluate Michaud (1998) Resampled Efficiency and compare with classical Markowitz frontier",
    )
    rf_parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ticker symbols (e.g. AAPL,MSFT,NVDA,GOOGL)",
    )
    rf_parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo resampling paths (default: 100)",
    )
    rf_parser.add_argument(
        "--points",
        type=int,
        default=15,
        help="Number of discrete efficient frontier points (default: 15)",
    )
    rf_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Annualized risk-free rate benchmark (default: 0.04)",
    )
    rf_parser.add_argument(
        "--json",
        action="store_true",
        help="Output resampled frontier receipt as JSON",
    )

    # Subparser: mrmr-select
    mrmr_parser = subparsers.add_parser(
        "mrmr-select",
        help="Perform Minimum Redundancy Maximum Relevance (mRMR) quantitative feature selection",
    )
    mrmr_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    mrmr_parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=[1, 5, 21],
        help="Forecast horizon in sessions: 1, 5, or 21 (default: 1)",
    )
    mrmr_parser.add_argument(
        "--n-features",
        type=int,
        default=5,
        help="Target number of non-redundant features to select (default: 5)",
    )
    mrmr_parser.add_argument(
        "--criterion",
        type=str,
        default="MID",
        choices=["MID", "MIQ"],
        help="mRMR objective criterion: 'MID' (Difference) or 'MIQ' (Quotient) (default: MID)",
    )
    mrmr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output mRMR selection receipt as JSON",
    )

    # Subparser: price-impact
    impact_parser = subparsers.add_parser(
        "price-impact",
        help="Evaluate Kyle's (1985) Informed Trading Lambda and market price impact",
    )
    impact_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    impact_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling regression window in sessions (default: 60)",
    )
    impact_parser.add_argument(
        "--json",
        action="store_true",
        help="Output price impact receipt as JSON",
    )

    # Subparser: tail-shape
    tail_parser = subparsers.add_parser(
        "tail-shape",
        help="Evaluate robust quantile tail shape, Moors kurtosis, and crash asymmetry",
    )
    tail_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    tail_parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Rolling observation window in sessions (default: 120)",
    )
    tail_parser.add_argument(
        "--json",
        action="store_true",
        help="Output tail shape receipt as JSON",
    )

    # Subparser: pain-index
    pain_parser = subparsers.add_parser(
        "pain-index",
        help="Evaluate Burke Ratio, Becker Pain Index, and Ulcer Performance Index",
    )
    pain_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    pain_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Annualized risk-free benchmark rate (default: 0.04)",
    )
    pain_parser.add_argument(
        "--json",
        action="store_true",
        help="Output drawdown analytics receipt as JSON",
    )

    # Subparser: markov-regime
    markov_parser = subparsers.add_parser(
        "markov-regime",
        help="Evaluate Hamilton (1989) two-state Markov-switching dynamic regime model",
    )
    markov_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    markov_parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Observation rolling window in sessions (default: 120)",
    )
    markov_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Markov regime receipt as JSON",
    )

    # Subparser: rmt-denoise
    rmt_parser = subparsers.add_parser(
        "rmt-denoise",
        help="Denoise empirical covariance matrix via Marchenko-Pastur (1967) Random Matrix Theory",
    )
    rmt_parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated stock tickers (e.g. AAPL,MSFT,NVDA,JPM)",
    )
    rmt_parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        help="Exchange MIC code for universe selection (default: XNYS)",
    )
    rmt_parser.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Lookback window in trading sessions (default: 252)",
    )
    rmt_parser.add_argument(
        "--method",
        type=str,
        default="constant_residual",
        choices=["constant_residual", "targeted_shrinkage"],
        help="RMT eigenvalue replacement algorithm (default: constant_residual)",
    )
    rmt_parser.add_argument(
        "--detrend",
        action="store_true",
        help="Remove dominant market eigenvalue mode (lambda_1)",
    )
    rmt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output RMT denoising receipt as JSON",
    )

    # Subparser: jump-test
    jump_parser = subparsers.add_parser(
        "jump-test",
        help="Evaluate Barndorff-Nielsen & Shephard (2004, 2006) Bipower Variation and Jump Decomposition",
    )
    jump_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    jump_parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Observation rolling window in sessions (default: 21)",
    )
    jump_parser.add_argument(
        "--json",
        action="store_true",
        help="Output jump diffusion receipt as JSON",
    )

    # Subparser: deflated-sharpe
    dsr_parser = subparsers.add_parser(
        "deflated-sharpe",
        help="Evaluate Bailey & López de Prado (2014) Probabilistic & Deflated Sharpe Ratio",
    )
    dsr_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    dsr_parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of strategy/model trials tested (default: 10)",
    )
    dsr_parser.add_argument(
        "--benchmark-sr",
        type=float,
        default=0.0,
        help="Hurdle annualized benchmark Sharpe ratio (default: 0.0)",
    )
    dsr_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Annualized risk-free rate (default: 0.04)",
    )
    dsr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Deflated Sharpe receipt as JSON",
    )

    # Subparser: effective-spread
    spread_parser = subparsers.add_parser(
        "effective-spread",
        help="Evaluate Richard Roll (1984) Effective Bid-Ask Spread and Composite Execution Friction",
    )
    spread_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    spread_parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Observation rolling window in sessions (default: 21)",
    )
    spread_parser.add_argument(
        "--json",
        action="store_true",
        help="Output effective spread receipt as JSON",
    )

    # Subparser: cpcv-validate
    cpcv_parser = subparsers.add_parser(
        "cpcv-validate",
        help="Evaluate Marcos López de Prado (2018) Combinatorial Purged & Embargoed Cross-Validation (CPCV)",
    )
    cpcv_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    cpcv_parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in sessions (default: 1)",
    )
    cpcv_parser.add_argument(
        "--groups",
        type=int,
        default=6,
        help="Total contiguous groups N (default: 6)",
    )
    cpcv_parser.add_argument(
        "--test-groups",
        type=int,
        default=2,
        help="Test groups per combination k (default: 2)",
    )
    cpcv_parser.add_argument(
        "--embargo",
        type=float,
        default=0.01,
        help="Post-test embargo proportion (default: 0.01)",
    )
    cpcv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output CPCV receipt as JSON",
    )

    # Subparser: downside-beta
    downside_parser = subparsers.add_parser(
        "downside-beta",
        help="Evaluate Ang, Chen & Xing (2006) Downside Beta and Realized Semivariance",
    )
    downside_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    downside_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    downside_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    downside_parser.add_argument(
        "--json",
        action="store_true",
        help="Output downside risk receipt as JSON",
    )

    # Subparser: optimal-execution
    execution_parser = subparsers.add_parser(
        "optimal-execution",
        help="Evaluate Almgren & Chriss (2000) optimal execution trajectory and market impact",
    )
    execution_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    execution_parser.add_argument(
        "--shares",
        type=float,
        required=True,
        help="Target shares to execute (positive for sell/liquidation, negative for buy)",
    )
    execution_parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Execution horizon in trading days/sessions (default: 5)",
    )
    execution_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1e-6,
        help="Investor risk aversion parameter lambda >= 0 (default: 1e-6, 0 for pure TWAP)",
    )
    execution_parser.add_argument(
        "--json",
        action="store_true",
        help="Output execution trajectory receipt as JSON",
    )

    # Subparser: fama-french
    fama_parser = subparsers.add_parser(
        "fama-french",
        help="Evaluate Fama-French (1993) & Carhart (1997) multi-factor risk and alpha attribution",
    )
    fama_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    fama_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    fama_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    fama_parser.add_argument(
        "--json",
        action="store_true",
        help="Output factor attribution receipt as JSON",
    )

    # Subparser: black-litterman
    bl_parser = subparsers.add_parser(
        "black-litterman",
        help="Execute Black-Litterman (1992) Bayesian portfolio optimization & active view blending",
    )
    bl_parser.add_argument(
        "--universe",
        type=str,
        required=True,
        help="Comma-separated stock universe (e.g. AAPL,MSFT,NVDA,GOOGL)",
    )
    bl_parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="Active views as key:value pairs or JSON (e.g. NVDA:0.25,AAPL:0.12)",
    )
    bl_parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="Prior uncertainty scalar (default: 0.05)",
    )
    bl_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=2.5,
        help="Market risk aversion parameter delta (default: 2.5)",
    )
    bl_parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback window in sessions (default: 120)",
    )
    bl_parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable Marchenko-Pastur RMT covariance denoising",
    )
    bl_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Black-Litterman receipt as JSON",
    )

    # Subparser: idiosyncratic-vol
    ivol_parser = subparsers.add_parser(
        "idiosyncratic-vol",
        help="Evaluate Ang, Hodrick, Xing & Zhang (2006, 2009) idiosyncratic volatility & higher moments",
    )
    ivol_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    ivol_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    ivol_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    ivol_parser.add_argument(
        "--json",
        action="store_true",
        help="Output idiosyncratic volatility receipt as JSON",
    )

    # Subparser: liquidity-beta
    liq_parser = subparsers.add_parser(
        "liquidity-beta",
        help="Evaluate Pástor & Stambaugh (2003) systematic liquidity risk factor & liquidity beta",
    )
    liq_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    liq_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    liq_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    liq_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Pástor-Stambaugh liquidity receipt as JSON",
    )

    # Subparser: fama-macbeth
    fm_parser = subparsers.add_parser(
        "fama-macbeth",
        help="Execute Fama & MacBeth (1973) two-pass cross-sectional regression with Shanken (1992) corrections",
    )
    fm_parser.add_argument(
        "--universe",
        type=str,
        required=True,
        help="Comma-separated stock universe (e.g. AAPL,MSFT,NVDA,GOOGL,AMZN)",
    )
    fm_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    fm_parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Observation lookback window in sessions (default: 120)",
    )
    fm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Fama-MacBeth receipt as JSON",
    )

    # Subparser: higher-moments
    hm_parser = subparsers.add_parser(
        "higher-moments",
        help="Evaluate Harvey-Siddique (2000) co-skewness & Dittmar (2002) co-kurtosis higher-moment asset pricing",
    )
    hm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    hm_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    hm_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    hm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output higher moments receipt as JSON",
    )

    # Subparser: merton-credit
    mc_parser = subparsers.add_parser(
        "merton-credit",
        help="Evaluate Robert C. Merton (1974) structural credit risk & distance-to-default",
    )
    mc_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    mc_parser.add_argument(
        "--debt",
        type=float,
        required=True,
        help="Face value of total debt liabilities / default point in dollars",
    )
    mc_parser.add_argument(
        "--shares",
        type=float,
        default=None,
        help="Common shares outstanding (optional; defaults to 1B shares or market cap)",
    )
    mc_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Explicit total equity market capitalization in dollars (optional)",
    )
    mc_parser.add_argument(
        "--volatility",
        type=float,
        default=None,
        help="Explicit annualized equity return volatility sigma_E (optional)",
    )
    mc_parser.add_argument(
        "--risk-free",
        type=float,
        default=0.045,
        help="Annualized risk-free interest rate (default: 0.045)",
    )
    mc_parser.add_argument(
        "--horizon",
        type=float,
        default=1.0,
        help="Credit maturity / forecast horizon in years (default: 1.0)",
    )
    mc_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback window in sessions to calculate equity volatility (default: 60)",
    )
    mc_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Merton credit receipt as JSON",
    )

    # Subparser: hasbrouck-var
    hv_parser = subparsers.add_parser(
        "hasbrouck-var",
        help="Evaluate Joel Hasbrouck (1991) VAR price discovery and trade information share",
    )
    hv_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    hv_parser.add_argument(
        "--lags",
        type=int,
        default=5,
        help="Number of autoregressive lags p in VAR (default: 5)",
    )
    hv_parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Impulse response function simulation horizon H (default: 20)",
    )
    hv_parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback window in sessions (default: 120)",
    )
    hv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hasbrouck receipt as JSON",
    )

    # Subparser: vrp
    vrp_parser = subparsers.add_parser(
        "vrp",
        help="Evaluate Bollerslev, Tauchen & Zhou (2009) variance risk premium & volatility-of-volatility",
    )
    vrp_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    vrp_parser.add_argument(
        "--iv",
        type=float,
        default=None,
        help="Annualized implied volatility (e.g. 0.28 for 28%; optional)",
    )
    vrp_parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Forward forecast horizon in sessions (default: 21)",
    )
    vrp_parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback window in sessions (default: 120)",
    )
    vrp_parser.add_argument(
        "--json",
        action="store_true",
        help="Output VRP receipt as JSON",
    )

    # Subparser: max-lottery
    ml_parser = subparsers.add_parser(
        "max-lottery",
        help="Evaluate Bali, Cakici & Whitelaw (2011) extreme daily return MAX effect & lottery demand",
    )
    ml_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    ml_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of extreme upside days N to average for MAX_N (default: 5)",
    )
    ml_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback observation window in sessions (default: 60)",
    )
    ml_parser.add_argument(
        "--json",
        action="store_true",
        help="Output MAX lottery receipt as JSON",
    )

    # Subparser: profitability
    gp_parser = subparsers.add_parser(
        "profitability",
        help="Evaluate Robert Novy-Marx (2013) gross profitability & Asness et al. (2019) QMJ quality",
    )
    gp_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    gp_parser.add_argument(
        "--revenue",
        type=float,
        default=None,
        help="Total annualized company revenue in dollars (optional)",
    )
    gp_parser.add_argument(
        "--cogs",
        type=float,
        default=None,
        help="Total Cost of Goods Sold in dollars (optional)",
    )
    gp_parser.add_argument(
        "--assets",
        type=float,
        default=None,
        help="Total Book Assets in dollars (optional)",
    )
    gp_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Total Debt Obligations in dollars (optional)",
    )
    gp_parser.add_argument(
        "--json",
        action="store_true",
        help="Output profitability receipt as JSON",
    )

    # Subparser: dual-momentum
    dm_parser = subparsers.add_parser(
        "dual-momentum",
        help="Evaluate Jegadeesh & Titman (1993) / Antonacci Dual Momentum & Industry Relative Strength",
    )
    dm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    dm_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Reference benchmark ticker (default: SPY)",
    )
    dm_parser.add_argument(
        "--rf-rate",
        type=float,
        default=4.0,
        help="Annualized risk-free hurdle rate percentage (default: 4.0)",
    )
    dm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output dual momentum receipt as JSON",
    )

    # Subparser: bab
    bab_parser = subparsers.add_parser(
        "bab",
        help="Evaluate Andrea Frazzini & Lasse Heje Pedersen (2014) Betting Against Beta & Leverage Multipliers",
    )
    bab_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    bab_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Reference benchmark ticker (default: SPY)",
    )
    bab_parser.add_argument(
        "--rf-rate",
        type=float,
        default=4.0,
        help="Annualized risk-free hurdle rate percentage (default: 4.0)",
    )
    bab_parser.add_argument(
        "--market-premium",
        type=float,
        default=6.0,
        help="Annualized expected equity risk premium ERP = E[Rm] - Rf (default: 6.0)",
    )
    bab_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BAB receipt as JSON",
    )

    # Subparser: reversal
    rev_parser = subparsers.add_parser(
        "reversal",
        help="Evaluate Narasimhan Jegadeesh (1990) & Bruce Lehmann (1990) Short-Term Reversal & Contrarian Alpha",
    )
    rev_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    rev_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Reference benchmark ticker (default: SPY)",
    )
    rev_parser.add_argument(
        "--json",
        action="store_true",
        help="Output reversal receipt as JSON",
    )

    # Subparser: distress
    distress_parser = subparsers.add_parser(
        "distress",
        help="Evaluate Campbell, Hilscher & Szilagyi (2008) Financial Distress Risk Hazard & Insolvency Alpha",
    )
    distress_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    distress_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Market capitalization in dollars (optional)",
    )
    distress_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Total debt liabilities in dollars (optional)",
    )
    distress_parser.add_argument(
        "--net-income",
        type=float,
        default=None,
        help="Annual net income in dollars (optional)",
    )
    distress_parser.add_argument(
        "--cash",
        type=float,
        default=None,
        help="Cash and liquid reserves in dollars (optional)",
    )
    distress_parser.add_argument(
        "--json",
        action="store_true",
        help="Output distress receipt as JSON",
    )

    # Subparser: sentiment
    sent_parser = subparsers.add_parser(
        "sentiment",
        help="Evaluate Malcolm Baker & Jeffrey Wurgler (2006) Cross-Sectional Investor Sentiment & Limits to Arbitrage",
    )
    sent_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    sent_parser.add_argument(
        "--market-sentiment",
        type=float,
        default=0.0,
        help="Standardized macro investor sentiment index I_sent in [-3.0, +3.0] (default: 0.0)",
    )
    sent_parser.add_argument(
        "--net-margin",
        type=float,
        default=None,
        help="Net profit margin percentage (optional)",
    )
    sent_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Market capitalization in dollars (optional)",
    )
    sent_parser.add_argument(
        "--json",
        action="store_true",
        help="Output sentiment receipt as JSON",
    )

    # Subparser: arbitrage-asymmetry
    asym_parser = subparsers.add_parser(
        "arbitrage-asymmetry",
        aliases=["asymmetry", "short-impediment"],
        help="Evaluate Robert F. Stambaugh, Jianfeng Yu & Yu Yuan (2012, 2015) Short-Sale Impediments & Arbitrage Asymmetry",
    )
    asym_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. TSLA, GME, AAPL)",
    )
    asym_parser.add_argument(
        "--market-sentiment",
        type=float,
        default=0.0,
        help="Standardized macro investor sentiment index I_sent in [-3.0, +3.0] (default: 0.0)",
    )
    asym_parser.add_argument(
        "--borrow-fee",
        type=float,
        default=None,
        help="Annualized cost to borrow equity shares (%/yr) (optional)",
    )
    asym_parser.add_argument(
        "--short-interest",
        type=float,
        default=None,
        help="Short interest as percentage of free float (%) (optional)",
    )
    asym_parser.add_argument(
        "--days-to-cover",
        type=float,
        default=None,
        help="Short ratio / days to cover in sessions (optional)",
    )
    asym_parser.add_argument(
        "--json",
        action="store_true",
        help="Output arbitrage asymmetry receipt as JSON",
    )

    # Subparser: q-factor
    q_parser = subparsers.add_parser(
        "q-factor",
        aliases=["qfactor", "hou-xue-zhang"],
        help="Evaluate Kewei Hou, Chen Xue & Lu Zhang (2015, 2020) q-Factor Investment & Profitability Profile",
    )
    q_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT, TSLA)",
    )
    q_parser.add_argument(
        "--roe",
        type=float,
        default=None,
        help="Return on Equity percentage (Net Income / Book Equity) (optional)",
    )
    q_parser.add_argument(
        "--investment-rate",
        type=float,
        default=None,
        help="Annual Investment-to-Assets I/A percentage (Total Asset Growth %) (optional)",
    )
    q_parser.add_argument(
        "--asset-growth",
        type=float,
        default=None,
        help="Alias for --investment-rate",
    )
    q_parser.add_argument(
        "--tobin-q",
        type=float,
        default=None,
        help="Approximate Tobin's q ratio (optional)",
    )
    q_parser.add_argument(
        "--json",
        action="store_true",
        help="Output q-factor receipt as JSON",
    )

    # Subparser: accruals
    accrual_parser = subparsers.add_parser(
        "accruals",
        aliases=["accrual", "earnings-quality", "sloan"],
        help="Evaluate Richard G. Sloan (1996) Accrual Anomaly & Earnings Quality Decomposition",
    )
    accrual_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, BRK.B, TSLA)",
    )
    accrual_parser.add_argument(
        "--net-income",
        type=float,
        default=None,
        help="Reported headline net income ($) (optional)",
    )
    accrual_parser.add_argument(
        "--operating-cash-flow",
        type=float,
        default=None,
        help="Operating cash flow from operations ($) (optional)",
    )
    accrual_parser.add_argument(
        "--cash-flow",
        type=float,
        default=None,
        help="Alias for --operating-cash-flow",
    )
    accrual_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Average total assets ($) (optional)",
    )
    accrual_parser.add_argument(
        "--json",
        action="store_true",
        help="Output accrual anomaly receipt as JSON",
    )

    # Subparser: beneish
    beneish_parser = subparsers.add_parser(
        "beneish",
        aliases=["mscore", "forensic-audit", "fraud-risk"],
        help="Evaluate Messod D. Beneish (1999) Forensic Accounting M-Score & Manipulation Risk",
    )
    beneish_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. ENRN, TSLA, AAPL)",
    )
    beneish_parser.add_argument(
        "--dsri",
        type=float,
        default=None,
        help="Days' Sales in Receivables Index (optional)",
    )
    beneish_parser.add_argument(
        "--gmi",
        type=float,
        default=None,
        help="Gross Margin Index (optional)",
    )
    beneish_parser.add_argument(
        "--aqi",
        type=float,
        default=None,
        help="Asset Quality Index (optional)",
    )
    beneish_parser.add_argument(
        "--sgi",
        type=float,
        default=None,
        help="Sales Growth Index (optional)",
    )
    beneish_parser.add_argument(
        "--depi",
        type=float,
        default=None,
        help="Depreciation Index (optional)",
    )
    beneish_parser.add_argument(
        "--sgai",
        type=float,
        default=None,
        help="SG&A Expense Index (optional)",
    )
    beneish_parser.add_argument(
        "--lvgi",
        type=float,
        default=None,
        help="Leverage Index (optional)",
    )
    beneish_parser.add_argument(
        "--tata",
        type=float,
        default=None,
        help="Total Accruals to Total Assets (optional)",
    )
    beneish_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Beneish M-Score receipt as JSON",
    )

    # Subparser: fscore
    fscore_parser = subparsers.add_parser(
        "fscore",
        aliases=["piotroski", "fundamental-health", "value-quality"],
        help="Evaluate Joseph D. Piotroski (2000) 9-Point Fundamental Accounting F-Score",
    )
    fscore_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. BRK.B, JNJ, TSLA)",
    )
    fscore_parser.add_argument(
        "--net-income",
        type=float,
        default=None,
        help="Current net income ($) (optional)",
    )
    fscore_parser.add_argument(
        "--operating-cash-flow",
        type=float,
        default=None,
        help="Operating cash flow ($) (optional)",
    )
    fscore_parser.add_argument(
        "--cash-flow",
        type=float,
        default=None,
        help="Alias for --operating-cash-flow",
    )
    fscore_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Total assets ($) (optional)",
    )
    fscore_parser.add_argument(
        "--roa",
        type=float,
        default=None,
        help="Return on assets % (optional)",
    )
    fscore_parser.add_argument(
        "--delta-roa",
        type=float,
        default=None,
        help="Change in return on assets % (optional)",
    )
    fscore_parser.add_argument(
        "--delta-debt",
        type=float,
        default=None,
        help="Change in long-term debt ($) (negative = debt repaid) (optional)",
    )
    fscore_parser.add_argument(
        "--delta-current-ratio",
        type=float,
        default=None,
        help="Change in current ratio (positive = liquidity expanded) (optional)",
    )
    fscore_parser.add_argument(
        "--shares-dilution",
        type=float,
        default=None,
        help="Share dilution % (negative = repurchased) (optional)",
    )
    fscore_parser.add_argument(
        "--delta-gross-margin",
        type=float,
        default=None,
        help="Change in gross margin % (optional)",
    )
    fscore_parser.add_argument(
        "--delta-asset-turnover",
        type=float,
        default=None,
        help="Change in asset turnover (optional)",
    )
    fscore_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Piotroski F-Score receipt as JSON",
    )

    # Hong & Stein (1999) Gradual Information Diffusion Parser
    diffusion_parser = subparsers.add_parser(
        "diffusion",
        aliases=["hong-stein", "information-diffusion", "momentum-regime"],
        help="Evaluate Hong & Stein (1999) gradual information diffusion and trend maturity",
    )
    diffusion_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    diffusion_parser.add_argument(
        "--analyst-coverage",
        type=int,
        default=None,
        help="Number of equity research analysts covering the firm (optional)",
    )
    diffusion_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hong-Stein diffusion receipt as JSON",
    )

    # Lakonishok, Shleifer & Vishny (1994) Contrarian Investment Parser
    contrarian_parser = subparsers.add_parser(
        "contrarian",
        aliases=["lsv", "extrapolation-bias", "glamour-overpricing"],
        help="Evaluate Lakonishok, Shleifer & Vishny (1994) contrarian investment and extrapolation bias",
    )
    contrarian_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    contrarian_parser.add_argument(
        "--sales-growth",
        type=float,
        default=None,
        help="Annualized 3-year sales growth rate % (optional)",
    )
    contrarian_parser.add_argument(
        "--earnings-growth",
        type=float,
        default=None,
        help="Annualized 3-year net income growth rate % (optional)",
    )
    contrarian_parser.add_argument(
        "--pe",
        type=float,
        default=None,
        help="Price-to-earnings (P/E) ratio (optional)",
    )
    contrarian_parser.add_argument(
        "--pcf",
        type=float,
        default=None,
        help="Price-to-cash-flow (P/CF) ratio (optional)",
    )
    contrarian_parser.add_argument(
        "--pb",
        type=float,
        default=None,
        help="Price-to-book (P/B) ratio (optional)",
    )
    contrarian_parser.add_argument(
        "--json",
        action="store_true",
        help="Output LSV Contrarian receipt as JSON",
    )

    # Daniel, Hirshleifer & Subrahmanyam (1998) Overconfidence & Self-Attribution Parser
    dhs_parser = subparsers.add_parser(
        "dhs",
        aliases=["overconfidence", "self-attribution", "behavioral-lifecycle"],
        help="Evaluate Daniel, Hirshleifer & Subrahmanyam (1998) overconfidence and self-attribution",
    )
    dhs_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    dhs_parser.add_argument(
        "--earnings-surprise",
        type=float,
        default=None,
        help="Public earnings surprise % (optional)",
    )
    dhs_parser.add_argument(
        "--dispersion",
        type=float,
        default=None,
        help="Analyst EPS forecast dispersion % (optional)",
    )
    dhs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output DHS Behavioral receipt as JSON",
    )

    # Barberis, Shleifer & Vishny (1998) Dual-Heuristic Sentiment Parser
    bsv_parser = subparsers.add_parser(
        "bsv",
        aliases=["barberis", "conservatism-representativeness", "dual-heuristic"],
        help="Evaluate Barberis, Shleifer & Vishny (1998) conservatism and representativeness sentiment",
    )
    bsv_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    bsv_parser.add_argument(
        "--earnings-surprise",
        type=float,
        default=None,
        help="Latest quarterly earnings surprise % (optional)",
    )
    bsv_parser.add_argument(
        "--streak",
        type=int,
        default=None,
        help="Consecutive quarters with earnings surprise in same direction (optional)",
    )
    bsv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BSV Sentiment receipt as JSON",
    )

    # Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Parser
    innovation_parser = subparsers.add_parser(
        "innovation",
        aliases=["rd-anomaly", "knowledge-capital", "rnd-intensity"],
        help="Evaluate Eberhart et al. (2004) / Cohen et al. (2013) innovation capital and R&D misvaluation",
    )
    innovation_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, BMY)",
    )
    innovation_parser.add_argument(
        "--rd-expense",
        type=float,
        default=None,
        help="Annual R&D expenditure in dollars (optional)",
    )
    innovation_parser.add_argument(
        "--revenues",
        type=float,
        default=None,
        help="Annual revenues in dollars (optional)",
    )
    innovation_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Total assets in dollars (optional)",
    )
    innovation_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Innovation Capital receipt as JSON",
    )

    # Charles M. C. Lee & Bhaskaran Swaminathan (2000) Momentum Life Cycle Parser
    lifecycle_parser = subparsers.add_parser(
        "lifecycle",
        aliases=["lee-swaminathan", "momentum-volume", "mlc-stage"],
        help="Evaluate Lee & Swaminathan (2000) price momentum and trading volume life-cycle",
    )
    lifecycle_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    lifecycle_parser.add_argument(
        "--turnover",
        type=float,
        default=None,
        help="Daily turnover ratio % override (optional)",
    )
    lifecycle_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Momentum Life Cycle receipt as JSON",
    )

    # Edward I. Altman (1968, 2000) Z-Score & James A. Ohlson (1980) O-Score Parser
    altman_parser = subparsers.add_parser(
        "altman",
        aliases=["zscore", "ohlson", "distress-risk"],
        help="Evaluate Altman (1968, 2000) Z-Score and Ohlson (1980) O-Score structural solvency",
    )
    altman_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    altman_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Total assets in $M (optional)",
    )
    altman_parser.add_argument(
        "--total-liabilities",
        type=float,
        default=None,
        help="Total liabilities in $M (optional)",
    )
    altman_parser.add_argument(
        "--working-capital",
        type=float,
        default=None,
        help="Working capital in $M (optional)",
    )
    altman_parser.add_argument(
        "--retained-earnings",
        type=float,
        default=None,
        help="Retained earnings in $M (optional)",
    )
    altman_parser.add_argument(
        "--ebit",
        type=float,
        default=None,
        help="EBIT operating profit in $M (optional)",
    )
    altman_parser.add_argument(
        "--sales",
        type=float,
        default=None,
        help="Total sales / revenues in $M (optional)",
    )
    altman_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Market capitalization in $M (optional)",
    )
    altman_parser.add_argument(
        "--current-assets",
        type=float,
        default=None,
        help="Current assets in $M (optional)",
    )
    altman_parser.add_argument(
        "--current-liabilities",
        type=float,
        default=None,
        help="Current liabilities in $M (optional)",
    )
    altman_parser.add_argument(
        "--net-income",
        type=float,
        default=None,
        help="Net income in $M (optional)",
    )
    altman_parser.add_argument(
        "--funds-from-operations",
        type=float,
        default=None,
        help="Funds from operations / CFO in $M (optional)",
    )
    altman_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Altman Z-Score & Ohlson O-Score receipt as JSON",
    )

    # Clifford S. Asness, Tobias J. Moskowitz & Lasse Heje Pedersen (2013) VME Parser
    vme_parser = subparsers.add_parser(
        "vme",
        aliases=["value-momentum", "asness-moskowitz", "vme-everywhere"],
        help="Evaluate Asness, Moskowitz & Pedersen (2013) 'Value and Momentum Everywhere' (VME)",
    )
    vme_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    vme_parser.add_argument(
        "--book-to-market",
        type=float,
        default=None,
        help="Book-to-Market equity ratio B/M (optional)",
    )
    vme_parser.add_argument(
        "--earnings-yield",
        type=float,
        default=None,
        help="Earnings-to-Price yield E/P % (optional)",
    )
    vme_parser.add_argument(
        "--correlation",
        type=float,
        default=None,
        help="Value-Momentum correlation rho override in [-0.95, 0.95] (optional)",
    )
    vme_parser.add_argument(
        "--json",
        action="store_true",
        help="Output VME receipt as JSON",
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
    elif args.command == "changepoint-detect":
        changepoint_detect_command(args)
    elif args.command == "toxicity-report":
        toxicity_report_command(args)
    elif args.command == "tail-dependence":
        tail_dependence_command(args)
    elif args.command == "factor-attribution":
        factor_attribution_command(args)
    elif args.command == "wavelet-denoise":
        wavelet_denoise_command(args)
    elif args.command == "momentum-crash-report":
        momentum_crash_report_command(args)
    elif args.command == "fracdiff":
        fracdiff_command(args)
    elif args.command == "kalman-spread":
        kalman_spread_command(args)
    elif args.command == "evt-report":
        evt_report_command(args)
    elif args.command == "entropy-report":
        entropy_report_command(args)
    elif args.command == "garch-volatility":
        garch_volatility_command(args)
    elif args.command == "resampled-frontier":
        resampled_frontier_command(args)
    elif args.command == "mrmr-select":
        mrmr_select_command(args)
    elif args.command == "price-impact":
        price_impact_command(args)
    elif args.command == "tail-shape":
        tail_shape_command(args)
    elif args.command == "pain-index":
        pain_index_command(args)
    elif args.command == "markov-regime":
        markov_regime_command(args)
    elif args.command == "rmt-denoise":
        rmt_denoise_command(args)
    elif args.command == "jump-test":
        jump_test_command(args)
    elif args.command == "deflated-sharpe":
        deflated_sharpe_command(args)
    elif args.command == "effective-spread":
        effective_spread_command(args)
    elif args.command == "cpcv-validate":
        cpcv_validate_command(args)
    elif args.command == "downside-beta":
        downside_beta_command(args)
    elif args.command == "optimal-execution":
        optimal_execution_command(args)
    elif args.command == "fama-french":
        fama_french_command(args)
    elif args.command == "black-litterman":
        black_litterman_command(args)
    elif args.command == "idiosyncratic-vol":
        idiosyncratic_vol_command(args)
    elif args.command == "liquidity-beta":
        liquidity_beta_command(args)
    elif args.command == "fama-macbeth":
        fama_macbeth_command(args)
    elif args.command == "higher-moments":
        higher_moments_command(args)
    elif args.command == "merton-credit":
        merton_credit_command(args)
    elif args.command == "hasbrouck-var":
        hasbrouck_var_command(args)
    elif args.command == "vrp":
        vrp_command(args)
    elif args.command == "max-lottery":
        max_lottery_command(args)
    elif args.command == "profitability":
        profitability_command(args)
    elif args.command == "dual-momentum":
        dual_momentum_command(args)
    elif args.command == "bab":
        bab_command(args)
    elif args.command == "reversal":
        reversal_command(args)
    elif args.command == "distress":
        distress_command(args)
    elif args.command == "sentiment":
        sentiment_command(args)
    elif args.command in ("arbitrage-asymmetry", "asymmetry", "short-impediment"):
        arbitrage_asymmetry_command(args)
    elif args.command in ("q-factor", "qfactor", "hou-xue-zhang"):
        q_factor_command(args)
    elif args.command in ("accruals", "accrual", "earnings-quality", "sloan"):
        accruals_command(args)
    elif args.command in ("beneish", "mscore", "forensic-audit", "fraud-risk"):
        beneish_command(args)
    elif args.command in ("fscore", "piotroski", "fundamental-health", "value-quality"):
        fscore_command(args)
    elif args.command in ("diffusion", "hong-stein", "information-diffusion", "momentum-regime"):
        diffusion_command(args)
    elif args.command in ("contrarian", "lsv", "extrapolation-bias", "glamour-overpricing"):
        contrarian_command(args)
    elif args.command in ("dhs", "overconfidence", "self-attribution", "behavioral-lifecycle"):
        dhs_command(args)
    elif args.command in ("bsv", "barberis", "conservatism-representativeness", "dual-heuristic"):
        bsv_command(args)
    elif args.command in ("innovation", "rd-anomaly", "knowledge-capital", "rnd-intensity"):
        innovation_command(args)
    elif args.command in ("lifecycle", "lee-swaminathan", "momentum-volume", "mlc-stage"):
        lifecycle_command(args)
    elif args.command in ("altman", "zscore", "ohlson", "distress-risk"):
        altman_command(args)
    elif args.command in ("vme", "value-momentum", "asness-moskowitz", "vme-everywhere"):
        vme_command(args)


if __name__ == "__main__":
    main()



