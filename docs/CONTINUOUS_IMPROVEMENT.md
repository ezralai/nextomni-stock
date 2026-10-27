# Continuous Improvement Framework (`nextomni-stock`)

This document provides actionable checklists, benchmarking criteria, and security audit rules for evaluating iterative improvements to the NextOmni Stock system.

---

## 1. Quantitative Benchmark Scorecard

Before adopting any new machine learning model (e.g., LightGBM, Random Forest, XGBoost, Neural Nets), record its out-of-sample walk-forward performance against the baseline in this scorecard:

| Model Architecture | Horizon | Test Ticker | Model MAE | Baseline MAE | MAE Delta % | Directional Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge (Iter 1: 10 Feats)** | 1 Session | `JPM` | 2.0365 | 1.9813 | -2.79% | 50.91% | Benchmark |
| **Ridge (Iter 1: 10 Feats)** | 5 Sessions | `JPM` | 4.5732 | 4.7168 | **+3.04%** | 55.66% | Production |
| **Ridge (Iter 1: 10 Feats)** | 21 Sessions | `JPM` | 6.2018 | 7.7491 | **+19.97%** | **73.33%** | Production |
| **HistGradientBoosting (Iter 2)** | 1 Session | `JPM` | 2.0742 | 1.9813 | -4.69% | **57.27%** | Benchmark |
| **HistGradientBoosting (Iter 2)** | 5 Sessions | `JPM` | 4.0913 | 4.7168 | **+13.26%** | **70.75%** | Production |
| **HistGradientBoosting (Iter 2)** | 21 Sessions | `JPM` | 6.8897 | 7.7491 | **+11.09%** | **72.22%** | Production |
| **Ridge (Iter 1: 10 Feats)** | 1 Session | `1155.KL` | 0.0424 | 0.0408 | -3.92% | 47.62% | Benchmark |
| **Ridge (Iter 1: 10 Feats)** | 5 Sessions | `1155.KL` | 0.0756 | 0.0842 | **+10.21%** | **64.36%** | Production |
| **Ridge (Iter 1: 10 Feats)** | 21 Sessions | `1155.KL` | 0.1703 | 0.2278 | **+25.24%** | **88.24%** | Production |
| **HistGradientBoosting (Iter 2)** | 1 Session | `1155.KL` | 0.0435 | 0.0408 | -6.62% | 45.45% | Benchmark |
| **HistGradientBoosting (Iter 2)** | 5 Sessions | `1155.KL` | 0.0704 | 0.0842 | **+16.39%** | **74.26%** | Production |
| **HistGradientBoosting (Iter 2: 10 Feats)** | 21 Sessions | `1155.KL` | 0.1353 | 0.2278 | **+40.61%** | **88.24%** | Production |
| **HistGradientBoosting (Iter 5: 12 Feats)** | 1 Session | `JPM` | 2.2169 | 2.0064 | -10.49% | 51.35% | Benchmark |
| **HistGradientBoosting (Iter 5: 12 Feats)** | 5 Sessions | `JPM` | 4.0360 | 4.6892 | **+13.93%** | **65.42%** | Production |
| **HistGradientBoosting (Iter 5: 12 Feats)** | 21 Sessions | `JPM` | 6.7692 | 7.7506 | **+12.66%** | **73.63%** | Production |
| **HistGradientBoosting (Iter 5: 12 Feats)** | 1 Session | `1155.KL` | 0.0450 | 0.0408 | -10.29% | 40.95% | Benchmark |
| **HistGradientBoosting (Iter 5: 12 Feats)** | 5 Sessions | `1155.KL` | 0.0681 | 0.0842 | **+19.12%** | **70.30%** | Production |
| **HistGradientBoosting (Iter 5: 12 Feats)** | 21 Sessions | `1155.KL` | 0.1224 | 0.2278 | **+46.27%** | **90.59%** | Production |
| **HistGradientBoosting (Iter 6: Adaptive Regularization)** | 1 Session | `JPM` | 2.0159 | 2.0064 | **-0.47%** | **54.95%** | Production |
| **HistGradientBoosting (Iter 6: Adaptive Regularization)** | 5 Sessions | `JPM` | 4.0360 | 4.6892 | **+13.93%** | **65.42%** | Production |
| **HistGradientBoosting (Iter 6: Adaptive Regularization)** | 21 Sessions | `JPM` | 6.7692 | 7.7506 | **+12.66%** | **73.63%** | Production |
| **HistGradientBoosting (Iter 6: Adaptive Regularization)** | 1 Session | `1155.KL` | 0.0413 | 0.0408 | **-1.20%** | **51.43%** | Production |
| **HistGradientBoosting (Iter 6: Adaptive Regularization)** | 5 Sessions | `1155.KL` | 0.0681 | 0.0842 | **+19.12%** | **70.30%** | Production |
| **HistGradientBoosting (Iter 6: Adaptive Regularization)** | 21 Sessions | `1155.KL` | 0.1224 | 0.2278 | **+46.27%** | **90.59%** | Production |
| **ConvexEnsemble (Iter 7: Ridge + GBDT)** | 1 Session | `JPM` | 2.0148 | 2.0064 | **-0.42%** | **54.95%** | Production |
| **ConvexEnsemble (Iter 7: Ridge + GBDT)** | 5 Sessions | `JPM` | 4.4130 | 4.6892 | **+5.89%** | **60.75%** | Production |
| **ConvexEnsemble (Iter 7: Ridge + GBDT)** | 21 Sessions | `JPM` | 5.9663 | 7.7506 | **+23.02%** | **76.92%** | **Production Champion** |

### Model Acceptance Criteria
1. **Short Horizon ($P_{t+1}$):** Model MAE within $\pm 5\%$ of baseline; Directional Accuracy $> 50\%$. (ConvexEnsemble: **-0.42%**, **54.95% DA**).
2. **Intermediate Horizon ($P_{t+5}$):** Model MAE must beat baseline by $\ge 0.5\%$; Directional Accuracy $\ge 58\%$. (ConvexEnsemble: **+5.89%**, **60.75% DA**).
3. **Long Horizon ($P_{t+21}$):** Model MAE must beat baseline by $\ge 10.0\%$; Directional Accuracy $\ge 65\%$. (ConvexEnsemble: **+23.02%**, **76.92% DA**, **IR = +0.335**).

---

## 2. System Security Verification Checklist

- [x] **SQL Query Safety:** All database queries utilize parameter binding (`?`). Zero string formatting in SQL statements.
- [x] **Input Allowlisting:** Ticker symbols match regex `^[A-Z0-9]{1,6}(\.[A-Z]{2})?$`; Exchange MIC matches `^(XNYS|XKLS)$` via `src/security/validators.py`.
- [x] **Model Family Allowlisting:** Model choice validated against `{"linear", "gradient_boosting", "baseline", "ensemble"}`.
- [x] **Floating-Point Finite Assertion:** Database writes validate `assert_finite_number` to protect against NaN/Inf float poisonings across prices and interval bounds.
- [x] **Path Traversal Protection:** Serialized model filenames sanitized and checked with `assert_safe_model_path`.
- [x] **Secret Isolation & Key Sanitization:** Environment variables (`NEXTOMNI_STOCK_*`) validated with `validate_api_key()` and masked via `mask_secret()` to prevent leaks in logs and UI.
- [x] **Rate Limiter Health:** Token bucket properly bounds requests (max 8 req/min for free tier) and logs warnings on failover.
- [x] **FSM Idempotency:** Running `evaluate-pending` multiple times produces identical records without duplicating evaluations.
- [x] **Safe Schema Evolution:** Idempotent `ALTER TABLE` migrations safely evolve SQLite tables (`lower_bound`, `upper_bound`, `confidence_level`, `audit_hash`).
- [x] **Cryptographic SHA-256 Audit Trail:** Deterministic canonical hashing protects `predictions` and `prediction_evaluations` records at rest.
- [x] **Tamper-Evident Verification:** `verify-audit` CLI command verifies database records against unauthorized modification.
- [x] **Cryptographic Audit Manifest Signing:** `list-predictions --export` generates SHA-256 manifest signatures for verifiable external compliance exports.
- [x] **Online Atomic SQLite Backup & Disaster Recovery:** `backup_database()` leverages SQLite's online backup API (`src_conn.backup(dst_conn)`) for transactionally consistent, zero-downtime snapshots under active WAL writes, followed by automatic cryptographic tamper verification.
- [x] **Cryptographic HMAC-SHA256 Model Signing at Rest:** All serialized models (`.joblib`) automatically write signed sidecars (`.joblib.sig`) verified upon loading via `verify_model_artifact()` to prevent weights tampering.
- [x] **Automated Continuous Database Tamper Alertness:** `EvaluationDaemon.run_once()` validates cryptographic database integrity during reconciliation cycles, raising critical system alerts upon any detected tamper event.
- [x] **Secure Parameter-Bound Storage Lifecycle Pruning:** `prune_raw_payloads()` safely purges expired raw JSON blobs (`retrieved_at < datetime('now', '-N days')`) using parameter binding `?`, protecting SQLite page storage while guaranteeing 100% inviolability of predictions and evaluation audit trails.

---

## 3. Architecture & Code Quality Checklist

- [x] **Polymorphic Forecaster Contract:** `BaseForecaster(ABC)` defines polymorphic interface for `fit`, `predict`, `predict_interval`, and serialization with `**kwargs` support.
- [x] **Convex Ensemble Meta-Model:** `ConvexEnsembleForecaster` combines Ridge and HistGradientBoosting via leak-free out-of-fold non-negative bounded least squares ($w_{\text{lin}} \ge 0, w_{\text{gb}} \ge 0, w_{\text{lin}} + w_{\text{gb}} = 1.0$), capturing non-linear volatility while retaining linear stability.
- [x] **Horizon-Adaptive Regularization:** Direct multi-horizon engine applies asymmetric capacity constraints (`max_depth=1`, `l2=20.0` for 1-session noise shrinkage vs `max_depth=3`, `l2=1.0` for multi-step momentum).
- [x] **Extended Quantitative Tail-Risk Engine:** `src/analytics/risk.py` computes Parametric Gaussian, Cornish-Fisher expansion VaR (accounting for skewness & excess kurtosis), Empirical Historical VaR, and Conditional Value-at-Risk (Expected Shortfall / CVaR) across multi-horizon forecasts.
- [x] **Quantitative Risk-Forecast Visual Overlay:** Tab 3 directly synchronizes Cornish-Fisher 95% multi-horizon VaR downside price floors onto forward forecasting trajectory cones.
- [x] **Storage Maintenance CLI Command:** `prune-db` command in `src/cli.py` exposes automated maintenance with input validation on retention horizons.
- [x] **Extended Risk Metrics:** `calculate_max_error` (Chebyshev $\ell_\infty$ norm worst-case risk) and `calculate_information_ratio` (benchmark-relative active alpha per tracking error) integrated into evaluation core and reports.
- [x] **Walk-Forward Quantile Bypass:** `fit(fit_quantiles=False)` accelerates walk-forward evaluation by ~65% while keeping intervals active in production.
- [x] **12-Signal Quantitative Feature Engine:** `vol_regime_ratio` ($\sigma_5 / \sigma_{21}$) and `macd_norm` integrated with zero lookahead leakage.
- [x] **Uncertainty Quantification:** Calibrated 80% prediction intervals ($[P_{\text{lower}}, P_{\text{upper}}]$) via both train-fold residual standard error and asymmetric quantile tree regression.
- [x] **Non-Parametric Quantile Regression:** Dual `HistGradientBoostingRegressor(loss="quantile", quantile=0.10/0.90)` models capture empirical fat tails and return skewness.
- [x] **Automated Session-Aware Daemon:** `EvaluationDaemon` and `daemon-reconcile` CLI command orchestrate market state checks and idempotent price reconciliation.
- [x] **Interactive Dashboard Security & Disaster Recovery Console:** Embedded real-time cryptographic audit verification, session-aware daemon triggers, and 1-click atomic database backup in Tab 4.
- [x] **Explainable AI (XAI) Additive Attribution:** `explain_forecast()` decomposes expected returns $\hat{r}_{t+k}$ into closed-form linear and marginal non-linear signal attributions across all models, ensuring exact additivity ($\sum c_i + \text{intercept} = \hat{r}$).
- [x] **Interactive Signal Attribution Visualizer:** Streamlit Tab 3 embeds an expander with horizontal color-coded contribution bar charts (bullish green vs. bearish red) for each horizon.
- [x] **CLI Explainability Flag:** `predict --explain` displays local signal attribution breakdown table in terminal.
- [x] **Walk-Forward Quantitative Backtester:** `src/analytics/backtester.py` simulates realistic execution with turnover transaction fee deductions (`fee_bps`), computing realized Annualized Sharpe, Sortino ratio, Max Drawdown, Profit Factor, and compounding equity curves vs. Buy & Hold benchmark.
- [x] **Interactive Strategy Equity Curve Visualizer:** Streamlit Tab 2 embeds an interactive backtest card with compounding equity curves ($100 base) and risk-adjusted return metrics.
- [x] **CLI Backtest Command:** `backtest` command in `src/cli.py` exposes strategy simulation with customizable horizons and fee parameters.
- [x] **Single-Main Consolidated CLI:** `src/cli.py` consolidated into a unified parser architecture with zero dead code, including `backup-db`, `risk-report`, `prune-db`, and `backtest`.
- [x] **Type Annotations:** All public functions include parameter types and return type annotations.
- [x] **Zero Regressions:** `pytest -v` passes 100% across all unit and integration tests (101/101 tests passing).
- [x] **Docstrings:** Standard Google/NumPy format explaining arguments, return values, and mathematical formulas.
- [x] **Decoupled Modules:** No circular imports between `src/data`, `src/features`, `src/models`, `src/evaluation`, `src/security`, and `src/dashboard`.
- [x] **Reproducibility:** Random seeds fixed (`random_state=42`) in all stochastic models or splitters.



