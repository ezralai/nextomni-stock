# NextOmni Stock (`nextomni-stock`) — AI Agent Continuous Refinement Rules

This document defines the core operational standards for any AI coding agent working on the NextOmni Stock quantitative intelligence system.

Every iteration must continuously refine the repository across three fundamental pillars: **System Architecture**, **System Security**, and **Machine Learning Abilities**.

---

## Pillar 1: System Architecture

1. **Domain-Driven Layering:**
   - Maintain strict separation of concerns:
     - `src/data/`: Data ingestion, database access, provider adapters, session tracking, and caching.
     - `src/features/`: Leak-free quantitative feature calculation and feature store logic.
     - `src/models/`: Direct multi-horizon regressors, baselines, and temporal splitters.
     - `src/evaluation/`: Reconciled prediction tracking, metrics, and walk-forward validation.
     - `src/analytics/`: Exploratory data analysis, statistical moments, and interactive charts.
     - `src/cli.py`: Terminal commands and orchestration.
     - `src/dashboard/`: Interactive Streamlit web user interface.
2. **Minimal, Targeted Changes:**
   - Do not introduce unnecessary abstractions, microservices, or external messaging queues.
   - Reuse existing services (`DataPipeline`, `MarketMonitorService`, `MultiHorizonForecaster`, `PredictionTracker`).
3. **Caching & Latency Optimization:**
   - Every external network query must be guarded with client-side TTL caching (in-memory or SQLite).
   - Prevent redundant API calls across CLI commands and dashboard re-renders.
4. **Clean Code & Type Safety:**
   - Maintain type annotations on all public functions.
   - Enforce PEP 8 style standards with zero dead code or unused imports.

---

## Pillar 2: System Security

1. **Input Validation & Sanitization:**
   - Strictly validate all tickers, exchange codes (MIC), date strings (`YYYY-MM-DD`), and model choices against allowlists.
   - Guard against path traversal when saving or loading serialized model files.
2. **Database Security:**
   - 100% of SQLite queries must use parameterized placeholders (`?`). Never concatenate user input into SQL strings.
   - Protect SQLite transactions with foreign key enforcement (`PRAGMA foreign_keys = ON;`).
3. **Credential & Secret Protection:**
   - Never commit API keys, tokens, or private secrets to the repository.
   - All credentials must be read exclusively from environment variables (e.g., `NEXTOMNI_STOCK_TWELVE_DATA_API_KEY`).
   - Keep `.env`, `*.db`, `.venv/`, and `models/**/*.joblib` strictly in `.gitignore`.
4. **Rate Limiting & Defensive Resilience:**
   - Maintain client-side token bucket rate limiters on all third-party market data providers (e.g., 8 requests/minute for Twelve Data).
   - Implement graceful failover to secondary providers (e.g., Yahoo Finance) on network errors or HTTP 429 quota exhaustion.
5. **Audit Trail Integrity:**
   - The predictions audit store (`predictions` and `prediction_evaluations`) must be treated as append-only.
   - Once evaluated, evaluation records must remain immutable and idempotent.

---

## Pillar 3: Machine Learning & Quantitative Intelligence

1. **Zero-Leakage Guarantee:**
   - Feature scalers (`StandardScaler`, `RobustScaler`), encoders, and imputers must be fitted **strictly on the training fold** of each expanding window.
   - All feature signals must be calculated exclusively using observations at or prior to time $t$ ($P_{\le t}$).
2. **Stationary Return Targets:**
   - Never predict raw non-stationary price levels directly with unconstrained regressors.
   - Formulate learning targets as expected percentage returns:
     $$\hat{r}_{t+k} = \frac{P_{t+k} - P_t}{P_t}, \quad \hat{P}_{t+k} = P_t \times (1 + \hat{r}_{t+k})$$
3. **Multi-Horizon Direct Forecasting:**
   - Train dedicated, direct models for each forecast horizon:
     - 1 Session ($P_{t+1}$)
     - 5 Sessions ($P_{t+5}$, 1 trading week)
     - 21 Sessions ($P_{t+21}$, 1 trading month)
   - Do not compound single-step autoregressive errors recursively across long horizons.
4. **Persistence Benchmark Standard:**
   - Every model proposal must be benchmarked against the naive financial persistence baseline:
     $$\hat{P}_{t+k} = P_t$$
   - A model is accepted only if it demonstrates statistically superior out-of-sample MAE and Directional Accuracy on walk-forward backtests.
5. **Walk-Forward Validation Protocol:**
   - Never use random k-fold cross-validation or shuffle time-series observations.
   - Use chronological expanding-window walk-forward validation (minimum initial training size: 120 sessions).
   - Always report MAE, RMSE, guarded MAPE, and Directional Accuracy.

---

## Iterative Refinement Cycle

When tasked with improving the system:
1. **Benchmark & Profile:** Inspect current performance, security boundaries, and feature matrices.
2. **Formulate Hypothesis:** Define a specific, testable improvement (e.g., adding Garman-Klass volatility, implementing LightGBM regressor, optimizing SQLite indices).
3. **Targeted Implementation:** Make minimal, clean code changes adhering to Pillars 1–3.
4. **Automated Verification:** Run `pytest -v` across the entire test suite. Verify 100% test pass rate.
5. **Document Impact:** Quantify the exact architectural, security, or ML performance improvement.
