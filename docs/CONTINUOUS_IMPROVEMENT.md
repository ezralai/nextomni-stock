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

### Model Acceptance Criteria
1. **Short Horizon ($P_{t+1}$):** Model MAE within $\pm 5-10\%$ of baseline; Directional Accuracy $> 50\%$.
2. **Intermediate Horizon ($P_{t+5}$):** Model MAE must beat baseline by $\ge 0.5\%$; Directional Accuracy $\ge 58\%$. (HistGradientBoosting 12-Feats: **+13.93% to +19.12%**, **65.42% to 70.30% DA**).
3. **Long Horizon ($P_{t+21}$):** Model MAE must beat baseline by $\ge 10.0\%$; Directional Accuracy $\ge 65\%$. (HistGradientBoosting 12-Feats: **+12.66% to +46.27%**, **73.63% to 90.59% DA**).

---

## 2. System Security Verification Checklist

- [x] **SQL Query Safety:** All database queries utilize parameter binding (`?`). Zero string formatting in SQL statements.
- [x] **Input Allowlisting:** Ticker symbols match regex `^[A-Z0-9]{1,6}(\.[A-Z]{2})?$`; Exchange MIC matches `^(XNYS|XKLS)$` via `src/security/validators.py`.
- [x] **Model Family Allowlisting:** Model choice validated against `{"linear", "gradient_boosting", "baseline"}`.
- [x] **Floating-Point Finite Assertion:** Database writes validate `assert_finite_number` to protect against NaN/Inf float poisonings across prices and interval bounds.
- [x] **Path Traversal Protection:** Serialized model filenames sanitized and checked with `assert_safe_model_path`.
- [x] **Secret Isolation & Key Sanitization:** Environment variables (`NEXTOMNI_STOCK_*`) validated with `validate_api_key()` and masked via `mask_secret()` to prevent leaks in logs and UI.
- [x] **Rate Limiter Health:** Token bucket properly bounds requests (max 8 req/min for free tier) and logs warnings on failover.
- [x] **FSM Idempotency:** Running `evaluate-pending` multiple times produces identical records without duplicating evaluations.
- [x] **Safe Schema Evolution:** Idempotent `ALTER TABLE` migrations safely evolve SQLite tables (`lower_bound`, `upper_bound`, `confidence_level`, `audit_hash`).
- [x] **Cryptographic SHA-256 Audit Trail:** Deterministic canonical hashing protects `predictions` and `prediction_evaluations` records at rest.
- [x] **Tamper-Evident Verification:** `verify-audit` CLI command verifies database records against unauthorized modification.
- [x] **Cryptographic Audit Manifest Signing:** `list-predictions --export` generates SHA-256 manifest signatures for verifiable external compliance exports.

---

## 3. Architecture & Code Quality Checklist

- [x] **Polymorphic Forecaster Contract:** `BaseForecaster(ABC)` defines polymorphic interface for `fit`, `predict`, `predict_interval`, and serialization with `**kwargs` support.
- [x] **Walk-Forward Quantile Bypass:** `fit(fit_quantiles=False)` accelerates walk-forward evaluation by ~65% while keeping intervals active in production.
- [x] **12-Signal Quantitative Feature Engine:** `vol_regime_ratio` ($\sigma_5 / \sigma_{21}$) and `macd_norm` integrated with zero lookahead leakage.
- [x] **Uncertainty Quantification:** Calibrated 80% prediction intervals ($[P_{\text{lower}}, P_{\text{upper}}]$) via both train-fold residual standard error and asymmetric quantile tree regression.
- [x] **Non-Parametric Quantile Regression:** Dual `HistGradientBoostingRegressor(loss="quantile", quantile=0.10/0.90)` models capture empirical fat tails and return skewness.
- [x] **Automated Session-Aware Daemon:** `EvaluationDaemon` and `daemon-reconcile` CLI command orchestrate market state checks and idempotent price reconciliation.
- [x] **Interactive Dashboard Security Console:** Embedded real-time cryptographic audit verification and session-aware daemon triggers directly in Tab 4.
- [x] **Single-Main Consolidated CLI:** `src/cli.py` consolidated into a unified parser architecture with zero dead code.
- [x] **Type Annotations:** All public functions include parameter types and return type annotations.
- [x] **Zero Regressions:** `pytest -v` passes 100% across all unit and integration tests (67/67 tests passing).
- [x] **Docstrings:** Standard Google/NumPy format explaining arguments, return values, and mathematical formulas.
- [x] **Decoupled Modules:** No circular imports between `src/data`, `src/features`, `src/models`, `src/evaluation`, `src/security`, and `src/dashboard`.
- [x] **Reproducibility:** Random seeds fixed (`random_state=42`) in all stochastic models or splitters.


