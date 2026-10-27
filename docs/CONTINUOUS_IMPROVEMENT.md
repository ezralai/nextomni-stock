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
| **Ridge (Iter 1: 10 Feats)** | 1 Session | `1155.KL` | 0.0424 | 0.0408 | -3.92% | 47.62% | Benchmark |
| **Ridge (Iter 1: 10 Feats)** | 5 Sessions | `1155.KL` | 0.0756 | 0.0842 | **+10.21%** | **64.36%** | Production |
| **Ridge (Iter 1: 10 Feats)** | 21 Sessions | `1155.KL` | 0.1703 | 0.2278 | **+25.24%** | **88.24%** | Production |

### Model Acceptance Criteria
1. **Short Horizon ($P_{t+1}$):** Model MAE within $\pm 3\%$ of baseline; Directional Accuracy $> 50\%$.
2. **Intermediate Horizon ($P_{t+5}$):** Model MAE must beat baseline by $\ge 0.5\%$; Directional Accuracy $\ge 58\%$.
3. **Long Horizon ($P_{t+21}$):** Model MAE must beat baseline by $\ge 10.0\%$; Directional Accuracy $\ge 65\%$.

---

## 2. System Security Verification Checklist

- [x] **SQL Query Safety:** All database queries utilize parameter binding (`?`). Zero string formatting in SQL statements.
- [x] **Input Allowlisting:** Ticker symbols match regex `^[A-Z0-9]{1,6}(\.[A-Z]{2})?$`; Exchange MIC matches `^(XNYS|XKLS)$` via `src/security/validators.py`.
- [x] **Path Traversal Protection:** Serialized model filenames sanitized and checked with `assert_safe_model_path`.
- [x] **Secret Isolation:** Environment variables (`NEXTOMNI_STOCK_*`) never logged to stdout or serialized in database payloads.
- [x] **Rate Limiter Health:** Token bucket properly bounds requests (max 8 req/min for free tier) and logs warnings on failover.
- [x] **FSM Idempotency:** Running `evaluate-pending` multiple times produces identical records without duplicating evaluations.

---

## 3. Architecture & Code Quality Checklist

- [ ] **Type Annotations:** All functions include parameter types and return type annotations.
- [ ] **Zero Regressions:** `pytest -v` passes 100% across all unit and integration tests.
- [ ] **Docstrings:** Standard Google/NumPy format explaining arguments, return values, and mathematical formulas.
- [ ] **Decoupled Modules:** No circular imports between `src/data`, `src/features`, `src/models`, `src/evaluation`, and `src/dashboard`.
- [ ] **Reproducibility:** Random seeds fixed (`random_state=42`) in all stochastic models or splitters.
