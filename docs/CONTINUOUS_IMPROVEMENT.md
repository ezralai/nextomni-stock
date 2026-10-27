# Continuous Improvement Framework (`nextomni-stock`)

This document provides actionable checklists, benchmarking criteria, and security audit rules for evaluating iterative improvements to the NextOmni Stock system.

---

## 1. Quantitative Benchmark Scorecard

Before adopting any new machine learning model (e.g., LightGBM, Random Forest, XGBoost, Neural Nets), record its out-of-sample walk-forward performance against the baseline in this scorecard:

| Model Architecture | Horizon | Test Ticker | Model MAE | Baseline MAE | MAE Delta % | Directional Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge (Current)** | 1 Session | `JPM` | 2.0531 | 1.9993 | -2.69% | 51.82% | Benchmark |
| **Ridge (Current)** | 5 Sessions | `JPM` | 4.6062 | 4.6367 | **+0.66%** | 59.43% | Production |
| **Ridge (Current)** | 21 Sessions | `JPM` | 9.0703 | 10.3134 | **+12.05%** | 68.89% | Production |
| **Ridge (Current)** | 1 Session | `1155.KL` | 0.0543 | 0.0522 | -3.92% | 47.62% | Benchmark |
| **Ridge (Current)** | 5 Sessions | `1155.KL` | 0.1065 | 0.1163 | **+8.43%** | 69.31% | Production |
| **Ridge (Current)** | 21 Sessions | `1155.KL` | 0.1477 | 0.2440 | **+39.46%** | 95.29% | Production |

### Model Acceptance Criteria
1. **Short Horizon ($P_{t+1}$):** Model MAE within $\pm 3\%$ of baseline; Directional Accuracy $> 50\%$.
2. **Intermediate Horizon ($P_{t+5}$):** Model MAE must beat baseline by $\ge 0.5\%$; Directional Accuracy $\ge 58\%$.
3. **Long Horizon ($P_{t+21}$):** Model MAE must beat baseline by $\ge 10.0\%$; Directional Accuracy $\ge 65\%$.

---

## 2. System Security Verification Checklist

- [ ] **SQL Query Safety:** All database queries utilize parameter binding (`?`). Zero string formatting in SQL statements.
- [ ] **Input Allowlisting:** Ticker symbols match regex `^[A-Z0-9]{1,6}(\.[A-Z]{2})?$`; Exchange MIC matches `^(XNYS|XKLS)$`.
- [ ] **Path Traversal Protection:** Serialized model filenames are generated from sanitized ticker strings without path manipulation characters (`../`, `/`).
- [ ] **Secret Isolation:** Environment variables (`NEXTOMNI_STOCK_*`) never logged to stdout or serialized in database payloads.
- [ ] **Rate Limiter Health:** Token bucket properly bounds requests (max 8 req/min for free tier) and logs warnings on failover.
- [ ] **FSM Idempotency:** Running `evaluate-pending` multiple times produces identical records without duplicating evaluations.

---

## 3. Architecture & Code Quality Checklist

- [ ] **Type Annotations:** All functions include parameter types and return type annotations.
- [ ] **Zero Regressions:** `pytest -v` passes 100% across all unit and integration tests.
- [ ] **Docstrings:** Standard Google/NumPy format explaining arguments, return values, and mathematical formulas.
- [ ] **Decoupled Modules:** No circular imports between `src/data`, `src/features`, `src/models`, `src/evaluation`, and `src/dashboard`.
- [ ] **Reproducibility:** Random seeds fixed (`random_state=42`) in all stochastic models or splitters.
