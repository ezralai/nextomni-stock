# NextOmni Stock (`nextomni-stock`)
## Real-Time Market Monitoring, Multi-Horizon Price Forecasting & Reconciled Prediction Evaluation

A portfolio-quality, production-ready quantitative financial intelligence system integrated within the **NextOmni** ecosystem. Designed for multi-exchange market monitoring, multi-horizon price forecasting, and automated predicted-versus-actual evaluation across two major international stock exchanges:
1. **New York Stock Exchange (NYSE)** — United States (`XNYS`, `America/New_York`, USD).
2. **Bursa Malaysia** — Malaysia (`XKLS`, `Asia/Kuala_Lumpur`, MYR).

---

## Architecture Overview

```
                                  +---------------------------------------+
                                  |         NextOmni Stock CLI &          |
                                  |     Interactive Streamlit Dashboard   |
                                  +-------------------+-------------------+
                                                      |
                   +----------------------------------+----------------------------------+
                   |                                  |                                  |
                   v                                  v                                  v
        +----------------------+          +----------------------+          +----------------------+
        |   Market Monitoring  |          | Multi-Horizon Engine |          |  Prediction Tracker  |
        |  & Session Status    |          | (1, 5, 21 Sessions)  |          |    & Reconciler      |
        +----------+-----------+          +----------+-----------+          +----------+-----------+
                   |                                 |                                 |
                   v                                 v                                 v
        +----------------------+          +----------------------+          +----------------------+
        | Exchange Calendars   |          | Feature Engineering  |          | Finite State Machine |
        | (XNYS / XKLS Hours)  |          | (Anti-Leakage Matrix)|          | (PENDING->EVALUATED) |
        +----------+-----------+          +----------+-----------+          +----------+-----------+
                   |                                 |                                 |
                   +---------------------------------+---------------------------------+
                                                     |
                                                     v
                                  +---------------------------------------+
                                  |    SQLite Storage & Audit Store       |
                                  | (stocks, observations, predictions)   |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Data Providers with Failover        |
                                  | (Twelve Data Real-time + Yahoo Fallback)|
                                  +---------------------------------------+
```

---

## Key Methodological Principles

### 1. Dual-Exchange Calendar Science
- Powered by `exchange_calendars` to handle exchange-specific holiday schedules, timezones, and trading hours.
- Accurately models **Bursa Malaysia's split daily sessions** (Morning: 09:00–12:30 MYT, Midday Lunch Break: 12:30–14:30 MYT, Afternoon: 14:30–17:00 MYT).
- Accurately models **NYSE trading hours** (Pre-Market: 04:00–09:30 ET, Regular Trading: 09:30–16:00 ET, After-Hours: 16:00–20:00 ET).

### 2. Multi-Horizon Direct Forecasting
Rather than compounding 1-step autoregressive errors recursively, we formulate independent **direct forecasting models** for each horizon $k \in \{1, 5, 21\}$:
- **Next Trading Session:** $P_{t+1}$
- **5 Trading Sessions Ahead (1 Trading Week):** $P_{t+5}$
- **21 Trading Sessions Ahead (1 Trading Month):** $P_{t+21}$

### 3. Return Target Formulation & Anti-Leakage Scaling
To ensure stationarity and guard against lookahead bias:
$$\hat{r}_{t+k} = \frac{P_{t+k} - P_t}{P_t}$$
$$\hat{P}_{t+k} = P_t \times (1 + \hat{r}_{t+k})$$
- Feature matrix consists of 7 quantitative signals: `log_return_1d`, `log_return_5d`, `price_to_sma_5`, `price_to_sma_20`, `rolling_vol_21`, `vol_ratio_5d`, `volume_change_1d`.
- Feature standardization (`StandardScaler`) is fit **strictly on training folds** inside each expanding window step to eliminate data leakage.

### 4. Transparent Persistence Benchmark
Every machine learning model is benchmarked against the standard financial random-walk persistence baseline:
$$\hat{P}_{t+k} = P_t$$
A model is only declared viable if it out-predicts persistence in out-of-sample walk-forward testing.

### 5. Prediction Lifecycle Finite State Machine (FSM)
Predictions are recorded immutably with UUID4 identifiers and evaluated against confirmed market prices:
```
[PENDING] ---> [AWAITING_ACTUAL] ---> [EVALUATED]
     |                 |
     +--------+--------+
              v
           [FAILED]
```

---

## Empirical Walk-Forward Validation Results

Chronological expanding-window walk-forward backtest (minimum initial training window: 120 trading sessions, evaluated fold-by-fold across historical out-of-sample data):

### JPMorgan Chase & Co. (`JPM` — NYSE)
| Horizon | Model MAE | Baseline MAE | MAE Improvement | Model RMSE | Baseline RMSE | Directional Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Session ($P_{t+1}$)** | 2.0531 | 1.9993 | -2.69% | 2.8028 | 2.7661 | 51.82% |
| **5 Sessions ($P_{t+5}$)** | 4.6062 | 4.6367 | **+0.66%** | 5.8644 | 5.9529 | **59.43%** |
| **21 Sessions ($P_{t+21}$)** | 9.0703 | 10.3134 | **+12.05%** | 10.7410 | 12.3789 | **68.89%** |

### Malayan Banking Berhad (`1155.KL` / Maybank — Bursa Malaysia)
| Horizon | Model MAE | Baseline MAE | MAE Improvement | Model RMSE | Baseline RMSE | Directional Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Session ($P_{t+1}$)** | 0.0543 | 0.0522 | -3.92% | 0.0734 | 0.0716 | 47.62% |
| **5 Sessions ($P_{t+5}$)** | 0.1065 | 0.1163 | **+8.43%** | 0.1345 | 0.1507 | **69.31%** |
| **21 Sessions ($P_{t+21}$)** | 0.1477 | 0.2440 | **+39.46%** | 0.1770 | 0.2762 | **95.29%** |

> **Key Quantitative Finding:** On ultra-short horizons (1 day), daily price variance is dominated by microstructural noise, meaning the persistence baseline ($\Delta P = 0$) achieves lower MAE. However, as the forecasting horizon extends to **5 sessions (1 week)** and **21 sessions (1 month)**, structural momentum and mean-reversion signals dominate noise, allowing the Ridge regression model to convincingly outperform the persistence baseline by **+12.05% (JPM)** and **+39.46% (Maybank)** with Directional Accuracy of **68.89%** and **95.29%**.

---

## Project Structure

```
stock/
├── .env.example              # Environment variables template (NEXTOMNI_STOCK_*)
├── .gitignore                # Git ignore configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Local SQLite database & raw payload store
│   └── stock_system.db
├── models/                   # Serialized multi-horizon model weights (*.joblib)
├── src/
│   ├── cli.py                # Comprehensive command-line interface
│   ├── config.py             # Exchanges, timezones, tickers, horizons
│   ├── analytics/            # EDA, statistical moments & Plotly charts
│   ├── data/                 # SQLite layer, TwelveData, YahooFinance & Pipeline
│   │   ├── db.py             # Database CRUD, indexes, and schema migrations
│   │   ├── base_provider.py  # Abstract market data provider interface
│   │   ├── yfinance_provider.py # Yahoo Finance adapter & quote resolver
│   │   ├── twelve_data_provider.py # Twelve Data client with rate limiting
│   │   ├── session_status.py # Real-time exchange operational status engine
│   │   ├── monitor.py        # Market monitor service with TTL caching
│   │   └── pipeline.py       # Data validation, sync & deduplication pipeline
│   ├── features/             # Anti-leakage quantitative feature matrix
│   ├── models/               # Direct multi-horizon regressors & persistence baseline
│   │   ├── splitter.py       # Chronological splitters & forward target alignment
│   │   ├── baseline.py       # Naive persistence baseline benchmark
│   │   ├── linear_model.py   # Ridge regressor with train-fold standardization
│   │   └── multi_horizon.py  # Multi-horizon direct forecasting orchestration
│   ├── evaluation/           # Evaluation metrics, validation & audit tracking
│   │   ├── metrics.py        # MAE, RMSE, guarded MAPE, Directional Accuracy
│   │   ├── validation.py     # Expanding-window walk-forward validator
│   │   └── tracker.py        # Prediction tracking finite state machine
│   └── dashboard/            # Interactive Streamlit intelligence dashboard
│       └── app.py            # 5-tab multi-page web application
└── tests/                    # Automated test suite (46 passing tests)
    ├── test_environment.py   # Calendar & dependency checks
    ├── test_data_pipeline.py # SQLite store, deduplication, and sync tests
    ├── test_analytics.py     # Returns, drawdowns, SMAs, volatility tests
    ├── test_baseline.py      # Persistence baseline and metrics tests
    ├── test_linear_model.py  # Feature matrix, model fit/predict/save tests
    ├── test_multi_horizon.py # Multi-horizon walk-forward validation tests
    ├── test_market_monitor.py # Session states, lunch break, and rate limit tests
    ├── test_prediction_tracker.py # Prediction FSM and reconciliation tests
    └── test_dashboard.py     # Streamlit app compilation and query tests
```

---

## Setup & Installation

### Prerequisites
- Python 3.12 (Recommended for macOS binary wheel support)
- Git

### Quickstart
```bash
# 1. Clone repository
git clone https://github.com/ezralai/nextomni-stock.git
cd nextomni-stock

# 2. Create and activate Python 3.12 virtual environment
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the full automated test suite (46 tests)
pytest -v
```

---

## Command-Line Interface (CLI) Guide

```bash
# 1. Sync historical data for default NYSE and Bursa Malaysia stocks
python -m src.cli sync --ticker JPM --exchange XNYS --start 2023-10-01 --end 2024-10-01
python -m src.cli sync --ticker 1155.KL --exchange XKLS --start 2023-10-01 --end 2024-10-01

# 2. Inspect database state
python -m src.cli status

# 3. Run Exploratory Data Analysis & export interactive HTML charts
python -m src.cli analyze --ticker JPM
python -m src.cli analyze --ticker 1155.KL

# 4. Generate next-session forecast & evaluate persistence benchmark
python -m src.cli forecast-baseline --ticker JPM
python -m src.cli forecast-baseline --ticker 1155.KL

# 5. Train Multi-Horizon direct models (1, 5, 21 sessions) with walk-forward validation
python -m src.cli train-all-horizons --ticker JPM
python -m src.cli train-all-horizons --ticker 1155.KL

# 6. Generate and register a forecast prediction into audit store
python -m src.cli predict --ticker JPM --model linear --horizon 1
python -m src.cli predict --ticker 1155.KL --model linear --horizon 5

# 7. Reconcile past predictions with confirmed exchange actual prices
python -m src.cli evaluate-pending

# 8. Inspect predictions audit log
python -m src.cli list-predictions

# 9. Inspect real-time exchange operational status and calendar countdowns
python -m src.cli session-status

# 10. Monitor real-time quotes with session status and latency badges
python -m src.cli monitor
python -m src.cli monitor --ticker JPM --exchange XNYS
python -m src.cli monitor --ticker 1155.KL --exchange XKLS
```

---

## Interactive Streamlit Dashboard

Launch the multi-page dashboard:
```bash
streamlit run src/dashboard/app.py
```

### Dashboard Tabs:
1. **Live Market Monitor:** Real-time exchange operational status (NYSE, Bursa Malaysia), countdown timers, live quotes table with data latency badges (`REALTIME` / `DELAYED_15MIN`), provider source, and cache refresh button.
2. **Stock Analytics & EDA:** Candlestick/Adjusted Close chart with SMA 5, SMA 20, SMA 50 overlays, volume histogram, 21-session rolling volatility, and peak-to-trough drawdown curves.
3. **Multi-Horizon Forecasts:** 3 horizon KPI cards ($P_{t+1}$, $P_{t+5}$, $P_{t+21}$), forward price projection trajectory, and one-click "Register Forecasts in Audit Log" button.
4. **Prediction Audit Log:** Reconciled predicted-versus-actual audit log, directional accuracy badges (✓ / ✗), and one-click "Reconcile Past Predictions" button.
5. **Model Performance:** Detailed walk-forward validation comparison table and quantitative Ridge beta coefficient importance breakdown.

---

## Development Roadmap

- [x] **Phase 0:** Repository setup, data provider research, exchange calendar verification, and environment foundation.
- [x] **Phase 1:** Historical data pipeline (SQLite storage, validation, Yahoo Finance adapter, corporate actions, and calendar cross-referencing).
- [x] **Phase 2:** Exploratory data analysis (charts, distributions, volatility, drawdown).
- [x] **Phase 3:** First forecast & persistence baseline ($P_{t+1}$).
- [x] **Phase 4:** First machine learning model (Linear Regression) with walk-forward validation.
- [x] **Phase 5:** Prediction tracking finite state machine & idempotent actual-price evaluation.
- [x] **Phase 6:** Multi-horizon expansion ($P_{t+5}$, $P_{t+21}$).
- [x] **Phase 7:** Market monitoring & data latency labeling (`REALTIME`, `DELAYED_15MIN`, `END_OF_DAY`).
- [x] **Phase 8:** Interactive Streamlit Dashboard.
- [x] **Phase 9:** Portfolio audit, performance documentation & reproducibility checks.

---

## Remote Repository & Git Push

To push this codebase to your newly created GitHub repository:
```bash
# Link remote origin
git remote add origin https://github.com/ezralai/nextomni-stock.git

# Push main branch
git push -u origin main
```
