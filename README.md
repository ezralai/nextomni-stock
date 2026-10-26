# Real-Time Stock Market Monitoring, Price Forecasting & Prediction Evaluation System

A portfolio-quality, maintainable Python system designed for stock market monitoring, multi-horizon price forecasting, and automated predicted-versus-actual evaluation across two stock exchanges:
1. **New York Stock Exchange (NYSE)** — United States (`XNYS`, `America/New_York`, USD).
2. **Bursa Malaysia** — Malaysia (`XKLS`, `Asia/Kuala_Lumpur`, MYR).

---

## Key Features

- **Dual-Exchange Calendar Science:** Powered by `exchange_calendars` to account for official trading sessions, market hours, split trading sessions in Malaysia, weekends, and exchange holidays.
- **Multi-Horizon Forecasting:**
  - **Next Trading Session:** $P_{t+1}$
  - **5 Trading Sessions Ahead (1 Trading Week):** $P_{t+5}$
  - **21 Trading Sessions Ahead (1 Trading Month):** $P_{t+21}$
- **Strict Anti-Leakage Feature Engineering:** Leak-free lagged returns, rolling volatility, and volume indicators.
- **Transparent Baselines:** Evaluates all machine learning models against a naive persistence baseline ($\hat{P}_{t+k} = P_t$).
- **Prediction Tracking & Reconciled Evaluation:** Immutable prediction records evaluated against actual market settlement prices once target sessions complete.
- **Interactive Streamlit Dashboard:** Multi-page interface for market monitoring, exploratory analysis, forecasting, and historical model audit.

---

## Project Structure

```
stock/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Local SQLite database & raw data store
│   └── stock_system.db
├── src/
│   ├── config.py             # Exchanges, timezones, tickers, horizons
│   ├── data/                 # Market data providers & cleaning pipeline
│   ├── features/             # Leak-free feature engineering
│   ├── models/               # Persistence baseline & ML regressors
│   ├── evaluation/           # Metrics (MAE, RMSE, DA), validation & tracker
│   └── dashboard/            # Streamlit multi-page dashboard
└── tests/
    └── test_environment.py   # Test suite
```

---

## Setup & Installation

### Prerequisites
- Python 3.12 (Recommended for macOS binary wheel support)
- Git

### Quickstart
```bash
# Clone or navigate to the repository
cd /Users/bluevypc3/stock

# Create and activate Python 3.12 virtual environment
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the test suite
pytest -v
```

---

## Development Roadmap

- [x] **Phase 0:** Repository setup, data provider research, exchange calendar verification, and environment foundation.
- [ ] **Phase 1:** Historical data pipeline (SQLite storage, validation, Yahoo Finance & Twelve Data adapters).
- [ ] **Phase 2:** Exploratory data analysis (charts, distributions, volatility, drawdown).
- [ ] **Phase 3:** First forecast & persistence baseline ($P_{t+1}$).
- [ ] **Phase 4:** First machine learning model (Linear Regression) with walk-forward validation.
- [ ] **Phase 5:** Prediction tracking finite state machine & idempotent actual-price evaluation.
- [ ] **Phase 6:** Multi-horizon expansion ($P_{t+5}$, $P_{t+21}$).
- [ ] **Phase 7:** Market monitoring & data latency labeling (`REALTIME`, `DELAYED_15MIN`, `END_OF_DAY`).
- [ ] **Phase 8:** Interactive Streamlit Dashboard.
- [ ] **Phase 9:** Portfolio audit, performance documentation & reproducibility checks.
