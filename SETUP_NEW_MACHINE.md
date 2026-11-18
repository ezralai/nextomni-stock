# NextOmni Stock — New Machine Setup & Reproduction Guide

This guide walks you through setting up and running **NextOmni Stock** (`nextomni-stock`) from a fresh clone on any computer (macOS, Linux, or Windows).

Because all code, architecture, 227 quantitative finance algorithms, and 2,012 unit tests are fully tracked in Git, you can recreate all local files, databases, virtual environments, and models in **under 2 minutes**.

---

## 📋 Prerequisites

- **Git** (`git --version`)
- **Python 3.10, 3.11, or 3.12** (`python3 --version`)
- Internet connection (to download dependencies and initial market data)

---

## 🚀 8-Step Complete Setup Walkthrough

### Step 1: Clone the GitHub Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/ezralai/nextomni-stock.git
cd nextomni-stock
```

---

### Step 2: Create & Activate the Virtual Environment (`.venv/`)

Set up an isolated Python environment:

#### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Upgrade `pip` and install all required project packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Rebuild the CodeGraph AST & Dataflow Index

NextOmni Stock uses an internal CodeGraph index to track function calls, class hierarchies, and blast radius impact. Rebuild it with a single command:

```bash
python .agents/skills/codegraph/scripts/build_graph.py
```

*Output:* Generates `.agents/skills/codegraph/graph.json` indexing 483 files, 2,077 functions, 226 CLI subparsers, and over 203,000 code relationship edges.

---

### Step 4: Verify the Complete Test Suite

Run the full automated test suite to verify that your environment is 100% operational:

```bash
pytest
```

*Expected output:* `2012 passed` (100% pass rate).

---

### Step 5: Recreate Local Databases & Sync Market Data (`data/*.db`)

NextOmni Stock uses SQLite for local storage and cryptographic HMAC audit ledgers. All database tables are created automatically on demand.

#### 1. Sync sample equity data from Yahoo Finance:
```bash
# Sync US Equities (NYSE / NASDAQ)
python -m src.cli sync --ticker SPY
python -m src.cli sync --ticker JPM

# Sync Bursa Malaysia Equities (XKLS)
python -m src.cli sync --ticker 1155.KL --exchange XKLS
```

*Result:* Automatically creates and populates `data/stock_system.db` with clean daily OHLCV observations, corporate actions, and calendar validation.

#### 2. Initialize Audit Ledger (`data/nextomni_stock.db`):
The audit store (`data/nextomni_stock.db`) will automatically generate its schema upon your first quantitative evaluation:
```bash
python -m src.cli mancini-threshold --ticker SPY --persist
python -m src.cli carr-wu-vrp --ticker SPY --persist
python -m src.cli gatheral-rough-vol --ticker SPY --persist
```

---

### Step 6: Train Multi-Horizon Machine Learning Models (`models/`)

To train and save local direct multi-horizon forecasting models ($P_{t+1}$, $P_{t+5}$, $P_{t+21}$):

```bash
# Train linear model for SPY
python -m src.cli train --ticker SPY --model linear

# Train linear model for JPM
python -m src.cli train --ticker JPM --model linear

# Train linear model for 1155.KL (Bursa Malaysia)
python -m src.cli train --ticker 1155.KL --model linear
```

*Result:* Serializes trained model weights and feature scalers into `models/`.

---

### Step 7: Generate Exploratory Data Analysis Reports (`reports/`)

To generate visual HTML EDA reports with price distributions, volatility cones, and drawdown curves:

```bash
python -m src.cli eda --ticker SPY
python -m src.cli eda --ticker JPM
```

*Result:* Generates interactive HTML reports in `reports/eda_SPY.html` and `reports/eda_JPM.html`.

---

### Step 8: Launch CLI & Streamlit Interactive Dashboard

You are now fully set up! You can launch any feature:

#### 1. Interactive Web Dashboard:
```bash
streamlit run src/dashboard/app.py
```
Open your browser to `http://localhost:8501` to view real-time market monitors, multi-horizon forecasts, and 220+ quantitative analytics panels.

#### 2. Live Market Monitor (Terminal):
```bash
python -m src.cli monitor
python -m src.cli session-status
```

#### 3. Multi-Horizon Forecasting (Terminal):
```bash
python -m src.cli predict --ticker SPY --model linear --horizon 1
python -m src.cli predict --ticker SPY --model linear --horizon 5
python -m src.cli predict --ticker SPY --model linear --horizon 21
```

---

## 📁 Summary of What Gets Recreated

| Component | How It Is Recreated | Time Required |
| :--- | :--- | :--- |
| **`.venv/`** | `python3 -m venv .venv && pip install -r requirements.txt` | ~60 seconds |
| **`graph.json`** | `python .agents/skills/codegraph/scripts/build_graph.py` | ~10 seconds |
| **`data/stock_system.db`** | `python -m src.cli sync --ticker SPY` | ~5 seconds |
| **`data/nextomni_stock.db`** | Auto-created on first CLI / test execution with `--persist` | Instant |
| **`models/*.joblib`** | `python -m src.cli train --ticker SPY` | ~5 seconds |
| **`reports/*.html`** | `python -m src.cli eda --ticker SPY` | ~3 seconds |

---

## 🔒 Security & Environment Variables (Optional)

If you have a **Twelve Data** API key for real-time NYSE/Bursa Malaysia streaming quotes, set it in your environment:

#### macOS / Linux:
```bash
export NEXTOMNI_STOCK_TWELVE_DATA_API_KEY="your_api_key_here"
```

#### Windows (PowerShell):
```powershell
$env:NEXTOMNI_STOCK_TWELVE_DATA_API_KEY="your_api_key_here"
```

*(Note: If no API key is provided, the system gracefully falls back to Yahoo Finance automatically).*
