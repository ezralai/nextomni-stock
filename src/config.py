"""Configuration constants and exchange settings for the stock monitoring & forecasting system.

This module acts as the single source of truth for exchange metadata,
timezones, supported initial tickers, and forecasting horizons.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "stock_system.db"


@dataclass(frozen=True)
class ExchangeConfig:
    mic: str
    name: str
    country: str
    timezone: str
    currency: str
    session_open: str
    session_close: str
    yahoo_suffix: str


# Supported Exchanges (Strictly NYSE and Bursa Malaysia)
SUPPORTED_EXCHANGES: Dict[str, ExchangeConfig] = {
    "XNYS": ExchangeConfig(
        mic="XNYS",
        name="New York Stock Exchange",
        country="United States",
        timezone="America/New_York",
        currency="USD",
        session_open="09:30",
        session_close="16:00",
        yahoo_suffix="",
    ),
    "XKLS": ExchangeConfig(
        mic="XKLS",
        name="Bursa Malaysia",
        country="Malaysia",
        timezone="Asia/Kuala_Lumpur",
        currency="MYR",
        session_open="09:00",
        session_close="17:00",
        yahoo_suffix=".KL",
    ),
}

# Supported Initial Tickers
INITIAL_TICKERS: Dict[str, List[str]] = {
    "XNYS": ["JPM", "KO", "WMT"],
    "XKLS": ["1155.KL", "1023.KL", "5347.KL"],
}

# Forecasting Horizons (Defined strictly in trading sessions)
HORIZON_NEXT_SESSION = 1
HORIZON_FIVE_SESSIONS = 5
HORIZON_TWENTY_ONE_SESSIONS = 21

SUPPORTED_HORIZONS = {
    "1_SESSION": HORIZON_NEXT_SESSION,
    "5_SESSIONS": HORIZON_FIVE_SESSIONS,
    "21_SESSIONS": HORIZON_TWENTY_ONE_SESSIONS,
}

# Price Type Standard: Adjusted Close to account for dividends & splits
PRICE_COLUMN = "adj_close"
RAW_CLOSE_COLUMN = "close"

# Prediction Lifecycle Statuses
STATUS_PENDING = "PENDING"
STATUS_AWAITING_ACTUAL = "AWAITING_ACTUAL"
STATUS_EVALUATED = "EVALUATED"
STATUS_FAILED = "FAILED"

# Data Feed Latency Statuses
FEED_REALTIME = "REALTIME"
FEED_DELAYED = "DELAYED_15MIN"
FEED_EOD = "END_OF_DAY"
