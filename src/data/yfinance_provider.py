"""Yahoo Finance data provider implementation for NextOmni Stock.

Retrieves historical OHLCV, dividends, and stock splits for NYSE and Bursa Malaysia.
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
import yfinance as yf

from src.config import FEED_DELAYED, FEED_EOD, SUPPORTED_EXCHANGES
from src.data.base_provider import BaseMarketDataProvider


class YahooFinanceProvider(BaseMarketDataProvider):
    """Market data provider using yfinance."""

    @property
    def name(self) -> str:
        return "YahooFinance"

    def _resolve_symbol(self, ticker: str, exchange: str) -> str:
        """Resolve ticker symbol according to exchange convention."""
        clean_ticker = ticker.strip().upper()
        if exchange == "XKLS" and not clean_ticker.endswith(".KL"):
            return f"{clean_ticker}.KL"
        return clean_ticker

    def fetch_historical_data(
        self,
        ticker: str,
        exchange: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch historical daily OHLCV observations."""
        symbol = self._resolve_symbol(ticker, exchange)
        yf_ticker = yf.Ticker(symbol)

        # Download historical data including corporate actions
        hist = yf_ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if hist.empty:
            return pd.DataFrame(), {"error": "Empty dataset returned from Yahoo Finance", "symbol": symbol}

        # Store raw payload for audit
        sample_records = hist.head(3).reset_index().astype(str).to_dict(orient="records")
        raw_payload = {
            "symbol": symbol,
            "exchange": exchange,
            "start_date": start_date,
            "end_date": end_date,
            "records_count": len(hist),
            "columns": list(hist.columns),
            "sample_head": sample_records,
        }

        # Normalize dataframe
        df = hist.reset_index()

        # Handle timezone in Date index
        date_col = "Date" if "Date" in df.columns else "Datetime"
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Strip timezone to get pure local exchange session date
            df["session_date"] = df[date_col].dt.strftime("%Y-%m-%d")
        else:
            df["session_date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

        # Column normalization
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = df.rename(columns=col_map)

        # If Adj Close is missing (rare in auto_adjust=False), fall back to close
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # Ensure all required numeric columns are present and clean
        req_cols = ["session_date", "open", "high", "low", "close", "adj_close", "volume"]
        df = df[req_cols].dropna().copy()

        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["adj_close"] = df["adj_close"].astype(float)
        df["volume"] = df["volume"].astype(int)

        df["data_status"] = FEED_EOD
        df["provider"] = self.name
        df["ticker"] = ticker
        df["exchange"] = exchange

        # Sort chronologically ascending
        df = df.sort_values("session_date", ascending=True).reset_index(drop=True)

        return df, raw_payload

    def fetch_corporate_actions(
        self,
        ticker: str,
        exchange: str,
    ) -> pd.DataFrame:
        """Fetch historical dividends and stock splits."""
        symbol = self._resolve_symbol(ticker, exchange)
        yf_ticker = yf.Ticker(symbol)

        records = []
        try:
            # Dividends
            dividends = yf_ticker.dividends
            if dividends is not None and not dividends.empty:
                for dt, val in dividends.items():
                    date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")
                    records.append(
                        {"action_date": date_str, "action_type": "DIVIDEND", "value": float(val)}
                    )

            # Splits
            splits = yf_ticker.splits
            if splits is not None and not splits.empty:
                for dt, val in splits.items():
                    date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")
                    records.append(
                        {"action_date": date_str, "action_type": "SPLIT", "value": float(val)}
                    )
        except Exception:
            pass

        if not records:
            return pd.DataFrame(columns=["action_date", "action_type", "value"])

        res_df = pd.DataFrame(records).sort_values("action_date").reset_index(drop=True)
        return res_df

    def fetch_latest_quote(
        self,
        ticker: str,
        exchange: str,
    ) -> Dict[str, Any]:
        """Fetch the latest available quote metadata."""
        symbol = self._resolve_symbol(ticker, exchange)
        yf_ticker = yf.Ticker(symbol)
        info = yf_ticker.fast_info

        try:
            last_price = float(info.last_price)
            prev_close = float(info.previous_close)
            change = last_price - prev_close
            pct_change = (change / prev_close) * 100 if prev_close > 0 else 0.0

            return {
                "ticker": ticker,
                "exchange": exchange,
                "symbol": symbol,
                "last_price": last_price,
                "previous_close": prev_close,
                "change": change,
                "pct_change": pct_change,
                "currency": getattr(info, "currency", "USD"),
                "status": FEED_DELAYED,
                "provider": self.name,
            }
        except Exception as e:
            return {
                "ticker": ticker,
                "exchange": exchange,
                "symbol": symbol,
                "error": str(e),
                "status": "UNAVAILABLE",
                "provider": self.name,
            }
