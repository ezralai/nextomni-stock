"""Exchange session engine for NextOmni Stock (nextomni-stock).

Accurately models real-time market operational status for NYSE and Bursa Malaysia,
including Bursa Malaysia's official midday lunch break and NYSE pre/post-market sessions.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo
import pandas as pd
import exchange_calendars as xcals

from src.config import SUPPORTED_EXCHANGES

# Operational Status Constants
STATUS_OPEN = "OPEN"
STATUS_CLOSED = "CLOSED"
STATUS_LUNCH_BREAK = "LUNCH_BREAK"
STATUS_PRE_MARKET = "PRE_MARKET"
STATUS_AFTER_HOURS = "AFTER_HOURS"
STATUS_WEEKEND = "WEEKEND"
STATUS_HOLIDAY = "HOLIDAY"


@dataclass(frozen=True)
class SessionStatusInfo:
    """Dataclass encapsulating comprehensive market status at a point in time."""
    exchange: str
    name: str
    status: str
    is_open: bool
    local_time: str
    timezone: str
    next_open: Optional[str]
    next_close: Optional[str]
    session_date: str
    details: str


class ExchangeSessionEngine:
    """Engine computing live trading session states and countdowns."""

    _calendars = {}

    @classmethod
    def get_calendar(cls, exchange: str):
        if exchange not in cls._calendars:
            cls._calendars[exchange] = xcals.get_calendar(exchange)
        return cls._calendars[exchange]

    @classmethod
    def _find_next_session(cls, cal, date_str: str) -> pd.Timestamp:
        """Find next trading session from either a session date or non-session date."""
        if cal.is_session(date_str):
            return cal.next_session(date_str)
        return cal.date_to_session(date_str, direction="next")

    @classmethod
    def get_status(
        cls,
        exchange: str,
        as_of: Optional[datetime] = None,
    ) -> SessionStatusInfo:
        """Determine detailed market status for an exchange.

        Args:
            exchange: Market Identifier Code ('XNYS' or 'XKLS').
            as_of: Datetime to evaluate. If None, uses current system time.

        Returns:
            SessionStatusInfo with operational status, localized timestamps, and descriptions.
        """
        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange '{exchange}'. Must be one of {list(SUPPORTED_EXCHANGES.keys())}")

        cfg = SUPPORTED_EXCHANGES[exchange]
        cal = cls.get_calendar(exchange)
        tz = ZoneInfo(cfg.timezone)

        # Standardize evaluation time
        if as_of is None:
            now_dt = datetime.now(tz)
        else:
            if as_of.tzinfo is None:
                now_dt = as_of.replace(tzinfo=tz)
            else:
                now_dt = as_of.astimezone(tz)

        date_str = now_dt.strftime("%Y-%m-%d")
        time_curr = now_dt.time()
        local_time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        # 1. Check Weekend (Saturday=5, Sunday=6)
        if now_dt.weekday() >= 5:
            next_sess = cls._find_next_session(cal, date_str)
            next_open_str = f"{next_sess.strftime('%Y-%m-%d')} {cfg.session_open}:00"
            return SessionStatusInfo(
                exchange=exchange,
                name=cfg.name,
                status=STATUS_WEEKEND,
                is_open=False,
                local_time=local_time_str,
                timezone=cfg.timezone,
                next_open=next_open_str,
                next_close=None,
                session_date=date_str,
                details="Market is closed for the weekend.",
            )

        # 2. Check Exchange Holiday
        is_trading_day = cal.is_session(date_str)
        if not is_trading_day:
            next_sess = cls._find_next_session(cal, date_str)
            next_open_str = f"{next_sess.strftime('%Y-%m-%d')} {cfg.session_open}:00"
            return SessionStatusInfo(
                exchange=exchange,
                name=cfg.name,
                status=STATUS_HOLIDAY,
                is_open=False,
                local_time=local_time_str,
                timezone=cfg.timezone,
                next_open=next_open_str,
                next_close=None,
                session_date=date_str,
                details=f"Market is closed for an official exchange holiday. Next session on {next_sess.strftime('%Y-%m-%d')}.",
            )

        # 3. Trading Day Operational Windows
        if exchange == "XKLS":
            # Bursa Malaysia trading schedule:
            # Morning: 09:00 - 12:30 MYT
            # Midday Break: 12:30 - 14:30 MYT
            # Afternoon: 14:30 - 17:00 MYT
            t_open_am = time(9, 0)
            t_close_am = time(12, 30)
            t_open_pm = time(14, 30)
            t_close_pm = time(17, 0)

            if time_curr < t_open_am:
                next_open = f"{date_str} 09:00:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_CLOSED,
                    is_open=False,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=next_open,
                    next_close=None,
                    session_date=date_str,
                    details="Pre-opening session. Regular morning trading begins at 09:00 MYT.",
                )
            elif t_open_am <= time_curr < t_close_am:
                next_close = f"{date_str} 12:30:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_OPEN,
                    is_open=True,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=None,
                    next_close=next_close,
                    session_date=date_str,
                    details="Morning trading session in progress (09:00 - 12:30 MYT).",
                )
            elif t_close_am <= time_curr < t_open_pm:
                next_open = f"{date_str} 14:30:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_LUNCH_BREAK,
                    is_open=False,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=next_open,
                    next_close=None,
                    session_date=date_str,
                    details="Midday lunch break (12:30 - 14:30 MYT). Afternoon trading resumes at 14:30 MYT.",
                )
            elif t_open_pm <= time_curr < t_close_pm:
                next_close = f"{date_str} 17:00:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_OPEN,
                    is_open=True,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=None,
                    next_close=next_close,
                    session_date=date_str,
                    details="Afternoon trading session in progress (14:30 - 17:00 MYT).",
                )
            else:
                next_sess = cls._find_next_session(cal, date_str)
                next_open = f"{next_sess.strftime('%Y-%m-%d')} 09:00:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_CLOSED,
                    is_open=False,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=next_open,
                    next_close=None,
                    session_date=date_str,
                    details=f"Market closed for the day. Next trading session opens {next_sess.strftime('%Y-%m-%d')} at 09:00 MYT.",
                )

        else:
            # NYSE trading schedule:
            # Pre-market: 04:00 - 09:30 ET
            # Regular session: 09:30 - 16:00 ET
            # After-hours: 16:00 - 20:00 ET
            t_pre = time(4, 0)
            t_open = time(9, 30)
            t_close = time(16, 0)
            t_post = time(20, 0)

            if time_curr < t_pre:
                next_open = f"{date_str} 09:30:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_CLOSED,
                    is_open=False,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=next_open,
                    next_close=None,
                    session_date=date_str,
                    details="Market closed overnight. Regular trading opens at 09:30 ET.",
                )
            elif t_pre <= time_curr < t_open:
                next_open = f"{date_str} 09:30:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_PRE_MARKET,
                    is_open=False,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=next_open,
                    next_close=None,
                    session_date=date_str,
                    details="Pre-market trading session in progress. Regular session opens at 09:30 ET.",
                )
            elif t_open <= time_curr < t_close:
                next_close = f"{date_str} 16:00:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_OPEN,
                    is_open=True,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=None,
                    next_close=next_close,
                    session_date=date_str,
                    details="Regular trading session in progress (09:30 - 16:00 ET).",
                )
            elif t_close <= time_curr < t_post:
                next_sess = cls._find_next_session(cal, date_str)
                next_open = f"{next_sess.strftime('%Y-%m-%d')} 09:30:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_AFTER_HOURS,
                    is_open=False,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=next_open,
                    next_close=None,
                    session_date=date_str,
                    details="After-hours trading session in progress until 20:00 ET.",
                )
            else:
                next_sess = cls._find_next_session(cal, date_str)
                next_open = f"{next_sess.strftime('%Y-%m-%d')} 09:30:00"
                return SessionStatusInfo(
                    exchange=exchange,
                    name=cfg.name,
                    status=STATUS_CLOSED,
                    is_open=False,
                    local_time=local_time_str,
                    timezone=cfg.timezone,
                    next_open=next_open,
                    next_close=None,
                    session_date=date_str,
                    details=f"Market closed for the day. Next regular session opens {next_sess.strftime('%Y-%m-%d')} at 09:30 ET.",
                )
