"""NSE trading-session helpers. All times are IST (UTC+05:30)."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

from .data import IST, interval_seconds

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)

# NSE trading holidays (YYYY-MM-DD). Kept empty by default: refresh yearly from
# https://www.nseindia.com/resources/exchange-communication-holidays and add
# dates here or to the ``market.holidays`` list in config.json.
HOLIDAYS: set[str] = set()


def now_ist() -> datetime:
    return datetime.now(IST)


def is_weekend(day) -> bool:
    return day.weekday() >= 5


def is_holiday(day, holidays=()) -> bool:
    return day.isoformat()[:10] in set(holidays)


def is_market_open(moment: datetime, holidays=()) -> bool:
    if is_weekend(moment) or is_holiday(moment, holidays):
        return False
    return SESSION_OPEN <= moment.time() <= SESSION_CLOSE


def next_session_open(moment: datetime, holidays=()) -> datetime:
    candidate = datetime.combine(moment.date(), SESSION_OPEN, tzinfo=IST)
    if moment < candidate and is_market_open(candidate, holidays):
        return candidate
    probe = candidate + timedelta(days=1)
    while True:
        if not is_weekend(probe) and not is_holiday(probe, holidays):
            return probe
        probe += timedelta(days=1)


def business_days_between(day1, day2, holidays=()) -> int:
    """Inclusive count of business days from ``day1`` to ``day2``."""
    if day2 < day1:
        return 0
    holidays = set(holidays)
    days = 0
    probe = day1
    while probe <= day2:
        if probe.weekday() < 5 and probe.isoformat() not in holidays:
            days += 1
        probe += timedelta(days=1)
    return days


def is_data_fresh(bar_ts: int, interval: str, now, multiplier: float = 4.0, holidays=()):
    """Return ``(fresh, age_seconds, detail)`` for the newest bar.

    ``now`` may be a datetime or an epoch timestamp.
    """
    if isinstance(now, (int, float)):
        now = datetime.fromtimestamp(now, IST)
    bar_dt = datetime.fromtimestamp(bar_ts, IST)
    age = (now - bar_dt).total_seconds()
    if interval.endswith("d"):
        gap = business_days_between(bar_dt.date(), now.date(), holidays)
        return gap <= 2, age, f"{gap} trading day(s) since last bar"
    limit = interval_seconds(interval) * multiplier
    return age <= limit, age, f"age {age:.0f}s > limit {limit:.0f}s"
