"""Market data providers for NSE symbols.

- ``YahooProvider``     : Yahoo Finance chart API (works for ``*.NS`` tickers).
- ``CSVProvider``       : bars read from local CSV files.
- ``SyntheticProvider`` : deterministic random walks for demos and tests.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30), "IST")

INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "1h": 3600,
    "1d": 86400,
}

_RANGE_TABLE = {
    "1m": [("1d", 300), ("5d", 1300), ("7d", 1900)],
    "5m": [("1d", 75), ("5d", 375), ("1mo", 1500), ("3mo", 4500)],
    "15m": [("5d", 125), ("1mo", 500), ("3mo", 1500)],
    "1h": [("1mo", 150), ("3mo", 450), ("6mo", 900), ("1y", 1800)],
    "1d": [("6mo", 125), ("1y", 250), ("2y", 500), ("5y", 1250)],
}

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def interval_seconds(interval: str) -> int:
    return INTERVAL_SECONDS.get(interval, 300)


def range_for_interval(interval: str, bars: int) -> str:
    options = _RANGE_TABLE.get(interval) or _RANGE_TABLE["5m"]
    for rng, approx in options:
        if approx >= bars:
            return rng
    return options[-1][0]


@dataclass(frozen=True)
class Bar:
    ts: int          # epoch seconds of the bar open
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.ts, IST)

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Bar":
        return cls(
            ts=int(data["ts"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=int(data.get("volume") or 0),
        )


class YahooProvider:
    """Fetch OHLCV bars from the Yahoo Finance chart API."""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def __init__(self, timeout: int = 20, retries: int = 3, backoff: float = 1.5):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _get_json(self, url: str) -> dict:
        last_error = None
        for attempt in range(self.retries):
            request = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(self.backoff * (attempt + 1))
                    continue
                raise RuntimeError(f"Yahoo HTTP {exc.code} for {url}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(self.backoff * (attempt + 1))
        raise RuntimeError(f"Yahoo request failed after {self.retries} attempts: {last_error}")

    def fetch(self, symbol: str, interval: str = "5m", bars: int = 200, range_: str = None) -> list:
        rng = range_ or range_for_interval(interval, bars)
        url = (
            f"{self.BASE_URL}{urllib.parse.quote(symbol)}"
            f"?range={rng}&interval={interval}&includePrePost=false"
        )
        payload = self._get_json(url)
        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise RuntimeError(f"Yahoo error for {symbol}: {chart['error']}")
        result = (chart.get("result") or [None])[0]
        if not result:
            raise RuntimeError(f"Yahoo returned no result for {symbol}")
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        out = []
        for i, ts in enumerate(timestamps):
            try:
                bar = Bar(
                    ts=int(ts),
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=int(volumes[i] or 0),
                )
            except (IndexError, TypeError, ValueError):
                continue
            if bar.close > 0 and bar.high >= bar.low:
                out.append(bar)
        return out[-bars:]


class CSVProvider:
    """Read bars from ``<directory>/<symbol>.csv``."""

    def __init__(self, directory: str):
        self.directory = Path(directory)

    def _path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("^", "_")
        return self.directory / f"{safe}.csv"

    def fetch(self, symbol: str, interval: str = "5m", bars: int = 200, range_: str = None) -> list:
        path = self._path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found for {symbol}: {path}")
        out = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ts = row.get("ts") or row.get("timestamp") or row.get("date") or row.get("time")
                if ts is None:
                    continue
                try:
                    epoch = int(float(ts))
                except ValueError:
                    epoch = int(
                        datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        .astimezone(IST)
                        .timestamp()
                    )
                out.append(
                    Bar(
                        ts=epoch,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(float(row.get("volume") or 0)),
                    )
                )
        out.sort(key=lambda bar: bar.ts)
        return out[-bars:]


class SyntheticProvider:
    """Deterministic synthetic OHLCV series for demos and tests.

    Consecutive ``fetch`` calls continue the same random walk, so a demo loop
    observes evolving price action while staying fully reproducible.
    """

    def __init__(self, seed: int = 1234, bars: int = 240, base_vol: float = 0.011):
        self.seed = seed
        self.bars = bars
        self.base_vol = base_vol
        self._cache = {}

    def _phase(self, symbol: str) -> float:
        return random.Random(f"{self.seed}:{symbol}:phase").uniform(0.0, 2.0 * math.pi)

    def _make_bar(self, symbol: str, index: int, prev_close: float, ts: int) -> Bar:
        rng = random.Random(f"{self.seed}:{symbol}:bar:{index}")
        drift = 0.0016 * math.sin(2.0 * math.pi * index / 100.0 + self._phase(symbol)) + 0.00015
        shock = rng.gauss(0.0, self.base_vol)
        if rng.random() < 0.02:
            shock *= 3.0
        open_ = prev_close
        close_ = max(prev_close * math.exp(drift + shock), 5.0)
        high_ = max(open_, close_) * (1.0 + abs(rng.gauss(0.0, self.base_vol * 0.4)))
        low_ = min(open_, close_) * (1.0 - abs(rng.gauss(0.0, self.base_vol * 0.4)))
        volume = int(150000 + 900000 * rng.random() * (1.0 + abs(drift) * 150.0))
        return Bar(ts=ts, open=open_, high=high_, low=low_, close=close_, volume=volume)

    def fetch(self, symbol: str, interval: str = "5m", bars: int = 200, range_: str = None) -> list:
        step = interval_seconds(interval)
        count = bars or self.bars
        key = (symbol, interval, count)
        cached = self._cache.get(key)

        if cached is not None:
            series, start_index = cached
            new_bar = self._make_bar(symbol, start_index + len(series), series[-1].close,
                                     series[-1].ts + step)
            series = series[1:] + [new_bar]
            self._cache[key] = (series, start_index + 1)
            return series

        start_price = 150.0 + random.Random(f"{self.seed}:{symbol}:start").random() * 4500.0
        end = int(datetime.now(IST).timestamp())
        end -= end % step
        series = []
        for i in range(count):
            ts = end - (count - 1 - i) * step
            prev_close = start_price if not series else series[-1].close
            series.append(self._make_bar(symbol, i, prev_close, ts))
        self._cache[key] = (series, count)
        return series
