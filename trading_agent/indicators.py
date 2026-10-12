"""Technical indicators computed on plain float lists.

No third-party dependencies are used, so the agent runs on a bare Python
install. Every function returns ``None`` when there is not enough data.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev


def sma(values, period):
    """Simple moving average of the last ``period`` values."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return None
    return fmean(values[-period:])


def ema_series(values, period):
    """Full EMA series (seeded with the SMA of the first ``period`` values)."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return []
    k = 2.0 / (period + 1.0)
    out = [fmean(values[:period])]
    for value in values[period:]:
        out.append(value * k + out[-1] * (1.0 - k))
    return out


def ema(values, period):
    series = ema_series(values, period)
    return series[-1] if series else None


def rsi(closes, period=14):
    """Wilder's RSI. 100.0 for a pure uptrend (no losses)."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = fmean(gains)
    avg_loss = fmean(losses)
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


@dataclass(frozen=True)
class Macd:
    line: float
    signal: float
    histogram: float


def macd(closes, fast=12, slow=26, signal=9):
    """MACD line / signal / histogram."""
    if len(closes) < slow + signal:
        return None
    fast_series = ema_series(closes, fast)
    slow_series = ema_series(closes, slow)
    if not fast_series or not slow_series:
        return None
    line = [f - s for f, s in zip(fast_series[-len(slow_series):], slow_series)]
    if len(line) < signal:
        return None
    sig = ema_series(line, signal)
    if not sig:
        return None
    return Macd(line=line[-1], signal=sig[-1], histogram=line[-1] - sig[-1])


def true_ranges(highs, lows, closes):
    out = []
    for i in range(1, len(closes)):
        out.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    return out


def atr(highs, lows, closes, period=14):
    """Wilder's Average True Range."""
    if len(closes) < period + 1:
        return None
    trs = true_ranges(highs, lows, closes)
    if len(trs) < period:
        return None
    value = fmean(trs[:period])
    for tr in trs[period:]:
        value = (value * (period - 1) + tr) / period
    return value


@dataclass(frozen=True)
class Bollinger:
    middle: float
    upper: float
    lower: float
    percent_b: float
    bandwidth_pct: float


def bollinger(closes, period=20, mult=2.0, price=None):
    if len(closes) < period:
        return None
    window = closes[-period:]
    middle = fmean(window)
    sd = pstdev(window)
    upper = middle + mult * sd
    lower = middle - mult * sd
    ref = price if price is not None else closes[-1]
    band = upper - lower
    percent_b = (ref - lower) / band if band else 0.5
    bandwidth = (band / middle * 100.0) if middle else 0.0
    return Bollinger(middle=middle, upper=upper, lower=lower, percent_b=percent_b, bandwidth_pct=bandwidth)


def pct_change(new, old):
    if not old:
        return None
    return (new - old) / old * 100.0


def volatility_pct(closes, period=20):
    """Standard deviation of the last ``period`` simple returns, in percent."""
    if len(closes) < period + 1:
        return None
    returns = []
    for i in range(len(closes) - period, len(closes)):
        prev = closes[i - 1]
        if prev > 0:
            returns.append((closes[i] - prev) / prev)
    if len(returns) < 2:
        return None
    return pstdev(returns) * 100.0
