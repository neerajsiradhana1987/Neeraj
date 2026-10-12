"""Build the compact market snapshot that the LLM reasons about."""
from __future__ import annotations

from . import indicators as ind


def _round(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def _trend_label(fast, slow):
    if fast is None or slow is None:
        return "unknown"
    return "up" if fast > slow else "down"


def build_snapshot(symbol: str, bars, interval: str, portfolio: dict = None) -> dict:
    portfolio = portfolio or {}
    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    volumes = [bar.volume for bar in bars]

    last = closes[-1]
    prev = closes[-2] if len(closes) > 1 else last

    sma20 = ind.sma(closes, 20)
    sma50 = ind.sma(closes, 50)
    sma200 = ind.sma(closes, 200)
    ema12 = ind.ema(closes, 12)
    ema26 = ind.ema(closes, 26)
    rsi14 = ind.rsi(closes, 14)
    macd = ind.macd(closes)
    atr14 = ind.atr(highs, lows, closes, 14)
    bands = ind.bollinger(closes, 20, 2.0, price=last)
    vol20 = ind.volatility_pct(closes, 20)

    high20 = max(highs[-20:]) if len(highs) >= 20 else max(highs)
    low20 = min(lows[-20:]) if len(lows) >= 20 else min(lows)
    span = high20 - low20

    short_trend = _trend_label(last, sma20)
    mid_trend = _trend_label(sma20, sma50)
    long_trend = _trend_label(sma50, sma200)
    trends = [short_trend, mid_trend, long_trend]
    aligned = len({t for t in trends if t != "unknown"}) == 1

    avg_volume_20 = ind.sma(volumes, 20)

    return {
        "symbol": symbol,
        "currency": "INR",
        "timeframe": interval,
        "as_of": bars[-1].dt.isoformat(),
        "bars": len(bars),
        "last_price": _round(last, 2),
        "change_pct": _round(ind.pct_change(last, prev), 3),
        "change_pct_5bars": _round(ind.pct_change(last, closes[-6]), 3) if len(closes) >= 6 else None,
        "change_pct_20bars": _round(ind.pct_change(last, closes[-21]), 3) if len(closes) >= 21 else None,
        "volume": volumes[-1],
        "avg_volume_20": _round(avg_volume_20, 0),
        "sma_20": _round(sma20, 2),
        "sma_50": _round(sma50, 2),
        "sma_200": _round(sma200, 2),
        "ema_12": _round(ema12, 2),
        "ema_26": _round(ema26, 2),
        "rsi_14": _round(rsi14, 2),
        "macd": None if macd is None else {
            "line": _round(macd.line, 4),
            "signal": _round(macd.signal, 4),
            "histogram": _round(macd.histogram, 4),
        },
        "atr_14": _round(atr14, 2),
        "atr_pct_of_price": _round((atr14 / last * 100.0) if atr14 else None, 3),
        "bollinger_20": None if bands is None else {
            "upper": _round(bands.upper, 2),
            "middle": _round(bands.middle, 2),
            "lower": _round(bands.lower, 2),
            "percent_b": _round(bands.percent_b, 3),
            "bandwidth_pct": _round(bands.bandwidth_pct, 3),
        },
        "high_20bars": _round(high20, 2),
        "low_20bars": _round(low20, 2),
        "range_position_pct": _round((last - low20) / span * 100.0 if span else 50.0, 2),
        "volatility_20_pct": _round(vol20, 3),
        "trend": {"short": short_trend, "medium": mid_trend, "long": long_trend, "aligned": aligned},
        "portfolio": portfolio,
    }


def portfolio_context(broker, equity: float, day_start_equity: float, symbol: str, last_prices: dict) -> dict:
    """Portfolio facts handed to the LLM alongside the indicators."""
    position = broker.positions.get(symbol)
    day_pnl_pct = None
    if day_start_equity:
        day_pnl_pct = (equity - day_start_equity) / day_start_equity * 100.0
    return {
        "cash": round(broker.cash, 2),
        "equity": round(equity, 2),
        "open_positions": len(broker.positions),
        "day_pnl_pct": round(day_pnl_pct, 3) if day_pnl_pct is not None else None,
        "position_qty": position.qty if position else 0,
        "position_avg_price": round(position.avg_price, 2) if position else None,
        "position_stop_loss": position.stop_loss if position else None,
        "position_take_profit": position.take_profit if position else None,
        "last_prices": {sym: round(price, 2) for sym, price in last_prices.items()},
    }
