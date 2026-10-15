"""Intraday analysis for the headline Indian indices.

For every index the module reports the four things an intraday desk looks at
before the open and during the session:

- **trend** - direction and strength of the current session (EMA stack, VWAP,
  market structure, momentum),
- **range** - the probable low/high for the session (previous session range,
  central pivot range, opening range, ATR and the option-implied straddle),
- **reversal warnings** - exhaustion signals that the move may be about to
  turn (RSI extremes, divergences, band rides, engulfing candles, VWAP
  rejection, failed opening-range breaks, ...),
- **options** - max pain, open-interest walls, put/call ratio, the side that
  is paying the most premium and the strikes where positions are built in bulk.

Everything here is pure computation over bars plus an optional
:class:`~trading_agent.options.OptionChain`; no network or state is touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median

from . import indicators as ind
from .data import Bar
from .market_hours import is_market_open, now_ist
from .options import OptionChain, SyntheticOptionChainProvider, analyze_chain

# Intraday index moves are small in percentage terms; these thresholds are the
# point at which a move is considered stretched enough to warn about.
STRETCH_PCT = 0.75          # price vs session open, in percent
STRETCH_VWAP_PCT = 0.004     # price vs VWAP, in percent of price
STRETCH_ATR_MULT = 1.5      # price vs VWAP, in ATR multiples
PARABOLIC_PCT = 1.1         # return over the last STRETCH_BARS bars

SEVERITY_POINTS = {"info": 6, "watch": 13, "alert": 25}


@dataclass(frozen=True)
class IndexSpec:
    """One tradable index: how to fetch it and how its option strike ladder is built."""

    key: str                        # NIFTY | BANKNIFTY | SENSEX
    name: str                       # display name
    symbol: str                     # Yahoo chart symbol for the index
    strike_step: float = 50.0       # strike ladder step
    lot_size: int = 25              # current option lot size (informational)


INDEX_SPECS: dict[str, IndexSpec] = {
    "NIFTY": IndexSpec("NIFTY", "NIFTY 50", "^NSEI", 50.0, 25),
    "BANKNIFTY": IndexSpec("BANKNIFTY", "NIFTY BANK", "^NSEBANK", 100.0, 15),
    "SENSEX": IndexSpec("SENSEX", "S&P BSE SENSEX", "^BSESN", 100.0, 10),
}

INDEX_ALIASES = {
    "NIFTY50": "NIFTY",
    "NIFTY 50": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTYBANK": "BANKNIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "SENSEX": "SENSEX",
    "BSE30": "SENSEX",
}


def resolve_index(name: str) -> IndexSpec:
    """Map user input (``nifty``, ``NIFTY 50``, ``banknifty``, ...) to a spec."""
    key = (name or "").upper().strip()
    if key in INDEX_SPECS:
        return INDEX_SPECS[key]
    compact = key.replace(" ", "").replace(".NS", "")
    if compact in INDEX_ALIASES:
        return INDEX_SPECS[INDEX_ALIASES[compact]]
    raise ValueError(f"unknown index {name!r}; choose from {sorted(INDEX_SPECS)}")


def resolve_indices(names) -> list:
    """Resolve a comma/space separated list of index names."""
    if not names:
        return list(INDEX_SPECS.values())
    out = []
    for name in str(names).replace(",", " ").split():
        spec = resolve_index(name)
        if spec not in out:
            out.append(spec)
    return out


# --------------------------------------------------------------------- sessions
def day_of(bar: Bar) -> str:
    """IST calendar day of a bar."""
    return bar.dt.strftime("%Y-%m-%d")


def split_sessions(bars: list) -> dict:
    """Bars grouped by IST calendar day, oldest day first."""
    out: dict[str, list] = {}
    for bar in bars:
        out.setdefault(day_of(bar), []).append(bar)
    return {day: sorted(rows, key=lambda bar: bar.ts) for day, rows in sorted(out.items())}


def session_bars(bars: list) -> list:
    """Bars of the most recent session present in the series."""
    days = split_sessions(bars)
    if not days:
        return []
    return list(days.values())[-1]


def previous_session(bars: list) -> list:
    """Bars of the session before the most recent one, if any."""
    days = list(split_sessions(bars).values())
    if len(days) < 2:
        return []
    return days[-2]


def session_vwap(bars: list) -> float | None:
    """Session VWAP from typical price and volume.

    Index feeds frequently report zero volume; in that case the function
    degrades to the mean typical price of the session, which is the usual
    fallback traders use for an index VWAP proxy.
    """
    if not bars:
        return None
    total_pv = 0.0
    total_volume = 0.0
    for bar in bars:
        typical = (bar.high + bar.low + bar.close) / 3.0
        total_pv += typical * (bar.volume or 0)
        total_volume += bar.volume or 0
    if total_volume > 0:
        return total_pv / total_volume
    return fmean((bar.high + bar.low + bar.close) / 3.0 for bar in bars)


def _rsi_series(closes: list, period: int = 14) -> list:
    """RSI per bar, aligned so ``result[i]`` belongs to ``closes[i]``."""
    return [ind.rsi(closes[:i + 1], period) for i in range(len(closes))]


# ----------------------------------------------------------------------- trend
def _market_structure(bars: list, lookback: int = 10) -> tuple[str, float]:
    """Higher-high/higher-low structure score of the recent bars."""
    recent = bars[-lookback:]
    prior = bars[-2 * lookback:-lookback] if len(bars) >= 2 * lookback else []
    if not recent or not prior:
        return "unknown", 0.0
    recent_high, recent_low = max(b.high for b in recent), min(b.low for b in recent)
    prior_high, prior_low = max(b.high for b in prior), min(b.low for b in prior)
    if recent_high > prior_high and recent_low > prior_low:
        return "higher high, higher low", 1.0
    if recent_high < prior_high and recent_low < prior_low:
        return "lower high, lower low", -1.0
    if recent_high > prior_high:
        return "expansion (new high, lower low)", 0.25
    if recent_low < prior_low:
        return "expansion (new low, higher high)", -0.25
    return "mixed", 0.0


def trend_analysis(bars: list) -> dict:
    """Direction and strength of the current move, on a -1..1 score."""
    closes = [bar.close for bar in bars]
    last = closes[-1]
    ema9 = ind.ema(closes, 9)
    ema21 = ind.ema(closes, 21)
    ema50 = ind.ema(closes, 50)
    sma20 = ind.sma(closes, 20)
    vwap = session_vwap(session_bars(bars))
    rsi14 = ind.rsi(closes, 14)
    slope_bars = min(10, len(closes) - 1)
    slope_pct = ind.pct_change(last, closes[-1 - slope_bars]) if slope_bars else None

    if ema9 and ema21 and ema50:
        if ema9 > ema21 > ema50:
            stack, stack_score = "bullish (9>21>50)", 1.0
        elif ema9 < ema21 < ema50:
            stack, stack_score = "bearish (9<21<50)", -1.0
        else:
            stack, stack_score = "mixed", 0.2 if ema9 > ema21 else -0.2
    else:
        stack, stack_score = "unknown", 0.0

    vwap_score = 0.0 if vwap is None else _sign_scale(last - vwap, last * 0.0008)
    ema_score = 0.0 if not ema9 else _sign_scale(last - ema9, last * 0.0005)
    structure, structure_score = _market_structure(bars)
    rsi_score = 0.0 if rsi14 is None else (rsi14 - 50.0) / 50.0
    slope_score = 0.0 if slope_pct is None else _sign_scale(slope_pct, 0.25)

    weights = [
        ("ema_stack", stack_score, 0.25),
        ("price_vs_vwap", vwap_score, 0.20),
        ("price_vs_ema9", ema_score, 0.10),
        ("structure", structure_score, 0.20),
        ("rsi_momentum", rsi_score, 0.15),
        ("slope_10bars", slope_score, 0.10),
    ]
    total_weight = sum(weight for _, _, weight in weights) or 1.0
    score = sum(value * weight for _, value, weight in weights) / total_weight

    if score >= 0.55:
        label = "strong_up"
    elif score >= 0.2:
        label = "up"
    elif score <= -0.55:
        label = "strong_down"
    elif score <= -0.2:
        label = "down"
    else:
        label = "sideways"

    return {
        "label": label,
        "score": round(score, 3),
        "strength": "strong" if abs(score) >= 0.55 else ("moderate" if abs(score) >= 0.2 else "weak"),
        "ema_stack": stack,
        "structure": structure,
        "slope_10bars_pct": round(slope_pct, 3) if slope_pct is not None else None,
        "ema_9": _round(ema9),
        "ema_21": _round(ema21),
        "ema_50": _round(ema50),
        "sma_20": _round(sma20),
        "vwap": _round(vwap),
        "rsi_14": round(rsi14, 2) if rsi14 is not None else None,
        "above_vwap": None if vwap is None else last > vwap,
        "components": {name: round(value, 3) for name, value, _ in weights},
    }


def _sign_scale(value: float, reference: float) -> float:
    """Map a signed difference to -1..1 using ``reference`` as full scale."""
    if not reference:
        return 0.0
    scaled = value / reference
    return max(-1.0, min(1.0, scaled))


def _round(value, digits: int = 2):
    return None if value is None else round(float(value), digits)


# ---------------------------------------------------------------------- levels
def pivot_points(prev: list) -> dict:
    """Classic floor pivots and the central pivot range of a closed session."""
    if not prev:
        return {}
    high = max(bar.high for bar in prev)
    low = min(bar.low for bar in prev)
    close = prev[-1].close
    pp = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = 2.0 * pp - bc
    span = high - low
    return {
        "pivot": _round(pp),
        "r1": _round(2 * pp - low),
        "r2": _round(pp + span),
        "r3": _round(high + 2 * (pp - low)),
        "s1": _round(2 * pp - high),
        "s2": _round(pp - span),
        "s3": _round(low - 2 * (high - pp)),
        "cpr_top": _round(tc),
        "cpr_bottom": _round(bc),
        "cpr_low": _round(min(tc, bc)),
        "cpr_high": _round(max(tc, bc)),
        # an inverted CPR (top below bottom) usually marks a trending session
        "cpr_inverted": bool(tc < bc),
        "previous_high": _round(high),
        "previous_low": _round(low),
        "previous_close": _round(close),
    }


def opening_range(bars: list, minutes: int = 15) -> dict | None:
    """Opening range (first ``minutes`` of the session) and its breakout levels."""
    if not bars:
        return None
    start = bars[0].dt
    window = [bar for bar in bars if (bar.dt - start).total_seconds() <= minutes * 60]
    if not window:
        return None
    high = _round(max(bar.high for bar in window))
    low = _round(min(bar.low for bar in window))
    span = high - low
    return {
        "minutes": minutes,
        "from": window[0].dt.strftime("%H:%M"),
        "high": high,
        "low": low,
        "size_pct": round(span / window[0].close * 100.0, 3) if window[0].close else None,
        "break_up": high,
        "break_down": low,
        "target_up": _round(high + span),
        "target_down": _round(max(low - span, 0.0)),
    }


def level_analysis(bars: list, prev: list) -> dict:
    """Support/resistance ladder for the current session."""
    current = session_bars(bars)
    if not current:
        current = bars
    spot = current[-1].close
    vwap = session_vwap(current)
    atr14 = ind.atr([b.high for b in bars], [b.low for b in bars], [b.close for b in bars], 14)
    pivots = pivot_points(prev)
    orbs = opening_range(current, 15)
    day_high = max(bar.high for bar in current)
    day_low = min(bar.low for bar in current)

    resistance = []
    support = []
    for label, value in (("day_high", day_high), ("pivot_r1", pivots.get("r1")),
                         ("pivot_r2", pivots.get("r2")), ("pivot_r3", pivots.get("r3")),
                         ("orb_high", (orbs or {}).get("high")),
                         ("prev_high", pivots.get("previous_high"))):
        if value and value > spot:
            resistance.append({"label": label, "price": _round(value),
                               "distance_pct": _pct_dist(value, spot)})
    for label, value in (("day_low", day_low), ("pivot_s1", pivots.get("s1")),
                         ("pivot_s2", pivots.get("s2")), ("pivot_s3", pivots.get("s3")),
                         ("orb_low", (orbs or {}).get("low")),
                         ("prev_low", pivots.get("previous_low"))):
        if value and value < spot:
            support.append({"label": label, "price": _round(value),
                            "distance_pct": _pct_dist(value, spot)})
    resistance.sort(key=lambda item: item["price"])
    support.sort(key=lambda item: -item["price"])

    return {
        "session_date": day_of(current[-1]) if current else None,
        "bars_in_session": len(current),
        "spot": _round(spot),
        "open": _round(current[0].open) if current else None,
        "high": _round(day_high),
        "low": _round(day_low),
        "prev_close": _round(pivots.get("previous_close")),
        "vwap": _round(vwap),
        "vwap_distance_pct": _pct_dist(spot, vwap) if vwap else None,
        "atr_14": _round(atr14),
        "atr_pct_of_price": _round(atr14 / spot * 100.0) if atr14 else None,
        "cpr": None if not pivots else {
            "top": pivots["cpr_top"], "bottom": pivots["cpr_bottom"],
            "low": pivots["cpr_low"], "high": pivots["cpr_high"],
            "inverted": pivots["cpr_inverted"],
            "pivot": pivots["pivot"],
            "inside": pivots["cpr_low"] <= spot <= pivots["cpr_high"],
        },
        "pivots": pivots or None,
        "opening_range": orbs,
        "resistance": resistance,
        "support": support,
    }


def _pct_dist(value: float, reference: float) -> float | None:
    if not reference:
        return None
    return round((value - reference) / reference * 100.0, 3)


# -------------------------------------------------------------------- reversal
def _warning(code: str, direction: str, severity: str, message: str) -> dict:
    return {"code": code, "direction": direction, "severity": severity, "message": message}


def _engulfing(bars: list) -> str | None:
    if len(bars) < 2:
        return None
    prev, last = bars[-2], bars[-1]
    prev_body = abs(prev.close - prev.open)
    last_body = abs(last.close - last.open)
    if last_body <= prev_body * 1.1 or prev_body == 0:
        return None
    if last.close > last.open and last.open <= prev.close and last.close >= prev.open:
        if last.close > prev.close:
            return "bullish"
    if last.close < last.open and last.open >= prev.close and last.close <= prev.open:
        if last.close < prev.close:
            return "bearish"
    return None


def _wick_rejection(bars: list) -> str | None:
    if not bars:
        return None
    bar = bars[-1]
    body = abs(bar.close - bar.open)
    span = bar.high - bar.low
    if span <= 0:
        return None
    upper = bar.high - max(bar.open, bar.close)
    lower = min(bar.open, bar.close) - bar.low
    if upper >= 2.0 * body and upper >= 0.4 * span:
        return "upper"
    if lower >= 2.0 * body and lower >= 0.4 * span:
        return "lower"
    return None


def _divergence(closes: list, rsi_values: list, lookback: int = 20) -> str | None:
    """Price/RSI divergence over the last ``lookback`` bars."""
    count = len(closes)
    if count < 12:
        return None

    def valid(index: int) -> bool:
        return rsi_values[index] is not None

    def extreme(indexes, pick: str) -> int:
        return (max if pick == "high" else min)(indexes, key=lambda i: closes[i])

    recent_end = count
    recent_start = max(0, count - lookback)
    prior_end = recent_start
    prior_start = max(0, count - 2 * lookback)
    if prior_start == prior_end or recent_start == recent_end:
        return None
    recent = range(recent_start, recent_end)
    prior = range(prior_start, prior_end)

    recent_high, prior_high = extreme(recent, "high"), extreme(prior, "high")
    if valid(recent_high) and valid(prior_high):
        if (closes[recent_high] > closes[prior_high]
                and rsi_values[recent_high] < rsi_values[prior_high] - 2.0):
            return "bearish"
    recent_low, prior_low = extreme(recent, "low"), extreme(prior, "low")
    if valid(recent_low) and valid(prior_low):
        if (closes[recent_low] < closes[prior_low]
                and rsi_values[recent_low] > rsi_values[prior_low] + 2.0):
            return "bullish"
    return None


def reversal_analysis(bars: list, levels: dict = None, chain=None, trend: dict = None) -> dict:
    """Exhaustion signals for the current session, with a 0-100 risk score."""
    levels = levels or {}
    closes = [bar.close for bar in bars]
    rsi_values = _rsi_series(closes, 14)
    last = closes[-1]
    spot = levels.get("spot") or last
    rsi14 = ind.rsi(closes, 14)
    macd_value = ind.macd(closes)
    bands = ind.bollinger(closes, 20, 2.0, price=last)
    vwap = levels.get("vwap")
    atr = levels.get("atr_14")
    session = session_bars(bars)
    session_high = max(bar.high for bar in session)
    session_low = min(bar.low for bar in session)
    session_span = session_high - session_low or 1.0
    session_open = levels.get("open") or session[0].open
    trend = trend or trend_analysis(bars)

    warnings: list[dict] = []

    def add(warning: dict | None):
        if warning:
            warnings.append(warning)

    if rsi14 is not None and rsi14 >= 70:
        add(_warning("RSI_OVERBOUGHT", "bearish",
                     "alert" if rsi14 >= 78 else "watch",
                     f"RSI {rsi14:.1f} - momentum is overbought, pullback risk"))
    if rsi14 is not None and rsi14 <= 30:
        add(_warning("RSI_OVERSOLD", "bullish",
                     "alert" if rsi14 <= 22 else "watch",
                     f"RSI {rsi14:.1f} - momentum is oversold, bounce risk"))
    divergence = _divergence(closes, rsi_values)
    if divergence == "bearish":
        add(_warning("RSI_BEARISH_DIVERGENCE", "bearish", "alert",
                     "price made a new high but RSI did not - distribution into strength"))
    elif divergence == "bullish":
        add(_warning("RSI_BULLISH_DIVERGENCE", "bullish", "alert",
                     "price made a new low but RSI did not - absorption"))
    if bands and bands.percent_b > 1.0:
        add(_warning("BAND_RIDE_UPPER", "bearish", "watch",
                     f"price is outside the upper Bollinger band (%B {bands.percent_b:.2f})"))
    elif bands and bands.percent_b < 0.0:
        add(_warning("BAND_RIDE_LOWER", "bullish", "watch",
                     f"price is outside the lower Bollinger band (%B {bands.percent_b:.2f})"))
    if vwap is not None:
        stretch = max(STRETCH_ATR_MULT * (atr or 0.0), STRETCH_VWAP_PCT * last)
        if abs(last - vwap) > stretch:
            add(_warning("VWAP_STRETCH", "bearish" if last > vwap else "bullish", "watch",
                         f"price is {abs(last - vwap) / last * 100:.2f}% away from VWAP "
                         "(mean-reversion zone)"))
        if trend["label"].startswith("up") and last < vwap:
            add(_warning("VWAP_CROSS_DOWN", "bearish", "alert",
                         "uptrend lost the VWAP - stop-loss reference broken"))
        elif trend["label"].startswith("down") and last > vwap:
            add(_warning("VWAP_CROSS_UP", "bullish", "alert",
                         "downtrend reclaimed the VWAP - shorts in trouble"))
    engulfing = _engulfing(bars)
    if engulfing == "bearish":
        add(_warning("ENGULFING_BEARISH", "bearish", "watch",
                     "bearish engulfing candle on the last bar"))
    elif engulfing == "bullish":
        add(_warning("ENGULFING_BULLISH", "bullish", "watch",
                     "bullish engulfing candle on the last bar"))
    wick = _wick_rejection(bars)
    if wick == "upper":
        add(_warning("UPPER_WICK_REJECTION", "bearish", "watch",
                     "long upper shadow - sellers rejected the highs"))
    elif wick == "lower":
        add(_warning("LOWER_WICK_REJECTION", "bullish", "watch",
                     "long lower shadow - buyers defended the lows"))
    move_pct = ind.pct_change(last, session_open)
    if move_pct is not None and abs(move_pct) >= STRETCH_PCT:
        add(_warning("STRETCH_FROM_OPEN", "bearish" if move_pct > 0 else "bullish",
                     "alert" if abs(move_pct) >= 2 * STRETCH_PCT else "watch",
                     f"price is {move_pct:+.2f}% from the open - extended move"))
    stretch_bars = min(10, len(closes) - 1)
    if stretch_bars >= 5:
        run_pct = ind.pct_change(last, closes[-1 - stretch_bars])
        if run_pct is not None and abs(run_pct) >= PARABOLIC_PCT:
            add(_warning("PARABOLIC_RUN", "bearish" if run_pct > 0 else "bullish", "alert",
                         f"{stretch_bars}-bar move of {run_pct:+.2f}% - exhaustion risk"))
    if macd_value is not None and len(closes) >= 4:
        previous = ind.macd(closes[:-1])
        if previous is not None:
            if macd_value.histogram > 0 > previous.histogram and closes[-1] < closes[-2]:
                add(_warning("MACD_FLIP_DOWN", "bearish", "watch",
                             "MACD histogram turned negative"))
            elif macd_value.histogram < 0 < previous.histogram and closes[-1] > closes[-2]:
                add(_warning("MACD_FLIP_UP", "bullish", "watch",
                             "MACD histogram turned positive"))
    position = (last - session_low) / session_span
    if position >= 0.9:
        add(_warning("AT_SESSION_HIGH", "bearish", "info",
                     "trading at the top of the session range"))
    elif position <= 0.1:
        add(_warning("AT_SESSION_LOW", "bullish", "info",
                     "trading at the bottom of the session range"))
    orbs = levels.get("opening_range")
    if orbs:
        if spot > orbs["high"] and last < orbs["high"]:
            add(_warning("ORB_FAILURE_UP", "bearish", "alert",
                         "broke the opening range high but closed back inside - failed breakout"))
        elif spot < orbs["low"] and last > orbs["low"]:
            add(_warning("ORB_FAILURE_DOWN", "bullish", "alert",
                         "broke the opening range low but closed back inside - failed breakdown"))
    warnings.extend(_option_warnings(spot, levels, chain))

    score = min(100, sum(SEVERITY_POINTS.get(w["severity"], 0) for w in warnings))
    bearish = sum(SEVERITY_POINTS.get(w["severity"], 0)
                  for w in warnings if w["direction"] == "bearish")
    bullish = sum(SEVERITY_POINTS.get(w["severity"], 0)
                  for w in warnings if w["direction"] == "bullish")
    if bearish or bullish:
        bias = "bearish" if bearish > bullish else ("bullish" if bullish > bearish else "mixed")
    else:
        bias = "none"
    return {
        "risk": "high" if score >= 60 else ("medium" if score >= 30 else "low"),
        "score": score,
        "bias": bias,
        "count": len(warnings),
        "warnings": warnings,
    }


def _option_warnings(spot: float, levels: dict, chain: OptionChain | None) -> list:
    """Reversal risks that only the option chain can show."""
    if chain is None or not spot:
        return []
    out = []
    analysis = levels.get("options") or {}
    walls = analysis.get("walls") or {}
    for row in walls.get("call_resistance") or []:
        if abs(row["distance_pct"]) <= 0.2:
            out.append(_warning("CE_WALL_AHEAD", "bearish", "watch",
                                f"call writers clustered at {row['strike']:.0f} "
                                f"({row['oi_lots'] or row['oi']} lots) - resistance"))
    for row in walls.get("put_support") or []:
        if abs(row["distance_pct"]) <= 0.2:
            out.append(_warning("PE_WALL_BELOW", "bullish", "watch",
                                f"put writers clustered at {row['strike']:.0f} "
                                f"({row['oi_lots'] or row['oi']} lots) - support"))
    pain = analysis.get("max_pain") or {}
    if pain.get("available") and pain.get("strike"):
        distance = abs(pain["strike"] - spot) / spot * 100.0
        if distance <= 0.15:
            direction = "bearish" if pain["strike"] < spot else "bullish"
            out.append(_warning("MAX_PIN_NEAR_SPOT", direction, "watch",
                                f"max pain {pain['strike']:.0f} sits just "
                                f"{distance:.2f}% away - magnet/pinning risk into expiry"))
    return out


# ----------------------------------------------------------------------- range
def typical_session_ranges(bars: list, sessions: int = 5) -> float | None:
    """Median high-low range of the last ``sessions`` complete sessions."""
    days = list(split_sessions(bars).values())
    if len(days) < 2:
        return None
    closed = days[:-1][-sessions:]
    ranges = [max(bar.high for bar in day) - min(bar.low for bar in day) for day in closed]
    return median(ranges) if ranges else None


def expected_range(bars: list, levels: dict = None, straddle: dict | None = None) -> dict:
    """Probable low/high for the session, in points and percent."""
    levels = levels or {}
    spot = levels.get("spot") or bars[-1].close
    centre = levels.get("vwap") or spot
    typical = typical_session_ranges(bars)
    sources = {
        "straddle_points": None if not straddle else straddle.get("straddle_points"),
        "typical_session_range": _round(typical),
        "atr_14": levels.get("atr_14"),
    }
    if straddle and straddle.get("straddle_points"):
        width = float(straddle["straddle_points"])
        basis = "straddle (options-implied)"
    elif typical:
        width = float(typical)
        basis = "median of the last sessions"
    else:
        width = None
        basis = "session so far"
    if width is None:
        session = session_bars(bars)
        low = min(bar.low for bar in session)
        high = max(bar.high for bar in session)
        width = high - low
    low, high = centre - width / 2.0, centre + width / 2.0
    outside = spot < low or spot > high
    low, high = min(low, spot), max(high, spot)
    span = high - low or 1.0
    return {
        "basis": basis,
        "low": _round(low),
        "high": _round(high),
        "width": _round(width),
        "width_pct": round(width / spot * 100.0, 3) if spot else None,
        "spot_outside": outside,
        "spot_position_pct": round((spot - low) / span * 100.0, 1),
        "spot_vs_high_pct": _pct_dist(spot, high),
        "spot_vs_low_pct": _pct_dist(spot, low),
        "sources": sources,
    }


# ---------------------------------------------------------------------- boards
def analyze_index(spec: IndexSpec, bars: list, interval: str = "5m",
                  chain: OptionChain | None = None, chain_error: str | None = None,
                  ladder_strikes: int | None = None) -> dict:
    """Full analysis block for one index."""
    prev = previous_session(bars)
    levels = level_analysis(bars, prev)
    chain_section = _chain_section(spec, chain, chain_error, ladder_strikes)
    levels["options"] = chain_section
    trend = trend_analysis(bars)
    reversal = reversal_analysis(bars, levels, chain, trend=trend)
    range_block = expected_range(bars, levels, (chain_section.get("straddle")
                                                if chain_section["available"] else None))
    session = session_bars(bars)
    spot = session[-1].close
    prior_close = prev[-1].close if prev else session[0].open
    return {
        "index": spec.key,
        "name": spec.name,
        "symbol": spec.symbol,
        "interval": interval,
        "as_of": bars[-1].dt.isoformat(timespec="seconds"),
        "strike_step": spec.strike_step,
        "lot_size": spec.lot_size,
        "spot": _round(spot),
        "change": _round(spot - prior_close),
        "change_pct": round((spot - prior_close) / prior_close * 100.0, 3) if prior_close else None,
        "trend": trend,
        "levels": levels,
        "range": range_block,
        "reversal": reversal,
        "options": chain_section,
    }


def _chain_section(spec: IndexSpec, chain: OptionChain | None, chain_error: str | None = None,
                   ladder_strikes: int | None = None) -> dict:
    if chain is None:
        return {"available": False, "reason": chain_error or "option chain not requested"}
    try:
        section = {"available": True, **analyze_chain(chain, lot_size=spec.lot_size)}
        if ladder_strikes:
            section["ladder"] = chain.to_dict(count=ladder_strikes, lot_size=spec.lot_size)
        return section
    except Exception as exc:  # analysis is best-effort, never breaks the board
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def fetch_index_bars(provider, spec: IndexSpec, interval: str, bars: int) -> list:
    """Bars for one index, raising on failure."""
    series = provider.fetch(spec.symbol, interval=interval, bars=bars)
    if not series:
        raise ValueError(f"no bars returned for {spec.symbol}")
    return series


def build_index_board(specs=None, provider=None, chain_provider=None, interval: str = "5m",
                      bars: int = 200, expiry: str | None = None,
                      ladder_strikes: int | None = None) -> dict:
    """Fetch data for every index and return the complete board payload.

    One failing index must never sink the whole board, so fetch and analysis
    errors are captured per index and reported in ``indices[].error``.
    """
    specs = specs or list(INDEX_SPECS.values())
    now = now_ist()
    entries = []
    for spec in specs:
        entry = {
            "index": spec.key, "name": spec.name, "symbol": spec.symbol,
            "strike_step": spec.strike_step, "lot_size": spec.lot_size,
            "interval": interval,
        }
        if provider is None:
            entry["error"] = "no bar provider configured"
            entries.append(entry)
            continue
        try:
            series = fetch_index_bars(provider, spec, interval, bars)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entries.append(entry)
            continue
        chain = None
        chain_error = None
        if chain_provider is not None:
            try:
                chain = chain_provider.fetch(spec.key, expiry=expiry)
            except Exception as exc:
                chain = None
                chain_error = str(exc)
        entry.update(analyze_index(spec, series, interval, chain=chain,
                                   chain_error=chain_error, ladder_strikes=ladder_strikes))
        entries.append(entry)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "market_open": is_market_open(now),
        "session_day": now.strftime("%A %d %b %Y"),
        "interval": interval,
        "bars": bars,
        "chain_source": getattr(chain_provider, "name", None),
        "indices": entries,
    }


def synthetic_chain_provider(bar_provider, interval: str = "5m", bars: int = 200, **kwargs):
    """Synthetic chains pinned to the spot the board prints for each index."""
    cache: dict[str, float] = {}

    def spot(index_key: str) -> float:
        spec = INDEX_SPECS.get(index_key.upper())
        if spec is None:
            raise ValueError(f"unknown index {index_key}")
        if index_key not in cache:
            series = bar_provider.fetch(spec.symbol, interval=interval, bars=bars)
            if not series:
                raise ValueError(f"no bars returned for {spec.symbol}")
            cache[index_key] = series[-1].close
        return cache[index_key]

    return SyntheticOptionChainProvider(spot_provider=spot, **kwargs)
