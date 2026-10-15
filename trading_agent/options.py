"""Option-chain providers and the derived analysis used by the indices board.

Two providers ship with the project:

- ``NSEOptionChainProvider``       : the live NSE indices option-chain API
  (``https://www.nseindia.com/api/option-chain-indices``) for NIFTY, BANKNIFTY,
  FINNIFTY and MIDCPNIFTY.
- ``SyntheticOptionChainProvider`` : deterministic strikes and open interest,
  used by ``demo``, by tests and as an offline fallback.

SENSEX options trade on BSE, which exposes no public option-chain API, so a live
chain is only available for the NSE indices. Both providers return the same
``OptionChain`` structure and every analysis helper below is a pure function of
that structure, so the numbers never depend on where the chain came from.
"""
from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta

from .data import IST

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# Index names the NSE option-chain API accepts.
NSE_INDEX_ALIASES = {
    "NIFTY": "NIFTY",
    "NIFTY50": "NIFTY",
    "NIFTY 50": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "NIFTYBANK": "BANKNIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "FINNIFTY": "FINNIFTY",
    "NIFTYFIN": "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
}

# Indices whose options do not trade on the NSE chain API.
NON_NSE_INDICES = {"SENSEX", "BANKEX", "SX40"}

LEG_SIDES = ("CE", "PE")


class OptionChainError(RuntimeError):
    """Raised when a chain cannot be fetched (network, blocking, bad payload)."""


@dataclass(frozen=True)
class OptionLeg:
    """One call or put contract on a strike."""

    strike: float
    side: str                     # "CE" or "PE"
    oi: int = 0                   # open interest (contracts)
    oi_change: int = 0            # change in open interest
    volume: int = 0               # contracts traded today
    ltp: float = 0.0              # last traded premium
    iv: float | None = None       # implied volatility, percent
    bid: float | None = None
    ask: float | None = None

    @property
    def turnover(self) -> float:
        """Premium actually paid on the strike today (volume x last price)."""
        return float(self.volume) * float(self.ltp or 0.0)

    def to_dict(self) -> dict:
        return {
            "strike": self.strike,
            "side": self.side,
            "oi": self.oi,
            "oi_change": self.oi_change,
            "volume": self.volume,
            "ltp": self.ltp,
            "iv": self.iv,
            "bid": self.bid,
            "ask": self.ask,
            "turnover": round(self.turnover, 2),
        }


@dataclass(frozen=True)
class OptionChain:
    """A full strike ladder for one index and one expiry."""

    index: str
    spot: float
    expiry: str
    expiries: tuple = ()
    legs: tuple = ()
    as_of: str = ""
    source: str = ""

    def side(self, side: str) -> list:
        side = side.upper()
        return [leg for leg in self.legs if leg.side == side]

    @property
    def strikes(self) -> list:
        return sorted({leg.strike for leg in self.legs})

    @property
    def step(self) -> float | None:
        unique = self.strikes
        if len(unique) < 2:
            return None
        diffs = [round(b - a, 2) for a, b in zip(unique, unique[1:]) if b > a]
        if not diffs:
            return None
        from statistics import median

        return float(median(diffs))

    def atm_strike(self) -> float | None:
        if not self.spot or not self.strikes:
            return None
        return min(self.strikes, key=lambda strike: abs(strike - self.spot))

    def legs_near_spot(self, count: int = 25) -> list:
        """The ``count`` strikes closest to spot, as a symmetric ladder."""
        strikes = self.strikes
        if not strikes:
            return []
        atm = self.atm_strike()
        centre = strikes.index(atm)
        half = max(1, count // 2)
        window = strikes[max(0, centre - half):centre + half + 1]
        wanted = set(window)
        return [leg for leg in self.legs if leg.strike in wanted]

    def to_dict(self, count: int = 25, lot_size: int | None = None) -> dict:
        ladder = self.legs_near_spot(count)
        rows = []
        by_strike: dict[float, dict[str, dict]] = {}
        for leg in ladder:
            row = by_strike.setdefault(leg.strike, {"strike": leg.strike})
            row[leg.side] = leg.to_dict()
        for strike in sorted(by_strike):
            row = by_strike[strike]
            row["atm"] = abs(strike - (self.spot or 0)) < 1e-9
            rows.append(row)
        return {
            "index": self.index,
            "spot": self.spot,
            "expiry": self.expiry,
            "expiries": list(self.expiries),
            "as_of": self.as_of,
            "source": self.source,
            "lot_size": lot_size,
            "ladder": rows,
        }


@dataclass
class OptionChainProvider:
    """Interface: every provider exposes ``fetch(index, expiry=None)``."""

    name: str = "base"

    def fetch(self, index: str, expiry: str | None = None) -> OptionChain:
        raise NotImplementedError


def _new_cookie_jar():
    import http.cookiejar

    return http.cookiejar.CookieJar()


def _as_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _leg_from_nse(block: dict, strike: float, side: str) -> OptionLeg:
    def number(key: str) -> float:
        return _as_float(block.get(key)) or 0.0

    iv = _as_float(block.get("impliedVolatility"))
    return OptionLeg(
        strike=strike,
        side=side,
        oi=int(number("openInterest")),
        oi_change=int(number("changeinOpenInterest")),
        volume=int(number("totalTradedVolume")),
        ltp=number("lastPrice"),
        iv=None if iv is None else iv,
        bid=_as_float(block.get("bidprice")),
        ask=_as_float(block.get("askPrice")),
    )


def _parse_expiry(value: str) -> datetime | None:
    for pattern in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(value).strip(), pattern)
        except ValueError:
            continue
    return None


def chain_from_nse_payload(index: str, payload: dict, expiry: str | None = None) -> OptionChain:
    """Turn the raw NSE option-chain JSON into an :class:`OptionChain`."""
    records = (payload or {}).get("records") or {}
    rows = records.get("data") or []
    expiries = tuple(str(value) for value in (records.get("expiryDates") or []))
    chosen = expiry
    if chosen is None and expiries:
        today = datetime.now(IST).date()
        dated = [value for value in expiries if _parse_expiry(value)]
        upcoming = sorted(
            (value for value in dated if _parse_expiry(value).date() >= today),
            key=lambda value: _parse_expiry(value),
        )
        chosen = upcoming[0] if upcoming else expiries[0]
    chosen = str(chosen) if chosen else ""

    legs = []
    for row in rows:
        if chosen and str(row.get("expiryDate")) != chosen:
            continue
        strike = _as_float(row.get("strikePrice"))
        if not strike:
            continue
        for side in LEG_SIDES:
            block = row.get(side)
            if not block:
                continue
            legs.append(_leg_from_nse(block, strike, side))
    spot = _as_float(records.get("underlyingValue")) or 0.0
    return OptionChain(
        index=index.upper(),
        spot=spot,
        expiry=chosen,
        expiries=expiries,
        legs=tuple(legs),
        as_of=str(records.get("timestamp") or datetime.now(IST).isoformat(timespec="seconds")),
        source="nse",
    )


class NSEOptionChainProvider(OptionChainProvider):
    """Live option chain from the NSE indices API.

    NSE only serves the API to browser-like clients, so a cookie jar is seeded
    from the home page first. Requests come from an Indian IP range for the API
    to answer at all; datacentre IPs are commonly rejected with HTTP 403.
    """

    name = "nse"
    HOME = "https://www.nseindia.com"
    CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices"

    def __init__(self, timeout: int = 20, retries: int = 3, backoff: float = 1.5):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._cookiejar = urllib.request.HTTPCookieProcessor(_new_cookie_jar())
        self._opener = urllib.request.build_opener(self._cookiejar)

    def _get(self, url: str) -> dict:
        last_error = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(url, headers=_HEADERS)
                with self._opener.open(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 403:
                    raise OptionChainError(
                        "NSE refused the option-chain API (HTTP 403). The exchange "
                        "blocks non-browser and non-Indian clients; retry from an "
                        "Indian network or use --option-source synthetic."
                    ) from exc
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(self.backoff * (attempt + 1))
                    continue
                raise OptionChainError(f"NSE HTTP {exc.code} for {url}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(self.backoff * (attempt + 1))
        raise OptionChainError(f"NSE option-chain request failed: {last_error}")

    def fetch(self, index: str, expiry: str | None = None) -> OptionChain:
        key = (index or "").upper().strip()
        if key in NON_NSE_INDICES:
            raise OptionChainError(
                f"{key} options trade on BSE, which has no public option-chain API; "
                "live chains are available for NIFTY/BANKNIFTY only "
                "(--option-source synthetic for an offline estimate)."
            )
        symbol = NSE_INDEX_ALIASES.get(key.replace(" ", ""))
        if not symbol:
            raise OptionChainError(f"unknown index for the NSE option chain: {index}")
        try:
            self._opener.open(urllib.request.Request(self.HOME, headers=_HEADERS),
                              timeout=self.timeout).close()
        except Exception as exc:  # cookies are optional, the API may still answer
            raise OptionChainError(f"NSE home page unreachable: {exc}") from exc
        url = f"{self.CHAIN_URL}?{urllib.parse.urlencode({'symbol': symbol})}"
        payload = self._get(url)
        chain = chain_from_nse_payload(symbol, payload, expiry=expiry)
        if not chain.legs:
            raise OptionChainError(f"NSE returned an empty option chain for {symbol}")
        return chain


class SyntheticOptionChainProvider(OptionChainProvider):
    """Deterministic option chain for demos, tests and offline runs.

    ``spot`` (or ``spot_provider``() ) pins the ladder to the same price the
    board fetched for the index, so max pain and the OI walls stay consistent
    with the printed range.
    """

    name = "synthetic"

    def __init__(self, seed: int = 1234, spot: float | None = None,
                 spot_provider=None, strikes: int = 12, strike_step: float | None = None):
        self.seed = seed
        self.spot = spot
        self.spot_provider = spot_provider
        self.strikes = strikes
        self.strike_step = strike_step

    def _rng(self, *parts) -> random.Random:
        return random.Random(":".join([str(self.seed), *[str(part) for part in parts]]))

    def _spot(self, index: str) -> float:
        if self.spot:
            return float(self.spot)
        if self.spot_provider is not None:
            return float(self.spot_provider(index))
        return float(20000.0 + self._rng(index, "spot").random() * 500.0)

    def _expiry(self) -> str:
        today = datetime.now(IST).date()
        days_ahead = (3 - today.weekday()) % 7 or 7
        return (today + timedelta(days=days_ahead)).isoformat()

    def fetch(self, index: str, expiry: str | None = None) -> OptionChain:
        index = (index or "").upper()
        spot = self._spot(index)
        step = float(self.strike_step or _guess_step(index, spot))
        atm = round(spot / step) * step
        spread = max(self.strikes, 4)
        expiry_value = expiry or self._expiry()
        legs = []
        for offset in range(-spread, spread + 1):
            strike = atm + offset * step
            distance = strike - spot
            for side in LEG_SIDES:
                rng = self._rng(index, expiry_value, strike, side)
                put_side = side == "PE"
                peak = -3.0 * step if put_side else 1.5 * step
                base = 2_400_000 if put_side else 1_900_000
                sigma = 5.5 * step
                oi = base * math.exp(-0.5 * ((distance - peak) / sigma) ** 2)
                oi *= rng.uniform(0.82, 1.18)
                oi = int(round(oi / 100.0) * 100.0)
                # fresh positions: a concentrated build-up on each side
                build_peak = -1.5 * step if put_side else 3.0 * step
                change = oi * 0.35 * math.exp(-0.5 * ((distance - build_peak) / (3.0 * step)) ** 2)
                change *= rng.uniform(0.5, 1.4)
                if rng.random() < 0.15:
                    change = -change
                volume = int(oi * rng.uniform(0.05, 0.5))
                iv = rng.uniform(9.0, 22.0)
                intrinsic = (spot - strike) if put_side else (strike - spot)
                intrinsic = max(0.0, intrinsic)
                tte = max(1.0 / 365.0, 3.0 / 365.0)
                time_value = spot * (iv / 100.0) * math.sqrt(tte) * math.exp(
                    -0.5 * ((distance / (0.85 * (iv / 100.0) * math.sqrt(tte) * spot)) ** 2)
                )
                ltp = round((intrinsic + time_value) * 20.0) / 20.0
                legs.append(OptionLeg(
                    strike=strike,
                    side=side,
                    oi=oi,
                    oi_change=int(round(change / 100.0) * 100.0),
                    volume=volume,
                    ltp=ltp,
                    iv=round(iv, 2),
                    bid=round(ltp * 0.995, 2),
                    ask=round(ltp * 1.005, 2),
                ))
        return OptionChain(
            index=index,
            spot=spot,
            expiry=expiry_value,
            expiries=(expiry_value,),
            legs=tuple(legs),
            as_of=datetime.now(IST).isoformat(timespec="seconds"),
            source="synthetic",
        )


def _guess_step(index: str, spot: float) -> float:
    guesses = {
        "NIFTY": 50.0,
        "BANKNIFTY": 100.0,
        "SENSEX": 100.0,
        "FINNIFTY": 50.0,
        "MIDCPNIFTY": 25.0,
    }
    step = guesses.get(index.upper())
    if step:
        return step
    return 50.0 if spot < 10000 else 100.0


def build_chain_provider(source: str, **kwargs) -> OptionChainProvider:
    source = (source or "nse").lower()
    if source in ("off", "none", "disabled"):
        return None
    if source == "synthetic":
        return SyntheticOptionChainProvider(**kwargs)
    return NSEOptionChainProvider(
        timeout=int(kwargs.pop("timeout", 20)),
        retries=int(kwargs.pop("retries", 2)),
    )


# --------------------------------------------------------------------- analysis
def pain_at(chain: OptionChain, price: float) -> dict:
    """Value left for option buyers if the index settles at ``price``.

    The strike that minimises this total is the *max pain* point: it is where
    buyers, taken as a whole, lose the most and writers keep the most.
    """
    calls = chain.side("CE")
    puts = chain.side("PE")
    call_loss = sum(leg.oi * max(0.0, price - leg.strike) for leg in calls)
    put_loss = sum(leg.oi * max(0.0, leg.strike - price) for leg in puts)
    return {
        "settle": round(float(price), 2),
        "call_loss": round(call_loss, 2),
        "put_loss": round(put_loss, 2),
        "total_loss": round(call_loss + put_loss, 2),
    }


def max_pain(chain: OptionChain, window_pct: float = 0.12) -> dict:
    """Max-pain strike plus the nearest few points of the pain profile."""
    strikes = [strike for strike in chain.strikes
               if any(leg.oi > 0 for leg in chain.legs if leg.strike == strike)]
    spot = chain.spot or 0.0
    if not strikes:
        return {"available": False, "reason": "no open interest in the chain"}
    low, high = spot * (1.0 - window_pct), spot * (1.0 + window_pct)
    candidates = [strike for strike in strikes if low <= strike <= high] or strikes
    profile = [(strike, pain_at(chain, strike)) for strike in candidates]
    strike, pain = min(profile, key=lambda item: item[1]["total_loss"])
    near = sorted(profile, key=lambda item: abs(item[0] - spot))[:5]
    distance_pct = ((strike - spot) / spot * 100.0) if spot else None
    return {
        "available": True,
        "strike": strike,
        "distance_pct": round(distance_pct, 3) if distance_pct is not None else None,
        "buyer_loss_at_strike": pain["total_loss"],
        "call_share_pct": round(pain["call_loss"] / pain["total_loss"] * 100.0, 1)
        if pain["total_loss"] else None,
        "put_share_pct": round(pain["put_loss"] / pain["total_loss"] * 100.0, 1)
        if pain["total_loss"] else None,
        "profile": [
            {"strike": item[0], "total_loss": item[1]["total_loss"],
             "call_loss": item[1]["call_loss"], "put_loss": item[1]["put_loss"]}
            for item in near
        ],
    }


def concentration(chain: OptionChain, side: str, key: str, top: int = 3,
                  lot_size: int | None = None) -> list:
    """Top strikes by ``key`` (``oi``, ``oi_change`` or ``turnover``)."""
    legs = [leg for leg in chain.side(side) if getattr(leg, key, 0)]
    legs.sort(key=lambda leg: getattr(leg, key), reverse=True)
    out = []
    for leg in legs[:top]:
        out.append({
            "strike": leg.strike,
            "side": leg.side,
            "oi": leg.oi,
            "oi_change": leg.oi_change,
            "volume": leg.volume,
            "turnover": round(leg.turnover, 2),
            "ltp": leg.ltp,
            "iv": leg.iv,
            "oi_lots": round(leg.oi / lot_size) if lot_size else None,
            "distance_pct": round((leg.strike - chain.spot) / chain.spot * 100.0, 3)
            if chain.spot else None,
        })
    return out


def oi_walls(chain: OptionChain, top: int = 3, lot_size: int | None = None) -> dict:
    """Highest open interest on each side: writer support and writer resistance."""
    return {
        "put_support": concentration(chain, "PE", "oi", top, lot_size),
        "call_resistance": concentration(chain, "CE", "oi", top, lot_size),
    }


def oi_buildup(chain: OptionChain, top: int = 3, lot_size: int | None = None) -> dict:
    """Where fresh positions (open-interest change) are being written today."""
    return {
        "put_buildup": concentration(chain, "PE", "oi_change", top, lot_size),
        "call_buildup": concentration(chain, "CE", "oi_change", top, lot_size),
    }


def pcr(chain: OptionChain) -> dict:
    """Put/call ratio on open interest and on traded volume."""
    puts = chain.side("PE")
    calls = chain.side("CE")
    put_oi = sum(leg.oi for leg in puts)
    call_oi = sum(leg.oi for leg in calls)
    put_volume = sum(leg.volume for leg in puts)
    call_volume = sum(leg.volume for leg in calls)
    oi_pcr = (put_oi / call_oi) if call_oi else None
    volume_pcr = (put_volume / call_volume) if call_volume else None
    return {
        "put_oi": put_oi,
        "call_oi": call_oi,
        "oi_pcr": round(oi_pcr, 3) if oi_pcr is not None else None,
        "put_volume": put_volume,
        "call_volume": call_volume,
        "volume_pcr": round(volume_pcr, 3) if volume_pcr is not None else None,
        "bias": "put-heavy (supportive)" if (oi_pcr or 0) >= 1.1
        else ("call-heavy (resistive)" if oi_pcr is not None and oi_pcr <= 0.9 else "balanced"),
    }


def buying_pressure(chain: OptionChain) -> dict:
    """Which side is actually paying premium today (volume x last price)."""
    puts = chain.side("PE")
    calls = chain.side("CE")
    put_turnover = sum(leg.turnover for leg in puts)
    call_turnover = sum(leg.turnover for leg in calls)
    put_oi_change = sum(leg.oi_change for leg in puts)
    call_oi_change = sum(leg.oi_change for leg in calls)
    total = put_turnover + call_turnover
    side = "PE" if put_turnover > call_turnover else "CE"
    return {
        "put_turnover": round(put_turnover, 2),
        "call_turnover": round(call_turnover, 2),
        "turnover_ratio": round(put_turnover / call_turnover, 3) if call_turnover else None,
        "max_buying_side": side,
        "max_buying_pct": round(max(put_turnover, call_turnover) / total * 100.0, 1) if total else None,
        "put_oi_change": put_oi_change,
        "call_oi_change": call_oi_change,
        "oi_change_side": "PE" if put_oi_change >= call_oi_change else "CE",
        "flow": _flow_label(put_turnover, call_turnover, put_oi_change, call_oi_change),
    }


def _flow_label(put_turnover, call_turnover, put_oi_change, call_oi_change) -> str:
    buying = "PE" if put_turnover >= call_turnover else "CE"
    writing = "PE" if put_oi_change >= call_oi_change else "CE"
    if buying == writing:
        return f"buildup on the {buying} side (buying and writing the same strikes)"
    return f"heavy {buying} buying against {writing} writing"


def bulk_strikes(chain: OptionChain, top: int = 3, lot_size: int | None = None) -> dict:
    """Strikes where the most money changed hands, per side."""
    return {
        "PE": concentration(chain, "PE", "turnover", top, lot_size),
        "CE": concentration(chain, "CE", "turnover", top, lot_size),
    }


def straddle(chain: OptionChain) -> dict | None:
    """ATM straddle premium and the implied end-of-day price range."""
    atm = chain.atm_strike()
    if atm is None:
        return None
    call = next((leg for leg in chain.side("CE") if leg.strike == atm), None)
    put = next((leg for leg in chain.side("PE") if leg.strike == atm), None)
    if not call or not put or not call.ltp or not put.ltp:
        return None
    premium = call.ltp + put.ltp
    return {
        "atm_strike": atm,
        "call_ltp": call.ltp,
        "put_ltp": put.ltp,
        "straddle_points": round(premium, 2),
        "straddle_pct": round(premium / chain.spot * 100.0, 3) if chain.spot else None,
        "breakeven_low": round(atm - premium, 2),
        "breakeven_high": round(atm + premium, 2),
    }


def analyze_chain(chain: OptionChain, lot_size: int | None = None,
                  top: int = 3) -> dict:
    """Every option-chain number the indices board reports."""
    return {
        "index": chain.index,
        "source": chain.source,
        "expiry": chain.expiry,
        "expiries": list(chain.expiries),
        "as_of": chain.as_of,
        "spot": chain.spot,
        "atm_strike": chain.atm_strike(),
        "max_pain": max_pain(chain),
        "walls": oi_walls(chain, top, lot_size),
        "buildup": oi_buildup(chain, top, lot_size),
        "bulk_strikes": bulk_strikes(chain, top, lot_size),
        "pcr": pcr(chain),
        "buying_pressure": buying_pressure(chain),
        "straddle": straddle(chain),
        "lot_size": lot_size,
        "legs": len(chain.legs),
    }
