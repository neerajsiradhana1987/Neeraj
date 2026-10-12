"""LLM clients: an OpenAI-compatible chat client plus a deterministic mock.

The real client talks to any OpenAI-compatible ``/chat/completions`` endpoint
(OpenRouter, OpenAI, Groq, Together, a local vLLM/Ollama server, ...), so no
provider SDK is required.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .decision import TradeDecision, normalise_stops, parse_decision_text

SYSTEM_PROMPT = """You are "NSE Paper Trader", an autonomous equities trading agent for the \
Indian stock market (NSE), operating a simulated paper portfolio denominated in INR.

For every market snapshot you receive you decide exactly one action: BUY, SELL or HOLD.

Hard rules:
1. Reply with a single JSON object and nothing else: no markdown, no code fences, no prose.
2. Schema (every key required unless nullable):
   {"action": "BUY" | "SELL" | "HOLD",
    "quantity": <integer number of shares, 0 for HOLD>,
    "confidence": <number between 0 and 1>,
    "stop_loss": <number or null>,
    "take_profit": <number or null>,
    "reasoning": <string, max 240 characters>}
3. quantity is a number of shares, not rupees. A risk engine will shrink your size when it \
breaks risk limits, so request the size you actually believe in.
4. For BUY: stop_loss must be strictly below the current price, take_profit strictly above. \
For SELL both may be null.
5. Capital preservation comes first. Prefer HOLD when signals conflict, volatility is extreme, \
or you already hold a position with no clear edge. A good HOLD is a valid, valuable decision.
6. Use ONLY the indicators and prices present in the snapshot. Never assume news, earnings, \
results or events that were not provided.
7. reasoning must mention the 1-3 indicators that drove the decision, in at most two sentences."""

_USER_TEMPLATE = """Symbol: {symbol}  |  timeframe: {interval}  |  prices in INR
Market snapshot:
{snapshot}

Portfolio: cash INR {cash:,.2f} | equity INR {equity:,.2f} | open positions {positions} | \
day P&L {day_pnl}% | current position in {symbol}: {position}
Choose the single best action for the next period. Reply with JSON only."""


def build_user_prompt(snapshot: dict) -> str:
    portfolio = snapshot.get("portfolio") or {}
    position = "none"
    if portfolio.get("position_qty"):
        position = f"{portfolio['position_qty']} shares @ avg INR {portfolio['position_avg_price']}"
    day_pnl = portfolio.get("day_pnl_pct")
    return _USER_TEMPLATE.format(
        symbol=snapshot.get("symbol", ""),
        interval=snapshot.get("timeframe", ""),
        snapshot=json.dumps(
            {k: v for k, v in snapshot.items() if k != "portfolio"},
            indent=2,
            sort_keys=True,
        ),
        cash=portfolio.get("cash") or 0.0,
        equity=portfolio.get("equity") or 0.0,
        positions=portfolio.get("open_positions") or 0,
        day_pnl="n/a" if day_pnl is None else f"{day_pnl:.2f}",
        position=position,
    )


class LLMClient:
    """OpenAI-compatible chat completions client (stdlib urllib only)."""

    def __init__(self, config):
        self.config = config

    @property
    def available(self) -> bool:
        return bool(getattr(self.config, "api_key", ""))

    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST", headers=headers)
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def decide(self, snapshot: dict, price: float = None):
        """Return ``(decision, error, latency_seconds)``."""
        started = time.monotonic()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://localhost/trading-agent",
            "X-Title": "nse-paper-trading-agent",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(snapshot)},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        try:
            data = self._post(url, payload, headers)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if not content:
                raise RuntimeError(f"unexpected response shape: {str(data)[:200]}")
        except Exception as exc:  # network, HTTP, parse and shape errors
            return None, f"llm_error: {exc}", time.monotonic() - started

        decision, error = parse_decision_text(content)
        if decision is not None and price:
            decision = normalise_stops(
                decision, price, snapshot.get("_ref_stop"), snapshot.get("_ref_target")
            )
        return decision, error, time.monotonic() - started


class MockLLMClient:
    """Deterministic rule-based stand-in used for demos and tests.

    It reads only the pre-computed indicators in the snapshot: trend alignment
    (price vs SMA20 vs SMA50), MACD histogram and RSI. This keeps demo runs
    reproducible without any API key.
    """

    def __init__(self, target_fraction: float = 0.10):
        self.target_fraction = target_fraction

    @property
    def available(self) -> bool:
        return True

    def decide(self, snapshot: dict, price: float = None):
        price = price or snapshot.get("last_price") or 0.0
        macd = snapshot.get("macd") or {}
        rsi = snapshot.get("rsi_14")
        sma20 = snapshot.get("sma_20")
        sma50 = snapshot.get("sma_50")
        atr = snapshot.get("atr_14") or price * 0.02
        trend = snapshot.get("trend") or {}
        portfolio = snapshot.get("portfolio") or {}
        position_qty = int(portfolio.get("position_qty") or 0)
        equity = portfolio.get("equity") or 0.0

        score = 0
        if sma20 and sma50:
            score += 1 if sma20 > sma50 else -1
        if sma20:
            score += 1 if price > sma20 else -1
        histogram = macd.get("histogram")
        if histogram is not None:
            score += 1 if histogram > 0 else -1
        if rsi is not None:
            if rsi > 75:
                score -= 2
            elif 45 <= rsi <= 68:
                score += 1
            elif rsi < 35:
                score += 1

        stop = round(price - 1.5 * atr, 2)
        target = round(price + 3.0 * atr, 2)

        if position_qty > 0:
            if price < (sma20 or price) or (histogram is not None and histogram < 0) or (rsi is not None and rsi < 42):
                decision = TradeDecision(
                    action="SELL",
                    quantity=position_qty,
                    confidence=round(min(0.9, 0.55 + 0.08 * abs(score)), 3),
                    stop_loss=None,
                    take_profit=None,
                    reasoning="Mock exit: price/MACD/RSI turned against the open position.",
                    raw={"mock": True},
                )
            else:
                decision = TradeDecision(
                    action="HOLD",
                    quantity=0,
                    confidence=0.5,
                    stop_loss=None,
                    take_profit=None,
                    reasoning="Mock hold: trend and momentum still favourable.",
                    raw={"mock": True},
                )
        elif score >= 2 and (not sma20 or price > sma20) and equity > 0:
            qty = max(1, int(equity * self.target_fraction / price))
            decision = TradeDecision(
                action="BUY",
                quantity=qty,
                confidence=round(min(0.9, 0.5 + 0.08 * score), 3),
                stop_loss=stop,
                take_profit=target,
                reasoning=f"Mock entry: trend aligned, MACD histogram and RSI supportive (score {score}).",
                raw={"mock": True},
            )
        else:
            decision = TradeDecision(
                action="HOLD",
                quantity=0,
                confidence=round(min(0.9, 0.4 + 0.05 * score), 3),
                stop_loss=None,
                take_profit=None,
                reasoning=f"Mock hold: signal score {score} below entry threshold.",
                raw={"mock": True},
            )
        return decision, "", 0.0
