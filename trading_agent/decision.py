"""Trade decision schema and tolerant JSON parsing for LLM replies."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

ACTIONS = {"BUY", "SELL", "HOLD"}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class TradeDecision:
    action: str
    quantity: int
    confidence: float
    stop_loss: float | None
    take_profit: float | None
    reasoning: str
    raw: dict


def _as_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    return _FENCE_RE.sub("", stripped).strip()


def _find_json_object(text: str):
    """Return the first decodable JSON object embedded in ``text``."""
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    return None


def parse_decision_text(text: str):
    """Parse an LLM reply into a :class:`TradeDecision`.

    Returns ``(decision, error)``; ``decision`` is ``None`` when the reply could
    not be understood, in which case ``error`` explains why.
    """
    if not text or not text.strip():
        return None, "empty response"

    payload = None
    candidate = _strip_fences(text)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        payload = _find_json_object(text)
    if payload is None:
        return None, "no JSON object found in response"

    action = str(payload.get("action", "")).strip().upper()
    if action not in ACTIONS:
        return None, f"unknown action {payload.get('action')!r}"

    quantity_raw = payload.get("quantity", 0)
    try:
        quantity = int(float(quantity_raw))
    except (TypeError, ValueError):
        quantity = 0
    if quantity < 0:
        quantity = 0

    confidence = _as_float(payload.get("confidence"))
    if confidence is None:
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    stop_loss = _as_float(payload.get("stop_loss"))
    take_profit = _as_float(payload.get("take_profit"))
    reasoning = str(payload.get("reasoning") or "").strip()[:240]

    return (
        TradeDecision(
            action=action,
            quantity=quantity,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            raw=payload,
        ),
        "",
    )


def normalise_stops(decision: TradeDecision, price: float, ref_stop: float = None, ref_target: float = None):
    """Validate/repair the LLM's stop and target levels for the current price.

    Returns a possibly-changed :class:`TradeDecision`; invalid or missing levels
    fall back to the reference values computed from ATR by the agent.
    """
    stop = decision.stop_loss
    target = decision.take_profit
    if decision.action == "BUY":
        if stop is None or stop <= 0 or stop >= price:
            stop = ref_stop
        if target is None or target <= 0 or target <= price:
            target = ref_target
    else:
        if stop is not None and stop <= 0:
            stop = None
        if target is not None and target <= 0:
            target = None
    return TradeDecision(
        action=decision.action,
        quantity=decision.quantity,
        confidence=decision.confidence,
        stop_loss=stop,
        take_profit=target,
        reasoning=decision.reasoning,
        raw=decision.raw,
    )
