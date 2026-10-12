"""Risk guardrails applied on top of LLM decisions.

The LLM chooses direction and conviction; this module decides the *maximum*
quantity that may actually be traded and can veto or halt trading entirely.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .broker import ChargeModel, compute_charges

_REFERENCE_VALUE = 100_000.0


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float = 0.15        # max % of equity in a single symbol
    max_positions: int = 5                # max concurrently open symbols
    risk_per_trade_pct: float = 0.01      # max % of equity risked per trade (stop distance)
    max_daily_loss_pct: float = 0.03      # halt trading for the day beyond this drawdown
    min_cash_pct: float = 0.10            # keep at least this fraction of equity in cash
    max_order_value: float = 200_000.0    # INR cap per order
    cooldown_minutes: int = 60            # no new entries in a symbol after a stop-out


@dataclass(frozen=True)
class RiskVerdict:
    allowed: bool
    quantity: int
    reason: str


class RiskManager:
    def __init__(self, config: RiskConfig, charge_model: ChargeModel = None, slippage_bps: float = 5.0):
        self.config = config
        self.charge_model = charge_model or ChargeModel()
        self.slippage_bps = float(slippage_bps)
        self.cooldowns: dict[str, int] = {}

    # ------------------------------------------------------------------ helpers
    def _charge_rate(self, side: str) -> float:
        return compute_charges(self.charge_model, side, _REFERENCE_VALUE).total / _REFERENCE_VALUE

    def _cash_cap(self, side: str, price: float, cash: float) -> int:
        rate = self._charge_rate(side)
        cost_per_share = price * (1.0 + rate)
        if cost_per_share <= 0:
            return 0
        return int(cash // cost_per_share)

    # ------------------------------------------------------------------ cooldown
    def note_stop_out(self, symbol: str, ts: int = None) -> None:
        self.cooldowns[symbol] = int(time.time() if ts is None else ts)

    def in_cooldown(self, symbol: str, now_ts: int = None) -> bool:
        until = self.cooldowns.get(symbol)
        if until is None:
            return False
        now_ts = now_ts or int(time.time())
        return (now_ts - until) < self.config.cooldown_minutes * 60

    # ------------------------------------------------------------------ halts
    def check_halt(self, day_start_equity: float, equity: float):
        """Return ``(halted, reason)`` based on the daily loss limit."""
        if not day_start_equity or day_start_equity <= 0:
            return False, ""
        drawdown = (day_start_equity - equity) / day_start_equity
        if drawdown >= self.config.max_daily_loss_pct:
            return True, (
                f"daily drawdown {drawdown:.2%} reached limit "
                f"{self.config.max_daily_loss_pct:.2%}; halted for the day"
            )
        return False, ""

    # ------------------------------------------------------------------ entries
    def approve_buy(
        self,
        symbol: str,
        suggested_qty: int,
        price: float,
        equity: float,
        cash: float,
        open_positions: int,
        stop_loss: float = None,
        halted: bool = False,
        entries_paused: bool = False,
        cooldown_active: bool = False,
    ) -> RiskVerdict:
        if halted:
            return RiskVerdict(False, 0, "trading halted for the day (daily loss limit)")
        if entries_paused:
            return RiskVerdict(False, 0, "new entries paused from the web UI")
        if cooldown_active:
            return RiskVerdict(False, 0, f"{symbol} in cooldown after stop-out")
        if price <= 0:
            return RiskVerdict(False, 0, "invalid price")
        if open_positions >= self.config.max_positions:
            return RiskVerdict(False, 0, f"max open positions ({self.config.max_positions}) reached")
        if suggested_qty <= 0:
            return RiskVerdict(False, 0, "LLM suggested no shares")

        usable_cash = max(0.0, cash - equity * self.config.min_cash_pct)
        equity_cap = int(equity * self.config.max_position_pct // price)
        value_cap = int(self.config.max_order_value // price)
        cash_cap = self._cash_cap("BUY", price, usable_cash)

        if stop_loss and 0 < stop_loss < price:
            risk_per_share = price - stop_loss
            risk_cap = int(equity * self.config.risk_per_trade_pct // risk_per_share)
        else:
            risk_cap = equity_cap

        quantity = min(int(suggested_qty), equity_cap, risk_cap, cash_cap, value_cap)
        if quantity <= 0:
            binding = min(
                (("max_position_pct", equity_cap), ("risk_per_trade", risk_cap),
                 ("min_cash/cash", cash_cap), ("max_order_value", value_cap)),
                key=lambda item: item[1],
            )
            return RiskVerdict(False, 0, f"quantity rounded to 0 by limit {binding[0]}")
        return RiskVerdict(True, quantity, "ok")

    # ------------------------------------------------------------------ exits
    def approve_sell(self, symbol: str, suggested_qty: int, position_qty: int) -> RiskVerdict:
        if position_qty <= 0:
            return RiskVerdict(False, 0, f"no open position in {symbol}")
        quantity = position_qty if suggested_qty <= 0 else min(int(suggested_qty), position_qty)
        if quantity <= 0:
            return RiskVerdict(False, 0, "sell quantity rounded to 0")
        return RiskVerdict(True, quantity, "ok")

    # ------------------------------------------------------------------ ref stops
    def reference_levels(self, price: float, atr: float, atr_mult_stop: float = 2.0, reward_risk: float = 2.0):
        """Fallback stop/target derived from ATR when the LLM gives none."""
        atr = atr or price * 0.02
        stop = max(price - atr_mult_stop * atr, price * 0.5)
        target = price + reward_risk * (price - stop)
        return round(stop, 2), round(target, 2)
