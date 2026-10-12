"""The trading agent: one cycle = fetch data, manage exits, ask the LLM, trade."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from . import signals
from .broker import ChargeModel, PaperBroker
from .config import AgentConfig
from .data import CSVProvider, SyntheticProvider, YahooProvider
from .llm import LLMClient, MockLLMClient
from .market_hours import is_data_fresh, is_market_open, next_session_open, now_ist
from .risk import RiskManager
from .storage import StateStore


@dataclass
class CycleReport:
    ts: int
    market_open: bool
    halted: bool
    equity: float
    cash: float
    positions: int
    fills: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    forced: bool = False

    def summary(self) -> str:
        stamp = datetime.fromtimestamp(self.ts).astimezone(now_ist().tzinfo).strftime("%H:%M:%S")
        parts = [f"[{stamp} IST]"]
        if not self.market_open:
            parts.append("outside session (forced)" if self.forced else "market closed")
        if self.halted:
            parts.append("HALTED")
        parts.append(f"equity INR {self.equity:,.0f}")
        parts.append(f"cash INR {self.cash:,.0f}")
        parts.append(f"positions {self.positions}")
        if self.fills:
            parts.append(
                "fills: "
                + ", ".join(f"{f.side} {f.qty} {f.symbol}@{f.price:.2f}({f.reason})" for f in self.fills)
            )
        if self.rejected:
            parts.append("rejected: " + ", ".join(self.rejected))
        if self.skipped:
            parts.append("skipped: " + ", ".join(self.skipped))
        parts.extend(f"note: {note}" for note in self.notes)
        return " | ".join(parts)


def build_provider(config: AgentConfig):
    source = (config.data.source or "yahoo").lower()
    if source == "synthetic":
        return SyntheticProvider(seed=config.data.seed, bars=config.market.bars)
    if source == "csv":
        return CSVProvider(config.data.csv_directory)
    return YahooProvider()


def build_llm(config: AgentConfig, force_mock: bool = False):
    if force_mock:
        return MockLLMClient(), True
    if config.llm.enabled and config.llm.api_key:
        return LLMClient(config.llm), False
    return MockLLMClient(), True


class TradingAgent:
    def __init__(self, config: AgentConfig, provider=None, llm=None, store=None, clock=None, force_mock: bool = False):
        self.config = config
        self.store = store or StateStore(config.state_dir)
        self.charge_model = ChargeModel(
            brokerage_flat=config.broker.brokerage_flat,
            brokerage_pct=config.broker.brokerage_pct,
            stt_pct_sell=config.broker.stt_pct_sell,
            exchange_txn_pct=config.broker.exchange_txn_pct,
            sebi_pct=config.broker.sebi_pct,
            stamp_duty_pct_buy=config.broker.stamp_duty_pct_buy,
            gst_pct=config.broker.gst_pct,
        )
        self.provider = provider or build_provider(config)
        if llm is None:
            llm, using_mock = build_llm(config, force_mock=force_mock)
        else:
            using_mock = isinstance(llm, MockLLMClient)
        self.llm = llm
        self.using_mock = using_mock
        self.risk = RiskManager(config.risk, self.charge_model, config.broker.slippage_bps)
        self.clock = clock or now_ist
        self.holidays = set(config.market.holidays or [])
        self._warned_no_key = False

    # ------------------------------------------------------------------ helpers
    def _broker_from_state(self, state: dict) -> PaperBroker:
        return PaperBroker.from_dict(
            state.get("broker") or {},
            charge_model=self.charge_model,
            slippage_bps=self.config.broker.slippage_bps,
            allow_short=self.config.broker.allow_short,
            default_cash=self.config.broker.starting_cash,
        )

    def symbols(self, override=None) -> list:
        symbols = override or self.config.market.symbols
        out = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            if not symbol:
                continue
            if "." not in symbol and "^" not in symbol:
                symbol += ".NS"
            out.append(symbol)
        return out

    def _note_mock(self) -> str:
        if self.using_mock and not self._warned_no_key:
            self._warned_no_key = True
            return "no LLM API key configured; using deterministic mock strategy"
        return ""

    # -------------------------------------------------------------------- cycle
    def run_cycle(self, force: bool = False, symbols=None) -> CycleReport:
        now = self.clock()
        state = self.store.load_state()
        broker = self._broker_from_state(state)
        self.risk.cooldowns = dict(state.get("cooldowns") or {})
        last_prices = dict(state.get("last_prices") or {})
        symbols = self.symbols(symbols)
        interval = self.config.market.interval

        report = CycleReport(
            ts=int(now.timestamp()),
            market_open=True,
            halted=False,
            equity=broker.equity(last_prices),
            cash=broker.cash,
            positions=len(broker.positions),
        )
        note = self._note_mock()
        if note:
            report.notes.append(note)

        equity = broker.equity(last_prices)
        today = now.date().isoformat()
        if state.get("current_day") != today:
            state["current_day"] = today
            state["day_start_equity"] = equity
            state["halted"] = False
            self.store.journal("DAY_START", day=today, equity=round(equity, 2))

        day_start_equity = state.get("day_start_equity") or equity
        halted, reason = self.risk.check_halt(day_start_equity, equity)
        if halted and not state.get("halted"):
            state["halted"] = True
            self.store.journal("HALT", reason=reason, equity=round(equity, 2))

        if state.get("halted"):
            report.halted = True
            report.notes.append("halted: daily loss limit reached")
            state["broker"] = broker.to_dict()
            state["last_prices"] = last_prices
            state["cooldowns"] = self.risk.cooldowns
            self.store.save_state(state)
            return report

        market_open = is_market_open(now, self.holidays)
        report.market_open = market_open
        report.forced = bool(force and not market_open)
        if not market_open and not force:
            upcoming = next_session_open(now, self.holidays)
            state["broker"] = broker.to_dict()
            state["last_prices"] = last_prices
            state["cooldowns"] = self.risk.cooldowns
            self.store.save_state(state)
            self.store.journal("MARKET_CLOSED", now=now.isoformat(), next_open=upcoming.isoformat())
            report.notes.append(f"market closed; next session opens {upcoming.strftime('%Y-%m-%d %H:%M IST')}")
            return report

        for symbol in symbols:
            try:
                bars = self.provider.fetch(symbol, interval=interval, bars=self.config.market.bars)
            except Exception as exc:
                report.skipped.append(f"{symbol}(data error)")
                self.store.journal("SKIP", symbol=symbol, reason=f"data_error: {exc}")
                continue
            if not bars:
                report.skipped.append(f"{symbol}(no data)")
                self.store.journal("SKIP", symbol=symbol, reason="no_data")
                continue

            bar = bars[-1]
            price = bar.close
            last_prices[symbol] = price
            fresh, age, detail = is_data_fresh(
                bar.ts,
                interval,
                now,
                multiplier=self.config.market.staleness_multiplier,
                holidays=self.holidays,
            )
            if not fresh:
                report.skipped.append(f"{symbol}(stale)")
                self.store.journal("SKIP", symbol=symbol, reason="stale_data", detail=detail)
                continue

            equity = broker.equity(last_prices)

            # 1) enforce stop-loss / take-profit on open positions
            position = broker.positions.get(symbol)
            if position and position.qty > 0:
                exit_reason = None
                if position.stop_loss and bar.low <= position.stop_loss:
                    exit_reason = "STOP_LOSS"
                elif position.take_profit and bar.high >= position.take_profit:
                    exit_reason = "TAKE_PROFIT"
                if exit_reason:
                    result = broker.market_order(
                        symbol, "SELL", position.qty, price,
                        reason=exit_reason, ts=int(now.timestamp()),
                    )
                    if result.ok:
                        if exit_reason == "STOP_LOSS":
                            self.risk.note_stop_out(symbol, int(now.timestamp()))
                        report.fills.append(result.fill)
                        self.store.journal("FILL", **result.fill.to_dict())
                    else:
                        self.store.journal("RISK_REJECT", symbol=symbol, action="SELL",
                                           reason=result.reason, trigger=exit_reason)

            # 2) ask the LLM for a fresh decision
            equity = broker.equity(last_prices)
            portfolio = signals.portfolio_context(
                broker, equity, state.get("day_start_equity"), symbol, last_prices
            )
            snapshot = signals.build_snapshot(symbol, bars, interval, portfolio)
            ref_stop, ref_target = self.risk.reference_levels(
                price,
                snapshot.get("atr_14"),
                self.config.risk.atr_stop_multiplier,
                self.config.risk.reward_risk_ratio,
            )
            snapshot["_ref_stop"] = ref_stop
            snapshot["_ref_target"] = ref_target

            decision, error, latency = self.llm.decide(snapshot, price)
            if decision is None:
                self.store.journal("DECISION", symbol=symbol, action="ERROR",
                                   error=error, latency_seconds=round(latency, 3))
                report.rejected.append(f"{symbol}(LLM error)")
                continue

            self.store.journal(
                "DECISION", symbol=symbol, action=decision.action,
                quantity=decision.quantity, confidence=decision.confidence,
                stop_loss=decision.stop_loss, take_profit=decision.take_profit,
                price=round(price, 2), latency_seconds=round(latency, 3),
                reasoning=decision.reasoning, mock=bool(decision.raw.get("mock")),
            )

            if decision.action == "BUY":
                verdict = self.risk.approve_buy(
                    symbol, decision.quantity, price, equity, broker.cash,
                    open_positions=len(broker.positions), stop_loss=decision.stop_loss,
                    halted=state.get("halted", False),
                    cooldown_active=self.risk.in_cooldown(symbol, int(now.timestamp())),
                )
                if verdict.allowed and verdict.quantity > 0:
                    result = broker.market_order(
                        symbol, "BUY", verdict.quantity, price,
                        reason="LLM_BUY", ts=int(now.timestamp()),
                        stop_loss=decision.stop_loss or ref_stop,
                        take_profit=decision.take_profit or ref_target,
                    )
                    if result.ok:
                        report.fills.append(result.fill)
                        self.store.journal("FILL", **result.fill.to_dict())
                    else:
                        report.rejected.append(f"{symbol}({result.reason})")
                        self.store.journal("RISK_REJECT", symbol=symbol, action="BUY", reason=result.reason)
                else:
                    report.rejected.append(f"{symbol}({verdict.reason})")
                    self.store.journal("RISK_REJECT", symbol=symbol, action="BUY",
                                      reason=verdict.reason, suggested=decision.quantity)

            elif decision.action == "SELL":
                held = broker.positions.get(symbol)
                held_qty = held.qty if held else 0
                verdict = self.risk.approve_sell(symbol, decision.quantity, held_qty)
                if verdict.allowed and verdict.quantity > 0:
                    result = broker.market_order(
                        symbol, "SELL", verdict.quantity, price,
                        reason="LLM_SELL", ts=int(now.timestamp()),
                    )
                    if result.ok:
                        report.fills.append(result.fill)
                        self.store.journal("FILL", **result.fill.to_dict())
                    else:
                        report.rejected.append(f"{symbol}({result.reason})")
                        self.store.journal("RISK_REJECT", symbol=symbol, action="SELL", reason=result.reason)
                else:
                    self.store.journal("RISK_REJECT", symbol=symbol, action="SELL",
                                      reason=verdict.reason, suggested=decision.quantity)

        equity = broker.equity(last_prices)
        report.equity = equity
        report.cash = broker.cash
        report.positions = len(broker.positions)
        state["broker"] = broker.to_dict()
        state["last_prices"] = last_prices
        state["cooldowns"] = self.risk.cooldowns
        state["cycles"] = int(state.get("cycles") or 0) + 1
        self.store.save_state(state)
        self.store.journal(
            "CYCLE", equity=round(equity, 2), cash=round(broker.cash, 2),
            positions=len(broker.positions), fills=len(report.fills),
            rejected=len(report.rejected), skipped=len(report.skipped),
        )
        return report

    # -------------------------------------------------------------------- loop
    def run_loop(self, poll_seconds: int = None, force: bool = False, symbols=None, max_cycles: int = None):
        poll = poll_seconds or self.config.poll_seconds
        cycles = 0
        try:
            while True:
                report = self.run_cycle(force=force, symbols=symbols)
                print(report.summary(), flush=True)
                cycles += 1
                if max_cycles and cycles >= max_cycles:
                    break
                time.sleep(poll)
        except KeyboardInterrupt:
            print("\nstopped by user; state saved", flush=True)

    # ------------------------------------------------------------------ reports
    def status(self) -> dict:
        state = self.store.load_state()
        broker = self._broker_from_state(state)
        last_prices = dict(state.get("last_prices") or {})
        equity = broker.equity(last_prices)
        day_start = state.get("day_start_equity")
        day_pnl = equity - day_start if day_start else 0.0
        day_pnl_pct = (day_pnl / day_start * 100.0) if day_start else None
        positions = []
        for symbol, position in sorted(broker.positions.items()):
            price = float(last_prices.get(symbol) or position.avg_price)
            positions.append({
                "symbol": symbol,
                "qty": position.qty,
                "avg_price": round(position.avg_price, 2),
                "last_price": round(price, 2),
                "market_value": round(position.qty * price, 2),
                "unrealized_pnl": round(position.qty * (price - position.avg_price), 2),
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
            })
        return {
            "state_dir": str(self.store.directory),
            "model": "mock (deterministic rules)" if self.using_mock else self.config.llm.model,
            "data_source": self.config.data.source,
            "day": state.get("current_day"),
            "halted": bool(state.get("halted")),
            "cash": round(broker.cash, 2),
            "invested": round(broker.positions_value(last_prices), 2),
            "equity": round(equity, 2),
            "realized_pnl": round(broker.realized_pnl, 2),
            "unrealized_pnl": round(broker.unrealized_pnl(last_prices), 2),
            "charges_paid": round(broker.total_charges, 2),
            "slippage_cost": round(broker.total_slippage, 2),
            "day_start_equity": round(day_start, 2) if day_start else None,
            "day_pnl": round(day_pnl, 2),
            "day_pnl_pct": round(day_pnl_pct, 3) if day_pnl_pct is not None else None,
            "cycles": int(state.get("cycles") or 0),
            "positions": positions,
        }

    def trades(self, limit: int = 20) -> list:
        fills = self.store.read_events(event="FILL", limit=limit)
        for fill in fills:
            stamp = fill.get("ts")
            if stamp:
                stamp = datetime.fromtimestamp(int(stamp)).astimezone(now_ist().tzinfo)
                fill["ts_readable"] = stamp.strftime("%Y-%m-%d %H:%M:%S")
        return fills

    def reset(self) -> None:
        self.store.reset()
