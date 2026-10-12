"""Simulated (paper) broker for NSE equities, including Indian charge model.

No real orders are ever placed: every fill is applied to an in-memory INR
cash/position ledger and persisted between cycles by the agent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

BROKERAGE_FLAT = 20.0            # INR per order (discount broker style)
BROKERAGE_PCT = 0.0003           # 0.03% of turnover, whichever is lower
STT_PCT_SELL = 0.001             # 0.10% Securities Transaction Tax on sell (delivery)
EXCHANGE_TXN_PCT = 0.0000297     # NSE exchange transaction charge
SEBI_PCT = 0.000001              # SEBI turnover fee (INR 10 per crore)
STAMP_DUTY_PCT_BUY = 0.00015     # stamp duty on buy side
GST_PCT = 0.18                   # GST on brokerage + exchange + SEBI charges


@dataclass(frozen=True)
class ChargeModel:
    brokerage_flat: float = BROKERAGE_FLAT
    brokerage_pct: float = BROKERAGE_PCT
    stt_pct_sell: float = STT_PCT_SELL
    exchange_txn_pct: float = EXCHANGE_TXN_PCT
    sebi_pct: float = SEBI_PCT
    stamp_duty_pct_buy: float = STAMP_DUTY_PCT_BUY
    gst_pct: float = GST_PCT


@dataclass(frozen=True)
class Charges:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    total: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_charges(model: ChargeModel, side: str, value: float) -> Charges:
    """NSE equity delivery charges for one order of ``value`` INR."""
    side = side.upper()
    value = max(float(value), 0.0)
    if model.brokerage_flat > 0 and model.brokerage_pct > 0:
        brokerage = min(model.brokerage_flat, model.brokerage_pct * value)
    elif model.brokerage_flat > 0:
        brokerage = model.brokerage_flat
    else:
        brokerage = model.brokerage_pct * value
    exchange = model.exchange_txn_pct * value
    sebi = model.sebi_pct * value
    stt = model.stt_pct_sell * value if side == "SELL" else 0.0
    stamp = model.stamp_duty_pct_buy * value if side == "BUY" else 0.0
    gst = model.gst_pct * (brokerage + exchange + sebi)
    total = brokerage + stt + exchange + sebi + stamp + gst
    return Charges(
        brokerage=round(brokerage, 4),
        stt=round(stt, 4),
        exchange=round(exchange, 4),
        sebi=round(sebi, 4),
        stamp=round(stamp, 4),
        gst=round(gst, 4),
        total=round(total, 2),
    )


@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(
            symbol=data["symbol"],
            qty=int(data.get("qty") or 0),
            avg_price=float(data.get("avg_price") or 0.0),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            opened_at=data.get("opened_at"),
        )


@dataclass
class Fill:
    id: int
    ts: int
    symbol: str
    side: str
    qty: int
    price: float
    value: float
    charges: float
    reason: str
    realized_pnl: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrderResult:
    ok: bool
    fill: Fill | None
    reason: str


class PaperBroker:
    """Cash-and-position ledger for a single paper portfolio."""

    def __init__(
        self,
        cash: float = 500000.0,
        charge_model: ChargeModel = None,
        slippage_bps: float = 5.0,
        allow_short: bool = False,
    ):
        self.cash = float(cash)
        self.charge_model = charge_model or ChargeModel()
        self.slippage_bps = float(slippage_bps)
        self.allow_short = allow_short
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.total_charges = 0.0
        self.total_slippage = 0.0
        self.fills: list[Fill] = []

    # ------------------------------------------------------------------ queries
    def positions_value(self, last_prices: dict) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            price = float(last_prices.get(symbol) or position.avg_price)
            total += position.qty * price
        return total

    def equity(self, last_prices: dict) -> float:
        return self.cash + self.positions_value(last_prices)

    def unrealized_pnl(self, last_prices: dict) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            price = float(last_prices.get(symbol) or position.avg_price)
            total += position.qty * (price - position.avg_price)
        return total

    # ------------------------------------------------------------------ orders
    def apply_slippage(self, price: float, side: str) -> float:
        slip = self.slippage_bps / 10000.0
        return price * (1.0 + slip) if side.upper() == "BUY" else price * (1.0 - slip)

    def market_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        ref_price: float,
        reason: str = "",
        ts: int = None,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> OrderResult:
        """Fill a market order at ``ref_price`` minus/plus slippage, with charges."""
        side = side.upper()
        qty = int(qty)
        if qty <= 0:
            return OrderResult(False, None, "quantity must be a positive integer")
        if ref_price is None or ref_price <= 0:
            return OrderResult(False, None, "invalid reference price")

        price = self.apply_slippage(ref_price, side)
        value = price * qty
        charges = compute_charges(self.charge_model, side, value)
        self.total_slippage += abs(price - ref_price) * qty
        ts = ts or int(time.time())
        realized = 0.0

        if side == "BUY":
            cost = value + charges.total
            if cost > self.cash + 1e-6:
                return OrderResult(
                    False, None, f"insufficient cash: need INR {cost:,.2f}, have INR {self.cash:,.2f}"
                )
            self.cash -= cost
            position = self.positions.get(symbol)
            if position is None or position.qty == 0:
                position = Position(
                    symbol=symbol,
                    qty=0,
                    avg_price=0.0,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    opened_at=ts,
                )
                self.positions[symbol] = position
            position.avg_price = (position.qty * position.avg_price + cost) / (position.qty + qty)
            position.qty += qty
            if stop_loss is not None:
                position.stop_loss = float(stop_loss)
            if take_profit is not None:
                position.take_profit = float(take_profit)
        else:
            position = self.positions.get(symbol)
            held = position.qty if position else 0
            if held < qty and not self.allow_short:
                return OrderResult(
                    False, None, f"insufficient position in {symbol}: hold {held} shares, want {qty}"
                )
            proceeds = value - charges.total
            self.cash += proceeds
            if position is not None and position.qty > 0:
                realized = qty * (price - position.avg_price) - charges.total
                position.qty -= qty
                if position.qty <= 0:
                    if self.allow_short and held < qty:
                        position.qty = -int(qty - held)
                        position.avg_price = price
                    else:
                        self.positions.pop(symbol, None)
            self.realized_pnl += realized

        self.total_charges += charges.total
        fill = Fill(
            id=len(self.fills) + 1,
            ts=ts,
            symbol=symbol,
            side=side,
            qty=qty,
            price=round(price, 4),
            value=round(value, 2),
            charges=charges.total,
            reason=reason,
            realized_pnl=round(realized, 2),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.fills.append(fill)
        return OrderResult(True, fill, "filled")

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
            "realized_pnl": round(self.realized_pnl, 2),
            "total_charges": round(self.total_charges, 2),
            "total_slippage": round(self.total_slippage, 2),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        charge_model: ChargeModel = None,
        slippage_bps: float = 5.0,
        allow_short: bool = False,
        default_cash: float = 500000.0,
    ) -> "PaperBroker":
        broker = cls(
            cash=float(data.get("cash", default_cash)),
            charge_model=charge_model,
            slippage_bps=slippage_bps,
            allow_short=allow_short,
        )
        for symbol, position in (data.get("positions") or {}).items():
            broker.positions[symbol] = Position.from_dict(dict(position, symbol=symbol))
        broker.realized_pnl = float(data.get("realized_pnl") or 0.0)
        broker.total_charges = float(data.get("total_charges") or 0.0)
        broker.total_slippage = float(data.get("total_slippage") or 0.0)
        return broker
