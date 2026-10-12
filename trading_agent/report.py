"""Human-readable rendering for status and trade listings."""
from __future__ import annotations


def _money(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def render_status(status: dict) -> str:
    lines = [
        "=== NSE paper trading agent ===",
        f"model        : {status['model']}",
        f"data source  : {status['data_source']}",
        f"session day  : {status['day']} ({'HALTED' if status['halted'] else 'active'})",
        f"cash         : INR {_money(status['cash'])}",
        f"invested     : INR {_money(status['invested'])}",
        f"equity       : INR {_money(status['equity'])}",
        f"realized P&L : INR {_money(status['realized_pnl'])}",
        f"unrealized   : INR {_money(status['unrealized_pnl'])}",
        f"charges paid : INR {_money(status['charges_paid'])} (slippage INR {_money(status['slippage_cost'])})",
        f"day P&L      : INR {_money(status['day_pnl'])}"
        + (f" ({status['day_pnl_pct']:+.2f}%)" if status["day_pnl_pct"] is not None else ""),
        f"cycles run   : {status['cycles']}",
        "",
    ]
    if status["positions"]:
        header = f"{'symbol':<14}{'qty':>7}{'avg':>10}{'last':>10}{'value':>13}{'unreal P&L':>12}  stop/target"
        lines.append(header)
        lines.append("-" * len(header))
        for pos in status["positions"]:
            stop = pos["stop_loss"] if pos["stop_loss"] else "-"
            target = pos["take_profit"] if pos["take_profit"] else "-"
            lines.append(
                f"{pos['symbol']:<14}{pos['qty']:>7}{pos['avg_price']:>10.2f}{pos['last_price']:>10.2f}"
                f"{pos['market_value']:>13,.0f}{pos['unrealized_pnl']:>12,.0f}  {stop}/{target}"
            )
    else:
        lines.append("no open positions")
    lines.append(f"\nstate: {status['state_dir']}/state.json, journal: {status['state_dir']}/journal.jsonl")
    return "\n".join(lines)


def render_trades(fills: list) -> str:
    if not fills:
        return "no fills recorded yet"
    header = f"{'time (IST)':<21}{'symbol':<14}{'side':<6}{'qty':>7}{'price':>10}{'value':>13}{'charges':>9}  reason"
    lines = [header, "-" * len(header)]
    for fill in fills:
        lines.append(
            f"{fill.get('ts_readable', fill.get('ts', ''))!s:<21}"
            f"{fill.get('symbol', ''):<14}{fill.get('side', ''):<6}{fill.get('qty', 0):>7}"
            f"{fill.get('price', 0):>10.2f}{fill.get('value', 0):>13,.0f}{fill.get('charges', 0):>9.2f}"
            f"  {fill.get('reason', '')}"
        )
    return "\n".join(lines)
