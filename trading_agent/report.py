"""Human-readable rendering for status, trades and the intraday indices board."""
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


# ------------------------------------------------------------- indices board
_TREND_TEXT = {
    "strong_up": "strong UP",
    "up": "UP",
    "sideways": "sideways / ranging",
    "down": "DOWN",
    "strong_down": "strong DOWN",
}
_TREND_ARROW = {
    "strong_up": "^^",
    "up": "^",
    "sideways": "->",
    "down": "v",
    "strong_down": "vv",
}
_RISK_LABEL = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


def _signed(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+,.2f}{suffix}"


def _levels_line(levels: list, blank: str = "no level nearby") -> str:
    if not levels:
        return blank
    return "  ".join(f"{item['price']:,.1f} ({item['label']})" for item in levels)


def _strike_rows(rows: list) -> str:
    if not rows:
        return "-"
    return ", ".join(
        f"{row['strike']:,.0f}"
        + (f" [{row['oi_change']:+,} OI]" if row.get("oi_change") else "")
        for row in rows
    )


def render_index_board(board: dict) -> str:
    """Render :func:`trading_agent.indices.build_index_board` output as text."""
    header = [
        "=== Intraday indices board ===",
        f"{board['session_day']} | {board['generated_at']} IST | "
        f"interval {board['interval']} | market {'OPEN' if board['market_open'] else 'closed'}",
        f"option chain: {board.get('chain_source') or 'off'}",
        "",
    ]
    lines = list(header)
    for entry in board.get("indices", []):
        lines.extend(_render_index_entry(entry))
        lines.append("")
    lines.append("read as: levels are intraday reference points, not trade signals. "
                 "max pain / OI walls describe option writer positioning into expiry.")
    return "\n".join(lines)


def _render_index_entry(entry: dict) -> list:
    if entry.get("error"):
        return [f"{entry.get('index', '?')}: data error - {entry['error']}"]

    name = f"{entry['name']} ({entry['index']})"
    spot = f"{entry['spot']:,.2f}" if entry.get("spot") is not None else "n/a"
    change = f"{_signed(entry['change'])} ({_signed(entry['change_pct'], '%')})"
    lines = [
        f"--- {name} {spot} {change} | {entry['interval']} bars, as of "
        f"{entry['as_of'][11:16]} IST ---",
    ]

    trend = entry["trend"]
    lines.append(
        f"  trend     {_TREND_ARROW.get(trend['label'], '?')} {_TREND_TEXT.get(trend['label'], '?')}"
        f" (score {trend['score']:+.2f}) | EMA stack {trend['ema_stack']} | "
        f"VWAP {trend['vwap']:,.2f} ({'above' if trend['above_vwap'] else 'below'}) | "
        f"RSI {trend['rsi_14']} | {trend['structure']}"
    )

    levels = entry["levels"]
    range_block = entry["range"]
    lines.append(
        f"  range     {range_block['low']:,.1f} - {range_block['high']:,.1f}"
        f" ({range_block['width_pct']:+.2f}% width, {range_block['basis']}) | "
        f"spot at {range_block['spot_position_pct']:.0f}% of the range"
        + (" | already outside the projected range" if range_block.get("spot_outside") else "")
    )
    lines.append(f"  support   {_levels_line(levels['support'][:3])}")
    lines.append(f"  resistance{_levels_line(levels['resistance'][:3], '')}")
    cpr = levels.get("cpr")
    if cpr:
        lines.append(f"  CPR       {cpr['low']:,.2f} - {cpr['high']:,.2f}"
                     f" (pivot {cpr['pivot']:,.2f}{', inverted' if cpr['inverted'] else ''})"
                     f" | price {'inside' if cpr['inside'] else 'outside'} the CPR")
    orb = levels.get("opening_range")
    if orb:
        lines.append(
            f"  open range{orb['low']:,.2f} - {orb['high']:,.2f} (first {orb['minutes']} min, "
            f"{orb['size_pct']:.2f}%) | break above {orb['target_up']:,.1f} / "
            f"below {orb['target_down']:,.1f}"
        )

    reversal = entry["reversal"]
    bias = f", bias {reversal['bias']}" if reversal["bias"] != "none" else ""
    lines.append(
        f"  reversal  {_RISK_LABEL.get(reversal['risk'], reversal['risk'])} "
        f"({reversal['score']}/100{bias})"
    )
    for warning in reversal["warnings"][:6]:
        lines.append(f"      [{warning['severity']:<5}] {warning['direction']:<8} {warning['message']}")

    lines.extend(_render_options(entry.get("options") or {}))
    return lines


def _render_options(options: dict) -> list:
    if not options.get("available"):
        return [f"  options   unavailable - {options.get('reason', 'no option chain')}"]
    pain = options.get("max_pain") or {}
    pcr = options.get("pcr") or {}
    pressure = options.get("buying_pressure") or {}
    lines = [
        f"  options   expiry {options.get('expiry')} | spot {options.get('spot'):,.2f} | "
        f"ATM {options.get('atm_strike'):,.0f} | max pain "
        f"{_pain_text(pain)}"
    ]
    lines.append(
        f"  PCR       OI {pcr.get('oi_pcr')} | volume {pcr.get('volume_pcr')} | {pcr.get('bias')}"
    )
    lines.append(
        f"  max buying side {pressure.get('max_buying_side')} "
        f"({pressure.get('max_buying_pct')}% of INR {_compact(pressure.get('put_turnover'))} PE / "
        f"{_compact(pressure.get('call_turnover'))} CE premium) - {pressure.get('flow')}"
    )
    walls = options.get("walls") or {}
    lines.append(
        f"  OI walls  PE support {_strike_rows(walls.get('put_support'))} | "
        f"CE resistance {_strike_rows(walls.get('call_resistance'))}"
    )
    buildup = options.get("buildup") or {}
    lines.append(
        f"  OI added  PE {_strike_rows(buildup.get('put_buildup'))} | "
        f"CE {_strike_rows(buildup.get('call_buildup'))}"
    )
    bulk = options.get("bulk_strikes") or {}
    lines.append(
        f"  bulk      PE {_strike_rows(bulk.get('PE'))} | CE {_strike_rows(bulk.get('CE'))}"
    )
    straddle = options.get("straddle")
    if straddle:
        lines.append(
            f"  straddle  {straddle['straddle_points']:,.2f} pts ({straddle['straddle_pct']}%) | "
            f"breakeven {straddle['breakeven_low']:,.2f} / {straddle['breakeven_high']:,.2f}"
        )
    return lines


def _pain_text(pain: dict) -> str:
    if not pain.get("available"):
        return pain.get("reason", "n/a")
    distance = pain.get("distance_pct")
    return (f"{pain['strike']:,.0f} "
            f"({distance:+.2f}% from spot, call share {pain.get('call_share_pct')}%)")


def _compact(value) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    for unit, scale in (("Cr", 1e7), ("L", 1e5), ("k", 1e3)):
        if value >= scale:
            return f"{value / scale:,.1f} {unit}"
    return f"{value:,.0f}"
