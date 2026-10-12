"""Command line interface for the NSE paper trading agent."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

from . import __version__
from .agent import TradingAgent
from .config import load_config, write_default_config, write_dotenv_example
from .data import YahooProvider
from .market_hours import now_ist
from .report import render_status, render_trades

BANNER = f"nse-paper-agent {__version__}"


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_init(args) -> int:
    try:
        path = write_default_config(args.config, force=args.force)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    env_path = write_dotenv_example(".env.example")
    print(f"wrote {path}")
    print(f"wrote {env_path}")
    print("next: copy .env.example to .env, set TRADING_LLM_API_KEY, then run:")
    print(f"  python3 -m trading_agent demo --config {args.config}")
    return 0


def _build_agent(args, force_mock: bool = False, source: str = None):
    config = load_config(args.config)
    if source:
        config.data.source = source
    if getattr(args, "interval", None):
        config.market.interval = args.interval
    if getattr(args, "cash", None) is not None:
        store_exists = os.path.exists(os.path.join(config.state_dir, "state.json"))
        if store_exists:
            print(f"note: --cash ignored, existing state found in {config.state_dir}")
        else:
            config.broker.starting_cash = float(args.cash)
    agent = TradingAgent(config, force_mock=force_mock)
    return agent


def cmd_run(args) -> int:
    agent = _build_agent(args, force_mock=args.mock_llm)
    symbols = [s for s in (args.symbols or "").replace(",", " ").split()] or None
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{BANNER} | symbols: {', '.join(agent.symbols(symbols))} | interval {agent.config.market.interval}")
    print(f"model: {'mock (no key)' if agent.using_mock else agent.config.llm.model} | state: {agent.config.state_dir}")
    if args.once:
        report = agent.run_cycle(force=args.force, symbols=symbols)
        print(report.summary())
        return 0
    agent.run_loop(
        poll_seconds=args.poll,
        force=args.force,
        symbols=symbols,
        max_cycles=args.max_cycles,
    )
    print(f"finished after starting {started}")
    return 0


def cmd_demo(args) -> int:
    agent = _build_agent(args, force_mock=True, source="synthetic")
    symbols = [s for s in (args.symbols or "").replace(",", " ").split()] or None
    print(f"{BANNER} demo | synthetic data + deterministic mock strategy (no API key needed)")
    print(f"symbols: {', '.join(agent.symbols(symbols))} | starting cash INR {agent.config.broker.starting_cash:,.0f}")
    for i in range(max(1, args.cycles)):
        report = agent.run_cycle(force=True, symbols=symbols)
        print(f"cycle {i + 1}: {report.summary()}")
        time.sleep(0.05)
    status = agent.status()
    if args.json:
        _print_json(status)
    else:
        print()
        print(render_status(status))
    return 0


def cmd_status(args) -> int:
    agent = _build_agent(args)
    status = agent.status()
    if args.json:
        _print_json(status)
    else:
        print(render_status(status))
    return 0


def cmd_trades(args) -> int:
    agent = _build_agent(args)
    if args.json:
        _print_json(agent.trades(limit=args.limit))
    else:
        print(render_trades(agent.trades(limit=args.limit)))
    return 0


def cmd_reset(args) -> int:
    if not args.yes:
        print("refusing to delete state without --yes", file=sys.stderr)
        return 1
    agent = _build_agent(args)
    agent.reset()
    print(f"state cleared in {agent.config.state_dir}")
    return 0


def cmd_doctor(args) -> int:
    checks = []
    try:
        config = load_config(args.config)
        checks.append(("config load", True, args.config))
    except Exception as exc:
        print(f"FAIL config load: {exc}")
        return 1

    provider = YahooProvider()
    symbol = config.market.symbols[0] if config.market.symbols else "RELIANCE.NS"
    try:
        bars = provider.fetch(symbol, interval=config.market.interval, bars=5)
        ok = bool(bars)
        checks.append((f"yahoo data ({symbol})", ok, f"{len(bars)} bars" if ok else "empty"))
    except Exception as exc:
        checks.append((f"yahoo data ({symbol})", False, str(exc)))

    if config.llm.enabled and config.llm.api_key:
        checks.append(("llm api key", True, f"{config.llm.model} via {config.llm.base_url}"))
    else:
        checks.append(("llm api key", False, "not set: agent will use the mock strategy"))

    state_dir = config.state_dir
    try:
        os.makedirs(state_dir, exist_ok=True)
        probe = os.path.join(state_dir, ".probe")
        with open(probe, "w") as handle:
            handle.write("ok")
        os.remove(probe)
        checks.append(("state dir writable", True, state_dir))
    except Exception as exc:
        checks.append(("state dir writable", False, str(exc)))

    from .market_hours import is_market_open

    market_open = "yes" if is_market_open(now_ist(), set(config.market.holidays)) else "no"

    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name:<{width}}  {detail}")
        failed += 0 if ok else 1
    print(f"market session (IST {now_ist().strftime('%H:%M')}): {market_open}")
    print(f"interval {config.market.interval}, {len(config.market.symbols)} symbols, "
          f"starting cash INR {config.broker.starting_cash:,.0f}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-agent",
        description="Paper trading agent for NSE equities with LLM-driven decisions.",
    )
    parser.add_argument("--version", action="version", version=BANNER)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sub_parser):
        sub_parser.add_argument("--config", default="config.json", help="path to config.json")
        sub_parser.add_argument("--json", action="store_true", help="machine-readable output")

    p_init = sub.add_parser("init", help="write a default config.json and .env.example")
    p_init.add_argument("--config", default="config.json")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="run the agent (single cycle with --once)")
    add_common(p_run)
    p_run.add_argument("--once", action="store_true", help="run exactly one cycle and exit")
    p_run.add_argument("--force", action="store_true", help="trade even outside market hours")
    p_run.add_argument("--poll", type=int, default=None, help="seconds between cycles (loop mode)")
    p_run.add_argument("--max-cycles", type=int, default=None, help="stop after N cycles")
    p_run.add_argument("--symbols", default=None, help="space/comma separated tickers, e.g. RELIANCE TCS")
    p_run.add_argument("--interval", default=None, choices=["1m", "5m", "15m", "1h", "1d"])
    p_run.add_argument("--cash", type=float, default=None, help="starting INR cash (fresh state only)")
    p_run.add_argument("--mock-llm", action="store_true", help="use the deterministic mock strategy")
    p_run.set_defaults(func=cmd_run)

    p_demo = sub.add_parser("demo", help="offline demo on synthetic data (no API key, no network)")
    add_common(p_demo)
    p_demo.add_argument("--cycles", type=int, default=5, help="number of cycles to simulate")
    p_demo.add_argument("--symbols", default=None)
    p_demo.add_argument("--interval", default=None, choices=["5m", "15m", "1h", "1d"])
    p_demo.add_argument("--cash", type=float, default=None)
    p_demo.set_defaults(func=cmd_demo)

    p_status = sub.add_parser("status", help="show portfolio state")
    add_common(p_status)
    p_status.set_defaults(func=cmd_status)

    p_trades = sub.add_parser("trades", help="show recent fills")
    add_common(p_trades)
    p_trades.add_argument("--limit", type=int, default=20)
    p_trades.set_defaults(func=cmd_trades)

    p_reset = sub.add_parser("reset", help="delete local state (paper portfolio)")
    add_common(p_reset)
    p_reset.add_argument("--yes", action="store_true", help="confirm deletion")
    p_reset.set_defaults(func=cmd_reset)

    p_doctor = sub.add_parser("doctor", help="check config, data and LLM connectivity")
    p_doctor.add_argument("--config", default="config.json")
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
