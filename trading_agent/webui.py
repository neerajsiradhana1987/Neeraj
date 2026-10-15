"""Mobile-first web dashboard and JSON API for the trading agent.

Uses only the standard library (``http.server``), so the project keeps its
zero-dependency promise. The dashboard is served as a single self-contained
HTML page and talks to the agent through a small JSON API.

Run it with::

    python3 -m trading_agent web --host 0.0.0.0 --port 8000

Security: the page is intended for a private network. Pass ``--token`` (or set
``TRADING_WEB_TOKEN``) to require ``Authorization: Bearer <token>`` on every
API call; without a token the server binds to 127.0.0.1 only.
"""
from __future__ import annotations

import json
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .agent import TradingAgent, build_provider
from .config import AgentConfig, load_config
from .data import INTERVAL_SECONDS
from .indices import build_index_board, resolve_indices, synthetic_chain_provider
from .market_hours import is_market_open, next_session_open, now_ist
from .options import build_chain_provider
from .storage import StateStore

INTERVALS = ["1m", "5m", "15m", "1h", "1d"]


class WebController:
    """Owns the agent instance, the background loop and the shared state."""

    def __init__(self, config: AgentConfig, agent: TradingAgent, config_path: str = "config.json",
                 poll_seconds: int = None):
        self.config = config
        self.agent = agent
        self.config_path = Path(config_path)
        self.store = agent.store
        self.poll_seconds = int(poll_seconds or config.poll_seconds)
        self.force = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self.last_report = None
        self.last_run_ts = None
        self.last_error = None
        self._bar_provider = None
        self._chain_provider = None
        self._indices_cache = None

    # ----------------------------------------------------------------- indices
    def _index_providers(self):
        """Lazily build (and remember) the providers used by the indices board."""
        if self._bar_provider is None:
            data_config = self.config.data
            data_config.source = self.config.indices.data_source or data_config.source
            self._bar_provider = build_provider(self.config)
        if self._chain_provider is None:
            self._chain_provider = self._build_chain_provider()
        return self._bar_provider, self._chain_provider

    def _build_chain_provider(self):
        source = (self.config.indices.option_source or "nse").lower()
        if source in ("off", "none", "disabled"):
            return None
        if source == "synthetic":
            return synthetic_chain_provider(
                self._bar_provider, self.config.indices.interval, self.config.indices.bars)
        return build_chain_provider(source)

    def indices(self) -> dict:
        """Indices board, refreshed at most once every ``indices.refresh_seconds``."""
        interval = self.config.indices.interval
        bars = self.config.indices.bars
        now = time.time()
        cached = self._indices_cache
        if cached and now - cached[0] < self.config.indices.refresh_seconds:
            return cached[1]
        with self._lock:
            cached = self._indices_cache
            if cached and now - cached[0] < self.config.indices.refresh_seconds:
                return cached[1]
            bar_provider, chain_provider = self._index_providers()
            board = build_index_board(
                specs=resolve_indices(",".join(self.config.indices.names)),
                provider=bar_provider,
                chain_provider=chain_provider,
                interval=interval,
                bars=bars,
                ladder_strikes=self.config.indices.ladder_strikes,
            )
            self._indices_cache = (now, board)
            return board

    # ------------------------------------------------------------------ running
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_cycle_now(self.force)
            except Exception as exc:  # keep the loop alive, surface in the UI
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._stop_event.wait(self.poll_seconds)

    def start_loop(self, poll_seconds: int = None, force: bool = None) -> dict:
        with self._lock:
            if self.running:
                return {"ok": False, "reason": "loop already running"}
            if poll_seconds:
                self.poll_seconds = max(15, int(poll_seconds))
            if force is not None:
                self.force = bool(force)
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="agent-loop", daemon=True)
            self._thread.start()
            self.store.journal(
                "WEB_ACTION", action="loop_start",
                poll_seconds=self.poll_seconds, force=self.force,
            )
            return {"ok": True, "poll_seconds": self.poll_seconds, "force": self.force}

    def stop_loop(self) -> dict:
        with self._lock:
            if not self.running:
                return {"ok": False, "reason": "loop not running"}
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=5)
            self._thread = None
            self.store.journal("WEB_ACTION", action="loop_stop")
            return {"ok": True}

    def run_cycle_now(self, force: bool = None) -> dict:
        with self._lock:
            if force is not None:
                self.force = bool(force)
            try:
                report = self.agent.run_cycle(force=self.force)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise
            self.last_report = report
            self.last_run_ts = int(time.time())
            self.last_error = None
            return {
                "ok": True,
                "summary": report.summary(),
                "fills": [fill.to_dict() for fill in report.fills],
                "rejected": report.rejected,
                "skipped": report.skipped,
                "notes": report.notes,
                "equity": round(report.equity, 2),
            }

    # ------------------------------------------------------------------- state
    def _entries_paused(self) -> bool:
        return bool(self.store.load_state().get("entries_paused"))

    def set_entries_paused(self, paused: bool) -> dict:
        state = self.store.load_state()
        state["entries_paused"] = bool(paused)
        self.store.save_state(state)
        self.store.journal("WEB_ACTION", action="pause" if paused else "resume",
                           entries_paused=bool(paused))
        return {"ok": True, "entries_paused": bool(paused)}

    def reset_portfolio(self) -> dict:
        with self._lock:
            self.agent.reset()
            self.store.journal("WEB_ACTION", action="reset")
            self.last_report = None
            self.last_run_ts = None
            return {"ok": True}

    def close_position(self, symbol: str) -> dict:
        """Manual exit for one symbol — still routed through the risk layer."""
        symbol = self.agent.symbols([symbol])[0]
        with self._lock:
            state = self.store.load_state()
            broker = self.agent._broker_from_state(state)
            position = broker.positions.get(symbol)
            if not position or position.qty <= 0:
                return {"ok": False, "reason": f"no open position in {symbol}"}
            price = float((state.get("last_prices") or {}).get(symbol) or position.avg_price)
            verdict = self.agent.risk.approve_sell(symbol, position.qty, position.qty)
            if not verdict.allowed:
                self.store.journal("RISK_REJECT", symbol=symbol, action="SELL",
                                  reason=verdict.reason, trigger="MANUAL_CLOSE")
                return {"ok": False, "reason": verdict.reason}
            result = broker.market_order(
                symbol, "SELL", verdict.quantity, price, reason="MANUAL_CLOSE",
                ts=int(time.time()),
            )
            if not result.ok:
                self.store.journal("RISK_REJECT", symbol=symbol, action="SELL",
                                  reason=result.reason, trigger="MANUAL_CLOSE")
                return {"ok": False, "reason": result.reason}
            state["broker"] = broker.to_dict()
            self.store.save_state(state)
            self.store.journal("FILL", **result.fill.to_dict())
            return {"ok": True, "fill": result.fill.to_dict()}

    # ------------------------------------------------------------------ config
    def _read_config_file(self) -> dict:
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_config_file(self, raw: dict) -> None:
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.config_path)

    def set_symbols(self, action: str, symbol: str) -> dict:
        action = (action or "").lower()
        if action not in ("add", "remove"):
            return {"ok": False, "reason": "action must be add or remove"}
        normalised = self.agent.symbols([symbol])[0]
        raw = self._read_config_file()
        market = dict(raw.get("market") or {})
        symbols = [s.upper() for s in (market.get("symbols") or [])]
        if action == "add":
            if normalised in symbols:
                return {"ok": False, "reason": f"{normalised} already in the watchlist"}
            symbols.append(normalised)
        else:
            if normalised not in symbols:
                return {"ok": False, "reason": f"{normalised} not in the watchlist"}
            symbols = [s for s in symbols if s != normalised]
        market["symbols"] = symbols
        raw["market"] = market
        self._write_config_file(raw)
        self.config.market.symbols = symbols
        self.store.journal("WEB_ACTION", action=f"symbol_{action}", symbol=normalised)
        return {"ok": True, "symbols": symbols}

    def set_interval(self, interval: str) -> dict:
        interval = (interval or "").lower().strip()
        if interval not in INTERVAL_SECONDS:
            return {"ok": False, "reason": f"interval must be one of {sorted(INTERVAL_SECONDS)}"}
        raw = self._read_config_file()
        market = dict(raw.get("market") or {})
        market["interval"] = interval
        raw["market"] = market
        self._write_config_file(raw)
        self.config.market.interval = interval
        self.store.journal("WEB_ACTION", action="interval", interval=interval)
        return {"ok": True, "interval": interval}

    def set_force(self, force: bool) -> dict:
        self.force = bool(force)
        return {"ok": True, "force": self.force}

    # ----------------------------------------------------------------- reports
    def status(self) -> dict:
        base = self.agent.status()
        now = now_ist()
        base.update({
            "running": self.running,
            "entries_paused": self._entries_paused(),
            "force": self.force,
            "poll_seconds": self.poll_seconds,
            "last_run": self.last_run_ts,
            "last_summary": self.last_report.summary() if self.last_report else None,
            "last_error": self.last_error,
            "market_open": is_market_open(now, self.agent.holidays),
            "market_time": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "next_open": next_session_open(now, self.agent.holidays).strftime("%Y-%m-%d %H:%M"),
            "interval": self.config.market.interval,
            "watchlist": self.agent.symbols(),
            "api_version": 1,
        })
        return base

    def journal(self, limit: int = 20) -> list:
        from datetime import datetime

        events = self.store.read_events(limit=limit)
        for event in events:
            ts = event.get("ts")
            if ts:
                event["ts_readable"] = datetime.fromtimestamp(
                    int(ts), now_ist().tzinfo
                ).strftime("%H:%M:%S")
        return events

    def trades(self, limit: int = 20) -> list:
        return self.agent.trades(limit=limit)


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0f14">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>NSE Paper Agent</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230b0f14'/%3E%3Cpath d='M5 22 L12 13 L18 18 L27 7' stroke='%233fb950' stroke-width='3' fill='none' stroke-linecap='round'/%3E%3C/svg%3E">
<style>
:root{--bg:#0b0f14;--card:#151b23;--card2:#1b232d;--line:#263041;--text:#e6edf3;
--muted:#8b98a9;--accent:#2f81f7;--green:#3fb950;--red:#f85149;--amber:#d29922}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
padding-bottom:32px}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:10px;
padding:calc(10px + env(safe-area-inset-top)) 14px 10px;background:rgba(11,15,20,.92);
backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.2px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted);flex:none}
.dot.run{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot.pause{background:var(--amber)}
.dot.halt{background:var(--red)}
.clock{margin-left:auto;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
main{padding:12px;max-width:760px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
@media(min-width:560px){.cards{grid-template-columns:repeat(4,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
.card .v{font-size:18px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}
.pos{background:var(--red)}.neg{color:var(--green)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
margin-top:12px;padding:12px}
.panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin:0}
.panel-head{display:flex;align-items:baseline;gap:8px;margin-bottom:10px}
.muted{color:var(--muted);font-size:12px;margin-left:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{color:var(--muted);font-size:11px;font-weight:500;text-align:right;padding:4px 6px;
border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:8px 6px;border-bottom:1px solid rgba(38,48,65,.5);text-align:right;font-size:14px}
tr:last-child td{border-bottom:none}
.sym{font-weight:600}
.sub{color:var(--muted);font-size:11px}
button{font:inherit;font-size:14px;font-weight:600;color:var(--text);background:var(--card2);
border:1px solid var(--line);border-radius:10px;padding:11px 14px;min-height:44px;cursor:pointer}
button:active{transform:scale(.98)}
button.primary{background:var(--accent);border-color:var(--accent)}
button.danger{color:var(--red);border-color:#5a2b2b}
button.sm{padding:7px 10px;min-height:34px;font-size:12px;font-weight:500}
.controls{display:flex;flex-wrap:wrap;gap:8px}
.row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
select,input[type=text],input[type=password]{font:inherit;font-size:14px;color:var(--text);
background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:10px 12px;min-height:44px}
input[type=text],input[type=password]{flex:1;min-width:130px}
.chk{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px;
padding:0 4px;min-height:44px}
.chk input{width:18px;height:18px;accent-color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{display:flex;align-items:center;gap:6px;background:var(--card2);border:1px solid var(--line);
border-radius:999px;padding:6px 8px 6px 12px;font-size:13px}
.chip button{padding:0 6px;min-height:22px;border:none;background:transparent;color:var(--muted);
font-size:15px;line-height:1}
.feed{list-style:none;margin:0;padding:0;max-height:300px;overflow:auto}
.feed li{display:flex;gap:8px;padding:7px 2px;border-bottom:1px solid rgba(38,48,65,.5);font-size:13px}
.feed li:last-child{border-bottom:none}
.feed .t{color:var(--muted);font-size:11px;flex:none;padding-top:2px;font-variant-numeric:tabular-nums}
.tag{font-size:10px;font-weight:700;letter-spacing:.4px;padding:2px 6px;border-radius:5px;flex:none;
height:fit-content;margin-top:1px}
.tag.FILL{background:rgba(63,185,80,.16);color:var(--green)}
.tag.RISK_REJECT{background:rgba(210,153,34,.16);color:var(--amber)}
.tag.SKIP,.tag.MARKET_CLOSED{background:rgba(139,152,169,.14);color:var(--muted)}
.tag.DECISION{background:rgba(47,129,247,.16);color:var(--accent)}
.tag.HALT{background:rgba(248,81,73,.16);color:var(--red)}
.tag.CYCLE,.tag.DAY_START,.tag.WEB_ACTION{background:rgba(139,152,169,.12);color:var(--muted)}
.note{color:var(--muted);font-size:12px;margin-top:8px}
.banner{background:rgba(210,153,34,.12);border:1px solid rgba(210,153,34,.35);color:#e8c36a;
border-radius:10px;padding:9px 11px;font-size:13px;margin-bottom:12px}
.hidden{display:none!important}
.tag.INFO{background:rgba(63,185,80,.16);color:var(--green)}
.tag.WATCH{background:rgba(210,153,34,.16);color:var(--amber)}
.tag.ALERT{background:rgba(248,81,73,.16);color:var(--red)}
.idx{background:var(--card2);border:1px solid var(--line);border-radius:12px;
  padding:12px;margin-bottom:10px}
.idx:last-child{margin-bottom:0}
.idx .head{display:flex;align-items:baseline;gap:8px}
.idx .name{font-weight:650}
.idx .lv{margin-left:auto;color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.4px}
.idx .lv.low{color:var(--green)}.idx .lv.medium{color:var(--amber)}.idx .lv.high{color:var(--red)}
.idx .row{display:flex;justify-content:space-between;gap:10px;margin-top:8px;font-size:13px}
.idx .k{color:var(--muted)}
.idx .warn{display:flex;gap:6px;margin-top:6px;font-size:12px;align-items:flex-start}
.idx .opt{margin-top:8px;border-top:1px dashed var(--line);padding-top:8px;font-size:12px;
  color:var(--muted)}
.idx .opt b{color:var(--text);font-weight:600}
.login{position:fixed;inset:0;background:rgba(11,15,20,.96);display:flex;align-items:center;
justify-content:center;padding:24px;z-index:20}
.login .box{width:100%;max-width:320px}
.login h2{color:var(--text);font-size:16px;text-transform:none;letter-spacing:0;margin-bottom:12px}
</style>
</head>
<body>
<header>
  <span id="dot" class="dot"></span>
  <h1>NSE Paper Agent</h1>
  <span class="clock" id="clock">--:--</span>
</header>
<main>
  <div id="banner" class="banner hidden"></div>
  <section class="cards" id="cards"></section>

  <section class="panel">
    <div class="panel-head"><h2>Positions</h2><span class="muted" id="posmeta"></span></div>
    <div id="positions"><p class="note">loading...</p></div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Indices</h2><span class="muted" id="idxmeta"></span></div>
    <div id="indices"><p class="note">loading...</p></div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Controls</h2><span class="muted" id="ctlmeta"></span></div>
    <div class="controls">
      <button id="btn-cycle" class="primary">Run cycle</button>
      <label class="chk"><input type="checkbox" id="chk-force"> force</label>
      <button id="btn-loop">Start loop</button>
      <button id="btn-pause">Pause entries</button>
      <button id="btn-reset" class="danger">Reset</button>
    </div>
    <div class="row">
      <select id="sel-interval" aria-label="interval"></select>
      <input type="text" id="inp-symbol" placeholder="Add symbol e.g. SBIN" autocomplete="off">
      <button id="btn-add">Add</button>
    </div>
    <div class="chips" id="watchlist"></div>
    <p class="note" id="lastrun"></p>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Activity</h2><span class="muted" id="feedmeta"></span></div>
    <ul class="feed" id="activity"></ul>
  </section>
</main>

<div id="login" class="login hidden">
  <div class="box">
    <h2>Enter access token</h2>
    <input type="password" id="inp-token" placeholder="token" autocomplete="off">
    <div class="row"><button id="btn-token" class="primary" style="flex:1">Unlock</button></div>
  </div>
</div>

<script>
const TOKEN_KEY = "nse_agent_token";
(function adoptTokenFromUrl() {
  // one-click links carry ?token=... - keep it so later visits need nothing
  const param = new URLSearchParams(location.search).get("token");
  if (param) {
    localStorage.setItem(TOKEN_KEY, param);
    history.replaceState(null, "", location.pathname);
  }
})();
let TOKEN = localStorage.getItem(TOKEN_KEY) || "";
const fmt = (n, d = 2) => (n === null || n === undefined || isNaN(n)) ? "n/a"
  : Number(n).toLocaleString("en-IN", {minimumFractionDigits: d, maximumFractionDigits: d});
const signed = (n, d = 2) => (n === null || n === undefined || isNaN(n)) ? "n/a"
  : (n >= 0 ? "+" : "") + Number(n).toLocaleString("en-IN", {minimumFractionDigits: d, maximumFractionDigits: d});
const esc = (s) => String(s === null || s === undefined ? "" : s)
  .replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

async function api(path, options = {}) {
  const headers = Object.assign({}, options.headers || {});
  if (TOKEN) headers["Authorization"] = "Bearer " + TOKEN;
  if (options.body) headers["Content-Type"] = "application/json";
  const res = await fetch(path, Object.assign({}, options, {headers}));
  if (res.status === 401) { showLogin(true); throw new Error("unauthorized"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || data.reason || res.statusText);
  return data;
}

function showLogin(show) {
  document.getElementById("login").classList.toggle("hidden", !show);
  if (show) document.getElementById("inp-token").focus();
}
document.getElementById("btn-token").onclick = () => {
  TOKEN = document.getElementById("inp-token").value.trim();
  localStorage.setItem(TOKEN_KEY, TOKEN);
  showLogin(false); refresh();
};

function dotClass(s) {
  if (s.halted) return "halt";
  if (s.entries_paused) return "pause";
  return s.running ? "run" : "";
}

function renderStatus(s) {
  const dot = document.getElementById("dot");
  dot.className = "dot " + dotClass(s);
  dot.title = s.running ? "loop running" : "loop stopped";
  document.getElementById("clock").textContent = s.market_time || "";

  const banner = document.getElementById("banner");
  const notes = [];
  if (String(s.model || "").includes("mock")) notes.push("No LLM key set - using the deterministic fallback strategy.");
  if (!s.market_open) notes.push("Market closed - cycles outside the session need force.");
  if (s.halted) notes.push("Daily loss limit reached; entries halted for today.");
  if (s.entries_paused && !s.halted) notes.push("New entries paused.");
  banner.className = "banner" + (notes.length ? "" : " hidden");
  banner.textContent = notes.join(" ");

  const cards = [
    ["Equity", "INR " + fmt(s.equity, 0)],
    ["Cash", "INR " + fmt(s.cash, 0)],
    ["Day P&L", signed(s.day_pnl, 0) + (s.day_pnl_pct === null || s.day_pnl_pct === undefined ? "" : " (" + signed(s.day_pnl_pct) + "%)")],
    ["Realized", signed(s.realized_pnl, 0)],
    ["Unrealized", signed(s.unrealized_pnl, 0)],
    ["Charges", "INR " + fmt(s.charges_paid, 0)],
    ["Positions", String(s.positions.length)],
    ["Cycles", String(s.cycles)]
  ];
  document.getElementById("cards").innerHTML = cards.map(([k, v]) => {
    const cls = k.endsWith("P&L") || k === "Realized" || k === "Unrealized"
      ? (String(v).startsWith("-") ? "neg" : "pos") : "";
    return '<div class="card"><div class="k">' + esc(k) + '</div><div class="v ' + cls + '">'
      + esc(v) + "</div></div>";
  }).join("");

  const wrap = document.getElementById("positions");
  if (!s.positions.length) {
    wrap.innerHTML = '<p class="note">No open positions.</p>';
  } else {
    const rows = s.positions.map((p) => '<tr><td><div class="sym">' + esc(p.symbol)
      + '</div><div class="sub">' + p.qty + " @ " + fmt(p.avg_price) + "</div></td><td>"
      + fmt(p.last_price) + '</td><td>' + fmt(p.market_value, 0) + '</td><td class="'
      + (p.unrealized_pnl < 0 ? "neg" : "pos") + '">' + signed(p.unrealized_pnl, 0)
      + '</td><td><button class="sm" data-close="' + esc(p.symbol) + '">Close</button></td></tr>'
      + '<tr><td class="sub" colspan="5">stop ' + fmt(p.stop_loss) + " / target " + fmt(p.take_profit)
      + "</td></tr>").join("");
    wrap.innerHTML = '<table><thead><tr><th>Symbol</th><th>Last</th><th>Value</th>'
      + "<th>P&amp;L</th><th></th></tr></thead><tbody>" + rows + "</tbody></table>";
    wrap.querySelectorAll("[data-close]").forEach((btn) => {
      btn.onclick = async () => {
        btn.disabled = true;
        try { await api("/api/close", {method: "POST", body: JSON.stringify({symbol: btn.dataset.close})}); }
        catch (e) { alert("Close failed: " + e.message); }
        refresh();
      };
    });
  }
  document.getElementById("posmeta").textContent = s.positions.length
    ? s.positions.length + " open" : "flat";

  document.getElementById("chk-force").checked = !!s.force;
  const loopBtn = document.getElementById("btn-loop");
  loopBtn.textContent = s.running ? "Stop loop" : "Start loop";
  loopBtn.dataset.running = s.running ? "1" : "";
  const pauseBtn = document.getElementById("btn-pause");
  pauseBtn.textContent = s.entries_paused ? "Resume entries" : "Pause entries";
  pauseBtn.dataset.paused = s.entries_paused ? "1" : "";

  const sel = document.getElementById("sel-interval");
  if (!sel.options.length) {
    ["1m", "5m", "15m", "1h", "1d"].forEach((iv) => {
      const opt = document.createElement("option");
      opt.value = iv; opt.textContent = iv; sel.appendChild(opt);
    });
    sel.onchange = () => api("/api/interval", {method: "POST", body: JSON.stringify({interval: sel.value})})
      .then(refresh).catch((e) => alert(e.message));
  }
  sel.value = s.interval;

  document.getElementById("ctlmeta").textContent =
    (s.market_open ? "market open" : "market closed") + " - " + s.model;

  document.getElementById("watchlist").innerHTML = (s.watchlist || []).map((sym) =>
    '<span class="chip">' + esc(sym) + '<button data-remove="' + esc(sym)
    + '" title="remove">&times;</button></span>').join("");
  document.querySelectorAll("[data-remove]").forEach((btn) => {
    btn.onclick = () => api("/api/symbols", {method: "POST",
      body: JSON.stringify({action: "remove", symbol: btn.dataset.remove})})
      .then(refresh).catch((e) => alert(e.message));
  });

  document.getElementById("lastrun").textContent = s.last_summary
    ? "last cycle: " + s.last_summary + (s.last_error ? " | error: " + s.last_error : "")
    : (s.last_error ? "error: " + s.last_error : "no cycles run yet in this session");
}

const IDX_TREND = {strong_up: "^^ strong UP", up: "^ UP", sideways: "-> sideways",
  down: "v DOWN", strong_down: "vv strong DOWN"};

function idxLevels(levels, key, count) {
  const rows = (levels && levels[key]) || [];
  if (!rows.length) return "-";
  return rows.slice(0, count || 2)
    .map((l) => fmt(l.price, 1) + " " + String(l.label || "").replace(/_/g, " ")).join(", ");
}

function renderIndices(board) {
  const wrap = document.getElementById("indices");
  const meta = document.getElementById("idxmeta");
  const rows = (board && board.indices) || [];
  meta.textContent = (board ? (board.market_open ? "market open" : "market closed") + " - "
    + board.interval : "");
  if (!rows.length) { wrap.innerHTML = '<p class="note">No index data.</p>'; return; }
  wrap.innerHTML = rows.map((r) => {
    if (r.error) {
      return '<div class="idx"><div class="head"><span class="name">' + esc(r.index)
        + '</span></div><div class="row"><span class="k">error</span><span>'
        + esc(r.error) + "</span></div></div>";
    }
    const chg = signed(r.change, 2) + " (" + signed(r.change_pct, 2) + "%)";
    const rev = r.reversal || {};
    const warnings = (rev.warnings || []).slice(0, 4).map((w) =>
      '<div class="warn"><span class="tag ' + esc(String(w.severity).toUpperCase())
      + '">' + esc(String(w.severity)) + "</span><span>" + esc(w.message) + "</span></div>").join("");
    let opt = '<div class="opt">option chain unavailable - '
      + esc((r.options && r.options.reason) || "no data") + "</div>";
    const o = r.options || {};
    if (o.available) {
      const pain = o.max_pain || {};
      const pressure = o.buying_pressure || {};
      const bulk = o.bulk_strikes || {};
      opt = '<div class="opt">'
        + "expiry " + esc(o.expiry) + " | ATM <b>" + fmt(o.atm_strike, 0) + "</b> | max pain <b>"
        + fmt(pain.strike, 0) + "</b> (" + fmt(pain.distance_pct, 2) + "%)"
        + " | PCR <b>" + fmt((o.pcr || {}).oi_pcr, 2) + "</b> "
        + esc((o.pcr || {}).bias || "") + "<br>"
        + "max buying side <b>" + esc(pressure.max_buying_side || "-") + "</b> ("
        + fmt(pressure.max_buying_pct, 1) + "% of premium) - " + esc(pressure.flow || "")
        + "<br>bulk PE " + esc((bulk.PE || []).map((b) => fmt(b.strike, 0)).join(", ") || "-")
        + " | CE " + esc((bulk.CE || []).map((b) => fmt(b.strike, 0)).join(", ") || "-")
        + "<br>CE resistance "
        + esc(((o.walls || {}).call_resistance || []).map((b) => fmt(b.strike, 0)).join(", ") || "-")
        + " | PE support "
        + esc(((o.walls || {}).put_support || []).map((b) => fmt(b.strike, 0)).join(", ") || "-")
        + "</div>";
    }
    return '<div class="idx"><div class="head"><span class="name">' + esc(r.name) + "</span>"
      + "<span>" + fmt(r.spot, 2) + "</span>"
      + '<span class="' + (r.change >= 0 ? "pos" : "neg") + '">' + esc(chg) + "</span>"
      + '<span class="lv ' + esc(rev.risk || "low") + '">' + esc(String(rev.risk || "").toUpperCase())
      + "</span></div>"
      + '<div class="row"><span class="k">trend</span><span>'
      + esc((IDX_TREND[r.trend.label] || r.trend.label)
        + " (" + fmt(r.trend.score, 2) + ")") + "</span></div>"
      + '<div class="row"><span class="k">range</span><span>' + fmt(r.range.low, 1) + " - "
      + fmt(r.range.high, 1) + " (" + fmt(r.range.width_pct, 2) + "%)</span></div>"
      + '<div class="row"><span class="k">resistance</span><span>'
      + esc(idxLevels(r.levels, "resistance")) + "</span></div>"
      + '<div class="row"><span class="k">support</span><span>'
      + esc(idxLevels(r.levels, "support")) + "</span></div>"
      + '<div class="row"><span class="k">reversal</span><span>'
      + fmt(rev.score, 0) + "/100" + (rev.bias && rev.bias !== "none" ? " - " + esc(rev.bias) : "")
      + "</span></div>" + warnings + opt + "</div>";
  }).join("");
}

function renderFeed(events) {  const list = document.getElementById("activity");
  document.getElementById("feedmeta").textContent = events.length + " events";
  list.innerHTML = events.map((e) => {
    const detail = e.event === "DECISION"
      ? (e.symbol + " " + e.action + (e.reasoning ? " - " + e.reasoning : ""))
      : e.event === "FILL"
        ? (e.symbol + " " + e.side + " " + e.qty + "@" + fmt(e.price) + " (" + e.reason + ")")
        : e.event === "SKIP" || e.event === "RISK_REJECT"
          ? (e.symbol + " " + (e.reason || e.detail || ""))
          : e.event === "WEB_ACTION" ? (e.action + " " + (e.symbol || e.interval || ""))
            : (e.reason || e.day || "");
    return '<li><span class="t">' + esc(e.ts_readable || "") + '</span><span class="tag '
      + esc(e.event) + '">' + esc(e.event) + "</span><span>" + esc(detail) + "</span></li>";
  }).join("");
}

async function refresh() {
  try {
    const [status, events, board] = await Promise.all([
      api("/api/status"), api("/api/journal?limit=20"), api("/api/indices")]);
    renderStatus(status); renderFeed(events); renderIndices(board);
  } catch (e) { /* login screen or transient error */ }
}

document.getElementById("btn-cycle").onclick = async (ev) => {
  ev.target.disabled = true;
  const force = document.getElementById("chk-force").checked;
  try { await api("/api/cycle", {method: "POST", body: JSON.stringify({force})}); }
  catch (e) { alert("Cycle failed: " + e.message); }
  ev.target.disabled = false; refresh();
};
document.getElementById("chk-force").onchange = (ev) =>
  api("/api/force", {method: "POST", body: JSON.stringify({force: ev.target.checked})}).catch(() => {});
document.getElementById("btn-loop").onclick = async (ev) => {
  const running = !!ev.target.dataset.running;
  const body = running ? {} : {poll_seconds: 300, force: document.getElementById("chk-force").checked};
  try { await api(running ? "/api/loop/stop" : "/api/loop/start", {method: "POST", body: JSON.stringify(body)}); }
  catch (e) { alert(e.message); }
  refresh();
};
document.getElementById("btn-pause").onclick = async (ev) => {
  const paused = !!ev.target.dataset.paused;
  try { await api("/api/pause", {method: "POST", body: JSON.stringify({paused: !paused})}); }
  catch (e) { alert(e.message); }
  refresh();
};
document.getElementById("btn-reset").onclick = async () => {
  if (!confirm("Delete the paper portfolio and journal history?")) return;
  try { await api("/api/reset", {method: "POST", body: "{}"}); }
  catch (e) { alert(e.message); }
  refresh();
};
document.getElementById("btn-add").onclick = () => {
  const input = document.getElementById("inp-symbol");
  const symbol = input.value.trim();
  if (!symbol) return;
  api("/api/symbols", {method: "POST", body: JSON.stringify({action: "add", symbol})})
    .then(() => { input.value = ""; refresh(); })
    .catch((e) => alert(e.message));
};

refresh();
setInterval(refresh, 4000);
</script>
</body>
</html>
"""


def make_handler(controller: WebController, token: str = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "NSEPaperAgent/1.0"

        # -------------------------------------------------------------- helpers
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str, status: int = 200) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self, query: dict) -> bool:
            if not token:
                return True
            if query.get("token", [""])[0] == token:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {token}"

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                return {}
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}
            return payload if isinstance(payload, dict) else {}

        def log_message(self, fmt_str, *args):  # quieter logs
            if os.environ.get("TRADING_WEB_VERBOSE"):
                super().log_message(fmt_str, *args)

        # ----------------------------------------------------------------- gets
        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if not self._authorized(query):
                return self._send_json({"error": "unauthorized"}, status=401)
            route = parsed.path
            try:
                if route in ("/", "/index.html"):
                    return self._send_html(HTML_PAGE)
                if route == "/api/status":
                    return self._send_json(controller.status())
                if route == "/api/journal":
                    limit = min(int(query.get("limit", ["20"])[0] or 20), 200)
                    return self._send_json(controller.journal(limit=limit))
                if route == "/api/trades":
                    limit = min(int(query.get("limit", ["20"])[0] or 20), 200)
                    return self._send_json(controller.trades(limit=limit))
                if route == "/api/indices":
                    return self._send_json(controller.indices())
                return self._send_json({"error": f"unknown route {route}"}, status=404)
            except Exception as exc:
                return self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        # ---------------------------------------------------------------- posts
        def do_POST(self):
            parsed = urlparse(self.path)
            if not self._authorized(parse_qs(parsed.query)):
                return self._send_json({"error": "unauthorized"}, status=401)
            route = parsed.path
            body = self._body()
            try:
                if route == "/api/cycle":
                    return self._send_json(controller.run_cycle_now(body.get("force")))
                if route == "/api/loop/start":
                    return self._send_json(controller.start_loop(
                        poll_seconds=body.get("poll_seconds"), force=body.get("force")))
                if route == "/api/loop/stop":
                    return self._send_json(controller.stop_loop())
                if route == "/api/pause":
                    return self._send_json(controller.set_entries_paused(bool(body.get("paused"))))
                if route == "/api/force":
                    return self._send_json(controller.set_force(bool(body.get("force"))))
                if route == "/api/close":
                    return self._send_json(controller.close_position(body.get("symbol", "")))
                if route == "/api/symbols":
                    return self._send_json(
                        controller.set_symbols(body.get("action", ""), body.get("symbol", "")))
                if route == "/api/interval":
                    return self._send_json(controller.set_interval(body.get("interval", "")))
                if route == "/api/reset":
                    return self._send_json(controller.reset_portfolio())
                return self._send_json({"error": f"unknown route {route}"}, status=404)
            except Exception as exc:
                return self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    return Handler


def _lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def build_controller(config_path: str = "config.json", poll_seconds: int = None,
                     force_mock: bool = False) -> WebController:
    config = load_config(config_path)
    agent = TradingAgent(config, force_mock=force_mock)
    return WebController(config, agent, config_path=config_path, poll_seconds=poll_seconds)


def resolve_token(state_dir: str, token: str = None, auto_token: bool = False):
    """Resolve the dashboard token.

    An explicit ``token`` always wins. With ``auto_token`` a random token is
    generated once and remembered in ``<state_dir>/.web_token`` (gitignored),
    so one-click launchers can hand out a link that logs itself in.
    """
    if token:
        return token
    if not auto_token:
        return None
    path = Path(state_dir) / ".web_token"
    try:
        saved = path.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    except OSError:
        pass
    generated = secrets.token_urlsafe(9)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated, encoding="utf-8")
        path.chmod(0o600)
    except OSError:
        pass
    return generated


def serve(host: str = "127.0.0.1", port: int = 8000, config_path: str = "config.json",
          token: str = None, poll_seconds: int = None, force_mock: bool = False,
          auto_token: bool = False, open_browser: bool = False, autostart: bool = False) -> None:
    """Start the dashboard (optionally one-click: token, browser and auto loop)."""
    controller = build_controller(config_path, poll_seconds, force_mock)
    token = resolve_token(controller.config.state_dir, token, auto_token)
    handler = make_handler(controller, token)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    bound_host, bound_port = server.server_address[:2]
    base = f"http://{bound_host}:{bound_port}"
    print(f"NSE paper agent dashboard listening on {base}")
    if host in ("0.0.0.0", "::", ""):
        print(f"  on your phone (same Wi-Fi): http://{_lan_ip()}:{bound_port}")
    print(f"  model: {'mock (no key)' if controller.agent.using_mock else controller.config.llm.model}")
    if token:
        print(f"  open this link (it remembers the token): {base}/?token={token}")
        if host in ("0.0.0.0", "::", ""):
            print(f"  phone link: http://{_lan_ip()}:{bound_port}/?token={token}")
    else:
        print("  no token set (local access only); use --token to require one")
    if autostart:
        controller.start_loop()
        print(f"  auto loop started: one cycle every {controller.poll_seconds}s "
              "(pause/resume from the dashboard)")
    print("Ctrl+C to stop")
    sys.stdout.flush()

    if open_browser:
        def _open():
            time.sleep(0.8)
            try:
                webbrowser.open(base + (f"/?token={token}" if token else ""))
                print("  browser opened", flush=True)
            except Exception:
                print("  could not open a browser automatically; use the link above", flush=True)

        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        controller.stop_loop()
        server.server_close()
