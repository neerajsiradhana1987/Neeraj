# NSE Paper Trading Agent

An autonomous **paper trading** agent for the Indian stock market (NSE). Every cycle it:

1. pulls OHLCV bars for your watchlist (Yahoo Finance, or local CSV, or synthetic data),
2. computes technical context locally (RSI, MACD, ATR, Bollinger, SMA/EMA, volatility, trend alignment),
3. asks an LLM for a single decision — `BUY` / `SELL` / `HOLD` with quantity, stop-loss, take-profit and reasoning — as strict JSON,
4. runs that decision through hard risk guardrails,
5. executes it on a **simulated INR broker** with realistic NSE charges and slippage,
6. persists the portfolio and an append-only trade journal.

No real money, no real orders, no broker account required. This is an educational/research tool, **not investment advice**.

## Requirements

- Python 3.9+ (developed on 3.10). **Zero third-party dependencies.**
- Optional: an API key for any OpenAI-compatible LLM endpoint (OpenRouter, OpenAI, Groq, Together, a local vLLM/Ollama server, ...). Without a key the agent falls back to a deterministic built-in strategy so demos and tests still run.

## Quickstart

**Just want to see it work?** Double-click `start-website.sh` — full website, no setup.

```bash
# 1. optional but recommended: fresh config + .env.example
python3 -m trading_agent init

# 2. sanity checks (config, data, LLM key, state dir, market session)
python3 -m trading_agent doctor

# 3. offline demo: synthetic prices + deterministic strategy, no key/network needed
python3 -m trading_agent demo --cycles 10

# 4. real NSE data, simulated money (one cycle)
python3 -m trading_agent run --once

# 5. loop during market hours (09:15-15:30 IST, Mon-Fri), one decision every 5 minutes
python3 -m trading_agent run --poll 300
```

For live LLM decisions:

```bash
cp .env.example .env
# edit .env: TRADING_LLM_API_KEY=sk-or-...   (OpenRouter, OpenAI, Groq, ...)
python3 -m trading_agent run --once --force     # --force trades outside the session too
```

## Commands

| Command | Purpose |
| --- | --- |
| `init` | write `config.json` and `.env.example` |
| `doctor` | check config, Yahoo connectivity, LLM key, NSE option chain, state dir, market session |
| `run --once` | single cycle (fetch data, decide, maybe trade) |
| `run --poll N` | loop forever, one cycle every N seconds |
| `run --force` | trade even outside market hours / with stale-ish data |
| `run --symbols "RELIANCE TCS HDFCBANK"` | override the watchlist (`.NS` is added automatically) |
| `run --interval 5m` | `1m`, `5m`, `15m`, `1h`, `1d` |
| `run --mock-llm` | force the deterministic fallback strategy |
| `demo` | offline demo on synthetic data (no key, no network) |
| `web` | mobile web dashboard (`--host 0.0.0.0 --port 8000 --token ...`) |
| `indices` | intraday trend / range / reversal warnings / option chain for NIFTY, BANKNIFTY, SENSEX |
| `status` | cash, equity, positions, P&L, charges, halt status (`--json` for machines) |
| `trades [--limit N]` | recent fills from the journal |
| `reset --yes` | delete the local paper portfolio |

Example output:

```
nse-paper-agent 0.1.0 | symbols: RELIANCE.NS, TCS.NS | interval 1d
model: openai/gpt-4o-mini | state: state
[09:20:11 IST] | equity INR 499,949 | cash INR 451,074 | positions 1 |
fills: BUY 23 TCS.NS@2126.06(LLM_BUY) | rejected: INFY.NS(min_cash/cash)
```

## How a decision is made

`trading_agent/llm.py` builds a compact snapshot (indicators + portfolio state) and a strict system prompt requiring a single JSON object:

```json
{"action": "BUY", "quantity": 23, "confidence": 0.74,
 "stop_loss": 2050.0, "take_profit": 2280.0,
 "reasoning": "Price above SMA20/50, MACD histogram positive, RSI 58."}
```

Parsing is deliberately tolerant (`trading_agent/decision.py`): markdown fences, surrounding prose, string quantities and crazy confidences are all handled; an unparseable or invalid reply is treated as `HOLD` and journaled, never guessed at.

## Guardrails (the LLM cannot bypass these)

Configured under `risk` in `config.json`, enforced by `trading_agent/risk.py`:

| Limit | Default |
| --- | --- |
| Max % of equity per symbol | 15% |
| Max concurrent positions | 5 |
| Max % of equity risked per trade (from stop distance) | 1% |
| Min cash kept aside | 10% |
| Max value per order | INR 200,000 |
| Daily loss limit (halts trading for the day) | 3% |
| Cooldown after a stop-out | 60 minutes per symbol |

The broker also:

- applies adverse **slippage** (default 5 bps) on every fill,
- charges Indian delivery taxes/fees: brokerage (min of INR 20 or 0.03%), STT 0.1% on sells, NSE transaction charge, SEBI turnover fee, stamp duty on buys and 18% GST (`trading_agent/broker.py`),
- enforces stop-loss/take-profit using each bar's high/low, so a position can be closed even if the LLM says nothing,
- only trades on fresh data inside the NSE session unless you pass `--force`.

## Configuration

`config.json` sections: `llm`, `risk`, `broker`, `market`, `data`, `indices`, plus top-level `state_dir` and `poll_seconds`. Secrets come from the environment, never from the file:

```bash
TRADING_LLM_API_KEY=...          # required for LLM mode
TRADING_LLM_BASE_URL=https://openrouter.ai/api/v1
TRADING_LLM_MODEL=openai/gpt-4o-mini
TRADING_LLM_ENABLED=true
TRADING_INDICES_OPTION_SOURCE=nse      # nse | synthetic | off
TRADING_INDICES_INTERVAL=5m
```

Swap the model/provider freely — anything speaking OpenAI `chat/completions` works (set `TRADING_LLM_BASE_URL` for a local vLLM/Ollama endpoint).

Add NSE holidays to `market.holidays` (dates as `YYYY-MM-DD`) to keep the agent idle on them; `doctor`/`status` show whether the session is open.

## Indices board (NIFTY, BANKNIFTY, SENSEX)

Read-only intraday screen for the three headline indices. No portfolio, no LLM, no
journal - it just reads bars (Yahoo) and, optionally, the NSE option chain:

```bash
# live: Yahoo bars + NSE option chain, refreshed every 60s
python3 -m trading_agent indices --watch 60

# offline demo: synthetic bars and a synthetic chain (no network at all)
python3 -m trading_agent indices --source synthetic --option-source synthetic

# just levels/trend, no option chain, machine-readable
python3 -m trading_agent indices --option-source off --json
```

For every index it prints:

- **trend** - score (-1..+1) from EMA stack, price vs VWAP, market structure, RSI momentum and slope,
- **range** - probable low/high (options straddle when available, otherwise the median of recent sessions),
  with the pivot ladder S3-R3, the central pivot range, the 15-minute opening range and breakout targets,
- **reversal warnings** - RSI extremes, price/RSI divergence, Bollinger band rides, engulfing candles,
  long rejection wicks, VWAP loss/stretch, failed opening-range breaks, parabolic runs, CE/PE open-interest
  walls and max-pain magnet nearby. Summarised as `LOW`/`MEDIUM`/`HIGH` (0-100 score) with a bias,
- **options** - max pain (where option buyers lose the most), PCR on volume and open interest, the side
  paying the most premium today, the strikes where positions are being built in bulk, the open-interest
  support/resistance walls and the ATM straddle breakevens.

Useful flags: `--names "NIFTY BANKNIFTY SENSEX"`, `--interval 5m|15m|1m|1h`,
`--bars N`, `--source yahoo|csv|synthetic`, `--option-source nse|synthetic|off`,
`--expiry 27-Oct-2026`, `--json`, `--watch SECONDS`.

Data notes:

- index bars come from Yahoo (`^NSEI`, `^NSEBANK`, `^BSESN`) exactly like the agent's bars,
- the live option chain uses the NSE API and needs an Indian/browser-reachable connection; the dashboard
  and CLI degrade to a clear message instead of failing when NSE refuses (HTTP 403),
- SENSEX options trade on BSE, which exposes no public option-chain API, so a live SENSEX chain is not
  available - use `--option-source synthetic` (or `off`) for SENSEX while NIFTY/BANKNIFTY stay live,
  or pass `--option-source synthetic` for a fully offline demo,
- the same board is available in the web dashboard (`/api/indices`, cached for
  `indices.refresh_seconds`).

## Layout

```
trading_agent/
  __main__.py     CLI (run/demo/status/trades/reset/doctor/indices/init)
  agent.py        cycle orchestration: data -> exits -> LLM -> risk -> fills
  broker.py       simulated INR ledger + NSE charge model + slippage
  risk.py         position sizing and veto rules
  llm.py          OpenAI-compatible client, prompt, deterministic mock
  decision.py     strict JSON schema for trade decisions
  signals.py      indicator snapshot builder
  indicators.py   SMA/EMA/RSI/MACD/ATR/Bollinger (stdlib only)
  data.py         Yahoo / CSV / synthetic bar providers
  indices.py      intraday index board: trend, range, reversal warnings
  options.py      NSE / synthetic option chains, max pain, OI walls, bulk strikes
  market_hours.py NSE session + data freshness (IST)
  storage.py      state.json + journal.jsonl persistence
  report.py       status, trade and indices-board rendering
  webui.py        stdlib HTTP server, JSON API and mobile dashboard
tests/            105 unit and end-to-end tests
start-website.sh  double-click launcher (Linux/macOS) + stop-website.sh
start-website.bat double-click launcher (Windows)
```

## One click (easiest)

Double-click **`start-website.sh`** (or **`start-website.bat`** on Windows, or the
**NSE Paper Agent** desktop entry). It will:

1. start the web server on port `8000`, reachable from your phone on the same Wi-Fi,
2. open your browser automatically,
3. generate an access token and give you a link that logs itself in,
4. **start trading immediately** (one cycle per minute).

Then open the printed phone link on your mobile and tap *Add to Home Screen*.
Stop it with **`stop-website.sh`** (or `Ctrl+C` in the launcher window).

<details>
<summary>Manual (terminal) alternative</summary>

```bash
python3 -m trading_agent web --host 0.0.0.0 --port 8000 --auto-token --open --autostart --poll 60
```

</details>

## Web dashboard (use it from your phone)

The same dashboard, started manually — standard library only, no extra installs:

```bash
python3 -m trading_agent web --host 0.0.0.0 --port 8000 --token pick-a-secret
```

Open `http://<your-computer-ip>:8000` on your phone (same Wi-Fi). On iOS/Android use
**Add to Home Screen** — it launches full-screen with a dark theme and refreshes itself.

What you get on the phone:

- **Live cards**: equity, cash, day P&L (₹ and %), realized/unrealized P&L, charges, positions, cycles.
- **Positions**: qty, average, last, value, P&L, stop/target levels, and a **Close** button for a manual exit.
- **Controls**: *Run cycle* (with a *force* toggle for outside session hours), *Start/Stop loop* (runs a cycle every N seconds in the background), *Pause entries* (stops new buys but still enforces stop-losses), *Reset*.
- **Watchlist**: add/remove symbols (e.g. `SBIN` → `SBIN.NS`) and switch the candle interval (`1m`–`1d`) — both are written back to `config.json`.
- **Activity feed**: decisions, fills, risk rejections, skips and halts in real time.

**Security**: with no token the server refuses non-local binds (`--host 0.0.0.0` warns loudly and
exposes your paper portfolio to the network). Always pass `--token` (or set `TRADING_WEB_TOKEN`) when
binding to anything other than localhost — every API call then needs `Authorization: Bearer <token>`,
and the page stores it in `localStorage` after a one-time unlock.

All web actions go through the same `RiskManager` and `StateStore.journal` path as the CLI, so the
phone can start/stop/pause the agent but can never bypass a risk limit.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

57 tests. Everything runs offline (synthetic provider + mocked HTTP), so the suite is safe in CI.

## Running it for real (still paper)

```bash
# long-lived loop, log to file, running after you log out
nohup python3 -m trading_agent run --poll 300 > agent.log 2>&1 &

# or keep the dashboard up and drive the loop from your phone
nohup python3 -m trading_agent web --host 0.0.0.0 --port 8000 --token secret > web.log 2>&1 &
```

Watch it with `python3 -m trading_agent status`, audit decisions with `tail -f state/journal.jsonl`, and stop with `python3 -m trading_agent reset --yes` when you want a clean slate.

## Safety notes

- This agent only simulates trades. It never contacts a broker and cannot move money.
- NSE holiday lists and exchange rules change — refresh `market.holidays` and the charge model in `broker.py` periodically.
- LLMs can be confidently wrong. The risk layer is what keeps the account alive; treat LLM output as one signal, not truth.
- Nothing here is financial advice.
