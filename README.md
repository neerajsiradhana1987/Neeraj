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
| `doctor` | check config, Yahoo connectivity, LLM key, state dir, market session |
| `run --once` | single cycle (fetch data, decide, maybe trade) |
| `run --poll N` | loop forever, one cycle every N seconds |
| `run --force` | trade even outside market hours / with stale-ish data |
| `run --symbols "RELIANCE TCS HDFCBANK"` | override the watchlist (`.NS` is added automatically) |
| `run --interval 5m` | `1m`, `5m`, `15m`, `1h`, `1d` |
| `run --mock-llm` | force the deterministic fallback strategy |
| `demo` | offline demo on synthetic data (no key, no network) |
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

`config.json` sections: `llm`, `risk`, `broker`, `market`, `data`, plus top-level `state_dir` and `poll_seconds`. Secrets come from the environment, never from the file:

```bash
TRADING_LLM_API_KEY=...          # required for LLM mode
TRADING_LLM_BASE_URL=https://openrouter.ai/api/v1
TRADING_LLM_MODEL=openai/gpt-4o-mini
TRADING_LLM_ENABLED=true
```

Swap the model/provider freely — anything speaking OpenAI `chat/completions` works (set `TRADING_LLM_BASE_URL` for a local vLLM/Ollama endpoint).

Add NSE holidays to `market.holidays` (dates as `YYYY-MM-DD`) to keep the agent idle on them; `doctor`/`status` show whether the session is open.

## Layout

```
trading_agent/
  __main__.py     CLI (run/demo/status/trades/reset/doctor/init)
  agent.py        cycle orchestration: data -> exits -> LLM -> risk -> fills
  broker.py       simulated INR ledger + NSE charge model + slippage
  risk.py         position sizing and veto rules
  llm.py          OpenAI-compatible client, prompt, deterministic mock
  decision.py     strict JSON schema for trade decisions
  signals.py      indicator snapshot builder
  indicators.py   SMA/EMA/RSI/MACD/ATR/Bollinger (stdlib only)
  data.py         Yahoo / CSV / synthetic bar providers
  market_hours.py NSE session + data freshness (IST)
  storage.py      state.json + journal.jsonl persistence
  report.py       status and trade rendering
tests/            45 unit and end-to-end tests
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Everything runs offline (synthetic provider + mocked HTTP), so the suite is safe in CI.

## Running it for real (still paper)

```bash
# long-lived loop, log to file, running after you log out
nohup python3 -m trading_agent run --poll 300 > agent.log 2>&1 &
```

Watch it with `python3 -m trading_agent status`, audit decisions with `tail -f state/journal.jsonl`, and stop with `python3 -m trading_agent reset --yes` when you want a clean slate.

## Safety notes

- This agent only simulates trades. It never contacts a broker and cannot move money.
- NSE holiday lists and exchange rules change — refresh `market.holidays` and the charge model in `broker.py` periodically.
- LLMs can be confidently wrong. The risk layer is what keeps the account alive; treat LLM output as one signal, not truth.
- Nothing here is financial advice.
