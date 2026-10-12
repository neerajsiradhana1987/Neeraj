# AGENTS.md

Conventions for working in this repository.

## Project

`nse-paper-agent`: a paper-trading agent for NSE equities with LLM-driven trade
decisions. Pure Python standard library only — no third-party runtime
dependencies, no pip requirements file.

## Commands

```bash
python3 -m unittest discover -s tests -v     # full test suite (offline)
python3 -m trading_agent doctor              # environment check
python3 -m trading_agent demo --cycles 10    # offline demo (synthetic data + mock LLM)
python3 -m trading_agent run --once --force  # single cycle against live Yahoo data
python3 -m trading_agent web --host 0.0.0.0 --port 8000 --token secret   # mobile dashboard
./start-website.sh                        # one-click: token + browser + auto loop
```

Run the test suite after any code change; do not commit if it is red.
There is no linter configured — match the surrounding style (module
docstrings, type hints on public functions, 4-space indent, ~100 col lines).

## Rules for changes

- Keep the standard-library-only constraint. If a new dependency is genuinely
  required, ask first.
- Data providers must stay interchangeable: `fetch(symbol, interval, bars)`.
- The risk layer (`risk.py`) is a hard boundary — the LLM may suggest but never
  bypass it. New order paths must go through `RiskManager`.
- Every action must be journaled to `state/journal.jsonl` via `StateStore.journal`
  (event types: `DAY_START`, `SKIP`, `DECISION`, `RISK_REJECT`, `FILL`, `HALT`,
  `MARKET_CLOSED`, `CYCLE`, `WEB_ACTION`).
- Secrets live in `.env` / environment variables only, never in `config.json`.
- Tests must run offline: use `SyntheticProvider` and mock `urllib.request.urlopen`.
- The `state/` directory is gitignored runtime data — do not commit it.
