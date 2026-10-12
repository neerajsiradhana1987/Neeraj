#!/usr/bin/env bash
# One-click launcher for the NSE paper trading dashboard.
#
#   * double-click this file in your file manager (choose "Run"), or
#   * run ./start-website.sh in a terminal
#
# It starts the web server, opens your browser automatically, remembers an
# access token and starts trading on its own. Open the printed phone link on
# your mobile (same Wi-Fi) - it logs itself in. Press Ctrl+C to stop.
cd "$(dirname "$0")" || exit 1
PORT="${PORT:-8000}"
export PYTHONUNBUFFERED=1
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
exec "$PY" -m trading_agent web \
  --host 0.0.0.0 \
  --port "$PORT" \
  --auto-token \
  --open \
  --autostart \
  --poll 60
