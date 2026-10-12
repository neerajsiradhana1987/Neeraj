#!/usr/bin/env bash
# Stop the dashboard started by start-website.sh.
cd "$(dirname "$0")" || exit 1
PORT="${PORT:-8000}"
if command -v fuser >/dev/null 2>&1 && fuser "${PORT}/tcp" >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null && echo "stopped dashboard on port ${PORT}"
elif pkill -f "trading_agent web" 2>/dev/null; then
  echo "stopped dashboard"
else
  echo "no dashboard process found"
fi
