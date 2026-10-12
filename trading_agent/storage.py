"""JSON state persistence and the append-only trade journal."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

DEFAULT_STATE = {
    "version": 1,
    "current_day": None,
    "day_start_equity": None,
    "halted": False,
    "last_prices": {},
    "cooldowns": {},
    "cycles": 0,
    "created_at": None,
}


class StateStore:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.state_path = self.directory / "state.json"
        self.journal_path = self.directory / "journal.jsonl"

    def ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> dict:
        default = json.loads(json.dumps(DEFAULT_STATE))
        default["created_at"] = int(time.time())
        default["broker"] = {}
        if not self.state_path.exists():
            return default
        try:
            stored = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default
        default.update({k: v for k, v in stored.items() if k != "version"})
        default["version"] = 1
        return default

    def save_state(self, state: dict) -> None:
        self.ensure()
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def journal(self, event: str, **fields) -> dict:
        self.ensure()
        record = {"ts": int(time.time()), "event": event}
        record.update(fields)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return record

    def read_events(self, event: str = None, limit: int = None) -> list:
        if not self.journal_path.exists():
            return []
        events = []
        with self.journal_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event and record.get("event") != event:
                    continue
                events.append(record)
        if limit:
            events = events[-limit:]
        return events

    def reset(self) -> None:
        for path in (self.state_path, self.journal_path):
            if path.exists():
                path.unlink()
