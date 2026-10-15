"""Tests for the stdlib web dashboard and its JSON API."""
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

from trading_agent.agent import TradingAgent
from trading_agent.config import (AgentConfig, BrokerConfig, DataConfig, IndicesConfig,
                                 MarketConfig, RiskConfig)
from trading_agent.data import IST, SyntheticProvider
from trading_agent.llm import MockLLMClient
from trading_agent.webui import HTML_PAGE, WebController, make_handler, resolve_token


def _request(url, method="GET", payload=None, headers=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") or "{}"
        return exc.code, json.loads(body)


class WebDashboardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config_path = os.path.join(self._tmp.name, "config.json")
        config = AgentConfig(
            state_dir=self._tmp.name,
            broker=BrokerConfig(starting_cash=500000.0, slippage_bps=0.0),
            market=MarketConfig(symbols=["RELIANCE.NS"], interval="15m", bars=120),
            risk=RiskConfig(),
        )
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2)
        self.config = config

    @staticmethod
    def _read(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def _start(self, token=None):
        agent = TradingAgent(self.config, provider=SyntheticProvider(seed=21, bars=120),
                             llm=MockLLMClient())
        controller = WebController(self.config, agent, config_path=self.config_path)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(controller, token))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return controller, base_url

    def test_page_is_served(self):
        _, url = self._start()
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
        self.assertEqual(response.status, 200)
        self.assertIn("NSE Paper Agent", body)
        self.assertIn("viewport", body)

    def test_status_and_cycle_api(self):
        _, url = self._start()
        status, payload = _request(f"{url}/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(payload["cash"], 500000.0)
        self.assertIn("positions", payload)
        self.assertIn("watchlist", payload)
        self.assertFalse(payload["running"])

        status, payload = _request(f"{url}/api/cycle", method="POST", payload={"force": True})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertIn("summary", payload)

        status, events = _request(f"{url}/api/journal?limit=10")
        self.assertEqual(status, 200)
        self.assertTrue(any(event["event"] == "CYCLE" for event in events))
        self.assertIn("ts_readable", events[-1])

    def test_pause_blocks_new_entries(self):
        _, url = self._start()
        status, payload = _request(f"{url}/api/pause", method="POST", payload={"paused": True})
        self.assertTrue(payload["entries_paused"])

        status, payload = _request(f"{url}/api/status")
        self.assertTrue(payload["entries_paused"])

        state = json.loads(self._read(os.path.join(self._tmp.name, "state.json")))
        self.assertTrue(state["entries_paused"])

        status, payload = _request(f"{url}/api/pause", method="POST", payload={"paused": False})
        self.assertFalse(payload["entries_paused"])

    def test_loop_start_stop(self):
        controller, url = self._start()
        status, payload = _request(f"{url}/api/loop/start", method="POST",
                                  payload={"poll_seconds": 15})
        self.assertTrue(payload["ok"])
        self.assertTrue(controller.running)
        status, payload = _request(f"{url}/api/loop/stop", method="POST", payload={})
        self.assertTrue(payload["ok"])
        self.assertFalse(controller.running)
        status, payload = _request(f"{url}/api/loop/stop", method="POST", payload={})
        self.assertFalse(payload["ok"])

    def test_symbols_and_interval_persist(self):
        _, url = self._start()
        status, payload = _request(f"{url}/api/symbols", method="POST",
                                  payload={"action": "add", "symbol": "sbin"})
        self.assertEqual(status, 200)
        self.assertIn("SBIN.NS", payload["symbols"])

        status, payload = _request(f"{url}/api/symbols", method="POST",
                                  payload={"action": "add", "symbol": "SBIN.NS"})
        self.assertFalse(payload["ok"])

        status, payload = _request(f"{url}/api/symbols", method="POST",
                                  payload={"action": "remove", "symbol": "RELIANCE.NS"})
        self.assertNotIn("RELIANCE.NS", payload["symbols"])

        status, payload = _request(f"{url}/api/interval", method="POST", payload={"interval": "1d"})
        self.assertEqual(payload["interval"], "1d")
        stored = json.loads(self._read(self.config_path))
        self.assertEqual(stored["market"]["interval"], "1d")
        self.assertEqual(stored["market"]["symbols"], ["SBIN.NS"])

        status, payload = _request(f"{url}/api/interval", method="POST", payload={"interval": "3d"})
        self.assertFalse(payload["ok"])

    def test_close_position_route(self):
        agent = TradingAgent(self.config, provider=SyntheticProvider(seed=21, bars=120),
                             llm=MockLLMClient())
        agent.store.save_state({
            "broker": {
                "cash": 400000.0,
                "positions": {
                    "RELIANCE.NS": {
                        "symbol": "RELIANCE.NS", "qty": 10, "avg_price": 1000.0,
                        "stop_loss": 900.0, "take_profit": 1200.0, "opened_at": 0,
                    }
                },
                "realized_pnl": 0.0, "total_charges": 0.0, "total_slippage": 0.0,
            },
            "last_prices": {"RELIANCE.NS": 1010.0},
            "current_day": None, "day_start_equity": None, "halted": False,
            "cooldowns": {}, "cycles": 0,
        })
        controller = WebController(self.config, agent, config_path=self.config_path)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(controller, None))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        status, payload = _request(f"{url}/api/close", method="POST",
                                  payload={"symbol": "RELIANCE"})
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["fill"]["side"], "SELL")
        self.assertEqual(payload["fill"]["qty"], 10)

        status, payload = _request(f"{url}/api/status")
        self.assertEqual(payload["positions"], [])
        self.assertGreater(payload["cash"], 400000.0)

        status, payload = _request(f"{url}/api/close", method="POST",
                                  payload={"symbol": "RELIANCE.NS"})
        self.assertFalse(payload["ok"])
        self.assertIn("no open position", payload["reason"])

    def test_token_is_required_when_set(self):
        _, url = self._start(token="s3cret")
        status, payload = _request(f"{url}/api/status")
        self.assertEqual(status, 401)
        self.assertIn("unauthorized", payload["error"])

        status, payload = _request(f"{url}/api/status?token=s3cret")
        self.assertEqual(status, 200)

        status, payload = _request(f"{url}/api/status", headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(status, 200)

        status, payload = _request(f"{url}/api/cycle", method="POST",
                                  headers={"Authorization": "Bearer wrong"}, payload={"force": True})
        self.assertEqual(status, 401)

    def test_reset_endpoint(self):
        _, url = self._start()
        _request(f"{url}/api/cycle", method="POST", payload={"force": True})
        status, payload = _request(f"{url}/api/reset", method="POST", payload={})
        self.assertTrue(payload["ok"])
        status, payload = _request(f"{url}/api/status")
        self.assertEqual(payload["equity"], 500000.0)
        self.assertEqual(payload["cycles"], 0)


class TokenHelperTests(unittest.TestCase):
    def test_explicit_token_wins_and_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolve_token(tmp, token="mine", auto_token=True)
            self.assertFalse((Path(tmp) / ".web_token").exists())
            self.assertEqual(resolve_token(tmp, token="mine", auto_token=True), "mine")

    def test_auto_token_is_created_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(resolve_token(tmp))
            first = resolve_token(tmp, auto_token=True)
            self.assertTrue(first)
            self.assertGreaterEqual(len(first), 8)
            self.assertTrue((Path(tmp) / ".web_token").exists())
            self.assertEqual(resolve_token(tmp, auto_token=True), first)

    def test_auto_token_reads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".web_token").write_text("saved-token\n", encoding="utf-8")
            self.assertEqual(resolve_token(tmp, auto_token=True), "saved-token")


class HaltBehaviourTests(unittest.TestCase):
    """A daily-loss halt must block entries but still enforce exits."""

    def test_halted_still_enforces_stop_loss(self):
        from trading_agent.storage import StateStore

        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(
                state_dir=tmp,
                broker=BrokerConfig(starting_cash=100000.0, slippage_bps=0.0),
                market=MarketConfig(symbols=["RELIANCE.NS"], interval="15m", bars=120),
                risk=RiskConfig(min_cash_pct=0.0, max_position_pct=0.5),
            )
            store = StateStore(tmp)
            store.save_state({
                "broker": {
                    "cash": 90000.0,
                    "positions": {
                        "RELIANCE.NS": {
                            "symbol": "RELIANCE.NS", "qty": 10, "avg_price": 1000.0,
                            "stop_loss": 99999.0, "take_profit": None, "opened_at": 0,
                        }
                    },
                    "realized_pnl": 0.0, "total_charges": 0.0, "total_slippage": 0.0,
                },
                "current_day": datetime.now(IST).date().isoformat(),
                "day_start_equity": 90000.0,
                "halted": True,
                "last_prices": {"RELIANCE.NS": 1000.0},
                "cooldowns": {}, "cycles": 0,
            })
            agent = TradingAgent(config, provider=SyntheticProvider(seed=8, bars=120),
                                 llm=MockLLMClient())
            report = agent.run_cycle(force=True)
            self.assertTrue(report.halted)
            self.assertTrue(report.fills, "stop-loss should still fire while halted")
            self.assertEqual(report.fills[0].reason, "STOP_LOSS")
            self.assertFalse(any(fill.side == "BUY" for fill in report.fills),
                             "no new entries while halted")
            self.assertEqual(agent.status()["positions"], [])


if __name__ == "__main__":
    unittest.main()


class IndicesApiTests(unittest.TestCase):
    """The dashboard exposes the same board the CLI prints at /api/indices."""

    def _controller(self, tmp, option_source="synthetic"):
        config = AgentConfig(
            state_dir=tmp,
            broker=BrokerConfig(starting_cash=500000.0, slippage_bps=0.0),
            market=MarketConfig(symbols=["RELIANCE.NS"], interval="15m", bars=120),
            risk=RiskConfig(),
            data=DataConfig(source="synthetic", seed=21),
            indices=IndicesConfig(names=["NIFTY"], interval="5m", bars=120,
                                  option_source=option_source, refresh_seconds=0),
        )
        agent = TradingAgent(config, provider=SyntheticProvider(seed=21, bars=120),
                             llm=MockLLMClient())
        return WebController(config, agent, config_path=os.path.join(tmp, "config.json"))

    def test_indices_endpoint_reports_the_whole_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(tmp)
            board = controller.indices()
            self.assertEqual([entry["index"] for entry in board["indices"]], ["NIFTY"])
            entry = board["indices"][0]
            self.assertTrue(entry["options"]["available"])
            for key in ("trend", "levels", "range", "reversal", "options"):
                self.assertIn(key, entry)
            self.assertIn("max_pain", entry["options"])

    def test_option_chain_off_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = self._controller(tmp, option_source="off")
            board = controller.indices()
            self.assertFalse(board["indices"][0]["options"]["available"])
            self.assertIsNone(board["chain_source"])

    def test_page_links_the_new_api(self):
        self.assertIn("/api/indices", HTML_PAGE)
        self.assertIn("renderIndices", HTML_PAGE)
