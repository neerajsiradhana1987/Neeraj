"""Tests for the OpenAI-compatible LLM client (network mocked out)."""
import io
import json
import unittest
from unittest import mock

from trading_agent.config import LLMConfig
from trading_agent.llm import LLMClient


def _http_response(payload: dict):
    body = json.dumps(payload).encode("utf-8")
    response = io.BytesIO(body)
    response.read = mock.Mock(return_value=body)
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *args: False
    return response


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.config = LLMConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            temperature=0.2,
            max_tokens=100,
            timeout_seconds=5,
        )
        self.snapshot = {
            "symbol": "RELIANCE.NS",
            "timeframe": "5m",
            "last_price": 1500.0,
            "rsi_14": 55.0,
            "atr_14": 12.0,
            "portfolio": {"cash": 100000.0, "equity": 200000.0, "open_positions": 0,
                          "day_pnl_pct": None, "position_qty": 0},
            "_ref_stop": 1476.0,
            "_ref_target": 1548.0,
        }

    def test_successful_decision_is_normalised(self):
        payload = {
            "choices": [
                {"message": {"content": json.dumps({
                    "action": "buy",
                    "quantity": "12",
                    "confidence": 0.82,
                    "stop_loss": 1480.0,
                    "take_profit": 1600.0,
                    "reasoning": "Trend and MACD aligned.",
                })}}
            ]
        }
        with mock.patch("urllib.request.urlopen", return_value=_http_response(payload)) as opener:
            decision, error, latency = LLMClient(self.config).decide(self.snapshot, price=1500.0)
        self.assertEqual(error, "")
        self.assertEqual(decision.action, "BUY")
        self.assertEqual(decision.quantity, 12)
        self.assertEqual(decision.stop_loss, 1480.0)
        self.assertGreaterEqual(latency, 0.0)
        request = opener.call_args[0][0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertIn("/chat/completions", request.full_url)
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["model"], "test-model")
        self.assertEqual(sent["messages"][0]["role"], "system")

    def test_invalid_levels_fall_back_to_reference(self):
        content = json.dumps({"action": "BUY", "quantity": 5, "stop_loss": 9999, "take_profit": 1})
        payload = {"choices": [{"message": {"content": content}}]}
        with mock.patch("urllib.request.urlopen", return_value=_http_response(payload)):
            decision, error, _ = LLMClient(self.config).decide(self.snapshot, price=1500.0)
        self.assertEqual(error, "")
        self.assertEqual(decision.stop_loss, 1476.0)
        self.assertEqual(decision.take_profit, 1548.0)

    def test_network_error_is_reported(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("boom")):
            decision, error, _ = LLMClient(self.config).decide(self.snapshot, price=1500.0)
        self.assertIsNone(decision)
        self.assertIn("llm_error", error)

    def test_unparseable_content_is_reported(self):
        payload = {"choices": [{"message": {"content": "I am not trading today."}}]}
        with mock.patch("urllib.request.urlopen", return_value=_http_response(payload)):
            decision, error, _ = LLMClient(self.config).decide(self.snapshot, price=1500.0)
        self.assertIsNone(decision)
        self.assertIn("no JSON object", error)

    def test_agent_uses_llm_when_key_present(self):
        import tempfile

        from trading_agent.agent import TradingAgent
        from trading_agent.config import AgentConfig, BrokerConfig, MarketConfig
        from trading_agent.data import SyntheticProvider

        content = json.dumps({"action": "HOLD", "quantity": 0, "confidence": 0.4,
                              "stop_loss": None, "take_profit": None, "reasoning": "wait"})
        payload = {"choices": [{"message": {"content": content}}]}
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(
                state_dir=tmp,
                broker=BrokerConfig(starting_cash=100000.0),
                market=MarketConfig(symbols=["RELIANCE.NS"], interval="15m", bars=120),
            )
            config.llm.api_key = "test-key"
            agent = TradingAgent(config, provider=SyntheticProvider(seed=11, bars=120))
            self.assertFalse(agent.using_mock)
            with mock.patch("urllib.request.urlopen", return_value=_http_response(payload)):
                report = agent.run_cycle(force=True)
            self.assertEqual(report.positions, 0)
            decisions = agent.store.read_events(event="DECISION")
            self.assertEqual(decisions[-1]["action"], "HOLD")


if __name__ == "__main__":
    unittest.main()
