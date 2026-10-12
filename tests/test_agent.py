"""Tests for the agent loop, broker accounting and risk limits."""
import json
import tempfile
import unittest
from pathlib import Path

from trading_agent.agent import TradingAgent
from trading_agent.broker import ChargeModel, PaperBroker, compute_charges
from trading_agent.config import AgentConfig
from trading_agent.data import Bar, CSVProvider, SyntheticProvider, range_for_interval
from trading_agent.llm import MockLLMClient
from trading_agent.risk import RiskConfig, RiskManager


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.model = ChargeModel()

    def test_buy_debits_cash_and_charges(self):
        broker = PaperBroker(cash=100000.0, charge_model=self.model, slippage_bps=0.0)
        result = broker.market_order("RELIANCE.NS", "BUY", 10, 1000.0)
        self.assertTrue(result.ok)
        charges = compute_charges(self.model, "BUY", 10000.0).total
        self.assertAlmostEqual(broker.cash, 100000.0 - 10000.0 - charges, places=2)
        position = broker.positions["RELIANCE.NS"]
        self.assertEqual(position.qty, 10)
        self.assertGreater(position.avg_price, 1000.0)  # charges are in the cost basis

    def test_slippage_is_adverse(self):
        broker = PaperBroker(cash=1000000.0, charge_model=self.model, slippage_bps=10.0)
        result = broker.market_order("TCS.NS", "BUY", 1, 1000.0)
        self.assertAlmostEqual(result.fill.price, 1001.0, places=4)
        sell = broker.market_order("TCS.NS", "SELL", 1, 1000.0)
        self.assertAlmostEqual(sell.fill.price, 999.0, places=4)

    def test_insufficient_cash_rejected(self):
        broker = PaperBroker(cash=5000.0, charge_model=self.model)
        result = broker.market_order("INFY.NS", "BUY", 10, 1000.0)
        self.assertFalse(result.ok)
        self.assertIn("insufficient cash", result.reason)
        self.assertEqual(broker.cash, 5000.0)

    def test_sell_without_position_rejected(self):
        broker = PaperBroker(cash=100000.0, charge_model=self.model)
        result = broker.market_order("INFY.NS", "SELL", 5, 100.0)
        self.assertFalse(result.ok)
        self.assertIn("insufficient position", result.reason)

    def test_round_trip_realized_pnl(self):
        broker = PaperBroker(cash=200000.0, charge_model=self.model, slippage_bps=0.0)
        broker.market_order("HDFCBANK.NS", "BUY", 10, 1500.0, reason="test")
        avg = broker.positions["HDFCBANK.NS"].avg_price
        result = broker.market_order("HDFCBANK.NS", "SELL", 10, 1600.0, reason="test")
        self.assertTrue(result.ok)
        expected = 10 * (1600.0 - avg) - compute_charges(self.model, "SELL", 16000.0).total
        self.assertAlmostEqual(broker.realized_pnl, expected, places=1)
        self.assertGreater(broker.realized_pnl, 0)
        self.assertNotIn("HDFCBANK.NS", broker.positions)

    def test_equity_and_unrealized(self):
        broker = PaperBroker(cash=50000.0, charge_model=self.model, slippage_bps=0.0)
        broker.market_order("ITC.NS", "BUY", 10, 400.0)
        prices = {"ITC.NS": 440.0}
        self.assertAlmostEqual(broker.equity(prices), 50000.0 - 4000.0 - broker.total_charges + 4400.0, places=2)
        self.assertAlmostEqual(broker.unrealized_pnl(prices), 400.0 - broker.total_charges, places=2)

    def test_state_round_trip(self):
        broker = PaperBroker(cash=75000.0, charge_model=self.model, slippage_bps=0.0)
        broker.market_order("SBIN.NS", "BUY", 5, 600.0, stop_loss=560.0, take_profit=700.0)
        restored = PaperBroker.from_dict(
            broker.to_dict(), charge_model=self.model, slippage_bps=0.0, default_cash=1.0
        )
        self.assertEqual(restored.cash, broker.cash)
        self.assertEqual(restored.positions["SBIN.NS"].qty, 5)
        self.assertEqual(restored.positions["SBIN.NS"].stop_loss, 560.0)


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.config = RiskConfig()
        self.risk = RiskManager(self.config, ChargeModel(), slippage_bps=0.0)

    def test_buy_limited_by_position_cap(self):
        equity, cash, price = 1_000_000.0, 1_000_000.0, 1000.0
        verdict = self.risk.approve_buy("RELIANCE.NS", 10_000, price, equity, cash, open_positions=0)
        self.assertTrue(verdict.allowed)
        self.assertEqual(verdict.quantity, int(equity * self.config.max_position_pct // price))

    def test_buy_limited_by_cash(self):
        equity, cash, price = 1_000_000.0, 5_000.0, 1000.0
        verdict = self.risk.approve_buy("TCS.NS", 100, price, equity, cash, open_positions=0)
        self.assertFalse(verdict.allowed)
        self.assertIn("rounded to 0", verdict.reason)

    def test_buy_limited_by_stop_distance(self):
        equity, cash, price = 1_000_000.0, 1_000_000.0, 1000.0
        verdict = self.risk.approve_buy(
            "INFY.NS", 5000, price, equity, cash, open_positions=0, stop_loss=900.0
        )
        expected = int(equity * self.config.risk_per_trade_pct // (price - 900.0))
        self.assertEqual(verdict.quantity, min(5000, expected))

    def test_halt_and_cooldown(self):
        self.assertTrue(self.risk.check_halt(100.0, 95.0)[0])
        self.assertFalse(self.risk.check_halt(100.0, 99.0)[0])
        self.risk.note_stop_out("SBIN.NS", 0)
        self.assertTrue(self.risk.in_cooldown("SBIN.NS", 60))
        self.assertFalse(self.risk.in_cooldown("SBIN.NS", 60 + 61 * 60))

    def test_sell_approval_caps_at_position(self):
        verdict = self.risk.approve_sell("ITC.NS", 100, 7)
        self.assertTrue(verdict.allowed)
        self.assertEqual(verdict.quantity, 7)
        self.assertTrue(self.risk.approve_sell("ITC.NS", 0, 7).allowed)
        self.assertFalse(self.risk.approve_sell("ITC.NS", 5, 0).allowed)


class DataTests(unittest.TestCase):
    def test_range_selection(self):
        self.assertEqual(range_for_interval("5m", 75), "1d")
        self.assertEqual(range_for_interval("1d", 250), "1y")
        self.assertEqual(range_for_interval("1m", 5000), "7d")

    def test_synthetic_is_deterministic_and_sane(self):
        first = SyntheticProvider(seed=99, bars=120).fetch("RELIANCE.NS", interval="15m", bars=120)
        again = SyntheticProvider(seed=99, bars=120).fetch("RELIANCE.NS", interval="15m", bars=120)
        self.assertEqual([bar.to_dict() for bar in first], [bar.to_dict() for bar in again])
        for bar in first:
            self.assertGreater(bar.close, 0)
            self.assertGreaterEqual(bar.high, bar.low)
            self.assertLessEqual(bar.low, bar.open)
            self.assertLessEqual(bar.high, bar.open * 1.5)

    def test_synthetic_evolves_between_calls(self):
        provider = SyntheticProvider(seed=5, bars=60)
        first = provider.fetch("TCS.NS", interval="15m", bars=60)
        second = provider.fetch("TCS.NS", interval="15m", bars=60)
        self.assertEqual(len(second), 60)
        self.assertNotEqual(first[-1].ts, second[-1].ts)
        self.assertEqual(second[-2].ts, first[-1].ts)      # one new bar per call
        self.assertEqual(second[-2].close, first[-1].close)

    def test_yahoo_parsing(self):
        import trading_agent.data as data_module

        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1700000000 + 60 * i for i in range(3)],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10.0, 11.0, 12.0],
                                    "high": [11.0, 12.0, 13.0],
                                    "low": [9.5, 10.5, 11.5],
                                    "close": [10.5, 11.5, None],
                                    "volume": [100, 200, 300],
                                }
                            ]
                        },
                    }
                ]
            }
        }

        class StubYahoo(data_module.YahooProvider):
            def __init__(self):
                super().__init__()
                self.urls = []

            def _get_json(self, url):
                self.urls.append(url)
                return payload

        provider = StubYahoo()
        bars = provider.fetch("RELIANCE.NS", interval="5m", bars=10)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[-1].close, 11.5)
        self.assertEqual(provider.urls[0].split("?")[1].split("&")[0], "range=1d")
        for url in provider.urls:
            self.assertNotIn("<class", url)

    def test_csv_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "TCS.NS.csv"
            path.write_text(
                "ts,open,high,low,close,volume\n"
                "1700000000,100,110,95,105,1000\n"
                "1700000300,105,115,100,112,1500\n",
                encoding="utf-8",
            )
            bars = CSVProvider(tmp).fetch("TCS.NS", interval="5m", bars=10)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[-1].close, 112.0)


class AgentTests(unittest.TestCase):
    def test_full_cycle_with_mock_llm(self):
        from trading_agent.config import AgentConfig, BrokerConfig, MarketConfig, RiskConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(
                state_dir=tmp,
                broker=BrokerConfig(starting_cash=500000.0, slippage_bps=0.0),
                market=MarketConfig(
                    symbols=["RELIANCE.NS", "TCS.NS"],
                    interval="15m",
                    bars=120,
                    session_only=True,
                    holidays=[],
                ),
                risk=RiskConfig(),
            )
            from trading_agent.data import SyntheticProvider

            provider = SyntheticProvider(seed=7, bars=120)
            agent = TradingAgent(config, provider=provider, llm=MockLLMClient())
            for _ in range(12):
                report = agent.run_cycle(force=True)
                self.assertIsNotNone(report)

            status = agent.status()
            self.assertGreater(status["equity"], 0)
            self.assertAlmostEqual(
                status["cash"] + status["invested"], status["equity"], places=1
            )
            events = agent.store.read_events()
            self.assertTrue(events)
            self.assertTrue(any(e["event"] == "DECISION" for e in events))

            fills = agent.trades(limit=50)
            for fill in fills:
                self.assertIn(fill["side"], ("BUY", "SELL"))
                self.assertGreater(fill["qty"], 0)
                for key in ("ts", "symbol", "price", "value", "charges"):
                    self.assertIn(key, fill)

            # persist across instances
            reopened = TradingAgent(config, provider=provider, llm=MockLLMClient())
            self.assertAlmostEqual(reopened.status()["equity"], status["equity"], places=6)

    def test_risk_limits_hold_up(self):
        from trading_agent.config import AgentConfig, BrokerConfig, MarketConfig, RiskConfig
        from trading_agent.data import SyntheticProvider

        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(
                state_dir=tmp,
                broker=BrokerConfig(starting_cash=50000.0, slippage_bps=0.0),
                market=MarketConfig(symbols=["RELIANCE.NS"], interval="15m", bars=120),
                risk=RiskConfig(max_position_pct=0.2, min_cash_pct=0.5),
            )
            agent = TradingAgent(config, provider=SyntheticProvider(seed=3, bars=120), llm=MockLLMClient())
            for _ in range(8):
                agent.run_cycle(force=True)
            status = agent.status()
            for position in status["positions"]:
                self.assertLessEqual(position["market_value"], status["equity"] * 0.2 + 1)
            self.assertGreaterEqual(status["cash"], status["equity"] * 0.5 - 1)

    def test_symbols_are_suffixed(self):
        from trading_agent.config import AgentConfig

        agent = TradingAgent.__new__(TradingAgent)
        agent.config = AgentConfig()
        self.assertEqual(agent.symbols(["reliance", "TCS.NS"]), ["RELIANCE.NS", "TCS.NS"])


class JournalTests(unittest.TestCase):
    def test_read_events_filtering(self):
        from trading_agent.storage import StateStore

        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp)
            store.journal("DECISION", symbol="RELIANCE.NS", action="BUY")
            store.journal("FILL", symbol="RELIANCE.NS", side="BUY")
            store.journal("FILL", symbol="TCS.NS", side="SELL")
            self.assertEqual(len(store.read_events()), 3)
            self.assertEqual(len(store.read_events(event="FILL")), 2)
            self.assertEqual(len(store.read_events(event="FILL", limit=1)), 1)


if __name__ == "__main__":
    unittest.main()
