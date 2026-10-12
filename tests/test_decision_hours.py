"""Tests for trade-decision parsing, market hours and prompt building."""
import unittest
from datetime import datetime, timedelta
from unittest import mock

from trading_agent.data import IST
from trading_agent.decision import normalise_stops, parse_decision_text
from trading_agent.llm import build_user_prompt
from trading_agent.market_hours import (
    business_days_between,
    is_data_fresh,
    is_market_open,
    next_session_open,
)

VALID_JSON = '{"action": "BUY", "quantity": 10, "confidence": 0.8, "stop_loss": 1480.0, "take_profit": 1600.0, "reasoning": "test"}'


class DecisionParsingTests(unittest.TestCase):
    def test_plain_json(self):
        decision, error = parse_decision_text(VALID_JSON)
        self.assertEqual(error, "")
        self.assertEqual(decision.action, "BUY")
        self.assertEqual(decision.quantity, 10)
        self.assertEqual(decision.stop_loss, 1480.0)

    def test_fenced_json_with_prose(self):
        text = "Here is my decision:\n```json\n" + VALID_JSON + '\n```\nThanks!'
        decision, error = parse_decision_text(text)
        self.assertEqual(error, "")
        self.assertEqual(decision.action, "BUY")

    def test_json_embedded_in_prose(self):
        text = "I think we should act. " + VALID_JSON + " Let me know."
        decision, error = parse_decision_text(text)
        self.assertEqual(error, "")
        self.assertEqual(decision.action, "BUY")

    def test_case_and_type_coercion(self):
        text = '{"action": "hold", "quantity": "5", "confidence": 1.7, "stop_loss": null, "take_profit": null, "reasoning": "x"}'
        decision, error = parse_decision_text(text)
        self.assertEqual(error, "")
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.quantity, 5)
        self.assertEqual(decision.confidence, 1.0)

    def test_rejects_garbage_and_bad_action(self):
        self.assertIsNone(parse_decision_text("no json here")[0])
        decision, error = parse_decision_text('{"action": "YOLO", "quantity": 1}')
        self.assertIsNone(decision)
        self.assertIn("unknown action", error)
        self.assertIsNone(parse_decision_text("")[0])

    def test_normalise_stops_repairs_levels(self):
        decision, _ = parse_decision_text('{"action": "BUY", "quantity": 1, "stop_loss": 2000, "take_profit": 100}')
        fixed = normalise_stops(decision, price=1500.0, ref_stop=1450.0, ref_target=1600.0)
        self.assertEqual(fixed.stop_loss, 1450.0)
        self.assertEqual(fixed.take_profit, 1600.0)

    def test_normalise_stops_keeps_valid_levels(self):
        decision, _ = parse_decision_text(VALID_JSON.replace("BUY", "BUY"))
        fixed = normalise_stops(decision, price=1500.0, ref_stop=1000.0, ref_target=2000.0)
        self.assertEqual(fixed.stop_loss, 1480.0)
        self.assertEqual(fixed.take_profit, 1600.0)


class MarketHoursTests(unittest.TestCase):
    def test_session_bounds(self):
        monday_930 = datetime(2026, 10, 12, 9, 30, tzinfo=IST)
        monday_915 = datetime(2026, 10, 12, 9, 15, tzinfo=IST)
        monday_1531 = datetime(2026, 10, 12, 15, 31, tzinfo=IST)
        saturday = datetime(2026, 10, 17, 10, 0, tzinfo=IST)
        pre_open = datetime(2026, 10, 12, 9, 0, tzinfo=IST)
        holiday = datetime(2026, 10, 14, 10, 0, tzinfo=IST)
        self.assertTrue(is_market_open(monday_930))
        self.assertTrue(is_market_open(monday_915))
        self.assertFalse(is_market_open(monday_1531))
        self.assertFalse(is_market_open(pre_open))
        self.assertFalse(is_market_open(saturday))
        self.assertFalse(is_market_open(holiday, ["2026-10-14"]))

    def test_next_session_open(self):
        friday_night = datetime(2026, 10, 16, 18, 0, tzinfo=IST)
        self.assertEqual(next_session_open(friday_night).date().isoformat(), "2026-10-19")
        self.assertEqual(next_session_open(friday_night).time().hour, 9)

    def test_business_days(self):
        friday = datetime(2026, 10, 16, tzinfo=IST).date()
        monday = datetime(2026, 10, 19, tzinfo=IST).date()
        self.assertEqual(business_days_between(friday, monday), 2)
        self.assertEqual(business_days_between(monday, friday), 0)

    def test_data_freshness(self):
        now = datetime(2026, 10, 12, 12, 0, tzinfo=IST)
        recent = int((now - timedelta(minutes=10)).timestamp())
        old = int((now - timedelta(minutes=90)).timestamp())
        self.assertTrue(is_data_fresh(recent, "5m", now)[0])
        self.assertFalse(is_data_fresh(old, "5m", now)[0])

        monday_open = int(datetime(2026, 10, 12, 9, 15, tzinfo=IST).timestamp())
        friday_close = int(datetime(2026, 10, 9, 15, 29, tzinfo=IST).timestamp())
        self.assertTrue(is_data_fresh(friday_close, "1d", monday_open)[0])
        stale_daily = int(datetime(2026, 10, 8, 15, 29, tzinfo=IST).timestamp())
        self.assertFalse(is_data_fresh(stale_daily, "1d", monday_open)[0])


class PromptTests(unittest.TestCase):
    def test_user_prompt_includes_snapshot(self):
        snapshot = {
            "symbol": "RELIANCE.NS",
            "timeframe": "5m",
            "last_price": 1500.0,
            "rsi_14": 55.0,
            "portfolio": {
                "cash": 100000.0,
                "equity": 250000.0,
                "open_positions": 2,
                "day_pnl_pct": -0.5,
                "position_qty": 0,
                "position_avg_price": None,
            },
        }
        prompt = build_user_prompt(snapshot)
        self.assertIn("RELIANCE.NS", prompt)
        self.assertIn("1500.0", prompt)
        self.assertIn("rsi_14", prompt)
        self.assertIn("250,000.00", prompt)


if __name__ == "__main__":
    unittest.main()
