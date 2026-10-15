"""Tests for the intraday indices board (trend, range, reversal, levels)."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from random import Random

from trading_agent import indices
from trading_agent.data import Bar, IST
from trading_agent.options import OptionChainError, SyntheticOptionChainProvider
from trading_agent.report import render_index_board

START = datetime(2026, 10, 13, 9, 15, tzinfo=IST)   # a Tuesday


def make_bars(days=3, per_day=75, start=24000.0, drift=0.0004, noise=0.0007, seed=11,
              volume=0, step_minutes=5):
    """Deterministic multi-session OHLC series (zero volume by default, like a feed)."""
    rng = Random(seed)
    bars = []
    price = start
    for day in range(days):
        moment = START + timedelta(days=day)
        for index in range(per_day):
            close = price * (1.0 + drift + rng.gauss(0.0, noise))
            open_ = bars[-1].close if bars else price
            high = max(open_, close) * (1.0 + abs(rng.gauss(0.0, noise / 2.5)))
            low = min(open_, close) * (1.0 - abs(rng.gauss(0.0, noise / 2.5)))
            bars.append(Bar(int(moment.timestamp()), open_, high, low, close, volume))
            price = close
            moment += timedelta(minutes=step_minutes)
    return bars


def walk(count, start, step, seed=5):
    """A tight, almost noiseless series - used for predictable trend labels."""
    rng = Random(seed)
    bars = []
    price = start
    moment = START
    for index in range(count):
        price += step
        open_ = bars[-1].close if bars else price - step
        bars.append(Bar(int(moment.timestamp()), open_, price + 0.5, price - 0.5, price,
                        rng.randint(1000, 2000)))
        moment += timedelta(minutes=5)
    return bars


class FakeBarProvider:
    def __init__(self, series=None, failures=()):
        self.series = series or {}
        self.failures = set(failures)
        self.calls = []

    def fetch(self, symbol, interval="5m", bars=200, range_=None):
        self.calls.append(symbol)
        if symbol in self.failures:
            raise RuntimeError(f"boom for {symbol}")
        return self.series.get(symbol, [])


class FakeChainProvider:
    name = "fake"

    def __init__(self, chains=None, failures=()):
        self.chains = chains or {}
        self.failures = set(failures)
        self.calls = []

    def fetch(self, index, expiry=None):
        self.calls.append(index)
        if index in self.failures:
            raise OptionChainError(f"chain down for {index}")
        return self.chains[index]


class IndexSpecTests(unittest.TestCase):
    def test_resolve_aliases(self):
        self.assertEqual(indices.resolve_index("nifty").key, "NIFTY")
        self.assertEqual(indices.resolve_index("NIFTY 50").key, "NIFTY")
        self.assertEqual(indices.resolve_index("banknifty").key, "BANKNIFTY")
        self.assertEqual(indices.resolve_index("NIFTY BANK").key, "BANKNIFTY")
        self.assertEqual(indices.resolve_index("SENSEX").key, "SENSEX")
        with self.assertRaises(ValueError):
            indices.resolve_index("NASDAQ")

    def test_resolve_list_defaults_to_all_three(self):
        self.assertEqual([spec.key for spec in indices.resolve_indices(None)],
                         ["NIFTY", "BANKNIFTY", "SENSEX"])
        specs = indices.resolve_indices("nifty, sensex nifty")
        self.assertEqual([spec.key for spec in specs], ["NIFTY", "SENSEX"])


class SessionTests(unittest.TestCase):
    def test_sessions_are_split_by_ist_day(self):
        bars = make_bars(days=3)
        days = indices.split_sessions(bars)
        self.assertEqual(list(days), ["2026-10-13", "2026-10-14", "2026-10-15"])
        self.assertEqual(len(days["2026-10-15"]), 75)
        self.assertEqual(len(indices.session_bars(bars)), 75)
        self.assertEqual(indices.previous_session(bars), days["2026-10-14"])

    def test_vwap_uses_volume_and_falls_back_to_typical_price(self):
        # 3 bars: typical prices 100, 200, 300 with volumes 1, 1, 2
        bars = [
            Bar(1, 100.0, 100.0, 100.0, 100.0, 1),
            Bar(2, 200.0, 200.0, 200.0, 200.0, 1),
            Bar(3, 300.0, 300.0, 300.0, 300.0, 2),
        ]
        self.assertAlmostEqual(indices.session_vwap(bars), (100 + 200 + 600) / 4.0)
        unweighted = [Bar(int((START + timedelta(minutes=5 * i)).timestamp()),
                          float(100 * (i + 1)), float(100 * (i + 1)), float(100 * (i + 1)),
                          float(100 * (i + 1)), 0) for i in range(3)]
        self.assertAlmostEqual(indices.session_vwap(unweighted), 200.0)
        self.assertIsNone(indices.session_vwap([]))


class LevelTests(unittest.TestCase):
    def test_pivot_and_cpr_math(self):
        bars = make_bars(days=2)
        prev = indices.previous_session(bars)
        pivots = indices.pivot_points(prev)
        high = max(b.high for b in prev)
        low = min(b.low for b in prev)
        close = prev[-1].close
        pp = (high + low + close) / 3.0
        self.assertAlmostEqual(pivots["pivot"], pp, places=2)
        self.assertAlmostEqual(pivots["r1"], 2 * pp - low, places=2)
        self.assertAlmostEqual(pivots["s1"], 2 * pp - high, places=2)
        self.assertAlmostEqual(pivots["r3"], high + 2 * (pp - low), places=2)
        self.assertAlmostEqual(pivots["cpr_bottom"], (high + low) / 2.0, places=2)
        self.assertAlmostEqual(pivots["cpr_top"], 2 * pp - pivots["cpr_bottom"], places=2)
        self.assertEqual(pivots["cpr_inverted"], pivots["cpr_top"] < pivots["cpr_bottom"])
        self.assertGreater(pivots["r3"], pivots["r2"])
        self.assertGreater(pivots["r2"], pivots["r1"])
        self.assertGreater(pivots["r1"], pivots["pivot"])
        self.assertGreater(pivots["pivot"], pivots["s1"])
        self.assertGreater(pivots["s1"], pivots["s2"])
        self.assertGreater(pivots["s2"], pivots["s3"])

    def test_opening_range_is_the_first_fifteen_minutes(self):
        session = indices.session_bars(make_bars(days=2))
        orb = indices.opening_range(session, 15)
        self.assertEqual(orb["minutes"], 15)
        self.assertEqual(len([b for b in session[:4]]), 4)
        self.assertAlmostEqual(orb["high"], max(b.high for b in session[:4]), places=2)
        self.assertAlmostEqual(orb["low"], min(b.low for b in session[:4]), places=2)
        self.assertEqual(orb["target_up"], round(orb["high"] + (orb["high"] - orb["low"]), 2))

    def test_levels_are_sorted_away_from_spot(self):
        entry = indices.analyze_index(indices.INDEX_SPECS["NIFTY"], make_bars(), "5m")
        levels = entry["levels"]
        spot = levels["spot"]
        self.assertTrue(all(level["price"] > spot for level in levels["resistance"]))
        self.assertTrue(all(level["price"] < spot for level in levels["support"]))
        resistances = [level["price"] for level in levels["resistance"]]
        supports = [level["price"] for level in levels["support"]]
        self.assertEqual(resistances, sorted(resistances))
        self.assertEqual(supports, sorted(supports, reverse=True))
        self.assertIn("day_high", [level["label"] for level in levels["resistance"]])
        self.assertIn("pivot_s1", [level["label"] for level in levels["support"]])


class TrendTests(unittest.TestCase):
    def test_uptrend_is_detected(self):
        bars = walk(120, 24000.0, 2.0, seed=3)
        trend = indices.trend_analysis(indices.session_bars(bars))
        self.assertIn(trend["label"], ("up", "strong_up"))
        self.assertGreater(trend["score"], 0.2)
        self.assertTrue(trend["above_vwap"])

    def test_downtrend_is_detected(self):
        bars = walk(120, 24000.0, -2.0, seed=4)
        trend = indices.trend_analysis(indices.session_bars(bars))
        self.assertIn(trend["label"], ("down", "strong_down"))
        self.assertLess(trend["score"], -0.2)
        self.assertFalse(trend["above_vwap"])

    def test_sideways_when_the_score_is_small(self):
        rng = Random(9)
        bars = []
        moment = START
        for index in range(120):
            close = 24000.0 + (5.0 if index % 2 else -5.0) + rng.gauss(0.0, 1.0)
            bars.append(Bar(int(moment.timestamp()), close, close + 1.0, close - 1.0, close, 1000))
            moment += timedelta(minutes=5)
        trend = indices.trend_analysis(bars)
        self.assertEqual(trend["label"], "sideways")
        self.assertEqual(trend["strength"], "weak")


class ReversalTests(unittest.TestCase):
    def test_overbought_parabolic_move_warns(self):
        bars = walk(90, 24000.0, 30.0, seed=7)   # +2700 points over 90 bars
        levels = indices.level_analysis(bars, bars[:75])
        reversal = indices.reversal_analysis(indices.session_bars(bars), levels)
        codes = {warning["code"] for warning in reversal["warnings"]}
        self.assertIn("RSI_OVERBOUGHT", codes)
        self.assertIn("STRETCH_FROM_OPEN", codes)
        self.assertIn("PARABOLIC_RUN", codes)
        self.assertEqual(reversal["bias"], "bearish")
        self.assertIn(reversal["risk"], ("medium", "high"))

    def test_calm_market_has_low_reversal_risk(self):
        rng = Random(21)
        bars = []
        moment = START
        for index in range(90):
            close = 24000.0 + (70.0 if index % 2 else -70.0) + rng.gauss(0.0, 5.0)
            open_ = close - (5.0 if index % 2 else -5.0)
            bars.append(Bar(int(moment.timestamp()), open_, close + 8.0, close - 8.0,
                            close, rng.randint(1000, 2000)))
            moment += timedelta(minutes=5)
        levels = indices.level_analysis(bars, bars[:75])
        reversal = indices.reversal_analysis(indices.session_bars(bars), levels)
        self.assertEqual(reversal["risk"], "low")
        self.assertEqual(reversal["warnings"], [])

    def test_option_warnings_come_from_the_chain(self):
        from trading_agent.options import analyze_chain
        from tests.test_options import make_chain

        chain = make_chain(spot=23000.0,
                           call_oi={23000: (5000, 500, 100, 120.0)},
                           put_oi={22975: (4000, 900, 200, 150.0)})
        levels = {"spot": 23000.0, "options": {"available": True, **analyze_chain(chain)}}
        warnings = indices._option_warnings(23000.0, levels, chain)
        codes = {warning["code"] for warning in warnings}
        self.assertIn("CE_WALL_AHEAD", codes)
        self.assertIn("PE_WALL_BELOW", codes)
        self.assertTrue(any("23000" in warning["message"] for warning in warnings))


class RangeTests(unittest.TestCase):
    def test_straddle_wins_and_the_range_brackets_spot(self):
        bars = make_bars()
        levels = indices.analyze_index(indices.INDEX_SPECS["NIFTY"], bars, "5m")["levels"]
        straddle = {"atm_strike": 24000.0, "straddle_points": 3000.0,
                    "straddle_pct": 12.5, "breakeven_low": 22500.0, "breakeven_high": 25500.0}
        block = indices.expected_range(bars, levels, straddle)
        self.assertEqual(block["basis"], "straddle (options-implied)")
        self.assertEqual(block["width"], 3000.0)
        self.assertLess(block["low"], levels["spot"])
        self.assertGreater(block["high"], levels["spot"])
        self.assertFalse(block["spot_outside"])

    def test_spot_outside_the_projected_range_is_flagged(self):
        bars = make_bars()
        levels = indices.analyze_index(indices.INDEX_SPECS["NIFTY"], bars, "5m")["levels"]
        block = indices.expected_range(bars, levels, {"straddle_points": 10.0})
        self.assertTrue(block["spot_outside"])
        self.assertLessEqual(block["low"], levels["spot"])
        self.assertGreaterEqual(block["high"], levels["spot"])

    def test_typical_session_range_is_used_without_a_chain(self):
        bars = make_bars(days=4)
        levels = indices.analyze_index(indices.INDEX_SPECS["NIFTY"], bars, "5m")["levels"]
        block = indices.expected_range(bars, levels)
        self.assertEqual(block["basis"], "median of the last sessions")
        self.assertIsNotNone(block["sources"]["typical_session_range"])
        self.assertIsNone(block["sources"]["straddle_points"])


class BoardTests(common := unittest.TestCase):
    def test_analyze_index_reports_every_block(self):
        spec = indices.INDEX_SPECS["BANKNIFTY"]
        spot = indices.session_bars(make_bars())[-1].close
        chain = SyntheticOptionChainProvider(seed=2, spot=spot).fetch("BANKNIFTY")
        entry = indices.analyze_index(spec, make_bars(), "5m", chain=chain)
        self.assertEqual(entry["index"], "BANKNIFTY")
        self.assertEqual(entry["lot_size"], 15)
        self.assertTrue(entry["options"]["available"])
        self.assertEqual(entry["options"]["source"], "synthetic")
        self.assertIsNotNone(entry["trend"]["label"])
        self.assertIsNotNone(entry["levels"]["cpr"])
        self.assertIn("max_pain", entry["options"])
        json.dumps(entry)  # must stay JSON-serialisable for the web API

    def test_one_failing_index_does_not_sink_the_board(self):
        provider = FakeBarProvider({
            "^NSEI": make_bars(),
        }, failures={"^NSEBANK"})
        chain_provider = FakeChainProvider(chains={}, failures={"NIFTY"})
        board = indices.build_index_board(provider=provider, chain_provider=chain_provider,
                                          interval="5m", bars=200)
        self.assertEqual(len(board["indices"]), 3)
        nifty = board["indices"][0]
        self.assertNotIn("error", nifty)
        self.assertFalse(nifty["options"]["available"])
        self.assertIn("chain down", nifty["options"]["reason"])
        self.assertIn("boom", board["indices"][1]["error"])
        self.assertIn("no bars", board["indices"][2]["error"])
        self.assertEqual(provider.calls, ["^NSEI", "^NSEBANK", "^BSESN"])
        self.assertEqual(chain_provider.calls, ["NIFTY"])   # no bars, no chain fetch

    def test_ladder_is_optional_in_the_json_output(self):
        bars = make_bars()
        spot = indices.session_bars(bars)[-1].close
        provider = FakeBarProvider({"^NSEI": bars})
        chain_provider = FakeChainProvider(
            {"NIFTY": SyntheticOptionChainProvider(seed=1, spot=spot).fetch("NIFTY")})
        compact = indices.build_index_board(specs=[indices.INDEX_SPECS["NIFTY"]],
                                            provider=provider, chain_provider=chain_provider)
        full = indices.build_index_board(specs=[indices.INDEX_SPECS["NIFTY"]],
                                         provider=provider, chain_provider=chain_provider,
                                         ladder_strikes=9)
        self.assertNotIn("ladder", compact["indices"][0]["options"])
        self.assertEqual(len(full["indices"][0]["options"]["ladder"]["ladder"]), 9)
        self.assertIn("atm", full["indices"][0]["options"]["ladder"]["ladder"][0])

    def test_board_without_a_chain_provider(self):
        provider = FakeBarProvider({"^NSEI": make_bars()})
        board = indices.build_index_board(specs=[indices.INDEX_SPECS["NIFTY"]],
                                          provider=provider)
        self.assertIsNone(board["chain_source"])
        self.assertFalse(board["indices"][0]["options"]["available"])

    def test_rendered_board_mentions_the_key_readings(self):
        provider = FakeBarProvider({"^NSEI": make_bars()})
        board = indices.build_index_board(specs=[indices.INDEX_SPECS["NIFTY"]],
                                          provider=provider)
        text = render_index_board(board)
        self.assertIn("NIFTY 50", text)
        self.assertIn("trend", text)
        self.assertIn("range", text)
        self.assertIn("reversal", text)
        self.assertIn("unavailable", text)

    def test_render_full_synthetic_chain_does_not_raise(self):
        bars = make_bars()
        spot = indices.session_bars(bars)[-1].close
        chain = SyntheticOptionChainProvider(seed=6, spot=spot).fetch("NIFTY")
        entry = indices.analyze_index(indices.INDEX_SPECS["NIFTY"], bars, "5m", chain=chain)
        text = render_index_board({"session_day": "x", "generated_at": "y", "market_open": True,
                                   "interval": "5m", "indices": [entry], "chain_source": "synthetic"})
        self.assertIn("max pain", text)
        self.assertIn("bulk", text)
        self.assertIn("max buying side", text)


if __name__ == "__main__":
    unittest.main()


class CliTests(unittest.TestCase):
    """The command must run offline and never raise, even with --json."""

    def test_synthetic_indices_run(self):
        import contextlib
        import io
        import json as jsonlib

        from trading_agent.__main__ import main
        from trading_agent.config import write_default_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            write_default_config(config_path)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = main([
                    "indices", "--config", config_path, "--json",
                    "--source", "synthetic", "--option-source", "synthetic",
                    "--bars", "120", "--names", "NIFTY,SENSEX",
                ])
            self.assertEqual(status, 0)
            payload = jsonlib.loads(buffer.getvalue())
            self.assertEqual([entry["index"] for entry in payload["indices"]], ["NIFTY", "SENSEX"])
            self.assertTrue(all(entry["options"]["available"] for entry in payload["indices"]))

    def test_text_render_run(self):
        import contextlib
        import io

        from trading_agent.__main__ import main
        from trading_agent.config import write_default_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            write_default_config(config_path)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = main([
                    "indices", "--config", config_path,
                    "--source", "synthetic", "--option-source", "off",
                    "--bars", "120", "--names", "BANKNIFTY",
                ])
            self.assertEqual(status, 0)
            self.assertIn("NIFTY BANK", buffer.getvalue())
            self.assertIn("unavailable", buffer.getvalue())

    def test_unknown_index_is_rejected(self):
        import contextlib
        import io

        from trading_agent.__main__ import main
        from trading_agent.config import write_default_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            write_default_config(config_path)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = main([
                    "indices", "--config", config_path,
                    "--names", "NIFTY,NASDAQ", "--source", "synthetic",
                ])
            self.assertEqual(status, 1)
            self.assertIn("unknown index", err.getvalue() + out.getvalue())


if __name__ == "__main__":
    unittest.main()
