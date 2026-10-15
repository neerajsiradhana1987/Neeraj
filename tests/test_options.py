"""Tests for the option-chain providers and the derived analysis."""
import json
import unittest
from datetime import datetime, timedelta

from trading_agent import options
from trading_agent.data import IST
from trading_agent.options import (
    OptionChain,
    OptionChainError,
    OptionLeg,
    SyntheticOptionChainProvider,
    analyze_chain,
    build_chain_provider,
    chain_from_nse_payload,
    max_pain,
    pain_at,
)


def make_chain(spot=23000.0, call_oi=None, put_oi=None, source="test") -> OptionChain:
    """Build a chain from explicit OI maps, volumes and prices.

    ``call_oi``/``put_oi`` are ``{strike: (oi, oi_change, volume, ltp)}`` maps.
    """
    call_oi = call_oi or {}
    put_oi = put_oi or {}
    legs = []
    strikes = sorted(set(call_oi) | set(put_oi))
    for strike in strikes:
        for side, book in (("CE", call_oi), ("PE", put_oi)):
            oi, change, volume, ltp = book.get(strike, (0, 0, 0, 0.0))
            legs.append(OptionLeg(strike=float(strike), side=side, oi=oi, oi_change=change,
                                  volume=volume, ltp=ltp))
    return OptionChain(index="NIFTY", spot=spot, expiry="16-Oct-2026", expiries=("16-Oct-2026",),
                       legs=tuple(legs), as_of="2026-10-15 15:30:00", source=source)


class SyntheticProviderTests(unittest.TestCase):
    def test_deterministic_and_shaped_around_spot(self):
        first = SyntheticOptionChainProvider(seed=9, spot=23000.0, strike_step=50.0).fetch("NIFTY")
        again = SyntheticOptionChainProvider(seed=9, spot=23000.0, strike_step=50.0).fetch("NIFTY")
        self.assertEqual([leg.to_dict() for leg in first.legs],
                         [leg.to_dict() for leg in again.legs])
        self.assertEqual(first.spot, 23000.0)
        self.assertEqual(first.step, 50.0)
        self.assertEqual(first.atm_strike(), 23000.0)
        self.assertGreater(len(first.legs), 20)
        self.assertTrue(all(leg.oi >= 0 for leg in first.legs))

    def test_spot_provider_is_used_when_no_spot(self):
        provider = SyntheticOptionChainProvider(spot_provider=lambda index: 55849.25)
        chain = provider.fetch("BANKNIFTY")
        self.assertEqual(chain.spot, 55849.25)
        self.assertEqual(chain.step, 100.0)

    def test_guess_step_for_unknown_index(self):
        provider = SyntheticOptionChainProvider(spot=1234.0)
        chain = provider.fetch("MIDCPNIFTY")
        self.assertEqual(chain.step, 25.0)


class ModelHelperTests(unittest.TestCase):
    def test_chain_helpers(self):
        chain = make_chain(call_oi={22900: (0, 0, 0, 0), 23000: (10, 0, 0, 0)},
                           put_oi={22900: (5, 0, 0, 0)})
        self.assertEqual(sorted(chain.strikes), [22900.0, 23000.0])
        self.assertEqual([leg.side for leg in chain.side("CE")], ["CE", "CE"])
        self.assertEqual(chain.atm_strike(), 23000.0)
        ladder = chain.legs_near_spot(2)
        self.assertTrue(all(leg.strike in (22900.0, 23000.0) for leg in ladder))
        payload = chain.to_dict(count=2, lot_size=25)
        self.assertEqual(len(payload["ladder"]), 2)
        self.assertTrue(any(row["atm"] for row in payload["ladder"]))
        self.assertEqual(payload["lot_size"], 25)


class PainTests(unittest.TestCase):
    def test_pain_split_and_max_pain_strike(self):
        chain = make_chain(
            call_oi={22900: (500, 0, 0, 0), 23000: (1000, 0, 0, 0), 23100: (1500, 0, 0, 0)},
            put_oi={22800: (1000, 0, 0, 0), 22900: (1500, 0, 0, 0), 23000: (1000, 0, 0, 0)},
        )
        split = pain_at(chain, 23000.0)
        self.assertEqual(split["call_loss"], 50000.0)     # 500 OI * 100 points ITM
        self.assertEqual(split["put_loss"], 0.0)
        self.assertEqual(split["total_loss"], 50000.0)

        result = max_pain(chain)
        self.assertTrue(result["available"])
        self.assertEqual(result["strike"], 23000.0)
        self.assertIsNotNone(result["profile"])
        self.assertGreaterEqual(result["distance_pct"], 0)

    def test_max_pain_moves_with_the_oi_cluster(self):
        heavy_calls = make_chain(
            call_oi={23500: (5000, 0, 0, 0)},
            put_oi={22500: (5000, 0, 0, 0)},
        )
        result = max_pain(heavy_calls)
        # every settlement below 22500 keeps the puts worthless, so pain is flat
        # there and the reported strike is the lowest candidate with zero loss.
        self.assertTrue(result["available"])
        self.assertLessEqual(result["strike"], 22500.0)

    def test_max_pain_without_open_interest(self):
        empty = make_chain(call_oi={23000: (0, 0, 0, 0)}, put_oi={23000: (0, 0, 0, 0)})
        result = max_pain(empty)
        self.assertFalse(result["available"])


class WallAndFlowTests(unittest.TestCase):
    def build(self) -> OptionChain:
        return make_chain(
            call_oi={23200: (4000, 500, 1000, 90.0), 23300: (3000, 100, 500, 40.0)},
            put_oi={22800: (6000, 700, 2000, 120.0), 22700: (2000, 100, 400, 200.0)},
        )

    def test_oi_walls_rank_by_open_interest(self):
        analysis = analyze_chain(self.build(), lot_size=25)
        self.assertEqual(analysis["walls"]["put_support"][0]["strike"], 22800.0)
        self.assertEqual(analysis["walls"]["call_resistance"][0]["strike"], 23200.0)
        self.assertEqual(analysis["walls"]["put_support"][0]["oi_lots"], 240)

    def test_buildup_uses_open_interest_change(self):
        analysis = analyze_chain(self.build())
        self.assertEqual(analysis["buildup"]["put_buildup"][0]["strike"], 22800.0)
        self.assertEqual(analysis["buildup"]["call_buildup"][0]["strike"], 23200.0)

    def test_pcr_and_bias(self):
        analysis = analyze_chain(self.build())
        pcr = analysis["pcr"]
        self.assertEqual(pcr["put_oi"], 8000)
        self.assertEqual(pcr["call_oi"], 7000)
        self.assertAlmostEqual(pcr["oi_pcr"], 8000 / 7000, places=3)
        self.assertAlmostEqual(pcr["volume_pcr"], 2400 / 1500, places=3)
        self.assertEqual(pcr["bias"], "put-heavy (supportive)")

    def test_buying_pressure_picks_the_larger_premium_turnover(self):
        analysis = analyze_chain(self.build())
        pressure = analysis["buying_pressure"]
        # PE turnover = 2000*120 + 400*200 = 320000, CE turnover = 1000*90 + 500*40 = 110000
        self.assertEqual(pressure["put_turnover"], 320000.0)
        self.assertEqual(pressure["call_turnover"], 110000.0)
        self.assertEqual(pressure["max_buying_side"], "PE")
        self.assertEqual(pressure["max_buying_pct"], 74.4)
        self.assertIn("PE side", pressure["flow"])

    def test_flow_labels_split_buying_and_writing(self):
        chain = make_chain(
            call_oi={23200: (4000, 900, 5000, 90.0)},
            put_oi={22800: (6000, 5000, 1000, 120.0)},
        )
        pressure = analyze_chain(chain)["buying_pressure"]
        self.assertEqual(pressure["put_turnover"], 120000.0)
        self.assertEqual(pressure["call_turnover"], 450000.0)
        self.assertEqual(pressure["max_buying_side"], "CE")
        self.assertEqual(pressure["oi_change_side"], "PE")
        self.assertEqual(pressure["flow"], "heavy CE buying against PE writing")

    def test_bulk_strikes_rank_by_turnover(self):
        analysis = analyze_chain(self.build(), lot_size=25)
        self.assertEqual(analysis["bulk_strikes"]["PE"][0]["strike"], 22800.0)
        self.assertEqual(analysis["bulk_strikes"]["CE"][0]["strike"], 23200.0)
        self.assertLess(analysis["bulk_strikes"]["PE"][1]["turnover"],
                        analysis["bulk_strikes"]["PE"][0]["turnover"])

    def test_straddle_from_the_atm_strike(self):
        chain = make_chain(
            call_oi={23000: (100, 0, 0, 150.0)},
            put_oi={23000: (100, 0, 0, 120.0), 22900: (50, 0, 0, 200.0)},
        )
        straddle = analyze_chain(chain)["straddle"]
        self.assertEqual(straddle["atm_strike"], 23000.0)
        self.assertEqual(straddle["straddle_points"], 270.0)
        self.assertEqual(straddle["breakeven_low"], 22730.0)
        self.assertEqual(straddle["breakeven_high"], 23270.0)
        self.assertAlmostEqual(straddle["straddle_pct"], 270.0 / 23000.0 * 100.0, places=2)


class NSEPayloadTests(unittest.TestCase):
    def payload(self, days_ahead=2):
        today = datetime.now(IST).date()
        near = (today + timedelta(days=days_ahead)).strftime("%d-%b-%Y")
        far = (today + timedelta(days=days_ahead + 7)).strftime("%d-%b-%Y")

        def legs(oi, change, volume, ltp):
            return {"openInterest": oi, "changeinOpenInterest": change,
                    "totalTradedVolume": volume, "impliedVolatility": 13.5,
                    "lastPrice": ltp}

        return {
            "records": {
                "expiryDates": [far, near],
                "underlyingValue": 23000.35,
                "timestamp": "15-Oct-2026 15:30:00",
                "data": [
                    {"strikePrice": 22900, "expiryDate": near,
                     "CE": legs(1000, 120, 900, 100.05),
                     "PE": legs(2000, -300, 700, 210.10)},
                    {"strikePrice": 23000, "expiryDate": near,
                     "CE": legs(2500, 400, 1000, 60.05),
                     "PE": legs(1800, 250, 800, 160.25)},
                    {"strikePrice": 22900, "expiryDate": far,
                     "CE": legs(9999, 0, 0, 1.0), "PE": legs(9999, 0, 0, 1.0)},
                ],
            }
        }

    def test_parsing_picks_the_nearest_expiry_and_both_sides(self):
        chain = chain_from_nse_payload("NIFTY", self.payload())
        self.assertEqual(chain.spot, 23000.35)
        self.assertEqual(len(chain.legs), 4)
        self.assertEqual([leg.strike for leg in chain.legs if leg.side == "CE"][0], 22900.0)
        self.assertEqual(chain.side("PE")[1].volume, 800)
        self.assertEqual(chain.side("CE")[0].oi_change, 120)
        self.assertEqual(chain.side("PE")[0].oi_change, -300)
        self.assertEqual(chain.source, "nse")
        self.assertEqual(chain.step, 100.0)

    def test_explicit_expiry_is_honoured(self):
        payload = self.payload()
        wanted = payload["records"]["expiryDates"][0]
        chain = chain_from_nse_payload("NIFTY", payload, expiry=wanted)
        self.assertEqual(chain.expiry, wanted)
        self.assertEqual(len(chain.legs), 2)
        self.assertEqual(chain.side("CE")[0].oi, 9999)

    def test_non_nse_index_and_unknown_index_raise(self):
        provider = options.NSEOptionChainProvider(retries=1)
        with self.assertRaises(OptionChainError):
            provider.fetch("SENSEX")
        with self.assertRaises(OptionChainError):
            provider.fetch("NOTANINDEX")


class ProviderFactoryTests(unittest.TestCase):
    def test_off_returns_none(self):
        self.assertIsNone(build_chain_provider("off"))

    def test_nse_source_builds_a_provider(self):
        provider = build_chain_provider("nse")
        self.assertIsInstance(provider, options.NSEOptionChainProvider)

    def test_synthetic_source_builds_a_provider(self):
        provider = build_chain_provider("synthetic", spot=23000.0)
        self.assertIsInstance(provider, SyntheticOptionChainProvider)
        chain = provider.fetch("NIFTY")
        self.assertEqual(chain.source, "synthetic")


class SerialisationTests(unittest.TestCase):
    def test_analysis_is_json_serialisable(self):
        chain = SyntheticOptionChainProvider(seed=3, spot=24900.0).fetch("NIFTY")
        analysis = analyze_chain(chain, lot_size=25)
        dumped = json.dumps(analysis)
        self.assertIn("max_pain", dumped)
        self.assertIn("bulk_strikes", dumped)


if __name__ == "__main__":
    unittest.main()
