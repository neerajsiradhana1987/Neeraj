"""Tests for pure-stdlib indicator math."""
import unittest

from trading_agent import indicators as ind


class IndicatorTests(unittest.TestCase):
    def test_sma_and_ema(self):
        values = [float(i) for i in range(1, 21)]
        self.assertAlmostEqual(ind.sma(values, 5), 18.0)
        self.assertAlmostEqual(ind.ema(values, 5), ind.ema_series(values, 5)[-1])
        self.assertGreater(ind.ema(values, 5), ind.ema(values, 20))

    def test_rsi_bounds_and_uptrend(self):
        rising = [100.0 + i for i in range(60)]
        self.assertEqual(ind.rsi(rising, 14), 100.0)
        self.assertIsNone(ind.rsi(rising[:10], 14))
        wobbling = [100.0, 101.0, 100.5, 101.5, 101.0, 102.0, 101.5, 102.5]
        rsi = ind.rsi(wobbling, 3)
        self.assertTrue(0.0 <= rsi <= 100.0)

    def test_macd_uptrend(self):
        rising = [100.0 + 0.5 * i for i in range(80)]
        result = ind.macd(rising)
        self.assertIsNotNone(result)
        self.assertGreater(result.line, 0)
        self.assertGreaterEqual(result.histogram, -1e-9)
        self.assertIsNone(ind.macd(rising[:20]))

    def test_macd_histogram_positive_when_accelerating(self):
        convex = [100.0 + 0.05 * i * i for i in range(80)]
        result = ind.macd(convex)
        self.assertGreater(result.histogram, 0)

    def test_atr_positive(self):
        highs = [float(i) + 1.0 for i in range(40)]
        lows = [float(i) - 1.0 for i in range(40)]
        closes = [float(i) for i in range(40)]
        self.assertAlmostEqual(ind.atr(highs, lows, closes, 14), 2.0, places=6)
        self.assertIsNone(ind.atr(highs[:10], lows[:10], closes[:10], 14))

    def test_bollinger_ordering(self):
        closes = [100.0 + (i % 5) for i in range(30)]
        bands = ind.bollinger(closes, 20, 2.0, price=104.0)
        self.assertGreater(bands.upper, bands.middle)
        self.assertGreater(bands.middle, bands.lower)
        self.assertAlmostEqual(bands.middle, ind.sma(closes, 20))
        self.assertGreater(bands.bandwidth_pct, 0)

    def test_helpers(self):
        self.assertAlmostEqual(ind.pct_change(110, 100), 10.0)
        self.assertIsNone(ind.pct_change(110, 0))
        self.assertGreater(ind.volatility_pct([100.0, 101.0, 100.5, 102.0, 101.0, 103.0], 5), 0)
        self.assertEqual(ind.sma([1.0, 2.0], 5), None)


if __name__ == "__main__":
    unittest.main()
