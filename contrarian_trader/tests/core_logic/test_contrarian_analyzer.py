import unittest
import statistics # For calculating expected averages if needed
from contrarian_trader.src.core_logic.contrarian_analyzer import ContrarianAnalyzer

# Helper data for tests
MOCK_NEWS_NEGATIVE_DOMINANT = [
    {'title': 'Disaster Strikes Market', 'sentiment_score_placeholder': -0.8, 'source_type': 'major'},
    {'title': 'Outlook Grim for Sector', 'sentiment_score_placeholder': -0.6, 'source_type': 'major'},
    {'title': 'Another Bad Day', 'sentiment_score_placeholder': -0.7, 'source_type': 'major'},
    {'title': 'Minor Good News', 'sentiment_score_placeholder': 0.5, 'source_type': 'minor'}, # Should be filtered out
]

MOCK_NEWS_POSITIVE_NEUTRAL = [
    {'title': 'Market Soars High', 'sentiment_score_placeholder': 0.8, 'source_type': 'major'},
    {'title': 'Neutral Day for Stocks', 'sentiment_score_placeholder': 0.1, 'source_type': 'major'},
    {'title': 'Slightly Up', 'sentiment_score_placeholder': 0.3, 'source_type': 'major'},
]

MOCK_NEWS_NO_MAJOR = [
    {'title': 'Blog Post Negative', 'sentiment_score_placeholder': -0.9, 'source_type': 'minor'},
    {'title': 'Tweet Positive', 'sentiment_score_placeholder': 0.7, 'source_type': 'social'},
]

# Market data: recent data first
MOCK_MARKET_HIGH_VOL_STABLE_PRICE = [ # High volume, price stable/rising
    {'date': '2023-10-05', 'close': 102, 'volume': 2000000}, # Day 1 (most recent)
    {'date': '2023-10-04', 'close': 101, 'volume': 2200000}, # Day 2
    {'date': '2023-10-03', 'close': 100, 'volume': 1800000}, # Day 3
    {'date': '2023-10-02', 'close': 99,  'volume': 2100000}, # Day 4
    {'date': '2023-10-01', 'close': 100, 'volume': 1900000}, # Day 5 (price change from here: 102 vs 100 = +2%)
] + [{'date': f'2023-09-{20-i:02d}', 'close': 95+i*0.1, 'volume': 1000000} for i in range(15)] # Prev 15 days (total 20)

MOCK_MARKET_LOW_VOL = [
    {'date': '2023-10-05', 'close': 102, 'volume': 800000}, # Low recent volume
    {'date': '2023-10-04', 'close': 101, 'volume': 700000},
    {'date': '2023-10-03', 'close': 100, 'volume': 900000},
    {'date': '2023-10-02', 'close': 99,  'volume': 850000},
    {'date': '2023-10-01', 'close': 100, 'volume': 750000},
] + [{'date': f'2023-09-{20-i:02d}', 'close': 95+i*0.1, 'volume': 1000000} for i in range(15)]

MOCK_MARKET_FALLING_PRICE = [
    {'date': '2023-10-05', 'close': 95,  'volume': 2000000}, # Price fell from 100 to 95 (-5%)
    {'date': '2023-10-04', 'close': 97,  'volume': 2200000},
    {'date': '2023-10-03', 'close': 98,  'volume': 1800000},
    {'date': '2023-10-02', 'close': 99,  'volume': 2100000},
    {'date': '2023-10-01', 'close': 100, 'volume': 1900000},
] + [{'date': f'2023-09-{20-i:02d}', 'close': 95+i*0.1, 'volume': 1000000} for i in range(15)]


class TestContrarianAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ContrarianAnalyzer()

    # --- Test analyze_sentiment_divergence ---
    def test_sentiment_divergence_negative_dominant(self):
        result = self.analyzer.analyze_sentiment_divergence(MOCK_NEWS_NEGATIVE_DOMINANT)
        expected_avg = statistics.mean([-0.8, -0.6, -0.7])
        self.assertAlmostEqual(result['average_sentiment'], expected_avg, places=3)
        self.assertTrue(result['is_negative_sentiment_dominant'])
        self.assertEqual(result['processed_articles_count'], 3)

    def test_sentiment_divergence_not_negative(self):
        result = self.analyzer.analyze_sentiment_divergence(MOCK_NEWS_POSITIVE_NEUTRAL)
        expected_avg = statistics.mean([0.8, 0.1, 0.3])
        self.assertAlmostEqual(result['average_sentiment'], expected_avg, places=3)
        self.assertFalse(result['is_negative_sentiment_dominant'])
        self.assertEqual(result['processed_articles_count'], 3)

    def test_sentiment_divergence_no_major_news(self):
        result = self.analyzer.analyze_sentiment_divergence(MOCK_NEWS_NO_MAJOR)
        self.assertEqual(result['average_sentiment'], 0.0)
        self.assertFalse(result['is_negative_sentiment_dominant'])
        self.assertEqual(result['processed_articles_count'], 0)
        
    def test_sentiment_divergence_empty_news(self):
        result = self.analyzer.analyze_sentiment_divergence([])
        self.assertEqual(result['average_sentiment'], 0.0)
        self.assertFalse(result['is_negative_sentiment_dominant'])
        self.assertEqual(result['processed_articles_count'], 0)

    # --- Test analyze_price_volume_anomaly ---
    def test_price_volume_high_accumulation_stable_price(self):
        result = self.analyzer.analyze_price_volume_anomaly(MOCK_MARKET_HIGH_VOL_STABLE_PRICE)
        self.assertTrue(result['is_high_volume_accumulation'])
        self.assertTrue(result['is_price_stable_or_rising'])

    def test_price_volume_low_volume(self):
        result = self.analyzer.analyze_price_volume_anomaly(MOCK_MARKET_LOW_VOL)
        self.assertFalse(result['is_high_volume_accumulation'])
        # Price might still be stable/rising in this mock data, check specific assertion
        # MOCK_MARKET_LOW_VOL prices: 102 vs 100 (5 days ago) = +2.0%, which is stable/rising
        self.assertTrue(result['is_price_stable_or_rising'])


    def test_price_volume_falling_price(self):
        result = self.analyzer.analyze_price_volume_anomaly(MOCK_MARKET_FALLING_PRICE)
        # Volume is high in this mock data
        self.assertTrue(result['is_high_volume_accumulation'])
        self.assertFalse(result['is_price_stable_or_rising'])
        
    def test_price_volume_insufficient_data(self):
        result = self.analyzer.analyze_price_volume_anomaly(MOCK_MARKET_HIGH_VOL_STABLE_PRICE[:5]) # Only 5 days
        self.assertFalse(result['is_high_volume_accumulation'])
        self.assertFalse(result['is_price_stable_or_rising'])
        self.assertEqual(result['reason'], "Insufficient data")

    # --- Test generate_overall_signal ---
    def test_generate_signal_strong_buy_contrarian(self):
        news_sentiment_analysis = {'is_negative_sentiment_dominant': True, 'average_sentiment': -0.7}
        price_volume_analysis = {'is_high_volume_accumulation': True, 'is_price_stable_or_rising': True}
        market_trap_analysis = {'trap_detected': False} # Placeholder
        
        result = self.analyzer.generate_overall_signal(
            "AAPL", news_sentiment_analysis, price_volume_analysis, market_trap_analysis
        )
        self.assertEqual(result['signal'], 'BUY')
        self.assertTrue("Contrarian BUY for AAPL" in result['reason'])
        self.assertGreater(result['confidence'], 0.5)

    def test_generate_signal_hold_due_to_positive_sentiment(self):
        news_sentiment_analysis = {'is_negative_sentiment_dominant': False, 'average_sentiment': 0.2}
        price_volume_analysis = {'is_high_volume_accumulation': True, 'is_price_stable_or_rising': True}
        market_trap_analysis = {'trap_detected': False}
        
        result = self.analyzer.generate_overall_signal(
            "MSFT", news_sentiment_analysis, price_volume_analysis, market_trap_analysis
        )
        self.assertEqual(result['signal'], 'HOLD')
        self.assertTrue("News sentiment is not dominantly negative" in result['reason'])

    def test_generate_signal_hold_due_to_low_volume(self):
        news_sentiment_analysis = {'is_negative_sentiment_dominant': True, 'average_sentiment': -0.6}
        price_volume_analysis = {'is_high_volume_accumulation': False, 'is_price_stable_or_rising': True}
        market_trap_analysis = {'trap_detected': False}
        
        result = self.analyzer.generate_overall_signal(
            "GOOG", news_sentiment_analysis, price_volume_analysis, market_trap_analysis
        )
        self.assertEqual(result['signal'], 'HOLD')
        self.assertTrue("Volume is not indicating high accumulation" in result['reason'])

    def test_generate_signal_hold_due_to_falling_price(self):
        news_sentiment_analysis = {'is_negative_sentiment_dominant': True, 'average_sentiment': -0.5}
        price_volume_analysis = {'is_high_volume_accumulation': True, 'is_price_stable_or_rising': False}
        market_trap_analysis = {'trap_detected': False}
        
        result = self.analyzer.generate_overall_signal(
            "TSLA", news_sentiment_analysis, price_volume_analysis, market_trap_analysis
        )
        self.assertEqual(result['signal'], 'HOLD')
        self.assertTrue("Price is not stable or rising as per criteria" in result['reason'])

if __name__ == '__main__':
    unittest.main()
