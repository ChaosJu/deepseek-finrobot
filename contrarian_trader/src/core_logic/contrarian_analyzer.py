import logging
from typing import List, Dict, Any, Optional
import statistics # For mean and stdev if needed

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ContrarianAnalyzer:
    """
    Analyzes market data and news sentiment to identify contrarian trading opportunities.
    """

    def analyze_sentiment_divergence(self, 
                                     news_articles: List[Dict[str, Any]], 
                                     social_media_posts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyzes sentiment from news articles.
        Assumes news_articles have 'sentiment_score_placeholder' and 'source_type'.

        Args:
            news_articles: A list of news articles.
            social_media_posts: Optional list of social media posts (currently unused).

        Returns:
            A dictionary with average sentiment and whether negative sentiment is dominant.
        """
        if not news_articles:
            logger.warning("No news articles provided for sentiment analysis.")
            return {'average_sentiment': 0.0, 'is_negative_sentiment_dominant': False, 'processed_articles_count': 0}

        major_news_sentiments = []
        for article in news_articles:
            # Assuming 'sentiment_score_placeholder' exists.
            # And 'source_type' helps filter for 'major' news.
            # If 'source_type' is not always present, this logic might need adjustment.
            if article.get('source_type', 'minor').lower() == 'major' and 'sentiment_score_placeholder' in article:
                try:
                    sentiment_score = float(article['sentiment_score_placeholder'])
                    major_news_sentiments.append(sentiment_score)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse sentiment score for article: {article.get('title', 'N/A')}")
        
        if not major_news_sentiments:
            logger.info("No 'major' news articles with sentiment scores found.")
            # Fallback: consider all articles if no major ones, or handle as per strategy.
            # For now, if no major news, sentiment is considered neutral or not strongly negative.
            return {'average_sentiment': 0.0, 'is_negative_sentiment_dominant': False, 'processed_articles_count': 0}

        avg_score = statistics.mean(major_news_sentiments)
        # Threshold for dominant negative sentiment (example: less than -0.3)
        is_negative_dominant = avg_score < -0.3 
        
        logger.info(f"Sentiment analysis: Avg score from {len(major_news_sentiments)} major articles = {avg_score:.3f}. Negative dominant: {is_negative_dominant}")
        return {
            'average_sentiment': round(avg_score, 3), 
            'is_negative_sentiment_dominant': is_negative_dominant,
            'processed_articles_count': len(major_news_sentiments)
        }

    def analyze_price_volume_anomaly(self, market_data_history: List[Dict[str, Any]], 
                                     short_term_days: int = 5, 
                                     long_term_days: int = 20,
                                     volume_factor_threshold: float = 1.5,
                                     price_min_change_pct: float = -1.0,
                                     price_max_change_pct: float = 5.0
                                     ) -> Dict[str, Any]:
        """
        Analyzes historical market data for price and volume anomalies.
        Checks for high recent volume and stable/rising price.

        Args:
            market_data_history: List of OHLCV dicts, e.g., {'date': ..., 'close': ..., 'volume': ...}.
                                 Expected to be sorted with most recent data first.
            short_term_days: Period for recent volume and price change calculation.
            long_term_days: Period for average volume calculation.
            volume_factor_threshold: Factor by which recent volume must exceed long-term average.
            price_min_change_pct: Min percentage price change over short_term_days for "stable/rising".
            price_max_change_pct: Max percentage price change over short_term_days for "stable/rising".


        Returns:
            Dict with 'is_high_volume_accumulation' and 'is_price_stable_or_rising'.
        """
        if not market_data_history or len(market_data_history) < long_term_days:
            logger.warning(f"Insufficient market data history for analysis (need at least {long_term_days} days, got {len(market_data_history)}).")
            return {'is_high_volume_accumulation': False, 'is_price_stable_or_rising': False, 'reason': "Insufficient data"}

        # Ensure data is sorted, most recent first (caller should ensure this, but good to be aware)
        # market_data_history.sort(key=lambda x: x['date'], reverse=True) # If not guaranteed

        # Volume Analysis
        is_high_volume_accumulation = False
        try:
            recent_volumes = [float(d['volume']) for d in market_data_history[:short_term_days] if 'volume' in d]
            long_term_volumes = [float(d['volume']) for d in market_data_history[:long_term_days] if 'volume' in d]

            if len(recent_volumes) < short_term_days or len(long_term_volumes) < long_term_days:
                logger.warning(f"Not enough volume data points. Recent: {len(recent_volumes)}/{short_term_days}, Long-term: {len(long_term_volumes)}/{long_term_days}")
                return {'is_high_volume_accumulation': False, 'is_price_stable_or_rising': False, 'reason': "Missing volume data points"}


            avg_recent_volume = statistics.mean(recent_volumes)
            avg_long_term_volume = statistics.mean(long_term_volumes)

            if avg_long_term_volume > 0: # Avoid division by zero
                if avg_recent_volume > volume_factor_threshold * avg_long_term_volume:
                    is_high_volume_accumulation = True
                logger.info(f"Volume analysis: Avg recent ({short_term_days}d) vol: {avg_recent_volume:.0f}, Avg long-term ({long_term_days}d) vol: {avg_long_term_volume:.0f}. High accumulation: {is_high_volume_accumulation}")
            else:
                logger.warning("Average long-term volume is zero, cannot perform volume anomaly check.")
        except (ValueError, TypeError, statistics.StatisticsError) as e:
            logger.error(f"Error during volume analysis: {e}", exc_info=True)
            return {'is_high_volume_accumulation': False, 'is_price_stable_or_rising': False, 'reason': f"Volume analysis error: {e}"}


        # Price Stability/Rise Analysis
        is_price_stable_or_rising = False
        try:
            if len(market_data_history) >= short_term_days:
                # Assuming data is sorted: most recent is market_data_history[0]
                current_price = float(market_data_history[0]['close'])
                past_price = float(market_data_history[short_term_days - 1]['close']) # Price from `short_term_days` ago

                if past_price > 0: # Avoid division by zero
                    price_change_pct = ((current_price - past_price) / past_price) * 100
                    if price_min_change_pct <= price_change_pct <= price_max_change_pct:
                        is_price_stable_or_rising = True
                    logger.info(f"Price analysis: Current price: {current_price:.2f}, Past price ({short_term_days}d ago): {past_price:.2f}. Change: {price_change_pct:.2f}%. Stable/Rising: {is_price_stable_or_rising}")
                else:
                    logger.warning("Past price is zero, cannot calculate price change percentage.")
            else:
                logger.warning(f"Not enough data points for price change analysis (need {short_term_days}, got {len(market_data_history)}).")
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error during price analysis: {e}", exc_info=True)
            return {'is_high_volume_accumulation': is_high_volume_accumulation, 'is_price_stable_or_rising': False, 'reason': f"Price analysis error: {e}"}

        return {
            'is_high_volume_accumulation': is_high_volume_accumulation, 
            'is_price_stable_or_rising': is_price_stable_or_rising,
            'reason': "Analysis complete"
        }

    def identify_market_trap(self, market_data_history: List[Dict[str, Any]], current_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identifies potential bull or bear traps. Placeholder for now.
        """
        logger.info("Market trap identification is a placeholder in this version.")
        return {'trap_detected': False, 'trap_type': None, 'reason': "Not implemented"}

    def generate_overall_signal(self, 
                                stock_symbol: str, 
                                news_sentiment_analysis: Dict[str, Any], 
                                price_volume_analysis: Dict[str, Any], 
                                market_trap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates an overall trading signal based on various analyses.
        Implements "Negative News, Positive Volume/Price" strategy.
        """
        logger.info(f"Generating overall signal for {stock_symbol} based on analyses.")
        logger.debug(f"News Sentiment Analysis input: {news_sentiment_analysis}")
        logger.debug(f"Price/Volume Analysis input: {price_volume_analysis}")
        logger.debug(f"Market Trap Analysis input: {market_trap_analysis}") # Currently unused

        is_negative_news = news_sentiment_analysis.get('is_negative_sentiment_dominant', False)
        is_high_volume = price_volume_analysis.get('is_high_volume_accumulation', False)
        is_price_ok = price_volume_analysis.get('is_price_stable_or_rising', False)

        # Core "Negative News, Positive Volume/Price" Contrarian BUY Signal
        if is_negative_news and is_high_volume and is_price_ok:
            reason = (f"Contrarian BUY for {stock_symbol}: Dominant negative news "
                      f"(avg sentiment: {news_sentiment_analysis.get('average_sentiment', 'N/A')}) "
                      "countered by high accumulation volume and stable/rising price.")
            logger.info(f"BUY signal condition met for {stock_symbol}.")
            return {
                'signal': 'BUY', 
                'confidence': 0.7, # Example confidence
                'reason': reason,
                'stock_symbol': stock_symbol,
                'supporting_evidence': {
                    'news_sentiment': news_sentiment_analysis,
                    'price_volume': price_volume_analysis
                }
            }
        else:
            reasons_for_hold = []
            if not is_negative_news:
                reasons_for_hold.append("News sentiment is not dominantly negative.")
            if not is_high_volume:
                reasons_for_hold.append("Volume is not indicating high accumulation.")
            if not is_price_ok:
                reasons_for_hold.append("Price is not stable or rising as per criteria.")
            
            hold_reason = f"HOLD for {stock_symbol}: Conditions for contrarian BUY not met. " + " ".join(reasons_for_hold)
            logger.info(f"HOLD signal condition met for {stock_symbol}. Reason: {' '.join(reasons_for_hold)}")
            return {
                'signal': 'HOLD', 
                'confidence': 0.5, # Example confidence
                'reason': hold_reason,
                'stock_symbol': stock_symbol,
                'supporting_evidence': {
                    'news_sentiment': news_sentiment_analysis,
                    'price_volume': price_volume_analysis
                }
            }

if __name__ == '__main__':
    # Example usage for testing ContrarianAnalyzer methods
    print("--- Testing ContrarianAnalyzer ---")
    analyzer = ContrarianAnalyzer()

    # 1. Test analyze_sentiment_divergence
    print("\n1. Testing Sentiment Divergence:")
    mock_news_positive = [
        {'title': 'Good News Inc', 'sentiment_score_placeholder': 0.5, 'source_type': 'major'},
        {'title': 'Also Good', 'sentiment_score_placeholder': 0.7, 'source_type': 'major'},
    ]
    mock_news_negative = [
        {'title': 'Bad News Corp', 'sentiment_score_placeholder': -0.6, 'source_type': 'major'},
        {'title': 'Very Bad', 'sentiment_score_placeholder': -0.8, 'source_type': 'major'},
        {'title': 'Minor Bad', 'sentiment_score_placeholder': -0.9, 'source_type': 'minor'}, # Should be ignored
    ]
    mock_news_mixed = [
        {'title': 'Mixed Bag', 'sentiment_score_placeholder': -0.7, 'source_type': 'major'},
        {'title': 'Neutralish', 'sentiment_score_placeholder': 0.1, 'source_type': 'major'},
        {'title': 'Positive Spin', 'sentiment_score_placeholder': 0.4, 'source_type': 'major'},
    ]
    print(f"Positive news: {analyzer.analyze_sentiment_divergence(mock_news_positive)}")
    print(f"Negative news: {analyzer.analyze_sentiment_divergence(mock_news_negative)}")
    print(f"Mixed news: {analyzer.analyze_sentiment_divergence(mock_news_mixed)}")
    print(f"Empty news: {analyzer.analyze_sentiment_divergence([])}")
    print(f"News with no major sources: {analyzer.analyze_sentiment_divergence([{'title': 'Minor news only', 'sentiment_score_placeholder': -0.5, 'source_type': 'minor'}])}")


    # 2. Test analyze_price_volume_anomaly
    print("\n2. Testing Price/Volume Anomaly:")
    # Recent data first
    mock_market_data_positive_anomaly = [ # High volume, price stable/rising
        {'date': '2023-10-05', 'close': 102, 'volume': 2000000}, # Day 1 (most recent)
        {'date': '2023-10-04', 'close': 101, 'volume': 2200000}, # Day 2
        {'date': '2023-10-03', 'close': 100, 'volume': 1800000}, # Day 3
        {'date': '2023-10-02', 'close': 99,  'volume': 2100000}, # Day 4
        {'date': '2023-10-01', 'close': 100, 'volume': 1900000}, # Day 5 (price change from here: 102 vs 100 = +2%)
    ] + [{'date': f'2023-09-{20-i:02d}', 'close': 95+i*0.1, 'volume': 1000000} for i in range(15)] # Prev 15 days (total 20)
    
    mock_market_data_no_anomaly = [ # Low volume, price falling
        {'date': '2023-10-05', 'close': 98, 'volume': 800000},
        {'date': '2023-10-04', 'close': 99, 'volume': 700000},
        {'date': '2023-10-03', 'close': 100, 'volume': 900000},
        {'date': '2023-10-02', 'close': 101, 'volume': 850000},
        {'date': '2023-10-01', 'close': 102, 'volume': 750000}, # Price change from here: 98 vs 102 = -3.9%
    ] + [{'date': f'2023-09-{20-i:02d}', 'close': 100-i*0.1, 'volume': 1000000} for i in range(15)]

    print(f"Positive anomaly case: {analyzer.analyze_price_volume_anomaly(mock_market_data_positive_anomaly)}")
    print(f"No anomaly case: {analyzer.analyze_price_volume_anomaly(mock_market_data_no_anomaly)}")
    print(f"Insufficient data: {analyzer.analyze_price_volume_anomaly(mock_market_data_no_anomaly[:5])}") # Only 5 days of data

    # 3. Test identify_market_trap (placeholder)
    print("\n3. Testing Market Trap (Placeholder):")
    print(analyzer.identify_market_trap(mock_market_data_no_anomaly, {}))

    # 4. Test generate_overall_signal
    print("\n4. Testing Generate Overall Signal:")
    news_neg = analyzer.analyze_sentiment_divergence(mock_news_negative)
    pv_pos = analyzer.analyze_price_volume_anomaly(mock_market_data_positive_anomaly)
    trap_none = analyzer.identify_market_trap([], {})

    # Case 1: BUY signal
    print("  Case 1: BUY (Negative News, Positive PV)")
    signal_buy = analyzer.generate_overall_signal("AAPL", news_neg, pv_pos, trap_none)
    print(f"    Signal: {signal_buy['signal']}, Confidence: {signal_buy['confidence']}, Reason: {signal_buy['reason']}")

    # Case 2: HOLD signal (e.g., news not negative enough)
    print("  Case 2: HOLD (Positive News, Positive PV)")
    news_pos = analyzer.analyze_sentiment_divergence(mock_news_positive)
    signal_hold_news = analyzer.generate_overall_signal("MSFT", news_pos, pv_pos, trap_none)
    print(f"    Signal: {signal_hold_news['signal']}, Confidence: {signal_hold_news['confidence']}, Reason: {signal_hold_news['reason']}")

    # Case 3: HOLD signal (e.g., PV not anomalous)
    print("  Case 3: HOLD (Negative News, Non-Anomalous PV)")
    pv_neg = analyzer.analyze_price_volume_anomaly(mock_market_data_no_anomaly)
    signal_hold_pv = analyzer.generate_overall_signal("GOOG", news_neg, pv_neg, trap_none)
    print(f"    Signal: {signal_hold_pv['signal']}, Confidence: {signal_hold_pv['confidence']}, Reason: {signal_hold_pv['reason']}")
    
    print("\n--- ContrarianAnalyzer tests complete ---")
