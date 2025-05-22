import logging
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from contrarian_trader.src.core_logic.contrarian_analyzer import ContrarianAnalyzer 

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ContrarianStrategyAgent(BaseAgent):
    """
    Agent that applies contrarian logic using the ContrarianAnalyzer.
    It takes inputs from various data-providing agents (Market, News, Social)
    and uses the analyzer to generate a trading signal.
    """

    def __init__(self, 
                 contrarian_analyzer: ContrarianAnalyzer,
                 agent_name: str = "ContrarianStrategyAgent", 
                 logger_instance: Optional[logging.Logger] = None): # Renamed logger to logger_instance to avoid conflict
        """
        Initializes the ContrarianStrategyAgent.

        Args:
            contrarian_analyzer: An instance of ContrarianAnalyzer.
            agent_name: Name of the agent.
            logger_instance: Optional logger instance.
        """
        super().__init__(agent_name=agent_name, logger=logger_instance) # Pass logger_instance to super
        if not isinstance(contrarian_analyzer, ContrarianAnalyzer):
            self.logger.error("ContrarianStrategyAgent requires a valid ContrarianAnalyzer instance.")
            raise ValueError("contrarian_analyzer must be an instance of ContrarianAnalyzer.")
        
        self.analyzer = contrarian_analyzer # Changed from self.contrarian_analyzer for clarity
        self.last_analysis_result: Optional[Dict[str, Any]] = None
        self.logger.info(f"ContrarianStrategyAgent initialized with {self.analyzer.__class__.__name__}.")

    def process(self, data: Dict[str, Any]) -> None:
        """
        Processes the combined data from other agents by feeding it to the ContrarianAnalyzer.
        This method orchestrates the analysis steps.

        Args:
            data: A dictionary containing data required for analysis.
                  Expected keys:
                  - 'stock_symbol': The symbol being analyzed (str).
                  - 'news_data': Output from NewsAgent (List[Dict[str, Any]]).
                                 Each dict should have 'sentiment_score_placeholder' and 'source_type'.
                  - 'market_data': Output from MarketDataAgent (List[Dict[str, Any]] - OHLCV).
                  - 'social_media_data': (Optional) Output from SocialMediaAgent.
        """
        self.logger.info(f"Processing data for symbol: {data.get('stock_symbol', 'N/A')} for contrarian analysis.")
        
        stock_symbol = data.get('stock_symbol')
        if not stock_symbol:
            self.logger.error("Missing 'stock_symbol' in data for ContrarianStrategyAgent.")
            self.last_analysis_result = {"signal": "error", "confidence": 0.0, "reason": "Missing stock_symbol."}
            return

        # Extract data for analyzer, ensuring they are in the correct format (list of dicts)
        # The CEOAgent is expected to structure `all_gathered_data` correctly.
        # news_data_output from NewsAgent is expected under data['news_agent_data']['content']['articles_with_sentiment']
        # market_data_output from MarketDataAgent is expected under data['market_data_agent_data']['content']['raw_market_data']
        
        news_agent_output = data.get('news_agent_data', {}).get('content', {})
        news_articles = news_agent_output.get('articles_with_sentiment', [])
        if not isinstance(news_articles, list):
            self.logger.warning(f"News articles for {stock_symbol} is not a list or not found. Found: {type(news_articles)}. Using empty list.")
            news_articles = []

        market_data_output = data.get('market_data_agent_data', {}).get('content', {})
        # Market data agent's 'raw_market_data' key was used in its own `run` method's process call.
        # The actual data structure for historical might be directly a list if `data_type` was 'historical'.
        # Let's check common structures based on MarketDataAgent output.
        market_history = market_data_output.get('raw_market_data', []) # if data_type was 'historical'
        if not market_history and 'price' in market_data_output : # if data_type was 'latest_price'
             self.logger.warning(f"Received latest price for {stock_symbol}, but historical data is needed for full analysis. Market data: {market_data_output}")
             # This strategy needs historical data. If only latest price, it can't proceed.
             market_history = [] # Or could be a list containing just the latest price if analyzer handles it.
        
        if not isinstance(market_history, list):
            self.logger.warning(f"Market data history for {stock_symbol} is not a list or not found. Found: {type(market_history)}. Using empty list.")
            market_history = []


        # Social media data is optional and not used by current core strategy logic in analyzer
        # social_media_posts = data.get('social_media_data', {}).get('content', {}).get('posts_with_sentiment', [])

        try:
            self.logger.debug(f"Calling ContrarianAnalyzer.analyze_sentiment_divergence for {stock_symbol} with {len(news_articles)} articles.")
            news_sentiment_result = self.analyzer.analyze_sentiment_divergence(news_articles=news_articles) # social_media_posts currently unused

            self.logger.debug(f"Calling ContrarianAnalyzer.analyze_price_volume_anomaly for {stock_symbol} with {len(market_history)} data points.")
            price_volume_result = self.analyzer.analyze_price_volume_anomaly(market_data_history=market_history)
            
            # Market trap analysis is a placeholder
            self.logger.debug(f"Calling ContrarianAnalyzer.identify_market_trap for {stock_symbol}.")
            trap_result = self.analyzer.identify_market_trap(market_data_history=market_history, current_sentiment=news_sentiment_result)

            self.logger.debug(f"Calling ContrarianAnalyzer.generate_overall_signal for {stock_symbol}.")
            self.last_analysis_result = self.analyzer.generate_overall_signal(
                stock_symbol=stock_symbol,
                news_sentiment_analysis=news_sentiment_result,
                price_volume_analysis=price_volume_result,
                market_trap_analysis=trap_result
            )
            self.logger.info(f"Contrarian analysis for {stock_symbol} complete. Signal: {self.last_analysis_result.get('signal')}")

        except Exception as e:
            self.logger.error(f"Error during contrarian analysis for {stock_symbol}: {e}", exc_info=True)
            self.last_analysis_result = {"signal": "error", "confidence": 0.0, "reason": str(e)}

    def make_recommendation(self) -> Dict[str, Any]:
        """
        Returns the latest analysis result from the ContrarianAnalyzer.
        """
        if self.last_analysis_result:
            self.logger.info(f"Providing contrarian analysis recommendation: {self.last_analysis_result.get('signal')}")
            return self.last_analysis_result
        else:
            self.logger.warning("No contrarian analysis result available to make a recommendation.")
            return {"signal": "hold", "confidence": 0.0, "reason": "No analysis performed yet."}

    def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrates the contrarian analysis process.
        The 'data' dictionary is expected to be prepared by the CEOAgent, containing all necessary inputs.
        """
        self.logger.info("ContrarianStrategyAgent run started.")
        if not data or 'stock_symbol' not in data:
            self.logger.error("ContrarianStrategyAgent run requires 'stock_symbol' in data.")
            return {"signal": "error", "confidence": 0.0, "reason": "Missing 'stock_symbol' in input data for run."}

        # The process method now handles the detailed logic of calling analyzer methods
        self.process(data) 
        
        recommendation = self.make_recommendation()
        self.logger.info(f"ContrarianStrategyAgent run finished. Recommendation: {recommendation.get('signal')}")
        return recommendation

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- Testing ContrarianStrategyAgent with implemented ContrarianAnalyzer ---")

    # Use the actual ContrarianAnalyzer
    actual_analyzer = ContrarianAnalyzer()
    strategy_agent = ContrarianStrategyAgent(contrarian_analyzer=actual_analyzer)

    # Mock data structures as they would come from data-providing agents (nested structure)
    # This aligns with how CEOAgent would pass data.
    
    # Case 1: Data that should trigger a BUY signal
    mock_news_negative_major = [
        {'title': 'Very Bad Corp News', 'sentiment_score_placeholder': -0.7, 'source_type': 'major'},
        {'title': 'Doom and Gloom Times', 'sentiment_score_placeholder': -0.5, 'source_type': 'major'},
        {'title': 'Minor issue', 'sentiment_score_placeholder': -0.8, 'source_type': 'minor'}, # Ignored
    ]
    # Data for positive price/volume anomaly (recent data first)
    mock_market_data_positive_pv = [ 
        {'date': '2023-10-05', 'close': 102, 'volume': 2000000}, 
        {'date': '2023-10-04', 'close': 101, 'volume': 2200000}, 
        {'date': '2023-10-03', 'close': 100, 'volume': 1800000}, 
        {'date': '2023-10-02', 'close': 99,  'volume': 2100000}, 
        {'date': '2023-10-01', 'close': 100, 'volume': 1900000}, 
    ] + [{'date': f'2023-09-{20-i:02d}', 'close': 95+i*0.1, 'volume': 1000000} for i in range(15)]
    
    run_data_for_buy = {
        "stock_symbol": "CONTRABUY",
        "news_agent_data": { # Simulating output from NewsAgent
            "status": "success", 
            "content": {"articles_with_sentiment": mock_news_negative_major}
        },
        "market_data_agent_data": { # Simulating output from MarketDataAgent
            "status": "success",
            "content": {"raw_market_data": mock_market_data_positive_pv, "data_type": "historical"}
        }
        # social_media_data can be omitted as it's optional
    }
    logger.info("\n--- Test Case 1: BUY Signal ---")
    result_buy = strategy_agent.run(data=run_data_for_buy)
    logger.info(f"Result (CONTRABUY): Signal: {result_buy.get('signal')}, Confidence: {result_buy.get('confidence')}, Reason: {result_buy.get('reason')}")
    assert result_buy.get('signal') == 'BUY'

    # Case 2: Data that should result in HOLD (e.g., news not negative enough)
    mock_news_neutral_major = [
        {'title': 'Neutral News Corp', 'sentiment_score_placeholder': -0.1, 'source_type': 'major'},
        {'title': 'Slightly Upbeat Times', 'sentiment_score_placeholder': 0.2, 'source_type': 'major'},
    ]
    run_data_for_hold_news = {
        "stock_symbol": "HOLDNEWS",
        "news_agent_data": {"content": {"articles_with_sentiment": mock_news_neutral_major}},
        "market_data_agent_data": {"content": {"raw_market_data": mock_market_data_positive_pv, "data_type": "historical"}}
    }
    logger.info("\n--- Test Case 2: HOLD Signal (News Not Negative) ---")
    result_hold_news = strategy_agent.run(data=run_data_for_hold_news)
    logger.info(f"Result (HOLDNEWS): Signal: {result_hold_news.get('signal')}, Confidence: {result_hold_news.get('confidence')}, Reason: {result_hold_news.get('reason')}")
    assert result_hold_news.get('signal') == 'HOLD'
    assert "News sentiment is not dominantly negative" in result_hold_news.get('reason', '')


    # Case 3: Data for HOLD (e.g., volume not high enough)
    mock_market_data_low_volume = [
        {'date': '2023-10-05', 'close': 102, 'volume': 1000000}, # Recent volume not high
        {'date': '2023-10-04', 'close': 101, 'volume': 1200000}, 
        {'date': '2023-10-03', 'close': 100, 'volume': 800000}, 
        {'date': '2023-10-02', 'close': 99,  'volume': 1100000}, 
        {'date': '2023-10-01', 'close': 100, 'volume': 900000}, 
    ] + [{'date': f'2023-09-{20-i:02d}', 'close': 95+i*0.1, 'volume': 1000000} for i in range(15)]
    run_data_for_hold_pv = {
        "stock_symbol": "HOLDPV",
        "news_agent_data": {"content": {"articles_with_sentiment": mock_news_negative_major}},
        "market_data_agent_data": {"content": {"raw_market_data": mock_market_data_low_volume, "data_type": "historical"}}
    }
    logger.info("\n--- Test Case 3: HOLD Signal (Price/Volume Anomaly Not Met) ---")
    result_hold_pv = strategy_agent.run(data=run_data_for_hold_pv)
    logger.info(f"Result (HOLDPV): Signal: {result_hold_pv.get('signal')}, Confidence: {result_hold_pv.get('confidence')}, Reason: {result_hold_pv.get('reason')}")
    assert result_hold_pv.get('signal') == 'HOLD'
    assert "Volume is not indicating high accumulation" in result_hold_pv.get('reason', '')

    # Case 4: Missing market_data
    run_data_missing_market = {
        "stock_symbol": "MISSINGMARKET",
        "news_agent_data": {"content": {"articles_with_sentiment": mock_news_negative_major}},
        # market_data_agent_data is missing
    }
    logger.info("\n--- Test Case 4: HOLD Signal (Missing Market Data) ---")
    result_missing_market = strategy_agent.run(data=run_data_missing_market)
    logger.info(f"Result (MISSINGMARKET): Signal: {result_missing_market.get('signal')}, Confidence: {result_missing_market.get('confidence')}, Reason: {result_missing_market.get('reason')}")
    assert result_missing_market.get('signal') == 'HOLD' # Should default to hold as P/V conditions won't be met
    assert "Volume is not indicating high accumulation" in result_missing_market.get('reason', '') # Because market_history will be []

    # Case 5: Missing stock_symbol
    logger.info("\n--- Test Case 5: ERROR (Missing Stock Symbol) ---")
    result_no_symbol = strategy_agent.run(data={"news_agent_data": {}, "market_data_agent_data": {}})
    logger.info(f"Result (no symbol): {result_no_symbol}")
    assert result_no_symbol.get('signal') == 'error'

    logger.info("\n--- ContrarianStrategyAgent tests with actual ContrarianAnalyzer complete ---")
