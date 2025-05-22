import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from contrarian_trader.src.data_sources import NewsDataSource # Adjusted import path

# Placeholder for a sentiment analysis tool/client
class MockSentimentAnalyzer:
    def analyze(self, text: str) -> float:
        """Mocks sentiment analysis, returning a score between -1 (negative) and 1 (positive)."""
        # Simple mock logic: longer text is more positive, shorter more negative
        # This is purely for demonstration and not realistic.
        score = (len(text) % 200 / 100.0) - 1.0 # Score between -1.0 and 1.0
        return round(max(-1.0, min(1.0, score)), 2)


class NewsAgent(BaseAgent):
    """
    Agent responsible for fetching, processing, and analyzing news data.
    """

    def __init__(self, 
                 news_data_source: NewsDataSource, 
                 sentiment_analyzer: Optional[Any] = None, # Placeholder for actual sentiment tool
                 agent_name: str = "NewsAgent", 
                 logger_instance: Optional[logging.Logger] = None): # Renamed logger
        """
        Initializes the NewsAgent.

        Args:
            news_data_source: An instance of NewsDataSource to fetch news.
            sentiment_analyzer: A tool/client for sentiment analysis. Uses MockSentimentAnalyzer if None.
            agent_name: Name of the agent.
            logger_instance: Optional logger instance.
        """
        super().__init__(agent_name=agent_name, logger_instance=logger_instance) # Pass logger_instance
        if not isinstance(news_data_source, NewsDataSource):
            self.logger.error("NewsAgent requires a valid NewsDataSource instance.")
            raise ValueError("news_data_source must be an instance of NewsDataSource.")
        
        self.news_data_source = news_data_source
        self.sentiment_analyzer = sentiment_analyzer or MockSentimentAnalyzer()
        self.processed_news_data: Optional[Dict[str, Any]] = None # Stores list of articles with sentiment
        self.logger.info(f"NewsAgent initialized with {news_data_source.__class__.__name__} and {self.sentiment_analyzer.__class__.__name__}.")

    def process(self, data: Dict[str, Any]) -> None:
        """
        Processes fetched news articles by adding sentiment scores.

        Args:
            data: A dictionary containing a list of raw news articles.
                  Expected key: 'articles' (list of dicts from NewsDataSource).
        """
        self.logger.info(f"Processing news data: {len(data.get('articles', []))} articles.")
        if not data or 'articles' not in data or not isinstance(data['articles'], list):
            self.logger.warning("No 'articles' list found in data or data is malformed.")
            self.processed_news_data = {"articles_with_sentiment": []}
            return

        articles_with_sentiment = []
        for article in data['articles']:
            try:
                # Use 'summary' or 'title' for sentiment analysis, depending on availability
                text_to_analyze = article.get('summary') or article.get('title', '')
                if text_to_analyze:
                    sentiment_score = self.sentiment_analyzer.analyze(text_to_analyze)
                else:
                    sentiment_score = 0.0 # Neutral if no text
                
                processed_article = article.copy() # Avoid modifying original data from source
                processed_article['sentiment_score'] = sentiment_score
                articles_with_sentiment.append(processed_article)
            except Exception as e:
                self.logger.error(f"Error processing sentiment for article titled '{article.get('title', 'N/A')}': {e}", exc_info=True)
                # Add article without sentiment if processing fails
                processed_article = article.copy()
                processed_article['sentiment_score'] = None 
                articles_with_sentiment.append(processed_article)
        
        self.processed_news_data = {"articles_with_sentiment": articles_with_sentiment}
        self.logger.debug(f"News data processed. {len(articles_with_sentiment)} articles with sentiment.")

    def make_recommendation(self) -> Dict[str, Any]:
        """
        Provides the processed news data including sentiment scores.
        This agent primarily provides data; "recommendation" here means analyzed news.

        Returns:
            A dictionary containing the news articles with their sentiment scores.
        """
        if self.processed_news_data and self.processed_news_data.get("articles_with_sentiment"):
            self.logger.info("Providing processed news data with sentiment.")
            # Calculate an aggregate sentiment score as an example
            scores = [art['sentiment_score'] for art in self.processed_news_data["articles_with_sentiment"] if art['sentiment_score'] is not None]
            overall_sentiment = sum(scores) / len(scores) if scores else 0.0
            return {
                "status": "success", 
                "data_type": "news_data_with_sentiment", 
                "content": self.processed_news_data,
                "overall_sentiment_placeholder": round(overall_sentiment, 3)
            }
        else:
            self.logger.warning("No processed news data available to provide.")
            return {"status": "no_data", "data_type": "news_data_with_sentiment", "content": {}}

    def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrates fetching news and processing it for sentiment.

        Args:
            data: A dictionary containing parameters for fetching news,
                  e.g., {'symbol': 'AAPL', 'limit': 10}.

        Returns:
            A dictionary containing the news articles with sentiment scores.
        """
        self.logger.info(f"NewsAgent run started with parameters: {data}")
        if not data or 'symbol' not in data:
            self.logger.error("Missing 'symbol' in run parameters for NewsAgent.")
            return {"status": "error", "message": "Missing 'symbol' in run parameters."}

        symbol = data['symbol']
        limit = data.get('limit', 10) # Default limit for news articles

        try:
            # Fetch raw news articles using the NewsDataSource
            # The NewsDataSource's fetch_data or get_news_for_stock can be used.
            # Assuming fetch_data is the generic entry point.
            raw_articles = self.news_data_source.fetch_data(symbol, limit=limit)
            
            if raw_articles:
                # Pass the fetched articles to the process method
                self.process({'articles': raw_articles, 'symbol': symbol})
            else:
                self.logger.warning(f"No raw news articles fetched for {symbol}.")
                # Ensure processed_data is initialized even if no articles are found
                self.processed_news_data = {'articles_with_sentiment': [], 'symbol': symbol}


        except Exception as e:
            self.logger.error(f"Error during NewsAgent run for {symbol}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        
        recommendation = self.make_recommendation()
        self.logger.info(f"NewsAgent run finished. Result status: {recommendation.get('status')}")
        return recommendation

if __name__ == '__main__':
    # This is for testing purposes and won't run when imported.
    logging.basicConfig(level=logging.DEBUG)

    # Mock NewsDataSource for testing
    class MockNewsDataSource(NewsDataSource):
        def __init__(self, api_key: str = "test_news_key"):
            super().__init__(api_key=api_key)
            self.logger.info("MockNewsDataSource initialized for testing.")

        def fetch_data(self, symbol: str, **kwargs) -> List[Dict[str, Any]]:
            limit = kwargs.get('limit', 2)
            self.logger.debug(f"Mock fetching {limit} news articles for {symbol}")
            return [
                {'title': f'{symbol} Soars!', 'summary': 'A very positive summary about {symbol}. Great performance!', 'source': 'Mock News', 'published_at': '2023-01-01T10:00:00Z'},
                {'title': f'{symbol} Plummets?', 'summary': 'Concerning news for {symbol}.', 'source': 'Mock Finance', 'published_at': '2023-01-01T11:00:00Z'}
            ][:limit]

    print("\n--- Testing NewsAgent ---")
    mock_news_source = MockNewsDataSource()
    news_agent = NewsAgent(news_data_source=mock_news_source) # Uses MockSentimentAnalyzer by default

    # Test run for fetching and processing news
    print("\nRunning agent for news data:")
    news_params = {'symbol': 'TESTNEWS', 'limit': 2}
    result_news = news_agent.run(data=news_params)
    print(f"Result (news): {result_news}")
    if result_news['status'] == 'success':
        for article in result_news['content']['articles_with_sentiment']:
            print(f"  Article: '{article['title']}', Sentiment: {article['sentiment_score']}")
        print(f"  Overall Sentiment (Placeholder): {result_news['overall_sentiment_placeholder']}")


    # Test run with missing parameters
    print("\nRunning agent with missing parameters:")
    result_missing_params = news_agent.run(data={}) # Missing 'symbol'
    print(f"Result (missing params): {result_missing_params}")

    # Test make_recommendation without run (should be empty)
    print("\nTesting make_recommendation without prior run:")
    # Re-initialize agent to ensure clean state
    news_agent_clean = NewsAgent(news_data_source=mock_news_source)
    direct_recommendation = news_agent_clean.make_recommendation()
    print(f"Direct recommendation (no data): {direct_recommendation}")
    
    print("\n--- NewsAgent tests complete ---")
