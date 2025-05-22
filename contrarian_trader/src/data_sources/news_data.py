import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from .base_data_source import BaseDataSource

logger = logging.getLogger(__name__)

class NewsDataSource(BaseDataSource):
    """
    Data source for fetching news articles and sentiment.
    """

    def __init__(self, api_key: str = None):
        """
        Initializes the news data source.
        Args:
            api_key: The API key for the news provider.
                     If None, attempts to read from NEWS_API_KEY environment variable.
        """
        resolved_api_key = api_key or os.getenv("NEWS_API_KEY")
        if not resolved_api_key:
            logger.warning("News API key not provided and NEWS_API_KEY environment variable not set.")
        super().__init__(api_key=resolved_api_key)
        logger.info("NewsDataSource initialized.")

    def fetch_data(self, symbol: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetches news data for a given symbol.
        This is a generic entry point, defaulting to get_news_for_stock.

        Args:
            symbol: The stock symbol.
            **kwargs: Can include 'limit' for the number of articles.

        Returns:
            A list of dictionaries representing news articles.
        """
        limit = kwargs.get('limit', 10)
        logger.info(f"Fetching news data for symbol: {symbol} with limit: {limit}")
        return self.get_news_for_stock(symbol, limit=limit)

    def get_news_for_stock(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches news articles for a given stock symbol.

        Args:
            symbol: The stock symbol.
            limit: The maximum number of news articles to return.

        Returns:
            A list of dictionaries, where each dictionary represents a news article.
            (e.g., {'title': '...', 'source': '...', 'published_at': '...', 'summary': '...', 'sentiment_score_placeholder': 0.0})
            Returns mock data for now.
        """
        logger.info(f"Getting {limit} news articles for {symbol}")
        # Mock implementation:
        mock_articles = []
        sources = ["News Network A", "Financial Times Clone", "Tech Blog X", "Market Watcher Pro"]
        sentiments = [-0.5, 0.0, 0.2, 0.8, -0.1] # Placeholder sentiment scores

        for i in range(limit):
            pub_date = datetime.now() - timedelta(days=i, hours=i*2)
            mock_articles.append({
                'title': f"{symbol} in Focus: Market Update {i+1}",
                'source': sources[i % len(sources)],
                'published_at': pub_date.isoformat(),
                'summary': f"A mock summary about {symbol} discussing recent trends and market sentiment. This is placeholder content for article {i+1}.",
                'url': f"http://mocknews.com/{symbol}/article{i+1}",
                'sentiment_score_placeholder': sentiments[i % len(sentiments)] # Cycle through placeholder sentiments
            })
        
        logger.debug(f"Generated {len(mock_articles)} mock news articles for {symbol}.")
        return mock_articles

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This part is for quick testing and won't run when imported.
    # Ensure environment variable NEWS_API_KEY is set if you want to test API key logic.
    # export NEWS_API_KEY='your_news_test_key'

    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for more verbose output

    news_source = NewsDataSource() # Will try to use os.getenv("NEWS_API_KEY")

    print("\n--- Testing get_news_for_stock ---")
    articles_aapl = news_source.get_news_for_stock('AAPL', limit=3)
    for article in articles_aapl:
        print(article)

    print("\n--- Testing fetch_data (defaulting to get_news_for_stock) ---")
    articles_tsla = news_source.fetch_data('TSLA', limit=2)
    for article in articles_tsla:
        print(article)
    
    print("\n--- Testing with explicit API key ---")
    news_source_with_key = NewsDataSource(api_key="explicit_news_key_456")
    articles_goog = news_source_with_key.get_news_for_stock("GOOG", limit=1)
    print(articles_goog[0])

    print("\n--- Testing without API key (should show warning) ---")
    # To really test this, ensure NEWS_API_KEY is not set in the environment
    if os.getenv("NEWS_API_KEY"):
        print("SKIPPING: Test without API key requires NEWS_API_KEY to be unset.")
    else:
        news_source_no_key = NewsDataSource()
        articles_msft = news_source_no_key.get_news_for_stock("MSFT", limit=1)
        print(articles_msft[0])
        print("Check logs for warning about missing API key.")

    print("\nDone with NewsDataSource examples.")
