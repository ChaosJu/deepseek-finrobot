import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from .base_data_source import BaseDataSource

logger = logging.getLogger(__name__)

class MarketDataSource(BaseDataSource):
    """
    Data source for fetching market data (price, volume, etc.).
    """

    def __init__(self, api_key: str = None):
        """
        Initializes the market data source.
        Args:
            api_key: The API key for the market data provider.
                     If None, attempts to read from MARKET_API_KEY environment variable.
        """
        resolved_api_key = api_key or os.getenv("MARKET_API_KEY")
        if not resolved_api_key:
            logger.warning("Market API key not provided and MARKET_API_KEY environment variable not set.")
            # Depending on the API, this might be a critical error or it might allow some free-tier access.
            # For now, we'll allow it to proceed, but real usage would likely require an API key.
        super().__init__(api_key=resolved_api_key)
        logger.info("MarketDataSource initialized.")

    def fetch_data(self, symbol: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetches generic market data for a given symbol.
        For now, this mock implementation returns historical data.

        Args:
            symbol: The stock symbol.
            **kwargs: Can include 'start_date', 'end_date' for historical data.

        Returns:
            A list of dictionaries representing market data points.
        """
        logger.info(f"Fetching market data for symbol: {symbol} with args: {kwargs}")
        # Mock implementation:
        start_date_str = kwargs.get('start_date', (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        end_date_str = kwargs.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        return self.get_historical_data(symbol, start_date_str, end_date_str)

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetches historical price/volume data for a given symbol between two dates.

        Args:
            symbol: The stock symbol.
            start_date: The start date for the historical data (YYYY-MM-DD).
            end_date: The end date for the historical data (YYYY-MM-DD).

        Returns:
            A list of dictionaries, where each dictionary represents a data point
            (e.g., {'date': '2023-01-01', 'open': 100, 'high': 105, 'low': 98, 'close': 102, 'volume': 10000}).
            Returns mock data for now.
        """
        logger.info(f"Getting historical data for {symbol} from {start_date} to {end_date}")
        # Mock implementation:
        mock_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        price = 100.0
        volume_base = 10000

        while current_date <= end_dt:
            price += (0.5 - (current_date.weekday() / 10)) # Some variation
            mock_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'open': round(price - 1, 2),
                'high': round(price + 2, 2),
                'low': round(price - 2, 2),
                'close': round(price, 2),
                'volume': int(volume_base * (1 + (current_date.weekday() * 0.1)))
            })
            current_date += timedelta(days=1)
        
        logger.debug(f"Generated {len(mock_data)} mock historical data points for {symbol}.")
        return mock_data

    def get_latest_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches the latest price for a given symbol.

        Args:
            symbol: The stock symbol.

        Returns:
            A dictionary representing the latest price data
            (e.g., {'symbol': 'AAPL', 'price': 150.25, 'timestamp': '...'})
            Returns mock data for now.
        """
        logger.info(f"Getting latest price for {symbol}")
        # Mock implementation:
        mock_price_data = {
            'symbol': symbol,
            'price': round(150.0 + (hash(symbol) % 10) - 5, 2), # Consistent mock price based on symbol
            'timestamp': datetime.now().isoformat()
        }
        logger.debug(f"Generated mock latest price for {symbol}: {mock_price_data}")
        return mock_price_data

# Example usage (for testing purposes, normally not here)
if __name__ == '__main__':
    # This part is for quick testing and won't run when imported.
    # Ensure environment variable MARKET_API_KEY is set if you want to test API key logic.
    # export MARKET_API_KEY='your_test_key' 
    
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for more verbose output during testing
    
    market_source = MarketDataSource() # Will try to use os.getenv("MARKET_API_KEY")
    
    print("\n--- Testing get_historical_data ---")
    historical_data = market_source.get_historical_data('AAPL', '2023-10-01', '2023-10-05')
    for day_data in historical_data:
        print(day_data)
        
    print("\n--- Testing get_latest_price ---")
    latest_price_aapl = market_source.get_latest_price('AAPL')
    print(latest_price_aapl)
    latest_price_goog = market_source.get_latest_price('GOOG')
    print(latest_price_goog)

    print("\n--- Testing fetch_data (defaulting to historical) ---")
    fetch_data_result = market_source.fetch_data('MSFT', start_date='2023-11-01', end_date='2023-11-03')
    for day_data in fetch_data_result:
        print(day_data)

    print("\n--- Testing fetch_data (no date, should use defaults) ---")
    fetch_data_default_result = market_source.fetch_data('TSLA')
    print(f"Fetched {len(fetch_data_default_result)} data points for TSLA with default dates.")
    # print(fetch_data_default_result[0]) # Print first element to check structure

    print("\n--- Testing with explicit API key ---")
    market_source_with_key = MarketDataSource(api_key="explicit_test_key_123")
    # This just tests initialization, actual API call with key is not mocked here beyond super().__init__
    data_with_key = market_source_with_key.get_latest_price("NVDA")
    print(data_with_key)

    print("\n--- Testing without API key (should show warning) ---")
    # To really test this, ensure MARKET_API_KEY is not set in the environment
    # For example, run: unset MARKET_API_KEY
    # Then run the script.
    if os.getenv("MARKET_API_KEY"):
        print("SKIPPING: Test without API key requires MARKET_API_KEY to be unset.")
    else:
        market_source_no_key = MarketDataSource()
        data_no_key = market_source_no_key.get_latest_price("AMD")
        print(data_no_key)
        print("Check logs for warning about missing API key.")
    
    print("\nDone with MarketDataSource examples.")
