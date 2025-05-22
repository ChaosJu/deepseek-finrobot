import logging
from typing import Dict, Any, Optional

from .base_agent import BaseAgent
from contrarian_trader.src.data_sources import MarketDataSource # Adjusted import path

class MarketDataAgent(BaseAgent):
    """
    Agent responsible for fetching, processing, and providing market data.
    """

    def __init__(self, market_data_source: MarketDataSource, agent_name: str = "MarketDataAgent", logger_instance: Optional[logging.Logger] = None): # Renamed logger
        """
        Initializes the MarketDataAgent.

        Args:
            market_data_source: An instance of MarketDataSource to fetch data.
            agent_name: Name of the agent.
            logger_instance: Optional logger instance.
        """
        super().__init__(agent_name=agent_name, logger_instance=logger_instance) # Pass logger_instance
        if not isinstance(market_data_source, MarketDataSource):
            self.logger.error("MarketDataAgent requires a valid MarketDataSource instance.")
            raise ValueError("market_data_source must be an instance of MarketDataSource.")
        self.market_data_source = market_data_source
        self.processed_data: Optional[Dict[str, Any]] = None
        self.logger.info(f"MarketDataAgent initialized with {market_data_source.__class__.__name__}.")

    def process(self, data: Dict[str, Any]) -> None:
        """
        Processes fetched market data.
        For this agent, "processing" might involve cleaning, transforming, or caching the data.

        Args:
            data: A dictionary containing raw market data fetched by the data source.
                  Expected keys depend on the data source, e.g., 'historical_data', 'latest_price'.
        """
        self.logger.info(f"Processing market data for symbol: {data.get('symbol', 'N/A')}")
        # Placeholder: For now, just stores the data.
        # Real implementation could involve validation, transformation, feature engineering, etc.
        if not data:
            self.logger.warning("Received empty data to process.")
            self.processed_data = {}
            return
            
        self.processed_data = data
        self.processed_data['processed_timestamp'] = self.market_data_source.get_latest_price(data.get('symbol', 'N/A')).get('timestamp') # Example enrichment
        self.logger.debug(f"Market data processed: {self.processed_data}")

    def make_recommendation(self) -> Dict[str, Any]:
        """
        Provides the processed market data.
        This agent doesn't make "recommendations" in the trading sense, but rather provides data.

        Returns:
            A dictionary containing the processed market data, or an empty dict if no data.
        """
        if self.processed_data:
            self.logger.info("Providing processed market data.")
            return {"status": "success", "data_type": "market_data", "content": self.processed_data}
        else:
            self.logger.warning("No processed market data available to provide.")
            return {"status": "no_data", "data_type": "market_data", "content": {}}

    def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Orchestrates the fetching and processing of market data.

        Args:
            data: A dictionary containing parameters for fetching data,
                  e.g., {'symbol': 'AAPL', 'data_type': 'historical', 'start_date': '...', 'end_date': '...'}.
                  If 'data_type' is 'latest_price', it fetches the latest price.
                  If 'data_type' is 'historical', it fetches historical data.

        Returns:
            A dictionary containing the processed market data.
        """
        self.logger.info(f"MarketDataAgent run started with parameters: {data}")
        if not data or 'symbol' not in data or 'data_type' not in data:
            self.logger.error("Missing 'symbol' or 'data_type' in run parameters.")
            return {"status": "error", "message": "Missing 'symbol' or 'data_type' in run parameters."}

        symbol = data['symbol']
        data_type = data['data_type']
        raw_data = None

        try:
            if data_type == 'latest_price':
                raw_data = self.market_data_source.get_latest_price(symbol)
            elif data_type == 'historical':
                start_date = data.get('start_date')
                end_date = data.get('end_date')
                if not start_date or not end_date: # Use defaults from MarketDataSource if not provided
                    self.logger.info(f"Using default date range for historical data for {symbol}.")
                    raw_data = self.market_data_source.get_historical_data(symbol, start_date=None, end_date=None) # Assuming source handles None
                else:
                    raw_data = self.market_data_source.get_historical_data(symbol, start_date, end_date)
            else:
                self.logger.warning(f"Unsupported data_type: {data_type}. Attempting generic fetch_data.")
                # Fallback to generic fetch_data if specific method not matched
                raw_data = self.market_data_source.fetch_data(symbol, **data)


            if raw_data:
                # Pass a dictionary that includes the symbol for context in process()
                self.process({'symbol': symbol, 'raw_market_data': raw_data, 'data_type': data_type})
            else:
                self.logger.warning(f"No raw data fetched for {symbol} with data_type {data_type}.")
                self.processed_data = {'symbol': symbol, 'raw_market_data': None, 'data_type': data_type} # Ensure processed_data is not None

        except Exception as e:
            self.logger.error(f"Error during MarketDataAgent run for {symbol}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        
        recommendation = self.make_recommendation()
        self.logger.info(f"MarketDataAgent run finished. Result: {recommendation.get('status')}")
        return recommendation

if __name__ == '__main__':
    # This is for testing purposes and won't run when imported.
    # Setup basic logging for the test.
    logging.basicConfig(level=logging.DEBUG)
    
    # Mock MarketDataSource for testing
    class MockMarketDataSource(MarketDataSource):
        def __init__(self, api_key: str = "test_key"):
            super().__init__(api_key=api_key)
            self.logger.info("MockMarketDataSource initialized for testing.")

        def get_historical_data(self, symbol: str, start_date: Optional[str], end_date: Optional[str]) -> List[Dict[str, Any]]:
            self.logger.debug(f"Mock fetching historical for {symbol} from {start_date} to {end_date}")
            return [{'date': '2023-01-01', 'close': 100}, {'date': '2023-01-02', 'close': 101}]

        def get_latest_price(self, symbol: str) -> Dict[str, Any]:
            self.logger.debug(f"Mock fetching latest price for {symbol}")
            return {'symbol': symbol, 'price': 102.5, 'timestamp': '2023-01-02T16:00:00Z'}

    print("\n--- Testing MarketDataAgent ---")
    mock_source = MockMarketDataSource()
    market_agent = MarketDataAgent(market_data_source=mock_source)

    # Test run for historical data
    print("\nRunning agent for historical data:")
    historical_params = {'symbol': 'TESTHIST', 'data_type': 'historical', 'start_date': '2023-01-01', 'end_date': '2023-01-02'}
    result_hist = market_agent.run(data=historical_params)
    print(f"Result (historical): {result_hist}")

    # Test run for latest price
    print("\nRunning agent for latest price data:")
    latest_price_params = {'symbol': 'TESTLATEST', 'data_type': 'latest_price'}
    result_latest = market_agent.run(data=latest_price_params)
    print(f"Result (latest price): {result_latest}")

    # Test run with missing parameters
    print("\nRunning agent with missing parameters:")
    result_missing_params = market_agent.run(data={'symbol': 'TESTBAD'})
    print(f"Result (missing params): {result_missing_params}")
    
    # Test run with invalid data_type
    print("\nRunning agent with invalid data_type (should use fetch_data):")
    # Mocking fetch_data in MockMarketDataSource for this test
    def mock_fetch_data(symbol, **kwargs):
        return {"data": "generic fetch_data result", "symbol": symbol, "params": kwargs}
    mock_source.fetch_data = mock_fetch_data 
    invalid_type_params = {'symbol': 'TESTINVALID', 'data_type': 'invalid_type', 'some_param': 'value'}
    result_invalid_type = market_agent.run(data=invalid_type_params)
    print(f"Result (invalid type): {result_invalid_type}")


    # Test make_recommendation without run (should be empty)
    print("\nTesting make_recommendation without prior run:")
    # Re-initialize agent to ensure clean state
    market_agent_clean = MarketDataAgent(market_data_source=mock_source)
    direct_recommendation = market_agent_clean.make_recommendation()
    print(f"Direct recommendation (no data): {direct_recommendation}")

    print("\n--- MarketDataAgent tests complete ---")
