import logging
from typing import Dict, List, Any, Optional

# Attempt to import setup_logger and config variables
try:
    from contrarian_trader.src.utils import setup_logger
    from contrarian_trader.src.utils.config import LOG_LEVEL, LOG_FILE
except ImportError:
    # Fallback for standalone testing or if paths are not perfectly set yet
    # This allows the file to be parsable and testable even if imports fail initially
    # during development. A more robust solution might involve better path management or stubs.
    print("TradingSimulator: Could not import setup_logger or config. Using basic logging.")
    logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_LEVEL = "INFO"
    LOG_FILE = None
    
    # Dummy setup_logger if import fails
    def setup_logger(name, log_level=None, log_file=None, **kwargs):
        logger = logging.getLogger(name)
        if not logger.handlers: # Avoid adding handlers multiple times
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level or "INFO")
        return logger

class TradingSimulator:
    """
    Simulates trading operations, managing a portfolio and cash.
    """

    def __init__(self, initial_capital: float = 100000.0, logger_name: str = "TradingSimulator"):
        """
        Initializes the TradingSimulator.

        Args:
            initial_capital: The starting cash balance.
            logger_name: The name for the logger instance.
        """
        self.cash: float = initial_capital
        self.portfolio: Dict[str, Dict[str, float]] = {}  # e.g., {'AAPL': {'shares': 10, 'average_price': 150.0}}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Use LOG_LEVEL and LOG_FILE from config for the logger if available
        self.logger = setup_logger(logger_name, log_level=LOG_LEVEL, log_file=LOG_FILE)
        self.logger.info(f"TradingSimulator initialized with initial capital: ${initial_capital:,.2f}")

    def execute_trade(self, stock_symbol: str, signal: str, quantity: int, price: float, reason: str = "N/A") -> bool:
        """
        Executes a trade (BUY or SELL) and updates the portfolio and cash.

        Args:
            stock_symbol: The stock symbol (e.g., 'AAPL').
            signal: 'BUY' or 'SELL'.
            quantity: The number of shares to trade.
            price: The price per share.
            reason: Optional reason for the trade.

        Returns:
            True if the trade was successful, False otherwise.
        """
        signal = signal.upper()
        trade_cost = quantity * price

        if quantity <= 0 or price <= 0:
            self.logger.error(f"Trade execution failed for {stock_symbol}: Quantity and price must be positive. Got Qty: {quantity}, Price: {price}")
            return False

        if signal == 'BUY':
            if self.cash >= trade_cost:
                self.cash -= trade_cost
                
                # Update portfolio
                if stock_symbol in self.portfolio:
                    current_shares = self.portfolio[stock_symbol]['shares']
                    current_avg_price = self.portfolio[stock_symbol]['average_price']
                    new_total_shares = current_shares + quantity
                    new_total_cost = (current_avg_price * current_shares) + trade_cost
                    self.portfolio[stock_symbol]['average_price'] = new_total_cost / new_total_shares
                    self.portfolio[stock_symbol]['shares'] = new_total_shares
                else:
                    self.portfolio[stock_symbol] = {'shares': float(quantity), 'average_price': price}
                
                trade_log = {
                    'type': 'BUY', 'symbol': stock_symbol, 'quantity': quantity, 
                    'price': price, 'total_cost': trade_cost, 'timestamp': self._current_timestamp(),
                    'reason': reason, 'cash_remaining': self.cash
                }
                self.trade_history.append(trade_log)
                self.logger.info(f"Executed BUY: {quantity} shares of {stock_symbol} @ ${price:,.2f}. Cost: ${trade_cost:,.2f}. Reason: {reason}. Cash: ${self.cash:,.2f}")
                return True
            else:
                self.logger.error(f"BUY failed for {stock_symbol}: Insufficient cash. Need ${trade_cost:,.2f}, have ${self.cash:,.2f}.")
                return False

        elif signal == 'SELL':
            if stock_symbol in self.portfolio and self.portfolio[stock_symbol]['shares'] >= quantity:
                proceeds = trade_cost # For sell, 'trade_cost' is actually proceeds
                self.cash += proceeds
                
                original_avg_price = self.portfolio[stock_symbol]['average_price']
                cost_of_shares_sold = original_avg_price * quantity
                profit_loss = proceeds - cost_of_shares_sold
                
                self.portfolio[stock_symbol]['shares'] -= quantity
                if self.portfolio[stock_symbol]['shares'] == 0:
                    del self.portfolio[stock_symbol] # Remove if no shares left
                
                trade_log = {
                    'type': 'SELL', 'symbol': stock_symbol, 'quantity': quantity, 
                    'price': price, 'total_proceeds': proceeds, 'pnl': profit_loss, 
                    'timestamp': self._current_timestamp(), 'reason': reason,
                    'cash_remaining': self.cash
                }
                self.trade_history.append(trade_log)
                self.logger.info(f"Executed SELL: {quantity} shares of {stock_symbol} @ ${price:,.2f}. Proceeds: ${proceeds:,.2f}. P&L: ${profit_loss:,.2f}. Reason: {reason}. Cash: ${self.cash:,.2f}")
                return True
            else:
                current_shares = self.portfolio.get(stock_symbol, {}).get('shares', 0)
                self.logger.error(f"SELL failed for {stock_symbol}: Insufficient shares. Need {quantity}, have {current_shares}.")
                return False
        else:
            self.logger.error(f"Trade execution failed: Unknown signal '{signal}' for {stock_symbol}.")
            return False

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculates the current total value of the portfolio (cash + holdings).

        Args:
            current_prices: A dictionary mapping stock symbols to their current prices.
                            e.g., {'AAPL': 170.50, 'MSFT': 300.00}

        Returns:
            The total portfolio value.
        """
        holdings_value = 0.0
        for symbol, data in self.portfolio.items():
            current_price = current_prices.get(symbol, 0.0) # Default to 0 if price not found for a holding
            if current_price == 0.0:
                 self.logger.warning(f"Price for {symbol} not found in current_prices for portfolio valuation. Using $0.")
            holdings_value += data['shares'] * current_price
        
        total_value = self.cash + holdings_value
        return total_value

    def log_portfolio_summary(self, current_prices: Dict[str, float]):
        """
        Logs a summary of the current portfolio status.
        """
        self.logger.info("--- Portfolio Summary ---")
        self.logger.info(f"Cash: ${self.cash:,.2f}")
        
        if not self.portfolio:
            self.logger.info("Holdings: No shares held.")
        else:
            self.logger.info("Holdings:")
            for symbol, data in self.portfolio.items():
                shares = data['shares']
                avg_price = data['average_price']
                current_price = current_prices.get(symbol, data['average_price']) # Use avg_price if current not available
                market_value = shares * current_price
                if current_prices.get(symbol) is None:
                    self.logger.warning(f"Current price for {symbol} not provided for summary; using average cost.")
                
                self.logger.info(
                    f"  {symbol}: {shares} shares @ avg ${avg_price:,.2f} | "
                    f"Current Price: ${current_price:,.2f} | Market Value: ${market_value:,.2f}"
                )
        
        total_value = self.get_portfolio_value(current_prices)
        self.logger.info(f"Total Portfolio Value: ${total_value:,.2f}")
        self.logger.info("-------------------------")

    def _current_timestamp(self) -> str:
        """Helper to get a consistent timestamp format."""
        from datetime import datetime
        return datetime.now().isoformat()

if __name__ == '__main__':
    # Example Usage (assuming setup_logger and config are correctly imported or mocked)
    print("--- Testing TradingSimulator ---")
    
    # Setup a test logger that prints to console for this example
    # This will use the dummy setup_logger if imports failed at the top
    test_sim_logger_name = "TestSim"
    if 'contrarian_trader' not in sys.modules: # Crude check if we are in a test/dev standalone mode
        global LOG_LEVEL, LOG_FILE
        LOG_LEVEL = "DEBUG" # More verbose for testing
        LOG_FILE = "test_simulator.log" 
        print(f"Standalone mode: Setting LOG_LEVEL={LOG_LEVEL}, LOG_FILE={LOG_FILE}")
        
    simulator = TradingSimulator(initial_capital=50000.0, logger_name=test_sim_logger_name)
    
    # Mock current prices for stocks
    mock_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOG': 2500.0}

    # Log initial state
    simulator.log_portfolio_summary(mock_prices)

    # Buy AAPL
    print("\n1. Buying AAPL...")
    simulator.execute_trade('AAPL', 'BUY', 10, 148.0, reason="Contrarian signal")
    simulator.log_portfolio_summary(mock_prices)
    
    # Buy MSFT
    print("\n2. Buying MSFT...")
    simulator.execute_trade('MSFT', 'BUY', 5, 295.0, reason="Analyst upgrade")
    simulator.log_portfolio_summary(mock_prices)

    # Attempt to buy GOOG with insufficient funds
    print("\n3. Attempting to buy GOOG (insufficient funds)...")
    simulator.execute_trade('GOOG', 'BUY', 20, 2500.0, reason="FOMO") # 20 * 2500 = 50000, likely not enough
    simulator.log_portfolio_summary(mock_prices)
    
    # Update prices and sell some AAPL
    mock_prices['AAPL'] = 155.0 # AAPL price went up
    print("\n4. Selling some AAPL (price increased)...")
    simulator.execute_trade('AAPL', 'SELL', 5, 155.0, reason="Take partial profit")
    simulator.log_portfolio_summary(mock_prices)

    # Sell all MSFT
    mock_prices['MSFT'] = 290.0 # MSFT price went down
    print("\n5. Selling all MSFT (price decreased)...")
    simulator.execute_trade('MSFT', 'SELL', 5, 290.0, reason="Stop loss triggered")
    simulator.log_portfolio_summary(mock_prices)

    # Attempt to sell shares not owned
    print("\n6. Attempting to sell GOOG (not owned)...")
    simulator.execute_trade('GOOG', 'SELL', 5, 2500.0, reason="Mistake")
    simulator.log_portfolio_summary(mock_prices)

    # Attempt to sell more AAPL than owned
    print("\n7. Attempting to sell more AAPL than owned...")
    simulator.execute_trade('AAPL', 'SELL', 100, 155.0, reason="Greed") # Own 5 shares
    simulator.log_portfolio_summary(mock_prices)

    # Buy more AAPL (testing average price update)
    mock_prices['AAPL'] = 140.0
    print("\n8. Buying more AAPL at a different price...")
    simulator.execute_trade('AAPL', 'BUY', 10, 140.0, reason="Buying the dip")
    simulator.log_portfolio_summary(mock_prices) # Expected: 15 shares of AAPL, avg price should be updated

    print("\n--- Trade History ---")
    for trade in simulator.trade_history:
        print(trade)
    
    print(f"\nFinal portfolio value with AAPL at $140: ${simulator.get_portfolio_value(mock_prices):,.2f}")
    mock_prices['AAPL'] = 160.0 # Assume AAPL recovers
    print(f"Final portfolio value with AAPL at $160: ${simulator.get_portfolio_value(mock_prices):,.2f}")
    
    print("\n--- TradingSimulator tests complete. Check 'test_simulator.log' if created. ---")
    # Clean up log file if it was created by the dummy logger for test
    import os, sys
    if 'contrarian_trader' not in sys.modules and os.path.exists("test_simulator.log"):
        # os.remove("test_simulator.log") # Optional: remove test log
        pass
