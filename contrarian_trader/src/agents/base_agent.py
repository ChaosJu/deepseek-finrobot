import abc
import logging
from typing import Dict, Any, Optional

# Attempt to import setup_logger and config variables
# This is to ensure BaseAgent can initialize its logger correctly.
try:
    from contrarian_trader.src.utils import setup_logger
    from contrarian_trader.src.utils.config import LOG_LEVEL, LOG_FILE
except ImportError:
    # Fallback for standalone testing or if paths are not perfectly set yet
    print("BaseAgent: Could not import setup_logger or config. Using basic logging.")
    logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_LEVEL = "INFO" # Default if config not loaded
    LOG_FILE = None    # Default if config not loaded
    
    # Dummy setup_logger if import fails, so the class can be defined
    def setup_logger(name, log_level=None, log_file=None, **kwargs):
        logger_instance = logging.getLogger(name)
        if not logger_instance.handlers: # Avoid adding handlers multiple times
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
            logger_instance.setLevel(log_level or "INFO")
        return logger_instance

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the trading bot.
    """

    def __init__(self, agent_name: str, logger_instance: Optional[logging.Logger] = None):
        """
        Initializes the base agent.

        Args:
            agent_name: The name of the agent.
            logger_instance: An optional, pre-configured logger instance. 
                             If None, a new logger is created using setup_logger.
        """
        self.agent_name = agent_name
        # Use the provided logger instance or set up a new one using the global config
        if logger_instance:
            self.logger = logger_instance
        else:
            # Use LOG_LEVEL and LOG_FILE from config for the logger if available
            self.logger = setup_logger(f"Agent.{self.agent_name}", log_level=LOG_LEVEL, log_file=LOG_FILE)
        
        self.logger.info(f"Agent {self.agent_name} initialized.")

    @abc.abstractmethod
    def process(self, data: Dict[str, Any]) -> None:
        """
        Processes input data.
        This method should be implemented by subclasses to handle data specific to the agent.

        Args:
            data: A dictionary containing input data for the agent.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        self.logger.debug(f"Processing data: {data}")
        raise NotImplementedError(f"{self.agent_name} has not implemented the process method.")

    @abc.abstractmethod
    def make_recommendation(self) -> Dict[str, Any]:
        """
        Generates a recommendation, analysis, or decision based on processed data.

        Returns:
            A dictionary containing the agent's output (e.g., a trading signal, analysis summary).

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        self.logger.debug("Making recommendation.")
        raise NotImplementedError(f"{self.agent_name} has not implemented the make_recommendation method.")

    @abc.abstractmethod
    def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main method to orchestrate the agent's logic.
        This typically involves processing data and then making a recommendation.

        Args:
            data: Optional input data to be processed by the agent.

        Returns:
            A dictionary containing the result of the agent's run (e.g., recommendation, analysis).

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        self.logger.info(f"Running agent {self.agent_name} with data: {data is not None}")
        raise NotImplementedError(f"{self.agent_name} has not implemented the run method.")

if __name__ == '__main__':
    # Example of how a subclass might look (for testing BaseAgent structure)
    class MockAgent(BaseAgent):
        def __init__(self):
            # For MockAgent, explicitly pass None for logger_instance to test default setup
            super().__init__("MockAgent", logger_instance=None) 
            self.internal_data_store = None

        def process(self, data: Dict[str, Any]) -> None:
            self.logger.info(f"MockAgent processing data: {data}")
            self.internal_data_store = data
            # Perform some mock processing
            self.internal_data_store['processed'] = True

        def make_recommendation(self) -> Dict[str, Any]:
            if self.internal_data_store and self.internal_data_store.get('processed'):
                self.logger.info("MockAgent making recommendation based on processed data.")
                return {"recommendation": "mock_action", "confidence": 0.9, "data": self.internal_data_store}
            else:
                self.logger.warning("MockAgent: No data processed to make a recommendation.")
                return {"recommendation": "hold", "confidence": 0.0, "reason": "No data processed."}

        def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            self.logger.info(f"MockAgent run started with data: {data is not None}")
            if data:
                self.process(data)
            recommendation = self.make_recommendation()
            self.logger.info(f"MockAgent run finished. Recommendation: {recommendation}")
            return recommendation

    print("--- Testing MockAgent that inherits from BaseAgent ---")
    mock_agent = MockAgent()
    
    # Test run without initial data (relies on internal state or default behavior)
    print("\nRunning agent without initial data:")
    result_no_data = mock_agent.run()
    print(f"Result (no data): {result_no_data}")

    # Test run with initial data
    print("\nRunning agent with initial data:")
    sample_data = {"symbol": "TEST", "price": 100}
    result_with_data = mock_agent.run(data=sample_data)
    print(f"Result (with data): {result_with_data}")

    # Test individual methods
    print("\nTesting process method directly:")
    mock_agent.process({"new_data": "some_value"})
    print(f"Agent's internal data after process: {mock_agent.internal_data_store}")

    print("\nTesting make_recommendation method directly:")
    recommendation = mock_agent.make_recommendation()
    print(f"Direct recommendation: {recommendation}")
    
    print("\n--- BaseAgent tests complete ---")
