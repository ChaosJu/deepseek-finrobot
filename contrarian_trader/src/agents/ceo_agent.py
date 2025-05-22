import logging
from typing import Dict, Any, Optional, List, Type

from .base_agent import BaseAgent

class CEOAgent(BaseAgent):
    """
    The CEO Agent orchestrates the workflow of other specialized agents
    to arrive at a final trading decision or market assessment.
    """

    def __init__(self, 
                 strategy_agents: Optional[List[BaseAgent]] = None, # Agents like ContrarianStrategyAgent
                 data_providing_agents: Optional[List[BaseAgent]] = None, # Agents like MarketDataAgent, NewsAgent
                 agent_name: str = "CEOAgent", 
                 logger_instance: Optional[logging.Logger] = None): # Renamed logger to logger_instance
        """
        Initializes the CEOAgent.

        Args:
            strategy_agents: A list of strategy agents (e.g., ContrarianStrategyAgent) that generate signals.
            data_providing_agents: A list of agents that fetch and process data.
            agent_name: Name of the agent.
            logger_instance: Optional, pre-configured logger instance.
        """
        super().__init__(agent_name=agent_name, logger_instance=logger_instance) # Pass logger_instance to super
        
        self.strategy_agents = strategy_agents if strategy_agents is not None else []
        self.data_providing_agents = data_providing_agents if data_providing_agents is not None else []
        
        self.final_decision_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"CEOAgent initialized with {len(self.strategy_agents)} strategy agent(s) and {len(self.data_providing_agents)} data provider(s).")
        for agent in self.strategy_agents:
            self.logger.debug(f"  Registered strategy agent: {agent.agent_name}")
        for agent in self.data_providing_agents:
            self.logger.debug(f"  Registered data providing agent: {agent.agent_name}")


    def register_strategy_agent(self, agent: BaseAgent):
        """Registers a new strategy agent."""
        if agent not in self.strategy_agents:
            self.strategy_agents.append(agent)
            self.logger.info(f"Strategy agent {agent.agent_name} registered with CEOAgent.")
        else:
            self.logger.warning(f"Strategy agent {agent.agent_name} already registered.")

    def register_data_providing_agent(self, agent: BaseAgent):
        """Registers a new data providing agent."""
        if agent not in self.data_providing_agents:
            self.data_providing_agents.append(agent)
            self.logger.info(f"Data providing agent {agent.agent_name} registered with CEOAgent.")
        else:
            self.logger.warning(f"Data providing agent {agent.agent_name} already registered.")

    def process(self, data: Dict[str, Any]) -> None:
        """
        Processes input data, which for the CEO agent typically means orchestrating
        its registered agents. The core logic is within the `run` method for CEO.

        Args:
            data: A dictionary containing input data, often including the target (e.g., stock symbol).
                  Example: {'stock_symbol': 'AAPL', 'parameters': {...}}
        """
        self.logger.info(f"CEOAgent processing task for: {data.get('stock_symbol', 'N/A')}")
        # The main orchestration logic is in `run`.
        # This method could be used for preliminary checks or logging if needed.
        if not data or 'stock_symbol' not in data:
            self.logger.error("CEOAgent: 'stock_symbol' is required in data for processing.")
            # Potentially set an internal error state or log for make_recommendation to pick up.
            return
        
        # Store or log parameters if necessary
        self.logger.debug(f"Task parameters: {data.get('parameters')}")


    def make_recommendation(self) -> Dict[str, Any]:
        """
        Makes a final decision based on the recommendations from strategy agents.
        This is a placeholder and would involve more sophisticated decision logic.

        Returns:
            A dictionary representing the final decision (e.g., trade signal, market assessment).
        """
        self.logger.info("CEOAgent making final recommendation based on strategy agent outputs.")
        
        if not self.final_decision_history: # Check if run has produced a decision
            self.logger.warning("No decision history available. Run the agent first or check for processing errors.")
            return {"final_signal": "hold_no_data", "confidence": 0.0, "reason": "No analysis performed by CEO."}

        # For this placeholder, let's assume the last decision recorded is the current one.
        last_decision = self.final_decision_history[-1]
        
        self.logger.info(f"Final decision: {last_decision.get('final_signal')} with confidence {last_decision.get('confidence')}")
        return last_decision


    def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main orchestration logic for the CEOAgent.
        1. Gathers data using data-providing agents.
        2. Passes data to strategy agents.
        3. Synthesizes results from strategy agents to make a final decision.

        Args:
            data: Input for the run, typically including 'stock_symbol' and any parameters
                  for data fetching or strategy execution.
                  Example: {'stock_symbol': 'MSFT', 
                            'market_data_params': {'data_type': 'historical', 'days': 30},
                            'news_params': {'limit': 10}
                           }
        Returns:
            A dictionary containing the final decision and supporting data.
        """
        self.logger.info(f"CEOAgent run started with data: {data is not None}")
        if not data or 'stock_symbol' not in data:
            self.logger.error("CEOAgent run requires 'stock_symbol' in input data.")
            final_decision = {"final_signal": "error", "reason": "Missing stock_symbol for CEO run."}
            self.final_decision_history.append(final_decision)
            return final_decision

        stock_symbol = data['stock_symbol']
        
        # 1. Gather data using data-providing agents
        # This is a simplified version. A real implementation would handle parameters for each agent.
        all_gathered_data = {'stock_symbol': stock_symbol}
        self.logger.info(f"Gathering data for {stock_symbol} using data-providing agents...")
        for dp_agent in self.data_providing_agents:
            self.logger.debug(f"Running data-providing agent: {dp_agent.agent_name}")
            try:
                # Construct parameters for each data providing agent based on 'data'
                # Example: agent_params = data.get(f"{dp_agent.agent_name}_params", {'symbol': stock_symbol})
                # For simplicity, we assume data providing agents take 'symbol' and type-specific params
                agent_run_params = {'symbol': stock_symbol, **data.get(f"{dp_agent.agent_name}_params", {})}

                # Ensure 'data_type' or similar is present if required by the agent
                # This is a bit of a hack for the current MarketDataAgent structure
                if "MarketDataAgent" in dp_agent.agent_name and 'data_type' not in agent_run_params:
                    agent_run_params['data_type'] = 'historical' # Default or make configurable

                agent_output = dp_agent.run(agent_run_params)
                # Store data under a key related to the agent's name or type
                all_gathered_data[dp_agent.agent_name.lower().replace("agent", "_data").strip("_")] = agent_output
            except Exception as e:
                self.logger.error(f"Error running data-providing agent {dp_agent.agent_name}: {e}", exc_info=True)
                all_gathered_data[dp_agent.agent_name.lower().replace("agent", "_data").strip("_")] = {"status": "error", "message": str(e)}
        
        self.logger.info(f"Data gathering complete for {stock_symbol}.")
        self.logger.debug(f"All gathered data: { {k: v.get('status', 'N/A') if isinstance(v,dict) else v for k,v in all_gathered_data.items()} }") # Log status or type

        # 2. Pass data to strategy agents
        strategy_outputs = []
        self.logger.info(f"Running strategy agents for {stock_symbol}...")
        for strat_agent in self.strategy_agents:
            self.logger.debug(f"Running strategy agent: {strat_agent.agent_name}")
            try:
                # Strategy agents receive all gathered data
                strategy_result = strat_agent.run(all_gathered_data.copy()) # Pass a copy
                strategy_outputs.append({"agent_name": strat_agent.agent_name, "output": strategy_result})
            except Exception as e:
                self.logger.error(f"Error running strategy agent {strat_agent.agent_name}: {e}", exc_info=True)
                strategy_outputs.append({"agent_name": strat_agent.agent_name, "output": {"signal": "error", "reason": str(e)}})

        self.logger.info(f"Strategy agents execution complete for {stock_symbol}.")

        # 3. Synthesize results (Placeholder Logic)
        # This is where the CEO's "intelligence" would be.
        # For now, a very simple synthesis: if any strategy agent says "buy", CEO says "buy".
        # If any says "sell", CEO says "sell" (unless a buy was already found). Otherwise "hold".
        final_signal = "hold"
        final_confidence = 0.0
        supporting_reasons = []
        
        # Prioritize signals: buy > sell > hold
        for res in strategy_outputs:
            agent_name = res['agent_name']
            output_signal = res['output'].get('signal', 'hold').lower()
            output_confidence = res['output'].get('confidence', 0.0)
            output_reason = res['output'].get('reason', f"Signal from {agent_name}")
            
            self.logger.info(f"Signal from {agent_name}: {output_signal.upper()}, Confidence: {output_confidence:.2f}, Reason: {output_reason}")

            if output_signal == "buy":
                if final_signal != "buy": # First buy signal takes precedence or highest confidence buy
                    final_signal = "buy"
                    final_confidence = output_confidence
                elif output_confidence > final_confidence: # Higher confidence buy replaces lower
                    final_confidence = output_confidence
                supporting_reasons.append(f"Buy signal from {agent_name}: {output_reason} (Conf: {output_confidence:.2f})")
            
            elif output_signal == "sell":
                if final_signal == "hold": # Sell takes precedence over hold
                    final_signal = "sell"
                    final_confidence = output_confidence
                elif final_signal == "sell" and output_confidence > final_confidence: # Higher confidence sell replaces lower
                    final_confidence = output_confidence
                supporting_reasons.append(f"Sell signal from {agent_name}: {output_reason} (Conf: {output_confidence:.2f})")

            elif output_signal == "hold":
                if final_signal == "hold": # If still hold, consider this hold's confidence
                     final_confidence = max(final_confidence, output_confidence)
                supporting_reasons.append(f"Hold signal from {agent_name}: {output_reason} (Conf: {output_confidence:.2f})")


        if not strategy_outputs:
            final_reason = "No strategy agents provided output."
            final_confidence = 0.0 # No confidence if no agents ran
        elif not supporting_reasons and strategy_outputs : # Agents ran but no reasons captured (edge case)
            final_reason = "Neutral or uncaptured signals from strategy agents."
            final_confidence = 0.5 # Default confidence for neutral outcome
        else:
            # Construct a more readable final reason
            pertinent_reasons = [r for r in supporting_reasons if final_signal in r.lower()]
            if not pertinent_reasons: # If final_signal is e.g. 'hold' by default, but no agent explicitly said 'hold' for it
                pertinent_reasons = supporting_reasons # Show all reasons
            final_reason = f"Final decision based on {len(strategy_outputs)} strategy agent(s). Key reason(s): " + "; ".join(pertinent_reasons)
            if final_signal == "hold" and not any(final_signal in r.lower() for r in supporting_reasons):
                 final_reason = "No strong buy/sell signals; defaulting to hold. " + "; ".join(supporting_reasons)


        final_decision_object = {
            "stock_symbol": stock_symbol,
            "final_signal": final_signal.upper(), # Ensure uppercase for consistency
            "confidence": round(final_confidence, 2),
            "reason": final_reason,
            "strategy_agent_outputs": strategy_outputs, # Detailed outputs from each strategy agent
            "supporting_data_summary": {
                k: v.get('status', 'N/A') if isinstance(v,dict) else type(v).__name__ 
                for k,v in all_gathered_data.items() if k != 'stock_symbol'
            }
        }
        
        self.final_decision_history.append(final_decision_object)
        self.logger.info(
            f"CEOAgent run finished for {stock_symbol}. "
            f"Final Decision: {final_decision_object['final_signal']}, "
            f"Confidence: {final_decision_object['confidence']:.2f}. "
            f"Reason: {final_decision_object['reason']}"
        )
        
        # The 'process' method is more about initial data intake for the agent itself,
        # not the primary orchestrator of sub-agents, which is `run`.
        # Calling self.process(data) here is mostly for logging the initial call if needed.
        # self.process(data) # Commented out as it's somewhat redundant after full run.
        
        return final_decision_object

if __name__ == '__main__':
    # Ensures that the logger for the main test script is set up if utils are available.
    # This is important if BaseAgent and other components rely on setup_logger from utils.
    try:
        from contrarian_trader.src.utils import setup_logger, config
        # Setup main logger for the test script itself
        script_logger = setup_logger(__name__, log_level=config.LOG_LEVEL, log_file=config.LOG_FILE)
        script_logger.info("CEOAgent test script starting. Logging configured via utils.")
    except ImportError:
        print("CEOAgent Test: Could not import utils.setup_logger or utils.config. Using basicConfig for script logging.")
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info("CEOAgent test script starting with basicConfig.")

    
    # --- Mock Agents and Data Sources for CEOAgent testing ---
    # These imports are moved inside `if __name__ == '__main__':` to avoid being top-level
    # when the module is imported elsewhere, especially if they also try to setup logging.
    from contrarian_trader.src.data_sources import MarketDataSource, NewsDataSource
    from .market_data_agent import MarketDataAgent
    from .news_agent import NewsAgent
    from .contrarian_strategy_agent import ContrarianStrategyAgent
    from contrarian_trader.src.core_logic.contrarian_analyzer import ContrarianAnalyzer

    # Mock Data Sources
    class MockMarketDataSource(MarketDataSource):
        def get_historical_data(self, symbol, start_date, end_date): 
            self.logger.debug(f"MockMarketDataSource: get_historical_data for {symbol}")
            return [{"date": "2023-01-01", "close": 100, "volume": 10000}]
        def get_latest_price(self, symbol): 
            self.logger.debug(f"MockMarketDataSource: get_latest_price for {symbol}")
            return {"symbol": symbol, "price": 101, "timestamp": "now"}
        # fetch_data needs to handle being called by MarketDataAgent if data_type is not specific
        def fetch_data(self, symbol, **kwargs):
            self.logger.debug(f"MockMarketDataSource: fetch_data for {symbol} with {kwargs}")
            if kwargs.get('data_type') == 'historical' or not kwargs.get('data_type'): # Default to historical if generic
                return self.get_historical_data(symbol, kwargs.get('start_date'), kwargs.get('end_date'))
            return self.get_latest_price(symbol)


    class MockNewsDataSource(NewsDataSource):
        def fetch_data(self, symbol, **kwargs): 
            self.logger.debug(f"MockNewsDataSource: fetch_data for {symbol} with {kwargs}")
            return [{"title": f"News for {symbol}", "summary": "Good news.", 'sentiment_score_placeholder': 0.5, 'source_type': 'major'}]

    # Mock Analyzer for ContrarianStrategyAgent
    class MockContrarianAnalyzer(ContrarianAnalyzer):
        # We need to override all methods that ContrarianStrategyAgent's process() will call on the analyzer
        def analyze_sentiment_divergence(self, news_articles: List[Dict[str, Any]], social_media_posts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            is_neg = any(article.get('sentiment_score_placeholder', 0) < -0.2 for article in news_articles)
            return {'average_sentiment': -0.5 if is_neg else 0.1, 'is_negative_sentiment_dominant': is_neg, 'processed_articles_count': len(news_articles)}

        def analyze_price_volume_anomaly(self, market_data_history: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
            # Assume positive anomaly if "GOOD" or "NEUTRAL" in symbol for test simplicity
            is_good_pv = "GOOD" in market_data_history[0].get('symbol_for_test', "") if market_data_history else False
            is_neutral_pv = "NEUTRAL" in market_data_history[0].get('symbol_for_test', "") if market_data_history else False

            return {'is_high_volume_accumulation': is_good_pv or is_neutral_pv, 'is_price_stable_or_rising': is_good_pv or is_neutral_pv, 'reason': "Mock PV analysis"}

        def identify_market_trap(self, market_data_history: List[Dict[str, Any]], current_sentiment: Dict[str, Any]) -> Dict[str, Any]:
            return {'trap_detected': False, 'trap_type': None, 'reason': "Mock trap - none detected"}
        
        # generate_overall_signal will be called by ContrarianStrategyAgent with results from above.
        # It uses the actual logic from ContrarianAnalyzer, so this mock doesn't need to reimplement it
        # unless we want to test specific inputs to generate_overall_signal itself.
        # For CEOAgent test, we rely on the ContrarianStrategyAgent to use the actual analyzer's generate_overall_signal.
        # For simplicity in testing CEO, let's make ContrarianStrategyAgent's output predictable based on symbol.
        # This means the ContrarianStrategyAgent itself should use a mock of ContrarianAnalyzer that provides this.
        
        # Let's ensure ContrarianStrategyAgent gets a MockContrarianAnalyzer that directly outputs signals
        # This simplifies testing CEO's aggregation logic.
        def generate_overall_signal(self, stock_symbol: str, news_sentiment_analysis: Dict[str, Any], 
                                    price_volume_analysis: Dict[str, Any], market_trap_analysis: Dict[str, Any]) -> Dict[str, Any]:
            # This method in MockContrarianAnalyzer can be simplified for CEO test
            if "GOOD" in stock_symbol:
                return {"signal": "BUY", "confidence": 0.8, "reason": "MockAnalyzer: Detected bullish setup via overall."}
            elif "BAD" in stock_symbol:
                 return {"signal": "SELL", "confidence": 0.7, "reason": "MockAnalyzer: Detected bearish setup via overall."}
            return {"signal": "HOLD", "confidence": 0.5, "reason": "MockAnalyzer: Neutral setup via overall."}


    logging.info("\n--- Initializing Mock Components for CEOAgent Test ---")
    # Ensure data sources also get loggers if they use them internally for debugging
    mock_market_source = MockMarketDataSource(api_key="mock_market_key") 
    mock_news_source = MockNewsDataSource(api_key="mock_news_key")
    
    # Pass None for logger_instance to make agents use default setup_logger via BaseAgent
    market_data_agent = MarketDataAgent(market_data_source=mock_market_source, logger_instance=None)
    news_agent = NewsAgent(news_data_source=mock_news_source, logger_instance=None) 
    
    # ContrarianStrategyAgent will use the MockContrarianAnalyzer defined above
    mock_contrarian_analyzer_for_strat_agent = MockContrarianAnalyzer()
    contrarian_strategy_agent = ContrarianStrategyAgent(
        contrarian_analyzer=mock_contrarian_analyzer_for_strat_agent, 
        logger_instance=None
    )

    logging.info("\n--- Initializing CEOAgent ---")
    ceo = CEOAgent(
        strategy_agents=[contrarian_strategy_agent],
        data_providing_agents=[market_data_agent, news_agent],
        logger_instance=None # CEO uses its own logger via BaseAgent
    )

    # --- Test Case 1: Good Stock ---
    logging.info("\n--- CEOAgent Test Case 1: GOOD_STOCK ---")
    # Add symbol_for_test to mock market data for MockContrarianAnalyzer's PV logic
    mock_market_source.get_historical_data = lambda symbol, start_date, end_date: [{"date": "2023-01-01", "close": 100, "volume": 10000, "symbol_for_test": symbol}]
    
    ceo_input_good = {
        "stock_symbol": "GOOD_STOCK_XYZ",
        # MarketDataAgent needs data_type for its run method
        "market_data_agent_params": {"data_type": "historical"}, 
        "news_agent_params": {"limit": 3}, # Params for NewsAgent
    }
    final_decision_good = ceo.run(data=ceo_input_good)
    logging.info(f"CEO Final Decision (GOOD_STOCK_XYZ): {final_decision_good.get('final_signal')}, Conf: {final_decision_good.get('confidence')}")
    # logging.debug(f"Full output (GOOD): {final_decision_good}") 
    assert final_decision_good.get('final_signal') == 'BUY'


    # --- Test Case 2: Bad Stock ---
    logging.info("\n--- CEOAgent Test Case 2: BAD_STOCK_ABC ---")
    ceo_input_bad = {
        "stock_symbol": "BAD_STOCK_ABC",
        "market_data_agent_params": {"data_type": "historical"},
        "news_agent_params": {"limit": 1},
    }
    final_decision_bad = ceo.run(data=ceo_input_bad)
    logging.info(f"CEO Final Decision (BAD_STOCK_ABC): {final_decision_bad.get('final_signal')}, Conf: {final_decision_bad.get('confidence')}")
    assert final_decision_bad.get('final_signal') == 'SELL'
    
    # --- Test Case 3: Neutral Stock ---
    logging.info("\n--- CEOAgent Test Case 3: NEUTRAL_STOCK ---")
    ceo_input_neutral = {
        "stock_symbol": "NEUTRAL_STOCK",
        "market_data_agent_params": {"data_type": "historical"},
    }
    final_decision_neutral = ceo.run(data=ceo_input_neutral)
    logging.info(f"CEO Final Decision (NEUTRAL_STOCK): {final_decision_neutral.get('final_signal')}, Conf: {final_decision_neutral.get('confidence')}")
    assert final_decision_neutral.get('final_signal') == 'HOLD'


    # --- Test Case 4: Missing stock_symbol ---
    logging.info("\n--- CEOAgent Test Case 4: Missing stock_symbol ---")
    final_decision_no_symbol = ceo.run(data={}) # Missing stock_symbol
    logging.info(f"CEO Final Decision (No Symbol): {final_decision_no_symbol.get('final_signal')}")
    assert final_decision_no_symbol.get('final_signal') == 'ERROR'
    
    # --- Test Case 5: CEO make_recommendation after a run ---
    logging.info("\n--- CEOAgent Test Case 5: make_recommendation after run ---")
    ceo_recommendation = ceo.make_recommendation() # Uses the last decision (NEUTRAL_STOCK)
    logging.info(f"CEO Recommendation (post-run): {ceo_recommendation.get('final_signal')}, Conf: {ceo_recommendation.get('confidence')}")
    assert ceo_recommendation.get('stock_symbol') == "NEUTRAL_STOCK"

    # --- Test Case 6: CEO make_recommendation before any run (for a new CEO instance) ---
    logging.info("\n--- CEOAgent Test Case 6: make_recommendation for new CEO (no run) ---")
    new_ceo = CEOAgent(logger_instance=None) # New instance
    new_ceo_recommendation = new_ceo.make_recommendation()
    logging.info(f"New CEO Recommendation (no run): {new_ceo_recommendation.get('final_signal')}")
    assert new_ceo_recommendation.get('final_signal') == "HOLD_NO_DATA"

    logging.info("\n--- CEOAgent tests complete ---")
