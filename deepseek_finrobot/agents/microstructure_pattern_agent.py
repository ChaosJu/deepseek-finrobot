import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import datetime

# Assuming BaseAgent is in a reachable path, adjust if necessary
# from .base_agent import BaseAgent 
# For now, as BaseAgent is not in the same directory, we'll define a placeholder or inherit from object.
# If BaseAgent is needed, its path must be correct relative to this file, or PYTHONPATH configured.
# Let's assume a simple BaseAgent for now for logging, or just use a standard logger.

class BaseAgent: # Placeholder if actual BaseAgent is not accessible
    def __init__(self, agent_name: str = "BaseAgent", logger_instance: Optional[logging.Logger] = None):
        self.agent_name = agent_name
        if logger_instance:
            self.logger = logger_instance
        else:
            self.logger = logging.getLogger(self.agent_name)
            if not self.logger.handlers: # Setup basic logging if not configured
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)


class MicrostructurePatternAgent(BaseAgent):
    """
    (微观结构模式分析智能体)
    分析高频市场数据（如分钟K线、Tick数据）以识别特定的微观结构模式，
    例如恐慌性抛售、冲高回落等，并评估其潜在含义。
    Analyzes high-frequency market data (e.g., 1-minute K-lines, Tick data) 
    to identify specific microstructure patterns like panic selling, blow-off tops, etc.,
    and assesses their potential implications.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_config: Optional[Dict[str, Any]] = None, agent_name: str = "MicrostructurePatternAgent", logger_instance: Optional[logging.Logger] = None):
        """
        初始化微观结构模式分析智能体。
        Initializes the MicrostructurePatternAgent.

        Args:
            config (Optional[Dict[str, Any]]): 包含模式检测参数的配置字典。
                                               Configuration dictionary containing parameters for pattern detection.
                                               Example:
                                               {
                                                   "panic_selling_volume_spike": {
                                                       "lookback_minutes_short": 5,
                                                       "lookback_minutes_long": 30,
                                                       "volume_surge_factor": 2.5,
                                                       "price_drop_threshold_pct": -2.0
                                                   },
                                                   "euphoria_blow_off_top": {
                                                       "lookback_minutes_price_run": 60,
                                                       "min_price_rise_pct_for_run": 10.0,
                                                       "volume_surge_factor_climax": 3.0,
                                                       "post_climax_stall_minutes": 15,
                                                       "reversal_threshold_pct_from_peak": -1.5
                                                   }
                                               }
            llm_config (Optional[Dict[str, Any]]): LLM配置（未来可能用于更复杂的模式解释）。
                                                    LLM configuration (potentially for more complex pattern interpretation in the future).
            agent_name (str): 智能体的名称。Name of the agent.
            logger_instance (Optional[logging.Logger]): 日志记录器实例。Optional logger instance.
        """
        super().__init__(agent_name=agent_name, logger_instance=logger_instance)
        self.config = config if config else {}
        self.llm_config = llm_config # Currently unused, for future enhancements
        
        # Load pattern-specific configurations
        self.panic_selling_config = self.config.get("panic_selling_volume_spike", {})
        self.euphoria_blow_off_config = self.config.get("euphoria_blow_off_top", {})
        
        self.logger.info("MicrostructurePatternAgent initialized.")
        self.logger.debug(f"Panic Selling Config: {self.panic_selling_config}")
        self.logger.debug(f"Euphoria Blow-off Config: {self.euphoria_blow_off_config}")


    def _detect_panic_selling_volume_spike(self, klines_1min: List[Dict], config_panic: Dict) -> Optional[Dict[str, Any]]:
        """
        (私有方法) 检测恐慌性抛售中的成交量激增和价格下跌模式。
        (Private method) Detects panic selling pattern characterized by volume spike and price drop.

        Args:
            klines_1min (List[Dict]): 1分钟OHLCV数据列表，按时间升序排列。
                                      List of 1-minute OHLCV data, sorted chronologically.
                                      Expected keys per dict: 'time', 'open', 'high', 'low', 'close', 'volume'.
            config_panic (Dict): 此模式的特定配置参数。
                                 Specific configuration parameters for this pattern.

        Returns:
            Optional[Dict[str, Any]]: 如果检测到模式，则返回包含模式详情的字典，否则返回None。
                                      Returns a dictionary with pattern details if detected, otherwise None.
        """
        lookback_short = config_panic.get("lookback_minutes_short", 5)
        lookback_long = config_panic.get("lookback_minutes_long", 30)
        volume_surge_factor = config_panic.get("volume_surge_factor", 2.5)
        price_drop_threshold_pct = config_panic.get("price_drop_threshold_pct", -2.0) # e.g., -2.0 for a 2% drop

        if not klines_1min or len(klines_1min) < lookback_long:
            self.logger.debug(f"Panic Selling: Not enough data. Need {lookback_long}, got {len(klines_1min)}.")
            return None

        df = pd.DataFrame(klines_1min)
        if not all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close', 'volume']):
            self.logger.warning("Panic Selling: Missing required columns in kline data.")
            return None
        
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['time'] = pd.to_datetime(df['time']) # Ensure time is datetime
        df = df.sort_values(by='time', ascending=True).reset_index(drop=True) # Ensure sorted

        if len(df) < lookback_long: # Re-check after potential NaN drops or if initial list was too short
            self.logger.debug(f"Panic Selling: Not enough data after processing. Need {lookback_long}, got {len(df)}.")
            return None

        # Volume analysis
        avg_volume_long = df['volume'].rolling(window=lookback_long, min_periods=min(lookback_long, len(df))).mean().iloc[-1]
        avg_volume_short = df['volume'].rolling(window=lookback_short, min_periods=min(lookback_short, len(df))).mean().iloc[-1]
        
        volume_condition_met = False
        if avg_volume_long > 0 and avg_volume_short > (volume_surge_factor * avg_volume_long):
            volume_condition_met = True
            self.logger.debug(f"Panic Selling: Volume surge detected. Short avg: {avg_volume_short:.2f}, Long avg: {avg_volume_long:.2f}, Factor: {avg_volume_short/avg_volume_long if avg_volume_long > 0 else 'inf'}")

        # Price drop analysis (over the short lookback period)
        price_condition_met = False
        if len(df) >= lookback_short:
            current_close = df['close'].iloc[-1]
            lookback_start_price = df['open'].iloc[-lookback_short] # Price at the start of the short lookback window
            
            if lookback_start_price > 0:
                price_change_pct = ((current_close - lookback_start_price) / lookback_start_price) * 100
                if price_change_pct <= price_drop_threshold_pct: # Price drop is negative
                    price_condition_met = True
                    self.logger.debug(f"Panic Selling: Price drop detected. Start: {lookback_start_price:.2f}, Current: {current_close:.2f}, Change: {price_change_pct:.2f}%")
        
        if volume_condition_met and price_condition_met:
            latest_kline_time = df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(f"Panic Selling Volume Spike DETECTED at {latest_kline_time}")
            return {
                "pattern_name": "PanicSelling_VolumeSpike",
                "timestamp_detected": latest_kline_time,
                "confidence": 0.7, # Base confidence, can be adjusted
                "key_metrics": {
                    "time_window_minutes_short": lookback_short,
                    "time_window_minutes_long": lookback_long,
                    "volume_spike_factor_observed": round(avg_volume_short / avg_volume_long if avg_volume_long > 0 else float('inf'), 2),
                    "price_drop_percentage_observed": round(price_change_pct, 2)
                },
                "implication": "Potential capitulation, short-term oversold. May indicate a reversal opportunity if other contrarian signals align."
            }
        return None

    def _detect_euphoria_blow_off_top(self, klines_1min: List[Dict], config_euphoria: Dict) -> Optional[Dict[str, Any]]:
        """
        (私有方法) 检测市场狂热情绪下的冲高回落模式。
        (Private method) Detects euphoria / blow-off top pattern.

        Args:
            klines_1min (List[Dict]): 1分钟OHLCV数据列表，按时间升序排列。
            config_euphoria (Dict): 此模式的特定配置参数。

        Returns:
            Optional[Dict[str, Any]]: 如果检测到模式，则返回包含模式详情的字典，否则返回None。
        """
        lookback_run = config_euphoria.get("lookback_minutes_price_run", 60)
        min_rise_pct = config_euphoria.get("min_price_rise_pct_for_run", 10.0)
        volume_surge_climax = config_euphoria.get("volume_surge_factor_climax", 3.0)
        stall_minutes = config_euphoria.get("post_climax_stall_minutes", 15)
        reversal_pct_from_peak = config_euphoria.get("reversal_threshold_pct_from_peak", -1.5) # e.g. -1.5% drop from peak

        if not klines_1min or len(klines_1min) < lookback_run + stall_minutes:
            self.logger.debug(f"Blow-off Top: Not enough data. Need {lookback_run + stall_minutes}, got {len(klines_1min)}.")
            return None

        df = pd.DataFrame(klines_1min)
        if not all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close', 'volume']):
            self.logger.warning("Blow-off Top: Missing required columns in kline data.")
            return None
        
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time', ascending=True).reset_index(drop=True)
        
        if len(df) < lookback_run + stall_minutes: 
            self.logger.debug(f"Blow-off Top: Not enough data after processing. Need {lookback_run + stall_minutes}, got {len(df)}.")
            return None

        # 1. Identify recent sharp price run
        price_run_df = df.iloc[- (lookback_run + stall_minutes) : -stall_minutes] # Data for the price run period
        if len(price_run_df) < lookback_run: self.logger.debug("Blow-off Top: Not enough data for price run period."); return None
        
        price_start_of_run = price_run_df['open'].iloc[0]
        peak_price_in_run = price_run_df['high'].max()
        price_rise_pct = ((peak_price_in_run - price_start_of_run) / price_start_of_run) * 100

        if price_rise_pct < min_rise_pct:
            self.logger.debug(f"Blow-off Top: Price rise ({price_rise_pct:.2f}%) less than threshold ({min_rise_pct:.2f}%)."); return None
        self.logger.debug(f"Blow-off Top: Price run identified. Rise: {price_rise_pct:.2f}%")

        # 2. Look for climactic volume during the run, especially near the peak
        peak_time_in_run = price_run_df['high'].idxmax() # Index of the peak high
        # Consider volume around the peak time, e.g., a few bars before and at the peak
        climax_volume_window_df = price_run_df.loc[max(0, peak_time_in_run - 5) : peak_time_in_run] # Example: 5 min window up to peak
        if climax_volume_window_df.empty: self.logger.debug("Blow-off Top: Empty climax volume window."); return None

        avg_volume_run_period = price_run_df['volume'].rolling(window=min(len(price_run_df),30)).mean() # 30-min avg volume in run period
        if avg_volume_run_period.empty or pd.isna(avg_volume_run_period.iloc[-1]) or avg_volume_run_period.iloc[-1] == 0:
            self.logger.debug("Blow-off Top: Could not calculate valid average volume for run period."); return None
            
        max_volume_at_climax = climax_volume_window_df['volume'].max()
        
        if max_volume_at_climax < (volume_surge_climax * avg_volume_run_period.iloc[-1]): # Compare with avg volume just before climax
            self.logger.debug(f"Blow-off Top: Climax volume ({max_volume_at_climax:.0f}) not surged enough over avg ({avg_volume_run_period.iloc[-1]:.0f})."); return None
        self.logger.debug(f"Blow-off Top: Climax volume detected: {max_volume_at_climax:.0f}")

        # 3. Check for subsequent stall or reversal
        stall_df = df.iloc[-stall_minutes:] # Data for the post-climax stall period
        if len(stall_df) < stall_minutes: self.logger.debug("Blow-off Top: Not enough data for stall period."); return None
        
        current_close_after_stall = stall_df['close'].iloc[-1]
        price_change_from_peak = ((current_close_after_stall - peak_price_in_run) / peak_price_in_run) * 100

        if price_change_from_peak > reversal_pct_from_peak: # Price hasn't dropped enough from peak
            self.logger.debug(f"Blow-off Top: Price change from peak ({price_change_from_peak:.2f}%) not sufficient for reversal ({reversal_pct_from_peak:.2f}%)."); return None
        self.logger.info(f"Blow-off Top: Reversal/stall condition met. Change from peak: {price_change_from_peak:.2f}%")
            
        latest_kline_time = df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info(f"Euphoria Blow-Off Top DETECTED at {latest_kline_time}")
        return {
            "pattern_name": "Euphoria_BlowOffTop",
            "timestamp_detected": latest_kline_time,
            "confidence": 0.75, # Base confidence
            "key_metrics": {
                "price_run_up_percentage": round(price_rise_pct, 2),
                "climax_volume_observed": round(max_volume_at_climax,0),
                "reversal_from_peak_percentage": round(price_change_from_peak, 2)
            },
            "implication": "Potential exhaustion of buying pressure, increased risk of sharp correction. Contrarian SELL opportunity may arise."
        }
        return None


    def analyze_microstructure(self, asset_id: str, intraday_data: Dict[str, List[Dict[str, Any]]], latest_news_sentiment: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析微观结构数据，检测预定义的模式。
        Analyzes microstructure data to detect predefined patterns.

        Args:
            asset_id (str): 资产ID。Asset ID.
            intraday_data (Dict[str, List[Dict[str, Any]]]): 包含日内数据的字典。
                                                             Dictionary containing intraday data.
                                                             Expected keys: '1min_klines', 'tick_data'.
            latest_news_sentiment (Optional[Dict]): 最新的新闻情绪分析（可选，用于上下文）。
                                                    Optional latest news sentiment for context.

        Returns:
            Dict[str, Any]: 包含检测到的模式和总体评估的分析结果。
                            Analysis result containing detected patterns and an overall assessment.
                            Example:
                            {
                                "asset_id": "000001.SZ",
                                "analysis_timestamp": "YYYY-MM-DD HH:MM:SS",
                                "detected_patterns": [
                                    {"pattern_name": "PanicSelling_VolumeSpike", ...},
                                    {"pattern_name": "Euphoria_BlowOffTop", ...}
                                ],
                                "overall_microstructure_assessment": "HighVolatility_PotentialReversal",
                                "raw_data_summary": {"1min_klines_count": 120, "tick_data_count": 0}
                            }
        """
        self.logger.info(f"Starting microstructure analysis for {asset_id}")
        detected_patterns_list = []
        
        klines_1min = intraday_data.get('1min_klines', [])
        # tick_data = intraday_data.get('tick_data', []) # For future use

        if klines_1min:
            # Detect Panic Selling
            if self.panic_selling_config.get("enabled", True): # Check if pattern detection is enabled
                panic_pattern = self._detect_panic_selling_volume_spike(klines_1min, self.panic_selling_config)
                if panic_pattern:
                    detected_patterns_list.append(panic_pattern)
            
            # Detect Euphoria Blow-Off Top
            if self.euphoria_blow_off_config.get("enabled", True):
                euphoria_pattern = self._detect_euphoria_blow_off_top(klines_1min, self.euphoria_blow_off_config)
                if euphoria_pattern:
                    detected_patterns_list.append(euphoria_pattern)
        else:
            self.logger.info(f"No 1-minute klines provided for {asset_id}, skipping kline-based pattern detection.")

        # (Placeholder for other pattern detection using tick_data or L2 data)

        # Determine overall assessment (simple heuristic for now)
        overall_assessment = "Neutral"
        if any(p["pattern_name"] == "PanicSelling_VolumeSpike" for p in detected_patterns_list):
            overall_assessment = "PanicSellingDetected_PotentialReversal"
        elif any(p["pattern_name"] == "Euphoria_BlowOffTop" for p in detected_patterns_list):
            overall_assessment = "EuphoriaBlowOff_PotentialReversal"
        elif detected_patterns_list: # If any other pattern was detected
            overall_assessment = "PatternsDetected_MonitorClosely"
        
        self.logger.info(f"Microstructure analysis for {asset_id} complete. Detected patterns: {len(detected_patterns_list)}. Assessment: {overall_assessment}")

        return {
            "asset_id": asset_id,
            "analysis_timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "detected_patterns": detected_patterns_list,
            "overall_microstructure_assessment": overall_assessment,
            "raw_data_summary": {
                "1min_klines_count": len(klines_1min),
                # "tick_data_count": len(tick_data) # For future
            },
            "latest_news_sentiment_context": latest_news_sentiment # Pass through for context
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    import json

    # --- Test Configuration ---
    test_config = {
        "panic_selling_volume_spike": {
            "enabled": True,
            "lookback_minutes_short": 5,
            "lookback_minutes_long": 20, # Shorter for test data
            "volume_surge_factor": 2.0,
            "price_drop_threshold_pct": -1.5 
        },
        "euphoria_blow_off_top": {
            "enabled": True,
            "lookback_minutes_price_run": 30, # Shorter for test data
            "min_price_rise_pct_for_run": 5.0,
            "volume_surge_factor_climax": 2.0,
            "post_climax_stall_minutes": 5,
            "reversal_threshold_pct_from_peak": -1.0
        }
    }
    agent = MicrostructurePatternAgent(config=test_config)

    # --- Test Case 1: Panic Selling ---
    logger.info("\n--- Test Case 1: Panic Selling ---")
    panic_klines = []
    base_time = datetime.datetime(2024, 1, 1, 9, 30, 0)
    # Long lookback period with normal volume/price
    for i in range(test_config["panic_selling_volume_spike"]["lookback_minutes_long"] - test_config["panic_selling_volume_spike"]["lookback_minutes_short"]):
        panic_klines.append({
            "time": (base_time + datetime.timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'),
            "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 100
        })
    # Short lookback with volume surge and price drop
    price = 10.0
    for i in range(test_config["panic_selling_volume_spike"]["lookback_minutes_short"]):
        ts = base_time + datetime.timedelta(minutes=test_config["panic_selling_volume_spike"]["lookback_minutes_long"] - test_config["panic_selling_volume_spike"]["lookback_minutes_short"] + i)
        price -= 0.1 # Gradual drop
        panic_klines.append({
            "time": ts.strftime('%Y-%m-%d %H:%M:%S'),
            "open": price + 0.05, "high": price + 0.1, "low": price - 0.05, "close": price, 
            "volume": 300 # Volume surge
        })
    
    # Ensure the last kline reflects the full drop for the check
    start_price_short_window = panic_klines[-test_config["panic_selling_volume_spike"]["lookback_minutes_short"]]["open"]
    final_price_short_window = panic_klines[-1]["close"]
    expected_drop_pct = ((final_price_short_window - start_price_short_window) / start_price_short_window) * 100
    logger.debug(f"Panic Test: Short window start price {start_price_short_window}, end price {final_price_short_window}, drop {expected_drop_pct:.2f}%")


    analysis_panic = agent.analyze_microstructure("TEST001", intraday_data={'1min_klines': panic_klines})
    print(json.dumps(analysis_panic, indent=2, ensure_ascii=False))
    assert any(p["pattern_name"] == "PanicSelling_VolumeSpike" for p in analysis_panic["detected_patterns"])

    # --- Test Case 2: Euphoria Blow-Off Top ---
    logger.info("\n--- Test Case 2: Euphoria Blow-Off Top ---")
    euphoria_klines = []
    base_time_eu = datetime.datetime(2024, 1, 1, 10, 0, 0)
    price = 10.0
    # Price run-up period
    for i in range(test_config["euphoria_blow_off_top"]["lookback_minutes_price_run"]):
        price += 0.2 # Steady rise
        euphoria_klines.append({
            "time": (base_time_eu + datetime.timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'),
            "open": price - 0.1, "high": price + 0.1, "low": price - 0.2, "close": price, "volume": 150
        })
    # Climax volume at peak
    peak_price = price + 0.5
    euphoria_klines.append({
        "time": (base_time_eu + datetime.timedelta(minutes=test_config["euphoria_blow_off_top"]["lookback_minutes_price_run"])).strftime('%Y-%m-%d %H:%M:%S'),
        "open": price, "high": peak_price, "low": price - 0.1, "close": peak_price - 0.1, "volume": 500 # Volume surge
    })
    # Stall/Reversal period
    current_price_stall = peak_price - 0.1
    for i in range(test_config["euphoria_blow_off_top"]["post_climax_stall_minutes"]):
        current_price_stall -= 0.05 # Slight drop
        euphoria_klines.append({
            "time": (base_time_eu + datetime.timedelta(minutes=test_config["euphoria_blow_off_top"]["lookback_minutes_price_run"] + 1 + i)).strftime('%Y-%m-%d %H:%M:%S'),
            "open": current_price_stall + 0.02, "high": current_price_stall + 0.05, "low": current_price_stall - 0.02, "close": current_price_stall, "volume": 100
        })
    
    analysis_euphoria = agent.analyze_microstructure("TEST002", intraday_data={'1min_klines': euphoria_klines})
    print(json.dumps(analysis_euphoria, indent=2, ensure_ascii=False))
    assert any(p["pattern_name"] == "Euphoria_BlowOffTop" for p in analysis_euphoria["detected_patterns"])

    logger.info("\n--- MicrostructurePatternAgent tests complete ---")
```
