import logging
from typing import List, Dict, Any, Optional
import statistics # For mean and stdev if needed

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define detailed sentiment categories that might influence signal strength
DETAILED_SENTIMENT_CATEGORIES_CONFIG = {
    "IrrationalPanic": {"bonus": 0.2, "description": "市场非理性恐慌"},
    "RumorDrivenDrop": {"bonus": 0.15, "description": "谣言驱动下跌"},
    "OverreactionToMinorNegative": {"bonus": 0.1, "description": "对次要负面新闻反应过度"},
    "TemporarySetback": {"bonus": 0.05, "description": "暂时性挫折"},
    "SustainedConcern": {"bonus": 0.0, "description": "持续担忧"}, 
    "FundamentalScandalOrCrisis": {"penalty": 1.0, "veto_buy": True, "description": "基本面丑闻或危机"},
    "OverhypedRally": {"bonus": 0.2, "description": "过度炒作上涨"},
    "RumorDrivenSpike": {"bonus": 0.15, "description": "谣言驱动上涨"},
    "OverreactionToMinorPositive": {"bonus": 0.1, "description": "对次要正面新闻反应过度"},
    "SpeculativeFrenzy": {"bonus": 0.1, "description": "投机狂热"},
    "SolidGrowthConfirmation": {"penalty": 0.1, "description": "稳健增长确认"},
    "NeutralObservation": {"bonus": 0.0, "description": "中性观察"},
    "FactualReport": {"bonus": 0.0, "description": "事实报道"},
    "Misinformation": {"penalty": 0.05, "description": "虚假信息(谨慎处理)"},
    "UnavailableDueToNoLLM": {"bonus": 0.0, "description": "LLM未配置，情绪细节不可用"},
    "ErrorInAnalysis": {"bonus": 0.0, "description": "情绪分析出错"},
    "ParsingError": {"bonus": 0.0, "description": "情绪分析解析错误"},
    "Unknown": {"bonus": 0.0, "description": "未知详细情绪分类"}
}


class ContrarianAnalyzer:
    """
    (逆向分析器)
    分析市场数据和结构化的新闻情绪，结合市场状态（波动性和趋势）动态调整阈值，以识别逆向交易机会。
    根据详细的情绪分类调整信号强度或否决信号。

    Analyzes market data and structured news sentiment, dynamically adjusting thresholds based on market state 
    (volatility and trend), to identify contrarian trading opportunities.
    Adjusts signal strength or vetoes signals based on detailed sentiment categories.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 ContrarianAnalyzer。
        加载基础阈值、详细情绪类别相关的评分调整配置，以及市场状态对阈值的调整因子。

        Initializes the ContrarianAnalyzer.
        Loads base thresholds, configuration for score adjustments related to detailed sentiment categories,
        and adjustment factors for market state.

        Args:
            config (Optional[Dict[str, Any]]): 配置字典。
                                               Configuration dictionary.
        """
        self.config = config if config else {}
        self.sentiment_category_adjustments = self.config.get(
            "sentiment_category_adjustments", 
            DETAILED_SENTIMENT_CATEGORIES_CONFIG 
        )
        
        # Base Thresholds
        self.base_threshold_sentiment_buy = self.config.get("BASE_THRESHOLD_SENTIMENT_CONTRARIAN_BUY", -0.3)
        self.base_threshold_sentiment_sell = self.config.get("BASE_THRESHOLD_SENTIMENT_CONTRARIAN_SELL", 0.3)
        # Base RSI thresholds (if used directly here, or passed to a PV analyzer that uses them)
        self.base_rsi_oversold = self.config.get("BASE_RSI_OVERSOLD", 30)
        self.base_rsi_overbought = self.config.get("BASE_RSI_OVERBOUGHT", 70)
        self.base_pv_price_min_change_pct = self.config.get("BASE_PV_PRICE_MIN_CHANGE_PCT", -1.0)
        self.base_pv_price_max_change_pct = self.config.get("BASE_PV_PRICE_MAX_CHANGE_PCT", 5.0)


        # Adjustment Factors from config for Market State
        # Format: self.adj_<param>_<trend/volatility>_<state> = self.config.get("ADJ_...", default_value)
        # Sentiment adjustments
        self.adj_sentiment_buy_trend_bearish = self.config.get("ADJ_SENTIMENT_BUY_TREND_BEARISH", -0.1) # More negative needed in bearish market
        self.adj_sentiment_buy_trend_bullish = self.config.get("ADJ_SENTIMENT_BUY_TREND_BULLISH", 0.05) # Less negative needed in bullish
        self.adj_sentiment_buy_vol_high = self.config.get("ADJ_SENTIMENT_BUY_VOL_HIGH", -0.1) # More negative needed in high vol
        self.adj_sentiment_buy_vol_low = self.config.get("ADJ_SENTIMENT_BUY_VOL_LOW", 0.05)   # Less negative needed in low vol
        
        # RSI adjustments (example, assuming RSI is part of PV analysis or checked here)
        self.adj_rsi_oversold_trend_bearish = self.config.get("ADJ_RSI_OVERSOLD_TREND_BEARISH", -5) # Lower RSI needed (e.g. 25)
        self.adj_rsi_oversold_vol_high = self.config.get("ADJ_RSI_OVERSOLD_VOL_HIGH", -5)       # Lower RSI needed

        # Price stability adjustments (for pv_price_min_change_pct)
        # Example: In high volatility, allow wider price swings for "stable"
        self.adj_pv_price_min_trend_bearish = self.config.get("ADJ_PV_PRICE_MIN_TREND_BEARISH", -0.5) # Allow slightly more dip in bearish
        self.adj_pv_price_min_vol_high = self.config.get("ADJ_PV_PRICE_MIN_VOL_HIGH", -1.0) # Allow more dip in high vol

        logger.info("ContrarianAnalyzer initialized with base thresholds and market state adjustment factors.")

    def _get_adjusted_thresholds(self, market_state_data: Dict[str, Any]) -> Dict[str, float]:
        """
        (私有方法) 根据当前市场状态动态调整交易信号生成的阈值。
        (Private method) Dynamically adjusts thresholds for trading signal generation based on the current market state.

        Args:
            market_state_data (Dict[str, Any]): 来自MarketDataAgent的市场状态数据。
                                                 Market state data from MarketDataAgent.
                                                 Expected keys: "market_volatility_level", "market_trend_state".
        Returns:
            Dict[str, float]: 包含所有调整后阈值的字典。
                              A dictionary containing all adjusted thresholds.
        """
        adj_thresholds = {
            "sentiment_buy": self.base_threshold_sentiment_buy,
            "sentiment_sell": self.base_threshold_sentiment_sell,
            "rsi_oversold": self.base_rsi_oversold,
            "rsi_overbought": self.base_rsi_overbought,
            "pv_price_min_change_pct": self.base_pv_price_min_change_pct,
            "pv_price_max_change_pct": self.base_pv_price_max_change_pct,
        }

        trend = market_state_data.get("market_trend_state", "Neutral/Ranging")
        volatility = market_state_data.get("market_volatility_level", "Medium")

        # Adjust sentiment_buy threshold
        if trend == "Bearish":
            adj_thresholds["sentiment_buy"] += self.adj_sentiment_buy_trend_bearish
        elif trend == "Bullish":
            adj_thresholds["sentiment_buy"] += self.adj_sentiment_buy_trend_bullish
        
        if volatility == "High":
            adj_thresholds["sentiment_buy"] += self.adj_sentiment_buy_vol_high
        elif volatility == "Low":
            adj_thresholds["sentiment_buy"] += self.adj_sentiment_buy_vol_low

        # Adjust RSI oversold threshold (example)
        if trend == "Bearish":
            adj_thresholds["rsi_oversold"] += self.adj_rsi_oversold_trend_bearish
        if volatility == "High":
            adj_thresholds["rsi_oversold"] += self.adj_rsi_oversold_vol_high
            
        # Adjust PV price min change percentage (example)
        if trend == "Bearish":
            adj_thresholds["pv_price_min_change_pct"] += self.adj_pv_price_min_trend_bearish
        if volatility == "High":
            adj_thresholds["pv_price_min_change_pct"] += self.adj_pv_price_min_vol_high

        logger.info(f"Adjusted thresholds based on market state (Trend: {trend}, Vol: {volatility}): {adj_thresholds}")
        return adj_thresholds

    def process_enhanced_news_sentiment(self, enhanced_sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        (中文文档已在上一版本中提供)
        Processes the enhanced news sentiment data from NewsAgent.
        Extracts key fields like overall_sentiment_score and aggregated_dominant_detailed_category.
        """
        if not enhanced_sentiment_data:
            logger.warning("No enhanced sentiment data provided.")
            return {'overall_sentiment_score': 0.0, 'dominant_detailed_category': 'Unknown', 'news_items_count': 0}

        overall_score = enhanced_sentiment_data.get('overall_sentiment_score', 0.0)
        dominant_category = enhanced_sentiment_data.get('aggregated_dominant_detailed_category', 'Unknown')
        news_count = len(enhanced_sentiment_data.get("news_items_analyzed", []))

        logger.info(f"Processed enhanced sentiment for asset {enhanced_sentiment_data.get('asset_id', 'N/A')}: "
                    f"Score={overall_score}, Category='{dominant_category}', Items={news_count}")
        
        return {
            'overall_sentiment_score': overall_score,
            'dominant_detailed_category': dominant_category,
            'news_items_count': news_count
        }

    def analyze_price_volume_anomaly(self, market_data_history: List[Dict[str, Any]], 
                                     adjusted_thresholds: Dict[str, float], # Now takes adjusted thresholds
                                     short_term_days: int = 5, 
                                     long_term_days: int = 20,
                                     volume_factor_threshold: float = 1.5
                                     ) -> Dict[str, Any]:
        """
        分析历史市场数据以寻找价格和成交量异常。
        使用动态调整后的价格变动百分比阈值。

        Analyzes historical market data for price and volume anomalies.
        Uses dynamically adjusted price change percentage thresholds.
        """
        price_min_change_pct = adjusted_thresholds.get("pv_price_min_change_pct", self.base_pv_price_min_change_pct)
        price_max_change_pct = adjusted_thresholds.get("pv_price_max_change_pct", self.base_pv_price_max_change_pct)

        if not market_data_history or len(market_data_history) < long_term_days:
            logger.warning(f"Insufficient market data history for PV analysis (need at least {long_term_days} days, got {len(market_data_history)}).")
            return {'is_high_volume_accumulation': False, 'is_price_stable_or_rising': False, 'reason': "Insufficient data"}

        is_high_volume_accumulation = False
        try:
            recent_volumes = [float(d['volume']) for d in market_data_history[:short_term_days] if 'volume' in d and d['volume'] is not None]
            long_term_volumes = [float(d['volume']) for d in market_data_history[:long_term_days] if 'volume' in d and d['volume'] is not None]

            if len(recent_volumes) < short_term_days or len(long_term_volumes) < long_term_days:
                logger.warning(f"Not enough valid volume data points for PV. Recent: {len(recent_volumes)}/{short_term_days}, Long-term: {len(long_term_volumes)}/{long_term_days}")
            else:
                avg_recent_volume = statistics.mean(recent_volumes)
                avg_long_term_volume = statistics.mean(long_term_volumes)
                if avg_long_term_volume > 0:
                    if avg_recent_volume > volume_factor_threshold * avg_long_term_volume:
                        is_high_volume_accumulation = True
                    logger.info(f"Volume analysis: Avg recent vol: {avg_recent_volume:.0f}, Avg long-term vol: {avg_long_term_volume:.0f}. High accumulation: {is_high_volume_accumulation}")
                else: logger.warning("Avg long-term volume is zero for PV.")
        except (ValueError, TypeError, statistics.StatisticsError) as e: logger.error(f"Error in PV volume analysis: {e}", exc_info=True)

        is_price_stable_or_rising = False
        try:
            if len(market_data_history) >= short_term_days:
                current_price = float(market_data_history[0]['close'])
                past_price = float(market_data_history[short_term_days - 1]['close'])
                if past_price > 0:
                    price_change_pct = ((current_price - past_price) / past_price) * 100
                    if price_min_change_pct <= price_change_pct <= price_max_change_pct:
                        is_price_stable_or_rising = True
                    logger.info(f"Price analysis: Current: {current_price:.2f}, Past({short_term_days}d): {past_price:.2f}. Change: {price_change_pct:.2f}%. Stable/Rising (Thresh: {price_min_change_pct:.2f}% to {price_max_change_pct:.2f}%): {is_price_stable_or_rising}")
                else: logger.warning("Past price is zero for PV price change calc.")
            else: logger.warning(f"Not enough data for PV price change (need {short_term_days}, got {len(market_data_history)}).")
        except (ValueError, TypeError, IndexError) as e: logger.error(f"Error in PV price analysis: {e}", exc_info=True)

        return {
            'is_high_volume_accumulation': is_high_volume_accumulation, 
            'is_price_stable_or_rising': is_price_stable_or_rising,
            'reason': "PV analysis complete"
        }

    def generate_overall_signal(self, 
                                stock_symbol: str, 
                                processed_sentiment_data: Dict[str, Any], 
                                price_volume_analysis_input_data: List[Dict[str, Any]], # Expects raw market data for PV
                                market_state_data: Dict[str, Any]) -> Dict[str, Any]: # New parameter
        """
        根据各项分析（包括增强后的新闻情绪和市场状态）生成总体交易信号。
        核心逻辑是“负面新闻（特定类型），正面量价”的逆向买入策略，
        以及“正面新闻（特定类型），警惕量价”的逆向卖出（或避免买入）策略。
        详细情绪类别用于调整信号强度/置信度或否决信号。市场状态用于动态调整阈值。

        Generates an overall trading signal based on various analyses, including enhanced news sentiment and market state.
        Core logic is "Negative News (specific types), Positive Volume/Price" for contrarian BUY,
        and "Positive News (specific types), Watchful Volume/Price" for contrarian SELL (or avoid BUY).
        Detailed sentiment categories adjust signal strength/confidence or veto signals. Market state dynamically adjusts thresholds.

        Args:
            stock_symbol (str): 股票代码。Stock symbol.
            processed_sentiment_data (Dict[str, Any]): 来自 `process_enhanced_news_sentiment` 的处理后情绪数据。
                                                      Processed sentiment data from `process_enhanced_news_sentiment`.
            price_volume_analysis_input_data (List[Dict[str, Any]]): 用于价格成交量分析的原始市场数据列表。
                                                                    Raw market data list for price-volume analysis.
            market_state_data (Dict[str, Any]): 当前市场状态数据 (波动性, 趋势)。
                                                Current market state data (volatility, trend).
        
        Returns:
            Dict[str, Any]: 包含交易信号、置信度、理由、股票代码和支持证据的字典。
                            A dictionary containing the trading signal, confidence, reason, stock symbol, and supporting evidence.
        """
        logger.info(f"Generating signal for {stock_symbol} with market state: {market_state_data}")
        
        adjusted_thresholds = self._get_adjusted_thresholds(market_state_data)
        logger.info(f"Using adjusted thresholds: {adjusted_thresholds}")

        # Perform PV analysis using adjusted thresholds
        # Assuming default periods for short/long term days and volume factor from original method if not in config
        pv_short_term_days = self.config.get("PV_SHORT_TERM_DAYS", 5)
        pv_long_term_days = self.config.get("PV_LONG_TERM_DAYS", 20)
        pv_volume_factor = self.config.get("PV_VOLUME_FACTOR_THRESHOLD", 1.5)
        
        price_volume_analysis = self.analyze_price_volume_anomaly(
            price_volume_analysis_input_data,
            adjusted_thresholds=adjusted_thresholds, # Pass the whole dict
            short_term_days=pv_short_term_days,
            long_term_days=pv_long_term_days,
            volume_factor_threshold=pv_volume_factor
        )

        logger.debug(f"Processed Sentiment Data: {processed_sentiment_data}")
        logger.debug(f"Price/Volume Analysis (with adjusted thresholds): {price_volume_analysis}")

        overall_sentiment_score = processed_sentiment_data.get('overall_sentiment_score', 0.0)
        detailed_category = processed_sentiment_data.get('dominant_detailed_category', 'Unknown')
        
        is_high_volume = price_volume_analysis.get('is_high_volume_accumulation', False)
        is_price_ok_for_buy = price_volume_analysis.get('is_price_stable_or_rising', False)

        signal = 'HOLD'
        base_confidence = 0.5 
        sentiment_adjustment = 0.0
        reason_parts = []

        category_config = self.sentiment_category_adjustments.get(detailed_category, {"bonus": 0.0, "penalty": 0.0, "veto_buy": False, "description": "未配置的类别"})
        
        # Use adjusted sentiment threshold for BUY
        current_sentiment_buy_threshold = adjusted_thresholds.get("sentiment_buy", self.base_threshold_sentiment_buy)

        # --- Contrarian BUY Logic ---
        if overall_sentiment_score < current_sentiment_buy_threshold:
            reason_parts.append(f"Overall sentiment score ({overall_sentiment_score:.2f}) is below adjusted BUY threshold ({current_sentiment_buy_threshold:.2f}).")
            if is_high_volume and is_price_ok_for_buy:
                reason_parts.append("Price/volume conditions favorable for contrarian BUY (high volume, stable/rising price based on adjusted thresholds).")
                signal = 'BUY'
                base_confidence = 0.6 
                if category_config.get("veto_buy", False):
                    signal = 'NO_TRADE_CRISIS'
                    base_confidence = 0.0 
                    reason_parts.append(f"VETO BUY due to critical sentiment: {detailed_category} ({category_config.get('description', '')}).")
                else:
                    sentiment_adjustment = category_config.get("bonus", 0.0) - category_config.get("penalty", 0.0)
                    reason_parts.append(f"Detailed sentiment '{detailed_category} ({category_config.get('description', '')})' adjusts confidence by {sentiment_adjustment:.2f}.")
            else:
                pv_reasons = []
                if not is_high_volume: pv_reasons.append("volume not high enough")
                if not is_price_ok_for_buy: pv_reasons.append("price not stable/rising per adjusted criteria")
                reason_parts.append(f"Price/volume conditions not met for contrarian BUY: {'; '.join(pv_reasons)}.")
        
        # --- Contrarian SELL Logic (Placeholder) ---
        # current_sentiment_sell_threshold = adjusted_thresholds.get("sentiment_sell", self.base_threshold_sentiment_sell)
        # if overall_sentiment_score > current_sentiment_sell_threshold:
        #     ... (similar logic for SELL)

        final_confidence = base_confidence + sentiment_adjustment
        final_confidence = round(max(0.0, min(1.0, final_confidence)), 2)

        if signal == 'HOLD' and not reason_parts:
            reason_parts.append("No strong contrarian signals detected based on current sentiment, P/V, and adjusted thresholds.")
        
        final_reason = f"{signal} signal for {stock_symbol} (Confidence: {final_confidence:.2f}): " + " ".join(reason_parts)
        logger.info(final_reason)

        return {
            'signal': signal, 
            'confidence': final_confidence,
            'reason': final_reason,
            'stock_symbol': stock_symbol,
            'detailed_sentiment_category': detailed_category,
            'adjusted_thresholds_used': adjusted_thresholds, # Log used thresholds
            'supporting_evidence': {
                'processed_sentiment': processed_sentiment_data,
                'price_volume': price_volume_analysis,
                'market_state': market_state_data
            }
        }

if __name__ == '__main__':
    print("--- Testing ContrarianAnalyzer with Market State and Adjusted Thresholds ---")
    
    test_config = {
        "BASE_THRESHOLD_SENTIMENT_CONTRARIAN_BUY": -0.3,
        "BASE_RSI_OVERSOLD": 30, # Example if RSI was used here
        "BASE_PV_PRICE_MIN_CHANGE_PCT": -2.0, # Base price stability min
        "BASE_PV_PRICE_MAX_CHANGE_PCT": 7.0,  # Base price stability max

        "ADJ_SENTIMENT_BUY_TREND_BEARISH": -0.1, 
        "ADJ_SENTIMENT_BUY_VOL_HIGH": -0.05,
        "ADJ_PV_PRICE_MIN_VOL_HIGH": -1.5, # Allow price to drop more in high vol for "stable"
        
        "sentiment_category_adjustments": DETAILED_SENTIMENT_CATEGORIES_CONFIG
    }
    analyzer = ContrarianAnalyzer(config=test_config)

    # Mock Market Data History for PV analysis
    mock_pv_input_data = [{'date': f'2023-10-{i+1:02d}', 'close': 100-i, 'volume': 1000000 + (i%3)*500000} for i in range(20)][::-1]


    # Test Case 1: Bullish Market, Low Volatility -> Easier BUY thresholds
    print("\n1. Bullish Market, Low Volatility")
    market_state_bull_low_vol = {"market_volatility_level": "Low", "market_trend_state": "Bullish"}
    sentiment_data_mild_neg = analyzer.process_enhanced_news_sentiment({
        "overall_sentiment_score": -0.25, # Slightly less negative than base threshold
        "aggregated_dominant_detailed_category": "TemporarySetback" 
    })
    pv_data = mock_pv_input_data[:5] # Ensure enough for short term
    pv_data[0]['close'] = 99 # current price
    pv_data[4]['close'] = 100 # past price for 5 days (-1% change, meets base_pv_price_min_change_pct)

    # Simulate PV analysis input data (needs to be a list of dicts for analyze_price_volume_anomaly)
    signal1 = analyzer.generate_overall_signal("STOCK_BLV", sentiment_data_mild_neg, pv_data, market_state_bull_low_vol)
    print(json.dumps(signal1, indent=2, ensure_ascii=False))
    # Expected: BUY signal due to threshold adjustments making -0.25 sufficiently negative.

    # Test Case 2: Bearish Market, High Volatility -> Harder BUY thresholds
    print("\n2. Bearish Market, High Volatility")
    market_state_bear_high_vol = {"market_volatility_level": "High", "market_trend_state": "Bearish"}
    sentiment_data_strong_neg = analyzer.process_enhanced_news_sentiment({
        "overall_sentiment_score": -0.4, # More negative than base
        "aggregated_dominant_detailed_category": "IrrationalPanic" 
    })
    pv_data_bear = mock_pv_input_data[:5]
    pv_data_bear[0]['close'] = 97 # current price
    pv_data_bear[4]['close'] = 100 # past price for 5 days (-3% change)
                                  # This should meet the more lenient pv_price_min_change_pct due to high vol
                                  # Base: -2.0, ADJ_PV_PRICE_MIN_VOL_HIGH: -1.5 -> adjusted = -3.5%

    signal2 = analyzer.generate_overall_signal("STOCK_BHV", sentiment_data_strong_neg, pv_data_bear, market_state_bear_high_vol)
    print(json.dumps(signal2, indent=2, ensure_ascii=False))
    # Expected: BUY signal if sentiment is negative enough despite harsher base, and P/V is ok with adjusted thresholds.

    # Test Case 3: Veto due to FundamentalScandalOrCrisis, regardless of market state
    print("\n3. VETO BUY (FundamentalScandalOrCrisis), Neutral Market")
    market_state_neutral = {"market_volatility_level": "Medium", "market_trend_state": "Neutral/Ranging"}
    sentiment_crisis = analyzer.process_enhanced_news_sentiment({
        "overall_sentiment_score": -0.8, 
        "aggregated_dominant_detailed_category": "FundamentalScandalOrCrisis"
    })
    signal3 = analyzer.generate_overall_signal("STOCK_CRISIS", sentiment_crisis, pv_data, market_state_neutral)
    print(json.dumps(signal3, indent=2, ensure_ascii=False))
    # Expected: NO_TRADE_CRISIS
    
    print("\n--- ContrarianAnalyzer tests with market state adjustments complete ---")
