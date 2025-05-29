import logging
from typing import Dict, Any, Optional, List 
import datetime 
import pandas as pd 
import numpy as np 
import akshare 

from .base_agent import BaseAgent


class MarketDataAgent(BaseAgent):
    """
    (市场数据智能体)
    负责获取、处理并提供市场数据，包括计算市场整体状态（波动性和趋势）、单个资产的ATR，
    以及获取更细粒度的市场数据如Tick数据、分钟K线和L2快照。
    Agent responsible for fetching, processing, and providing market data,
    including calculating overall market state (volatility and trend), ATR for individual assets,
    and fetching finer-grained market data like Tick data, 1-minute K-lines, and L2 snapshots.
    """

    def __init__(self, market_data_source: Any, agent_name: str = "MarketDataAgent", logger_instance: Optional[logging.Logger] = None):
        """
        初始化 MarketDataAgent。
        Initializes the MarketDataAgent.

        Args:
            market_data_source (MarketDataSource): 用于获取数据的 MarketDataSource 实例。
                                                   An instance of MarketDataSource to fetch data.
                                                   This is currently less used as Akshare is called directly for many functions.
            agent_name (str): 智能体的名称。Name of the agent.
            logger_instance (Optional[logging.Logger]): 可选的日志记录器实例。
                                                        Optional logger instance.
        """
        super().__init__(agent_name=agent_name, logger_instance=logger_instance)
        # The market_data_source might be a generic one or specific.
        # For Akshare specific calls, we might not always use the market_data_source methods directly.
        self.market_data_source = market_data_source 
        self.processed_data: Optional[Dict[str, Any]] = None
        self.logger.info(f"MarketDataAgent initialized with data source: {market_data_source.__class__.__name__ if market_data_source else 'None'}.")

    def _normalize_akshare_symbol(self, asset_id: str, for_tick_tx_js: bool = False, for_level2: bool = False) -> str:
        """
        (私有方法) 将 'XXXXXX.SZ' 或 'XXXXXX.SH' 格式的 asset_id 转换为 Akshare 特定函数所需的格式。
        (Private method) Converts asset_id from 'XXXXXX.SZ' or 'XXXXXX.SH' format 
        to the format required by specific Akshare functions.
        """
        parts = asset_id.split('.')
        if len(parts) != 2:
            self.logger.warning(f"Invalid asset_id format: {asset_id}. Expected XXXXXX.SZ or XXXXXX.SH. Returning as is.")
            return asset_id
        
        code, market = parts[0], parts[1].upper()
        if for_tick_tx_js or for_level2: # For stock_zh_a_tick_tx_js or stock_zh_a_level2_snapshot_em
            return market.lower() + code 
        return code # For stock_zh_a_hist_min_em which just needs the code part

    def get_tick_data(self, asset_id: str, trade_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取指定资产特定交易日的Tick数据。
        Fetches Tick data for a given asset on a specific trading day.

        Args:
            asset_id (str): 资产代码 (例如 "000001.SZ")。
                            Asset ID (e.g., "000001.SZ").
            trade_date (Optional[str]): 交易日期 ("YYYYMMDD" 格式)。如果为 None，则尝试获取最新交易日数据。
                                       Trading date in "YYYYMMDD" format. If None, attempts to fetch for the latest available trading day.
        Returns:
            List[Dict[str, Any]]: Tick数据列表，每个Tick包含'time', 'price', 'volume', 'type'等字段。
                                  A list of tick data, where each tick is a dictionary with fields like 
                                  'time', 'price', 'volume', 'type' (buy/sell/neutral if available).
        """
        self.logger.info(f"Fetching tick data for {asset_id} on trade date {trade_date if trade_date else 'latest'}.")
        ak_symbol = self._normalize_akshare_symbol(asset_id, for_tick_tx_js=True)
        
        if trade_date is None: # Try to get the latest trading day if not specified
            try:
                # Fetch a single day of daily data to find the latest trading day
                latest_daily_data = akshare.stock_zh_a_hist(symbol=ak_symbol.replace("sz","").replace("sh",""), period="daily", adjust="qfq", end_date=datetime.date.today().strftime("%Y%m%d"))
                if not latest_daily_data.empty:
                    trade_date = latest_daily_data['日期'].iloc[-1] # Akshare returns YYYY-MM-DD
                    trade_date = trade_date.replace("-","") # Convert to YYYYMMDD
                    self.logger.info(f"No trade_date provided, using latest available: {trade_date}")
                else:
                    self.logger.warning(f"Could not determine latest trade date for {asset_id}. Tick data fetch might fail.")
                    return []
            except Exception as e:
                self.logger.error(f"Error determining latest trade date for {asset_id}: {e}", exc_info=True)
                return []

        try:
            tick_df = akshare.stock_zh_a_tick_tx_js(symbol=ak_symbol, trade_date=trade_date)
            if tick_df is None or tick_df.empty:
                self.logger.warning(f"No tick data returned by Akshare for {asset_id} on {trade_date}.")
                return []

            # 标准化列名 (Standardize column names)
            # Akshare '成交时间', '成交价格', '价格变动', '成交量(手)', '成交额(元)', '性质'
            tick_data_list = []
            for _, row in tick_df.iterrows():
                tick_data_list.append({
                    "time": row.get("成交时间"),
                    "price": row.get("成交价格"),
                    "volume": row.get("成交量(手)"), # Volume in lots (手)
                    "turnover": row.get("成交额(元)"), # Turnover in Yuan
                    "type": row.get("性质") # '买盘', '卖盘', '中性盘' (Buy, Sell, Neutral)
                })
            return tick_data_list
        except Exception as e:
            self.logger.error(f"Error fetching tick data for {asset_id} on {trade_date} from Akshare: {e}", exc_info=True)
            return []

    def get_historical_klines(self, asset_id: str, period: str = '1', start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取指定资产和周期的历史K线数据 (OHLCV)。
        Fetches historical K-line data (OHLCV) for a given asset and period.

        Args:
            asset_id (str): 资产代码 (例如 "000001.SZ")。
                            Asset ID (e.g., "000001.SZ").
            period (str): K线周期 ('1', '5', '15', '30', '60' 分钟)。
                          K-line period ('1', '5', '15', '30', '60' minutes).
            start_date (Optional[str]): 开始日期 ("YYYYMMDD" 格式)。如果为 None，则获取近期数据。
                                       Start date in "YYYYMMDD" format. If None, fetches recent data.
            end_date (Optional[str]): 结束日期 ("YYYYMMDD" 格式)。如果为 None，默认为今天。
                                     End date in "YYYYMMDD" format. If None, defaults to today.
        Returns:
            List[Dict[str, Any]]: K线数据列表，每条记录包含 'time', 'open', 'high', 'low', 'close', 'volume'。
                                  A list of K-line data, each record containing 'time', 'open', 'high', 'low', 'close', 'volume'.
        """
        self.logger.info(f"Fetching {period}-min Klines for {asset_id} from {start_date} to {end_date if end_date else 'today'}.")
        ak_symbol_code_only = self._normalize_akshare_symbol(asset_id) # stock_zh_a_hist_min_em needs only code

        if start_date is None: # Default to a recent period if start_date is not provided
            start_date = (datetime.date.today() - datetime.timedelta(days=3)).strftime("%Y%m%d") 
        if end_date is None:
            end_date = datetime.date.today().strftime("%Y%m%d")
        
        try:
            # stock_zh_a_hist_min_em: 日期 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
            # For 1-minute data, it might only return data for the latest trading day if start_date/end_date are not specific enough
            # or if they span too short a period for older data.
            # If fetching for a specific single day, set start_date and end_date to that day.
            if period == '1' and start_date == end_date: # Specific handling for single day 1-min data
                 kline_df = akshare.stock_zh_a_hist_min_em(symbol=ak_symbol_code_only, period=period, start_date=start_date, end_date=end_date)
            else: # For other periods or ranges
                 kline_df = akshare.stock_zh_a_hist_min_em(symbol=ak_symbol_code_only, period=period, start_date=start_date, end_date=end_date)


            if kline_df is None or kline_df.empty:
                self.logger.warning(f"No {period}-min Klines returned by Akshare for {asset_id}.")
                return []

            kline_data_list = []
            for _, row in kline_df.iterrows():
                kline_data_list.append({
                    "time": row.get("日期"), # This is actually datetime for minute klines
                    "open": row.get("开盘"),
                    "high": row.get("最高"),
                    "low": row.get("最低"),
                    "close": row.get("收盘"),
                    "volume": row.get("成交量") 
                })
            return kline_data_list
        except Exception as e:
            self.logger.error(f"Error fetching {period}-min Klines for {asset_id} from Akshare: {e}", exc_info=True)
            return []

    def get_l2_snapshot(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定资产的L2快照数据 (注意: 这是单次快照，非连续数据流)。
        Fetches Level 2 snapshot data for a given asset (Note: this is a single snapshot, not a continuous stream).

        Args:
            asset_id (str): 资产代码 (例如 "600000.SH")。
                            Asset ID (e.g., "600000.SH").
        Returns:
            Optional[Dict[str, Any]]: L2快照数据字典，或在失败时返回None。
                                     A dictionary of L2 snapshot data, or None on failure.
                                     Includes bid/ask prices and volumes for multiple levels.
        """
        self.logger.info(f"Fetching L2 snapshot for {asset_id}.")
        ak_symbol = self._normalize_akshare_symbol(asset_id, for_level2=True)
        try:
            # Example output: dict_keys(['name', 'code', 'time', 'price', 'p_change', 'volume', 'amount', 
            # 'vol_ratio', 'buy_vol', 'sell_vol', 'pe', 'pb', 'amplitude', 'high', 'low', 'open', 
            # 'prev_close', 'bid_ask_price', 'bid_ask_volume'])
            # 'bid_ask_price' and 'bid_ask_volume' are lists of lists for buy1-5/sell1-5 prices and volumes.
            l2_data = akshare.stock_zh_a_level2_snapshot_em(symbol=ak_symbol) 
            if not l2_data: # Akshare might return an empty dict if symbol is wrong or no data
                self.logger.warning(f"No L2 snapshot data returned by Akshare for {asset_id}.")
                return None
            # Return the raw dictionary from Akshare as the structure is complex
            return l2_data 
        except Exception as e:
            self.logger.error(f"Error fetching L2 snapshot for {asset_id} from Akshare: {e}", exc_info=True)
            return None

    # ... (Existing methods: _calculate_atr, get_atr, get_market_state, process, make_recommendation) ...
    # (Keep existing methods as they were in Turn 95, unless they need minor adjustments for new data types)
    def _calculate_atr(self, high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int) -> pd.Series:
        if len(close_prices) < period +1 : 
            self.logger.warning(f"Not enough data for ATR period {period}, need {period+1} got {len(close_prices)}")
            return pd.Series([np.nan] * len(close_prices))
        prev_close = close_prices.shift(1); tr1 = high_prices - low_prices; tr2 = np.abs(high_prices - prev_close); tr3 = np.abs(low_prices - prev_close)
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = true_range.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        return atr

    def get_atr(self, asset_id: str, period: int = 14, historical_data_input: Optional[List[Dict[str, float]]] = None) -> Optional[float]:
        self.logger.info(f"Calculating ATR({period}) for asset: {asset_id}")
        hist_df = None
        if historical_data_input:
            if len(historical_data_input) < period + 50: self.logger.warning(f"Provided hist data for {asset_id} has {len(historical_data_input)} points, might be insufficient for ATR({period}). Fetching more."); historical_data_input = None
            else:
                hist_df = pd.DataFrame(historical_data_input)
                if not all(col in hist_df.columns for col in ['high', 'low', 'close']): self.logger.error(f"Provided hist data for {asset_id} missing cols."); return None
                hist_df[['high', 'low', 'close']] = hist_df[['high', 'low', 'close']].astype(float)
        if hist_df is None:
            end_date = datetime.date.today(); start_date_fetch = end_date - datetime.timedelta(days=period * 2 + 60) 
            try:
                # Assuming market_data_source is an Akshare wrapper or direct Akshare calls for individual stocks
                # For this POC, we directly use akshare.stock_zh_a_hist for individual stock daily data to calculate ATR
                # For more robust solution, self.market_data_source should abstract this.
                ak_symbol_code_only = self._normalize_akshare_symbol(asset_id)
                raw_hist_data = akshare.stock_zh_a_hist(symbol=ak_symbol_code_only, period="daily", start_date=start_date_fetch.strftime('%Y%m%d'), end_date=end_date.strftime('%Y%m%d'), adjust="qfq")

                if not isinstance(raw_hist_data, pd.DataFrame) or raw_hist_data.empty or len(raw_hist_data) < period + 1:
                    self.logger.warning(f"Insufficient daily data for {asset_id} to calc ATR({period}). Got {len(raw_hist_data) if isinstance(raw_hist_data, pd.DataFrame) else 0} points."); return None
                hist_df = raw_hist_data.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
                if not all(col in hist_df.columns for col in ['high', 'low', 'close']): self.logger.error(f"Fetched daily data for {asset_id} missing cols."); return None
                hist_df[['high', 'low', 'close']] = hist_df[['high', 'low', 'close']].astype(float)
            except Exception as e: self.logger.error(f"Error fetching daily data for ATR for {asset_id}: {e}", exc_info=True); return None
        if 'date' in hist_df.columns:
            try: hist_df['date'] = pd.to_datetime(hist_df['date']); hist_df = hist_df.sort_values(by='date', ascending=True)
            except Exception as e: self.logger.warning(f"Could not parse/sort date for ATR: {e}")
        if len(hist_df) < period + 1: self.logger.warning(f"Not enough data points ({len(hist_df)}) for ATR({period}) for {asset_id}."); return None
        atr_series = self._calculate_atr(hist_df['high'], hist_df['low'], hist_df['close'], period)
        if not atr_series.empty and not pd.isna(atr_series.iloc[-1]): return round(atr_series.iloc[-1], 4)
        else: self.logger.warning(f"ATR calc resulted NaN for {asset_id}."); return None

    def get_market_state(self, market_index_symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # (Implementation from previous turn - unchanged)
        self.logger.info(f"Calculating market state for index: {market_index_symbol}"); atr_period = config.get("atr_period", 14); short_ma_period = config.get("short_ma_period", 20); long_ma_period = config.get("long_ma_period", 50); vol_low_cutoff = config.get("volatility_cutoffs", {}).get("low", 0.012); vol_medium_cutoff = config.get("volatility_cutoffs", {}).get("medium", 0.025); required_days = long_ma_period + atr_period + 50; end_date = datetime.date.today(); start_date = end_date - datetime.timedelta(days=required_days * 1.5); market_volatility_level = "Unknown"; market_trend_state = "Unknown"; raw_volatility_score = np.nan; raw_short_ma = np.nan; raw_long_ma = np.nan; current_price = np.nan
        try:
            ak_symbol = ""; parts = market_index_symbol.split('.'); ak_symbol = parts[1].lower() + parts[0] if len(parts) == 2 else market_index_symbol
            hist_df = akshare.index_zh_a_hist(symbol=ak_symbol, period="daily", start_date=start_date.strftime("%Y%m%d"), end_date=end_date.strftime("%Y%m%d"))
            if hist_df.empty or len(hist_df) < long_ma_period: return {"market_volatility_level": "Unknown", "market_trend_state": "Unknown", "error": "Insufficient data"}
            hist_df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True); hist_df['date'] = pd.to_datetime(hist_df['date']); hist_df.set_index('date', inplace=True); hist_df[['open', 'close', 'high', 'low', 'volume']] = hist_df[['open', 'close', 'high', 'low', 'volume']].astype(float)
            atr_series = self._calculate_atr(hist_df['high'], hist_df['low'], hist_df['close'], atr_period)
            if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) and hist_df['close'].iloc[-1] > 0:
                raw_volatility_score = atr_series.iloc[-1] / hist_df['close'].iloc[-1]
                if raw_volatility_score < vol_low_cutoff: market_volatility_level = "Low"
                elif raw_volatility_score < vol_medium_cutoff: market_volatility_level = "Medium"
                else: market_volatility_level = "High"
            hist_df['short_ma'] = hist_df['close'].rolling(window=short_ma_period).mean(); hist_df['long_ma'] = hist_df['close'].rolling(window=long_ma_period).mean()
            if not pd.isna(hist_df['short_ma'].iloc[-1]) and not pd.isna(hist_df['long_ma'].iloc[-1]):
                raw_short_ma = hist_df['short_ma'].iloc[-1]; raw_long_ma = hist_df['long_ma'].iloc[-1]; current_price = hist_df['close'].iloc[-1]
                if raw_short_ma > raw_long_ma and current_price > raw_short_ma: market_trend_state = "Bullish"
                elif raw_short_ma < raw_long_ma and current_price < raw_short_ma: market_trend_state = "Bearish"
                else: market_trend_state = "Neutral/Ranging"
        except Exception as e: self.logger.error(f"Error calculating market state for {market_index_symbol}: {e}", exc_info=True); return {"market_volatility_level": "Error", "market_trend_state": "Error", "error": str(e)}
        return {"market_volatility_level": market_volatility_level, "market_trend_state": market_trend_state, "raw_volatility_score": round(raw_volatility_score, 5) if not pd.isna(raw_volatility_score) else None, "raw_short_ma": round(raw_short_ma, 2) if not pd.isna(raw_short_ma) else None, "raw_long_ma": round(raw_long_ma, 2) if not pd.isna(raw_long_ma) else None, "current_price": round(current_price, 2) if not pd.isna(current_price) else None}

    def process(self, data: Dict[str, Any]) -> None:
        # (Implementation from previous turn - unchanged)
        self.logger.info(f"Processing market data for symbol: {data.get('symbol', 'N/A')}, data_type: {data.get('data_type')}")
        if data.get('data_type') == 'market_state': self.processed_data = data.get('market_state', data)
        elif data.get('data_type') in ['price_and_atr', 'tick_data', '1min_klines', '5min_klines', 'l2_snapshot']: self.processed_data = data 
        else: self.processed_data = data.get('raw_market_data', data)
        if isinstance(self.processed_data, dict) and 'processed_timestamp' not in self.processed_data:
            if data.get('symbol') and data.get('data_type') == 'latest_price': self.processed_data['processed_timestamp'] = self.processed_data.get('timestamp')
            elif data.get('symbol') and data.get('data_type') == 'price_and_atr': self.processed_data['processed_timestamp'] = self.processed_data.get('price_info',{}).get('timestamp')
            elif data.get('data_type') not in ['market_state', 'tick_data', '1min_klines', '5min_klines', 'l2_snapshot']: self.processed_data['processed_timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.logger.debug(f"Market data processed: {self.processed_data}")


    def make_recommendation(self) -> Dict[str, Any]:
        # (Implementation from previous turn - unchanged)
        if self.processed_data: return {"status": "success", "data_type": "market_data", "content": self.processed_data}
        else: return {"status": "no_data", "data_type": "market_data", "content": {}}

    def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        (中文注释已更新)
        协调市场数据的获取和处理。
        可以获取历史数据、最新价格、ATR值、Tick数据、分钟K线、L2快照或计算市场状态。

        Orchestrates the fetching and processing of market data.
        Can fetch historical data, latest price, ATR, Tick data, minute K-lines, L2 snapshots, or calculate market state.

        Args:
            data: A dictionary containing parameters for fetching data.
                  - 'data_type': 'market_state', 'latest_price', 'historical', 'price_and_atr', 
                                 'tick_data', '1min_klines', '5min_klines', 'l2_snapshot'.
                  - 'symbol': Required for most data_types.
                  - Other params as needed by specific methods (e.g., 'trade_date', 'period').

        Returns:
            A dictionary containing the processed market data.
        """
        self.logger.info(f"MarketDataAgent run started with parameters: {data}")
        if not data or 'data_type' not in data:
            self.logger.error("Missing 'data_type' in run parameters.")
            return {"status": "error", "message": "Missing 'data_type' in run parameters."}

        data_type = data['data_type']
        processed_payload: Dict[str, Any] = {'data_type': data_type} 

        try:
            if data_type == 'market_state':
                # ... (market_state logic from previous turn)
                if 'market_index_symbol' not in data or 'market_state_config' not in data: return {"status": "error", "message": "Missing params for market_state."}
                market_state_result = self.get_market_state(data['market_index_symbol'], data['market_state_config']); processed_payload = market_state_result; processed_payload['data_type'] = 'market_state'
            
            elif 'symbol' in data:
                symbol = data['symbol']; processed_payload['symbol'] = symbol
                if data_type == 'latest_price': processed_payload['raw_market_data'] = self.market_data_source.get_latest_price(symbol)
                elif data_type == 'historical': processed_payload['raw_market_data'] = self.market_data_source.get_historical_data(symbol, data.get('start_date'), data.get('end_date'))
                elif data_type == 'price_and_atr':
                    latest_price = self.market_data_source.get_latest_price(symbol); atr_value = self.get_atr(symbol, period=data.get('atr_period', 14))
                    processed_payload['price_info'] = latest_price; processed_payload['atr_value'] = atr_value
                elif data_type == 'tick_data':
                    processed_payload['tick_data'] = self.get_tick_data(symbol, trade_date=data.get('trade_date'))
                elif data_type.endswith('_klines'): # e.g., '1min_klines', '5min_klines'
                    period_map = {'1min_klines': '1', '5min_klines': '5', '15min_klines': '15', '30min_klines': '30', '60min_klines': '60'}
                    kline_period = period_map.get(data_type, '1') # Default to 1 min if not specified
                    processed_payload[f'{kline_period}min_klines_data'] = self.get_historical_klines(symbol, period=kline_period, start_date=data.get('start_date'), end_date=data.get('end_date'))
                elif data_type == 'l2_snapshot':
                    processed_payload['l2_snapshot_data'] = self.get_l2_snapshot(symbol)
                else: # Fallback to generic fetch_data or market_data_source methods
                    self.logger.warning(f"Attempting generic fetch for data_type: {data_type}"); processed_payload['raw_market_data'] = self.market_data_source.fetch_data(symbol, **data)
                
                if processed_payload.get('raw_market_data') is None and data_type not in ['price_and_atr', 'tick_data', '1min_klines', '5min_klines', 'l2_snapshot', 'market_state']:
                     self.logger.warning(f"No raw data fetched for {symbol} with data_type {data_type}.")
            else: 
                 self.logger.error(f"Missing 'symbol' for data_type '{data_type}'."); return {"status": "error", "message": f"Missing 'symbol' for data_type '{data_type}'."}
            
            self.process(processed_payload) # Sets self.processed_data

        except Exception as e: self.logger.error(f"Error during MarketDataAgent run: {e}", exc_info=True); return {"status": "error", "message": str(e)}
        return self.make_recommendation()


if __name__ == '__main__':
    # (Test cases from previous turn, with new ones to be added)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import json # For pretty printing dicts
    
    class MockMarketDataSource: # Simplified
        def get_historical_data(self, symbol, start_date, end_date): return [{"date": (datetime.date.today() - datetime.timedelta(days=i)).strftime('%Y-%m-%d'), "close": 100-i, "high":102-i, "low":98-i, "open":100-i, "volume": 10000+i*100} for i in range(60)][::-1]
        def get_latest_price(self, symbol): return {"symbol": symbol, "price": 110.5, "timestamp": "now"}
        def fetch_data(self, symbol, **kwargs): return {"data": "generic"}

    mock_source = MockMarketDataSource()
    market_agent = MarketDataAgent(market_data_source=mock_source)

    print("\n--- Testing MarketDataAgent (Microstructure Data) ---")
    
    # Test Tick Data (uses Akshare directly, may need live market or specific setup if not mocked deeper)
    # For automated testing, akshare.stock_zh_a_tick_tx_js would need to be mocked.
    # Assuming it runs where Akshare can be called.
    print("\n1. Testing Tick Data (000001.SZ for a recent valid trade date):")
    # trade_date_ticks = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y%m%d") # Example: yesterday
    # For CI/CD or environments where live data access for specific dates is tricky, this might be problematic.
    # A fixed, known past date with data would be better for repeatable tests if Akshare allows.
    # Let's assume a fixed date for now for conceptual testing; replace if needed.
    trade_date_ticks = "20240614" # Replace with a known valid date if needed for testing
    tick_params = {'symbol': '000001.SZ', 'data_type': 'tick_data', 'trade_date': trade_date_ticks}
    result_ticks = market_agent.run(data=tick_params)
    print(f"Result (ticks for 000001.SZ on {trade_date_ticks}):")
    if result_ticks['status'] == 'success' and result_ticks['content'].get('tick_data'):
        print(json.dumps(result_ticks['content']['tick_data'][:5], indent=2, ensure_ascii=False)) # Print first 5 ticks
    else:
        print(json.dumps(result_ticks, indent=2, ensure_ascii=False))

    # Test 1-min Klines (uses Akshare)
    print("\n2. Testing 1-min Klines (000001.SZ for today):")
    # Note: akshare.stock_zh_a_hist_min_em might only return current day's 1-min data.
    today_date_str = datetime.date.today().strftime("%Y%m%d")
    kline_params = {'symbol': '000001.SZ', 'data_type': '1min_klines', 'start_date': today_date_str, 'end_date': today_date_str}
    result_klines = market_agent.run(data=kline_params)
    print(f"Result (1-min klines for 000001.SZ on {today_date_str}):")
    if result_klines['status'] == 'success' and result_klines['content'].get('1min_klines_data'):
        print(json.dumps(result_klines['content']['1min_klines_data'][:5], indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result_klines, indent=2, ensure_ascii=False))

    # Test L2 Snapshot (uses Akshare)
    print("\n3. Testing L2 Snapshot (000001.SZ):")
    l2_params = {'symbol': '000001.SZ', 'data_type': 'l2_snapshot'}
    result_l2 = market_agent.run(data=l2_params)
    print(f"Result (L2 snapshot for 000001.SZ):")
    # L2 data can be large, just print status or a few keys
    if result_l2['status'] == 'success' and result_l2['content'].get('l2_snapshot_data'):
        print(f"  L2 Snapshot fetched. Keys: {list(result_l2['content']['l2_snapshot_data'].keys())[:5]}...") # Print some keys
    else:
        print(json.dumps(result_l2, indent=2, ensure_ascii=False))

    print("\n--- MarketDataAgent Microstructure tests complete ---")
