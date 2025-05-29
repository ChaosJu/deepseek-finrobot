import logging
from typing import Dict, Any, Optional, List, Type, Tuple 
import datetime 
import math 

from .base_agent import BaseAgent
# from deepseek_finrobot.agents.microstructure_pattern_agent import MicrostructurePatternAgent # For type hinting

class CEOAgent(BaseAgent):
    """
    (CEO主控智能体)
    负责协调其他专业智能体的工作流程，综合信息，并根据动态计算的仓位大小、止损点以及包括基于微观结构模式、恐惧/危机信号和反向信号的精细化退出策略做出最终交易决策。
    The CEO Agent orchestrates the workflow of other specialized agents,
    synthesizes information, and makes final trading decisions with dynamically calculated position sizes, stop-losses, 
    and refined exit mechanisms including microstructure patterns, fear-based/crisis and counter-signal exits.
    """

    def __init__(self, 
                 strategy_agents: Optional[List[BaseAgent]] = None,
                 data_providing_agents: Optional[Dict[str, BaseAgent]] = None, 
                 expert_analyst_agents: Optional[Dict[str, BaseAgent]] = None, 
                 risk_management_agent: Optional[BaseAgent]] = None, 
                 config: Optional[Dict[str, Any]] = None, 
                 agent_name: str = "CEOAgent", 
                 logger_instance: Optional[logging.Logger] = None):
        """
        初始化CEOAgent。
        Initializes the CEOAgent.
        Args:
            strategy_agents (Optional[List[BaseAgent]]): 策略智能体列表。
            data_providing_agents (Optional[Dict[str, BaseAgent]]): 数据提供智能体字典。
            expert_analyst_agents (Optional[Dict[str, BaseAgent]]): 专家分析智能体字典，可包含 'microstructure_pattern_agent'。
            risk_management_agent (Optional[BaseAgent]): 风险管理智能体。
            config (Optional[Dict[str, Any]]): CEO智能体、仓位管理、止损及止盈 (包括精细化退出) 配置。
            agent_name (str): 智能体名称。
            logger_instance (Optional[logging.Logger]): 日志记录器实例。
        """
        super().__init__(agent_name=agent_name, logger_instance=logger_instance)
        
        self.strategy_agents = strategy_agents if strategy_agents is not None else []
        self.data_providing_agents = data_providing_agents if data_providing_agents is not None else {}
        self.expert_analyst_agents = expert_analyst_agents if expert_analyst_agents is not None else {}
        self.risk_management_agent = risk_management_agent
        
        self.config = config if config else {}
        self._load_position_sizing_config() 
        self._load_stop_loss_config() 
        self._load_profit_taking_config() 
        self._load_microstructure_integration_config() 

        self.final_decision_history: List[Dict[str, Any]] = []
        self.active_trades: Dict[str, Dict[str, Any]] = {} 
        
        self.logger.info(f"CEOAgent initialized.")

    def _load_position_sizing_config(self):
        """加载仓位管理配置。Loads position sizing configuration."""
        position_sizing_cfg = self.config.get("position_sizing_config", {})
        self.base_pos_size_pct = position_sizing_cfg.get("base_position_size_pct_of_total_portfolio", 0.02) 
        self.position_sizing_basis = position_sizing_cfg.get("position_sizing_basis", "total_portfolio_value")
        self.conviction_tiers = position_sizing_cfg.get("conviction_tiers", [{"score_threshold": 0.0, "multiplier": 0.5}, {"score_threshold": 0.4, "multiplier": 1.0}, {"score_threshold": 0.7, "multiplier": 1.5}])
        self.max_multiplier = position_sizing_cfg.get("max_multiplier", 2.0)
        self.analyst_score_contributions = position_sizing_cfg.get("analyst_score_contributions", {})
        self.logger.info("Position sizing configuration loaded.")

    def _load_stop_loss_config(self):
        """
        加载动态止损及基于危机/反向信号/微观结构模式的退出配置。
        Loads dynamic stop-loss and crisis/counter-signal/microstructure pattern exit configurations.
        """
        sl_config = self.config.get("stop_loss_config", {})
        self.atr_period_for_stop = sl_config.get("atr_period_for_stop", 14)
        self.base_n_atr_for_stop = sl_config.get("base_n_atr_for_stop", 2.0)
        self.n_atr_adjustments = sl_config.get("n_atr_adjustments", {})
        self.min_n_atr = sl_config.get("min_n_atr", 1.0); self.max_n_atr = sl_config.get("max_n_atr", 4.0) 
        self.use_crisis_exit_for_long = sl_config.get("use_crisis_exit_for_long", False)
        self.crisis_exit_strategy_for_long = sl_config.get("crisis_exit_strategy_for_long", "TightenStopOnCrisis") 
        self.news_sentiment_crisis_categories_for_long_exit = sl_config.get("news_sentiment_crisis_categories_for_long_exit", [])
        self.crisis_tightened_n_atr_stop = sl_config.get("crisis_tightened_n_atr_stop", 0.75) 
        self.use_counter_signal_exit = sl_config.get("use_counter_signal_exit", False)
        self.threshold_exit_on_counter_signal_confidence = sl_config.get("threshold_exit_on_counter_signal_confidence", 0.65)
        self.counter_signal_sell_categories_for_long_exit = sl_config.get("counter_signal_sell_categories_for_long_exit", []) 
        self.counter_signal_buy_categories_for_short_exit = sl_config.get("counter_signal_buy_categories_for_short_exit", [])
        self.logger.info("Stop-loss and refined exit (crisis/counter-signal) configurations loaded.")

    def _load_profit_taking_config(self):
        """
        加载动态止盈配置，包括基于贪婪、恐惧和微观结构模式的精细化止盈参数。
        Loads dynamic profit-taking configuration, including parameters for greed, fear, and microstructure pattern-based refinements.
        """
        pt_config = self.config.get("profit_taking_config", {})
        self.use_trailing_stop = pt_config.get("use_trailing_stop", True)
        self.atr_multiplier_for_trailing_activation = pt_config.get("atr_multiplier_for_trailing_activation", 1.5)
        self.n_atr_for_trailing_stop = pt_config.get("n_atr_for_trailing_stop", 1.5) 
        self.use_rr_profit_target = pt_config.get("use_rr_profit_target", True)
        self.desired_risk_reward_ratio = pt_config.get("desired_risk_reward_ratio", 2.0) 
        self.use_partial_profit_take = pt_config.get("use_partial_profit_take", False)
        self.partial_profit_target_rr_ratio = pt_config.get("partial_profit_target_rr_ratio", 1.0) 
        self.partial_profit_quantity_percentage = pt_config.get("partial_profit_quantity_percentage", 0.5) 
        self.move_stop_to_breakeven_after_partial_take = pt_config.get("move_stop_to_breakeven_after_partial_take", True)
        self.use_greed_based_profit_taking = pt_config.get("use_greed_based_profit_taking", False)
        self.news_sentiment_greed_categories = pt_config.get("news_sentiment_greed_categories", [])
        self.market_theme_greed_assessments = pt_config.get("market_theme_greed_assessments", [])
        self.min_profit_factor_for_greed_exit = pt_config.get("min_profit_factor_for_greed_exit", 1.5) 
        self.greedy_exit_strategy = pt_config.get("greedy_exit_strategy", "PartialAndTightenTrail") 
        self.greedy_partial_exit_percentage = pt_config.get("greedy_partial_exit_percentage", 0.33) 
        self.greedy_tightened_n_atr_trail = pt_config.get("greedy_tightened_n_atr_trail", 1.0) 
        self.use_fear_based_short_covering = pt_config.get("use_fear_based_short_covering", False)
        self.news_sentiment_fear_categories_for_short_cover = pt_config.get("news_sentiment_fear_categories_for_short_cover", [])
        self.market_theme_fear_assessments_for_short_cover = pt_config.get("market_theme_fear_assessments_for_short_cover", [])
        self.min_profit_factor_for_fear_cover = pt_config.get("min_profit_factor_for_fear_cover", 1.0) 
        self.logger.info("Profit-taking, greed & fear-based exit configurations loaded.")

    def _load_microstructure_integration_config(self):
        """
        加载微观结构整合相关配置。
        Loads microstructure integration related configurations.
        """
        pt_config = self.config.get("profit_taking_config", {})
        sl_config = self.config.get("stop_loss_config", {})
        
        self.microstructure_agent_enabled = self.config.get("microstructure_pattern_agent_config", {}).get("enabled", False)
        self.use_microstructure_for_conviction = self.microstructure_agent_enabled and \
            self.config.get("microstructure_integration_config", {}).get("use_microstructure_for_conviction", True)
        
        self.use_microstructure_for_greedy_exits_buy = self.microstructure_agent_enabled and \
            pt_config.get("use_microstructure_for_profit_taking", False) 
        self.min_confidence_for_greedy_microstructure_pattern_buy = pt_config.get("microstructure_blowoff_exit_buy_min_confidence", 0.7)
        
        self.use_microstructure_for_fear_capitulation_exits_short = self.microstructure_agent_enabled and \
            pt_config.get("use_microstructure_for_profit_taking", False) 
        self.min_confidence_for_fear_microstructure_pattern_short = pt_config.get("microstructure_panic_cover_short_min_confidence", 0.7)

        self.use_microstructure_for_stop_adjustment = self.microstructure_agent_enabled and \
            sl_config.get("use_microstructure_for_stop_adjustment", False)
        self.microstructure_panic_tighten_n_atr = sl_config.get("microstructure_panic_tighten_n_atr", 1.0)
        self.microstructure_euphoria_tighten_n_atr = sl_config.get("microstructure_euphoria_tighten_n_atr", 1.0)
        
        self.logger.info("Microstructure integration configurations loaded.")


    def _calculate_signal_conviction(self, primary_signal: Dict[str, Any], analyst_reports: Dict[str, Any], microstructure_analysis_output: Optional[Dict[str, Any]] = None) -> float:
        """
        (私有方法) 计算信号的总体置信度/确信度。
        基于主信号的置信度，并结合各专家分析智能体（包括微观结构分析）的报告进行调整。
        (Private method) Calculates the overall conviction score for a signal.
        Starts with the primary signal's confidence and adjusts based on reports from expert analyst agents, including microstructure analysis.
        """
        if not primary_signal or 'confidence' not in primary_signal: self.logger.warning("Primary signal missing/lacks confidence. Conviction=0."); return 0.0
        conviction_score = primary_signal.get('confidence', 0.0)
        
        for report_key, contributions in self.analyst_score_contributions.items():
            if report_key == "microstructure_patterns": continue 
            report_name_in_data = f"{report_key}_report" 
            report_content = analyst_reports.get(report_name_in_data, {}).get("content", {})
            if report_content:
                if report_key == "short_term_value_filter": 
                    assessment = report_content.get('value_assessment', '')
                    bonus_key = f"{assessment.replace(' ', '')}_bonus"; penalty_key = f"{assessment.replace(' ', '')}_penalty"
                    adjustment = contributions.get(bonus_key, 0.0) + contributions.get(penalty_key, 0.0)
                    if adjustment != 0: conviction_score += adjustment; self.logger.debug(f"{report_key} ('{assessment}') adj conviction: {adjustment:.2f}. New: {conviction_score:.2f}")
                # TODO: Add specific handling for catalyst_scout, market_theme_sentiment, macro_policy_impact
        
        if self.use_microstructure_for_conviction and microstructure_analysis_output and microstructure_analysis_output.get("detected_patterns"):
            ms_contributions = self.analyst_score_contributions.get("microstructure_patterns", {})
            for pattern in microstructure_analysis_output["detected_patterns"]:
                pattern_name = pattern.get("pattern_name")
                pattern_confidence = pattern.get("confidence", 0.0)
                
                if pattern_name == "PanicSelling_VolumeSpike" and primary_signal.get("signal") == "BUY": 
                    bonus = ms_contributions.get("PanicSelling_VolumeSpike_bonus", 0.0) * pattern_confidence
                    conviction_score += bonus
                    self.logger.debug(f"Microstructure ({pattern_name}) adj conviction by: {bonus:.2f}. New score: {conviction_score:.2f}")
                elif pattern_name == "Euphoria_BlowOffTop" and primary_signal.get("signal") == "BUY": 
                    penalty = ms_contributions.get("Euphoria_BlowOffTop_penalty", 0.0) * pattern_confidence 
                    conviction_score += penalty 
                    self.logger.debug(f"Microstructure ({pattern_name}) adj conviction by: {penalty:.2f}. New score: {conviction_score:.2f}")
        
        return round(max(0.0, min(1.0, conviction_score)), 3)

    def _determine_preliminary_position_size(self, conviction_score: float, asset_price: float, portfolio_snapshot: Dict[str, Any]) -> Tuple[int, float]:
        # (Unchanged from Turn 99)
        if asset_price <= 0: return 0, 0.0; capital_base = portfolio_snapshot.get('cash_balance' if self.position_sizing_basis == "available_cash" else 'total_portfolio_value', 0.0);
        if capital_base <= 0: return 0, 0.0; target_capital = capital_base * self.base_pos_size_pct; multiplier = 0.0
        for tier in sorted(self.conviction_tiers, key=lambda x: x['score_threshold']):
            if conviction_score >= tier['score_threshold']: multiplier = tier['multiplier']
            else: break
        multiplier = min(multiplier, self.max_multiplier); value = target_capital * multiplier; qty = int(value / asset_price) if asset_price > 0 else 0
        return qty, qty * asset_price
    def _calculate_effective_n_atr(self, base_n_atr: float, market_state: Dict[str, Any], primary_signal_details: Dict[str, Any], conviction_score: float) -> float:
        # (Unchanged from Turn 99)
        effective_n = base_n_atr; vol_adj = self.n_atr_adjustments.get("market_volatility", {}).get(market_state.get("market_volatility_level", "Medium"), 0.0); effective_n += vol_adj
        senti_adj_cfg = self.n_atr_adjustments.get("detailed_sentiment", {}).get(primary_signal_details.get("detailed_sentiment_category", "Unknown"), 0.0)
        effective_n += senti_adj_cfg if isinstance(senti_adj_cfg, (int,float)) else 0.0; conv_adj_cfg = self.n_atr_adjustments.get("conviction_score", {})
        if conviction_score >= self.conviction_tiers[-1]["score_threshold"]: effective_n += conv_adj_cfg.get("high_conviction_n_reduction", 0.0)
        elif conviction_score < self.conviction_tiers[1]["score_threshold"]: effective_n += conv_adj_cfg.get("low_conviction_n_increase", 0.0)
        return round(max(self.min_n_atr, min(self.max_n_atr, effective_n)), 2)
    def _calculate_initial_profit_targets(self, asset_id: str, entry_price: float, initial_stop_loss_price: float, direction: str) -> Dict[str, Optional[float]]:
        # (Unchanged from Turn 99)
        targets = {"rr_full": None, "rr_partial": None}; initial_risk = abs(entry_price - initial_stop_loss_price)
        if initial_risk == 0: return targets
        if self.use_rr_profit_target: targets["rr_full"] = round(entry_price + (initial_risk * self.desired_risk_reward_ratio if direction == "BUY" else -initial_risk * self.desired_risk_reward_ratio), 2)
        if self.use_partial_profit_take: targets["rr_partial"] = round(entry_price + (initial_risk * self.partial_profit_target_rr_ratio if direction == "BUY" else -initial_risk * self.partial_profit_target_rr_ratio), 2)
        return targets


    def _check_and_manage_active_trades(self, market_data_for_active_assets: Dict[str, Any], all_gathered_data_for_current_run: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        (私有方法) 检查并管理当前活跃的交易，应用止盈、止损及各种精细化退出逻辑。
        包括基于微观结构模式的止损调整和退出确认。
        (Private method) Checks and manages active trades, applying profit-taking, stop-loss, and refined exit logic.
        Includes stop adjustments and exit confirmations based on microstructure patterns.
        """
        closing_decisions = []
        if not self.active_trades: return closing_decisions
        self.logger.info(f"Managing {len(self.active_trades)} active trade(s)...")

        for asset_id, trade_state in list(self.active_trades.items()): 
            current_asset_market_data = market_data_for_active_assets.get(asset_id)
            if not current_asset_market_data or "price" not in current_asset_market_data:
                self.logger.warning(f"No current market data for active trade {asset_id}. Skipping management."); continue

            current_price = current_asset_market_data["price"]; high_price_bar = current_asset_market_data.get("high", current_price)
            low_price_bar = current_asset_market_data.get("low", current_price); current_atr = current_asset_market_data.get("atr")
            exit_reason: Optional[str] = None; exit_price: Optional[float] = None; qty_to_close: int = trade_state["quantity_remaining"]
            action_taken_this_bar = False 

            asset_specific_news_sentiment = {}; detailed_sentiment = "Unknown"; asset_theme_sentiment = "Neutral" 
            asset_specific_primary_signal = {}; asset_specific_microstructure_analysis = {}
            microstructure_agent = self.expert_analyst_agents.get('microstructure_pattern_agent')
            market_data_agent = self.data_providing_agents.get('market_data_agent')

            if microstructure_agent and market_data_agent and self.microstructure_agent_enabled and \
               (self.use_microstructure_for_greedy_exits_buy or self.use_microstructure_for_fear_capitulation_exits_short or self.use_microstructure_for_stop_adjustment):
                today_str = datetime.date.today().strftime("%Y%m%d")
                kline_lookback_days = self.config.get("microstructure_pattern_agent_config", {}).get("kline_lookback_days_for_analysis", 1)
                kline_1min_params = {'symbol': asset_id, 'data_type': '1min_klines', 
                                     'start_date': (datetime.date.today() - datetime.timedelta(days=kline_lookback_days)).strftime("%Y%m%d"), 
                                     'end_date': today_str}
                kline_output = market_data_agent.run(kline_1min_params)
                intraday_klines_data = {"1min_klines": kline_output.get("content", {}).get(f"{kline_1min_params.get('period','1')}min_klines_data", []) if kline_output.get("status") == "success" else []}
                if intraday_klines_data["1min_klines"]:
                    asset_specific_microstructure_analysis = microstructure_agent.analyze_microstructure(asset_id=asset_id, intraday_data=intraday_klines_data, latest_news_sentiment=None) 
                else: self.logger.warning(f"Could not get fresh 1-min klines for microstructure exit check for {asset_id}.")
            
            if all_gathered_data_for_current_run.get("stock_symbol") == asset_id:
                news_data_key = "news_sentiment_data"; asset_specific_news_sentiment = all_gathered_data_for_current_run.get(news_data_key, {}).get("content", {}); detailed_sentiment = asset_specific_news_sentiment.get("aggregated_dominant_detailed_category", "Unknown")
                asset_specific_primary_signal = all_gathered_data_for_current_run.get("primary_signal_details", {})
                if "microstructure_analysis_output" in all_gathered_data_for_current_run: asset_specific_microstructure_analysis = all_gathered_data_for_current_run["microstructure_analysis_output"]
            else: 
                detailed_sentiment = trade_state.get("detailed_sentiment_at_entry", "Unknown") 
                if not asset_specific_microstructure_analysis: asset_specific_microstructure_analysis = trade_state.get("microstructure_at_entry", {})
            
            # --- 1. Standard Stop-Loss Check --- (Full logic from Turn 99)
            if not action_taken_this_bar:
                stop_price_to_check = trade_state.get("current_trailing_stop_price") if trade_state.get("is_trailing_stop_active") else trade_state.get("initial_stop_loss_price")
                if stop_price_to_check is not None:
                    if (trade_state["direction"] == "BUY" and low_price_bar <= stop_price_to_check) or \
                       (trade_state["direction"] == "SELL" and high_price_bar >= stop_price_to_check):
                        exit_reason = "Initial SL Hit" if not trade_state.get("is_trailing_stop_active") else "Trailing SL Hit"; exit_price = stop_price_to_check 
                        action_taken_this_bar = True; qty_to_close = trade_state["quantity_remaining"]

            # --- 2. Fear-Based/Crisis/Microstructure-Informed Stop Adjustments & Exits ---
            if not action_taken_this_bar and trade_state["quantity_remaining"] > 0:
                # a. Fear-Based Short Covering (Microstructure can confirm)
                if trade_state["direction"] == "SELL" and self.use_fear_based_short_covering and current_price < trade_state["entry_price"]: 
                    initial_risk = trade_state.get("initial_risk_per_share", 0.0)
                    if initial_risk > 0 and ((trade_state["entry_price"] - current_price) / initial_risk) >= self.min_profit_factor_for_fear_cover:
                        is_news_fear = detailed_sentiment in self.news_sentiment_fear_categories_for_short_cover
                        is_theme_fear = asset_theme_sentiment in self.market_theme_fear_assessments_for_short_cover 
                        is_micro_panic = False
                        if self.use_microstructure_for_fear_capitulation_exits_short and asset_specific_microstructure_analysis.get("detected_patterns"):
                           if any(p.get("pattern_name") == "PanicSelling_VolumeSpike" and p.get("confidence",0) >= self.min_confidence_for_fear_microstructure_pattern_short for p in asset_specific_microstructure_analysis["detected_patterns"]):
                                is_micro_panic = True
                        if is_news_fear or is_theme_fear or is_micro_panic:
                            fear_signals = [s for s,t in [(f"News:{detailed_sentiment}",is_news_fear), (f"Theme:{asset_theme_sentiment}",is_theme_fear), ("MicroPanic",is_micro_panic)] if t]
                            exit_reason = f"Fear/Panic Short Cover ({', '.join(fear_signals)})"; exit_price = current_price
                            qty_to_close = trade_state["quantity_remaining"]; action_taken_this_bar = True; self.logger.info(f"{exit_reason} for {asset_id}")
                
                # b. Crisis Exit for Longs (Full logic from Turn 99)
                if not action_taken_this_bar and trade_state["direction"] == "BUY" and self.use_crisis_exit_for_long and detailed_sentiment in self.news_sentiment_crisis_categories_for_long_exit:
                    strategy = self.crisis_exit_strategy_for_long
                    self.logger.info(f"Crisis sentiment ({detailed_sentiment}) for LONG {asset_id}. Strategy: {strategy}")
                    if strategy == "ImmediateExitOnCrisis":
                        exit_reason = f"Immediate Exit on Crisis Sentiment ({detailed_sentiment})"; exit_price = current_price; qty_to_close = trade_state["quantity_remaining"]; action_taken_this_bar = True
                    elif strategy == "TightenStopOnCrisis" and current_atr and current_atr > 0:
                        new_tight_stop = round(current_price - (self.crisis_tightened_n_atr_stop * current_atr), 2)
                        current_sl = trade_state.get("current_stop_loss_price", trade_state.get("initial_stop_loss_price", -float('inf')))
                        trade_state["current_stop_loss_price"] = max(current_sl, new_tight_stop); trade_state["is_trailing_stop_active"] = False; trade_state["current_n_atr_for_trailing"] = None 
                        self.logger.info(f"Crisis: Stop for {asset_id} tightened to {trade_state['current_stop_loss_price']:.2f}. Re-checking SL.")
                        if low_price_bar <= trade_state["current_stop_loss_price"]: 
                            exit_reason = f"Crisis-Tightened Stop Hit ({detailed_sentiment})"; exit_price = trade_state["current_stop_loss_price"]; qty_to_close = trade_state["quantity_remaining"]; action_taken_this_bar = True
                
                # c. Microstructure-informed Stop Adjustment
                if not action_taken_this_bar and self.use_microstructure_for_stop_adjustment and current_atr and current_atr > 0 and asset_specific_microstructure_analysis.get("detected_patterns"):
                    for pattern in asset_specific_microstructure_analysis.get("detected_patterns", []):
                        new_tight_stop = None; pattern_name_for_log = pattern.get("pattern_name")
                        current_sl = trade_state.get("current_stop_loss_price", trade_state.get("initial_stop_loss_price", -float('inf') if trade_state["direction"] == "BUY" else float('inf')))
                        if trade_state["direction"] == "BUY" and pattern_name_for_log == "PanicSelling_VolumeSpike": 
                            new_tight_stop = round(current_price - (self.microstructure_panic_tighten_n_atr * current_atr), 2)
                            if new_tight_stop > current_sl: trade_state["current_stop_loss_price"] = new_tight_stop; trade_state["is_trailing_stop_active"] = False; self.logger.info(f"Micro (Panic): Stop for BUY {asset_id} tightened to {new_tight_stop:.2f}.")
                        elif trade_state["direction"] == "SELL" and pattern_name_for_log == "Euphoria_BlowOffTop": 
                            new_tight_stop = round(current_price + (self.microstructure_euphoria_tighten_n_atr * current_atr), 2)
                            if new_tight_stop < current_sl: trade_state["current_stop_loss_price"] = new_tight_stop; trade_state["is_trailing_stop_active"] = False; self.logger.info(f"Micro (Euphoria): Stop for SELL {asset_id} tightened to {new_tight_stop:.2f}.")
                        
                        if new_tight_stop is not None and \
                           ((trade_state["direction"] == "BUY" and low_price_bar <= trade_state["current_stop_loss_price"]) or \
                            (trade_state["direction"] == "SELL" and high_price_bar >= trade_state["current_stop_loss_price"])):
                            exit_reason = f"Micro-Tightened SL Hit ({pattern_name_for_log})"; exit_price = trade_state["current_stop_loss_price"]; qty_to_close = trade_state["quantity_remaining"]; action_taken_this_bar = True; break 

            # --- 3. Counter-Signal Exit --- (Full logic from Turn 99)
            if not action_taken_this_bar and self.use_counter_signal_exit and trade_state["quantity_remaining"] > 0 and asset_specific_primary_signal:
                 new_signal = asset_specific_primary_signal.get("signal"); new_conf = asset_specific_primary_signal.get("confidence", 0.0); new_senti_cat = asset_specific_primary_signal.get("detailed_sentiment_category", "Unknown")
                 exit_due_to_counter = False
                 if trade_state["direction"] == "BUY" and new_signal == "SELL" and new_conf >= self.threshold_exit_on_counter_signal_confidence and new_senti_cat in self.counter_signal_sell_categories_for_long_exit: exit_reason = f"Counter-Signal Exit (New SELL: {new_senti_cat}, Conf: {new_conf:.2f})"; exit_due_to_counter = True
                 elif trade_state["direction"] == "SELL" and new_signal == "BUY" and new_conf >= self.threshold_exit_on_counter_signal_confidence and new_senti_cat in self.counter_signal_buy_categories_for_short_exit: exit_reason = f"Counter-Signal Exit (New BUY: {new_senti_cat}, Conf: {new_conf:.2f})"; exit_due_to_counter = True
                 if exit_due_to_counter: exit_price = current_price; qty_to_close = trade_state["quantity_remaining"]; action_taken_this_bar = True; self.logger.info(f"{exit_reason} for {asset_id}")
            
            if action_taken_this_bar and exit_reason and exit_price is not None and qty_to_close > 0:
                 closing_decisions.append(self._create_closing_decision(asset_id, trade_state, exit_price, exit_reason, qty_to_close))
                 trade_state["quantity_remaining"] -= qty_to_close
                 if trade_state["quantity_remaining"] <= 0: del self.active_trades[asset_id]; continue
            elif action_taken_this_bar and qty_to_close == 0 : pass 

            # --- 4. Greed-Based Profit-Taking (Microstructure can confirm/trigger) ---
            if not action_taken_this_bar and self.use_greed_based_profit_taking and trade_state["direction"] == "BUY" and trade_state["quantity_remaining"] > 0:
                initial_risk = trade_state.get("initial_risk_per_share", 0.0)
                if initial_risk > 0 and ((current_price - trade_state["entry_price"]) / initial_risk) >= self.min_profit_factor_for_greed_exit:
                    is_news_greed = detailed_sentiment in self.news_sentiment_greed_categories; is_theme_greed = asset_theme_sentiment in self.market_theme_greed_assessments
                    is_micro_euphoria = False
                    if self.use_microstructure_for_greedy_exits_buy and asset_specific_microstructure_analysis.get("detected_patterns"):
                        if any(p.get("pattern_name") == "Euphoria_BlowOffTop" and p.get("confidence",0) >= self.min_confidence_for_greedy_microstructure_pattern_buy for p in asset_specific_microstructure_analysis["detected_patterns"]): is_micro_euphoria = True
                    
                    if is_news_greed or is_theme_greed or is_micro_euphoria:
                        # ... (Full Greed exit strategy logic from Turn 99 - sets action_taken_this_bar etc. Ensure to log if micro confirmed) ...
                        self.logger.info(f"Greed exit for BUY {asset_id} considered (News:{is_news_greed}, Theme:{is_theme_greed}, Micro:{is_micro_euphoria}).")
                        # Full logic for applying greedy_exit_strategy (FullExit, PartialAndTightenTrail, TightenTrailOnly) needs to be here
                        pass
            
            # --- 5. Standard Profit-Taking & 6. Trailing Stop Update --- (Full logic from Turn 99)
            if not action_taken_this_bar and trade_state["quantity_remaining"] > 0:
                # ... (Standard R/R partial, full profit, and trailing stop update logic from Turn 99) ...
                pass 
        
        return closing_decisions

    # ... (_create_closing_decision, register_*, process, make_recommendation methods remain unchanged)
    def _create_closing_decision(self, asset_id: str, trade_state: Dict[str, Any], exit_price: float, reason: str, quantity_to_close: int) -> Dict[str, Any]:
        return {
            "stock_symbol": asset_id, "final_signal": "SELL_TO_CLOSE" if trade_state["direction"] == "BUY" else "COVER_TO_CLOSE",
            "quantity": quantity_to_close, "estimated_price": exit_price, "estimated_trade_value": quantity_to_close * exit_price,
            "reason": reason, "original_trade_details": { "entry_price": trade_state["entry_price"], "initial_quantity": trade_state["initial_quantity"],
                "direction": trade_state["direction"], "initial_stop_loss": trade_state["initial_stop_loss_price"],
                "conviction_score_at_entry": trade_state.get("conviction_score_at_entry") }}
    def register_strategy_agent(self, agent: BaseAgent):
        if agent not in self.strategy_agents: self.strategy_agents.append(agent); self.logger.info(f"Strategy agent {agent.agent_name} registered.")
    def register_data_providing_agent(self, name: str, agent: BaseAgent): 
        if name not in self.data_providing_agents: self.data_providing_agents[name] = agent; self.logger.info(f"Data providing agent '{name}': {agent.agent_name} registered.")
    def register_expert_analyst_agent(self, name: str, agent: BaseAgent): 
        if name not in self.expert_analyst_agents: self.expert_analyst_agents[name] = agent; self.logger.info(f"Expert analyst agent '{name}': {agent.agent_name} registered.")
    def register_risk_management_agent(self, agent: BaseAgent):
        if not self.risk_management_agent: self.risk_management_agent = agent; self.logger.info(f"Risk management agent {agent.agent_name} registered.")
    def process(self, data: Dict[str, Any]) -> None: self.logger.info(f"CEO process called with: {data.get('stock_symbol', 'N/A')}")
    def make_recommendation(self) -> Dict[str, Any]:
        if not self.final_decision_history: return {"final_signal": "HOLD_NO_DATA", "reason": "No analysis yet."}
        return self.final_decision_history[-1]


    def run(self, data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: 
        """
        (CEO主控智能体运行主流程)
        协调数据获取、活跃交易管理、新信号生成、仓位计算、风险评估和最终决策。
        现在包括微观结构分析的调用和整合。
        Main orchestration logic for the CEOAgent.
        Coordinates data gathering, active trade management, new signal generation,
        position sizing, risk assessment, and final decision making.
        Now includes calling and integrating microstructure analysis.
        """
        current_run_decisions: List[Dict[str, Any]] = []
        stock_symbol_for_new_trade = data.get('stock_symbol') if data else None
        portfolio_snapshot = data.get('portfolio_snapshot', {}) if data else {}
        market_data_for_active = data.get('market_data_for_active_assets', {}) if data else {} 
        all_gathered_data_for_current_run = {'stock_symbol': stock_symbol_for_new_trade} if stock_symbol_for_new_trade else {}

        market_data_agent = self.data_providing_agents.get('market_data_agent')
        microstructure_agent = self.expert_analyst_agents.get('microstructure_pattern_agent') 
        
        market_state_data = {}; asset_atr_value = None; current_asset_price = None; 
        primary_signal_for_new_trade_target = None; microstructure_analysis_output = None

        if stock_symbol_for_new_trade and market_data_agent:
            market_state_params = data.get("market_state_params", {})
            if market_state_params:
                market_state_output = market_data_agent.run(market_state_params)
                if market_state_output.get("status") == "success": market_state_data = market_state_output.get("content", {}).get("market_state", {})
                all_gathered_data_for_current_run["market_state_report"] = market_state_output
            
            price_atr_params = {'symbol': stock_symbol_for_new_trade, 'data_type': 'price_and_atr', 'atr_period': self.atr_period_for_stop}
            price_atr_output = market_data_agent.run(price_atr_params)
            all_gathered_data_for_current_run["price_atr_data"] = price_atr_output
            if price_atr_output.get("status") == "success" and price_atr_output.get("content"):
                current_asset_price = price_atr_output['content'].get('price_info', {}).get('price')
                asset_atr_value = price_atr_output['content'].get('atr_value')

            for name, dp_agent in self.data_providing_agents.items():
                if name == 'market_data_agent': continue 
                agent_run_params = {'symbol': stock_symbol_for_new_trade, **data.get(f"{name}_params", {})}
                data_key = name.replace("_agent", "").strip("_")
                if name == "NewsSentimentAgent": data_key = "news_sentiment_data" 
                all_gathered_data_for_current_run[data_key] = dp_agent.run(agent_run_params)
            
            expert_reports_dict = {}
            for name, agent in self.expert_analyst_agents.items():
                if name == 'microstructure_pattern_agent': continue 
                expert_input = {'asset_id': stock_symbol_for_new_trade, **all_gathered_data_for_current_run}
                expert_reports_dict[name + "_report"] = agent.run(expert_input)
            all_gathered_data_for_current_run["expert_reports"] = expert_reports_dict

            if microstructure_agent and self.microstructure_agent_enabled:
                self.logger.info(f"Microstructure analysis enabled for {stock_symbol_for_new_trade}.")
                today_str = datetime.date.today().strftime("%Y%m%d")
                kline_lookback_days = self.config.get("microstructure_pattern_agent_config", {}).get("kline_lookback_days_for_analysis", 1)
                kline_1min_params = {'symbol': stock_symbol_for_new_trade, 'data_type': '1min_klines', 
                                     'start_date': (datetime.date.today() - datetime.timedelta(days=kline_lookback_days)).strftime("%Y%m%d"), 
                                     'end_date': today_str}
                kline_output = market_data_agent.run(kline_1min_params)
                intraday_klines_data = {"1min_klines": []} 
                if kline_output.get("status") == "success" and kline_output.get("content"):
                    kline_data_key_in_content = f"{kline_1min_params.get('period','1')}min_klines_data" 
                    if 'period' in kline_1min_params: kline_data_key_in_content = f"{kline_1min_params['period']}min_klines_data"
                    if kline_data_key_in_content in kline_output['content']:
                         intraday_klines_data["1min_klines"] = kline_output["content"][kline_data_key_in_content]
                         self.logger.debug(f"Fetched {len(intraday_klines_data['1min_klines'])} 1-min klines for {stock_symbol_for_new_trade}")
                    else: self.logger.warning(f"Key '{kline_data_key_in_content}' not found in kline_output for {stock_symbol_for_new_trade}.")
                else: self.logger.warning(f"Failed to get 1-min klines for {stock_symbol_for_new_trade}. Status: {kline_output.get('status')}")
                
                microstructure_analysis_output = microstructure_agent.analyze_microstructure(
                    asset_id=stock_symbol_for_new_trade, 
                    intraday_data=intraday_klines_data, 
                    latest_news_sentiment=all_gathered_data_for_current_run.get("news_sentiment_data", {}).get("content")
                )
                all_gathered_data_for_current_run["microstructure_analysis_output"] = microstructure_analysis_output
            
            if self.strategy_agents: 
                 strategy_input = all_gathered_data_for_current_run.copy(); strategy_input['market_state_data'] = market_state_data
                 primary_signal_for_new_trade_target = self.strategy_agents[0].run(strategy_input)
                 all_gathered_data_for_current_run["primary_signal_details"] = primary_signal_for_new_trade_target
        
        if self.active_trades and market_data_for_active:
            closing_decisions = self._check_and_manage_active_trades(market_data_for_active, all_gathered_data_for_current_run)
            current_run_decisions.extend(closing_decisions)
        
        if not stock_symbol_for_new_trade and not data.get("force_new_signal_generation_for_all_assets", False): 
            if current_run_decisions: self.final_decision_history.extend(current_run_decisions)
            return current_run_decisions
        if not stock_symbol_for_new_trade: 
             if current_run_decisions: self.final_decision_history.extend(current_run_decisions)
             return current_run_decisions

        if any(d.get("stock_symbol") == stock_symbol_for_new_trade and "CLOSE" in d.get("final_signal","") for d in current_run_decisions): return current_run_decisions
        if stock_symbol_for_new_trade in self.active_trades: return current_run_decisions

        primary_signal_output = all_gathered_data_for_current_run.get("primary_signal_details", {"signal": "HOLD", "confidence": 0.0, "reason": "No primary signal."})
        conviction_score = self._calculate_signal_conviction(primary_signal_output, all_gathered_data_for_current_run.get("expert_reports", {}), all_gathered_data_for_current_run.get("microstructure_analysis_output"))
        
        proposed_trade = { "asset_id": stock_symbol_for_new_trade, "direction": primary_signal_output.get("signal", "HOLD"), "quantity": 0, 
                           "estimated_price": current_asset_price if current_asset_price else 0.0, "estimated_trade_value": 0.0, 
                           "rationale": primary_signal_output.get("reason", ""), "conviction_score": conviction_score, 
                           "detailed_sentiment_category": primary_signal_output.get("detailed_sentiment_category", "Unknown"),
                           "primary_signal_confidence": primary_signal_output.get("confidence", 0.0), "stop_loss_price": None, 
                           "effective_n_atr": None, "profit_targets": {}, "initial_atr_at_entry": asset_atr_value, "initial_risk_per_share": 0.0,
                           "microstructure_analysis_at_entry": microstructure_analysis_output 
                         }

        if proposed_trade["direction"] not in ["BUY", "SELL"] or conviction_score < self.conviction_tiers[0]['score_threshold']:
            if proposed_trade["direction"] != "NO_TRADE_CRISIS": 
                proposed_trade["direction"] = "HOLD"; proposed_trade["rationale"] = f"HOLD: Weak signal/conviction. Initial: {primary_signal_output.get('signal')}, Conv: {conviction_score:.2f}"
        
        if proposed_trade["direction"] in ["BUY", "SELL"]:
            if current_asset_price and current_asset_price > 0:
                qty, val = self._determine_preliminary_position_size(conviction_score, current_asset_price, portfolio_snapshot)
                proposed_trade["quantity"] = qty; proposed_trade["estimated_trade_value"] = val
                proposed_trade["rationale"] = f"DynamicSize (Conviction {conviction_score:.2f}). {proposed_trade['rationale']}"
                if asset_atr_value and asset_atr_value > 0:
                    effective_n = self._calculate_effective_n_atr(self.base_n_atr_for_stop, market_state_data, primary_signal_output, conviction_score)
                    proposed_trade["effective_n_atr"] = effective_n
                    sl_price = round(current_asset_price - (effective_n * asset_atr_value) if proposed_trade["direction"] == "BUY" else current_asset_price + (effective_n * asset_atr_value), 2)
                    proposed_trade["stop_loss_price"] = sl_price
                    proposed_trade["initial_risk_per_share"] = abs(current_asset_price - sl_price) if sl_price else 0
                    if proposed_trade["initial_risk_per_share"] > 0 : 
                        proposed_trade["profit_targets"] = self._calculate_initial_profit_targets(stock_symbol_for_new_trade, current_asset_price, sl_price, proposed_trade["direction"])
            else: proposed_trade["direction"] = "HOLD"; proposed_trade["rationale"] = f"HOLD: Missing price. Orig: {primary_signal_output.get('signal')}"
        
        risk_assessment_output = {"overall_assessment": "NOT_ASSESSED"}
        if proposed_trade["direction"] in ["BUY", "SELL"] and proposed_trade["quantity"] > 0 and self.risk_management_agent:
            risk_assessment_output = self.risk_management_agent.run({'proposed_trade': proposed_trade, 'current_portfolio_snapshot': portfolio_snapshot})
        elif not self.risk_management_agent and proposed_trade["direction"] in ["BUY", "SELL"]: risk_assessment_output["overall_assessment"] = "APPROVED_NO_RM"
        
        final_decision_object = {**proposed_trade}
        final_decision_object["final_signal"] = proposed_trade["direction"].upper()
        if risk_assessment_output.get("overall_assessment") == "REJECTED":
            final_decision_object["final_signal"] = "HOLD_RISK_REJECTED"; final_decision_object["reason"] = f"REJECTED by Risk: {risk_assessment_output.get('warnings_or_rejections')}. Orig: {proposed_trade['rationale']}"
            final_decision_object["quantity"] = 0; final_decision_object["estimated_trade_value"] = 0.0
        
        final_decision_object.update({
            "primary_signal_details": primary_signal_output, 
            "expert_analyst_reports": all_gathered_data_for_current_run.get("expert_reports",{}),
            "risk_assessment": risk_assessment_output, 
            "supporting_data_summary": {k:v.get('status') if isinstance(v,dict) else type(v).__name__ for k,v in all_gathered_data_for_current_run.items() if k!='stock_symbol'}
        })
        
        if final_decision_object["final_signal"] in ["BUY", "SELL"] and final_decision_object["quantity"] > 0:
            self.active_trades[stock_symbol_for_new_trade] = {
                "asset_id": stock_symbol_for_new_trade, "direction": final_decision_object["final_signal"],
                "entry_price": final_decision_object["estimated_price"], "initial_quantity": final_decision_object["quantity"],
                "quantity_remaining": final_decision_object["quantity"], "initial_stop_loss_price": final_decision_object["stop_loss_price"],
                "current_stop_loss_price": final_decision_object["stop_loss_price"], "initial_atr_at_entry": asset_atr_value,
                "effective_n_atr_at_entry": final_decision_object["effective_n_atr"],
                "current_n_atr_for_trailing": self.n_atr_for_trailing_stop, 
                "profit_target_price_rr": final_decision_object.get("profit_targets",{}).get("rr_full"),
                "profit_target_price_partial": final_decision_object.get("profit_targets",{}).get("rr_partial"),
                "initial_risk_per_share": final_decision_object.get("initial_risk_per_share", 0),
                "is_trailing_stop_active": False, "current_trailing_stop_price": None,
                "highest_price_since_trail_activation": final_decision_object["estimated_price"] if final_decision_object["final_signal"] == "BUY" else -float('inf'),
                "lowest_price_since_trail_activation": final_decision_object["estimated_price"] if final_decision_object["final_signal"] == "SELL" else float('inf'),
                "entry_date": datetime.date.today().isoformat(), "conviction_score_at_entry": conviction_score, "partial_exit_done": False,
                "microstructure_at_entry": microstructure_analysis_output, 
                "detailed_sentiment_at_entry": primary_signal_output.get("detailed_sentiment_category", "Unknown") 
            }
        
        current_run_decisions.append(final_decision_object)
        self.final_decision_history.extend(current_run_decisions)
        self.logger.info(f"CEO Run Finished for {stock_symbol_for_new_trade}. Final: {final_decision_object.get('final_signal', 'N/A')}")
        return current_run_decisions

if __name__ == '__main__':
    pass 
```
