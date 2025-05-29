# 潜在的导入 autogen 或其他库的占位符
# import autogen
from typing import List, Dict, Any # Added for type hinting

class ShortTermValueFilterAgent:
    """
    ShortTermValueFilterAgent (短期价值筛选器代理) 为逆向交易信号提供快速的估值检查。
    其灵感来源于格雷厄姆和伯里的原则，专注于识别短期错误定价或异常，而非完整的贴现现金流 (DCF) 估值。
    它综合分析市盈率 (P/E)、市净率 (P/B)、股息率，并结合近期新闻背景（可选LLM辅助）来评估资产价值。
    
    ShortTermValueFilterAgent provides a quick valuation check for contrarian trading signals.
    It's inspired by principles from Graham and Burry, focusing on identifying short-term
    mispricings or anomalies rather than full discounted cash flow (DCF) valuations.
    It synthesizes P/E, P/B, dividend yield, and recent news context (optionally LLM-assisted) to assess asset value.
    """

    def __init__(self, config: Dict[str, Any] = None, llm_config: Dict[str, Any] = None):
        """
        初始化 ShortTermValueFilterAgent。
        该构造函数加载用于P/E、P/B、股息率和新闻分析的阈值及评分参数。

        Initializes the ShortTermValueFilterAgent.
        This constructor loads thresholds and scoring parameters for P/E, P/B, dividend yield, and news analysis.

        Args:
            config (dict, optional): 代理逻辑的配置参数。默认为 None。
                                     这些参数包括如 `VALUE_THRESHOLD_PE_DISCOUNT_SECTOR` (P/E相对行业折扣阈值),
                                     `SCORE_PB_LESS_THAN_ONE` (P/B小于1的得分),
                                     `MIN_ACCEPTABLE_DIVIDEND_YIELD` (最低可接受股息率),
                                     以及新闻分析相关的关键词和得分。
                                     Configuration parameters for the agent's logic. Defaults to None.
                                     These include parameters like `VALUE_THRESHOLD_PE_DISCOUNT_SECTOR`, 
                                     `SCORE_PB_LESS_THAN_ONE`, `MIN_ACCEPTABLE_DIVIDEND_YIELD`,
                                     and news analysis related keywords and scores.
            llm_config (dict, optional): 如果代理使用大型语言模型 (LLM) 进行细致分析（例如新闻背景），则为 LLM 的配置。默认为 None。
                                       Configuration for an LLM if the agent uses one
                                       for nuanced analysis (e.g., news context). Defaults to None.
        """
        self.config = config if config else {}
        self.llm_config = llm_config

        # Load P/E thresholds
        self.pe_discount_sector = self.config.get("VALUE_THRESHOLD_PE_DISCOUNT_SECTOR", 0.3)
        self.pe_premium_sector = self.config.get("VALUE_THRESHOLD_PE_PREMIUM_SECTOR", 0.3)
        self.pe_positive_score = self.config.get("SCORE_PE_DISCOUNT", 0.3)
        self.pe_negative_score = self.config.get("SCORE_PE_PREMIUM", -0.2)

        # Load P/B thresholds
        self.pb_discount_sector = self.config.get("VALUE_THRESHOLD_PB_DISCOUNT_SECTOR", 0.3)
        self.pb_less_than_one_score = self.config.get("SCORE_PB_LESS_THAN_ONE", 0.3)
        self.pb_sector_discount_score = self.config.get("SCORE_PB_SECTOR_DISCOUNT", 0.2)
        
        # Load Dividend Yield thresholds
        self.min_dividend_yield = self.config.get("MIN_ACCEPTABLE_DIVIDEND_YIELD", 0.03) # e.g. 3%
        self.dividend_yield_score = self.config.get("SCORE_DIVIDEND_YIELD", 0.1)

        # News contextualization scores
        self.news_overreaction_score = self.config.get("SCORE_NEWS_OVERREACTION", 0.3)
        self.news_fundamental_impairment_score = self.config.get("SCORE_NEWS_FUNDAMENTAL_IMPAIRMENT", -0.5)
        self.news_heuristic_overreaction_score = self.config.get("SCORE_NEWS_HEURISTIC_OVERREACTION", 0.1)
        self.news_heuristic_severe_score = self.config.get("SCORE_NEWS_HEURISTIC_SEVERE", -0.3)
        self.severe_news_keywords = self.config.get("SEVERE_NEWS_KEYWORDS", ["fraud", "bankruptcy", "delisting", "investigation"])


    def _analyze_pe(self, pe_ratio: float, sector_average_pe_ratio: float, contributing_factors: list) -> float:
        """
        分析市盈率 (P/E) 并返回其对总体价值评分的贡献。
        将其与行业平均P/E进行比较，考虑配置的折扣和溢价阈值。

        Analyzes P/E ratio and returns its contribution to the overall value score.
        Compares it against sector average P/E, considering configured discount and premium thresholds.

        Args:
            pe_ratio (float): 资产的当前市盈率。
                              Current P/E ratio of the asset.
            sector_average_pe_ratio (float): 资产所在行业的平均市盈率。
                                           Average P/E ratio of the asset's sector.
            contributing_factors (list): 用于记录分析细节的列表。
                                         List to record analysis details.

        Returns:
            float: P/E分析对价值评分的贡献值。
                   The score contribution from P/E analysis.
        """
        score_contribution = 0.0
        if pe_ratio is not None and sector_average_pe_ratio is not None and sector_average_pe_ratio > 0: # Avoid division by zero or meaningless comparison
            if pe_ratio < sector_average_pe_ratio * (1 - self.pe_discount_sector):
                score_contribution += self.pe_positive_score
                contributing_factors.append({
                    'factor': 'P/E vs Sector',
                    'assessment': 'Positive',
                    'details': f'P/E ({pe_ratio:.2f}) is significantly below sector average ({sector_average_pe_ratio:.2f} * (1-{self.pe_discount_sector:.2f}) = {sector_average_pe_ratio * (1 - self.pe_discount_sector):.2f}).'
                })
            elif pe_ratio > sector_average_pe_ratio * (1 + self.pe_premium_sector):
                score_contribution += self.pe_negative_score
                contributing_factors.append({
                    'factor': 'P/E vs Sector',
                    'assessment': 'Negative',
                    'details': f'P/E ({pe_ratio:.2f}) is significantly above sector average ({sector_average_pe_ratio:.2f} * (1+{self.pe_premium_sector:.2f}) = {sector_average_pe_ratio * (1 + self.pe_premium_sector):.2f}).'
                })
        return score_contribution

    def _analyze_pb(self, pb_ratio: float, sector_average_pb_ratio: float, contributing_factors: list) -> float:
        """
        分析市净率 (P/B) 并返回其对总体价值评分的贡献。
        检查P/B是否小于1，并将其与行业平均P/B进行比较。

        Analyzes P/B ratio and returns its contribution to the overall value score.
        Checks if P/B is less than 1 and compares it against sector average P/B.

        Args:
            pb_ratio (float): 资产的当前市净率。
                              Current P/B ratio of the asset.
            sector_average_pb_ratio (float): 资产所在行业的平均市净率。
                                           Average P/B ratio of the asset's sector.
            contributing_factors (list): 用于记录分析细节的列表。
                                         List to record analysis details.

        Returns:
            float: P/B分析对价值评分的贡献值。
                   The score contribution from P/B analysis.
        """
        score_contribution = 0.0
        pb_less_than_one_triggered = False
        if pb_ratio is not None:
            if pb_ratio < 1.0:
                score_contribution += self.pb_less_than_one_score
                contributing_factors.append({
                    'factor': 'P/B Ratio',
                    'assessment': 'Strongly Positive',
                    'details': f'P/B ratio ({pb_ratio:.2f}) is less than 1.0.'
                })
                pb_less_than_one_triggered = True
            
            if sector_average_pb_ratio is not None and sector_average_pb_ratio > 0: # Avoid division by zero or meaningless comparison
                 # Only apply sector discount if P/B < 1 didn't already give a stronger signal, or if P/B > 1
                if not pb_less_than_one_triggered or pb_ratio >= 1.0:
                    if pb_ratio < sector_average_pb_ratio * (1 - self.pb_discount_sector):
                        score_contribution += self.pb_sector_discount_score
                        contributing_factors.append({
                            'factor': 'P/B vs Sector',
                            'assessment': 'Positive',
                            'details': f'P/B ratio ({pb_ratio:.2f}) is significantly below sector average ({sector_average_pb_ratio:.2f} * (1-{self.pb_discount_sector:.2f}) = {sector_average_pb_ratio * (1 - self.pb_discount_sector):.2f}).'
                        })
        return score_contribution

    def _analyze_dividend(self, dividend_yield: float, contributing_factors: list) -> float:
        """
        分析股息率并返回其对总体价值评分的贡献。
        如果股息率高于配置的最低可接受水平，则增加正面评分。

        Analyzes dividend yield and returns its contribution to the overall value score.
        Adds a positive score if the dividend yield is above a configured minimum acceptable level.

        Args:
            dividend_yield (float): 资产的当前股息率。
                                    Current dividend yield of the asset.
            contributing_factors (list): 用于记录分析细节的列表。
                                         List to record analysis details.

        Returns:
            float: 股息率分析对价值评分的贡献值。
                   The score contribution from dividend yield analysis.
        """
        score_contribution = 0.0
        if dividend_yield is not None and dividend_yield > self.min_dividend_yield:
            score_contribution += self.dividend_yield_score
            contributing_factors.append({
                'factor': 'Dividend Yield',
                'assessment': 'Positive',
                'details': f'Dividend yield ({dividend_yield:.2%}) is above minimum acceptable threshold ({self.min_dividend_yield:.2%}).'
            })
        return score_contribution

    def _analyze_news_context(self, asset_id: str, news_summary: str, contributing_factors: list) -> (float, bool):
        """
        分析新闻背景，返回其对总体价值评分的贡献和一个价值陷阱标志。
        如果配置了LLM，则使用LLM评估新闻；否则，使用基于关键词的启发式方法。

        Analyzes news context, returning its contribution to the overall value score and a value trap flag.
        Uses LLM for news assessment if configured; otherwise, uses a keyword-based heuristic.

        Args:
            asset_id (str): 资产ID。Asset ID.
            news_summary (str): 近期新闻摘要。Summary of recent news.
            contributing_factors (list): 用于记录分析细节的列表。List to record analysis details.

        Returns:
            tuple: 包含新闻分析的评分贡献值 (float) 和价值陷阱标志 (bool) 的元组。
                   A tuple containing the score contribution (float) from news analysis and a value trap flag (bool).
        """
        score_contribution = 0.0
        value_trap_suspected = False

        if not news_summary:
            return score_contribution, value_trap_suspected

        if self.llm_config:
            # Placeholder for LLM call
            # prompt = f"News: '{news_summary}' for {asset_id}. Does this typically cause temporary market overreaction or indicate fundamental long-term value impairment for short-term trading? Respond 'Overreaction', 'Fundamental Impairment', or 'Uncertain'."
            # llm_response = "Overreaction" # Dummy LLM response for now
            llm_response = "Uncertain" # Safer default for placeholder

            if llm_response == "Overreaction":
                score_contribution += self.news_overreaction_score
                contributing_factors.append({
                    'factor': 'News Context (LLM)',
                    'assessment': 'Positive',
                    'details': f'LLM assessed news ("{news_summary[:50]}...") as likely temporary overreaction.'
                })
            elif llm_response == "Fundamental Impairment":
                score_contribution += self.news_fundamental_impairment_score
                value_trap_suspected = True
                contributing_factors.append({
                    'factor': 'News Context (LLM)',
                    'assessment': 'Strongly Negative',
                    'details': f'LLM assessed news ("{news_summary[:50]}...") as potential fundamental impairment.'
                })
            else: # Uncertain or other response
                 contributing_factors.append({
                    'factor': 'News Context (LLM)',
                    'assessment': 'Neutral',
                    'details': f'LLM assessment of news ("{news_summary[:50]}...") was uncertain.'
                })
        else:
            # Heuristic if no LLM
            news_summary_lower = news_summary.lower()
            is_severe = any(keyword in news_summary_lower for keyword in self.severe_news_keywords)
            
            if is_severe:
                score_contribution += self.news_heuristic_severe_score
                value_trap_suspected = True
                contributing_factors.append({
                    'factor': 'News Context (Heuristic)',
                    'assessment': 'Negative',
                    'details': f'News summary ("{news_summary[:50]}...") contains keywords indicating potential severity.'
                })
            else: # Assume non-critical if no severe keywords (weak heuristic)
                score_contribution += self.news_heuristic_overreaction_score
                contributing_factors.append({
                    'factor': 'News Context (Heuristic)',
                    'assessment': 'Mildly Positive',
                    'details': f'News summary ("{news_summary[:50]}...") assessed by heuristic as potential minor overreaction.'
                })
        return score_contribution, value_trap_suspected

    def assess_short_term_value(self, request_data: dict) -> dict:
        """
        根据提供的数据评估资产的短期价值。
        此方法综合P/E、P/B、股息率和新闻背景分析的结果，计算数值评分并生成最终的价值评估。

        Assesses the short-term value of an asset based on provided data.
        This method synthesizes results from P/E, P/B, dividend yield, and news context analyses 
        to calculate a numeric score and generate a final value assessment.

        Args:
            request_data (dict): 包含评估所需数据的字典，
                                 A dictionary containing the data needed for assessment,
                                 including:
                                 - 'asset_id' (str): 资产ID
                                 - 'current_market_data' (dict): {'price', 'pe_ratio'?, 'pb_ratio'?, 'dividend_yield'? } 当前市场数据，包含价格、市盈率、市净率、股息率等
                                 - 'sector_financial_metrics' (dict, optional): {'average_pe_ratio'?, 'average_pb_ratio'? } 行业财务指标（可选），包含平均市盈率、平均市净率等
                                 - 'recent_news_summary' (str, optional): 主要负面新闻摘要（可选）

        Returns:
            dict: 包含价值评估的字典，
                  A dictionary containing the value assessment, for example:
                  {
                      'asset_id': str, # 资产ID
                      'value_assessment': str ("Strongly Undervalued", "Undervalued", # 价值评估结果，例如：“严重低估”，“低估”
                                               "Fairly Valued", "Overvalued",          # “合理估值”，“高估”
                                               "Strongly Overvalued", "Value Trap Suspected"), # “严重高估”，“疑似价值陷阱”
                      'value_score': float (optional, e.g., -1.0 to 1.0), # 价值评分（可选），例如 -1.0 到 1.0
                      'rationale': str (brief explanation), # 理由（简要说明）
                      'contributing_factors': list (optional, factors that led to the assessment) # 促成评估的因素（可选）
                  }
        """
        asset_id = request_data.get('asset_id', 'Unknown')
        current_market_data = request_data.get('current_market_data', {})
        sector_financial_metrics = request_data.get('sector_financial_metrics', {})
        recent_news_summary = request_data.get('recent_news_summary')

        pe_ratio = current_market_data.get('pe_ratio')
        pb_ratio = current_market_data.get('pb_ratio')
        dividend_yield = current_market_data.get('dividend_yield')
        
        sector_average_pe_ratio = sector_financial_metrics.get('average_pe_ratio')
        sector_average_pb_ratio = sector_financial_metrics.get('average_pb_ratio')

        numeric_score = 0.0
        contributing_factors = []
        value_trap_suspected = False
        
        # P/E Analysis
        numeric_score += self._analyze_pe(pe_ratio, sector_average_pe_ratio, contributing_factors)
        
        # P/B Analysis
        numeric_score += self._analyze_pb(pb_ratio, sector_average_pb_ratio, contributing_factors)

        # Dividend Yield Analysis
        numeric_score += self._analyze_dividend(dividend_yield, contributing_factors)
        
        # News Contextualization
        if recent_news_summary:
            news_score, news_value_trap = self._analyze_news_context(asset_id, recent_news_summary, contributing_factors)
            numeric_score += news_score
            if news_value_trap:
                value_trap_suspected = True
        
        # Determine final assessment string
        value_assessment_str = "Fairly Valued"
        if value_trap_suspected:
            value_assessment_str = "Value Trap Suspected"
        elif numeric_score > 0.5: # Thresholds for assessment categories
            value_assessment_str = "Strongly Undervalued"
        elif numeric_score > 0.2:
            value_assessment_str = "Undervalued"
        elif numeric_score < -0.5:
            value_assessment_str = "Strongly Overvalued"
        elif numeric_score < -0.2:
            value_assessment_str = "Overvalued"

        # Build rationale
        rationale_parts = [f["details"] for f in contributing_factors]
        rationale = f"Overall score: {numeric_score:.2f}. Assessment: {value_assessment_str}. "
        if rationale_parts:
            rationale += "Key factors: " + "; ".join(rationale_parts)
        else:
            rationale += "No strong valuation signals detected based on available data."
        if value_trap_suspected:
            rationale += " Caution: Potential value trap indicated by news analysis."

        return {
            'asset_id': asset_id,
            'value_assessment': value_assessment_str,
            'value_score': round(numeric_score, 3),
            'rationale': rationale,
            'contributing_factors': contributing_factors
        }

# 示例用法（用于测试或作为脚本运行时，并非 autogen 代理的典型用法）
# Example usage (for testing or if run as a script, not typical for autogen agents)
if __name__ == '__main__':
    # 本节用于直接测试代理的方法，
    # This section would be for direct testing of the agent's methods,
    # 而非其在 autogen 框架内的操作。
    # not for its operation within the autogen framework.
    agent_config_test = {
        "VALUE_THRESHOLD_PE_DISCOUNT_SECTOR": 0.3, # PE significantly below sector
        "VALUE_THRESHOLD_PE_PREMIUM_SECTOR": 0.3,  # PE significantly above sector
        "SCORE_PE_DISCOUNT": 0.3,
        "SCORE_PE_PREMIUM": -0.2,
        "VALUE_THRESHOLD_PB_DISCOUNT_SECTOR": 0.3, # PB significantly below sector
        "SCORE_PB_LESS_THAN_ONE": 0.3,
        "SCORE_PB_SECTOR_DISCOUNT": 0.2,
        "MIN_ACCEPTABLE_DIVIDEND_YIELD": 0.03,     # Min 3% dividend yield to be positive
        "SCORE_DIVIDEND_YIELD": 0.1,
        "SCORE_NEWS_OVERREACTION": 0.3,            # LLM based news assessment
        "SCORE_NEWS_FUNDAMENTAL_IMPAIRMENT": -0.5, # LLM based news assessment
        "SCORE_NEWS_HEURISTIC_OVERREACTION": 0.1,  # Heuristic news assessment
        "SCORE_NEWS_HEURISTIC_SEVERE": -0.3,       # Heuristic news assessment
        "SEVERE_NEWS_KEYWORDS": ["fraud", "bankruptcy", "delisting", "investigation", "scandal", "halted"]
    }
    # Test case 1: Undervalued
    value_agent_test = ShortTermValueFilterAgent(config=agent_config_test, llm_config=None) # Test heuristic news
    sample_request_undervalued = {
        'asset_id': 'STOCK_UNDER',
        'current_market_data': {'pe_ratio': 8.0, 'pb_ratio': 0.8, 'dividend_yield': 0.04},
        'sector_financial_metrics': {'average_pe_ratio': 15.0, 'average_pb_ratio': 1.5},
        'recent_news_summary': "Market sentiment slightly down, but company fundamentals remain solid."
    }
    assessment1 = value_agent_test.assess_short_term_value(sample_request_undervalued)
    print("--- Assessment 1 (Undervalued Heuristic) ---")
    for key, value in assessment1.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")

    # Test case 2: Overvalued
    sample_request_overvalued = {
        'asset_id': 'STOCK_OVER',
        'current_market_data': {'pe_ratio': 30.0, 'pb_ratio': 2.5, 'dividend_yield': 0.01},
        'sector_financial_metrics': {'average_pe_ratio': 18.0, 'average_pb_ratio': 1.8},
        'recent_news_summary': "Stock price surged on speculative buying."
    }
    assessment2 = value_agent_test.assess_short_term_value(sample_request_overvalued)
    print("\n--- Assessment 2 (Overvalued Heuristic) ---")
    for key, value in assessment2.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")

    # Test case 3: Value Trap (with LLM placeholder for fundamental impairment)
    # To simulate LLM, we would need to mock the LLM call or adjust _analyze_news_context
    # For simplicity, let's use a severe news heuristic
    sample_request_trap = {
        'asset_id': 'STOCK_TRAP',
        'current_market_data': {'pe_ratio': 5.0, 'pb_ratio': 0.5}, # Looks cheap
        'sector_financial_metrics': {'average_pe_ratio': 15.0, 'average_pb_ratio': 1.5},
        'recent_news_summary': "Company under investigation for accounting fraud."
    }
    assessment3 = value_agent_test.assess_short_term_value(sample_request_trap)
    print("\n--- Assessment 3 (Value Trap Heuristic) ---")
    for key, value in assessment3.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")

    # Test case 4: Fairly Valued
    sample_request_fair = {
        'asset_id': 'STOCK_FAIR',
        'current_market_data': {'pe_ratio': 14.0, 'pb_ratio': 1.4, 'dividend_yield': 0.02},
        'sector_financial_metrics': {'average_pe_ratio': 15.0, 'average_pb_ratio': 1.5}
        # No news
    }
    assessment4 = value_agent_test.assess_short_term_value(sample_request_fair)
    print("\n--- Assessment 4 (Fairly Valued) ---")
    for key, value in assessment4.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")

    # Test case 5: LLM "Overreaction" (Simulated - requires modifying _analyze_news_context or mocking)
    # This would require temporarily changing the llm_response in _analyze_news_context
    # or setting up a mock. For this execution, it will use the heuristic.
    # To truly test LLM path, one would need to set llm_config and mock the call.
    value_agent_llm_test = ShortTermValueFilterAgent(config=agent_config_test, llm_config={"api_key": "dummy_key"}) # Enable LLM path
    
    # To simulate LLM response "Overreaction", we'd typically mock the LLM call.
    # For this self-contained script, we'll just note it will use heuristic due to no actual LLM call.
    # If _analyze_news_context is modified to force "Overreaction" for testing:
    # numeric_score += self.news_overreaction_score (0.3)
    sample_request_llm_overreaction = {
        'asset_id': 'STOCK_LLM_OVER',
        'current_market_data': {'pe_ratio': 10.0, 'pb_ratio': 1.0, 'dividend_yield': 0.025},
        'sector_financial_metrics': {'average_pe_ratio': 15.0, 'average_pb_ratio': 1.5},
        'recent_news_summary': "Analyst downgrades stock due to short-term supply chain issues, stock plunges."
    }
    # To properly test LLM "Overreaction", the LLM path in _analyze_news_context needs to be triggered
    # and the LLM response mocked or forced.
    # For now, this will fall back to heuristic.
    print("\n--- Assessment 5 (LLM Overreaction - Simulated via Heuristic due to no live LLM) ---")
    assessment5 = value_agent_llm_test.assess_short_term_value(sample_request_llm_overreaction)
    for key, value in assessment5.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")

    # Test case 6: LLM "Fundamental Impairment" (Simulated - requires modifying _analyze_news_context or mocking)
    # If _analyze_news_context is modified to force "Fundamental Impairment" for testing:
    # numeric_score += self.news_fundamental_impairment_score (-0.5)
    # value_trap_suspected = True
    print("\n--- Assessment 6 (LLM Fundamental Impairment - Simulated via Heuristic due to no live LLM) ---")
    sample_request_llm_impairment = {
        'asset_id': 'STOCK_LLM_IMP',
        'current_market_data': {'pe_ratio': 7.0, 'pb_ratio': 0.7}, # Looks cheap
        'sector_financial_metrics': {'average_pe_ratio': 15.0, 'average_pb_ratio': 1.5},
        'recent_news_summary': "Company announces major product recall and faces class action lawsuit." # Should be caught by heuristic too
    }
    assessment6 = value_agent_llm_test.assess_short_term_value(sample_request_llm_impairment)
    for key, value in assessment6.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")

    # Test case 7: PE significantly above sector
    sample_request_high_pe = {
        'asset_id': 'STOCK_HIGH_PE',
        'current_market_data': {'pe_ratio': 100.0, 'pb_ratio': 5.0, 'dividend_yield': 0.001},
        'sector_financial_metrics': {'average_pe_ratio': 20.0, 'average_pb_ratio': 2.0},
    }
    assessment7 = value_agent_test.assess_short_term_value(sample_request_high_pe)
    print("\n--- Assessment 7 (High PE vs Sector) ---")
    for key, value in assessment7.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")
            
    # Test case 8: Only P/B < 1, no other strong signals
    sample_request_pb_only = {
        'asset_id': 'STOCK_PB_ONLY',
        'current_market_data': {'pb_ratio': 0.7},
        'sector_financial_metrics': {'average_pb_ratio': 0.6}, # Sector PB also low, so no sector discount
    }
    assessment8 = value_agent_test.assess_short_term_value(sample_request_pb_only)
    print("\n--- Assessment 8 (P/B < 1, no sector discount) ---")
    for key, value in assessment8.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")
            
    # Test case 9: P/B > 1 but below sector average
    sample_request_pb_sector_discount = {
        'asset_id': 'STOCK_PB_SECTOR',
        'current_market_data': {'pb_ratio': 1.2},
        'sector_financial_metrics': {'average_pb_ratio': 2.0}, 
    }
    assessment9 = value_agent_test.assess_short_term_value(sample_request_pb_sector_discount)
    print("\n--- Assessment 9 (P/B > 1 but below sector average) ---")
    for key, value in assessment9.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")
            
    # Test case 10: Data missing for some metrics
    sample_request_missing_data = {
        'asset_id': 'STOCK_MISSING',
        'current_market_data': {'pb_ratio': 1.2}, # Only PB available
        # No sector metrics
    }
    assessment10 = value_agent_test.assess_short_term_value(sample_request_missing_data)
    print("\n--- Assessment 10 (Missing some data) ---")
    for key, value in assessment10.items():
        if key == 'contributing_factors':
            print(f"  {key}:")
            for item_val in value: print(f"    - {item_val}")
        else:
            print(f"  {key}: {value}")
