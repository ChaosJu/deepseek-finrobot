# 潜在的导入
import autogen # type: ignore
from typing import List, Dict, Any, Tuple # Added Tuple
import datetime
import akshare # Ensure akshare is installed

# Placeholder for an LLM client if used
# from ...openai_adapter import get_chat_completion # Assuming an adapter exists

class CatalystScoutAgent:
    """
    CatalystScoutAgent (催化剂侦察代理) 负责识别可能影响资产价格的近期潜在催化剂。
    它会扫描数据源，寻找与资产或其行业相关的即将发生的事件/新闻。
    其灵感来源于彼得·林奇和菲利普·费雪的投资理念。

    CatalystScoutAgent identifies near-term potential catalysts that could impact an asset's price.
    It scans data sources for upcoming events/news related to the asset or its sector.
    Inspired by Peter Lynch and Phil Fisher.
    """

    def __init__(self, config: Dict[str, Any] = None, llm_config: Dict[str, Any] = None):
        """
        初始化 CatalystScoutAgent。
        加载配置参数，例如预看天数、公告扫描历史天数、默认催化剂类型和关键词。

        Initializes the CatalystScoutAgent.
        Loads configuration parameters such as look-ahead days, announcement scan history days, 
        default catalyst types, and keywords.

        Args:
            config (dict, optional): 代理逻辑的配置参数。默认为 None。
                                     Configuration parameters for the agent's logic. Defaults to None.
            llm_config (dict, optional): 如果代理使用大型语言模型 (LLM) 对公告或新闻进行分类，则为 LLM 的配置。默认为 None。
                                       Configuration for an LLM if the agent uses one
                                       for classifying announcements or news. Defaults to None.
        """
        self.config = config if config else {}
        self.llm_config = llm_config
        
        self.default_look_ahead_days = self.config.get("DEFAULT_LOOK_AHEAD_DAYS", 30)
        self.announcement_scan_past_days = self.config.get("ANNOUNCEMENT_SCAN_PAST_DAYS", 7)
        self.default_catalyst_types = self.config.get("DEFAULT_CATALYST_TYPES", 
            ["earnings_release", "product_launch", "regulatory_approval", "major_contract", "partnership", "acquisition", "restructuring", "management_change", "analyst_rating_change"])
        self.announcement_keywords = self.config.get("ANNOUNCEMENT_KEYWORDS", {
            "product_launch": ["推出", "发布", "新产品", "上市"],
            "regulatory_approval": ["批准", "获批", "许可", "认证"],
            "major_contract": ["合同", "订单", "中标", "协议"],
            "partnership": ["合作", "合资", "联盟"],
            "acquisition": ["收购", "并购", "购买资产"],
            "restructuring": ["重组", "资产重组"],
            "management_change": ["辞职", "任命", "变更", "新任", "离任"]
        })
        # Example: self.look_ahead_days = self.config.get("DEFAULT_LOOK_AHEAD_DAYS", 30)
        # 从配置中初始化其他参数
        # Initialize other parameters from config

    def _fetch_earnings_catalysts_akshare(self, asset_id: str, look_ahead_days: int) -> List[Dict[str, Any]]:
        """
        (私有方法) 使用 Akshare 获取与财报相关的催化剂。
        包括业绩预告和预约披露时间。

        (Private method) Helper to fetch earnings related catalysts using Akshare.
        Includes earnings guidance and scheduled disclosure dates.

        Args:
            asset_id (str): 资产ID (例如 '000001.SZ')。Asset ID (e.g., '000001.SZ').
            look_ahead_days (int): 向未来看多少天以寻找催化剂。Number of days to look ahead for catalysts.

        Returns:
            List[Dict[str, Any]]: 找到的财报相关催化剂列表。
                                  A list of found earnings-related catalysts.
        """
        catalysts = []
        today = datetime.date.today()
        limit_date = today + datetime.timedelta(days=look_ahead_days)
        
        # Normalize asset_id for akshare (e.g., 000001.SZ -> 000001)
        stock_code = asset_id.split('.')[0]

        try:
            # 业绩预告 (Earnings Guidance)
            yjyg_df = akshare.stock_yjyg_em(symbol=stock_code)
            if not yjyg_df.empty:
                for _, row in yjyg_df.iterrows():
                    forecast_date_str = str(row.get("预告日期", "")) # Assuming '预告日期' is the announcement date
                    if forecast_date_str:
                        try:
                            event_date = datetime.datetime.strptime(forecast_date_str.split("T")[0], "%Y-%m-%d").date()
                            if today <= event_date <= limit_date: # Check if this is future-looking or recent past
                                description = f"业绩预告: {row.get('业绩预告标题', '')} - 类型: {row.get('预告类型', 'N/A')}"
                                impact = "Positive" if "增" in str(row.get('预告类型')) or "预增" in str(row.get('预告类型')) else \
                                         "Negative" if "减" in str(row.get('预告类型')) or "预亏" in str(row.get('预告类型')) else "Uncertain"
                                catalysts.append({
                                    'catalyst_type': "EarningsGuidance",
                                    'description': description,
                                    'date_event': event_date.isoformat(),
                                    'date_identified': today.isoformat(),
                                    'source_type': "CompanyFiling_Akshare_YJYG",
                                    'potential_impact_assessment': impact,
                                    'source_reference': f"Akshare stock_yjyg_em for {stock_code}"
                                })
                        except ValueError:
                            pass # Date parsing error
                            
            # 预约披露时间 (Scheduled Disclosure Dates)
            yysj_df = akshare.stock_yysj_em(stock=stock_code) # May need adjustment based on actual function parameters
            if not yysj_df.empty and '预约披露日期' in yysj_df.columns:
                for _, row in yysj_df.iterrows():
                    disclosure_date_str = str(row.get("预约披露日期", ""))
                    if disclosure_date_str:
                        try:
                            event_date = datetime.datetime.strptime(disclosure_date_str.split("T")[0], "%Y-%m-%d").date()
                            if today <= event_date <= limit_date:
                                catalysts.append({
                                    'catalyst_type': "EarningsReleaseScheduled",
                                    'description': f"预约披露时间: {row.get('定期报告类型', 'N/A')}",
                                    'date_event': event_date.isoformat(),
                                    'date_identified': today.isoformat(),
                                    'source_type': "FinancialCalendar_Akshare_YYSJ",
                                    'potential_impact_assessment': "Uncertain",
                                    'source_reference': f"Akshare stock_yysj_em for {stock_code}"
                                })
                        except ValueError:
                            pass # Date parsing error
        except Exception as e:
            print(f"Error fetching earnings catalysts from Akshare for {asset_id}: {e}")
        return catalysts

    def _analyze_announcement_keywords(self, title: str, summary: str, catalyst_types_of_interest: List[str]) -> Tuple[str | None, str]: # Updated return type
        """
        (私有方法) 对公告内容进行基于关键词的分析。
        根据预定义的关键词匹配来识别催化剂类型和初步评估影响。

        (Private method) Keyword-based analysis of announcement content.
        Identifies catalyst type and preliminary impact based on pre-defined keyword matching.

        Args:
            title (str): 公告标题。Announcement title.
            summary (str): 公告摘要或部分内容。Announcement summary or partial content.
            catalyst_types_of_interest (List[str]): 感兴趣的催化剂类型列表。List of catalyst types of interest.

        Returns:
            Tuple[str | None, str]: 包含识别的催化剂类型 (如果找到) 和评估的影响的元组。
                                    A tuple containing the identified catalyst type (if any) and assessed impact.
        """
        for catalyst_type, keywords in self.announcement_keywords.items():
            if catalyst_type in catalyst_types_of_interest or "all" in catalyst_types_of_interest:
                for keyword in keywords:
                    if keyword in title or (summary and keyword in summary):
                        # Simple impact assessment based on type (can be refined)
                        impact = "Uncertain"
                        if catalyst_type in ["major_contract", "regulatory_approval", "product_launch"]:
                            impact = "Positive"
                        elif catalyst_type in ["restructuring"] and ("失败" in title or "终止" in title):
                             impact = "Negative"
                        return catalyst_type, impact
        return None, "Uncertain"


    def _analyze_announcement_content(self, asset_id:str, title: str, summary: str, content_url: str, 
                                      look_ahead_days: int, catalyst_types_of_interest: List[str]) -> Dict[str, Any] | None: # Updated return type
        """
        (私有方法) 使用 LLM 或关键词分析公告内容。
        如果配置了 LLM，则优先使用 LLM；否则回退到关键词分析。

        (Private method) Analyzes announcement content using LLM or keywords.
        Prioritizes LLM if configured; otherwise, falls back to keyword analysis.

        Args:
            asset_id (str): 资产ID。Asset ID.
            title (str): 公告标题。Announcement title.
            summary (str): 公告摘要或部分内容。Announcement summary or partial content.
            content_url (str): 公告内容的URL (用于参考)。URL of the announcement content (for reference).
            look_ahead_days (int): 向未来看多少天。Number of days to look ahead.
            catalyst_types_of_interest (List[str]): 感兴趣的催化剂类型列表。List of catalyst types of interest.
            
        Returns:
            Dict[str, Any] | None: 如果识别出催化剂，则返回包含催化剂信息的字典；否则返回 None。
                                   A dictionary with catalyst information if identified, otherwise None.
        """
        if self.llm_config:
            # Placeholder for LLM call
            # prompt = f"Announcement Title: '{title}', Text Snippet: '{summary[:200]}...'. Does this indicate an upcoming catalyst of types {catalyst_types_of_interest} for {asset_id} within {look_ahead_days} days? If yes, identify catalyst type, summarize, estimate event date if possible, and assess potential impact (Positive/Negative/Uncertain). Respond in JSON format with fields: 'catalyst_type', 'description', 'date_event', 'potential_impact_assessment'."
            # llm_response_str = get_chat_completion(prompt, model=self.llm_config.get("model"), ...) 
            # try:
            #     llm_parsed = json.loads(llm_response_str)
            #     if llm_parsed.get('catalyst_type'): # LLM identified a catalyst
            #          return {
            #             'catalyst_type': llm_parsed['catalyst_type'],
            #             'description': llm_parsed.get('description', title),
            #             'date_event': llm_parsed.get('date_event', "Unknown"), # LLM might estimate this
            #             'potential_impact_assessment': llm_parsed.get('potential_impact_assessment', "Uncertain")
            #         }
            # except: pass # LLM response parsing error or no catalyst identified
            # Fallback to keywords if LLM fails or doesn't identify
            catalyst_type_found, impact = self._analyze_announcement_keywords(title, summary, catalyst_types_of_interest)
            if catalyst_type_found:
                return {
                    'catalyst_type': catalyst_type_found,
                    'description': title, # Use title as description for keyword-based
                    'date_event': "Unknown", # Hard to determine from just keywords
                    'potential_impact_assessment': impact
                }
        else: # Keyword-based only
            catalyst_type_found, impact = self._analyze_announcement_keywords(title, summary, catalyst_types_of_interest)
            if catalyst_type_found:
                return {
                    'catalyst_type': catalyst_type_found,
                    'description': title,
                    'date_event': "Unknown",
                    'potential_impact_assessment': impact
                }
        return None # No catalyst identified by any method

    def _scan_company_announcements_akshare(self, asset_id: str, look_ahead_days: int, catalyst_types_of_interest: List[str]) -> List[Dict[str, Any]]:
        """
        (私有方法) 从 Akshare 扫描近期的公司公告。
        获取指定公司近期的公告，并使用 `_analyze_announcement_content` 方法分析它们是否包含潜在催化剂。

        (Private method) Scans recent company announcements from Akshare.
        Fetches recent announcements for the given company and analyzes them for potential catalysts 
        using the `_analyze_announcement_content` method.

        Args:
            asset_id (str): 资产ID。Asset ID.
            look_ahead_days (int): 向未来看多少天 (主要用于 `_analyze_announcement_content` 的上下文)。
                                   Number of days to look ahead (primarily for context in `_analyze_announcement_content`).
            catalyst_types_of_interest (List[str]): 感兴趣的催化剂类型列表。List of catalyst types of interest.

        Returns:
            List[Dict[str, Any]]: 从公司公告中识别出的催化剂列表。
                                  A list of catalysts identified from company announcements.
        """
        catalysts = []
        today = datetime.date.today()
        stock_code = asset_id.split('.')[0]
        
        # Use '巨潮资讯网-公司公告-最新公告' as an example.
        # This might need adjustment based on available functions and desired sources.
        try:
            # Fetch recent announcements (e.g., past N days specified by self.announcement_scan_past_days)
            # The function stock_notice_report might not be ideal as it requires specific dates.
            # Let's assume we have a way to get announcements for a period or recent ones.
            # For now, using a placeholder for fetching recent announcements (e.g., stock_ggcg_em - 股本股东, stock_gsrl_gsgg_em - 公司公告)
            # This part highly depends on what akshare function is most suitable for generic announcements.
            # stock_gsrl_gsgg_em seems promising for general company announcements.
            # Example: df_announcements = akshare.stock_gsrl_gsgg_em(symbol=stock_code, period="季度")
            # Let's simulate fetching announcements for the past 'announcement_scan_past_days'
            
            # Simulating a fetch of recent announcements, actual akshare function might differ
            # For this example, let's use stock_notice_report for a specific date range
            # to simulate fetching recent news.
            end_date_str = today.strftime("%Y%m%d")
            start_date = today - datetime.timedelta(days=self.announcement_scan_past_days)
            start_date_str = start_date.strftime("%Y%m%d")

            # Using stock_notice_report - this gets reports, not necessarily future catalysts
            # Need to parse content for future implications.
            # A more suitable function might be stock_ggcg_em (股东大会) or similar event-focused ones if available.
            # For now, we'll process titles/summaries of generic announcements.
            # Actual function might be stock_gsrl_gsgg_em for general announcements
            
            # Using stock_gsrl_gsgg_em as it seems more appropriate for general announcements
            df_announcements = akshare.stock_gsrl_gsgg_em(symbol=stock_code) # This fetches recent announcements
            
            if not df_announcements.empty:
                for _, row in df_announcements.iterrows():
                    title = row.get("公告标题", "")
                    # summary = row.get("公告摘要", "") # May not be available, content might be in URL
                    content_url = row.get("公告链接", "") 
                    announcement_date_str = row.get("公告日期", "")
                    
                    if not title or not announcement_date_str:
                        continue

                    announcement_date = datetime.datetime.strptime(announcement_date_str.split("T")[0], "%Y-%m-%d").date()

                    # Analyze content for catalysts
                    analysis_result = self._analyze_announcement_content(asset_id, title, title, content_url, look_ahead_days, catalyst_types_of_interest) # Using title as summary for now
                    
                    if analysis_result:
                        catalysts.append({
                            'catalyst_type': analysis_result['catalyst_type'],
                            'description': analysis_result['description'],
                            'date_event': analysis_result.get('date_event', announcement_date.isoformat()), # If LLM provides future date, use it
                            'date_identified': announcement_date.isoformat(),
                            'source_type': "CompanyAnnouncement_Akshare",
                            'potential_impact_assessment': analysis_result['potential_impact_assessment'],
                            'source_reference': content_url if content_url else title
                        })
        except Exception as e:
            print(f"Error scanning company announcements from Akshare for {asset_id}: {e}")
        return catalysts

    def scan_for_catalysts(self, request_data: dict) -> dict:
        """
        扫描给定资产的即将发生的催化剂。
        此方法协调不同来源（如Akshare的财报日历和公司公告）的催化剂扫描，
        并根据预看天数筛选结果。

        Scans for upcoming catalysts for a given asset.
        This method orchestrates catalyst scanning from different sources 
        (e.g., Akshare for earnings calendar and company announcements)
        and filters results based on look_ahead_days.

        Args:
            request_data (dict): 包含扫描所需数据的字典，
                                 A dictionary containing the data needed for scanning,
                                 including:
                                 - 'asset_id' (str): 资产ID (例如 '000001.SZ')
                                 - 'asset_industry' (str, optional): 资产所属行业 (可选, 当前未使用但可用于新闻API)
                                                                    Industry of the asset (optional, currently unused but could be for news APIs).
                                 - 'look_ahead_days' (int, optional): 向未来看多少天 (可选)。
                                                                    How many days to look into the future (optional).
                                 - 'catalyst_types_of_interest' (List[str], optional): 感兴趣的特定催化剂类型 (可选, 例如 ["earnings_release", "major_contract"] 或 ["all"])。
                                                                                     Specific types to look for (optional, e.g., ["earnings_release", "major_contract"] or ["all"]).
        
        Returns:
            dict: 包含找到的催化剂列表的字典，
                  A dictionary containing the list of found catalysts, for example:
                  {
                      'asset_id': str, # 资产ID
                      'catalysts_found': [ # 找到的催化剂列表
                          {
                              'catalyst_type': str, # 例如："EarningsRelease" (财报发布), "ProductLaunch_NewDrug" (新药发布)
                              'description': str,   # 简要描述
                              'date_event': str,    # 事件日期 (ISO日期或文本描述, "Unknown" 如果不确定)
                              'date_identified': str, # 信息发现日期 (ISO日期)
                              'source_type': str,   # 例如："FinancialCalendar_Akshare_YYSJ", "CompanyAnnouncement_Akshare"
                              'potential_impact_assessment': str, # 例如："Positive" (正面), "Negative" (负面), "Uncertain" (不确定)
                              'source_reference': str (optional) # URL或新闻标题 (可选)
                          },
                          # ... 更多催化剂
                      ],
                      'error': str (optional) # 如果发生错误，则包含错误信息
                  }
        """
        asset_id = request_data.get('asset_id')
        if not asset_id:
            return {'asset_id': 'Unknown', 'catalysts_found': [], 'error': 'asset_id is required'}

        # asset_industry = request_data.get('asset_industry') # Currently unused, but could be for news APIs
        look_ahead_days = request_data.get('look_ahead_days', self.default_look_ahead_days)
        catalyst_types_of_interest = request_data.get('catalyst_types_of_interest', self.default_catalyst_types)
        if not isinstance(catalyst_types_of_interest, list): # Ensure it's a list
            catalyst_types_of_interest = self.default_catalyst_types
        if "all" in catalyst_types_of_interest: # If "all" is specified, use all default types
            catalyst_types_of_interest = self.default_catalyst_types


        all_found_catalysts = []

        # 1. Fetch Scheduled Events (Earnings)
        if "earnings_release" in catalyst_types_of_interest or "all_scheduled" in catalyst_types_of_interest:
             all_found_catalysts.extend(self._fetch_earnings_catalysts_akshare(asset_id, look_ahead_days))
        
        # 2. Scan Company Announcements
        all_found_catalysts.extend(self._scan_company_announcements_akshare(asset_id, look_ahead_days, catalyst_types_of_interest))

        # 3. Scan News APIs (Placeholder)
        # if "general_news" in catalyst_types_of_interest or "all" in catalyst_types_of_interest:
        #     # Placeholder: news_api_catalysts = self._scan_general_news_apis(asset_id, asset_industry, look_ahead_days)
        #     # all_found_catalysts.extend(news_api_catalysts)
        #     pass


        # Filter catalysts to be within look_ahead_days if date_event is known and in the future
        # For those with "Unknown" date_event from announcements, they are kept if identified recently.
        # This filtering is partially done within helper methods for scheduled events.
        # For announcements, if date_event is "Unknown", it's assumed potentially relevant if identified recently.
        
        final_catalysts = []
        today_iso = datetime.date.today().isoformat()
        limit_date_iso = (datetime.date.today() + datetime.timedelta(days=look_ahead_days)).isoformat()

        for cat in all_found_catalysts:
            event_date_str = cat.get('date_event', "")
            # If event date is known and in the past (but not today), skip, unless it's a guidance identified today.
            if event_date_str != "Unknown" and event_date_str < today_iso and cat.get('catalyst_type') != "EarningsGuidance":
                continue
            # If event date is known and beyond the look_ahead period, skip.
            if event_date_str != "Unknown" and event_date_str > limit_date_iso:
                continue
            final_catalysts.append(cat)


        return {
            'asset_id': asset_id,
            'catalysts_found': final_catalysts
        }

# 示例用法 (用于测试)
# Example usage (for testing)
if __name__ == '__main__':
    # 此部分用于直接测试代理的方法，
    # This section would be for direct testing of the agent's methods,
    # 而非其在 autogen 框架内的典型操作。
    # not for its typical operation within the autogen framework.
    
    # Important: Akshare calls can be slow and might require specific environment setups.
    # These tests might not run correctly in all environments without internet access or akshare working.
    print("Note: Akshare calls can be slow and might require specific environment setups/internet.")

    test_agent_config = {
        "DEFAULT_LOOK_AHEAD_DAYS": 90, # Look further for testing
        "ANNOUNCEMENT_SCAN_PAST_DAYS": 30, # Scan more announcements for testing
        "DEFAULT_CATALYST_TYPES": ["earnings_release", "major_contract", "product_launch"],
         "ANNOUNCEMENT_KEYWORDS": { # Example keywords
            "product_launch": ["新产品", "发布"],
            "major_contract": ["重大合同", "中标"],
        }
    }
    
    # Test with no LLM (keyword-based)
    catalyst_agent_no_llm = CatalystScoutAgent(config=test_agent_config, llm_config=None)
    
    # Example for a well-known stock for which announcements are likely
    # Using a common stock like Ping An Bank (000001.SZ) or Vanke (000002.SZ)
    sample_request_pa = {
        'asset_id': '000001.SZ', # Ping An Bank
        'asset_industry': 'Banking',
        'look_ahead_days': 60,
        'catalyst_types_of_interest': ["earnings_release", "major_contract"] 
    }
    
    print(f"\n--- Test 1: Scanning catalysts for {sample_request_pa['asset_id']} (No LLM) ---")
    results_pa = catalyst_agent_no_llm.scan_for_catalysts(sample_request_pa)
    print(f"Asset ID: {results_pa.get('asset_id')}")
    if results_pa.get('error'):
        print(f"Error: {results_pa.get('error')}")
    print("Catalysts Found:")
    if results_pa.get('catalysts_found'):
        for c in results_pa['catalysts_found']:
            print(f"  - Type: {c['catalyst_type']}, Date: {c['date_event']}, Desc: {c['description'][:60]}..., Impact: {c['potential_impact_assessment']}, Source: {c['source_type']}")
    else:
        print("  (No catalysts found or error occurred)")

    # Example for another stock
    sample_request_vke = {
        'asset_id': '000002.SZ', # Vanke
        'asset_industry': 'Real Estate',
        'catalyst_types_of_interest': ["all"] # Test "all" types
    }
    print(f"\n--- Test 2: Scanning catalysts for {sample_request_vke['asset_id']} (No LLM, all types) ---")
    results_vke = catalyst_agent_no_llm.scan_for_catalysts(sample_request_vke)
    print(f"Asset ID: {results_vke.get('asset_id')}")
    if results_vke.get('error'):
        print(f"Error: {results_vke.get('error')}")
    print("Catalysts Found:")
    if results_vke.get('catalysts_found'):
        for c in results_vke['catalysts_found']:
            print(f"  - Type: {c['catalyst_type']}, Date: {c['date_event']}, Desc: {c['description'][:60]}..., Impact: {c['potential_impact_assessment']}, Source: {c['source_type']}")
    else:
        print("  (No catalysts found or error occurred)")

    # Simulate LLM config for conceptual testing path (actual LLM call is commented out)
    # test_llm_config = {"model": "gpt-3.5-turbo", "api_key": "YOUR_API_KEY_HERE_IF_TESTING_LIVE_LLM"}
    # catalyst_agent_with_llm = CatalystScoutAgent(config=test_agent_config, llm_config=test_llm_config)
    # print(f"\n--- Test 3: Scanning catalysts for {sample_request_pa['asset_id']} (Simulated LLM path) ---")
    # results_llm_pa = catalyst_agent_with_llm.scan_for_catalysts(sample_request_pa) # Will use keyword fallback as LLM call is placeholder
    # print(f"Asset ID: {results_llm_pa.get('asset_id')}")
    # print("Catalysts Found (LLM path - uses keyword fallback):")
    # if results_llm_pa.get('catalysts_found'):
    #     for c in results_llm_pa['catalysts_found']:
    #         print(f"  - Type: {c['catalyst_type']}, Date: {c['date_event']}, Desc: {c['description'][:60]}..., Impact: {c['potential_impact_assessment']}, Source: {c['source_type']}")
    # else:
    #     print("  (No catalysts found or error occurred)")
    
    print("\nNote: Live Akshare calls were attempted. Results depend on current data and network access.")
