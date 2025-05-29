import logging
from typing import Dict, Any, Optional, List, Tuple # Added List, Tuple
import json # For LLM response parsing

from .base_agent import BaseAgent
from contrarian_trader.src.data_sources import NewsDataSource # Adjusted import path

# Defined detailed sentiment categories as per design
DETAILED_SENTIMENT_CATEGORIES = [
    "IrrationalPanic", "RumorDrivenDrop", "GenuineConcern", "OverreactionToMinorNews",
    "NeutralObservation", "FactualReport",
    "CautiousOptimism", "GenuinePositiveDevelopment", "HypeDrivenRally", "IrrationalExuberance",
    "Misinformation" # Could be positive or negative misinformation
]

class NewsAgent(BaseAgent):
    """
    (NewsSentimentAgent - 新闻情绪分析智能体)
    负责获取、处理和分析新闻数据，并进行详细的情绪分类。
    Agent responsible for fetching, processing, and analyzing news data, including detailed sentiment classification.
    """

    def __init__(self, 
                 news_data_source: NewsDataSource, 
                 llm_config: Optional[Dict[str, Any]] = None, # For detailed sentiment via LLM
                 agent_name: str = "NewsSentimentAgent", # Renamed for clarity
                 logger_instance: Optional[logging.Logger] = None):
        """
        初始化 NewsSentimentAgent。
        Initializes the NewsSentimentAgent.

        Args:
            news_data_source (NewsDataSource): 用于获取新闻的 NewsDataSource 实例。
                                               An instance of NewsDataSource to fetch news.
            llm_config (Optional[Dict[str, Any]]): LLM 配置，用于详细情绪分析。
                                                   LLM configuration for detailed sentiment analysis.
            agent_name (str): 智能体的名称。Name of the agent.
            logger_instance (Optional[logging.Logger]): 可选的日志记录器实例。
                                                        Optional logger instance.
        """
        super().__init__(agent_name=agent_name, logger_instance=logger_instance)
        if not isinstance(news_data_source, NewsDataSource):
            self.logger.error("NewsAgent requires a valid NewsDataSource instance.")
            raise ValueError("news_data_source must be an instance of NewsDataSource.")
        
        self.news_data_source = news_data_source
        self.llm_config = llm_config # Store LLM config
        self.analyzed_asset_news_data: Optional[Dict[str, Any]] = None # Stores final aggregated analysis
        self.logger.info(f"{self.agent_name} initialized with {news_data_source.__class__.__name__} and LLM config {'provided' if llm_config else 'not provided'}.")

    def _call_llm(self, prompt: str, model_name: Optional[str] = "default_model") -> str | None:
        """
        (私有方法) 概念性的LLM调用封装。
        实际应用中需替换为真正的LLM API调用实现。

        (Private method) Conceptual LLM call wrapper.
        Replace with actual LLM API call implementation in a real application.

        Args:
            prompt (str): 发送给LLM的提示。Prompt to send to the LLM.
            model_name (Optional[str]): 使用的LLM模型名称。Name of the LLM model to use.

        Returns:
            str | None: LLM的文本响应，如果调用失败则返回None。
                        Text response from LLM, or None if the call fails.
        """
        if not self.llm_config:
            self.logger.warning("LLM config not provided. Cannot make LLM call for detailed sentiment.")
            return None
        
        self.logger.debug(f"Conceptual LLM Call with model '{model_name}'. Prompt:\n{prompt[:500]}...") # Log truncated prompt
        
        # Placeholder for actual LLM call.
        # Replace this with your actual LLM client integration.
        # from ...openai_adapter import get_chat_completion # Example
        # return get_chat_completion(messages=[{"role":"user", "content":prompt}], model=model_name, ...)

        # Simulate LLM response based on prompt content for testing
        if "Detailed sentiment analysis for news item" in prompt:
            # Simulate a plausible JSON response for detailed sentiment
            mock_response = {
                "original_sentiment_score": round((len(prompt) % 20 - 10) / 10, 2), # Score -1.0 to 0.9
                "detailed_sentiment_category": DETAILED_SENTIMENT_CATEGORIES[len(prompt) % len(DETAILED_SENTIMENT_CATEGORIES)],
                "rationale": "LLM-generated rationale: The news exhibits characteristics of the identified category due to specific phrasing and market context.",
                "key_phrases": [f"phrase_{i+1}" for i in range(min(3, len(prompt) % 4))]
            }
            return json.dumps(mock_response)
        return None

    def analyze_news_item_detailed(self, news_headline: str, news_content: str, news_date: Optional[str] = None, news_source: Optional[str] = None) -> Dict[str, Any]:
        """
        对单个新闻条目进行详细的情绪分析。
        构造针对LLM的提示，要求进行细粒度情绪分类、原始情绪评分、理由和关键短语提取。
        解析LLM的JSON响应。

        Performs detailed sentiment analysis on a single news item.
        Constructs an advanced LLM prompt for granular sentiment classification, original sentiment score,
        rationale, and key phrase extraction. Parses the JSON response from the LLM.

        Args:
            news_headline (str): 新闻标题。The headline of the news item.
            news_content (str): 新闻的主要内容或摘要。The main content or summary of the news item.
            news_date (Optional[str]): 新闻发布日期 (ISO格式)。Publication date of the news (ISO format).
            news_source (Optional[str]): 新闻来源。Source of the news.

        Returns:
            Dict[str, Any]: 包含详细分析结果的字典。
                            A dictionary containing the detailed analysis results.
        """
        self.logger.debug(f"Analyzing news item: '{news_headline}'")
        if not self.llm_config:
            self.logger.warning("LLM not configured. Skipping detailed sentiment analysis, returning basic structure.")
            return {
                "headline": news_headline,
                "source": news_source,
                "date": news_date,
                "original_sentiment_score": None, # Placeholder
                "detailed_sentiment_category": "UnavailableDueToNoLLM",
                "rationale": "LLM not configured for detailed analysis.",
                "key_phrases": []
            }

        prompt = (
            f"Perform detailed sentiment analysis for news item:\n"
            f"Headline: \"{news_headline}\"\n"
            f"Content/Summary: \"{news_content[:500]}...\"\n" # Use a snippet for brevity
            f"Consider the A-share market context. Respond in JSON format with the following fields:\n"
            f"1. `original_sentiment_score`: A float between -1.0 (very negative) and 1.0 (very positive).\n"
            f"2. `detailed_sentiment_category`: Choose ONE most fitting category from this list: {json.dumps(DETAILED_SENTIMENT_CATEGORIES)}.\n"
            f"3. `rationale`: A brief (1-2 sentences) explanation for your chosen category and score.\n"
            f"4. `key_phrases`: A list of up to 3 key phrases (strings) from the text that support your analysis."
        )

        llm_response_str = self._call_llm(prompt, model_name=self.llm_config.get("model", "default_model"))

        analysis_result = {
            "headline": news_headline,
            "source": news_source,
            "date": news_date,
            "original_sentiment_score": None,
            "detailed_sentiment_category": "ErrorInAnalysis",
            "rationale": "LLM call failed or parsing error.",
            "key_phrases": []
        }

        if llm_response_str:
            try:
                data = json.loads(llm_response_str)
                analysis_result["original_sentiment_score"] = data.get("original_sentiment_score")
                analysis_result["detailed_sentiment_category"] = data.get("detailed_sentiment_category", "ParsingError")
                analysis_result["rationale"] = data.get("rationale", "Rationale not provided by LLM.")
                key_phrases = data.get("key_phrases", [])
                analysis_result["key_phrases"] = key_phrases if isinstance(key_phrases, list) else []
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM JSON response for news item '{news_headline}': {e}. Response: {llm_response_str}", exc_info=True)
        else:
            self.logger.warning(f"No response from LLM for news item '{news_headline}'.")
            
        return analysis_result

    def analyze_asset_news(self, asset_id: str, news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析给定资产的一系列新闻条目，并聚合情绪结果。
        对列表中的每个新闻条目调用 `analyze_news_item_detailed`，然后计算聚合的总体情绪评分和
        主要的详细情绪类别。

        Analyzes a list of news items for a given asset and aggregates the sentiment results.
        Calls `analyze_news_item_detailed` for each news item in the list, then calculates
        an aggregated overall sentiment score and a dominant detailed sentiment category.

        Args:
            asset_id (str): 资产ID。The ID of the asset.
            news_list (List[Dict[str, Any]]): 从数据源获取的新闻条目列表。
                                               A list of news items (dictionaries) from the data source.

        Returns:
            Dict[str, Any]: 包含已分析新闻条目和聚合情绪的字典。
                            A dictionary containing the analyzed news items and aggregated sentiment.
        """
        self.logger.info(f"Starting detailed news analysis for asset: {asset_id}, {len(news_list)} articles.")
        analyzed_items = []
        for article in news_list:
            headline = article.get('title', 'N/A')
            content = article.get('summary') or article.get('content') or headline # Use summary, content, or fallback to title
            date = article.get('published_at') or article.get('date') # Handle different date field names
            source = article.get('source')
            
            detailed_analysis = self.analyze_news_item_detailed(headline, content, date, source)
            analyzed_items.append(detailed_analysis)

        # Aggregation logic
        overall_sentiment_score = 0.0
        category_counts: Dict[str, int] = {}
        valid_scores_count = 0

        for item in analyzed_items:
            score = item.get("original_sentiment_score")
            if isinstance(score, (int, float)): # Check if score is a number
                overall_sentiment_score += score
                valid_scores_count += 1
            
            category = item.get("detailed_sentiment_category")
            if category and category not in ["ErrorInAnalysis", "ParsingError", "UnavailableDueToNoLLM"]:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        aggregated_dominant_category = "NeutralObservation" # Default if no clear dominant
        if category_counts:
            aggregated_dominant_category = max(category_counts, key=category_counts.get)
        
        if valid_scores_count > 0:
            overall_sentiment_score /= valid_scores_count
        else:
            overall_sentiment_score = 0.0 # Default if no valid scores

        self.analyzed_asset_news_data = {
            "asset_id": asset_id,
            "news_items_analyzed": analyzed_items,
            "aggregated_dominant_detailed_category": aggregated_dominant_category,
            "overall_sentiment_score": round(overall_sentiment_score, 3)
        }
        self.logger.info(f"Completed detailed news analysis for {asset_id}. Dominant category: {aggregated_dominant_category}, Overall score: {overall_sentiment_score:.3f}")
        return self.analyzed_asset_news_data


    def make_recommendation(self) -> Dict[str, Any]:
        """
        提供经过详细分析和聚合的新闻数据。
        此智能体主要提供数据；此处的“推荐”指已分析的新闻。

        Provides the detailed analyzed and aggregated news data.
        This agent primarily provides data; "recommendation" here means analyzed news.

        Returns:
            Dict[str, Any]: 包含带有详细情绪分析的新闻条目和聚合结果的字典。
                            A dictionary containing news items with detailed sentiment and aggregated results.
        """
        if self.analyzed_asset_news_data:
            self.logger.info(f"Providing analyzed news data for asset: {self.analyzed_asset_news_data.get('asset_id')}")
            return {
                "status": "success", 
                "data_type": "detailed_news_sentiment_analysis", 
                "content": self.analyzed_asset_news_data,
            }
        else:
            self.logger.warning("No analyzed news data available to provide.")
            return {"status": "no_data", "data_type": "detailed_news_sentiment_analysis", "content": {}}

    def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        协调获取新闻并对其进行详细情绪处理。
        Orchestrates fetching news and processing it for detailed sentiment.

        Args:
            data (Optional[Dict[str, Any]]): 包含获取新闻参数的字典，
                                              例如：{'symbol': 'AAPL', 'limit': 10}。
                                              A dictionary containing parameters for fetching news,
                                              e.g., {'symbol': 'AAPL', 'limit': 10}.

        Returns:
            Dict[str, Any]: 包含带有详细情绪分析的新闻条目和聚合结果的字典。
                            A dictionary containing news items with detailed sentiment and aggregated results.
        """
        self.logger.info(f"{self.agent_name} run started with parameters: {data}")
        if not data or 'symbol' not in data: # 'symbol' is used as asset_id here
            self.logger.error("Missing 'symbol' (asset_id) in run parameters for NewsSentimentAgent.")
            return {"status": "error", "message": "Missing 'symbol' (asset_id) in run parameters."}

        asset_id = data['symbol']
        limit = data.get('limit', 10) 

        try:
            raw_articles = self.news_data_source.fetch_data(asset_id, limit=limit)
            
            if raw_articles:
                self.analyze_asset_news(asset_id=asset_id, news_list=raw_articles)
            else:
                self.logger.warning(f"No raw news articles fetched for {asset_id}.")
                self.analyzed_asset_news_data = { # Ensure it's initialized
                    "asset_id": asset_id, "news_items_analyzed": [], 
                    "aggregated_dominant_detailed_category": "NeutralObservation", 
                    "overall_sentiment_score": 0.0
                }
        except Exception as e:
            self.logger.error(f"Error during {self.agent_name} run for {asset_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        
        recommendation = self.make_recommendation()
        self.logger.info(f"{self.agent_name} run finished. Result status: {recommendation.get('status')}")
        return recommendation

if __name__ == '__main__':
    # This is for testing purposes and won't run when imported.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock NewsDataSource for testing
    class MockNewsDataSource(NewsDataSource):
        def __init__(self, api_key: str = "test_news_key"):
            super().__init__(api_key=api_key) # Call to parent's __init__
            self.logger_instance = logging.getLogger(self.__class__.__name__) # Ensure logger is initialized
            self.logger_instance.info("MockNewsDataSource initialized for testing.")


        def fetch_data(self, symbol: str, **kwargs) -> List[Dict[str, Any]]:
            limit = kwargs.get('limit', 2)
            self.logger_instance.debug(f"Mock fetching {limit} news articles for {symbol}")
            return [
                {'title': f'{symbol} Soars on New Tech!', 'summary': 'A very positive summary about {symbol}. Great performance due to breakthrough innovation.', 'source': 'Mock News', 'published_at': '2023-01-01T10:00:00Z', 'content': 'Detailed content about soaring.'},
                {'title': f'{symbol} Plummets on Rumors?', 'summary': 'Concerning news for {symbol} as market reacts to unconfirmed information.', 'source': 'Mock Finance', 'published_at': '2023-01-01T11:00:00Z', 'content': 'Detailed content about plummeting.'},
                {'title': f'{symbol} Stable Outlook', 'summary': 'Neutral observation for {symbol}, maintaining current trajectory.', 'source': 'Mock Analysis', 'published_at': '2023-01-01T12:00:00Z', 'content': 'Detailed content about stability.'}

            ][:limit]

    print("\n--- Testing NewsSentimentAgent (Enhanced) ---")
    mock_news_source = MockNewsDataSource()
    
    # Simulate LLM config for testing the flow
    dummy_llm_config = {
        "model": "test_llm_model", 
        "api_key": "test_key", 
        "some_other_param": "value"
    }
    
    news_agent_enhanced = NewsAgent(news_data_source=mock_news_source, llm_config=dummy_llm_config)

    print("\nRunning agent for news data (TESTAAPL, limit 2):")
    news_params_aapl = {'symbol': 'TESTAAPL', 'limit': 2}
    result_aapl = news_agent_enhanced.run(data=news_params_aapl)
    
    print("\n--- Analysis Result for TESTAAPL ---")
    # Pretty print the JSON structure
    print(json.dumps(result_aapl, indent=2, ensure_ascii=False))

    if result_aapl['status'] == 'success' and result_aapl['content']:
        print(f"\nAsset ID: {result_aapl['content']['asset_id']}")
        print(f"Overall Sentiment Score: {result_aapl['content']['overall_sentiment_score']}")
        print(f"Aggregated Dominant Detailed Category: {result_aapl['content']['aggregated_dominant_detailed_category']}")
        print("Analyzed News Items:")
        for item in result_aapl['content']['news_items_analyzed']:
            print(f"  Headline: {item['headline']}")
            print(f"    Original Sentiment Score: {item['original_sentiment_score']}")
            print(f"    Detailed Category: {item['detailed_sentiment_category']}")
            print(f"    Rationale: {item['rationale']}")
            print(f"    Key Phrases: {item['key_phrases']}")
            print("-" * 20)
            
    print("\n--- NewsSentimentAgent (Enhanced) tests complete ---")
