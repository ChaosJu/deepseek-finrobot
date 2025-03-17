"""
代理库模块 - 实现各种专业代理
"""

import autogen
from typing import Dict, List, Optional, Union, Any
import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from ..data_source import akshare_utils, cn_news_utils
from ..utils import get_current_date, format_financial_number, cached

class MarketForecasterAgent:
    """
    市场预测代理 - 预测股票走势
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化市场预测代理
        
        Args:
            llm_config: LLM配置
        """
        self.llm_config = llm_config
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建市场分析师代理
        self.market_analyst = autogen.AssistantAgent(
            name="MarketAnalyst",
            llm_config=llm_config,
            system_message="""你是一位专业的市场分析师，擅长分析股票市场数据和新闻，预测股票走势。
你需要分析提供的股票数据、财务指标和相关新闻，给出对股票未来走势的预测。
你的分析应该包括以下几个方面：
1. 技术分析：基于股票价格、交易量和技术指标
2. 基本面分析：基于公司财务数据和行业状况
3. 情绪分析：基于市场新闻和舆论
4. 综合预测：给出未来一周的股票走势预测（上涨/下跌/震荡）和可能的价格区间

请确保你的分析逻辑清晰，预测有理有据。"""
        )
    
    @cached("market_forecast", expire_seconds=3600)
    def predict(self, symbol: str, days: int = 7) -> str:
        """
        预测股票走势
        
        Args:
            symbol: 股票代码（如：000001）
            days: 分析过去几天的数据和新闻
            
        Returns:
            预测结果
        """
        try:
            # 获取股票信息
            stock_info = akshare_utils.get_stock_info(symbol)
            if "error" in stock_info:
                return f"获取股票信息失败: {stock_info['error']}"
                
            # 获取股票历史数据
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y%m%d")
            stock_history = akshare_utils.get_stock_history(symbol, start_date=start_date, end_date=end_date)
            
            # 获取股票财务指标
            financial_indicator = akshare_utils.get_stock_financial_indicator(symbol)
            
            # 获取股票相关新闻
            company_name = stock_info.get("股票简称", symbol)
            news = cn_news_utils.search_news(company_name, days=days)
            
            # 获取行业新闻
            industry = stock_info.get("所处行业", "")
            industry_news = cn_news_utils.get_stock_industry_news(industry, limit=5)
            
            # 获取市场情绪
            market_sentiment = cn_news_utils.get_stock_market_sentiment()
            
            # 构建分析请求
            analysis_request = f"""
请分析以下股票数据并预测未来一周的走势：

股票代码: {symbol}
股票名称: {company_name}
所属行业: {industry}
当前价格: {stock_info.get('最新价', 'N/A')}
市盈率: {stock_info.get('市盈率(动态)', 'N/A')}
市净率: {stock_info.get('市净率', 'N/A')}

最近30天的价格走势:
{stock_history[['收盘', '成交量']].tail(10).to_string()}

主要财务指标:
{financial_indicator.head(5).to_string() if not financial_indicator.empty else "无财务数据"}

相关新闻:
{news[['title', 'content']].head(5).to_string() if not news.empty else "无相关新闻"}

行业新闻:
{industry_news[['title', 'content']].head(3).to_string() if not industry_news.empty else "无行业新闻"}

市场情绪指标:
{json.dumps(market_sentiment, ensure_ascii=False, indent=2) if market_sentiment and "error" not in market_sentiment else "无市场情绪数据"}

请基于以上数据，分析该股票未来一周的可能走势。你的分析应包括技术面、基本面和市场情绪三个方面，并给出明确的预测（上涨/下跌/震荡）和可能的价格区间。
"""
            
            # 发起对话
            self.user_proxy.initiate_chat(
                self.market_analyst,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为预测结果
            prediction = self.user_proxy.chat_messages[self.market_analyst.name][-1]["content"]
            
            return prediction
        except Exception as e:
            return f"预测股票走势时出错: {str(e)}"
    
    def batch_predict(self, symbols: List[str], days: int = 7, max_workers: int = 5) -> Dict[str, str]:
        """
        批量预测多个股票走势
        
        Args:
            symbols: 股票代码列表
            days: 分析过去几天的数据和新闻
            max_workers: 最大并行工作线程数
            
        Returns:
            预测结果字典，键为股票代码，值为预测结果
        """
        results = {}
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_symbol = {executor.submit(self.predict, symbol, days): symbol for symbol in symbols}
            
            # 获取结果
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    prediction = future.result()
                    results[symbol] = prediction
                except Exception as e:
                    results[symbol] = f"预测股票走势时出错: {str(e)}"
        
        return results
    
    def export_prediction(self, prediction: str, format: str = "markdown", output_file: Optional[str] = None) -> str:
        """
        导出预测结果
        
        Args:
            prediction: 预测结果
            format: 导出格式，支持 "markdown", "html", "text"
            output_file: 输出文件路径，如果为None则返回字符串
            
        Returns:
            导出的内容或文件路径
        """
        # 获取当前日期
        current_date = get_current_date()
        
        # 根据格式处理内容
        if format == "markdown":
            content = f"# 股票走势预测报告\n\n**生成日期**: {current_date}\n\n{prediction}"
        elif format == "html":
            content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>股票走势预测报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .date {{ color: #666; margin-bottom: 20px; }}
        .prediction {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <h1>股票走势预测报告</h1>
    <div class="date"><strong>生成日期</strong>: {current_date}</div>
    <div class="prediction">{prediction.replace('\n', '<br>')}</div>
</body>
</html>"""
        else:  # text
            content = f"股票走势预测报告\n\n生成日期: {current_date}\n\n{prediction}"
        
        # 如果指定了输出文件，则写入文件
        if output_file:
            # 确保文件扩展名与格式匹配
            if format == "markdown" and not output_file.endswith((".md", ".markdown")):
                output_file += ".md"
            elif format == "html" and not output_file.endswith((".html", ".htm")):
                output_file += ".html"
            elif format == "text" and not output_file.endswith((".txt")):
                output_file += ".txt"
            
            # 写入文件
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            return output_file
        
        return content
    
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        self.market_analyst.reset()

class FinancialReportAgent:
    """
    财务报告代理 - 生成财务分析报告
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化财务报告代理
        
        Args:
            llm_config: LLM配置
        """
        self.llm_config = llm_config
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建财务分析师代理
        self.financial_analyst = autogen.AssistantAgent(
            name="FinancialAnalyst",
            llm_config=llm_config,
            system_message="""你是一位专业的财务分析师，擅长分析公司财务数据，生成财务分析报告。
你需要分析提供的公司财务数据、行业数据和相关新闻，给出对公司财务状况的分析报告。
你的报告应该包括以下几个方面：
1. 公司概况：公司基本信息和业务介绍
2. 财务状况分析：基于资产负债表、利润表和现金流量表
3. 财务比率分析：包括盈利能力、偿债能力、运营能力和成长能力
4. 行业对比分析：与行业平均水平和主要竞争对手的对比
5. 风险分析：潜在的财务风险和经营风险
6. 投资建议：基于财务分析的投资建议

请确保你的分析逻辑清晰，报告专业全面。"""
        )
    
    def generate_report(self, symbol: str) -> str:
        """
        生成财务分析报告
        
        Args:
            symbol: 股票代码（如：000001）
            
        Returns:
            财务分析报告
        """
        try:
            # 获取股票信息
            stock_info = akshare_utils.get_stock_info(symbol)
            if "error" in stock_info:
                return f"获取股票信息失败: {stock_info['error']}"
                
            # 获取股票财务指标
            financial_indicator = akshare_utils.get_stock_financial_indicator(symbol)
            
            # 获取行业成分股
            industry = stock_info.get("所处行业", "")
            industry_list = akshare_utils.get_stock_industry_list()
            industry_code = None
            for _, row in industry_list.iterrows():
                if row["板块名称"] == industry:
                    industry_code = row["板块代码"]
                    break
                    
            industry_stocks = None
            if industry_code:
                industry_stocks = akshare_utils.get_stock_industry_constituents(industry_code)
            
            # 获取研究报告
            research_reports = akshare_utils.get_stock_research_report(symbol)
            
            # 构建分析请求
            analysis_request = f"""
请为以下公司生成一份财务分析报告：

公司基本信息:
股票代码: {symbol}
股票名称: {stock_info.get('股票简称', 'N/A')}
所属行业: {industry}
总市值: {stock_info.get('总市值', 'N/A')}
流通市值: {stock_info.get('流通市值', 'N/A')}

主要财务指标:
{financial_indicator.to_string() if not financial_indicator.empty else "无财务数据"}

行业对比:
{industry_stocks[['代码', '名称', '最新价', '涨跌幅', '市盈率']].head(5).to_string() if industry_stocks is not None and not industry_stocks.empty else "无行业对比数据"}

研究报告:
{research_reports[['title', 'author']].head(3).to_string() if not research_reports.empty else "无研究报告"}

请基于以上数据，生成一份全面的财务分析报告。你的报告应包括公司概况、财务状况分析、财务比率分析、行业对比分析、风险分析和投资建议。
"""
            
            # 发起对话
            self.user_proxy.initiate_chat(
                self.financial_analyst,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为报告
            report = self.user_proxy.chat_messages[self.financial_analyst.name][-1]["content"]
            
            return report
        except Exception as e:
            return f"生成财务分析报告时出错: {str(e)}"
    
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        self.financial_analyst.reset()

class NewsAnalysisAgent:
    """
    新闻分析代理 - 分析财经新闻
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化新闻分析代理
        
        Args:
            llm_config: LLM配置
        """
        self.llm_config = llm_config
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建新闻分析师代理
        self.news_analyst = autogen.AssistantAgent(
            name="NewsAnalyst",
            llm_config=llm_config,
            system_message="""你是一位专业的财经新闻分析师，擅长分析财经新闻对市场和个股的影响。
你需要分析提供的财经新闻，给出对市场或个股的影响分析。
你的分析应该包括以下几个方面：
1. 新闻摘要：简要概括新闻内容
2. 影响分析：分析新闻对市场或个股的潜在影响
3. 情绪评估：评估新闻的市场情绪（积极/消极/中性）
4. 投资建议：基于新闻分析的投资建议

请确保你的分析逻辑清晰，观点客观中立。"""
        )
    
    def analyze_news(self, keywords: str = None, days: int = 3, limit: int = 10) -> str:
        """
        分析财经新闻
        
        Args:
            keywords: 搜索关键词，如果为None则获取最新财经新闻
            days: 分析过去几天的新闻
            limit: 分析的新闻数量
            
        Returns:
            新闻分析结果
        """
        try:
            # 获取新闻
            if keywords:
                news = cn_news_utils.search_news(keywords, days=days, limit=limit)
            else:
                news = cn_news_utils.get_financial_news(limit=limit)
                
            if news.empty:
                return "未找到相关新闻"
                
            # 获取重大新闻
            major_news = cn_news_utils.get_major_news()
            
            # 获取市场情绪
            market_sentiment = cn_news_utils.get_stock_market_sentiment()
            
            # 获取热门股票
            hot_stocks = cn_news_utils.get_stock_hot_rank()
            
            # 构建分析请求
            analysis_request = f"""
请分析以下财经新闻并给出市场影响分析：

最新财经新闻:
{news[['title', 'content']].to_string() if not news.empty else "无相关新闻"}

重大财经新闻:
{major_news[['title', 'content']].head(3).to_string() if not major_news.empty else "无重大新闻"}

市场情绪指标:
{json.dumps(market_sentiment, ensure_ascii=False, indent=2) if market_sentiment and "error" not in market_sentiment else "无市场情绪数据"}

热门股票:
{hot_stocks[['代码', '名称', '最新价', '涨跌幅']].head(5).to_string() if not hot_stocks.empty else "无热门股票数据"}

请基于以上新闻，分析其对市场或相关个股的潜在影响。你的分析应包括新闻摘要、影响分析、情绪评估和投资建议。
"""
            
            # 发起对话
            self.user_proxy.initiate_chat(
                self.news_analyst,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为分析结果
            analysis = self.user_proxy.chat_messages[self.news_analyst.name][-1]["content"]
            
            return analysis
        except Exception as e:
            return f"分析财经新闻时出错: {str(e)}"
    
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        self.news_analyst.reset()

class IndustryAnalysisAgent:
    """
    行业分析代理 - 分析行业趋势和机会
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化行业分析代理
        
        Args:
            llm_config: LLM配置
        """
        self.llm_config = llm_config
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建行业分析师代理
        self.industry_analyst = autogen.AssistantAgent(
            name="IndustryAnalyst",
            llm_config=llm_config,
            system_message="""你是一位专业的行业分析师，擅长分析行业趋势、竞争格局和投资机会。
你需要分析提供的行业数据、成分股表现和相关新闻，给出对行业未来发展的分析。
你的分析应该包括以下几个方面：
1. 行业概况：行业定义、范围和主要特点
2. 行业格局：主要参与者、市场份额和竞争态势
3. 行业趋势：技术发展、政策环境和消费者行为变化
4. 投资机会：成长潜力、风险因素和投资建议

请确保你的分析逻辑清晰，观点有理有据。"""
        )
    
    @cached("industry_analysis", expire_seconds=86400)  # 缓存1天
    def analyze_industry(self, industry_name: str, days: int = 30) -> str:
        """
        分析行业趋势和机会
        
        Args:
            industry_name: 行业名称（如：银行、医药、计算机）
            days: 分析过去几天的数据和新闻
            
        Returns:
            行业分析结果
        """
        try:
            # 获取行业列表
            industry_list = akshare_utils.get_stock_industry_list()
            
            # 查找行业代码
            industry_code = None
            for _, row in industry_list.iterrows():
                if row["板块名称"] == industry_name:
                    industry_code = row["板块代码"]
                    break
            
            if not industry_code:
                return f"未找到行业: {industry_name}"
            
            # 获取行业成分股
            industry_stocks = akshare_utils.get_stock_industry_constituents(industry_code)
            
            if industry_stocks.empty:
                return f"获取行业成分股失败: {industry_name}"
            
            # 获取行业新闻
            industry_news = cn_news_utils.get_stock_industry_news(industry_name, limit=10)
            
            # 获取市场情绪
            market_sentiment = cn_news_utils.get_stock_market_sentiment()
            
            # 计算行业整体表现
            top_stocks = industry_stocks.sort_values("涨跌幅", ascending=False).head(5)
            bottom_stocks = industry_stocks.sort_values("涨跌幅").head(5)
            
            # 构建分析请求
            analysis_request = f"""
请分析以下行业数据并给出行业趋势和投资机会分析：

行业名称: {industry_name}
行业代码: {industry_code}
成分股数量: {len(industry_stocks)}

行业整体表现:
平均涨跌幅: {industry_stocks['涨跌幅'].mean():.2f}%
平均市盈率: {industry_stocks['市盈率'].mean():.2f}
平均市净率: {industry_stocks['市净率'].mean():.2f}

表现最好的5只股票:
{top_stocks[['代码', '名称', '最新价', '涨跌幅', '市盈率']].to_string()}

表现最差的5只股票:
{bottom_stocks[['代码', '名称', '最新价', '涨跌幅', '市盈率']].to_string()}

行业相关新闻:
{industry_news[['title', 'content']].head(10).to_string() if not industry_news.empty else "无行业新闻"}

市场情绪指标:
{json.dumps(market_sentiment, ensure_ascii=False, indent=2) if market_sentiment and "error" not in market_sentiment else "无市场情绪数据"}

请基于以上数据，分析该行业的整体趋势和投资机会。你的分析应包括行业概况、行业格局、行业趋势和投资机会四个方面。
"""
            
            # 发起对话
            self.user_proxy.initiate_chat(
                self.industry_analyst,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为分析结果
            analysis = self.user_proxy.chat_messages[self.industry_analyst.name][-1]["content"]
            
            return analysis
        except Exception as e:
            return f"分析行业趋势时出错: {str(e)}"
    
    def batch_analyze(self, industries: List[str], days: int = 30, max_workers: int = 3) -> Dict[str, str]:
        """
        批量分析多个行业
        
        Args:
            industries: 行业名称列表
            days: 分析过去几天的数据和新闻
            max_workers: 最大并行工作线程数
            
        Returns:
            分析结果字典，键为行业名称，值为分析结果
        """
        results = {}
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_industry = {executor.submit(self.analyze_industry, industry, days): industry for industry in industries}
            
            # 获取结果
            for future in concurrent.futures.as_completed(future_to_industry):
                industry = future_to_industry[future]
                try:
                    analysis = future.result()
                    results[industry] = analysis
                except Exception as e:
                    results[industry] = f"分析行业趋势时出错: {str(e)}"
        
        return results
    
    def export_analysis(self, analysis: str, format: str = "markdown", output_file: Optional[str] = None) -> str:
        """
        导出行业分析结果
        
        Args:
            analysis: 行业分析结果
            format: 导出格式，支持 "markdown", "html", "text"
            output_file: 输出文件路径，如果为None则返回字符串
            
        Returns:
            导出的内容或文件路径
        """
        # 获取当前日期
        current_date = get_current_date()
        
        # 根据格式处理内容
        if format == "markdown":
            content = f"# 行业趋势分析报告\n\n**生成日期**: {current_date}\n\n{analysis}"
        elif format == "html":
            content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>行业趋势分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .date {{ color: #666; margin-bottom: 20px; }}
        .analysis {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <h1>行业趋势分析报告</h1>
    <div class="date"><strong>生成日期</strong>: {current_date}</div>
    <div class="analysis">{analysis.replace('\n', '<br>')}</div>
</body>
</html>"""
        else:  # text
            content = f"行业趋势分析报告\n\n生成日期: {current_date}\n\n{analysis}"
        
        # 如果指定了输出文件，则写入文件
        if output_file:
            # 确保文件扩展名与格式匹配
            if format == "markdown" and not output_file.endswith((".md", ".markdown")):
                output_file += ".md"
            elif format == "html" and not output_file.endswith((".html", ".htm")):
                output_file += ".html"
            elif format == "text" and not output_file.endswith((".txt")):
                output_file += ".txt"
            
            # 写入文件
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            return output_file
        
        return content
    
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        self.industry_analyst.reset()

class PortfolioManagerAgent:
    """
    投资组合代理 - 构建和优化投资组合
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化投资组合代理
        
        Args:
            llm_config: LLM配置
        """
        self.llm_config = llm_config
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建投资组合管理代理
        self.portfolio_manager = autogen.AssistantAgent(
            name="PortfolioManager",
            llm_config=llm_config,
            system_message="""你是一位专业的投资组合管理师，擅长构建和优化投资组合。
你需要分析提供的股票数据、行业数据和市场情绪，给出投资组合建议。
你的分析应该包括以下几个方面：
1. 资产配置：不同资产类别（股票、债券、现金等）的配置比例
2. 行业配置：不同行业的配置比例
3. 个股选择：具体的股票选择和权重
4. 风险分析：投资组合的风险特征和分散化程度
5. 预期收益：投资组合的预期收益和风险调整后收益
6. 再平衡策略：投资组合的再平衡频率和触发条件

请确保你的分析逻辑清晰，建议符合投资者的风险偏好和投资目标。"""
        )
    
    def construct_portfolio(self, stocks: List[str], risk_preference: str = "中等", 
                           investment_horizon: str = "长期", 
                           investment_amount: float = 100000) -> str:
        """
        构建投资组合
        
        Args:
            stocks: 股票代码列表
            risk_preference: 风险偏好，可选 "保守", "中等", "激进"
            investment_horizon: 投资期限，可选 "短期", "中期", "长期"
            investment_amount: 投资金额
            
        Returns:
            投资组合建议
        """
        try:
            # 获取股票信息
            stocks_info = {}
            for symbol in stocks:
                stock_info = akshare_utils.get_stock_info(symbol)
                if "error" not in stock_info:
                    stocks_info[symbol] = stock_info
            
            if not stocks_info:
                return "获取股票信息失败，请检查股票代码是否正确"
            
            # 获取股票历史数据
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
            
            stocks_history = {}
            for symbol in stocks:
                history = akshare_utils.get_stock_history(symbol, start_date=start_date, end_date=end_date)
                if not history.empty:
                    stocks_history[symbol] = history
            
            # 计算股票收益率和波动率
            returns_data = {}
            volatility_data = {}
            for symbol, history in stocks_history.items():
                if '收盘' in history.columns and len(history) > 20:
                    # 计算日收益率
                    returns = history['收盘'].pct_change().dropna()
                    # 计算年化收益率
                    annual_return = returns.mean() * 252
                    # 计算年化波动率
                    annual_volatility = returns.std() * (252 ** 0.5)
                    
                    returns_data[symbol] = annual_return
                    volatility_data[symbol] = annual_volatility
            
            # 获取行业信息
            industry_data = {}
            for symbol, info in stocks_info.items():
                industry = info.get("所处行业", "未知")
                if industry not in industry_data:
                    industry_data[industry] = []
                industry_data[industry].append(symbol)
            
            # 获取市场情绪
            market_sentiment = cn_news_utils.get_stock_market_sentiment()
            
            # 构建分析请求
            analysis_request = f"""
请为以下投资需求构建一个投资组合：

投资者信息:
风险偏好: {risk_preference}
投资期限: {investment_horizon}
投资金额: {format_financial_number(investment_amount)}元

可选股票:
"""
            
            # 添加股票信息
            for symbol, info in stocks_info.items():
                stock_name = info.get("股票简称", symbol)
                industry = info.get("所处行业", "未知")
                price = info.get("最新价", "N/A")
                pe = info.get("市盈率(动态)", "N/A")
                pb = info.get("市净率", "N/A")
                
                annual_return = returns_data.get(symbol, "N/A")
                annual_volatility = volatility_data.get(symbol, "N/A")
                
                if annual_return != "N/A":
                    annual_return = f"{annual_return*100:.2f}%"
                if annual_volatility != "N/A":
                    annual_volatility = f"{annual_volatility*100:.2f}%"
                
                analysis_request += f"""
{symbol} ({stock_name}):
  行业: {industry}
  当前价格: {price}
  市盈率: {pe}
  市净率: {pb}
  年化收益率(过去一年): {annual_return}
  年化波动率(过去一年): {annual_volatility}
"""
            
            # 添加行业分布
            analysis_request += "\n行业分布:\n"
            for industry, symbols in industry_data.items():
                analysis_request += f"  {industry}: {len(symbols)}只股票 ({', '.join(symbols)})\n"
            
            # 添加市场情绪
            analysis_request += f"""
市场情绪指标:
{json.dumps(market_sentiment, ensure_ascii=False, indent=2) if market_sentiment and "error" not in market_sentiment else "无市场情绪数据"}

请基于以上数据，构建一个适合该投资者的投资组合。你的建议应包括资产配置、行业配置、个股选择、风险分析、预期收益和再平衡策略。
请给出具体的配置比例和金额，并解释你的投资逻辑。
"""
            
            # 发起对话
            self.user_proxy.initiate_chat(
                self.portfolio_manager,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为建议
            recommendation = self.user_proxy.chat_messages[self.portfolio_manager.name][-1]["content"]
            
            return recommendation
        except Exception as e:
            return f"构建投资组合时出错: {str(e)}"
    
    def optimize_portfolio(self, current_portfolio: Dict[str, float], 
                          risk_preference: str = "中等", 
                          investment_horizon: str = "长期") -> str:
        """
        优化现有投资组合
        
        Args:
            current_portfolio: 当前投资组合，键为股票代码，值为持仓比例
            risk_preference: 风险偏好，可选 "保守", "中等", "激进"
            investment_horizon: 投资期限，可选 "短期", "中期", "长期"
            
        Returns:
            投资组合优化建议
        """
        try:
            stocks = list(current_portfolio.keys())
            
            # 获取股票信息
            stocks_info = {}
            for symbol in stocks:
                stock_info = akshare_utils.get_stock_info(symbol)
                if "error" not in stock_info:
                    stocks_info[symbol] = stock_info
            
            if not stocks_info:
                return "获取股票信息失败，请检查股票代码是否正确"
            
            # 获取股票历史数据
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
            
            stocks_history = {}
            for symbol in stocks:
                history = akshare_utils.get_stock_history(symbol, start_date=start_date, end_date=end_date)
                if not history.empty:
                    stocks_history[symbol] = history
            
            # 计算股票收益率和波动率
            returns_data = {}
            volatility_data = {}
            for symbol, history in stocks_history.items():
                if '收盘' in history.columns and len(history) > 20:
                    # 计算日收益率
                    returns = history['收盘'].pct_change().dropna()
                    # 计算年化收益率
                    annual_return = returns.mean() * 252
                    # 计算年化波动率
                    annual_volatility = returns.std() * (252 ** 0.5)
                    
                    returns_data[symbol] = annual_return
                    volatility_data[symbol] = annual_volatility
            
            # 计算投资组合收益率和波动率
            portfolio_return = sum(returns_data.get(symbol, 0) * weight for symbol, weight in current_portfolio.items())
            
            # 获取市场情绪
            market_sentiment = cn_news_utils.get_stock_market_sentiment()
            
            # 获取行业信息
            industry_data = {}
            for symbol, info in stocks_info.items():
                industry = info.get("所处行业", "未知")
                if industry not in industry_data:
                    industry_data[industry] = []
                industry_data[industry].append(symbol)
            
            # 构建分析请求
            analysis_request = f"""
请优化以下投资组合：

投资者信息:
风险偏好: {risk_preference}
投资期限: {investment_horizon}

当前投资组合:
"""
            
            # 添加当前投资组合信息
            for symbol, weight in current_portfolio.items():
                stock_name = stocks_info.get(symbol, {}).get("股票简称", symbol)
                analysis_request += f"  {symbol} ({stock_name}): {weight*100:.2f}%\n"
            
            # 添加投资组合收益率
            analysis_request += f"\n投资组合年化收益率(估计): {portfolio_return*100:.2f}%\n"
            
            # 添加股票信息
            analysis_request += "\n股票详细信息:\n"
            for symbol, info in stocks_info.items():
                stock_name = info.get("股票简称", symbol)
                industry = info.get("所处行业", "未知")
                price = info.get("最新价", "N/A")
                pe = info.get("市盈率(动态)", "N/A")
                pb = info.get("市净率", "N/A")
                
                annual_return = returns_data.get(symbol, "N/A")
                annual_volatility = volatility_data.get(symbol, "N/A")
                
                if annual_return != "N/A":
                    annual_return = f"{annual_return*100:.2f}%"
                if annual_volatility != "N/A":
                    annual_volatility = f"{annual_volatility*100:.2f}%"
                
                analysis_request += f"""
{symbol} ({stock_name}):
  行业: {industry}
  当前价格: {price}
  市盈率: {pe}
  市净率: {pb}
  年化收益率(过去一年): {annual_return}
  年化波动率(过去一年): {annual_volatility}
  当前权重: {current_portfolio.get(symbol, 0)*100:.2f}%
"""
            
            # 添加行业分布
            analysis_request += "\n行业分布:\n"
            for industry, symbols in industry_data.items():
                industry_weight = sum(current_portfolio.get(symbol, 0) for symbol in symbols)
                analysis_request += f"  {industry}: {industry_weight*100:.2f}% ({', '.join(symbols)})\n"
            
            # 添加市场情绪
            analysis_request += f"""
市场情绪指标:
{json.dumps(market_sentiment, ensure_ascii=False, indent=2) if market_sentiment and "error" not in market_sentiment else "无市场情绪数据"}

请基于以上数据，优化该投资组合。你的建议应包括调整后的资产配置、行业配置、个股选择、风险分析、预期收益和再平衡策略。
请给出具体的调整建议，包括应该增加或减少哪些股票的权重，以及调整的理由。
"""
            
            # 发起对话
            self.user_proxy.initiate_chat(
                self.portfolio_manager,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为建议
            recommendation = self.user_proxy.chat_messages[self.portfolio_manager.name][-1]["content"]
            
            return recommendation
        except Exception as e:
            return f"优化投资组合时出错: {str(e)}"
    
    def export_recommendation(self, recommendation: str, format: str = "markdown", output_file: Optional[str] = None) -> str:
        """
        导出投资组合建议
        
        Args:
            recommendation: 投资组合建议
            format: 导出格式，支持 "markdown", "html", "text"
            output_file: 输出文件路径，如果为None则返回字符串
            
        Returns:
            导出的内容或文件路径
        """
        # 获取当前日期
        current_date = get_current_date()
        
        # 根据格式处理内容
        if format == "markdown":
            content = f"# 投资组合建议\n\n**生成日期**: {current_date}\n\n{recommendation}"
        elif format == "html":
            content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>投资组合建议</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .date {{ color: #666; margin-bottom: 20px; }}
        .recommendation {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <h1>投资组合建议</h1>
    <div class="date"><strong>生成日期</strong>: {current_date}</div>
    <div class="recommendation">{recommendation.replace('\n', '<br>')}</div>
</body>
</html>"""
        else:  # text
            content = f"投资组合建议\n\n生成日期: {current_date}\n\n{recommendation}"
        
        # 如果指定了输出文件，则写入文件
        if output_file:
            # 确保文件扩展名与格式匹配
            if format == "markdown" and not output_file.endswith((".md", ".markdown")):
                output_file += ".md"
            elif format == "html" and not output_file.endswith((".html", ".htm")):
                output_file += ".html"
            elif format == "text" and not output_file.endswith((".txt")):
                output_file += ".txt"
            
            # 写入文件
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            return output_file
        
        return content
    
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        self.portfolio_manager.reset()

class TechnicalAnalysisAgent:
    """
    技术分析代理 - 进行股票技术分析
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        初始化技术分析代理
        
        Args:
            llm_config: LLM配置
        """
        self.llm_config = llm_config
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建技术分析师代理
        self.technical_analyst = autogen.AssistantAgent(
            name="TechnicalAnalyst",
            llm_config=llm_config,
            system_message="""你是一位专业的技术分析师，擅长使用技术指标和图表模式分析股票走势。
你需要分析提供的股票价格数据和技术指标，给出技术分析结论和交易建议。
你的分析应该包括以下几个方面：
1. 趋势分析：判断股票的主要趋势（上升、下降或横盘）
2. 支撑位和阻力位：识别重要的价格水平
3. 技术指标分析：分析各种技术指标的信号（如MACD、RSI、KDJ等）
4. 图表形态：识别关键的图表形态（如头肩顶、双底等）
5. 交易信号：给出明确的买入、卖出或观望建议
6. 止损和目标价位：建议的止损位和目标价位

请确保你的分析逻辑清晰，建议具有可操作性。"""
        )
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 股票历史数据DataFrame，必须包含'收盘'列
            
        Returns:
            添加了技术指标的DataFrame
        """
        if df.empty or '收盘' not in df.columns:
            return df
        
        # 计算移动平均线
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['MA60'] = df['收盘'].rolling(window=60).mean()
        
        # 计算MACD
        df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])
        
        # 计算RSI
        delta = df['收盘'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算KDJ
        low_min = df['最低'].rolling(window=9).min()
        high_max = df['最高'].rolling(window=9).max()
        df['RSV'] = 100 * ((df['收盘'] - low_min) / (high_max - low_min))
        df['K'] = df['RSV'].ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        # 计算布林带
        df['BOLL_MIDDLE'] = df['收盘'].rolling(window=20).mean()
        df['BOLL_STD'] = df['收盘'].rolling(window=20).std()
        df['BOLL_UPPER'] = df['BOLL_MIDDLE'] + 2 * df['BOLL_STD']
        df['BOLL_LOWER'] = df['BOLL_MIDDLE'] - 2 * df['BOLL_STD']
        
        return df
    
    @cached("technical_analysis", expire_seconds=3600)
    def analyze(self, symbol: str, period: str = "daily", days: int = 120) -> str:
        """
        进行技术分析
        
        Args:
            symbol: 股票代码（如：000001）
            period: 时间周期，可选 daily, weekly, monthly
            days: 分析过去几天的数据
            
        Returns:
            技术分析结果
        """
        try:
            # 获取股票信息
            stock_info = akshare_utils.get_stock_info(symbol)
            if "error" in stock_info:
                return f"获取股票信息失败: {stock_info['error']}"
                
            # 获取股票历史数据
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
            stock_history = akshare_utils.get_stock_history(symbol, period=period, start_date=start_date, end_date=end_date)
            
            if stock_history.empty:
                return f"获取股票历史数据失败: {symbol}"
            
            # 计算技术指标
            stock_history = self.calculate_technical_indicators(stock_history)
            
            # 获取最近的技术指标数据
            recent_data = stock_history.tail(30)
            
            # 构建分析请求
            analysis_request = f"""
请对以下股票进行技术分析：

股票代码: {symbol}
股票名称: {stock_info.get('股票简称', 'N/A')}
当前价格: {stock_info.get('最新价', 'N/A')}
时间周期: {period}

最近30个交易日的价格和技术指标:
{recent_data[['收盘', '成交量', 'MA5', 'MA20', 'RSI', 'K', 'D', 'J', 'DIF', 'DEA', 'MACD']].tail(10).to_string()}

技术指标说明:
- MA5/MA20: 5日/20日移动平均线
- RSI: 相对强弱指标
- KDJ: 随机指标
- MACD: 指数平滑异同移动平均线

请基于以上数据，进行全面的技术分析。你的分析应包括趋势分析、支撑位和阻力位、技术指标分析、图表形态、交易信号以及止损和目标价位。
"""
            
            # 发起对话
            self.user_proxy.initiate_chat(
                self.technical_analyst,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为分析结果
            analysis = self.user_proxy.chat_messages[self.technical_analyst.name][-1]["content"]
            
            return analysis
        except Exception as e:
            return f"进行技术分析时出错: {str(e)}"
    
    def batch_analyze(self, symbols: List[str], period: str = "daily", days: int = 120, max_workers: int = 3) -> Dict[str, str]:
        """
        批量进行技术分析
        
        Args:
            symbols: 股票代码列表
            period: 时间周期
            days: 分析过去几天的数据
            max_workers: 最大并行工作线程数
            
        Returns:
            分析结果字典，键为股票代码，值为分析结果
        """
        results = {}
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_symbol = {executor.submit(self.analyze, symbol, period, days): symbol for symbol in symbols}
            
            # 获取结果
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    analysis = future.result()
                    results[symbol] = analysis
                except Exception as e:
                    results[symbol] = f"进行技术分析时出错: {str(e)}"
        
        return results
    
    def export_analysis(self, analysis: str, format: str = "markdown", output_file: Optional[str] = None) -> str:
        """
        导出技术分析结果
        
        Args:
            analysis: 技术分析结果
            format: 导出格式，支持 "markdown", "html", "text"
            output_file: 输出文件路径，如果为None则返回字符串
            
        Returns:
            导出的内容或文件路径
        """
        # 获取当前日期
        current_date = get_current_date()
        
        # 根据格式处理内容
        if format == "markdown":
            content = f"# 股票技术分析报告\n\n**生成日期**: {current_date}\n\n{analysis}"
        elif format == "html":
            content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>股票技术分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .date {{ color: #666; margin-bottom: 20px; }}
        .analysis {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <h1>股票技术分析报告</h1>
    <div class="date"><strong>生成日期</strong>: {current_date}</div>
    <div class="analysis">{analysis.replace('\n', '<br>')}</div>
</body>
</html>"""
        else:  # text
            content = f"股票技术分析报告\n\n生成日期: {current_date}\n\n{analysis}"
        
        # 如果指定了输出文件，则写入文件
        if output_file:
            # 确保文件扩展名与格式匹配
            if format == "markdown" and not output_file.endswith((".md", ".markdown")):
                output_file += ".md"
            elif format == "html" and not output_file.endswith((".html", ".htm")):
                output_file += ".html"
            elif format == "text" and not output_file.endswith((".txt")):
                output_file += ".txt"
            
            # 写入文件
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            return output_file
        
        return content
    
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        self.technical_analyst.reset() 