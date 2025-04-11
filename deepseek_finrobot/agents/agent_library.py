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
from ..utils import get_current_date, format_financial_number, cached, get_llm_config_for_autogen

# 添加行业映射字典，用于在行业数据缺失时提供行业信息
INDUSTRY_MAPPING = {
    # 银行
    "000001": "银行", "601398": "银行", "601288": "银行", "601939": "银行", "600036": "银行",
    # 白酒
    "600519": "白酒", "000858": "白酒", "002304": "白酒", "000568": "白酒",
    # 科技
    "000063": "科技", "000977": "科技", "002415": "科技", "600570": "科技", "600588": "科技",
    # 医疗健康
    "600276": "医药", "000538": "医药", "600196": "医药", "000661": "医药", "300015": "医药",
}

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
{json.dumps(market_sentiment, ensure_ascii=False, indent=2) if market_sentiment and isinstance(market_sentiment, dict) and "error" not in market_sentiment else "无市场情绪数据"}

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
        try:
            # 获取当前日期
            current_date = get_current_date()
            
            # 根据格式处理内容
            if format == "markdown":
                content = f"# 股票走势预测报告\n\n**生成日期**: {current_date}\n\n{prediction}"
            elif format == "html":
                content = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>股票走势预测报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .date { color: #666; margin-bottom: 20px; }
        .prediction { line-height: 1.6; }
    </style>
</head>
<body>
    <h1>股票走势预测报告</h1>
    <div class="date"><strong>生成日期</strong>: """ + current_date + """</div>
    <div class="prediction">""" + prediction.replace('\n', '<br>') + """</div>
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
                try:
                    with open(output_file, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(content)
                except UnicodeEncodeError:
                    # 如果遇到编码错误，尝试使用其他编码或回退处理
                    with open(output_file, "w", encoding="utf-8", errors="backslashreplace") as f:
                        f.write(content)
                
                return output_file
            
            return content
        except Exception as e:
            print(f"导出预测结果时出错: {e}")
            # 确保函数总是返回一个字符串
            return f"导出出错: {str(e)}"
    
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
        # 构建兼容两个版本API的llm_config
        self.llm_config = get_llm_config_for_autogen(**llm_config)
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建财务分析师代理
        self.financial_analyst = autogen.AssistantAgent(
            name="FinancialAnalyst",
            llm_config=self.llm_config,
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
        # 构建兼容两个版本API的llm_config
        self.llm_config = get_llm_config_for_autogen(**llm_config)
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建新闻分析师代理
        self.news_analyst = autogen.AssistantAgent(
            name="NewsAnalyst",
            llm_config=self.llm_config,
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
{json.dumps(market_sentiment, ensure_ascii=False, indent=2) if market_sentiment and isinstance(market_sentiment, dict) and "error" not in market_sentiment else "无市场情绪数据"}

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
        # 构建兼容两个版本API的llm_config
        self.llm_config = get_llm_config_for_autogen(**llm_config)
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建行业分析师代理
        self.industry_analyst = autogen.AssistantAgent(
            name="IndustryAnalyst",
            llm_config=self.llm_config,
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
            
            # 确保必要的列存在
            required_columns = ['市盈率', '市净率']
            for col in required_columns:
                if col not in industry_stocks.columns:
                    industry_stocks[col] = 0.0
                    print(f"行业成分股数据中缺少 {col} 列，已添加默认值")
            
            # 计算行业整体表现
            top_stocks = industry_stocks.sort_values("涨跌幅", ascending=False).head(5)
            bottom_stocks = industry_stocks.sort_values("涨跌幅").head(5)
            
            # 确保显示的股票数据中包含所有必要的列
            display_columns = ['代码', '名称', '最新价', '涨跌幅', '市盈率']
            for col in display_columns:
                if col not in top_stocks.columns:
                    if col == '市盈率':
                        top_stocks[col] = 0.0
                        bottom_stocks[col] = 0.0
                    else:
                        top_stocks[col] = f'未提供{col}'
                        bottom_stocks[col] = f'未提供{col}'
            
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
{top_stocks[display_columns].to_string()}

表现最差的5只股票:
{bottom_stocks[display_columns].to_string()}

行业相关新闻:
{industry_news[['title', 'content']].head(10).to_string() if not industry_news.empty else "无行业新闻"}

市场情绪指标:
{json.dumps(market_sentiment, ensure_ascii=False, indent=2) if market_sentiment and isinstance(market_sentiment, dict) and "error" not in market_sentiment else "无市场情绪数据"}

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
            content = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>行业趋势分析报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .date { color: #666; margin-bottom: 20px; }
        .analysis { line-height: 1.6; }
    </style>
</head>
<body>
    <h1>行业趋势分析报告</h1>
    <div class="date"><strong>生成日期</strong>: """ + current_date + """</div>
    <div class="analysis">""" + analysis.replace('\n', '<br>') + """</div>
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
        # 构建兼容两个版本API的llm_config
        self.llm_config = get_llm_config_for_autogen(**llm_config)
        
        # 创建用户代理
        self.user = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建投资组合管理代理
        self.portfolio_manager = autogen.AssistantAgent(
            name="PortfolioManager",
            llm_config=self.llm_config,
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
        
        # 行业适配性字典 - 根据不同风险偏好和市场环境调整权重
        self.industry_adaptability = {
            # 保守型投资者的行业适应性
            "保守": {
                "积极市场": {
                    "银行": 0.8,      # 保守型投资者在积极市场中仍然偏好银行股
                    "必需消费": 1.0,   # 必需消费品在任何市场环境都适合保守型投资者
                    "医药": 0.9,      # 医药股通常波动较小，适合保守型投资者
                    "公用事业": 1.0,   # 公用事业稳定性高
                    "白酒": 0.6,      # 白酒股波动较大，保守型投资者即使在积极市场也应谨慎
                    "科技": 0.4,      # 科技股波动大，保守型投资者配置有限
                    "新能源": 0.5,    # 新能源波动较大但有政策支持
                },
                "中性市场": {
                    "银行": 0.9,
                    "必需消费": 1.0,
                    "医药": 0.8,
                    "公用事业": 1.0,
                    "白酒": 0.5,
                    "科技": 0.3,
                    "新能源": 0.4,
                },
                "消极市场": {
                    "银行": 1.0,      # 在消极市场中银行股是最佳避风港之一
                    "必需消费": 1.0,
                    "医药": 0.7,
                    "公用事业": 1.0,
                    "白酒": 0.3,      # 消极市场中应显著降低白酒配置
                    "科技": 0.2,      # 消极市场中应最小化科技配置
                    "新能源": 0.3,
                }
            },
            # 中等风险偏好投资者的行业适应性
            "中等": {
                "积极市场": {
                    "银行": 0.7,
                    "必需消费": 0.8,
                    "医药": 0.9,
                    "公用事业": 0.7,
                    "白酒": 0.9,      # 中等风险投资者在积极市场可以适当增加白酒配置
                    "科技": 0.8,      # 中等风险投资者在积极市场可以适当增加科技配置
                    "新能源": 0.9,    # 新能源在积极市场中适合中等风险投资者
                },
                "中性市场": {
                    "银行": 0.8,
                    "必需消费": 0.8,
                    "医药": 0.8,
                    "公用事业": 0.8,
                    "白酒": 0.7,
                    "科技": 0.7,
                    "新能源": 0.7,
                },
                "消极市场": {
                    "银行": 0.9,
                    "必需消费": 0.9,
                    "医药": 0.7,
                    "公用事业": 0.9,
                    "白酒": 0.5,
                    "科技": 0.5,
                    "新能源": 0.5,
                }
            },
            # 激进型投资者的行业适应性
            "激进": {
                "积极市场": {
                    "银行": 0.5,      # 积极市场中，激进投资者对银行等传统行业配置较少
                    "必需消费": 0.5,
                    "医药": 0.8,
                    "公用事业": 0.4,
                    "白酒": 1.0,      # 激进投资者在积极市场中可以满配白酒股
                    "科技": 1.0,      # 激进投资者在积极市场中可以满配科技股
                    "新能源": 1.0,    # 新能源在积极市场有很高增长潜力
                },
                "中性市场": {
                    "银行": 0.6,
                    "必需消费": 0.6,
                    "医药": 0.7,
                    "公用事业": 0.5,
                    "白酒": 0.8,
                    "科技": 0.8,
                    "新能源": 0.8,
                },
                "消极市场": {
                    "银行": 0.7,      # 即使是激进投资者在消极市场也应增加银行等防御性行业配置
                    "必需消费": 0.7,
                    "医药": 0.6,
                    "公用事业": 0.7,
                    "白酒": 0.6,
                    "科技": 0.7,      # 激进投资者在消极市场也可以寻找科技股的逢低机会
                    "新能源": 0.6,
                }
            }
        }
        
        # 投资期限对行业的适配性
        self.horizon_industry_fit = {
            "短期": {
                "银行": 0.8,      # 短期内银行股较稳定
                "必需消费": 0.9,   # 必需消费短期波动小
                "医药": 0.7,      
                "公用事业": 0.8,   
                "白酒": 0.5,      # 短期内白酒股波动可能较大
                "科技": 0.5,      # 短期内科技股波动较大
                "新能源": 0.6,    
            },
            "中期": {
                "银行": 0.7,
                "必需消费": 0.8,
                "医药": 0.8,
                "公用事业": 0.7,
                "白酒": 0.7,
                "科技": 0.8,      # 中期内科技股增长潜力较好
                "新能源": 0.8,    
            },
            "长期": {
                "银行": 0.6,
                "必需消费": 0.7,
                "医药": 0.9,      # 长期内医药行业有良好增长性
                "公用事业": 0.6,
                "白酒": 0.8,      # 长期内白酒有品牌价值和增长潜力
                "科技": 0.9,      # 长期内科技行业增长潜力最大
                "新能源": 0.9,    # 长期内新能源是重要发展方向
            }
        }

    def get_sector_adjustment(self, risk_preference: str, market_sentiment: Dict[str, Any], investment_horizon: str) -> Dict[str, float]:
        """
        根据风险偏好、市场情绪和投资期限获取行业调整系数
        
        Args:
            risk_preference: 风险偏好，可选 "保守", "中等", "激进"
            market_sentiment: 市场情绪数据
            investment_horizon: 投资期限，可选 "短期", "中期", "长期"
            
        Returns:
            行业调整系数字典
        """
        # 默认为中性市场
        market_condition = "中性市场"
        
        # 根据市场情绪确定市场状况
        if market_sentiment and "error" not in market_sentiment:
            if "market_trend" in market_sentiment:
                sentiment = market_sentiment["market_trend"].get("sentiment", "中性")
                trend = market_sentiment["market_trend"].get("trend", "未知")
                
                if sentiment == "积极" or (sentiment == "中性" and trend == "上涨"):
                    market_condition = "积极市场"
                elif sentiment == "消极" or (sentiment == "中性" and trend == "下跌"):
                    market_condition = "消极市场"
        
        # 获取风险偏好下的市场环境调整系数
        risk_adjustments = self.industry_adaptability.get(risk_preference, self.industry_adaptability["中等"])
        market_adjustments = risk_adjustments.get(market_condition, risk_adjustments["中性市场"])
        
        # 获取投资期限的调整系数
        horizon_adjustments = self.horizon_industry_fit.get(investment_horizon, self.horizon_industry_fit["中期"])
        
        # 合并两个调整系数
        final_adjustments = {}
        all_industries = set(list(market_adjustments.keys()) + list(horizon_adjustments.keys()))
        
        for industry in all_industries:
            market_adj = market_adjustments.get(industry, 0.7)  # 默认为0.7
            horizon_adj = horizon_adjustments.get(industry, 0.7)  # 默认为0.7
            
            # 取两个调整系数的加权平均，市场环境权重0.6，投资期限权重0.4
            final_adjustments[industry] = market_adj * 0.6 + horizon_adj * 0.4
        
        return final_adjustments

    def _get_industry_info(self, symbol: str, stock_info: Dict[str, Any]) -> str:
        """
        获取股票行业信息，如果API数据中没有，则从映射表中获取
        
        Args:
            symbol: 股票代码
            stock_info: 股票信息字典
            
        Returns:
            行业名称
        """
        # 尝试从股票信息中获取行业
        industry = stock_info.get("所处行业")
        
        # 如果行业为空或未知，尝试从映射表获取
        if not industry or industry == "N/A" or industry == "未知":
            industry = INDUSTRY_MAPPING.get(symbol, "未知")
            
        return industry

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
            print("\n开始获取股票信息...")
            for symbol in stocks:
                print(f"获取股票 {symbol} 信息...")
                stock_info = akshare_utils.get_stock_info(symbol)
                if "error" in stock_info:
                    print(f"股票 {symbol} 信息获取失败: {stock_info['error']}")
                else:
                    print(f"股票 {symbol} 信息获取成功: {stock_info.get('股票简称', symbol)}")
                    stocks_info[symbol] = stock_info
            
            if not stocks_info:
                return "获取股票信息失败，请检查股票代码是否正确"
            
            # 获取股票历史数据
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
            
            stocks_history = {}
            print("\n开始获取股票历史数据...")
            for symbol in stocks:
                print(f"获取股票 {symbol} 历史数据...")
                history = akshare_utils.get_stock_history(symbol, start_date=start_date, end_date=end_date)
                if history.empty:
                    print(f"股票 {symbol} 历史数据获取失败或为空")
                else:
                    print(f"股票 {symbol} 历史数据获取成功，共 {len(history)} 条记录")
                    stocks_history[symbol] = history
            
            # 计算股票收益率和波动率
            returns_data = {}
            volatility_data = {}
            print("\n开始计算股票收益率和波动率...")
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
                    print(f"股票 {symbol} 年化收益率: {annual_return*100:.2f}%, 年化波动率: {annual_volatility*100:.2f}%")
                else:
                    print(f"股票 {symbol} 历史数据不足或没有'收盘'列，无法计算收益率和波动率")
            
            # 获取行业信息
            industry_data = {}
            print("\n整理行业分布...")
            for symbol, info in stocks_info.items():
                # 使用增强的行业获取方法
                industry = self._get_industry_info(symbol, info)
                
                if industry not in industry_data:
                    industry_data[industry] = []
                industry_data[industry].append(symbol)
                print(f"股票 {symbol} 所属行业: {industry}")
            
            # 获取市场情绪
            print("\n获取市场情绪...")
            market_sentiment = cn_news_utils.get_stock_market_sentiment()
            if market_sentiment and "error" not in market_sentiment:
                print("市场情绪获取成功")
            else:
                print("市场情绪获取失败或为空")
                
            # 计算行业调整系数
            print("\n计算行业调整系数...")
            sector_adjustments = self.get_sector_adjustment(risk_preference, market_sentiment, investment_horizon)
            for industry, adjustment in sector_adjustments.items():
                if industry in industry_data:
                    print(f"行业 {industry} 的调整系数: {adjustment:.2f}")
            
            # 构建分析请求
            analysis_request = f"""
请为以下投资需求构建一个投资组合：

投资者信息:
风险偏好: {risk_preference}
投资期限: {investment_horizon}
投资金额: {format_financial_number(investment_amount)}元

风险偏好说明:
- 保守: 追求资本保值，对波动性极为敏感，可接受较低回报以换取更高安全性
- 中等: 寻求资本增值与风险平衡，可接受适度波动以获取合理回报
- 激进: 追求资本高增长，能够承受较大波动，追求高回报

投资期限说明:
- 短期: 1年以内，更关注短期波动和流动性
- 中期: 1-3年，关注中期价值增长和合理风险
- 长期: 3年以上，关注长期价值增长，可以承受短期波动

可选股票:
"""
            
            # 添加股票信息
            for symbol, info in stocks_info.items():
                stock_name = info.get("股票简称", symbol)
                # 使用增强的行业获取方法
                industry = self._get_industry_info(symbol, info)
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
                
            # 添加行业调整系数
            analysis_request += "\n行业适配性评分 (0-1，越高代表越适合当前组合):\n"
            for industry in industry_data.keys():
                adjustment = sector_adjustments.get(industry, 0.7)  # 默认为0.7
                adjustment_desc = "非常适合" if adjustment >= 0.9 else (
                                "适合" if adjustment >= 0.7 else (
                                "中性" if adjustment >= 0.5 else (
                                "较不适合" if adjustment >= 0.3 else "不适合")))
                analysis_request += f"  {industry}: {adjustment:.2f} ({adjustment_desc})\n"
            
            # 添加市场情绪和市场环境建议
            analysis_request += "\n市场情绪指标:\n"
            if market_sentiment and "error" not in market_sentiment:
                # 检查是否是合成数据
                is_synthetic = any(isinstance(v, dict) and v.get("is_synthetic", False) for v in market_sentiment.values())
                
                if is_synthetic:
                    analysis_request += "注意: 由于实时数据无法获取，以下是合成的市场情绪数据，仅供参考。\n"
                
                # 提取最关键的市场情绪数据
                if "market_trend" in market_sentiment:
                    trend = market_sentiment["market_trend"]
                    analysis_request += f"上证指数最近趋势: {trend.get('trend', 'N/A')}, 变化: {trend.get('recent_change', 'N/A')}, 情绪: {trend.get('sentiment', 'N/A')}\n"
                
                if "north_flow" in market_sentiment:
                    flow = market_sentiment["north_flow"]
                    analysis_request += f"北向资金: {flow.get('direction', 'N/A')}, 数值: {flow.get('value', 'N/A')}\n"
                
                if "market_activity" in market_sentiment:
                    activity = market_sentiment["market_activity"]
                    if "activity" in activity:
                        analysis_request += f"市场活跃度: {activity.get('activity', 'N/A')}\n"
                    elif "total_volume" in activity:
                        analysis_request += f"市场成交量: {activity.get('total_volume', 'N/A')}, 成交额: {activity.get('total_amount', 'N/A')}\n"
                    else:
                        analysis_request += "市场活跃度数据可用\n"
                
                # 添加市场环境评估
                market_sentiment_value = "中性"
                if "market_trend" in market_sentiment:
                    sentiment = market_sentiment["market_trend"].get("sentiment", "中性")
                    trend_value = market_sentiment["market_trend"].get("trend", "未知")
                    market_sentiment_value = sentiment
                    
                # 添加基于市场情绪的投资建议
                analysis_request += "\n基于当前市场环境的建议:\n"
                if market_sentiment_value == "积极" or (market_sentiment_value == "中性" and trend_value == "上涨"):
                    if risk_preference == "保守":
                        analysis_request += "当前市场情绪积极，但考虑到投资者风险偏好保守，建议仍以稳健型股票为主，可适当增加配置比例，但保持谨慎。\n"
                    elif risk_preference == "中等":
                        analysis_request += "当前市场情绪积极，对于中等风险偏好投资者，可以适度增加成长型股票配置，同时保留部分防御性资产。\n"
                    else:  # 激进
                        analysis_request += "当前市场情绪积极，对于激进型投资者，可以显著增加成长型股票占比，降低现金持有，关注高增长板块。\n"
                elif market_sentiment_value == "消极" or (market_sentiment_value == "中性" and trend_value == "下跌"):
                    if risk_preference == "保守":
                        analysis_request += "当前市场情绪偏消极，考虑到投资者风险偏好保守，建议增加防御性资产占比，关注高股息蓝筹和必需消费品类股票。\n"
                    elif risk_preference == "中等":
                        analysis_request += "当前市场情绪偏消极，对于中等风险偏好投资者，建议采取相对均衡的配置，增加优质蓝筹比例，减少高波动性股票占比。\n"
                    else:  # 激进
                        analysis_request += "当前市场情绪偏消极，但对于激进型投资者，可以视为逢低布局机会，关注优质但已显著回调的成长股，同时保留部分现金应对可能的进一步波动。\n"
                else:  # 中性
                    if risk_preference == "保守":
                        analysis_request += "当前市场情绪中性，对于保守型投资者，建议采取均衡配置，侧重蓝筹股和稳健型行业，关注股息回报。\n"
                    elif risk_preference == "中等":
                        analysis_request += "当前市场情绪中性，对于中等风险偏好投资者，建议配置均衡组合，结合价值型和成长型股票，关注行业龙头。\n"
                    else:  # 激进
                        analysis_request += "当前市场情绪中性，对于激进型投资者，可以适度增加高成长性行业占比，但同时保留部分价值型股票以分散风险。\n"
                
                # 如果有其他市场指标，也添加到请求中
                for key, value in market_sentiment.items():
                    if key not in ["market_trend", "north_flow", "market_activity", "error", "data_source", "note", "message"]:
                        analysis_request += f"{key}: {value}\n"
            else:
                analysis_request += "当前无法获取市场情绪数据，请基于其他因素进行分析。\n"
                
            # 添加投资期限相关建议
            analysis_request += "\n基于投资期限的建议:\n"
            if investment_horizon == "短期":
                analysis_request += "对于短期投资（1年以内），建议关注流动性较好、波动性较低的股票，可以考虑蓝筹股、必需消费品等防御性板块。避免过度集中在单一行业或个股。\n"
            elif investment_horizon == "中期":
                analysis_request += "对于中期投资（1-3年），可以平衡配置价值型和成长型股票，关注具有持续竞争优势的行业龙头企业和合理估值的成长股。\n"
            else:  # 长期
                analysis_request += "对于长期投资（3年以上），可以增加优质成长股比例，关注具有长期竞争优势和行业格局改善的企业，可以承受短期波动以换取长期增值。\n"

            analysis_request += """
请基于以上数据和指导，构建一个特别适合该投资者风险偏好和当前市场环境的投资组合。你的建议应包括:

1. 资产配置：具体的股票配置比例和金额
2. 行业配置：不同行业的配置比例和理由
3. 个股选择：每只推荐股票的具体配置理由
4. 风险分析：评估组合整体风险水平和分散化程度
5. 预期收益：基于历史数据的组合预期收益率
6. 再平衡策略：何时以及如何调整投资组合

请确保你的建议与投资者的风险偏好和投资期限紧密匹配，并根据当前市场环境进行适当调整。
"""
            
            # 发起对话
            self.user.initiate_chat(
                self.portfolio_manager,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为建议
            recommendation = self.user.chat_messages[self.portfolio_manager.name][-1]["content"]
            
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
                # 使用增强的行业获取方法
                industry = self._get_industry_info(symbol, info)
                
                if industry not in industry_data:
                    industry_data[industry] = []
                industry_data[industry].append(symbol)
                
            # 计算行业调整系数
            sector_adjustments = self.get_sector_adjustment(risk_preference, market_sentiment, investment_horizon)
            
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
                # 使用增强的行业获取方法
                industry = self._get_industry_info(symbol, info)
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
                
            # 添加行业调整系数
            analysis_request += "\n行业适配性评分 (0-1，越高代表越适合当前组合):\n"
            for industry in industry_data.keys():
                adjustment = sector_adjustments.get(industry, 0.7)  # 默认为0.7
                adjustment_desc = "非常适合" if adjustment >= 0.9 else (
                                "适合" if adjustment >= 0.7 else (
                                "中性" if adjustment >= 0.5 else (
                                "较不适合" if adjustment >= 0.3 else "不适合")))
                industry_weight = sum(current_portfolio.get(symbol, 0) for symbol in industry_data[industry])
                
                # 计算调整建议
                ideal_direction = ""
                if adjustment >= 0.8 and industry_weight < 0.2:
                    ideal_direction = "【建议增加】"
                elif adjustment <= 0.4 and industry_weight > 0.1:
                    ideal_direction = "【建议减少】"
                elif adjustment >= 0.6 and industry_weight < 0.1:
                    ideal_direction = "【可适度增加】"
                elif adjustment <= 0.5 and industry_weight > 0.2:
                    ideal_direction = "【可适度减少】"
                
                analysis_request += f"  {industry}: {adjustment:.2f} ({adjustment_desc}) - 当前配置: {industry_weight*100:.2f}% {ideal_direction}\n"
            
            # 添加市场情绪和市场环境建议
            analysis_request += "\n市场情绪指标:\n"
            if market_sentiment and "error" not in market_sentiment:
                # 检查是否是合成数据
                is_synthetic = any(isinstance(v, dict) and v.get("is_synthetic", False) for v in market_sentiment.values())
                
                if is_synthetic:
                    analysis_request += "注意: 由于实时数据无法获取，以下是合成的市场情绪数据，仅供参考。\n"
                
                # 提取最关键的市场情绪数据
                if "market_trend" in market_sentiment:
                    trend = market_sentiment["market_trend"]
                    analysis_request += f"上证指数最近趋势: {trend.get('trend', 'N/A')}, 变化: {trend.get('recent_change', 'N/A')}, 情绪: {trend.get('sentiment', 'N/A')}\n"
                
                if "north_flow" in market_sentiment:
                    flow = market_sentiment["north_flow"]
                    analysis_request += f"北向资金: {flow.get('direction', 'N/A')}, 数值: {flow.get('value', 'N/A')}\n"
                
                if "market_activity" in market_sentiment:
                    activity = market_sentiment["market_activity"]
                    if "activity" in activity:
                        analysis_request += f"市场活跃度: {activity.get('activity', 'N/A')}\n"
                    elif "total_volume" in activity:
                        analysis_request += f"市场成交量: {activity.get('total_volume', 'N/A')}, 成交额: {activity.get('total_amount', 'N/A')}\n"
                    else:
                        analysis_request += "市场活跃度数据可用\n"
                
                # 添加市场环境评估
                market_sentiment_value = "中性"
                if "market_trend" in market_sentiment:
                    sentiment = market_sentiment["market_trend"].get("sentiment", "中性")
                    trend_value = market_sentiment["market_trend"].get("trend", "未知")
                    market_sentiment_value = sentiment
                    
                # 添加基于市场情绪的投资建议
                analysis_request += "\n基于当前市场环境的建议:\n"
                if market_sentiment_value == "积极" or (market_sentiment_value == "中性" and trend_value == "上涨"):
                    if risk_preference == "保守":
                        analysis_request += "当前市场情绪积极，但考虑到投资者风险偏好保守，建议仍以稳健型股票为主，可适当增加配置比例，但保持谨慎。\n"
                    elif risk_preference == "中等":
                        analysis_request += "当前市场情绪积极，对于中等风险偏好投资者，可以适度增加成长型股票配置，同时保留部分防御性资产。\n"
                    else:  # 激进
                        analysis_request += "当前市场情绪积极，对于激进型投资者，可以显著增加成长型股票占比，降低现金持有，关注高增长板块。\n"
                elif market_sentiment_value == "消极" or (market_sentiment_value == "中性" and trend_value == "下跌"):
                    if risk_preference == "保守":
                        analysis_request += "当前市场情绪偏消极，考虑到投资者风险偏好保守，建议增加防御性资产占比，关注高股息蓝筹和必需消费品类股票。\n"
                    elif risk_preference == "中等":
                        analysis_request += "当前市场情绪偏消极，对于中等风险偏好投资者，建议采取相对均衡的配置，增加优质蓝筹比例，减少高波动性股票占比。\n"
                    else:  # 激进
                        analysis_request += "当前市场情绪偏消极，但对于激进型投资者，可以视为逢低布局机会，关注优质但已显著回调的成长股，同时保留部分现金应对可能的进一步波动。\n"
                else:  # 中性
                    if risk_preference == "保守":
                        analysis_request += "当前市场情绪中性，对于保守型投资者，建议采取均衡配置，侧重蓝筹股和稳健型行业，关注股息回报。\n"
                    elif risk_preference == "中等":
                        analysis_request += "当前市场情绪中性，对于中等风险偏好投资者，建议配置均衡组合，结合价值型和成长型股票，关注行业龙头。\n"
                    else:  # 激进
                        analysis_request += "当前市场情绪中性，对于激进型投资者，可以适度增加高成长性行业占比，但同时保留部分价值型股票以分散风险。\n"
                
                # 如果有其他市场指标，也添加到请求中
                for key, value in market_sentiment.items():
                    if key not in ["market_trend", "north_flow", "market_activity", "error", "data_source", "note", "message"]:
                        analysis_request += f"{key}: {value}\n"
            else:
                analysis_request += "当前无法获取市场情绪数据，请基于其他因素进行分析。\n"
                
            # 添加投资期限相关建议
            analysis_request += "\n基于投资期限的建议:\n"
            if investment_horizon == "短期":
                analysis_request += "对于短期投资（1年以内），建议关注流动性较好、波动性较低的股票，可以考虑蓝筹股、必需消费品等防御性板块。避免过度集中在单一行业或个股。\n"
            elif investment_horizon == "中期":
                analysis_request += "对于中期投资（1-3年），可以平衡配置价值型和成长型股票，关注具有持续竞争优势的行业龙头企业和合理估值的成长股。\n"
            else:  # 长期
                analysis_request += "对于长期投资（3年以上），可以增加优质成长股比例，关注具有长期竞争优势和行业格局改善的企业，可以承受短期波动以换取长期增值。\n"

            analysis_request += f"""
请基于以上数据和指导，优化该投资组合。你的建议应包括调整后的资产配置、行业配置、个股选择、风险分析、预期收益和再平衡策略。
请给出具体的调整建议，包括应该增加或减少哪些股票的权重，以及调整的理由。
"""
            
            # 发起对话
            self.user.initiate_chat(
                self.portfolio_manager,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为建议
            recommendation = self.user.chat_messages[self.portfolio_manager.name][-1]["content"]
            
            return recommendation
        except Exception as e:
            return f"优化投资组合时出错: {str(e)}"
    
    def export_recommendation(self, recommendation: str, format: str = "markdown", output_file: Optional[str] = None) -> str:
        """
        导出投资组合建议为指定格式
        
        Args:
            recommendation: 投资组合建议文本
            format: 输出格式，支持 "markdown", "html", "text"
            output_file: 输出文件路径，如果不指定则返回内容
            
        Returns:
            如果指定了输出文件，则返回文件路径；否则返回格式化后的内容
        """
        # 转换为指定格式
        if format == "markdown":
            # 已经是Markdown格式，不需要转换
            content = recommendation
        elif format == "html":
            # 使用markdown2模块转换为HTML
            try:
                import markdown2
                content = markdown2.markdown(recommendation)
                # 添加基本的HTML结构
                content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>投资组合建议</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 2em; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>投资组合建议</h1>
    {content}
</body>
</html>"""
            except ImportError:
                content = f"<pre>{recommendation}</pre>"
        elif format == "text":
            # 简单移除Markdown格式
            import re
            content = recommendation
            # 移除Markdown标题
            content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
            # 移除Markdown列表
            content = re.sub(r'^\s*[-*]\s+', '- ', content, flags=re.MULTILINE)
            # 移除Markdown链接，仅保留文本
            content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
            # 移除Markdown粗体和斜体
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
            content = re.sub(r'\*([^*]+)\*', r'\1', content)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
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
    
    def dynamic_portfolio_adjustment(self, current_portfolio: Dict[str, float],
                                    market_trend_prediction: str = None,
                                    risk_preference: str = "中等",
                                    investment_horizon: str = "长期") -> str:
        """
        根据市场趋势预测动态调整投资组合
        
        Args:
            current_portfolio: 当前投资组合，键为股票代码，值为持仓比例
            market_trend_prediction: 市场趋势预测，可为 "看涨", "看跌", "震荡"，如果为None则自动获取
            risk_preference: 风险偏好，可选 "保守", "中等", "激进"
            investment_horizon: 投资期限，可选 "短期", "中期", "长期"
            
        Returns:
            动态调整建议
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
            
            # 获取市场情绪
            market_sentiment = cn_news_utils.get_stock_market_sentiment()
            
            # 如果没有提供市场趋势预测，则根据市场情绪推断
            if market_trend_prediction is None:
                if market_sentiment and "error" not in market_sentiment:
                    if "market_trend" in market_sentiment:
                        trend = market_sentiment["market_trend"].get("trend", "未知")
                        sentiment = market_sentiment["market_trend"].get("sentiment", "中性")
                        
                        if trend == "上涨" or sentiment == "积极":
                            market_trend_prediction = "看涨"
                        elif trend == "下跌" or sentiment == "消极":
                            market_trend_prediction = "看跌"
                        else:
                            market_trend_prediction = "震荡"
                else:
                    market_trend_prediction = "震荡"  # 默认为震荡
            
            # 获取行业信息
            industry_data = {}
            for symbol, info in stocks_info.items():
                industry = self._get_industry_info(symbol, info)
                
                if industry not in industry_data:
                    industry_data[industry] = []
                industry_data[industry].append(symbol)
            
            # 计算行业调整系数
            sector_adjustments = self.get_sector_adjustment(risk_preference, market_sentiment, investment_horizon)
            
            # 根据市场趋势预测调整配置策略
            trend_adjustments = {}
            if market_trend_prediction == "看涨":
                # 看涨环境下增加成长性行业权重
                trend_adjustments = {
                    "银行": 0.8 if risk_preference == "保守" else (0.7 if risk_preference == "中等" else 0.5),
                    "必需消费": 0.7 if risk_preference == "保守" else 0.6,
                    "医药": 0.9,
                    "公用事业": 0.6,
                    "白酒": 0.9 if risk_preference != "保守" else 0.7,
                    "科技": 1.0 if risk_preference == "激进" else (0.9 if risk_preference == "中等" else 0.6),
                    "新能源": 1.0 if risk_preference == "激进" else (0.9 if risk_preference == "中等" else 0.7),
                }
            elif market_trend_prediction == "看跌":
                # 看跌环境下增加防御性行业权重
                trend_adjustments = {
                    "银行": 1.0,
                    "必需消费": 1.0,
                    "医药": 0.9,
                    "公用事业": 1.0,
                    "白酒": 0.5 if risk_preference == "保守" else (0.6 if risk_preference == "中等" else 0.7),
                    "科技": 0.3 if risk_preference == "保守" else (0.5 if risk_preference == "中等" else 0.7),
                    "新能源": 0.4 if risk_preference == "保守" else (0.6 if risk_preference == "中等" else 0.7),
                }
            else:  # 震荡
                # 震荡环境下均衡配置
                trend_adjustments = {
                    "银行": 0.8,
                    "必需消费": 0.8,
                    "医药": 0.8,
                    "公用事业": 0.8,
                    "白酒": 0.7,
                    "科技": 0.7,
                    "新能源": 0.7,
                }
            
            # 合并行业调整系数和趋势调整系数
            final_adjustments = {}
            all_industries = set(list(sector_adjustments.keys()) + list(trend_adjustments.keys()))
            
            for industry in all_industries:
                sector_adj = sector_adjustments.get(industry, 0.7)
                trend_adj = trend_adjustments.get(industry, 0.7)
                
                # 趋势预测权重0.7，一般性适配性权重0.3
                final_adjustments[industry] = trend_adj * 0.7 + sector_adj * 0.3
            
            # 构建分析请求
            analysis_request = f"""
请基于市场趋势预测，对以下投资组合进行动态调整：

投资者信息:
风险偏好: {risk_preference}
投资期限: {investment_horizon}

市场趋势预测: {market_trend_prediction}

当前投资组合:
"""
            
            # 添加当前投资组合信息
            for symbol, weight in current_portfolio.items():
                stock_name = stocks_info.get(symbol, {}).get("股票简称", symbol)
                analysis_request += f"  {symbol} ({stock_name}): {weight*100:.2f}%\n"
            
            # 添加股票详细信息
            analysis_request += "\n股票详细信息:\n"
            for symbol, info in stocks_info.items():
                stock_name = info.get("股票简称", symbol)
                industry = self._get_industry_info(symbol, info)
                price = info.get("最新价", "N/A")
                pe = info.get("市盈率(动态)", "N/A")
                pb = info.get("市净率", "N/A")
                
                analysis_request += f"""
{symbol} ({stock_name}):
  行业: {industry}
  当前价格: {price}
  市盈率: {pe}
  市净率: {pb}
  当前权重: {current_portfolio.get(symbol, 0)*100:.2f}%
"""
            
            # 添加行业分布
            analysis_request += "\n行业分布:\n"
            for industry, symbols in industry_data.items():
                industry_weight = sum(current_portfolio.get(symbol, 0) for symbol in symbols)
                analysis_request += f"  {industry}: {industry_weight*100:.2f}% ({', '.join(symbols)})\n"
            
            # 添加动态行业调整系数
            analysis_request += f"\n基于'{market_trend_prediction}'趋势的行业适配性评分:\n"
            for industry in industry_data.keys():
                adjustment = final_adjustments.get(industry, 0.7)
                adjustment_desc = "非常适合" if adjustment >= 0.9 else (
                                "适合" if adjustment >= 0.7 else (
                                "中性" if adjustment >= 0.5 else (
                                "较不适合" if adjustment >= 0.3 else "不适合")))
                industry_weight = sum(current_portfolio.get(symbol, 0) for symbol in industry_data[industry])
                
                # 计算调整建议
                ideal_direction = ""
                if adjustment >= 0.8 and industry_weight < 0.2:
                    ideal_direction = "【建议增加】"
                elif adjustment <= 0.4 and industry_weight > 0.1:
                    ideal_direction = "【建议减少】"
                elif adjustment >= 0.6 and industry_weight < 0.1:
                    ideal_direction = "【可适度增加】"
                elif adjustment <= 0.5 and industry_weight > 0.2:
                    ideal_direction = "【可适度减少】"
                
                analysis_request += f"  {industry}: {adjustment:.2f} ({adjustment_desc}) - 当前配置: {industry_weight*100:.2f}% {ideal_direction}\n"
            
            # 添加市场情绪
            analysis_request += "\n市场情绪指标:\n"
            if market_sentiment and "error" not in market_sentiment:
                # 检查是否是合成数据
                is_synthetic = any(isinstance(v, dict) and v.get("is_synthetic", False) for v in market_sentiment.values())
                
                if is_synthetic:
                    analysis_request += "注意: 由于实时数据无法获取，以下是合成的市场情绪数据，仅供参考。\n"
                
                # 提取最关键的市场情绪数据
                if "market_trend" in market_sentiment:
                    trend = market_sentiment["market_trend"]
                    analysis_request += f"上证指数最近趋势: {trend.get('trend', 'N/A')}, 变化: {trend.get('recent_change', 'N/A')}, 情绪: {trend.get('sentiment', 'N/A')}\n"
                
                if "north_flow" in market_sentiment:
                    flow = market_sentiment["north_flow"]
                    analysis_request += f"北向资金: {flow.get('direction', 'N/A')}, 数值: {flow.get('value', 'N/A')}\n"
                
                if "market_activity" in market_sentiment:
                    activity = market_sentiment["market_activity"]
                    if "activity" in activity:
                        analysis_request += f"市场活跃度: {activity.get('activity', 'N/A')}\n"
                    elif "total_volume" in activity:
                        analysis_request += f"市场成交量: {activity.get('total_volume', 'N/A')}, 成交额: {activity.get('total_amount', 'N/A')}\n"
                    else:
                        analysis_request += "市场活跃度数据可用\n"
            else:
                analysis_request += "当前无法获取市场情绪数据，请基于其他因素进行分析。\n"
            
            # 基于市场趋势的具体调整策略
            analysis_request += f"\n针对'{market_trend_prediction}'趋势的调整策略建议:\n"
            if market_trend_prediction == "看涨":
                if risk_preference == "保守":
                    analysis_request += "市场趋势看涨，保守型投资者可以：\n"
                    analysis_request += "1. 适度增加优质蓝筹股配置，特别是具有稳定增长的行业龙头\n"
                    analysis_request += "2. 保持较低仓位，增量资金可以分批投入\n"
                    analysis_request += "3. 关注高股息、低估值板块，在市场上涨中获取稳健收益\n"
                elif risk_preference == "中等":
                    analysis_request += "市场趋势看涨，中等风险偏好投资者可以：\n"
                    analysis_request += "1. 增加高景气度行业龙头配置\n"
                    analysis_request += "2. 适当配置具有成长性的科技和新能源股票\n"
                    analysis_request += "3. 维持合理仓位，可以适度提高权益类资产占比\n"
                else:  # 激进
                    analysis_request += "市场趋势看涨，激进型投资者可以：\n"
                    analysis_request += "1. 显著增加成长股配置，特别是科技、新能源等高增长板块\n"
                    analysis_request += "2. 提高整体仓位至较高水平\n"
                    analysis_request += "3. 可以适量配置具有高Beta值的股票，放大上涨收益\n"
            elif market_trend_prediction == "看跌":
                if risk_preference == "保守":
                    analysis_request += "市场趋势看跌，保守型投资者可以：\n"
                    analysis_request += "1. 显著降低权益类资产占比，增加现金持有\n"
                    analysis_request += "2. 保留的股票以必需消费、公用事业等防御性板块为主\n"
                    analysis_request += "3. 避免高估值、高波动性板块\n"
                elif risk_preference == "中等":
                    analysis_request += "市场趋势看跌，中等风险偏好投资者可以：\n"
                    analysis_request += "1. 适度降低仓位，增加现金比例\n"
                    analysis_request += "2. 调整持仓结构，增加防御性板块比例\n"
                    analysis_request += "3. 保留估值合理、基本面稳健的优质成长股\n"
                else:  # 激进
                    analysis_request += "市场趋势看跌，激进型投资者可以：\n"
                    analysis_request += "1. 降低部分仓位，但可以保留核心资产配置\n"
                    analysis_request += "2. 关注超跌优质股票的逢低布局机会\n"
                    analysis_request += "3. 保留部分成长股，但应回避高估值、业绩存疑的股票\n"
            else:  # 震荡
                if risk_preference == "保守":
                    analysis_request += "市场趋势震荡，保守型投资者可以：\n"
                    analysis_request += "1. 维持均衡配置，侧重蓝筹股和稳健型行业\n"
                    analysis_request += "2. 关注股息回报，选择高股息稳定增长的公司\n"
                    analysis_request += "3. 保持适度仓位，避免过度交易\n"
                elif risk_preference == "中等":
                    analysis_request += "市场趋势震荡，中等风险偏好投资者可以：\n"
                    analysis_request += "1. 均衡配置价值型和成长型股票\n"
                    analysis_request += "2. 关注结构性机会，适当进行行业轮动\n"
                    analysis_request += "3. 采取波段操作策略，在市场波动中获利\n"
                else:  # 激进
                    analysis_request += "市场趋势震荡，激进型投资者可以：\n"
                    analysis_request += "1. 关注景气度向上的细分行业，寻找结构性机会\n"
                    analysis_request += "2. 可以适当增加交易频率，把握波段机会\n"
                    analysis_request += "3. 关注市场情绪变化，及时调整持仓结构\n"
            
            analysis_request += """
请基于以上数据和市场趋势预测，提供具体的投资组合动态调整建议。你的建议应包括：

1. 调整后的目标配置比例（具体到每只股票的目标权重）
2. 调整理由（基于市场趋势、行业景气度和个股基本面）
3. 调整优先级（哪些调整最为紧急或重要）
4. 建议的执行策略（如何分批实施调整，以减少市场冲击）

请确保调整建议与投资者风险偏好和投资期限相匹配，并充分考虑当前市场趋势预测。
"""
            
            # 发起对话
            self.user.initiate_chat(
                self.portfolio_manager,
                message=analysis_request,
            )
            
            # 获取最后一条消息作为建议
            recommendation = self.user.chat_messages[self.portfolio_manager.name][-1]["content"]
            
            return recommendation
        except Exception as e:
            return f"动态调整投资组合时出错: {str(e)}"
    
    def reset(self):
        """
        重置聊天历史
        """
        self.user.reset()
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
        # 构建兼容两个版本API的llm_config
        self.llm_config = get_llm_config_for_autogen(**llm_config)
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",  # 改为NEVER允许对话继续进行
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建技术分析师代理
        self.technical_analyst = autogen.AssistantAgent(
            name="TechnicalAnalyst",
            llm_config=self.llm_config,
            system_message="""你是一位专业的技术分析师，擅长使用各种技术指标分析股票走势。
你需要分析提供的股票数据和技术指标，给出对股票未来走势的技术分析。
你的分析应该包括以下几个方面：
1. 趋势分析：识别股票的上升、下降或横盘趋势
2. 支撑位和阻力位：识别关键价格水平
3. 技术指标分析：基于RSI、KDJ、MACD等指标的分析
4. 图表形态：识别头肩顶、双底等图表形态
5. 交易信号：基于技术分析的买入或卖出信号
6. 止损和目标价位：建议的止损位和目标价位

请确保你的分析逻辑清晰，观点有理有据。"""
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

请基于以上数据，进行全面的技术分析。你的分析必须包含以下内容：
1. 趋势分析（明确指出上升/下降/横盘趋势）
2. 支撑位和阻力位（给出具体价格）
3. 技术指标分析（必须分析RSI、KDJ和MACD指标）
4. 交易信号（明确的买入/卖出/观望建议）
5. 目标价位和止损位（给出具体价格）

请确保分析全面覆盖以上五个方面，并给出明确的结论。
"""
            
            # 创建判断函数，用于评估回答质量
            def is_answer_satisfactory(response):
                required_elements = [
                    "趋势分析", "支撑位", "阻力位", "RSI", "KDJ", "MACD", 
                    "交易信号", "目标价位", "止损位"
                ]
                element_count = sum(1 for element in required_elements if element in response)
                return element_count >= 7  # 至少包含7个必要元素
            
            # 初始化对话变量
            max_turns = 3  # 最大对话轮次
            current_turn = 0
            analysis_result = ""
            
            # 准备初始消息
            current_message = analysis_request
            
            print(f"开始对{symbol}进行技术分析，请稍候...")
            
            # 循环对话直到得到满意答案或达到最大轮次
            while current_turn < max_turns:
                print(f"进行第 {current_turn+1} 轮技术分析对话...")
                
                # 发起对话
                self.user_proxy.initiate_chat(
                    self.technical_analyst,
                    message=current_message,
                )
                
                # 获取最后一条消息
                if self.technical_analyst.name in self.user_proxy.chat_messages:
                    response = self.user_proxy.chat_messages[self.technical_analyst.name][-1]["content"]
                    
                    # 检查回答是否满意
                    if is_answer_satisfactory(response):
                        analysis_result = response
                        print("已获得满意的技术分析结果")
                        break
                    else:
                        # 准备追问 - 找出缺失的元素
                        missing_elements = [
                            elem for elem in ["趋势分析", "支撑位和阻力位", "RSI/KDJ/MACD指标分析", 
                                           "交易信号", "目标价位和止损位"] 
                            if not any(key in response for key in elem.split("/"))
                        ]
                        
                        if missing_elements:
                            missing_text = "、".join(missing_elements)
                            current_message = f"你的分析不够全面，请补充以下缺失内容：{missing_text}。需要给出更明确的分析结果和具体数值。"
                        else:
                            current_message = "请提供更明确的分析结论，包括具体的支撑位/阻力位价格、明确的交易信号和具体的目标价位/止损位价格。"
                else:
                    print("未获得回应，终止对话")
                    break
                    
                current_turn += 1
            
            # 如果达到最大轮次但没有满意答案，使用最后一次回应
            if not analysis_result and self.technical_analyst.name in self.user_proxy.chat_messages:
                analysis_result = self.user_proxy.chat_messages[self.technical_analyst.name][-1]["content"]
                print("已达到最大对话轮次，使用最后一次分析结果")
            
            # 主动结束对话
            self.user_proxy.send(
                recipient=self.technical_analyst,
                message="分析完成，对话结束。"
            )
            
            return analysis_result or "未能获得有效的技术分析结果"
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
            content = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>股票技术分析报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .date { color: #666; margin-bottom: 20px; }
        .analysis { line-height: 1.6; }
    </style>
</head>
<body>
    <h1>股票技术分析报告</h1>
    <div class="date"><strong>生成日期</strong>: """ + current_date + """</div>
    <div class="analysis">""" + analysis.replace('\n', '<br>') + """</div>
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