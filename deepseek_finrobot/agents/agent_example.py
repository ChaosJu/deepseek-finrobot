"""
代理示例模块 - 展示如何使用代理库中的代理
"""

import os
import json
from typing import Dict, Any, List
from .agent_library import MarketForecasterAgent, FinancialReportAgent, NewsAnalysisAgent, IndustryAnalysisAgent, PortfolioManagerAgent, TechnicalAnalysisAgent
from ..utils import get_current_date, format_financial_number

def load_llm_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载LLM配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        LLM配置字典
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            "config_list": [
                {
                    "model": "deepseek-chat",
                    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
                    "base_url": "https://api.deepseek.com/v1",
                }
            ],
            "temperature": 0.7,
            "request_timeout": 120,
            "seed": 42,
            "config_list_timeout": 60,
        }
    
    return config

def market_forecaster_example(symbol: str = "000001", days: int = 7):
    """
    市场预测代理示例
    
    Args:
        symbol: 股票代码
        days: 分析过去几天的数据和新闻
    """
    print(f"===== 市场预测代理示例 - {get_current_date()} =====")
    print(f"分析股票: {symbol}, 时间范围: 过去{days}天\n")
    
    # 加载LLM配置
    llm_config = load_llm_config()
    
    # 创建市场预测代理
    forecaster = MarketForecasterAgent(llm_config)
    
    # 预测股票走势
    prediction = forecaster.predict(symbol, days)
    
    print("预测结果:")
    print(prediction)
    print("\n" + "="*50 + "\n")

def financial_report_example(symbol: str = "000001"):
    """
    财务报告代理示例
    
    Args:
        symbol: 股票代码
    """
    print(f"===== 财务报告代理示例 - {get_current_date()} =====")
    print(f"分析股票: {symbol}\n")
    
    # 加载LLM配置
    llm_config = load_llm_config()
    
    # 创建财务报告代理
    reporter = FinancialReportAgent(llm_config)
    
    # 生成财务分析报告
    report = reporter.generate_report(symbol)
    
    print("财务分析报告:")
    print(report)
    print("\n" + "="*50 + "\n")

def news_analysis_example(keywords: str = "人工智能", days: int = 3, limit: int = 5):
    """
    新闻分析代理示例
    
    Args:
        keywords: 搜索关键词
        days: 分析过去几天的新闻
        limit: 分析的新闻数量
    """
    print(f"===== 新闻分析代理示例 - {get_current_date()} =====")
    print(f"关键词: {keywords}, 时间范围: 过去{days}天, 数量: {limit}\n")
    
    # 加载LLM配置
    llm_config = load_llm_config()
    
    # 创建新闻分析代理
    analyzer = NewsAnalysisAgent(llm_config)
    
    # 分析财经新闻
    analysis = analyzer.analyze_news(keywords, days, limit)
    
    print("新闻分析结果:")
    print(analysis)
    print("\n" + "="*50 + "\n")

def industry_analysis_example(industry: str = "银行", days: int = 30):
    """
    行业分析代理示例
    
    Args:
        industry: 行业名称
        days: 分析过去几天的数据和新闻
    """
    print(f"===== 行业分析代理示例 - {get_current_date()} =====")
    print(f"分析行业: {industry}, 时间范围: 过去{days}天\n")
    
    # 加载LLM配置
    llm_config = load_llm_config()
    
    # 创建行业分析代理
    analyzer = IndustryAnalysisAgent(llm_config)
    
    # 分析行业趋势
    analysis = analyzer.analyze_industry(industry, days)
    
    print("行业分析结果:")
    print(analysis)
    
    # 导出分析结果
    output_file = f"industry_analysis_{industry}_{get_current_date()}.md"
    analyzer.export_analysis(analysis, format="markdown", output_file=output_file)
    print(f"\n分析结果已导出到: {output_file}")
    
    print("\n" + "="*50 + "\n")

def portfolio_manager_example(stocks: List[str] = ["000001", "600519", "000858"], 
                             risk_preference: str = "中等", 
                             investment_horizon: str = "长期",
                             investment_amount: float = 100000):
    """
    投资组合代理示例
    
    Args:
        stocks: 股票代码列表
        risk_preference: 风险偏好
        investment_horizon: 投资期限
        investment_amount: 投资金额
    """
    print(f"===== 投资组合代理示例 - {get_current_date()} =====")
    print(f"股票列表: {', '.join(stocks)}")
    print(f"风险偏好: {risk_preference}, 投资期限: {investment_horizon}, 投资金额: {format_financial_number(investment_amount)}元\n")
    
    # 加载LLM配置
    llm_config = load_llm_config()
    
    # 创建投资组合代理
    portfolio_manager = PortfolioManagerAgent(llm_config)
    
    # 构建投资组合
    recommendation = portfolio_manager.construct_portfolio(
        stocks=stocks,
        risk_preference=risk_preference,
        investment_horizon=investment_horizon,
        investment_amount=investment_amount
    )
    
    print("投资组合建议:")
    print(recommendation)
    
    # 导出建议
    output_file = f"portfolio_recommendation_{get_current_date()}.md"
    portfolio_manager.export_recommendation(recommendation, format="markdown", output_file=output_file)
    print(f"\n建议已导出到: {output_file}")
    
    # 优化投资组合示例
    print("\n===== 投资组合优化示例 =====\n")
    
    # 假设的当前投资组合
    current_portfolio = {
        "000001": 0.3,  # 平安银行 30%
        "600519": 0.5,  # 贵州茅台 50%
        "000858": 0.2   # 五粮液 20%
    }
    
    print("当前投资组合:")
    for symbol, weight in current_portfolio.items():
        print(f"  {symbol}: {weight*100:.2f}%")
    
    # 优化投资组合
    optimization = portfolio_manager.optimize_portfolio(
        current_portfolio=current_portfolio,
        risk_preference=risk_preference,
        investment_horizon=investment_horizon
    )
    
    print("\n优化建议:")
    print(optimization)
    
    print("\n" + "="*50 + "\n")

def technical_analysis_example(symbol: str = "000001", period: str = "daily", days: int = 120):
    """
    技术分析代理示例
    
    Args:
        symbol: 股票代码
        period: 时间周期
        days: 分析过去几天的数据
    """
    print(f"===== 技术分析代理示例 - {get_current_date()} =====")
    print(f"分析股票: {symbol}, 时间周期: {period}, 时间范围: 过去{days}天\n")
    
    # 加载LLM配置
    llm_config = load_llm_config()
    
    # 创建技术分析代理
    analyst = TechnicalAnalysisAgent(llm_config)
    
    # 进行技术分析
    analysis = analyst.analyze(symbol, period, days)
    
    print("技术分析结果:")
    print(analysis)
    
    # 导出分析结果
    output_file = f"technical_analysis_{symbol}_{get_current_date()}.md"
    analyst.export_analysis(analysis, format="markdown", output_file=output_file)
    print(f"\n分析结果已导出到: {output_file}")
    
    # 批量分析示例
    print("\n===== 批量技术分析示例 =====\n")
    
    symbols = ["000001", "600519", "000858"]  # 平安银行、贵州茅台、五粮液
    print(f"批量分析股票: {', '.join(symbols)}\n")
    
    # 批量进行技术分析
    analyses = analyst.batch_analyze(symbols, period, days, max_workers=3)
    
    for symbol, analysis in analyses.items():
        print(f"股票 {symbol} 技术分析结果:")
        print(analysis[:200] + "...\n")  # 只显示前200个字符
    
    print("\n" + "="*50 + "\n")

def batch_analysis_example():
    """
    批量分析示例
    """
    print(f"===== 批量分析示例 - {get_current_date()} =====")
    
    # 加载LLM配置
    llm_config = load_llm_config()
    
    # 创建市场预测代理
    forecaster = MarketForecasterAgent(llm_config)
    
    # 批量预测股票走势
    symbols = ["000001", "600519", "000858"]  # 平安银行、贵州茅台、五粮液
    print(f"批量预测股票: {', '.join(symbols)}\n")
    
    predictions = forecaster.batch_predict(symbols, days=7, max_workers=3)
    
    for symbol, prediction in predictions.items():
        print(f"股票 {symbol} 预测结果:")
        print(prediction[:200] + "...\n")  # 只显示前200个字符
    
    # 创建行业分析代理
    industry_analyzer = IndustryAnalysisAgent(llm_config)
    
    # 批量分析行业
    industries = ["银行", "医药", "计算机"]
    print(f"批量分析行业: {', '.join(industries)}\n")
    
    analyses = industry_analyzer.batch_analyze(industries, days=30, max_workers=3)
    
    for industry, analysis in analyses.items():
        print(f"行业 {industry} 分析结果:")
        print(analysis[:200] + "...\n")  # 只显示前200个字符
    
    print("\n" + "="*50 + "\n")

def run_all_examples():
    """
    运行所有示例
    """
    # 市场预测示例
    market_forecaster_example()
    
    # 财务报告示例
    financial_report_example()
    
    # 新闻分析示例
    news_analysis_example()
    
    # 行业分析示例
    industry_analysis_example()
    
    # 投资组合示例
    portfolio_manager_example()
    
    # 技术分析示例
    technical_analysis_example()
    
    # 批量分析示例
    batch_analysis_example()

if __name__ == "__main__":
    # 运行单个示例
    # market_forecaster_example("000001")  # 平安银行
    
    # 或者运行行业分析示例
    # industry_analysis_example("银行")
    
    # 或者运行投资组合示例
    portfolio_manager_example()
    
    # 或者运行技术分析示例
    # technical_analysis_example("000001")
    
    # 或者运行批量分析示例
    # batch_analysis_example()
    
    # 或者运行所有示例
    # run_all_examples() 