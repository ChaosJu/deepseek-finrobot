"""
命令行工具 - 提供命令行接口
"""

import argparse
import sys
import os
import json
from typing import Dict, Any, List, Optional
from .agents import MarketForecasterAgent, FinancialReportAgent, NewsAnalysisAgent, IndustryAnalysisAgent, PortfolioManagerAgent, TechnicalAnalysisAgent
from .utils import get_current_date, get_deepseek_config, register_keys_from_json

def load_config() -> Dict[str, Any]:
    """
    加载配置
    
    Returns:
        配置字典
    """
    try:
        # 尝试加载API密钥
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_api_keys.json")
        if os.path.exists(config_path):
            keys = register_keys_from_json(config_path)
            if not keys:
                print("\n警告: 未找到有效的API密钥配置")
                print("请确保您已从config_api_keys_sample创建了config_api_keys.json文件，并填入有效的API密钥")
            
            # 检查是否有DeepSeek API密钥
            if "DEEPSEEK_API_KEY" not in keys or not keys["DEEPSEEK_API_KEY"]:
                print("\n错误: 未找到有效的DeepSeek API密钥")
                print("请确保config_api_keys.json中包含DEEPSEEK_API_KEY")
                sys.exit(1)
        else:
            print(f"\n错误: 找不到配置文件 {config_path}")
            print("请从config_api_keys_sample创建config_api_keys.json文件")
            sys.exit(1)
        
        # 加载DeepSeek配置
        try:
            llm_config = {
                "config_list": get_deepseek_config(),
                "temperature": 0.7,
            }
            return llm_config
        except FileNotFoundError as e:
            print(f"\n错误: {str(e)}")
            print("请从DEEPSEEK_CONFIG_sample创建DEEPSEEK_CONFIG文件")
            sys.exit(1)
    except Exception as e:
        print(f"\n加载配置出错: {str(e)}")
        print("请确保您已正确设置配置文件，并安装了所有必需的依赖项")
        sys.exit(1)

def predict_stock(args):
    """
    预测股票走势
    
    Args:
        args: 命令行参数
    """
    try:
        llm_config = load_config()
        forecaster = MarketForecasterAgent(llm_config)
        
        if args.batch:
            # 批量预测
            symbols = args.symbol.split(',')
            print(f"批量预测股票: {', '.join(symbols)}")
            
            predictions = forecaster.batch_predict(symbols, days=args.days, max_workers=args.workers)
            
            for symbol, prediction in predictions.items():
                print(f"\n===== 股票 {symbol} 预测结果 =====\n")
                try:
                    print(prediction)
                except UnicodeEncodeError as e:
                    print(f"预测股票走势时出错: {e}")
                
                if args.export:
                    try:
                        output_file = f"stock_prediction_{symbol}_{get_current_date()}.{args.format}"
                        forecaster.export_prediction(prediction, format=args.format, output_file=output_file)
                        print(f"\n预测结果已导出到: {output_file}")
                    except Exception as ex:
                        print(f"导出预测结果时出错: {ex}")
        else:
            # 单个预测
            print(f"预测股票: {args.symbol}")
            
            try:
                prediction = forecaster.predict(args.symbol, days=args.days)
                print("\n===== 预测结果 =====\n")
                try:
                    print(prediction)
                except UnicodeEncodeError as e:
                    print(f"预测股票走势时出错: {e}")
                
                if args.export:
                    try:
                        output_file = f"stock_prediction_{args.symbol}_{get_current_date()}.{args.format}"
                        forecaster.export_prediction(prediction, format=args.format, output_file=output_file)
                        print(f"\n预测结果已导出到: {output_file}")
                    except Exception as ex:
                        print(f"导出预测结果时出错: {ex}")
            except Exception as e:
                print(f"预测股票走势时出错: {e}")
    except Exception as e:
        print(f"预测过程发生错误: {e}")

def analyze_industry(args):
    """
    分析行业趋势
    
    Args:
        args: 命令行参数
    """
    llm_config = load_config()
    analyzer = IndustryAnalysisAgent(llm_config)
    
    if args.batch:
        # 批量分析
        industries = args.industry.split(',')
        print(f"批量分析行业: {', '.join(industries)}")
        
        analyses = analyzer.batch_analyze(industries, days=args.days, max_workers=args.workers)
        
        for industry, analysis in analyses.items():
            print(f"\n===== 行业 {industry} 分析结果 =====\n")
            print(analysis)
            
            if args.export:
                output_file = f"industry_analysis_{industry}_{get_current_date()}.{args.format}"
                analyzer.export_analysis(analysis, format=args.format, output_file=output_file)
                print(f"\n分析结果已导出到: {output_file}")
    else:
        # 单个分析
        print(f"分析行业: {args.industry}")
        
        analysis = analyzer.analyze_industry(args.industry, days=args.days)
        print("\n===== 分析结果 =====\n")
        print(analysis)
        
        if args.export:
            output_file = f"industry_analysis_{args.industry}_{get_current_date()}.{args.format}"
            analyzer.export_analysis(analysis, format=args.format, output_file=output_file)
            print(f"\n分析结果已导出到: {output_file}")

def analyze_news(args):
    """
    分析财经新闻
    
    Args:
        args: 命令行参数
    """
    llm_config = load_config()
    analyzer = NewsAnalysisAgent(llm_config)
    
    print(f"分析关键词: {args.keywords}")
    
    analysis = analyzer.analyze_news(args.keywords, days=args.days, limit=args.limit)
    print("\n===== 分析结果 =====\n")
    print(analysis)

def generate_report(args):
    """
    生成财务分析报告
    
    Args:
        args: 命令行参数
    """
    llm_config = load_config()
    reporter = FinancialReportAgent(llm_config)
    
    print(f"生成财务报告: {args.symbol}")
    
    report = reporter.generate_report(args.symbol)
    print("\n===== 财务分析报告 =====\n")
    print(report)

def construct_portfolio(args):
    """
    构建投资组合
    
    Args:
        args: 命令行参数
    """
    llm_config = load_config()
    portfolio_manager = PortfolioManagerAgent(llm_config)
    
    # 解析股票代码列表
    stocks = args.stocks.split(',')
    print(f"构建投资组合，股票列表: {', '.join(stocks)}")
    
    recommendation = portfolio_manager.construct_portfolio(
        stocks=stocks,
        risk_preference=args.risk,
        investment_horizon=args.horizon,
        investment_amount=args.amount
    )
    
    print("\n===== 投资组合建议 =====\n")
    print(recommendation)
    
    if args.export:
        output_file = f"portfolio_recommendation_{get_current_date()}.{args.format}"
        portfolio_manager.export_recommendation(recommendation, format=args.format, output_file=output_file)
        print(f"\n建议已导出到: {output_file}")

def optimize_portfolio(args):
    """
    优化投资组合
    
    Args:
        args: 命令行参数
    """
    llm_config = load_config()
    portfolio_manager = PortfolioManagerAgent(llm_config)
    
    # 解析当前投资组合
    current_portfolio = {}
    for item in args.portfolio.split(','):
        parts = item.split(':')
        if len(parts) == 2:
            symbol = parts[0].strip()
            weight = float(parts[1].strip()) / 100  # 转换为小数
            current_portfolio[symbol] = weight
    
    if not current_portfolio:
        print("错误: 无效的投资组合格式，应为 'symbol1:weight1,symbol2:weight2'")
        sys.exit(1)
    
    print("优化投资组合:")
    for symbol, weight in current_portfolio.items():
        print(f"  {symbol}: {weight*100:.2f}%")
    
    recommendation = portfolio_manager.optimize_portfolio(
        current_portfolio=current_portfolio,
        risk_preference=args.risk,
        investment_horizon=args.horizon
    )
    
    print("\n===== 投资组合优化建议 =====\n")
    print(recommendation)
    
    if args.export:
        output_file = f"portfolio_optimization_{get_current_date()}.{args.format}"
        portfolio_manager.export_recommendation(recommendation, format=args.format, output_file=output_file)
        print(f"\n建议已导出到: {output_file}")

def dynamic_adjust_portfolio(args):
    """
    根据市场趋势动态调整投资组合
    
    Args:
        args: 命令行参数
    """
    llm_config = load_config()
    portfolio_manager = PortfolioManagerAgent(llm_config)
    
    # 解析当前投资组合
    current_portfolio = {}
    for item in args.portfolio.split(','):
        parts = item.split(':')
        if len(parts) == 2:
            symbol = parts[0].strip()
            weight = float(parts[1].strip()) / 100  # 转换为小数
            current_portfolio[symbol] = weight
    
    if not current_portfolio:
        print("错误: 无效的投资组合格式，应为 'symbol1:weight1,symbol2:weight2'")
        sys.exit(1)
    
    print(f"动态调整投资组合（市场趋势: {args.trend if args.trend else '自动判断'}）:")
    for symbol, weight in current_portfolio.items():
        print(f"  {symbol}: {weight*100:.2f}%")
    
    recommendation = portfolio_manager.dynamic_portfolio_adjustment(
        current_portfolio=current_portfolio,
        market_trend_prediction=args.trend,
        risk_preference=args.risk,
        investment_horizon=args.horizon
    )
    
    print("\n===== 投资组合动态调整建议 =====\n")
    print(recommendation)
    
    if args.export:
        output_file = f"portfolio_adjustment_{get_current_date()}.{args.format}"
        portfolio_manager.export_recommendation(recommendation, format=args.format, output_file=output_file)
        print(f"\n建议已导出到: {output_file}")

def technical_analyze(args):
    """
    进行技术分析
    
    Args:
        args: 命令行参数
    """
    llm_config = load_config()
    analyst = TechnicalAnalysisAgent(llm_config)
    
    if args.batch:
        # 批量分析
        symbols = args.symbol.split(',')
        print(f"批量技术分析: {', '.join(symbols)}")
        
        analyses = analyst.batch_analyze(symbols, period=args.period, days=args.days, max_workers=args.workers)
        
        for symbol, analysis in analyses.items():
            print(f"\n===== 股票 {symbol} 技术分析结果 =====\n")
            print(analysis)
            
            if args.export:
                output_file = f"technical_analysis_{symbol}_{get_current_date()}.{args.format}"
                analyst.export_analysis(analysis, format=args.format, output_file=output_file)
                print(f"\n分析结果已导出到: {output_file}")
    else:
        # 单个分析
        print(f"技术分析股票: {args.symbol}, 周期: {args.period}, 时间范围: 过去{args.days}天")
        
        analysis = analyst.analyze(args.symbol, period=args.period, days=args.days)
        print("\n===== 技术分析结果 =====\n")
        print(analysis)
        
        if args.export:
            output_file = f"technical_analysis_{args.symbol}_{get_current_date()}.{args.format}"
            analyst.export_analysis(analysis, format=args.format, output_file=output_file)
            print(f"\n分析结果已导出到: {output_file}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="DeepSeek FinRobot 命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 预测股票走势
    predict_parser = subparsers.add_parser("predict", help="预测股票走势")
    predict_parser.add_argument("symbol", help="股票代码，批量预测时用逗号分隔")
    predict_parser.add_argument("--days", type=int, default=7, help="分析过去几天的数据和新闻")
    predict_parser.add_argument("--batch", action="store_true", help="批量预测")
    predict_parser.add_argument("--workers", type=int, default=3, help="批量预测时的最大并行工作线程数")
    predict_parser.add_argument("--export", action="store_true", help="导出预测结果")
    predict_parser.add_argument("--format", choices=["markdown", "html", "text"], default="markdown", help="导出格式")
    predict_parser.set_defaults(func=predict_stock)
    
    # 分析行业趋势
    industry_parser = subparsers.add_parser("industry", help="分析行业趋势")
    industry_parser.add_argument("industry", help="行业名称，批量分析时用逗号分隔")
    industry_parser.add_argument("--days", type=int, default=30, help="分析过去几天的数据和新闻")
    industry_parser.add_argument("--batch", action="store_true", help="批量分析")
    industry_parser.add_argument("--workers", type=int, default=3, help="批量分析时的最大并行工作线程数")
    industry_parser.add_argument("--export", action="store_true", help="导出分析结果")
    industry_parser.add_argument("--format", choices=["markdown", "html", "text"], default="markdown", help="导出格式")
    industry_parser.set_defaults(func=analyze_industry)
    
    # 分析财经新闻
    news_parser = subparsers.add_parser("news", help="分析财经新闻")
    news_parser.add_argument("keywords", help="搜索关键词")
    news_parser.add_argument("--days", type=int, default=3, help="分析过去几天的新闻")
    news_parser.add_argument("--limit", type=int, default=10, help="分析的新闻数量")
    news_parser.set_defaults(func=analyze_news)
    
    # 生成财务分析报告
    report_parser = subparsers.add_parser("report", help="生成财务分析报告")
    report_parser.add_argument("symbol", help="股票代码")
    report_parser.set_defaults(func=generate_report)
    
    # 构建投资组合
    portfolio_parser = subparsers.add_parser("portfolio", help="构建投资组合")
    portfolio_parser.add_argument("stocks", help="股票代码列表，用逗号分隔")
    portfolio_parser.add_argument("--risk", choices=["保守", "中等", "激进"], default="中等", help="风险偏好")
    portfolio_parser.add_argument("--horizon", choices=["短期", "中期", "长期"], default="长期", help="投资期限")
    portfolio_parser.add_argument("--amount", type=float, default=100000, help="投资金额")
    portfolio_parser.add_argument("--export", action="store_true", help="导出建议")
    portfolio_parser.add_argument("--format", choices=["markdown", "html", "text"], default="markdown", help="导出格式")
    portfolio_parser.set_defaults(func=construct_portfolio)
    
    # 优化投资组合
    optimize_parser = subparsers.add_parser("optimize", help="优化投资组合")
    optimize_parser.add_argument("portfolio", help="当前投资组合，格式为'symbol1:weight1,symbol2:weight2'，权重为百分比")
    optimize_parser.add_argument("--risk", choices=["保守", "中等", "激进"], default="中等", help="风险偏好")
    optimize_parser.add_argument("--horizon", choices=["短期", "中期", "长期"], default="长期", help="投资期限")
    optimize_parser.add_argument("--export", action="store_true", help="导出建议")
    optimize_parser.add_argument("--format", choices=["markdown", "html", "text"], default="markdown", help="导出格式")
    optimize_parser.set_defaults(func=optimize_portfolio)
    
    # 动态调整投资组合
    adjust_parser = subparsers.add_parser("adjust", help="根据市场趋势动态调整投资组合")
    adjust_parser.add_argument("portfolio", help="当前投资组合，格式为'symbol1:weight1,symbol2:weight2'，权重为百分比")
    adjust_parser.add_argument("--trend", choices=["看涨", "看跌", "震荡"], help="市场趋势预测，不提供则自动判断")
    adjust_parser.add_argument("--risk", choices=["保守", "中等", "激进"], default="中等", help="风险偏好")
    adjust_parser.add_argument("--horizon", choices=["短期", "中期", "长期"], default="长期", help="投资期限")
    adjust_parser.add_argument("--export", action="store_true", help="导出建议")
    adjust_parser.add_argument("--format", choices=["markdown", "html", "text"], default="markdown", help="导出格式")
    adjust_parser.set_defaults(func=dynamic_adjust_portfolio)
    
    # 技术分析
    technical_parser = subparsers.add_parser("technical", help="进行技术分析")
    technical_parser.add_argument("symbol", help="股票代码，批量分析时用逗号分隔")
    technical_parser.add_argument("--period", choices=["daily", "weekly", "monthly"], default="daily", help="时间周期")
    technical_parser.add_argument("--days", type=int, default=120, help="分析过去几天的数据")
    technical_parser.add_argument("--batch", action="store_true", help="批量分析")
    technical_parser.add_argument("--workers", type=int, default=3, help="批量分析时的最大并行工作线程数")
    technical_parser.add_argument("--export", action="store_true", help="导出分析结果")
    technical_parser.add_argument("--format", choices=["markdown", "html", "text"], default="markdown", help="导出格式")
    technical_parser.set_defaults(func=technical_analyze)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main() 