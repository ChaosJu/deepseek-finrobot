"""
中国新闻数据源工具 - 使用AKShare获取中国金融新闻
"""

import akshare as ak
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import datetime
import re
import requests
from bs4 import BeautifulSoup

def get_financial_news(limit: int = 20) -> pd.DataFrame:
    """
    获取财经新闻
    
    Args:
        limit: 返回的新闻数量
        
    Returns:
        财经新闻DataFrame
    """
    try:
        # 获取东方财富网财经新闻
        df = ak.stock_news_em()
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"获取财经新闻时出错: {e}")
        return pd.DataFrame()

def get_stock_news_sina(symbol: str, limit: int = 10) -> pd.DataFrame:
    """
    获取新浪财经股票新闻
    
    Args:
        symbol: 股票代码（如：sh000001 或 sz000001）
        limit: 返回的新闻数量
        
    Returns:
        股票新闻DataFrame
    """
    try:
        # 获取新浪财经股票新闻
        df = ak.stock_news_sina(symbol=symbol)
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"获取新浪财经股票新闻时出错: {e}")
        return pd.DataFrame()

def get_major_news() -> pd.DataFrame:
    """
    获取重大财经新闻
    
    Returns:
        重大财经新闻DataFrame
    """
    try:
        # 获取金十数据重大财经新闻
        df = ak.js_news()
        
        return df
    except Exception as e:
        print(f"获取重大财经新闻时出错: {e}")
        return pd.DataFrame()

def get_cctv_news() -> pd.DataFrame:
    """
    获取央视新闻
    
    Returns:
        央视新闻DataFrame
    """
    try:
        # 获取央视新闻
        df = ak.news_cctv()
        
        return df
    except Exception as e:
        print(f"获取央视新闻时出错: {e}")
        return pd.DataFrame()

def get_stock_research_report(symbol: str = None, category: str = None) -> pd.DataFrame:
    """
    获取股票研究报告
    
    Args:
        symbol: 股票代码（如：000001）
        category: 报告类别
        
    Returns:
        研究报告DataFrame
    """
    try:
        # 获取研究报告
        if symbol:
            df = ak.stock_research_report_em(symbol=symbol)
        elif category:
            df = ak.stock_research_report_em(symbol=category)
        else:
            df = ak.stock_research_report_em()
            
        return df
    except Exception as e:
        print(f"获取股票研究报告时出错: {e}")
        return pd.DataFrame()

def get_market_news_baidu(keywords: str = "股市", limit: int = 10) -> pd.DataFrame:
    """
    获取百度股市新闻
    
    Args:
        keywords: 搜索关键词
        limit: 返回的新闻数量
        
    Returns:
        百度股市新闻DataFrame
    """
    try:
        # 获取百度股市新闻
        df = ak.news_baidu(keywords=keywords)
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"获取百度股市新闻时出错: {e}")
        return pd.DataFrame()

def get_stock_notice(symbol: str, report_type: str = None) -> pd.DataFrame:
    """
    获取股票公告
    
    Args:
        symbol: 股票代码（如：000001）
        report_type: 公告类型
        
    Returns:
        股票公告DataFrame
    """
    try:
        # 获取股票公告
        if report_type:
            df = ak.stock_notice_report(symbol=symbol, report_type=report_type)
        else:
            df = ak.stock_notice_report(symbol=symbol)
            
        return df
    except Exception as e:
        print(f"获取股票公告时出错: {e}")
        return pd.DataFrame()

def get_stock_report_disclosure(symbol: str, period: str = "2023") -> pd.DataFrame:
    """
    获取股票财报披露时间
    
    Args:
        symbol: 股票代码（如：000001）
        period: 年份
        
    Returns:
        财报披露时间DataFrame
    """
    try:
        # 获取财报披露时间
        df = ak.stock_report_disclosure(symbol=symbol, period=period)
        
        return df
    except Exception as e:
        print(f"获取股票财报披露时间时出错: {e}")
        return pd.DataFrame()

def get_stock_industry_news(industry: str, limit: int = 10) -> pd.DataFrame:
    """
    获取行业新闻
    
    Args:
        industry: 行业名称
        limit: 返回的新闻数量
        
    Returns:
        行业新闻DataFrame
    """
    try:
        # 获取财经新闻
        df = get_financial_news(limit=100)
        
        # 过滤行业新闻
        df = df[df['content'].str.contains(industry)]
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"获取行业新闻时出错: {e}")
        return pd.DataFrame()

def get_stock_market_sentiment() -> Dict[str, Any]:
    """
    获取股市情绪指标
    
    Returns:
        股市情绪指标字典
    """
    try:
        # 获取股市情绪指标
        df_fear = ak.index_vix()
        df_up_down = ak.stock_market_activity_legu()
        
        # 计算最新情绪指标
        latest_fear = df_fear.iloc[-1].to_dict() if not df_fear.empty else {}
        latest_up_down = df_up_down.iloc[-1].to_dict() if not df_up_down.empty else {}
        
        # 合并结果
        result = {
            "fear_index": latest_fear,
            "market_activity": latest_up_down
        }
        
        return result
    except Exception as e:
        print(f"获取股市情绪指标时出错: {e}")
        return {"error": str(e)}

def search_news(keywords: str, days: int = 7, limit: int = 10) -> pd.DataFrame:
    """
    搜索新闻
    
    Args:
        keywords: 搜索关键词
        days: 过去几天的新闻
        limit: 返回的新闻数量
        
    Returns:
        新闻DataFrame
    """
    try:
        # 获取财经新闻
        df = get_financial_news(limit=100)
        
        # 计算日期范围
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # 转换日期列
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # 过滤日期
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
        # 过滤关键词
        df = df[df['content'].str.contains(keywords) | df['title'].str.contains(keywords)]
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"搜索新闻时出错: {e}")
        return pd.DataFrame()

def get_stock_hot_rank() -> pd.DataFrame:
    """
    获取股票热门排行
    
    Returns:
        股票热门排行DataFrame
    """
    try:
        # 获取东方财富网股票热门排行
        df = ak.stock_hot_rank_em()
        
        return df
    except Exception as e:
        print(f"获取股票热门排行时出错: {e}")
        return pd.DataFrame()

def get_stock_hot_keyword() -> pd.DataFrame:
    """
    获取股市热门关键词
    
    Returns:
        股市热门关键词DataFrame
    """
    try:
        # 获取股市热门关键词
        df = ak.stock_hot_keyword()
        
        return df
    except Exception as e:
        print(f"获取股市热门关键词时出错: {e}")
        return pd.DataFrame() 