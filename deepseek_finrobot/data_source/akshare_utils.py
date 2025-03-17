"""
AKShare数据源工具 - 用于获取中国市场的金融数据
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
import datetime
import os

def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    获取股票基本信息
    
    Args:
        symbol: 股票代码（如：000001，不带市场前缀）
        
    Returns:
        股票信息字典
    """
    try:
        # 获取股票实时行情
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        
        if stock_info.empty:
            return {"error": "未找到股票信息"}
            
        # 转换为字典
        info_dict = {}
        for _, row in stock_info.iterrows():
            info_dict[row['item']] = row['value']
            
        return info_dict
    except Exception as e:
        print(f"获取股票信息时出错: {e}")
        return {"error": str(e)}

def get_stock_history(symbol: str, period: str = "daily", 
                     start_date: str = None, end_date: str = None,
                     adjust: str = "qfq") -> pd.DataFrame:
    """
    获取股票历史行情数据
    
    Args:
        symbol: 股票代码（如：000001，不带市场前缀）
        period: 时间周期，可选 daily, weekly, monthly
        start_date: 开始日期，格式 YYYYMMDD，默认为近一年
        end_date: 结束日期，格式 YYYYMMDD，默认为今天
        adjust: 复权类型，qfq: 前复权, hfq: 后复权, 空: 不复权
        
    Returns:
        股票历史数据DataFrame
    """
    try:
        # 设置默认日期
        if not end_date:
            end_date = datetime.datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
            
        # 根据周期选择不同的函数
        if period == "daily":
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                   start_date=start_date, end_date=end_date, 
                                   adjust=adjust)
        elif period == "weekly":
            df = ak.stock_zh_a_hist(symbol=symbol, period="weekly", 
                                   start_date=start_date, end_date=end_date, 
                                   adjust=adjust)
        elif period == "monthly":
            df = ak.stock_zh_a_hist(symbol=symbol, period="monthly", 
                                   start_date=start_date, end_date=end_date, 
                                   adjust=adjust)
        else:
            return pd.DataFrame()
            
        # 重命名列
        df.columns = [col.lower() for col in df.columns]
        
        # 将日期列转换为索引
        if '日期' in df.columns:
            df = df.set_index('日期')
        elif 'date' in df.columns:
            df = df.set_index('date')
            
        return df
    except Exception as e:
        print(f"获取股票历史数据时出错: {e}")
        return pd.DataFrame()

def get_stock_realtime_quote(symbol: str) -> Dict[str, Any]:
    """
    获取股票实时行情
    
    Args:
        symbol: 股票代码（如：000001，不带市场前缀）
        
    Returns:
        股票实时行情字典
    """
    try:
        # 获取A股实时行情
        df = ak.stock_zh_a_spot_em()
        
        # 过滤指定股票
        df = df[df['代码'] == symbol]
        
        if df.empty:
            return {"error": "未找到股票实时行情"}
            
        # 转换为字典
        result = df.iloc[0].to_dict()
        
        return result
    except Exception as e:
        print(f"获取股票实时行情时出错: {e}")
        return {"error": str(e)}

def get_stock_financial_indicator(symbol: str) -> pd.DataFrame:
    """
    获取股票财务指标
    
    Args:
        symbol: 股票代码（如：000001，不带市场前缀）
        
    Returns:
        股票财务指标DataFrame
    """
    try:
        # 获取财务指标
        df = ak.stock_financial_analysis_indicator(symbol=symbol)
        
        return df
    except Exception as e:
        print(f"获取股票财务指标时出错: {e}")
        return pd.DataFrame()

def get_stock_news(limit: int = 10) -> pd.DataFrame:
    """
    获取股票相关新闻
    
    Args:
        limit: 返回的新闻数量
        
    Returns:
        新闻DataFrame
    """
    try:
        # 获取财经新闻
        df = ak.stock_news_em()
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"获取股票新闻时出错: {e}")
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
        # 获取行业新闻
        df = ak.stock_news_em()
        
        # 过滤行业新闻
        df = df[df['content'].str.contains(industry)]
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"获取行业新闻时出错: {e}")
        return pd.DataFrame()

def get_stock_index_data(symbol: str = "000001", period: str = "daily",
                        start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取股票指数数据
    
    Args:
        symbol: 指数代码（如：000001 表示上证指数）
        period: 时间周期，可选 daily, weekly, monthly
        start_date: 开始日期，格式 YYYYMMDD，默认为近一年
        end_date: 结束日期，格式 YYYYMMDD，默认为今天
        
    Returns:
        指数数据DataFrame
    """
    try:
        # 设置默认日期
        if not end_date:
            end_date = datetime.datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
            
        # 获取指数数据
        df = ak.stock_zh_index_daily(symbol=symbol)
        
        # 过滤日期
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # 根据周期重采样
        if period == "weekly":
            df = df.resample('W').last()
        elif period == "monthly":
            df = df.resample('M').last()
            
        return df
    except Exception as e:
        print(f"获取股票指数数据时出错: {e}")
        return pd.DataFrame()

def plot_stock_price(symbol: str, period: str = "daily", 
                    start_date: str = None, end_date: str = None,
                    ma: List[int] = [5, 20, 60], figsize: tuple = (12, 6)) -> plt.Figure:
    """
    绘制股票价格图表
    
    Args:
        symbol: 股票代码
        period: 时间周期
        start_date: 开始日期
        end_date: 结束日期
        ma: 移动平均线天数列表
        figsize: 图表大小
        
    Returns:
        matplotlib图表对象
    """
    try:
        # 获取股票数据
        df = get_stock_history(symbol, period, start_date, end_date)
        
        if df.empty:
            return None
            
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制收盘价
        ax.plot(df.index, df['收盘'], label='收盘价', color='blue')
        
        # 添加移动平均线
        for m in ma:
            if len(df) > m:
                df[f'MA{m}'] = df['收盘'].rolling(window=m).mean()
                ax.plot(df.index, df[f'MA{m}'], label=f'{m}日均线')
                
        # 设置图表标题和标签
        ax.set_title(f'{symbol} 股票价格', fontsize=16)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('价格 (元)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 格式化x轴日期
        fig.autofmt_xdate()
        
        return fig
    except Exception as e:
        print(f"绘制股票价格图表时出错: {e}")
        return None

def get_stock_industry_list() -> pd.DataFrame:
    """
    获取股票行业列表
    
    Returns:
        行业列表DataFrame
    """
    try:
        # 获取行业列表
        df = ak.stock_sector_spot_em()
        
        return df
    except Exception as e:
        print(f"获取股票行业列表时出错: {e}")
        return pd.DataFrame()

def get_stock_concept_list() -> pd.DataFrame:
    """
    获取股票概念列表
    
    Returns:
        概念列表DataFrame
    """
    try:
        # 获取概念列表
        df = ak.stock_board_concept_name_em()
        
        return df
    except Exception as e:
        print(f"获取股票概念列表时出错: {e}")
        return pd.DataFrame()

def get_stock_industry_constituents(industry_code: str) -> pd.DataFrame:
    """
    获取行业成分股
    
    Args:
        industry_code: 行业代码
        
    Returns:
        行业成分股DataFrame
    """
    try:
        # 获取行业成分股
        df = ak.stock_board_industry_cons_em(symbol=industry_code)
        
        return df
    except Exception as e:
        print(f"获取行业成分股时出错: {e}")
        return pd.DataFrame()

def get_stock_concept_constituents(concept_code: str) -> pd.DataFrame:
    """
    获取概念成分股
    
    Args:
        concept_code: 概念代码
        
    Returns:
        概念成分股DataFrame
    """
    try:
        # 获取概念成分股
        df = ak.stock_board_concept_cons_em(symbol=concept_code)
        
        return df
    except Exception as e:
        print(f"获取概念成分股时出错: {e}")
        return pd.DataFrame() 