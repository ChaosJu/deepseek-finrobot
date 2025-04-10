"""
TuShare数据源工具 - 提供中国市场金融数据（已移除Tushare依赖，改用AKShare）
"""

# 移除tushare依赖
# import tushare as ts
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import datetime
import matplotlib.pyplot as plt
import os
import json
from . import akshare_utils

# 一个警告标记，用于显示此函数已经重定向到akshare或禁用
def warn_tushare_disabled(func_name):
    print(f"警告: TuShare功能 '{func_name}' 已被禁用，请使用AKShare替代")
    return pd.DataFrame({"message": ["TuShare功能已禁用，请使用AKShare替代"]})

# 获取TuShare API Token
def get_tushare_token() -> str:
    """
    获取TuShare API Token（已禁用）
    
    Returns:
        空字符串
    """
    return ""

# 初始化TuShare
def init_tushare():
    """
    初始化TuShare（已禁用）
    """
    print("警告: TuShare功能已被禁用，请使用AKShare替代")
    return None

# 初始化TuShare API
pro = None

def get_stock_basic_info(symbol: str = None) -> pd.DataFrame:
    """
    获取股票基本信息（重定向到AKShare）
    
    Args:
        symbol: 股票代码（如果为None则获取所有股票）
    
    Returns:
        股票基本信息DataFrame
    """
    if symbol:
        return akshare_utils.get_stock_info(symbol)
    else:
        return akshare_utils.get_stock_list()

def get_stock_daily_data(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取股票日线数据（重定向到AKShare）
    
    Args:
        symbol: 股票代码
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
    
    Returns:
        股票日线数据DataFrame
    """
    return akshare_utils.get_stock_history(symbol, start_date=start_date, end_date=end_date)

def get_stock_financial_data(symbol: str, period: str = None) -> pd.DataFrame:
    """
    获取股票财务数据（重定向到AKShare）
    
    Args:
        symbol: 股票代码
        period: 报告期，如20231231
    
    Returns:
        股票财务数据DataFrame
    """
    return akshare_utils.get_stock_financial_indicator(symbol)

def get_stock_income_statement(symbol: str, period: str = None) -> pd.DataFrame:
    """
    获取股票利润表（已禁用）
    
    Args:
        symbol: 股票代码
        period: 报告期，如20231231
    
    Returns:
        空DataFrame
    """
    return warn_tushare_disabled("get_stock_income_statement")

def get_stock_balance_sheet(symbol: str, period: str = None) -> pd.DataFrame:
    """
    获取股票资产负债表（已禁用）
    
    Args:
        symbol: 股票代码
        period: 报告期，如20231231
    
    Returns:
        空DataFrame
    """
    return warn_tushare_disabled("get_stock_balance_sheet")

def get_stock_cash_flow(symbol: str, period: str = None) -> pd.DataFrame:
    """
    获取股票现金流量表（已禁用）
    
    Args:
        symbol: 股票代码
        period: 报告期，如20231231
    
    Returns:
        空DataFrame
    """
    return warn_tushare_disabled("get_stock_cash_flow")

def get_stock_dividend(symbol: str) -> pd.DataFrame:
    """
    获取股票分红数据（已禁用）
    
    Args:
        symbol: 股票代码
    
    Returns:
        空DataFrame
    """
    return warn_tushare_disabled("get_stock_dividend")

def get_stock_major_holders(symbol: str) -> pd.DataFrame:
    """
    获取股票主要股东数据（已禁用）
    
    Args:
        symbol: 股票代码
    
    Returns:
        空DataFrame
    """
    return warn_tushare_disabled("get_stock_major_holders")

def get_index_constituents(index_code: str) -> pd.DataFrame:
    """
    获取指数成分股（重定向到AKShare）
    
    Args:
        index_code: 指数代码
    
    Returns:
        指数成分股DataFrame
    """
    return akshare_utils.get_index_constituents(index_code)

def get_industry_list() -> pd.DataFrame:
    """
    获取行业列表（重定向到AKShare）
    
    Returns:
        行业列表DataFrame
    """
    return akshare_utils.get_stock_industry_list()

def get_industry_stocks(industry_code: str) -> pd.DataFrame:
    """
    获取行业股票（重定向到AKShare）
    
    Args:
        industry_code: 行业代码
    
    Returns:
        行业股票DataFrame
    """
    return akshare_utils.get_stock_industry_constituents(industry_code)

def plot_stock_price(symbol: str, start_date: str = None, end_date: str = None, ma: List[int] = [5, 20, 60], figsize: tuple = (12, 6)) -> None:
    """
    绘制股票价格走势图（重定向到AKShare）
    
    Args:
        symbol: 股票代码
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        ma: 移动平均线列表
        figsize: 图表大小
    """
    # 获取股票历史数据
    data = akshare_utils.get_stock_history(symbol, start_date=start_date, end_date=end_date)
    
    if data.empty or "error" in data.columns:
        print(f"无法获取股票 {symbol} 数据")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制价格线
    ax.plot(data.index, data['收盘'], label='收盘价')
    
    # 绘制移动平均线
    for m in ma:
        if len(data) > m:
            ma_name = f'MA{m}'
            data[ma_name] = data['收盘'].rolling(window=m).mean()
            ax.plot(data.index, data[ma_name], label=ma_name)
    
    # 设置图表标题和标签
    ax.set_title(f"{symbol} 价格走势图")
    ax.set_xlabel('日期')
    ax.set_ylabel('价格')
    ax.legend()
    ax.grid(True)
    
    # 显示图表
    plt.tight_layout()
    plt.show() 