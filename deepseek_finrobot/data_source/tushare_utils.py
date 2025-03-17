"""
TuShare数据源工具 - 提供中国市场金融数据
"""

import tushare as ts
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import datetime
import matplotlib.pyplot as plt
import os
import json

# 获取TuShare API Token
def get_tushare_token() -> str:
    """
    获取TuShare API Token
    
    Returns:
        TuShare API Token
    """
    # 尝试从环境变量获取
    token = os.environ.get("TUSHARE_TOKEN", "")
    
    # 如果环境变量中没有，尝试从配置文件获取
    if not token:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config_api_keys.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    token = config.get("TUSHARE_TOKEN", "")
            except Exception as e:
                print(f"读取配置文件出错: {str(e)}")
    
    return token

# 初始化TuShare
def init_tushare():
    """
    初始化TuShare
    """
    token = get_tushare_token()
    if token:
        ts.set_token(token)
        return ts.pro_api()
    else:
        print("警告: 未设置TuShare Token，部分功能可能受限")
        return None

# 初始化TuShare API
pro = init_tushare()

def get_stock_basic_info(symbol: str = None) -> pd.DataFrame:
    """
    获取股票基本信息
    
    Args:
        symbol: 股票代码（如果为None则获取所有股票）
    
    Returns:
        股票基本信息DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 获取所有股票列表
        data = pro.stock_basic(exchange='', list_status='L', 
                              fields='ts_code,symbol,name,area,industry,list_date')
        
        # 如果指定了股票代码，则筛选
        if symbol:
            # 处理股票代码格式
            if symbol.startswith('6'):
                ts_code = f"{symbol}.SH"
            else:
                ts_code = f"{symbol}.SZ"
            
            data = data[data['ts_code'] == ts_code]
            
            if data.empty:
                return pd.DataFrame({"error": [f"未找到股票代码 {symbol}"]})
        
        return data
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_stock_daily_data(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取股票日线数据
    
    Args:
        symbol: 股票代码
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
    
    Returns:
        股票日线数据DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 处理日期
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y%m%d')
        
        if start_date is None:
            # 默认获取最近一年的数据
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
        
        # 处理股票代码格式
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        # 获取日线数据
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到股票 {symbol} 在指定日期范围的数据"]})
        
        # 按日期排序
        df = df.sort_values('trade_date')
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_stock_financial_data(symbol: str, period: str = None) -> pd.DataFrame:
    """
    获取股票财务数据
    
    Args:
        symbol: 股票代码
        period: 报告期，如20231231
    
    Returns:
        股票财务数据DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 处理股票代码格式
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        # 获取财务指标数据
        if period:
            df = pro.fina_indicator(ts_code=ts_code, period=period)
        else:
            df = pro.fina_indicator(ts_code=ts_code)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到股票 {symbol} 的财务数据"]})
        
        # 按报告期排序
        df = df.sort_values('end_date', ascending=False)
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_stock_income_statement(symbol: str, period: str = None) -> pd.DataFrame:
    """
    获取股票利润表
    
    Args:
        symbol: 股票代码
        period: 报告期，如20231231
    
    Returns:
        股票利润表DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 处理股票代码格式
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        # 获取利润表数据
        if period:
            df = pro.income(ts_code=ts_code, period=period)
        else:
            df = pro.income(ts_code=ts_code)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到股票 {symbol} 的利润表数据"]})
        
        # 按报告期排序
        df = df.sort_values('end_date', ascending=False)
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_stock_balance_sheet(symbol: str, period: str = None) -> pd.DataFrame:
    """
    获取股票资产负债表
    
    Args:
        symbol: 股票代码
        period: 报告期，如20231231
    
    Returns:
        股票资产负债表DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 处理股票代码格式
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        # 获取资产负债表数据
        if period:
            df = pro.balancesheet(ts_code=ts_code, period=period)
        else:
            df = pro.balancesheet(ts_code=ts_code)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到股票 {symbol} 的资产负债表数据"]})
        
        # 按报告期排序
        df = df.sort_values('end_date', ascending=False)
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_stock_cash_flow(symbol: str, period: str = None) -> pd.DataFrame:
    """
    获取股票现金流量表
    
    Args:
        symbol: 股票代码
        period: 报告期，如20231231
    
    Returns:
        股票现金流量表DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 处理股票代码格式
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        # 获取现金流量表数据
        if period:
            df = pro.cashflow(ts_code=ts_code, period=period)
        else:
            df = pro.cashflow(ts_code=ts_code)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到股票 {symbol} 的现金流量表数据"]})
        
        # 按报告期排序
        df = df.sort_values('end_date', ascending=False)
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_stock_dividend(symbol: str) -> pd.DataFrame:
    """
    获取股票分红信息
    
    Args:
        symbol: 股票代码
    
    Returns:
        股票分红信息DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 处理股票代码格式
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        # 获取分红信息
        df = pro.dividend(ts_code=ts_code)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到股票 {symbol} 的分红信息"]})
        
        # 按除权除息日排序
        df = df.sort_values('ex_date', ascending=False)
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_stock_major_holders(symbol: str) -> pd.DataFrame:
    """
    获取股票主要股东信息
    
    Args:
        symbol: 股票代码
    
    Returns:
        股票主要股东信息DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 处理股票代码格式
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
        else:
            ts_code = f"{symbol}.SZ"
        
        # 获取主要股东信息
        df = pro.top10_holders(ts_code=ts_code)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到股票 {symbol} 的主要股东信息"]})
        
        # 按报告期和持股比例排序
        df = df.sort_values(['end_date', 'hold_ratio'], ascending=[False, False])
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_index_constituents(index_code: str) -> pd.DataFrame:
    """
    获取指数成分股
    
    Args:
        index_code: 指数代码，如000001.SH（上证指数）、399001.SZ（深证成指）
    
    Returns:
        指数成分股DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 获取指数成分股
        df = pro.index_weight(index_code=index_code)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到指数 {index_code} 的成分股信息"]})
        
        # 按权重排序
        df = df.sort_values('weight', ascending=False)
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_industry_list() -> pd.DataFrame:
    """
    获取行业列表
    
    Returns:
        行业列表DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 获取行业列表
        df = pro.index_classify(level='L1', src='SW')
        
        if df.empty:
            return pd.DataFrame({"error": ["未找到行业列表信息"]})
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def get_industry_stocks(industry_code: str) -> pd.DataFrame:
    """
    获取行业成分股
    
    Args:
        industry_code: 行业代码
    
    Returns:
        行业成分股DataFrame
    """
    try:
        if pro is None:
            return pd.DataFrame({"error": ["未设置TuShare Token"]})
        
        # 获取行业成分股
        df = pro.index_member(index_code=industry_code)
        
        if df.empty:
            return pd.DataFrame({"error": [f"未找到行业 {industry_code} 的成分股信息"]})
        
        # 获取股票基本信息
        stocks = get_stock_basic_info()
        
        # 合并股票信息
        if "error" not in stocks.columns:
            df = pd.merge(df, stocks, on='ts_code', how='left')
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def plot_stock_price(symbol: str, start_date: str = None, end_date: str = None, ma: List[int] = [5, 20, 60], figsize: tuple = (12, 6)) -> None:
    """
    绘制股票价格走势图
    
    Args:
        symbol: 股票代码
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        ma: 移动平均线天数列表
        figsize: 图表大小
    """
    try:
        # 获取股票日线数据
        df = get_stock_daily_data(symbol, start_date, end_date)
        
        if "error" in df.columns:
            print(f"获取股票数据出错: {df['error'][0]}")
            return
        
        # 计算移动平均线
        for m in ma:
            df[f'MA{m}'] = df['close'].rolling(window=m).mean()
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 绘制收盘价
        plt.plot(df['trade_date'], df['close'], label='收盘价')
        
        # 绘制移动平均线
        for m in ma:
            plt.plot(df['trade_date'], df[f'MA{m}'], label=f'{m}日均线')
        
        # 设置图表属性
        plt.title(f"{symbol} 股价走势图")
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # 显示图表
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"绘制股票价格走势图出错: {str(e)}") 