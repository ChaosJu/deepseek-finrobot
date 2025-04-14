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
import pypinyin

def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    获取股票基本信息
    
    Args:
        symbol: 股票代码（如：000001，不带市场前缀）
        
    Returns:
        股票信息字典
    """
    try:
        # 获取股票基本信息
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        
        if stock_info.empty:
            return {"error": "未找到股票信息"}
            
        # 转换为字典
        info_dict = {}
        for _, row in stock_info.iterrows():
            info_dict[row['item']] = row['value']
        
        # 补充获取实时行情数据
        try:
            # 获取A股实时行情
            realtime_data = ak.stock_zh_a_spot_em()
            
            # 只在调试时输出列名
            # print(f"实时行情数据列: {realtime_data.columns.tolist()}")
            
            # 过滤指定股票
            realtime_data = realtime_data[realtime_data['代码'] == symbol]
            
            if not realtime_data.empty:
                # 安全获取最新价
                if '最新价' in realtime_data.columns:
                    info_dict["最新价"] = realtime_data['最新价'].iloc[0]
                
                # 正确处理市盈率字段 - 测试结果显示字段名为"市盈率-动态"
                if '市盈率-动态' in realtime_data.columns:
                    pe_value = realtime_data['市盈率-动态'].iloc[0]
                    info_dict["市盈率"] = pe_value
                    info_dict["市盈率(动态)"] = pe_value
                
                # 获取市净率
                if '市净率' in realtime_data.columns:
                    info_dict["市净率"] = realtime_data['市净率'].iloc[0]
                
                # 获取行业信息
                # 通常股票基本信息中应该包含行业，但保险起见也从实时数据补充
                if ('所处行业' not in info_dict or info_dict['所处行业'] == '未知') and '行业' in realtime_data.columns:
                    info_dict["所处行业"] = realtime_data['行业'].iloc[0]
                
                # 添加其他有用的行情数据
                for key in ['涨跌幅', '成交量', '换手率', '总市值', '流通市值']:
                    if key in realtime_data.columns:
                        info_dict[key] = realtime_data[key].iloc[0]
            
        except Exception as e:
            print(f"获取实时行情数据出错: {e}")
        
        # 确保关键字段存在
        for key in ["最新价", "市盈率", "市盈率(动态)", "市净率", "所处行业"]:
            if key not in info_dict:
                info_dict[key] = "N/A"
            
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
            
        # 调用AKShare获取A股历史数据
        df = ak.stock_zh_a_hist(
            symbol=symbol, 
            period=period, 
            start_date=start_date, 
            end_date=end_date, 
            adjust=adjust
        )
        
        if df.empty:
            print(f"获取股票历史数据为空: {symbol}")
            return pd.DataFrame()
            
        # 重命名列，确保列名为小写
        df.columns = [col.lower() for col in df.columns]
        
        # 处理日期列并转换为索引
        date_columns = ['日期', 'date']
        date_col = next((col for col in date_columns if col in df.columns), None)
        
        if date_col:
            # 确保日期格式一致
            if isinstance(df[date_col].iloc[0], str):
                df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        
        # 确保数值列是浮点数
        numeric_cols = ['开盘', '收盘', '最高', '最低', '成交额', '涨跌幅', '涨跌额', '振幅', '换手率']
        for col in [c.lower() for c in numeric_cols]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
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
        
        # 确保返回标准字段名称
        field_mapping = {
            '最新价': '最新价',
            '涨跌幅': '涨跌幅',
            '涨跌额': '涨跌额', 
            '成交量': '成交量',
            '成交额': '成交额',
            '振幅': '振幅',
            '最高': '最高',
            '最低': '最低',
            '今开': '今开',
            '昨收': '昨收',
            '量比': '量比',
            '换手率': '换手率',
            '市盈率-动态': '市盈率',
            '市净率': '市净率',
            '总市值': '总市值',
            '流通市值': '流通市值'
        }
        
        standardized_result = {}
        for original_key, standard_key in field_mapping.items():
            if original_key in result:
                standardized_result[standard_key] = result[original_key]
            else:
                standardized_result[standard_key] = None
                
        # 添加代码和名称字段
        standardized_result['代码'] = symbol
        if '名称' in result:
            standardized_result['名称'] = result['名称']
        
        return standardized_result
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
        
        # 检查并处理列名变化问题
        if 'content' not in df.columns and '内容' in df.columns:
            df = df.rename(columns={'内容': 'content'})
        if 'title' not in df.columns and '标题' in df.columns:
            df = df.rename(columns={'标题': 'title'})
            
        # 如果仍然没有content列，则创建一个空的content列
        if 'content' not in df.columns:
            df['content'] = df.apply(lambda row: row.iloc[0] if len(row) > 0 else "", axis=1)
            print("警告: 新闻数据结构已变化，已自动适配")
            
        # 确保title列存在
        if 'title' not in df.columns:
            first_col_name = df.columns[0] if len(df.columns) > 0 else "新闻"
            df['title'] = df[first_col_name]
            print(f"警告: 新闻标题列不存在，已使用{first_col_name}列作为标题")
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"获取股票新闻时出错: {e}")
        # 返回一个包含必要列的空DataFrame
        return pd.DataFrame(columns=['title', 'content'])

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
        # 获取所有新闻
        df = ak.stock_news_em()
        
        # 检查并处理列名变化问题
        if 'content' not in df.columns and '内容' in df.columns:
            df = df.rename(columns={'内容': 'content'})
        if 'title' not in df.columns and '标题' in df.columns:
            df = df.rename(columns={'标题': 'title'})
            
        # 如果仍然没有content列，创建一个
        if 'content' not in df.columns:
            df['content'] = df.apply(lambda row: row.iloc[0] if len(row) > 0 else "", axis=1)
            print("警告: 行业新闻数据结构已变化，已自动适配")
            
        # 确保title列存在
        if 'title' not in df.columns:
            first_col_name = df.columns[0] if len(df.columns) > 0 else "新闻"
            df['title'] = df[first_col_name]
            print(f"警告: 行业新闻标题列不存在，已使用{first_col_name}列作为标题")
        
        # 现在我们尝试过滤所有内容中包含行业名称的新闻
        # 首先在标题中查找
        mask = df['title'].str.contains(industry, na=False)
        
        # 如果content列存在，也在内容中查找
        if 'content' in df.columns:
            mask = mask | df['content'].str.contains(industry, na=False)
        
        # 过滤行业新闻
        df = df[mask]
        
        if limit and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        print(f"获取行业新闻时出错: {e}")
        # 返回一个包含必要列的空DataFrame
        return pd.DataFrame(columns=['title', 'content'])

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
    
    Note:
        使用pypinyin模块可以将行业名称转换为拼音，用于排序和搜索
        使用示例: 
        from pypinyin import lazy_pinyin
        df['拼音'] = df['板块名称'].apply(lambda x: ''.join(lazy_pinyin(x)))
    """
    try:
        # 获取行业列表
        df = ak.stock_board_industry_name_em()
        
        # 确保列名一致性
        if '板块名称' not in df.columns and '板块名' in df.columns:
            df = df.rename(columns={'板块名': '板块名称'})
        if '板块名称' not in df.columns and '名称' in df.columns:
            df = df.rename(columns={'名称': '板块名称'})
        if '板块代码' not in df.columns and '代码' in df.columns:
            df = df.rename(columns={'代码': '板块代码'})
            
        print(f"成功获取行业列表，共找到 {len(df)} 个行业")
        return df
    except Exception as e:
        print(f"获取股票行业列表时出错: {e}")
        # 尝试替代方法获取行业列表
        try:
            # 尝试使用板块行情接口
            print("尝试使用替代接口获取行业列表...")
            df = ak.stock_sector_spot(indicator="行业")
            
            # 重命名列以匹配原来的接口
            if '板块名称' not in df.columns and '板块' in df.columns:
                df = df.rename(columns={'板块': '板块名称'})
                
            print(f"成功使用替代接口获取行业列表，共找到 {len(df)} 个行业")
            return df
        except Exception as inner_e:
            print(f"使用替代接口获取行业列表时出错: {inner_e}")
            
            # 如果所有方法都失败，创建一个固定的小型行业列表作为备选
            fallback_industries = {
                '银行': 'BK0475', 
                '医药': 'BK0465',
                '食品饮料': 'BK0438',
                '电子': 'BK0448',
                '计算机': 'BK0447',
                '有色金属': 'BK0478',
                '房地产': 'BK0451'
            }
            
            # 创建备选DataFrame
            df = pd.DataFrame({
                '板块名称': list(fallback_industries.keys()),
                '板块代码': list(fallback_industries.values())
            })
            
            print(f"使用内置备选行业列表，共 {len(df)} 个行业")
            return df

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
    :param industry_code: 行业代码
    :return: 成分股数据
    """
    try:
        # 使用东方财富行业成分股接口
        df = ak.stock_board_industry_cons_em(symbol=industry_code)
        
        # 确保必要的列存在
        required_columns = {
            '代码': str,
            '名称': str,
            '最新价': float,
            '涨跌幅': float,
            '市盈率': float,
            '市净率': float
        }
        
        # 重命名列（如果需要）
        rename_map = {
            '股票代码': '代码',
            '股票名称': '名称',
            '市盈率-动态': '市盈率'
        }
        df = df.rename(columns=rename_map)
        
        # 添加缺失的列并设置默认值
        for col, dtype in required_columns.items():
            if col not in df.columns:
                df[col] = dtype(0)
            df[col] = df[col].astype(dtype)
            
        # 如果市盈率列为空，尝试获取个股数据
        if '市盈率' in df.columns and df['市盈率'].isna().any():
            for idx, row in df.iterrows():
                try:
                    stock_info = ak.stock_a_lg_indicator(symbol=row['代码'])
                    if not stock_info.empty and '市盈率' in stock_info.columns:
                        df.at[idx, '市盈率'] = stock_info['市盈率'].iloc[0]
                except:
                    df.at[idx, '市盈率'] = 0.0
                    
        # 选择需要的列
        df = df[list(required_columns.keys())]
        return df
    except Exception as e:
        print(f"获取行业成分股失败: {e}")
        # 返回空DataFrame，但包含所需的列
        return pd.DataFrame(columns=list(required_columns.keys()))

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
        
        # 如果DataFrame为空，返回默认数据
        if df.empty:
            print("未获取到研究报告数据，创建默认数据")
            data = {
                'title': [f"{symbol or '市场'}行业分析报告"],
                'author': ["分析师团队"]
            }
            return pd.DataFrame(data)
        
        # 标准化列名
        if '报告名称' in df.columns and 'title' not in df.columns:
            df = df.rename(columns={'报告名称': 'title'})
        elif '标题' in df.columns and 'title' not in df.columns:
            df = df.rename(columns={'标题': 'title'})
            
        if '研究员' in df.columns and 'author' not in df.columns:
            df = df.rename(columns={'研究员': 'author'})
        elif '分析师' in df.columns and 'author' not in df.columns:
            df = df.rename(columns={'分析师': 'author'})
        
        # 确保必要的列存在
        if 'title' not in df.columns:
            df['title'] = f"{symbol or '市场'}行业分析报告"
            
        if 'author' not in df.columns:
            df['author'] = "未知分析师"
            
        return df
    except Exception as e:
        print(f"获取股票研究报告时出错: {e}")
        # 返回一个默认的研究报告数据
        data = {
            'title': [f"{symbol or '市场'}行业分析报告"],
            'author': ["分析师团队"]
        }
        return pd.DataFrame(data)

def get_stock_concept_history(concept_code: str, period: str = "daily", 
                            start_date: str = None, end_date: str = None,
                            adjust: str = "") -> pd.DataFrame:
    """
    获取板块概念的历史行情数据
    
    Args:
        concept_code: 概念代码（如：BK0815 表示半导体概念）
        period: 时间周期，可选 daily, weekly, monthly
        start_date: 开始日期，格式 YYYYMMDD，默认为近一年
        end_date: 结束日期，格式 YYYYMMDD，默认为今天
        adjust: 复权类型，默认为不复权
        
    Returns:
        概念历史数据DataFrame
    
    Note:
        概念代码可以通过get_stock_concept_list函数获取
    """
    try:
        # 设置默认日期
        if not end_date:
            end_date = datetime.datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
            
        # 调用AKShare获取概念历史数据
        df = ak.stock_board_concept_hist_em(
            symbol=concept_code, 
            period=period, 
            start_date=start_date, 
            end_date=end_date, 
            adjust=adjust
        )
        
        if df.empty:
            print(f"获取概念历史数据为空: {concept_code}")
            return pd.DataFrame()
        
        # 重命名列，确保列名为小写
        df.columns = [col.lower() for col in df.columns]
        
        # 处理日期列并转换为索引
        date_columns = ['日期', 'date']
        date_col = next((col for col in date_columns if col in df.columns), None)
        
        if date_col:
            # 确保日期格式一致
            if isinstance(df[date_col].iloc[0], str):
                df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            
        return df
    except Exception as e:
        print(f"获取概念历史数据时出错: {e}")
        return pd.DataFrame()

def get_stock_industry_hist_min(industry_code: str, period: str = "1", 
                              adjust: str = "") -> pd.DataFrame:
    """
    获取板块行业的分钟级历史行情数据
    
    Args:
        industry_code: 行业代码（如：BK0475 表示银行行业）
        period: 分钟周期，可选 1, 5, 15, 30, 60
        adjust: 复权类型，默认为不复权
        
    Returns:
        行业分钟级历史数据DataFrame
    
    Note:
        行业代码可以通过get_stock_industry_list函数获取
    """
    try:
        # 调用AKShare获取行业分钟级历史数据
        df = ak.stock_board_industry_hist_min_em(
            symbol=industry_code, 
            period=period, 
            adjust=adjust
        )
        
        if df.empty:
            print(f"获取行业分钟级历史数据为空: {industry_code}")
            return pd.DataFrame()
        
        # 重命名列，确保列名为小写
        df.columns = [col.lower() for col in df.columns]
        
        # 处理日期列并转换为索引
        time_columns = ['时间', 'time', 'datetime']
        time_col = next((col for col in time_columns if col in df.columns), None)
        
        if time_col:
            # 确保日期格式一致
            if isinstance(df[time_col].iloc[0], str):
                df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
            
        return df
    except Exception as e:
        print(f"获取行业分钟级历史数据时出错: {e}")
        return pd.DataFrame()

def get_stock_concept_hist_min(concept_code: str, period: str = "1", 
                             adjust: str = "") -> pd.DataFrame:
    """
    获取板块概念的分钟级历史行情数据
    
    Args:
        concept_code: 概念代码（如：BK0815 表示半导体概念）
        period: 分钟周期，可选 1, 5, 15, 30, 60
        adjust: 复权类型，默认为不复权
        
    Returns:
        概念分钟级历史数据DataFrame
    
    Note:
        概念代码可以通过get_stock_concept_list函数获取
    """
    try:
        # 调用AKShare获取概念分钟级历史数据
        df = ak.stock_board_concept_hist_min_em(
            symbol=concept_code, 
            period=period, 
            adjust=adjust
        )
        
        if df.empty:
            print(f"获取概念分钟级历史数据为空: {concept_code}")
            return pd.DataFrame()
        
        # 重命名列，确保列名为小写
        df.columns = [col.lower() for col in df.columns]
        
        # 处理日期列并转换为索引
        time_columns = ['时间', 'time', 'datetime']
        time_col = next((col for col in time_columns if col in df.columns), None)
        
        if time_col:
            # 确保日期格式一致
            if isinstance(df[time_col].iloc[0], str):
                df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
            
        return df
    except Exception as e:
        print(f"获取概念分钟级历史数据时出错: {e}")
        return pd.DataFrame() 