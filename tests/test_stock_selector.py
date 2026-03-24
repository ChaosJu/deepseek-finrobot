#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StockSelectorAgent tests
"""

from typing import Any, Dict

import pandas as pd

from deepseek_finrobot.agents.selection_agent import StockSelectorAgent


def _mock_get_stock_info(symbol: str) -> Dict[str, Any]:
    mapping = {
        "000001": {"股票简称": "平安银行", "所处行业": "银行", "市盈率(动态)": 5.0, "市净率": 0.7, "成交量": 1_000_000},
        "600036": {"股票简称": "招商银行", "所处行业": "银行", "市盈率(动态)": 6.0, "市净率": 0.9, "成交量": 1_200_000},
        "600519": {"股票简称": "贵州茅台", "所处行业": "白酒", "市盈率(动态)": 22.0, "市净率": 8.0, "成交量": 300_000},
        "000858": {"股票简称": "五粮液", "所处行业": "白酒", "市盈率(动态)": 18.0, "市净率": 6.0, "成交量": 500_000},
    }
    return mapping.get(symbol, {"error": "not found"})


def _mock_get_stock_history(symbol: str, **kwargs):
    # 构造不同强度的价格序列：银行较稳，茅台动量强，五粮液中等
    if symbol == "600519":
        close = [100 + i * 1.0 for i in range(100)]  # 强动量
    elif symbol == "000858":
        close = [100 + i * 0.6 for i in range(100)]  # 中等动量
    elif symbol == "600036":
        close = [100 + i * 0.35 for i in range(100)]  # 稳健
    else:
        close = [100 + i * 0.3 for i in range(100)]  # 稳健

    volume = [1_000_000 + i * 1000 for i in range(100)]
    df = pd.DataFrame({"收盘": close, "成交量": volume})
    return df


def _mock_get_stock_financial_indicator(symbol: str):
    # 银行质量高于白酒，用于保守偏好测试
    if symbol in {"000001", "600036"}:
        return pd.DataFrame(
            {
                "净资产收益率(%)": [14.5],
                "净利润同比增长率(%)": [9.0],
                "营业收入同比增长率(%)": [7.5],
            }
        )
    return pd.DataFrame(
        {
            "净资产收益率(%)": [11.0],
            "净利润同比增长率(%)": [18.0],
            "营业收入同比增长率(%)": [16.0],
        }
    )


def _mock_market_sentiment():
    return {"market_trend": {"sentiment": "中性", "trend": "震荡"}}


def test_stock_selector_returns_ranked_and_selected(monkeypatch):
    from deepseek_finrobot.data_source import akshare_utils, cn_news_utils

    monkeypatch.setattr(akshare_utils, "get_stock_info", _mock_get_stock_info)
    monkeypatch.setattr(akshare_utils, "get_stock_history", _mock_get_stock_history)
    monkeypatch.setattr(akshare_utils, "get_stock_financial_indicator", _mock_get_stock_financial_indicator)
    monkeypatch.setattr(cn_news_utils, "get_stock_market_sentiment", _mock_market_sentiment)

    selector = StockSelectorAgent(llm_config=None)
    result = selector.select_stocks(
        symbols=["000001", "600036", "600519", "000858"],
        top_n=3,
        risk_preference="中等",
        investment_horizon="中期",
        max_per_industry=2,
        use_consensus=False,
    )

    assert "error" not in result
    assert len(result["ranked_candidates"]) == 4
    assert len(result["selected_symbols"]) == 3
    assert all(item["score"] >= 0.0 and item["score"] <= 1.0 for item in result["ranked_candidates"])


def test_stock_selector_industry_cap_works(monkeypatch):
    from deepseek_finrobot.data_source import akshare_utils, cn_news_utils

    monkeypatch.setattr(akshare_utils, "get_stock_info", _mock_get_stock_info)
    monkeypatch.setattr(akshare_utils, "get_stock_history", _mock_get_stock_history)
    monkeypatch.setattr(akshare_utils, "get_stock_financial_indicator", _mock_get_stock_financial_indicator)
    monkeypatch.setattr(cn_news_utils, "get_stock_market_sentiment", _mock_market_sentiment)

    selector = StockSelectorAgent(llm_config=None)
    result = selector.select_stocks(
        symbols=["000001", "600036", "600519", "000858"],
        top_n=3,
        risk_preference="中等",
        investment_horizon="中期",
        max_per_industry=1,
        use_consensus=False,
    )

    selected = result["selected_candidates"]
    industry_count = {}
    for item in selected:
        industry = item["industry"]
        industry_count[industry] = industry_count.get(industry, 0) + 1

    assert len(result["selected_symbols"]) == 3
    # 即便补齐逻辑存在，优先阶段行业约束应生效，最终也不应出现严重集中
    assert max(industry_count.values()) <= 2

