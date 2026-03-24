#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from deepseek_finrobot.agents.selection_agent import StockSelectorAgent
from deepseek_finrobot.data_source import akshare_utils, cn_news_utils


def test_get_stock_industry_list_fallback_normalizes_board_code(monkeypatch):
    def mock_primary():
        raise RuntimeError("primary api failed")

    def mock_fallback(indicator=None):
        return pd.DataFrame({"label": ["BK0475"], "板块": ["银行"]})

    monkeypatch.setattr(akshare_utils.ak, "stock_board_industry_name_em", mock_primary)
    monkeypatch.setattr(akshare_utils.ak, "stock_sector_spot", mock_fallback)

    df = akshare_utils.get_stock_industry_list()
    assert "板块名称" in df.columns
    assert "板块代码" in df.columns
    assert str(df.iloc[0]["板块代码"]) == "BK0475"


def test_get_stock_market_sentiment_uses_cache(monkeypatch):
    cn_news_utils._SENTIMENT_CACHE["timestamp"] = 0.0
    cn_news_utils._SENTIMENT_CACHE["data"] = None

    calls = {"index": 0, "north": 0, "activity": 0}

    def mock_index_spot():
        calls["index"] += 1
        return pd.DataFrame({"名称": ["上证指数"], "涨跌幅": [0.5], "最新价": [3100]})

    def mock_north_hist():
        calls["north"] += 1
        return pd.DataFrame({"日期": ["2026-03-24"], "当日资金流入": [12.3]})

    def mock_activity():
        calls["activity"] += 1
        return pd.DataFrame([{"日期": "2026-03-24", "活跃度": "中等"}])

    monkeypatch.setattr(cn_news_utils.ak, "stock_zh_index_spot_sina", mock_index_spot)
    monkeypatch.setattr(cn_news_utils.ak, "stock_hsgt_hist_em", mock_north_hist)
    monkeypatch.setattr(cn_news_utils.ak, "stock_market_activity_legu", mock_activity)
    if hasattr(cn_news_utils.ak, "stock_em_hsgt_north_net_flow_in_hist"):
        monkeypatch.setattr(cn_news_utils.ak, "stock_em_hsgt_north_net_flow_in_hist", lambda: pd.DataFrame())
    if hasattr(cn_news_utils.ak, "stock_hsgt_north_net_flow_em"):
        monkeypatch.setattr(cn_news_utils.ak, "stock_hsgt_north_net_flow_em", lambda: pd.DataFrame())

    first = cn_news_utils.get_stock_market_sentiment()
    second = cn_news_utils.get_stock_market_sentiment()

    assert "market_trend" in first
    assert "north_flow" in first
    assert first == second
    assert calls["index"] == 1
    assert calls["north"] == 1
    assert calls["activity"] == 1


def test_get_stock_info_reuses_snapshot_cache(monkeypatch):
    akshare_utils._SPOT_CACHE["timestamp"] = 0.0
    akshare_utils._SPOT_CACHE["data"] = None

    calls = {"spot": 0}

    def mock_info(symbol):
        return pd.DataFrame({"item": ["股票简称"], "value": [f"name-{symbol}"]})

    def mock_spot():
        calls["spot"] += 1
        return pd.DataFrame(
            {
                "代码": ["000001", "000002"],
                "最新价": [10.0, 11.0],
                "市盈率-动态": [5.0, 6.0],
                "市净率": [1.0, 1.1],
                "行业": ["银行", "银行"],
                "涨跌幅": [0.1, 0.2],
                "成交量": [1000, 1200],
                "换手率": [0.5, 0.6],
                "总市值": [100000, 120000],
                "流通市值": [80000, 90000],
            }
        )

    monkeypatch.setattr(akshare_utils.ak, "stock_individual_info_em", mock_info)
    monkeypatch.setattr(akshare_utils.ak, "stock_zh_a_spot_em", mock_spot)

    r1 = akshare_utils.get_stock_info("000001")
    r2 = akshare_utils.get_stock_info("000002")

    assert "error" not in r1
    assert "error" not in r2
    assert calls["spot"] == 1


def test_selection_agent_skips_symbol_collection_error(monkeypatch):
    selector = StockSelectorAgent(llm_config=None)

    def mock_collect(symbol):
        if symbol == "BAD":
            raise RuntimeError("network error")
        return {
            "symbol": symbol,
            "name": symbol,
            "industry": "测试",
            "raw_metrics": {
                "momentum_raw": 0.1,
                "trend_win_rate": 0.5,
                "volatility_raw": 0.2,
                "valuation_raw": 10.0,
                "liquidity_raw": 1000.0,
                "quality_raw": 1.0,
                "pe": 10.0,
                "pb": 1.0,
                "roe": 12.0,
                "profit_growth": 8.0,
                "revenue_growth": 7.0,
            },
        }

    monkeypatch.setattr(selector, "_collect_symbol_metrics", mock_collect)
    monkeypatch.setattr(cn_news_utils, "get_stock_market_sentiment", lambda: {"market_trend": {"sentiment": "中性", "trend": "震荡"}})

    result = selector.select_stocks(symbols=["BAD", "GOOD"], top_n=1)
    assert "error" not in result
    assert result["selected_symbols"] == ["GOOD"]
