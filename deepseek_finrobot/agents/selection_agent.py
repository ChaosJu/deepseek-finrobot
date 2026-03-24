"""
智能选股代理 - 提供多因子量化选股与多代理共识能力
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import autogen
import numpy as np
import pandas as pd

from ..data_source import akshare_utils, cn_news_utils


class StockSelectorAgent:
    """
    智能选股代理

    特性：
    1) 多因子评分：动量、波动、估值、流动性、质量
    2) 权重自适应：根据风险偏好和投资期限动态调整
    3) 分散约束：限制单一行业入选数量
    4) AutoGen群聊共识（可选）：量化/基本面/风控多代理协同给出结论
    """

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.llm_config = llm_config

    @staticmethod
    def _safe_float(value: Any) -> float:
        """将任意输入尽可能转换为float，失败时返回NaN。"""
        if value is None:
            return np.nan
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if cleaned in {"", "N/A", "nan", "None", "--"}:
                return np.nan
            matched = re.search(r"-?\d+(\.\d+)?", cleaned)
            if matched:
                try:
                    return float(matched.group(0))
                except ValueError:
                    return np.nan
        return np.nan

    @staticmethod
    def _extract_metric(df: pd.DataFrame, candidate_columns: List[str]) -> float:
        """从财务DataFrame中按候选列提取首个可用数值。"""
        if df is None or df.empty:
            return np.nan
        for col in candidate_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if not series.empty:
                    return float(series.iloc[0])
        return np.nan

    @staticmethod
    def _normalize(values: List[float], reverse: bool = False) -> List[float]:
        """Min-Max归一化，缺失值回填0.5。"""
        arr = np.array(values, dtype=float)
        valid_mask = np.isfinite(arr)
        if valid_mask.sum() == 0:
            return [0.5] * len(values)

        valid = arr[valid_mask]
        v_min = float(np.min(valid))
        v_max = float(np.max(valid))

        norm = np.full_like(arr, 0.5, dtype=float)
        if not math.isclose(v_min, v_max):
            norm[valid_mask] = (valid - v_min) / (v_max - v_min)
        else:
            norm[valid_mask] = 0.5

        if reverse:
            norm = 1.0 - norm
        return np.clip(norm, 0.0, 1.0).tolist()

    @staticmethod
    def _risk_weights(risk_preference: str) -> Dict[str, float]:
        """不同风险偏好的基础因子权重。"""
        mapping = {
            "保守": {
                "quality": 0.30,
                "valuation": 0.25,
                "momentum": 0.10,
                "volatility": 0.25,
                "liquidity": 0.10,
            },
            "中等": {
                "quality": 0.22,
                "valuation": 0.20,
                "momentum": 0.25,
                "volatility": 0.15,
                "liquidity": 0.18,
            },
            "激进": {
                "quality": 0.15,
                "valuation": 0.10,
                "momentum": 0.40,
                "volatility": 0.10,
                "liquidity": 0.25,
            },
        }
        return mapping.get(risk_preference, mapping["中等"]).copy()

    @staticmethod
    def _horizon_adjustments(investment_horizon: str) -> Dict[str, float]:
        """不同投资期限对因子进行再加权。"""
        mapping = {
            "短期": {
                "quality": 0.85,
                "valuation": 0.85,
                "momentum": 1.25,
                "volatility": 1.10,
                "liquidity": 1.20,
            },
            "中期": {
                "quality": 1.00,
                "valuation": 1.00,
                "momentum": 1.00,
                "volatility": 1.00,
                "liquidity": 1.00,
            },
            "长期": {
                "quality": 1.25,
                "valuation": 1.20,
                "momentum": 0.85,
                "volatility": 0.95,
                "liquidity": 0.90,
            },
        }
        return mapping.get(investment_horizon, mapping["中期"]).copy()

    def _build_final_weights(self, risk_preference: str, investment_horizon: str) -> Dict[str, float]:
        """融合风险偏好与投资期限，得到最终权重。"""
        base = self._risk_weights(risk_preference)
        adj = self._horizon_adjustments(investment_horizon)
        weighted = {k: base[k] * adj.get(k, 1.0) for k in base}
        total = sum(weighted.values()) or 1.0
        return {k: v / total for k, v in weighted.items()}

    def _collect_symbol_metrics(self, symbol: str, lookback_days: int = 260) -> Optional[Dict[str, Any]]:
        """采集单个股票的多因子原始指标。"""
        info = akshare_utils.get_stock_info(symbol)
        if not info or "error" in info:
            return None

        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime("%Y%m%d")
        history = akshare_utils.get_stock_history(symbol, start_date=start_date, end_date=end_date)
        if history is None or history.empty or "收盘" not in history.columns:
            return None

        close = pd.to_numeric(history["收盘"], errors="coerce").dropna()
        if len(close) < 30:
            return None

        volume_series = pd.to_numeric(history["成交量"], errors="coerce").dropna() if "成交量" in history.columns else pd.Series(dtype=float)
        returns = close.pct_change().dropna()

        momentum_20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 21 else np.nan
        momentum_60 = float(close.iloc[-1] / close.iloc[-61] - 1.0) if len(close) > 61 else np.nan
        trend_win_rate = float((returns.tail(60) > 0).mean()) if len(returns) >= 20 else np.nan
        volatility_60 = float(returns.tail(60).std() * math.sqrt(252)) if len(returns) >= 20 else np.nan

        pe = self._safe_float(info.get("市盈率(动态)", info.get("市盈率")))
        pb = self._safe_float(info.get("市净率"))
        liquidity = float(volume_series.tail(20).mean()) if not volume_series.empty else self._safe_float(info.get("成交量"))

        financial = akshare_utils.get_stock_financial_indicator(symbol)
        roe = self._extract_metric(financial, ["净资产收益率(%)", "净资产收益率", "ROE", "净资产收益率-摊薄"])
        profit_growth = self._extract_metric(financial, ["净利润同比增长率(%)", "净利润同比增长率", "净利润同比"])
        revenue_growth = self._extract_metric(financial, ["营业收入同比增长率(%)", "营业收入同比增长率", "营业收入同比"])

        quality_components = [x for x in [roe, profit_growth, revenue_growth] if np.isfinite(x)]
        quality_raw = float(np.mean(quality_components)) if quality_components else np.nan

        momentum_raw = float(np.nanmean([momentum_20, momentum_60])) if np.isfinite(momentum_20) or np.isfinite(momentum_60) else np.nan
        valuation_raw = float(np.nanmean([pe, pb])) if np.isfinite(pe) or np.isfinite(pb) else np.nan

        return {
            "symbol": symbol,
            "name": info.get("股票简称", symbol),
            "industry": info.get("所处行业", "未知") or "未知",
            "raw_metrics": {
                "momentum_raw": momentum_raw,
                "trend_win_rate": trend_win_rate,
                "volatility_raw": volatility_60,
                "valuation_raw": valuation_raw,
                "liquidity_raw": liquidity,
                "quality_raw": quality_raw,
                "pe": pe,
                "pb": pb,
                "roe": roe,
                "profit_growth": profit_growth,
                "revenue_growth": revenue_growth,
            },
        }

    @staticmethod
    def _market_regime_bias(market_sentiment: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """根据市场情绪给出因子偏置（轻量，避免过拟合）。"""
        if not market_sentiment or "error" in market_sentiment:
            return {"momentum": 0.0, "valuation": 0.0, "volatility": 0.0}

        trend = market_sentiment.get("market_trend", {})
        sentiment = trend.get("sentiment", "中性")
        trend_dir = trend.get("trend", "未知")

        if sentiment == "积极" or trend_dir == "上涨":
            return {"momentum": 0.03, "valuation": -0.01, "volatility": -0.01}
        if sentiment == "消极" or trend_dir == "下跌":
            return {"momentum": -0.02, "valuation": 0.02, "volatility": 0.03}
        return {"momentum": 0.0, "valuation": 0.0, "volatility": 0.0}

    def _run_groupchat_consensus(
        self,
        selected_candidates: List[Dict[str, Any]],
        risk_preference: str,
        investment_horizon: str,
        market_sentiment: Optional[Dict[str, Any]],
    ) -> str:
        """使用AutoGen GroupChat生成多代理共识意见。"""
        if not self.llm_config:
            return "未启用共识分析：缺少llm_config。"

        try:
            user_proxy = autogen.UserProxyAgent(
                name="ChiefPM",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            quant_analyst = autogen.AssistantAgent(
                name="QuantAnalyst",
                llm_config=self.llm_config,
                system_message="你是量化分析师，重点评估因子有效性、风险收益比和组合分散度。",
            )
            fundamental_analyst = autogen.AssistantAgent(
                name="FundamentalAnalyst",
                llm_config=self.llm_config,
                system_message="你是基本面分析师，重点关注估值、盈利质量与增长可持续性。",
            )
            risk_manager = autogen.AssistantAgent(
                name="RiskManager",
                llm_config=self.llm_config,
                system_message="你是风控经理，重点评估波动、回撤风险与行业集中度，给出约束建议。",
            )

            groupchat = autogen.GroupChat(
                agents=[user_proxy, quant_analyst, fundamental_analyst, risk_manager],
                messages=[],
                max_round=8,
                speaker_selection_method="round_robin",
            )
            manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

            compact_candidates = []
            for c in selected_candidates:
                compact_candidates.append(
                    {
                        "symbol": c["symbol"],
                        "name": c["name"],
                        "industry": c["industry"],
                        "score": round(c["score"], 4),
                        "factors": {k: round(v, 4) for k, v in c["factors"].items()},
                    }
                )

            prompt = (
                f"请对以下选股结果做多代理共识评审，并输出最终结论：\n"
                f"- 风险偏好: {risk_preference}\n"
                f"- 投资期限: {investment_horizon}\n"
                f"- 市场情绪: {json.dumps(market_sentiment, ensure_ascii=False) if market_sentiment else '未知'}\n"
                f"- 候选池(已按分数排序):\n{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}\n\n"
                "请给出：\n"
                "1. 最终前3只核心标的及理由\n"
                "2. 组合层面的主要风险点\n"
                "3. 仓位与再平衡建议（简要）\n"
                "结尾请用“共识结论：”开头给出一句话总结。"
            )

            user_proxy.initiate_chat(manager, message=prompt, clear_history=True)

            if manager.name in user_proxy.chat_messages and user_proxy.chat_messages[manager.name]:
                return user_proxy.chat_messages[manager.name][-1].get("content", "")
            if groupchat.messages:
                return groupchat.messages[-1].get("content", "")
            return "共识分析已执行，但未捕获到有效输出。"
        except Exception as e:
            return f"共识分析执行失败: {e}"

    def select_stocks(
        self,
        symbols: List[str],
        top_n: int = 5,
        risk_preference: str = "中等",
        investment_horizon: str = "长期",
        max_per_industry: int = 2,
        use_consensus: bool = False,
    ) -> Dict[str, Any]:
        """
        多因子智能选股。
        """
        try:
            # region agent log
            os.makedirs("/opt/cursor/logs", exist_ok=True)
            open("/opt/cursor/logs/debug.log", "a", encoding="utf-8").write(
                json.dumps(
                    {
                        "hypothesisId": "A",
                        "location": "selection_agent.py:302",
                        "message": "select_stocks_entry",
                        "data": {"input_symbol_count": len(symbols), "top_n": top_n},
                        "timestamp": int(datetime.datetime.now().timestamp() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            # endregion
        except Exception:
            pass

        unique_symbols = []
        seen = set()
        for s in symbols:
            symbol = str(s).strip()
            if symbol and symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)

        metrics = []
        for symbol in unique_symbols:
            try:
                m = self._collect_symbol_metrics(symbol)
            except Exception as e:
                # 单个标的网络抖动不应导致整轮选股失败
                print(f"采集股票 {symbol} 指标失败，已跳过: {e}")
                m = None
            if m:
                metrics.append(m)
            try:
                # region agent log
                os.makedirs("/opt/cursor/logs", exist_ok=True)
                open("/opt/cursor/logs/debug.log", "a", encoding="utf-8").write(
                    json.dumps(
                        {
                            "hypothesisId": "A",
                            "location": "selection_agent.py:323",
                            "message": "collect_symbol_metrics_result",
                            "data": {"symbol": symbol, "success": bool(m)},
                            "timestamp": int(datetime.datetime.now().timestamp() * 1000),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # endregion
            except Exception:
                pass

        if not metrics:
            return {
                "error": "没有可用的候选股票数据，请检查候选池或稍后重试。",
                "ranked_candidates": [],
                "selected_symbols": [],
            }

        # 组装因子原始值
        momentum_values = [
            0.7 * m["raw_metrics"]["momentum_raw"] + 0.3 * m["raw_metrics"]["trend_win_rate"]
            if np.isfinite(m["raw_metrics"]["momentum_raw"]) or np.isfinite(m["raw_metrics"]["trend_win_rate"])
            else np.nan
            for m in metrics
        ]
        valuation_values = [m["raw_metrics"]["valuation_raw"] for m in metrics]
        volatility_values = [m["raw_metrics"]["volatility_raw"] for m in metrics]
        liquidity_values = [m["raw_metrics"]["liquidity_raw"] for m in metrics]
        quality_values = [m["raw_metrics"]["quality_raw"] for m in metrics]

        # 归一化（估值、波动为逆向因子）
        momentum_scores = self._normalize(momentum_values, reverse=False)
        valuation_scores = self._normalize(valuation_values, reverse=True)
        volatility_scores = self._normalize(volatility_values, reverse=True)
        liquidity_scores = self._normalize(liquidity_values, reverse=False)
        quality_scores = self._normalize(quality_values, reverse=False)

        weights = self._build_final_weights(risk_preference, investment_horizon)
        market_sentiment = cn_news_utils.get_stock_market_sentiment()
        regime_bias = self._market_regime_bias(market_sentiment)

        ranked_candidates = []
        for idx, m in enumerate(metrics):
            factors = {
                "quality": quality_scores[idx],
                "valuation": valuation_scores[idx],
                "momentum": momentum_scores[idx],
                "volatility": volatility_scores[idx],
                "liquidity": liquidity_scores[idx],
            }
            base_score = sum(weights[k] * factors[k] for k in factors)
            # 轻量市场偏置，增强策略“环境自适应”能力
            adjusted_score = (
                base_score
                + regime_bias["momentum"] * factors["momentum"]
                + regime_bias["valuation"] * factors["valuation"]
                + regime_bias["volatility"] * factors["volatility"]
            )
            final_score = float(np.clip(adjusted_score, 0.0, 1.0))

            ranked_candidates.append(
                {
                    "symbol": m["symbol"],
                    "name": m["name"],
                    "industry": m["industry"],
                    "score": final_score,
                    "factors": factors,
                    "raw_metrics": m["raw_metrics"],
                }
            )

        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 行业分散约束
        selected = []
        industry_counts: Dict[str, int] = {}
        for candidate in ranked_candidates:
            if len(selected) >= top_n:
                break
            industry = candidate["industry"] or "未知"
            if max_per_industry > 0 and industry_counts.get(industry, 0) >= max_per_industry:
                continue
            selected.append(candidate)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1

        # 若约束过强导致数量不足，则补齐
        if len(selected) < top_n:
            selected_symbols = {s["symbol"] for s in selected}
            for candidate in ranked_candidates:
                if len(selected) >= top_n:
                    break
                if candidate["symbol"] in selected_symbols:
                    continue
                selected.append(candidate)
                selected_symbols.add(candidate["symbol"])
                industry = candidate["industry"] or "未知"
                industry_counts[industry] = industry_counts.get(industry, 0) + 1

        result: Dict[str, Any] = {
            "strategy": "multi_factor_v1",
            "weights": weights,
            "market_bias": regime_bias,
            "ranked_candidates": ranked_candidates,
            "selected_candidates": selected,
            "selected_symbols": [c["symbol"] for c in selected],
            "industry_distribution": industry_counts,
        }

        if use_consensus:
            result["consensus"] = self._run_groupchat_consensus(
                selected_candidates=selected,
                risk_preference=risk_preference,
                investment_horizon=investment_horizon,
                market_sentiment=market_sentiment if isinstance(market_sentiment, dict) else None,
            )

        return result

    @staticmethod
    def format_selection_report(result: Dict[str, Any], top_show: int = 10) -> str:
        """格式化选股结果为可读文本。"""
        if "error" in result:
            return f"选股失败: {result['error']}"

        lines = []
        lines.append("# 智能选股报告")
        lines.append("")
        lines.append(f"- 策略版本: {result.get('strategy', 'unknown')}")
        lines.append(f"- 最终入选: {', '.join(result.get('selected_symbols', []))}")
        lines.append(f"- 行业分布: {json.dumps(result.get('industry_distribution', {}), ensure_ascii=False)}")
        lines.append("")
        lines.append("## 因子权重")
        for k, v in result.get("weights", {}).items():
            lines.append(f"- {k}: {v:.2%}")
        lines.append("")
        lines.append("## 候选排名")

        ranked = result.get("ranked_candidates", [])[:top_show]
        for i, c in enumerate(ranked, 1):
            lines.append(
                f"{i}. {c['symbol']} {c['name']} | 行业:{c['industry']} | "
                f"总分:{c['score']:.4f} | "
                f"动量:{c['factors']['momentum']:.3f} 估值:{c['factors']['valuation']:.3f} "
                f"质量:{c['factors']['quality']:.3f} 波动:{c['factors']['volatility']:.3f} "
                f"流动性:{c['factors']['liquidity']:.3f}"
            )

        if result.get("consensus"):
            lines.append("")
            lines.append("## AutoGen 多代理共识")
            lines.append(str(result["consensus"]))

        return "\n".join(lines)

    @staticmethod
    def export_selection_report(result: Dict[str, Any], fmt: str = "markdown", output_file: Optional[str] = None) -> str:
        """导出选股报告。"""
        content = StockSelectorAgent.format_selection_report(result)
        if fmt == "html":
            content = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>智能选股报告</title></head><body><pre>"
                + content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                + "</pre></body></html>"
            )
        elif fmt == "text":
            # markdown本身已可读，这里保持同样内容
            pass

        if output_file:
            if fmt == "html" and not output_file.endswith((".html", ".htm")):
                output_file += ".html"
            elif fmt == "text" and not output_file.endswith(".txt"):
                output_file += ".txt"
            elif fmt == "markdown" and not output_file.endswith((".md", ".markdown")):
                output_file += ".md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            return output_file
        return content
