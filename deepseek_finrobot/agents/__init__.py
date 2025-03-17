"""
代理模块 - 提供各种专业金融代理
"""

from .agent_library import MarketForecasterAgent, FinancialReportAgent, NewsAnalysisAgent, IndustryAnalysisAgent, PortfolioManagerAgent, TechnicalAnalysisAgent
from .workflow import SingleAssistant, SingleAssistantShadow, MultiAgentWorkflow

__all__ = [
    'MarketForecasterAgent',
    'FinancialReportAgent',
    'NewsAnalysisAgent',
    'IndustryAnalysisAgent',
    'PortfolioManagerAgent',
    'TechnicalAnalysisAgent',
    'SingleAssistant',
    'SingleAssistantShadow',
    'MultiAgentWorkflow'
] 