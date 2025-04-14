#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek FinRobot 项目全面测试脚本
"""

import os
import sys

def test_utils():
    """测试工具函数模块"""
    print("\n=== 测试工具函数模块 ===")
    try:
        from deepseek_finrobot.utils import (
            get_current_date, format_financial_number, 
            get_deepseek_config, get_deepseek_config_from_api_keys
        )
        
        print(f"当前日期: {get_current_date()}")
        print(f"格式化金额: {format_financial_number(1234567.89)}")
        print(f"格式化小额: {format_financial_number(123.45)}")
        print(f"格式化大额: {format_financial_number(123456789.12)}")
        
        # 测试config相关函数，但不实际加载配置
        print("\n测试配置相关函数:")
        os.environ["DEEPSEEK_API_KEY"] = "dummy-api-key-for-testing"
        config = get_deepseek_config_from_api_keys()
        print(f"配置列表长度: {len(config)}")
        print(f"配置项示例: {config[0]}")
        
        return True
    except Exception as e:
        print(f"工具函数测试失败: {e}")
        return False
        
def test_openai_adapter():
    """测试OpenAI适配器模块"""
    print("\n=== 测试OpenAI适配器模块 ===")
    try:
        from deepseek_finrobot.openai_adapter import (
            get_llm_config_for_autogen
        )
        
        os.environ["DEEPSEEK_API_KEY"] = "dummy-api-key-for-testing"
        config = get_llm_config_for_autogen(
            model="deepseek-chat",
            temperature=0.7
        )
        
        print(f"适配器配置: {config}")
        print(f"配置列表长度: {len(config['config_list'])}")
        print(f"模型名称: {config['config_list'][0]['model']}")
        
        return True
    except Exception as e:
        print(f"OpenAI适配器测试失败: {e}")
        return False
        
def test_agents():
    """测试代理模块"""
    print("\n=== 测试代理模块 ===")
    try:
        from deepseek_finrobot.agents import (
            MarketForecasterAgent, FinancialReportAgent, 
            NewsAnalysisAgent, IndustryAnalysisAgent,
            PortfolioManagerAgent, TechnicalAnalysisAgent
        )
        
        # 创建一个通用的LLM配置
        llm_config = {
            "config_list": [
                {
                    "model": "deepseek-chat",
                    "api_key": "dummy-api-key-for-testing",
                    "base_url": "https://api.deepseek.com/v1",
                }
            ],
            "temperature": 0.7,
        }
        
        # 测试各种代理的实例化
        print("\n测试代理实例化:")
        market = MarketForecasterAgent(llm_config)
        print(f"- MarketForecasterAgent: {type(market).__name__}")
        
        report = FinancialReportAgent(llm_config)
        print(f"- FinancialReportAgent: {type(report).__name__}")
        
        news = NewsAnalysisAgent(llm_config)
        print(f"- NewsAnalysisAgent: {type(news).__name__}")
        
        industry = IndustryAnalysisAgent(llm_config)
        print(f"- IndustryAnalysisAgent: {type(industry).__name__}")
        
        portfolio = PortfolioManagerAgent(llm_config)
        print(f"- PortfolioManagerAgent: {type(portfolio).__name__}")
        
        technical = TechnicalAnalysisAgent(llm_config)
        print(f"- TechnicalAnalysisAgent: {type(technical).__name__}")
        
        return True
    except Exception as e:
        print(f"代理模块测试失败: {e}")
        return False
        
def test_cli():
    """测试命令行接口"""
    print("\n=== 测试命令行接口 ===")
    try:
        import deepseek_finrobot.cli as cli
        print(f"CLI模块: {cli.__name__}")
        
        # 检查是否有主要的命令行接口相关函数或变量
        functions = [
            name for name in dir(cli) 
            if callable(getattr(cli, name)) and not name.startswith('_')
        ]
        print(f"CLI模块可用函数: {', '.join(functions) if functions else '无'}")
        
        # 直接查看命令行帮助
        import subprocess
        result = subprocess.run(
            ["python", "-m", "deepseek_finrobot.cli", "--help"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("\nCLI帮助信息:")
            commands = []
            for line in result.stdout.split('\n'):
                if "positional arguments" in line:
                    print(f"  {line.strip()}")
                elif "{" in line and "}" in line:
                    commands_str = line[line.find("{")+1:line.find("}")].strip()
                    commands = [cmd.strip() for cmd in commands_str.split(',')]
                    print(f"  可用命令: {', '.join(commands)}")
        
        return True
    except Exception as e:
        print(f"CLI模块测试失败: {e}")
        return False

if __name__ == "__main__":
    print("\n==================================================")
    print("DeepSeek FinRobot 项目全面测试开始")
    print("==================================================")
    
    success = []
    
    # 测试工具函数模块
    if test_utils():
        success.append("工具函数模块")
    
    # 测试OpenAI适配器模块
    if test_openai_adapter():
        success.append("OpenAI适配器模块")
    
    # 测试代理模块
    if test_agents():
        success.append("代理模块")
    
    # 测试命令行接口
    if test_cli():
        success.append("命令行接口")
    
    # 打印测试结果
    print("\n==================================================")
    print(f"测试完成! 成功模块: {len(success)}/{4}")
    for i, module in enumerate(success, 1):
        print(f"{i}. {module}")
    print("==================================================")
    
    sys.exit(0 if len(success) == 4 else 1) 