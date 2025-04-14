#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试DeepSeek FinRobot市场预测代理
"""

import os
import sys
from deepseek_finrobot.utils import get_deepseek_config
from deepseek_finrobot.agents import MarketForecasterAgent

def test_market_forecaster():
    """测试市场预测代理"""
    print("=" * 50)
    print("测试市场预测代理")
    print("=" * 50)
    
    try:
        # 从配置文件获取API密钥
        config = get_deepseek_config()
        api_key = config[0]['api_key']
        print(f"获取到API密钥: {api_key[:5]}...{api_key[-4:]}")
        
        # 设置环境变量
        os.environ["DEEPSEEK_API_KEY"] = api_key
        
        # 创建LLM配置
        llm_config = {
            "config_list": config,
            "temperature": 0.7,
        }
        
        # 创建市场预测代理
        print("\n创建市场预测代理...")
        forecaster = MarketForecasterAgent(llm_config)
        
        # 进行预测测试
        print("\n预测股票: 000001 (平安银行)")
        prediction = forecaster.predict("000001", days=3)
        
        print("\n预测结果:")
        print("-" * 50)
        print(prediction.strip())
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"\n代理测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_market_forecaster()
    sys.exit(0 if success else 1) 