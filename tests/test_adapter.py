#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试DeepSeek API适配器
"""

import os
from deepseek_finrobot.openai_adapter import get_llm_config_for_autogen

print("=" * 50)
print("测试DeepSeek API适配器")
print("=" * 50)

# 设置dummy API密钥
os.environ["DEEPSEEK_API_KEY"] = "dummy-api-key-for-testing"

try:
    # 获取AutoGen兼容的配置
    config = get_llm_config_for_autogen(
        model="deepseek-chat",
        temperature=0.7,
        base_url="https://api.deepseek.com/v1"
    )
    
    print("\n成功创建AutoGen兼容配置:")
    print(f"模型: {config['config_list'][0]['model']}")
    print(f"温度: {config['temperature']}")
    print(f"基础URL: {config['config_list'][0]['base_url']}")
    print("=" * 50)
    
    # 尝试导入autogen
    try:
        import autogen
        print("\n成功导入autogen模块")
        
        # 尝试创建autogen代理
        assistant = autogen.AssistantAgent(
            name="金融助手",
            llm_config=config,
            system_message="您是一位专业的金融分析师，擅长分析市场趋势和投资机会。"
        )
        print("\n成功创建AutoGen金融助手代理")
        
    except ImportError:
        print("\nautogen模块导入失败，但这不影响适配器的功能测试")
    
except Exception as e:
    print(f"\n创建配置时出错: {e}") 