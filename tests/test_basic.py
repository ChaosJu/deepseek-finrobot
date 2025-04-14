#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek FinRobot 基本功能测试脚本
用于验证安装是否成功
"""

print("开始测试基本功能...")

try:
    print("导入工具模块...")
    from deepseek_finrobot.utils import get_current_date, format_financial_number
    
    print(f"当前日期: {get_current_date()}")
    print(f"格式化数字: {format_financial_number(1234567.89)}")
    
    print("基本工具函数测试成功!")
    
    # 尝试导入openai_adapter模块
    print("导入API适配器模块...")
    from deepseek_finrobot.openai_adapter import get_completion, get_chat_completion
    print("API适配器模块导入成功!")
    
    # 注意：以下代码需要设置环境变量 DEEPSEEK_API_KEY 才能正常运行
    # print("测试API调用... (需要设置DEEPSEEK_API_KEY环境变量)")
    # response = get_completion("你好，请介绍自己")
    # print(f"API响应: {response[:100]}...")
    
    print("所有基本功能测试成功!")
except Exception as e:
    print(f"测试失败: {str(e)}")
    import traceback
    traceback.print_exc() 