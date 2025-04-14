#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试DeepSeek FinRobot基本功能的脚本
"""

import sys
import os

try:
    # 导入工具函数
    from deepseek_finrobot.utils import get_current_date, format_financial_number
    
    # 导入适配器函数（如果已安装openai和适配器已更新）
    try:
        from deepseek_finrobot.openai_adapter import get_completion
        has_openai = True
    except ImportError:
        has_openai = False
    
    # 测试基本工具函数
    print("=" * 50)
    print("测试基本工具函数:")
    print(f"当前日期: {get_current_date()}")
    print(f"格式化金额: {format_financial_number(1234567.89)}")
    print("=" * 50)
    
    # 测试openai适配器（如果已安装）
    if has_openai:
        # 检查环境变量是否设置
        has_api_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
        print("\n测试DeepSeek API适配器:")
        if has_api_key:
            try:
                response = get_completion("你好，请简单介绍一下你自己。", max_tokens=100)
                print(f"API响应: {response[:100]}...")
                print("DeepSeek API测试成功！")
            except Exception as e:
                print(f"DeepSeek API测试失败: {e}")
        else:
            print("未设置DEEPSEEK_API_KEY环境变量，跳过API测试")
        print("=" * 50)
    
    print("\n安装验证完成！基本功能测试成功。")
    
except Exception as e:
    print(f"测试失败: {e}")
    print("请检查安装是否正确完成。")
    sys.exit(1) 