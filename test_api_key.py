#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试DeepSeek API密钥有效性
"""

import os
import sys
from deepseek_finrobot.utils import get_deepseek_config
from deepseek_finrobot.openai_adapter import get_completion

def test_api_key():
    """测试API密钥是否有效"""
    print("=" * 50)
    print("测试DeepSeek API密钥")
    print("=" * 50)
    
    try:
        # 从配置文件获取API密钥
        config = get_deepseek_config()
        api_key = config[0]['api_key']
        print(f"获取到API密钥: {api_key[:5]}...{api_key[-4:]}")
        
        # 设置环境变量
        os.environ["DEEPSEEK_API_KEY"] = api_key
        
        # 进行一个简单的API调用
        print("\n发起API调用...")
        response = get_completion(
            prompt="你好，请用一句话介绍一下你自己。",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=100
        )
        
        print("\nAPI调用成功！")
        print("响应内容:")
        print("-" * 50)
        print(response.strip())
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"\nAPI调用失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_key()
    sys.exit(0 if success else 1) 