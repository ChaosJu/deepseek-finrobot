"""
DeepSeek API 工具 - 提供DeepSeek API访问功能 (兼容OpenAI API 1.9.0+)
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncIterator, Iterator

def get_completion(
    prompt: str, 
    model: str = "deepseek-chat", 
    temperature: float = 0.7, 
    max_tokens: int = 1000, 
    **kwargs
) -> str:
    """
    DeepSeek文本补全接口
    
    Args:
        prompt: 提示文本
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大生成token数
        **kwargs: 其他参数
        
    Returns:
        生成的文本
    """
    # 适配新版OpenAI API
    from openai import OpenAI
    
    # 从环境变量获取API密钥
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("未设置DEEPSEEK_API_KEY环境变量")
    
    # 构建API请求
    base_url = kwargs.get("base_url", "https://api.deepseek.com/v1")
    
    # 创建客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 创建请求参数
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # 添加其他参数
    for key, value in kwargs.items():
        if key not in ["base_url"]:
            params[key] = value
    
    # 发送请求
    response = client.chat.completions.create(**params)
    
    # 返回结果
    return response.choices[0].message.content

def get_chat_completion(
    messages: List[Dict[str, str]], 
    model: str = "deepseek-chat", 
    temperature: float = 0.7, 
    max_tokens: int = 1000, 
    **kwargs
) -> str:
    """
    DeepSeek对话补全接口
    
    Args:
        messages: 对话消息列表
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大生成token数
        **kwargs: 其他参数
        
    Returns:
        生成的回复文本
    """
    # 适配新版OpenAI API
    from openai import OpenAI
    
    # 从环境变量获取API密钥
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("未设置DEEPSEEK_API_KEY环境变量")
    
    # 构建API请求
    base_url = kwargs.get("base_url", "https://api.deepseek.com/v1")
    
    # 创建客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 创建请求参数
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # 添加其他参数
    for key, value in kwargs.items():
        if key not in ["base_url"]:
            params[key] = value
    
    # 发送请求
    response = client.chat.completions.create(**params)
    
    # 返回结果
    return response.choices[0].message.content

def get_llm_config_for_autogen(api_key: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    获取适用于autogen的LLM配置 (兼容新版autogen)
    
    Args:
        api_key: DeepSeek API密钥，如果为None则使用环境变量
        **kwargs: 其他配置参数
        
    Returns:
        LLM配置字典
    """
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key
    
    # 确保API密钥存在于环境变量中
    if "DEEPSEEK_API_KEY" not in os.environ or not os.environ["DEEPSEEK_API_KEY"]:
        raise ValueError("未设置DEEPSEEK_API_KEY环境变量或通过参数提供API密钥")
    
    # 基础配置
    config = {
        "temperature": kwargs.get("temperature", 0.7),
        "model": kwargs.get("model", "deepseek-chat"),
    }
    
    # 构建配置列表 (适配新版autogen)
    base_url = kwargs.get("base_url", "https://api.deepseek.com/v1")
    config_list = [{
        "model": config["model"],
        "api_key": os.environ["DEEPSEEK_API_KEY"],
        "base_url": base_url
    }]
    
    # 返回autogen格式的配置
    return {
        "config_list": config_list, 
        "temperature": config["temperature"]
    } 