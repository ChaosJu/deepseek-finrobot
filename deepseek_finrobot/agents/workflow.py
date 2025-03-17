"""
工作流模块 - 实现各种代理工作流
"""

import autogen
from typing import Dict, List, Optional, Union, Any
import os
import json
import datetime

class SingleAssistant:
    """
    单一助手代理
    """
    
    def __init__(self, name: str, llm_config: Dict[str, Any], 
                max_consecutive_auto_reply: Optional[int] = None,
                human_input_mode: str = "TERMINATE"):
        """
        初始化单一助手代理
        
        Args:
            name: 代理名称
            llm_config: LLM配置
            max_consecutive_auto_reply: 最大连续自动回复次数
            human_input_mode: 人类输入模式
        """
        self.name = name
        self.llm_config = llm_config
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.human_input_mode = human_input_mode
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
        )
        
        # 创建助手代理
        self.assistant = autogen.AssistantAgent(
            name=name,
            llm_config=llm_config,
        )
    
    def chat(self, message: str, use_cache: bool = True, max_turns: int = 10, 
            summary_method: str = "last_msg"):
        """
        开始聊天
        
        Args:
            message: 初始消息
            use_cache: 是否使用缓存
            max_turns: 最大对话轮次
            summary_method: 摘要方法
        """
        # 初始化聊天
        self.user_proxy.initiate_chat(
            self.assistant,
            message=message,
            clear_history=not use_cache,
            max_turns=max_turns,
            summary_method=summary_method,
        )
        
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        self.assistant.reset()

class SingleAssistantShadow:
    """
    单一助手影子代理 - 支持工具使用
    """
    
    def __init__(self, name: str, llm_config: Dict[str, Any], 
                max_consecutive_auto_reply: Optional[int] = None,
                human_input_mode: str = "TERMINATE"):
        """
        初始化单一助手影子代理
        
        Args:
            name: 代理名称
            llm_config: LLM配置
            max_consecutive_auto_reply: 最大连续自动回复次数
            human_input_mode: 人类输入模式
        """
        self.name = name
        self.llm_config = llm_config
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.human_input_mode = human_input_mode
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建助手代理
        self.assistant = autogen.AssistantAgent(
            name=name,
            llm_config=llm_config,
        )
    
    def chat(self, message: str, use_cache: bool = True, max_turns: int = 10, 
            summary_method: str = "last_msg"):
        """
        开始聊天
        
        Args:
            message: 初始消息
            use_cache: 是否使用缓存
            max_turns: 最大对话轮次
            summary_method: 摘要方法
        """
        # 初始化聊天
        self.user_proxy.initiate_chat(
            self.assistant,
            message=message,
            clear_history=not use_cache,
            max_turns=max_turns,
            summary_method=summary_method,
        )
        
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        self.assistant.reset()

class MultiAgentWorkflow:
    """
    多代理工作流
    """
    
    def __init__(self, agents_config: List[Dict[str, Any]], llm_config: Dict[str, Any]):
        """
        初始化多代理工作流
        
        Args:
            agents_config: 代理配置列表
            llm_config: LLM配置
        """
        self.llm_config = llm_config
        self.agents = {}
        
        # 创建用户代理
        self.user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            code_execution_config={"work_dir": ".", "use_docker": False},
        )
        
        # 创建代理
        for config in agents_config:
            name = config.get("name")
            system_message = config.get("system_message", "")
            
            agent = autogen.AssistantAgent(
                name=name,
                llm_config=llm_config,
                system_message=system_message,
            )
            
            self.agents[name] = agent
    
    def chat(self, message: str, agent_name: str = None, use_cache: bool = True):
        """
        开始聊天
        
        Args:
            message: 初始消息
            agent_name: 代理名称，如果为None则使用第一个代理
            use_cache: 是否使用缓存
        """
        if agent_name is None:
            agent_name = list(self.agents.keys())[0]
            
        if agent_name not in self.agents:
            raise ValueError(f"代理 {agent_name} 不存在")
            
        # 初始化聊天
        self.user_proxy.initiate_chat(
            self.agents[agent_name],
            message=message,
            clear_history=not use_cache,
        )
        
    def reset(self):
        """
        重置聊天历史
        """
        self.user_proxy.reset()
        for agent in self.agents.values():
            agent.reset()

    def group_chat(self, message: str, use_cache: bool = True):
        """
        开始群聊
        
        Args:
            message: 初始消息
            use_cache: 是否使用缓存
        """
        # 创建群聊管理器
        groupchat = autogen.GroupChat(
            agents=[self.user_proxy] + list(self.agents.values()),
            messages=[],
            max_round=50,
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config,
        )
        
        # 初始化聊天
        self.user_proxy.initiate_chat(
            manager,
            message=message,
            clear_history=not use_cache,
        ) 