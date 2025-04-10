"""
工具函数模块 - 提供各种辅助函数
"""

import os
import json
import datetime
import pandas as pd
import autogen
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable

# 导入 DeepSeek API 工具
try:
    from .openai_adapter import get_llm_config_for_autogen
except ImportError:
    # 如果导入失败，创建一个默认的适配函数
    def get_llm_config_for_autogen(**kwargs):
        """当无法导入adapter模块时的替代函数"""
        config = kwargs.copy()
        # 确保有config_list字段
        if "config_list" in config:
            new_config_list = []
            for cfg in config["config_list"]:
                new_cfg = cfg.copy()
                # 将api_base转换为base_url (如果存在)
                if "api_base" in new_cfg:
                    new_cfg["base_url"] = new_cfg.pop("api_base")
                new_config_list.append(new_cfg)
            config["config_list"] = new_config_list
        return config

def get_current_date() -> str:
    """
    获取当前日期，格式为YYYY-MM-DD
    
    Returns:
        当前日期字符串
    """
    return datetime.datetime.now().strftime("%Y-%m-%d")

def register_keys_from_json(file_path: str) -> Dict[str, str]:
    """
    从JSON文件中读取API密钥并注册到环境变量
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        读取的API密钥字典
    """
    try:
        with open(file_path, 'r') as f:
            api_keys = json.load(f)
            
        # 将API密钥注册到环境变量
        for key, value in api_keys.items():
            os.environ[key] = value
            
        return api_keys
    except Exception as e:
        print(f"读取API密钥时出错: {e}")
        return {}

def format_financial_number(number: Union[float, int], precision: int = 2) -> str:
    """
    格式化金融数字，大数字使用万/亿为单位
    
    Args:
        number: 要格式化的数字
        precision: 小数点后位数
        
    Returns:
        格式化后的字符串
    """
    if number is None:
        return "N/A"
        
    try:
        number = float(number)
    except (ValueError, TypeError):
        return str(number)
        
    if abs(number) >= 1e8:  # 亿
        return f"{number/1e8:.{precision}f}亿"
    elif abs(number) >= 1e4:  # 万
        return f"{number/1e4:.{precision}f}万"
    else:
        return f"{number:.{precision}f}"

def load_api_keys(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    加载API密钥配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        API密钥字典
    """
    if config_path is None:
        # 默认配置文件路径
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_api_keys.json")
    
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 {config_path} 不存在")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"读取配置文件出错: {str(e)}")
        return {}

def get_deepseek_config_from_api_keys(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    从config_api_keys.json文件生成DeepSeek API配置列表
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        DeepSeek API配置列表
    """
    # 加载API密钥
    api_keys = load_api_keys(config_path)
    
    # 检查是否有DeepSeek API密钥
    if "DEEPSEEK_API_KEY" not in api_keys or not api_keys["DEEPSEEK_API_KEY"]:
        raise ValueError("未找到有效的DeepSeek API密钥，请检查config_api_keys.json文件")
    
    # 创建配置列表
    config_list = [{
        "model": "deepseek-chat",
        "api_key": api_keys["DEEPSEEK_API_KEY"],
        "base_url": "https://api.deepseek.com/v1"
    }]
    
    return config_list

def get_deepseek_config(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    获取DeepSeek API配置列表（从config_api_keys.json文件）
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        DeepSeek API配置列表
    """
    return get_deepseek_config_from_api_keys(config_path)

def save_to_csv(data: pd.DataFrame, filename: str, index: bool = True) -> str:
    """
    保存数据到CSV文件
    
    Args:
        data: 要保存的数据
        filename: 文件名
        index: 是否保存索引
        
    Returns:
        保存的文件路径
    """
    try:
        # 确保文件名有.csv后缀
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # 保存数据
        data.to_csv(filename, index=index, encoding='utf-8-sig')
        
        return os.path.abspath(filename)
    except Exception as e:
        print(f"保存CSV文件出错: {str(e)}")
        return ""

def read_from_csv(filename: str, **kwargs) -> pd.DataFrame:
    """
    从CSV文件读取数据
    
    Args:
        filename: 文件名
        **kwargs: 传递给pd.read_csv的参数
        
    Returns:
        读取的数据
    """
    try:
        return pd.read_csv(filename, **kwargs)
    except Exception as e:
        print(f"读取CSV文件出错: {str(e)}")
        return pd.DataFrame()

def get_date_range(days: int = 30) -> tuple:
    """
    获取日期范围
    
    Args:
        days: 天数
        
    Returns:
        (开始日期, 结束日期) 元组，格式为YYYYMMDD
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    return (
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d")
    )

# 缓存相关函数
def get_cache_dir() -> Path:
    """
    获取缓存目录
    
    Returns:
        缓存目录路径
    """
    # 在项目根目录下创建.cache目录
    cache_dir = Path(os.path.dirname(os.path.dirname(__file__))) / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    生成缓存键
    
    Args:
        prefix: 前缀
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        缓存键
    """
    # 将参数转换为字符串
    args_str = str(args) + str(sorted(kwargs.items()))
    # 计算MD5哈希
    hash_obj = hashlib.md5(args_str.encode())
    return f"{prefix}_{hash_obj.hexdigest()}"

def cache_data(data: Any, cache_key: str, expire_seconds: int = 3600) -> None:
    """
    缓存数据到本地
    
    Args:
        data: 要缓存的数据
        cache_key: 缓存键
        expire_seconds: 过期时间（秒）
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    # 保存数据和过期时间
    cache_data = {
        "data": data,
        "expire_time": time.time() + expire_seconds
    }
    
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"缓存数据时出错: {str(e)}")

def get_cached_data(cache_key: str) -> Optional[Any]:
    """
    获取缓存数据
    
    Args:
        cache_key: 缓存键
        
    Returns:
        缓存的数据，如果不存在或已过期则返回None
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        
        # 检查是否过期
        if time.time() > cache_data["expire_time"]:
            # 删除过期缓存
            os.remove(cache_file)
            return None
        
        return cache_data["data"]
    except Exception as e:
        print(f"读取缓存数据时出错: {str(e)}")
        return None

def cached(prefix: str, expire_seconds: int = 3600):
    """
    缓存装饰器
    
    Args:
        prefix: 缓存键前缀
        expire_seconds: 过期时间（秒）
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            # 尝试从缓存获取数据
            cached_result = get_cached_data(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 调用原函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache_data(result, cache_key, expire_seconds)
            
            return result
        return wrapper
    return decorator

def clear_cache(prefix: Optional[str] = None) -> int:
    """
    清除缓存
    
    Args:
        prefix: 缓存键前缀，如果为None则清除所有缓存
        
    Returns:
        清除的缓存文件数量
    """
    cache_dir = get_cache_dir()
    count = 0
    
    for cache_file in cache_dir.glob("*.pkl"):
        if prefix is None or cache_file.name.startswith(f"{prefix}_"):
            try:
                os.remove(cache_file)
                count += 1
            except Exception as e:
                print(f"删除缓存文件 {cache_file} 时出错: {str(e)}")
    
    return count 