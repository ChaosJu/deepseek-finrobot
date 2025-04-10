"""
DeepSeek FinRobot - 基于DeepSeek API的金融机器人
"""

__version__ = '0.1.0'

# 导入所有子模块
from . import agents
from . import data_source
from . import utils

__all__ = [
    'agents',
    'data_source',
    'utils',
    '__version__'
]
