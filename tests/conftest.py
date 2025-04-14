"""
pytest配置文件
"""
import os
import sys
import pytest

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 全局fixture
@pytest.fixture(scope="session")
def api_key():
    """返回API密钥"""
    return os.environ.get("DEEPSEEK_API_KEY", "your_api_key_here")

@pytest.fixture(scope="session")
def test_data_dir():
    """返回测试数据目录"""
    return os.path.join(os.path.dirname(__file__), "test_data") 