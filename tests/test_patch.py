#!/usr/bin/env python3
"""
Test script to verify that the Pydantic namespace warnings are fixed
"""

import logging
import sys

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def test_import_without_warnings():
    """测试导入deepseek_finrobot包时没有Pydantic警告"""
    logger.info("正在测试导入deepseek_finrobot包...")
    
    # 重定向标准错误以捕获警告
    import io
    original_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        # 导入包
        import deepseek_finrobot
        
        # 获取捕获的输出
        captured_stderr = sys.stderr.getvalue()
        
        # 判断是否有关于protected_namespaces的警告
        if "protected namespace \"model_\"" in captured_stderr:
            logger.error("🚫 仍然存在Pydantic命名空间警告:")
            print(captured_stderr)
            return False
        else:
            logger.info("✅ 导入成功，没有检测到Pydantic命名空间警告")
            return True
    except Exception as e:
        logger.error(f"❌ 导入时出错: {str(e)}")
        return False
    finally:
        # 恢复标准错误
        sys.stderr = original_stderr

if __name__ == "__main__":
    logger.info("===== 开始测试 =====")
    success = test_import_without_warnings()
    logger.info(f"===== 测试{'成功' if success else '失败'} =====")
    sys.exit(0 if success else 1) 