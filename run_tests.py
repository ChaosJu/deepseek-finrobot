#!/usr/bin/env python
"""
测试运行脚本
"""
import os
import sys
import subprocess

def main():
    """运行测试"""
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建测试命令
    cmd = [
        sys.executable,  # 当前Python解释器
        "-m", "pytest",
        "tests",  # 测试目录
        "-v",  # 详细输出
        "--cov=deepseek_finrobot",  # 覆盖率报告
        "--cov-report=term",  # 终端输出覆盖率
        "--cov-report=html:coverage_report",  # HTML覆盖率报告
    ]
    
    # 添加命令行参数
    cmd.extend(sys.argv[1:])
    
    # 运行测试
    result = subprocess.run(cmd, cwd=current_dir)
    
    # 返回测试结果
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 