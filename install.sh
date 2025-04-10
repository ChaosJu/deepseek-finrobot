#!/bin/bash

# 打印彩色文本的函数
print_green() {
    echo -e "\033[0;32m$1\033[0m"
}

print_yellow() {
    echo -e "\033[0;33m$1\033[0m"
}

print_red() {
    echo -e "\033[0;31m$1\033[0m"
}

# 创建虚拟环境
if [ ! -d "finrobot_env" ]; then
    print_green "创建虚拟环境..."
    python -m venv finrobot_env
fi

# 激活虚拟环境
print_green "激活虚拟环境..."
source finrobot_env/bin/activate

# 安装基本工具
print_green "安装基本工具..."
python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --upgrade pip setuptools wheel

# 安装依赖
print_green "安装依赖..."
python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-build-isolation -r requirements.txt || {
    print_red "部分依赖安装失败，尝试跳过问题依赖继续安装..."
    # 如果安装失败，尝试安装核心依赖
    python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com requests pandas numpy typing_extensions>=4.9.0
}

# 安装项目
print_green "安装项目..."
python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --no-build-isolation -e .

print_green "安装完成！"
print_yellow "如需验证安装是否成功，请运行: python test_minimal.py"
print_yellow "如果您需要完整功能，可能需要手动安装其他依赖"