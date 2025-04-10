from setuptools import setup, find_packages

# 读取requirements.txt，忽略注释
try:
    with open("requirements.txt", "r") as f:
        REQUIRES = [line.split("#", 1)[0].strip() for line in f if line.strip()]
except:
    print("'requirements.txt' not found!")
    REQUIRES = list()

# 注意：akshare和aiohttp有依赖冲突，这里我们从依赖列表中移除它们
# 安装后需要手动安装akshare: pip install akshare
REQUIRES = [req for req in REQUIRES if not req.startswith(("akshare", "aiohttp"))]

# 不再修改OpenAI和pyautogen版本，直接使用requirements.txt中指定的版本

setup(
    name="deepseek_finrobot",
    version="0.1.0",
    include_package_data=True,
    author="DeepSeek FinRobot Team",
    author_email="your_email@example.com",
    url="https://github.com/yourusername/deepseek_finrobot",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=REQUIRES,
    description="DeepSeek FinRobot: 基于DeepSeek API的开源金融AI代理平台",
    long_description="""DeepSeek FinRobot是一个基于DeepSeek API的开源金融AI代理平台，用于金融分析和决策支持。
    
注意：安装后可能需要手动安装akshare: pip install akshare py-mini-racer
""",
    entry_points={
        'console_scripts': [
            'finrobot=deepseek_finrobot.cli:main',
        ],
    },
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Financial Large Language Models, AI Agents, DeepSeek",
    platforms=["any"],
    python_requires=">=3.8, <=3.13.2",
) 