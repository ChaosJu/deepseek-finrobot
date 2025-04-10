# DeepSeek FinRobot: 基于DeepSeek API的开源金融AI代理平台

**DeepSeek FinRobot** 是一个基于DeepSeek API的开源金融AI代理平台，专为金融应用设计。它集成了多种AI技术，不仅限于语言模型，还包括数据处理、可视化和分析工具，以满足金融行业的多样化需求。

**AI代理概念**：AI代理是一个智能实体，使用大型语言模型作为其"大脑"来感知环境、做出决策并执行操作。与传统人工智能不同，AI代理具有独立思考和利用工具逐步实现给定目标的能力。

## 项目结构

DeepSeek FinRobot 项目由以下主要组件构成：

- **agents/** - 多种金融AI代理的实现，包括市场预测、财务报告、新闻分析等专业代理
- **data_source/** - 数据源接口，用于获取金融市场数据、新闻和公司信息
- **utils.py** - 工具函数模块，提供各种辅助功能
- **cli.py** - 命令行接口，便于用户快速使用各种代理功能
- **openai_adapter.py** - DeepSeek API适配器，提供与DeepSeek大语言模型交互的标准接口

### openai_adapter.py

`openai_adapter.py` 模块是整个项目的核心组件之一，它提供了与DeepSeek API进行交互的标准接口。主要功能包括：

- **get_completion()** - 单轮问答接口，向DeepSeek模型发送单个提示并获取回复
- **get_chat_completion()** - 多轮对话接口，支持发送消息列表实现连续对话
- **get_llm_config_for_autogen()** - 生成适用于AutoGen框架的配置，是连接DeepSeek API和AutoGen框架的桥梁

这个适配器的设计类似于OpenAI API的接口格式，使得项目可以方便地使用DeepSeek的大语言模型服务，同时保持与其他LLM提供商的兼容性。

### utils.py中的关键函数

`utils.py` 文件包含多个重要的工具函数，其中包括：

- **get_deepseek_config()** - 获取DeepSeek API配置列表，用于设置与DeepSeek API的连接
- **get_deepseek_config_from_api_keys()** - 从config_api_keys.json文件生成DeepSeek API配置列表
- **cache_data()**, **get_cached_data()** - 提供数据缓存功能，提高性能
- **format_financial_number()** - 格式化金融数字，便于阅读和展示

## 快速开始

### 下载仓库

首先，下载项目代码：

```bash
git clone https://github.com/yourusername/deepseek_finrobot.git
cd deepseek-finrobot
```

### 配置API密钥

在安装依赖之前，需要配置DeepSeek API密钥。创建或编辑`config_api_keys.json`文件，添加您的API密钥：

```json
{
  "DEEPSEEK_API_KEY": "your-deepseek-api-key",
  "OPENAI_API_KEY": "placeholder"  // 可选，用于AutoGen框架兼容性
}
```

### 安装

完成API密钥配置后，您可以使用以下命令快速安装：

```bash
# 使用安装脚本（自动创建虚拟环境并安装所有依赖）
bash install.sh
```

您也可以手动安装：

```bash
# 创建虚拟环境（推荐）
python -m venv finrobot_env
source finrobot_env/bin/activate  # Linux/Mac
# 或 finrobot_env\Scripts\activate  # Windows

# 安装所有依赖
pip install -r requirements.txt  # 安装所有必要的依赖
pip install -e .                 # 以开发模式安装项目本身
```

#### 关键依赖

项目依赖以下主要包：

- **openai>=1.72.0** - OpenAI API客户端，用于与DeepSeek API交互
- **pyautogen>=0.8.5** - AutoGen框架，用于构建AI代理
- **pandas & numpy** - 数据处理核心组件
- **matplotlib>=3.10.0** - 用于数据可视化
- **akshare>=1.16.0** - 中国金融数据接口
- **requests>=2.31.0** - 处理HTTP请求

### 验证安装

安装完成后，您可以运行以下测试脚本来验证安装是否正确：

```bash
# 测试基本功能
python test_minimal.py

# 测试API密钥（需要先配置API密钥）
python test_api_key.py

# 测试市场预测代理（需要先配置API密钥）
python test_market_agent.py
```

### 使用示例

#### 市场预测代理

```python
from deepseek_finrobot.agents import MarketForecasterAgent

# 加载LLM配置
llm_config = {
    "config_list": [
        {
            "model": "deepseek-chat",
            "api_key": "your-deepseek-api-key",
            "base_url": "https://api.deepseek.com/v1",
        }
    ],
    "temperature": 0.7,
}

# 创建市场预测代理
forecaster = MarketForecasterAgent(llm_config)

# 预测股票走势
prediction = forecaster.predict("000001", days=7)
print(prediction)

# 批量预测多个股票
symbols = ["000001", "600519", "000858"]  # 平安银行、贵州茅台、五粮液
predictions = forecaster.batch_predict(symbols, days=7, max_workers=3)
for symbol, pred in predictions.items():
    print(f"股票 {symbol} 预测结果: {pred[:100]}...")

# 导出预测结果
forecaster.export_prediction(prediction, format="markdown", output_file="prediction.md")
```

#### 行业分析代理

```python
from deepseek_finrobot.agents import IndustryAnalysisAgent

# 加载LLM配置
llm_config = {
    "config_list": [
        {
            "model": "deepseek-chat",
            "api_key": "your-deepseek-api-key",
            "base_url": "https://api.deepseek.com/v1",
        }
    ],
    "temperature": 0.7,
}

# 创建行业分析代理
analyzer = IndustryAnalysisAgent(llm_config)

# 分析行业趋势
analysis = analyzer.analyze_industry("银行", days=30)
print(analysis)

# 批量分析多个行业
industries = ["银行", "医药", "计算机"]
analyses = analyzer.batch_analyze(industries, days=30, max_workers=3)
for industry, analysis in analyses.items():
    print(f"行业 {industry} 分析结果: {analysis[:100]}...")

# 导出分析结果
analyzer.export_analysis(analysis, format="html", output_file="industry_analysis.html")
```

#### 投资组合代理

```python
from deepseek_finrobot.agents import PortfolioManagerAgent

# 加载LLM配置
llm_config = {
    "config_list": [
        {
            "model": "deepseek-chat",
            "api_key": "your-deepseek-api-key",
            "base_url": "https://api.deepseek.com/v1",
        }
    ],
    "temperature": 0.7,
}

# 创建投资组合代理
portfolio_manager = PortfolioManagerAgent(llm_config)

# 构建投资组合
stocks = ["000001", "600519", "000858"]  # 平安银行、贵州茅台、五粮液
recommendation = portfolio_manager.construct_portfolio(
    stocks=stocks,
    risk_preference="中等",
    investment_horizon="长期",
    investment_amount=100000
)
print(recommendation)

# 优化现有投资组合
current_portfolio = {
    "000001": 0.3,  # 平安银行 30%
    "600519": 0.5,  # 贵州茅台 50%
    "000858": 0.2   # 五粮液 20%
}
optimization = portfolio_manager.optimize_portfolio(
    current_portfolio=current_portfolio,
    risk_preference="中等",
    investment_horizon="长期"
)
print(optimization)

# 导出建议
portfolio_manager.export_recommendation(recommendation, format="markdown", output_file="portfolio.md")
```

#### 其他代理

项目还支持以下代理：
- **技术分析代理** (`TechnicalAnalysisAgent`)
- **财务报告代理** (`FinancialReportAgent`)
- **新闻分析代理** (`NewsAnalysisAgent`)

### 使用命令行工具

```bash
# 预测股票走势
python -m deepseek_finrobot.cli predict 000001 --days 7 --export --format markdown

# 分析行业趋势
python -m deepseek_finrobot.cli industry 银行 --days 30 --export

# 分析财经新闻
python -m deepseek_finrobot.cli news 人工智能 --days 3 --limit 10

# 构建投资组合
python -m deepseek_finrobot.cli portfolio 000001,600519,000858 --risk 中等 --horizon 长期 --amount 100000 --export
```

## 直接使用DeepSeek API适配器

除了使用预构建的代理，您还可以直接使用`openai_adapter.py`模块中的函数来与DeepSeek API交互：

```python
import os
from deepseek_finrobot.openai_adapter import get_completion, get_chat_completion

# 设置API密钥环境变量
os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"

# 单轮问答示例
response = get_completion(
    prompt="分析贵州茅台(600519)的投资价值",
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1500
)
print(response)

# 多轮对话示例
messages = [
    {"role": "user", "content": "你是一个专业的金融分析师"},
    {"role": "assistant", "content": "您好，我是一名专业的金融分析师，擅长股票分析、行业研究和投资组合管理。请问有什么可以帮助您的？"},
    {"role": "user", "content": "分析一下最近A股市场的整体走势"}
]
response = get_chat_completion(
    messages=messages,
    model="deepseek-chat",
    temperature=0.5,
    max_tokens=2000
)
print(response)
```

### 与AutoGen框架集成

您也可以使用`get_llm_config_for_autogen`函数生成与AutoGen框架兼容的配置：

```python
import os
import autogen
from deepseek_finrobot.openai_adapter import get_llm_config_for_autogen

# 设置API密钥环境变量
os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"

# 获取AutoGen兼容的配置
config = get_llm_config_for_autogen(
    model="deepseek-chat",
    temperature=0.7,
    base_url="https://api.deepseek.com/v1"
)

# 创建AutoGen代理
assistant = autogen.AssistantAgent(
    name="金融助手",
    llm_config=config,
    system_message="您是一位专业的金融分析师，擅长分析市场趋势和投资机会。"
)

user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# 启动对话
user_proxy.initiate_chat(
    assistant,
    message="分析一下当前A股市场的投资机会，重点关注科技和消费板块。"
)
```

## 测试工具

项目提供了多个测试脚本，帮助验证安装和功能：

- **test_minimal.py** - 测试基本工具函数和环境设置
- **test_api_key.py** - 测试DeepSeek API密钥有效性
- **test_agent.py** - 测试市场预测代理的基本功能
- **test_market_agent.py** - 使用真实API密钥测试市场预测功能
- **test_all.py** - 全面测试项目的所有主要组件

使用这些测试脚本可以帮助您验证安装是否正确，以及API密钥是否有效。

## 免责声明

本代码和文档根据Apache-2.0许可证发布。它们不应被视为财务建议或实时交易的建议。在进行任何交易或投资行动之前，务必谨慎行事并咨询合格的金融专业人士。 