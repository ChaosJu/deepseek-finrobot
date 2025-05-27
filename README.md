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
- **tests/** - 测试目录，包含所有测试用例和测试数据

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
python -m deepseek_finrobot.cli predict 000001 --days 7 --export --format markdown （已经测试）

# 构建投资组合
python -m deepseek_finrobot.cli portfolio 000001,600519,000858 --risk 中等 --horizon 长期 --amount 100000 --export （已经测试）

# 分析行业趋势
# 行业名称按照股票行业分类
python -m deepseek_finrobot.cli industry 银行 --days 30 --export（已经测试）

# 分析财经新闻
python -m deepseek_finrobot.cli news 人工智能 --days 3 --limit 10 （未测试）
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

## 运行测试

项目包含一套完整的测试套件，用于验证各个组件的功能。测试文件位于`tests/`目录中。

### 运行所有测试

使用提供的测试运行脚本：

```bash
python run_tests.py
```

这将运行所有测试并生成覆盖率报告。

### 运行特定测试

您也可以使用pytest直接运行特定测试：

```bash
# 运行特定测试文件
pytest tests/test_minimal.py

# 运行特定测试函数
pytest tests/test_minimal.py::test_basic_functionality

# 运行包含特定关键字的测试
pytest -k "basic"
```

### 测试覆盖率

测试运行后，覆盖率报告将生成在`coverage_report/`目录中。您可以在浏览器中打开`coverage_report/index.html`查看详细的覆盖率报告。

## 贡献指南

本代码和文档根据Apache-2.0许可证发布。它们不应被视为财务建议或实时交易的建议。在进行任何交易或投资行动之前，务必谨慎行事并咨询合格的金融专业人士。 


# 逆向思维交易机器人 (中文文档)

## 一、程序简介

本机器人实现了一个基于市场心理学的逆向交易策略，旨在“在恐惧时买入，在贪婪时卖出”。它采用 `pyautogen` 构建的多智能体系统，并利用 DeepSeek 大语言模型进行高级分析。其核心逻辑是反向利用市场情绪，例如，在负面新闻引发恐慌时寻找买入机会，在市场普遍乐观、新闻利好时警惕风险并考虑卖出。

## 二、核心设计与运行逻辑

这个程序的核心思想是利用市场参与者的“人性弱点”（如恐慌性抛售、贪婪性追高）来执行逆向操作，从而盈利。它认为，当市场上充斥着负面新闻导致散户恐慌抛售时，往往是主力资金吸筹的机会（买入时机）；反之，当市场一片叫好，散户疯狂追高时，则可能是主力资金派发筹码的机会（卖出时机）。

整个系统基于 `deepseek-finrobot` 框架，并融合了 `ai-hedge-fund` 项目中多智能体协作的概念，构建了一个类似对冲基金的决策体系：

1.  **入口点 (`cli.py` - 命令行界面):**
    *   用户通过命令行启动机器人，指定要分析的股票代码列表、策略配置文件路径以及日志文件路径。
    *   命令示例: `python -m deepseek_finrobot.cli contrarian_trader --assets <股票列表> --config-file <配置文件> --output-log <日志文件>`

2.  **智能体系统 (基于 `pyautogen`):**
    *   **总指挥智能体 (`ChiefExecutiveOfficerAgent` - CEO Agent):**
        *   这是系统的“大脑”，负责协调其他分析师智能体的工作。
        *   它接收来自用户的指令（要分析哪些股票）。
        *   它会指示“新闻分析智能体”和“市场数据智能体”收集所需信息。
    *   **新闻分析智能体 (`NewsSentimentAgent`):**
        *   负责从数据源（例如 `akshare`）获取指定股票的最新新闻。
        *   利用DeepSeek大语言模型分析这些新闻的情绪（例如，是利好、利空还是中性），并给出一个情绪评分。
        *   将分析结果（新闻摘要、情绪评分）汇报给CEO。
    *   **市场数据智能体 (`MarketDataAgent`):**
        *   负责从数据源（例如 `akshare`）获取指定股票的市场数据，包括当前价格、成交量等。
        *   计算关键的技术指标，如相对强弱指数 (RSI)、移动平均线等。
        *   将市场数据和技术指标汇报给CEO。

3.  **逆向信号生成模块 (`ContrarianSignalGenerator`):**
    *   CEO智能体将从新闻分析智能体和市场数据智能体收集到的信息（新闻情绪、市场价格、成交量、RSI等）输入到这个模块。
    *   该模块根据预设的“逆向逻辑”判断是否产生交易信号：
        *   **买入逻辑触发条件 (示例):** 市场新闻情绪极度负面 + 成交量显著放大 (可能是恐慌盘涌出) + RSI指标显示超卖。
        *   **卖出逻辑触发条件 (示例):** 市场新闻情绪极度乐观 + 股价近期已有大幅上涨 + RSI指标显示超买。
    *   模块会输出一个“逆向信号”，包括信号类型（如强烈买入、买入、持有、卖出、强烈卖出）、信号的置信度以及产生的理由。

4.  **CEO智能体的决策:**
    *   CEO智能体获取逆向信号后，会结合更多上下文信息（例如，它自身的LLM可能会被提示考虑整体市场趋势、是否有重大的基本面风险等），并优先根据逆向策略的核心思想做出最终的交易决策。
    *   例如，即使逆向信号是“买入”，但如果CEO通过其LLM分析认为该股票存在即将破产等基本面重大利空，它可能会否决这个买入信号。
    *   决策的目的是识别并利用“主力诱多/诱空，散户接盘/割肉”的局面。

5.  **记录与执行:**
    *   CEO做出的所有决策（买入、卖出、持有）、决策的理由、相关的新闻情绪、市场数据以及逆向模块给出的信号，都会被详细记录在用户指定的日志文件中。这个日志文件方便后续复盘和策略优化。
    *   在当前设计中，机器人主要进行分析和决策记录。真实的交易执行需要对接券商API，这部分通常作为后续扩展。

**总结来说，该机器人的运行流程是：**
用户启动 -> CEO调度分析师收集信息 -> 分析师获取新闻并分析情绪、获取市场数据并计算指标 -> CEO将信息汇总给逆向模块生成信号 -> CEO结合逆向信号和自身LLM的判断做出最终决策 -> 记录决策。

整个过程强调“反人性操作”，试图捕捉因市场情绪过度反应而产生的错误定价机会。

## 三、使用方法

### 1. 先决条件

*   **Python 环境：** 推荐 Python 3.9+。
*   **依赖项：** 请安装所有必要的依赖包。如果您已遵循主项目的安装步骤（通过 `bash install.sh` 或 `pip install -r requirements.txt`），大部分依赖应已包含。针对本智能体系统的关键依赖包括 `pyautogen`。
*   **API密钥：**
    *   **DeepSeek API密钥：** 您必须拥有一个有效的DeepSeek API密钥。请在项目根目录下的 `config_api_keys.json` 文件中进行配置，格式如下：
        ```json
        {
          "DEEPSEEK_API_KEY": "your-deepseek-api-key",
          "OPENAI_API_KEY": "placeholder" 
        }
        ```
    *   **(可选) 数据源API密钥：** 如果您扩展机器人以使用其他数据源（例如用Alpha Vantage获取国际市场数据），则同样需要根据其要求配置相应的API密钥。`akshare`（用于中国市场数据）通常获取公开数据不需要API密钥。

### 2. 配置

*   **API密钥：** 确保 `config_api_keys.json` 文件已按上文所述正确配置。
*   **策略配置文件：**
    *   逆向交易机器人使用YAML或JSON配置文件来定义其策略参数。请创建一个文件（例如 `contrarian_strategy_config.yaml`）。
    *   **示例 `contrarian_strategy_config.yaml`：**
        ```yaml
        # Parameters for the ContrarianSignalGenerator
        contrarian_logic_params:
          THRESHOLD_EXTREME_NEGATIVE: -0.7
          THRESHOLD_EXTREME_POSITIVE: 0.7
          VOLUME_SPIKE_FACTOR: 1.5 
          RSI_OVERSOLD_THRESHOLD: 30
          RSI_OVERBOUGHT_THRESHOLD: 70
          MIN_RUNUP_PERCENTAGE_FOR_SELL_SIGNAL: 10

        # LLM configurations for agents
        llm_configs:
          ceo_agent:
            model: "deepseek-chat" 
            temperature: 0.5
          news_sentiment_agent:
            model: "deepseek-chat"
            temperature: 0.7
        ```
    *   **策略配置文件 (`contrarian_logic_params`) 中的关键参数：**
        *   `THRESHOLD_EXTREME_NEGATIVE`/`POSITIVE`: 触发逆向逻辑的情绪得分阈值。
        *   `VOLUME_SPIKE_FACTOR`: 用于确认恐慌/狂热情绪的平均成交量倍数。
        *   `RSI_OVERSOLD`/`OVERBOUGHT_THRESHOLD`: 标准的RSI超卖/超买水平。
        *   `MIN_RUNUP_PERCENTAGE_FOR_SELL_SIGNAL`: 近期资产价格应上涨多少百分比，才使得正面新闻被视为强烈的卖出信号（用于捕捉“利好出尽”行情）。

### 3. 运行机器人

机器人通过主 `cli.py` 脚本启动：

```bash
python -m deepseek_finrobot.cli contrarian_trader --assets <ASSET_LIST> --config-file <PATH_TO_STRATEGY_CONFIG> --output-log <PATH_TO_LOG_FILE>
```

*   **`--assets <ASSET_LIST>`：** （必需）逗号分隔的资产ID（股票代码）列表。示例：`000001.SZ,600519.SH`
*   **`--config-file <PATH_TO_STRATEGY_CONFIG>`：** （必需）策略配置文件（YAML或JSON）的路径。示例：`configs/contrarian_strategy_config.yaml`
*   **`--output-log <PATH_TO_LOG_FILE>`：** （可选）交易决策和理由将被记录到的文件路径。默认为当前目录下的 `contrarian_trades.log`。示例：`logs/trades_zh.md`

**命令示例：**

```bash
python -m deepseek_finrobot.cli contrarian_trader --assets 000001.SZ,600519.SH --config-file configs/contrarian_strategy_config.yaml --output-log logs/trades_zh.md
```

## 四、如何解读日志文件

输出日志文件（例如 `contrarian_trades.log` 或 `logs/trades_zh.md`）将包含机器人做出的每一项决策的时间戳条目。每个条目通常包括：

*   **时间戳：** 决策做出的时间。
*   **资产ID：** 被分析的资产。
*   **新闻情绪摘要：** 关键情绪发现。
*   **市场数据摘要：** 相关的价格、成交量和指标数据。
*   **逆向信号：** 由 `ContrarianSignalGenerator` 生成的信号（例如，买入、卖出、强烈买入），包括其置信度得分和理由。
*   **CEO的最终决策：** `ChiefExecutiveOfficerAgent` 采取的最终行动（例如，“行动：买入，数量：100股（模拟），理由：强烈的逆向买入信号，因恐慌性抛售……”）。

此日志是机器人活动和推理的详细记录，对于复盘和策略优化至关重要。

---