# DeepSeek FinRobot: 基于DeepSeek API的开源金融AI代理平台

**DeepSeek FinRobot** 是一个基于DeepSeek API的开源金融AI代理平台，专为金融应用设计。它集成了多种AI技术，不仅限于语言模型，还包括数据处理、可视化和分析工具，以满足金融行业的多样化需求。

**AI代理概念**：AI代理是一个智能实体，使用大型语言模型作为其"大脑"来感知环境、做出决策并执行操作。与传统人工智能不同，AI代理具有独立思考和利用工具逐步实现给定目标的能力。

## 安装

**1. (推荐) 创建新的虚拟环境**

```bash
conda create --name deepseek_finrobot python=3.10
conda activate deepseek_finrobot
```

**2. 下载DeepSeek FinRobot仓库**

```bash
git clone https://github.com/yourusername/deepseek_finrobot.git
cd deepseek_finrobot
```

**3. 安装deepseek_finrobot及其依赖**

```bash
pip install -e .
```

**4. 配置DEEPSEEK_CONFIG文件**

1) 重命名DEEPSEEK_CONFIG_sample为DEEPSEEK_CONFIG
2) 添加您的DeepSeek API密钥

**5. 配置config_api_keys文件**

1) 重命名config_api_keys_sample为config_api_keys
2) 添加您的各种API密钥（如有需要）

**6. 开始使用教程**

```
# 在tutorials目录中查找这些笔记本
1) agent_market_forecaster.ipynb
2) agent_annual_report.ipynb
3) agent_trade_strategist.ipynb
```

## DeepSeek FinRobot生态系统

### DeepSeek FinRobot的整体框架分为四个不同的层，每个层都旨在解决金融AI处理和应用的特定方面：

1. **金融AI代理层**：金融AI代理层包括金融思维链（CoT）提示，增强复杂分析和决策能力。市场预测代理、文档分析代理和交易策略代理利用CoT将金融挑战分解为逻辑步骤，将其先进算法和领域专业知识与金融市场的不断变化动态相结合，以获得精确、可操作的见解。
2. **金融LLM算法层**：金融LLM算法层配置并使用专门针对特定领域和全球市场分析调整的模型。
3. **LLMOps和DataOps层**：LLMOps层实施多源集成策略，为特定金融任务选择最合适的LLM，利用一系列最先进的模型。
4. **多源LLM基础模型层**：这一基础层支持各种通用和专业LLM的即插即用功能。

## 功能特性

- 中国金融数据源支持：
  - **AKShare**：提供丰富的中国市场数据，包括股票、债券、基金等
  - **TuShare**：提供中国金融市场数据，包括股票、指数、基金等
- 中国财经新闻数据源支持：
  - **东方财富**：获取财经新闻、股票公告等
  - **新浪财经**：获取股票新闻、行情等
  - **金十数据**：获取重大财经新闻
  - **央视财经**：获取央视财经新闻
  - **百度财经**：获取市场新闻
- 专业金融代理：
  - **市场预测代理**：分析A股数据和新闻，预测股票走势
  - **财务报告代理**：生成A股公司财务分析报告
  - **新闻分析代理**：分析中国财经新闻对市场和个股的影响
  - **行业分析代理**：分析行业趋势和投资机会
  - **投资组合代理**：构建和优化投资组合，提供资产配置建议
  - **技术分析代理**：使用技术指标分析股票走势，提供交易信号
- 高级功能：
  - **数据缓存**：自动缓存数据，提高性能
  - **批量处理**：支持批量分析多个股票或行业
  - **结果导出**：支持导出分析结果为Markdown、HTML或文本格式
  - **命令行工具**：提供简单易用的命令行接口

## 快速开始

### 安装

```bash
pip install -e .
```

### 配置API密钥

1. 复制配置文件模板：

```bash
cp config_api_keys_sample.json config_api_keys.json
```

2. 编辑`config_api_keys.json`，填入你的API密钥：

```json
{
  "DEEPSEEK_API_KEY": "your-deepseek-api-key",
  "AKSHARE_TOKEN": "your-akshare-token",
  "TUSHARE_TOKEN": "your-tushare-token"
}
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

#### 技术分析代理

```python
from deepseek_finrobot.agents import TechnicalAnalysisAgent

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

# 创建技术分析代理
analyst = TechnicalAnalysisAgent(llm_config)

# 进行技术分析
analysis = analyst.analyze("000001", period="daily", days=120)
print(analysis)

# 批量分析多个股票
symbols = ["000001", "600519", "000858"]
analyses = analyst.batch_analyze(symbols, period="daily", days=120, max_workers=3)
for symbol, result in analyses.items():
    print(f"股票 {symbol} 分析结果: {result[:100]}...")

# 导出分析结果
analyst.export_analysis(analysis, format="html", output_file="technical_analysis.html")
```

#### 财务报告代理

```python
from deepseek_finrobot.agents import FinancialReportAgent

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

# 创建财务报告代理
reporter = FinancialReportAgent(llm_config)

# 生成财务分析报告
report = reporter.generate_report("000001")
print(report)
```

#### 新闻分析代理

```python
from deepseek_finrobot.agents import NewsAnalysisAgent

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

# 创建新闻分析代理
analyzer = NewsAnalysisAgent(llm_config)

# 分析财经新闻
analysis = analyzer.analyze_news("人工智能", days=3, limit=5)
print(analysis)
```

### 命令行工具

安装后，可以使用`finrobot`命令行工具：

```bash
# 预测股票走势
finrobot predict 000001 --days 7 --export --format markdown

# 批量预测多个股票
finrobot predict 000001,600519,000858 --batch --workers 3 --export

# 分析行业趋势
finrobot industry 银行 --days 30 --export

# 批量分析多个行业
finrobot industry 银行,医药,计算机 --batch --workers 3

# 分析财经新闻
finrobot news 人工智能 --days 3 --limit 10

# 生成财务分析报告
finrobot report 000001

# 构建投资组合
finrobot portfolio 000001,600519,000858 --risk 中等 --horizon 长期 --amount 100000 --export

# 优化投资组合
finrobot optimize 000001:30,600519:50,000858:20 --risk 中等 --horizon 长期 --export

# 进行技术分析
finrobot technical 000001 --period daily --days 120 --export

# 批量进行技术分析
finrobot technical 000001,600519,000858 --batch --workers 3 --export
```

## 免责声明

本代码和文档根据Apache-2.0许可证发布。它们不应被视为财务建议或实时交易的建议。在进行任何交易或投资行动之前，务必谨慎行事并咨询合格的金融专业人士。 