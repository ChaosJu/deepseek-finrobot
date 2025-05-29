# System Architecture: Expert-Enhanced Contrarian Trading Bot

## 1. Overall System Philosophy

This system is designed as an advanced A-share short-term trading bot. Its core trading philosophy is **contrarian investing**, aiming to capitalize on market psychology by "buying fear and selling greed." This core is then significantly enhanced by a suite of **expert-inspired specialist analyst agents** providing diverse contextual insights (value, catalysts, market themes, macro-policy). Finally, all trading decisions are subject to **rigorous pre-trade risk management** and sophisticated **dynamic trade management** for exits.

The system operates as a multi-agent ensemble, orchestrated by a Chief Executive Officer (CEO) Agent, leveraging Large Language Models (LLMs) for complex analysis, decision-making, and interpretation tasks.
(系统理念：本系统定位为高级A股短期交易机器人，核心采用**逆向投资**策略，旨在利用市场心理“买入恐惧，卖出贪婪”。核心策略之上，辅以一套**专家启发的专业分析智能体**，提供价值、催化剂、市场主题、宏观政策等多维度背景信息。所有交易决策最终都需经过**严格的交易前风险管理**以及精细的**动态交易管理**以实现退出。系统以多智能体协作模式运行，由CEO主控智能体负责调度，并利用大语言模型LLM完成复杂分析、决策及解读任务。)

## 2. Main Components / Agent Roles (主要组件 / 智能体角色)

The system is composed of the following key agents and modules, primarily within the `deepseek_finrobot/agents/` and `contrarian_trader/src/` directories:
(系统由以下关键智能体和模块构成，主要位于 `deepseek_finrobot/agents/` 和 `contrarian_trader/src/` 目录下：)

*   **`ChiefExecutiveOfficerAgent (CEO)` (主控智能体) (`contrarian_trader/src/agents/ceo_agent.py`):**
    *   **Role (角色):** The central orchestrator and decision-making unit. (中央调度与决策单元。)
    *   **Responsibilities (职责):**
        *   Receives assets to analyze from the `cli.py` interface. (从 `cli.py` 命令行接口接收待分析资产列表。)
        *   Manages the overall workflow by querying various analyst agents. (通过查询各类分析智能体来管理整体工作流程。)
        *   Synthesizes all gathered information (primary contrarian signal + specialist analyses) using its LLM to formulate a trading proposal, including dynamic position sizing based on conviction. (利用其LLM综合所有收集到的信息（包括主要逆向信号和专业分析师的分析结果），形成包含基于确信度的动态仓位大小的交易提案。)
        *   Calculates initial stop-loss (ATR-based, dynamically adjusted) and profit targets for new trades. (为新交易计算初始止损（基于ATR动态调整）和止盈目标。)
        *   Submits the proposal to the `RiskManagementAgent`. (将交易提案提交给 `RiskManagementAgent` 进行风险评估。)
        *   **Manages active trades**: (管理活跃交易：)
            *   Checks for stop-loss hits (initial or trailing). (检查止损（初始或追踪）。)
            *   Implements **Fear-Based Short Covering**: Covers profitable short positions if extreme fear sentiment is detected for the asset. (实施基于恐惧的空头回补：若检测到资产出现极端恐惧情绪且空头头寸盈利，则回补。)
            *   Implements **Crisis Exit for Longs**: Exits or significantly tightens stops on long positions if critical negative news (e.g., "FundamentalScandalOrCrisis") is detected for the asset. (实施基于危机的多头退出：若检测到资产出现严重负面新闻（如“基本面丑闻或危机”），则立即退出或大幅收紧止损。)
            *   Implements **Counter-Signal Exit**: Exits a position if a fresh, strong, opposing primary signal is generated for the asset. (实施基于反向信号的退出：若系统为当前持仓资产生成了明确且高置信度的反向交易信号，则考虑退出。)
            *   Implements **Greed-Based Profit-Taking for BUYs**: Takes full/partial profits or tightens trailing stop for profitable long positions if extreme greed/euphoria is detected in news or market themes. (实施针对买入交易的基于贪婪的止盈：若在新闻或市场主题中检测到极度贪婪/狂热情绪且多头头寸盈利，则全部/部分止盈或收紧追踪止损。)
            *   Manages standard R/R profit targets (partial and full) and ATR-based trailing stops. (管理标准的风险回报比止盈目标（部分和全部）以及基于ATR的追踪止损。)
        *   Makes the final trading decision based on risk assessment and logs it. (基于风险评估结果做出最终交易决策，并记录日志。)

*   **`ContrarianSignalGenerator` (逆向信号生成模块 - Implemented within `contrarian_trader/src/core_logic/contrarian_analyzer.py`):**
    *   **Role (角色):** Generates the primary short-term trading signal. (生成主要的短期交易信号。)
    *   **Logic (逻辑):** Analyzes news sentiment (from `NewsSentimentAgent`) and market data/technicals (from `MarketDataAgent`), potentially adjusted by market state (volatility, trend), to identify contrarian opportunities. (分析新闻情绪（来自 `NewsSentimentAgent`）和市场数据/技术指标（来自 `MarketDataAgent`），并可能根据市场状态（波动性、趋势）进行调整，以识别逆向交易机会。)

*   **`NewsSentimentAgent` (新闻情绪分析智能体 - `contrarian_trader/src/agents/news_agent.py`):**
    *   **Role (角色):** Provides news articles and their detailed sentiment scores (including categories like "IrrationalPanic", "OverhypedRally", "FundamentalScandalOrCrisis", etc.) for specific assets using LLMs. (为特定资产提供新闻文章及其详细的情绪评分（包括“非理性恐慌”、“过度炒作上涨”、“基本面丑闻或危机”等类别），使用LLM进行分析。)
    *   **Output (输出):** Structured news data with detailed sentiment analysis (overall score, dominant detailed category, rationale, key phrases). (结构化的新闻数据及详细情绪分析结果（总体评分、主导详细类别、理由、关键短语）。)

*   **`MarketDataAgent` (市场数据智能体 - `contrarian_trader/src/agents/market_data_agent.py`):**
    *   **Role (角色):** Provides current/historical market data, technical indicators (RSI, MAs), ATR for individual assets, and overall market state (volatility, trend) based on a market index. (提供当前/历史市场数据、技术指标（RSI、移动平均线）、单个资产的ATR值，以及基于市场指数的整体市场状态（波动性、趋势）。)
    *   **Output (输出):** Structured market data, technical indicators, asset ATR, and market state. (结构化的市场数据、技术指标、资产ATR和市场状态。)

*   **`ShortTermValueFilterAgent` (短期价值筛选智能体 - `deepseek_finrobot/agents/short_term_value_filter_agent.py`):**
    *   **(职责和逻辑描述保持不变)**
*   **`CatalystScoutAgent` (催化剂侦察智能体 - `deepseek_finrobot/agents/catalyst_scout_agent.py`):**
    *   **(职责和逻辑描述保持不变)**
*   **`MarketThemeSentimentAgent` (市场主题情绪智能体 - `deepseek_finrobot/agents/market_theme_sentiment_agent.py`):**
    *   **(职责和逻辑描述保持不变)**
*   **`MacroPolicyImpactAgent` (宏观政策影响智能体 - `deepseek_finrobot/agents/macro_policy_impact_agent.py`):**
    *   **(职责和逻辑描述保持不变)**
*   **`RiskManagementAgent` (风险管理智能体 - `deepseek_finrobot/agents/risk_management_agent.py`):**
    *   **(职责和逻辑描述保持不变)**

## 3. Workflow Description (工作流程描述)

1.  **Initiation (初始化):** User launches via `cli.py`. (用户通过 `cli.py` 启动。)
2.  **CEO Orchestration (CEO调度):** For each target asset (or for managing active trades): (针对每个目标资产（或管理活跃交易时）：)
    a.  **Active Trade Management (活跃交易管理 - 优先执行):**
        i.  CEO fetches current market data (price, high, low, ATR) for all active positions via `MarketDataAgent`. (CEO通过 `MarketDataAgent` 获取所有活跃头寸的当前市场数据（价格、高/低价、ATR）。)
        ii. Optionally, for assets also targeted for new signals, fresh sentiment/primary signals might be available from `all_gathered_data_for_current_run`. (可选地，如果活跃头寸的资产也是新信号的目标，则 `all_gathered_data_for_current_run` 中可能包含其最新的情绪/主要信号数据。)
        iii. The `_check_and_manage_active_trades` method is called. This method applies exit logic in a specific order:
            1.  Standard Stop-Loss (initial or trailing). (标准止损（初始或追踪）。)
            2.  Fear-Based Short Covering / Crisis Exit for Longs (based on new sentiment, if available). (基于恐惧的空头回补 / 基于危机的多头退出（基于新的情绪数据，如果可用且适用）。)
            3.  Counter-Signal Exit (if a strong opposing signal is freshly generated for the asset). (反向信号退出（如果资产刚生成了强烈的反向主要信号）。)
            4.  Greed-Based Profit-Taking for BUYs (based on news/theme sentiment, if available and profitable). (针对买入交易的基于贪婪的止盈（基于新闻/主题情绪，如果可用且盈利）。)
            5.  Standard R/R Profit Targets (partial and full). (标准的风险回报比止盈目标（部分和全部）。)
            6.  ATR Trailing Stop Update (if active and not exited). (ATR追踪止损更新（如果激活且未退出）。)
        iv. Any generated closing decisions are added to the list for the current run. (产生的任何平仓决策会被加入当前运行的决策列表。)
    b.  **New Signal Generation (新信号生成 - 如果指定了目标资产且未平仓):** If `stock_symbol_for_new_trade` is provided and no existing active trade or recent closing action for it: (如果提供了 `stock_symbol_for_new_trade` 且该资产没有活跃仓位或近期未平仓：)
        i.  **Data Gathering (数据收集):** CEO directs `MarketDataAgent` to fetch market state, price/ATR for the target asset, and `NewsSentimentAgent` for detailed news sentiment. (CEO指示 `MarketDataAgent` 获取市场状态、目标资产价格/ATR，并指示 `NewsSentimentAgent` 获取详细新闻情绪。)
        ii. **Primary Signal (主要信号):** This data is fed to `ContrarianStrategyAgent` (which uses `ContrarianAnalyzer`) to generate a core signal, now potentially using dynamic thresholds adjusted by market state. (此数据被送入 `ContrarianStrategyAgent`（其内部使用 `ContrarianAnalyzer`）以产生核心信号，该过程现在可能使用由市场状态动态调整的阈值。)
        iii. **Contextual Analysis (辅助分析):** Queries specialist analyst agents (`ShortTermValueFilterAgent`, `CatalystScoutAgent`, `MarketThemeSentimentAgent`, `MacroPolicyImpactAgent`). (向专业分析智能体请求辅助分析。)
        iv. **Trade Proposal Formulation (交易提案形成):** CEO synthesizes all information, calculates conviction score, determines dynamic position size, calculates initial stop-loss and profit targets. (CEO综合所有信息，计算确信度分数，确定动态仓位大小，并计算初始止损和止盈目标。)
        v.  **Risk Assessment (风险评估):** Proposal sent to `RiskManagementAgent`. (提案送至 `RiskManagementAgent`。)
        vi. **Final Decision & Logging (最终决策与记录):** CEO makes final decision. If opening a trade, it's added to `active_trades`. All decisions logged. (CEO做出最终决策。如果开仓，则加入 `active_trades`。所有决策均被记录。)
3.  **Iteration (迭代):** Process repeats. (流程重复。)

## 4. Data Flow (Conceptual) (数据流概念)
*   **External Data (Akshare, News APIs) -> Data Providing Agents (外部数据 -> 数据提供智能体):** `MarketDataAgent` fetches price/volume, ATR, and index data. `NewsSentimentAgent` fetches news. Other specialist agents (Catalyst, Theme, Macro) also pull from relevant sources (e.g., Akshare). (数据提供智能体从外部源拉取数据。)
    *   `MarketDataAgent` 新增获取单个资产的ATR值和整体市场状态（波动性、趋势）的能力。
*   **Data Providing Agents -> CEO (数据提供智能体 -> CEO):**
    *   `MarketDataAgent` provides: latest price, historical data, asset-specific ATR, and market state (volatility, trend). (市场数据智能体提供：最新价格、历史数据、特定资产ATR、市场状态。)
    *   `NewsSentimentAgent` provides: detailed news sentiment analysis (overall score, dominant detailed category, rationale). (新闻情绪智能体提供：详细新闻情绪分析。)
    *   Other specialist agents provide their respective reports.
*   **CEO -> ContrarianStrategyAgent (CEO -> 逆向策略智能体):** Stock symbol, market data, news sentiment data, market state. (股票代码、市场数据、新闻情绪数据、市场状态。)
*   **ContrarianStrategyAgent -> CEO (逆向策略智能体 -> CEO):** Primary signal (BUY/SELL/HOLD, confidence, rationale, detailed_sentiment_category from news). (主要信号，包含详细情绪分类。)
*   **CEO (for active trades) -> (itself for management logic):** Current price, ATR, high/low for active assets; entry price, stop-loss, quantity, initial ATR, profit targets from `active_trades` state. (对于活跃交易，CEO使用当前价格、ATR、高/低价；从 `active_trades` 状态中获取入场价、止损价、数量、初始ATR、止盈目标。)
    *   **新增**: For fear/crisis/counter-signal exits, the CEO's trade management logic also considers fresh sentiment data and primary signals for the specific active asset if it's also the target of the current analysis run. (对于基于恐惧/危机/反向信号的退出，CEO的交易管理逻辑还会考虑当前活跃资产的最新情绪数据和主要信号（如果该资产也是本轮分析的目标）。)
*   **CEO (for new trades) -> RiskManagementAgent (CEO -> 风险管理智能体):** `proposed_trade` (now includes dynamic quantity, initial stop-loss, profit targets) + `current_portfolio_snapshot`.
*   **RiskManagementAgent -> CEO (风险管理智能体 -> CEO):** `risk_assessment`.
*   **CEO -> Logger (CEO -> 日志模块):** `final_decision` (can be new trade, closing trade, or HOLD), including all supporting data.

## 5. (Textual) Diagram Description (文本化架构图描述)
(No major changes to the overall agent connections, but the data exchanged is richer.)
Imagine a central `ChiefExecutiveOfficerAgent`.
*   `MarketDataAgent` now supplies not just price/technicals but also **asset-specific ATR** and **overall market state** (volatility/trend) to the CEO.
*   `NewsSentimentAgent` supplies **detailed sentiment categories** to the CEO.
*   The CEO uses market state to adjust thresholds for its internal `ContrarianSignalGenerator` (via `ContrarianStrategyAgent`).
*   When formulating a new trade, the CEO calculates **dynamic position size**, **ATR-based stop-loss** (itself dynamically adjusted by market state/sentiment/conviction), and **profit targets**.
*   For **active trade management**, the CEO continuously checks market data against stored trade states (entry price, current stop-loss, profit targets, ATR at entry) to trigger exits based on the refined logic (standard SL/PT, trailing stops, greed, fear, crisis, counter-signals).
*   Other flows remain similar: specialist agents provide context, Risk Management agent vets proposals.

## 6. Configuration Notes (配置说明)
*   The `contrarian_strategy_config.yaml` file is central.
*   **New/Updated Sections in Config:**
    *   `position_sizing_config`: For dynamic position sizing (base percentage, conviction tiers, multipliers, analyst score contributions). (用于动态仓位调整的配置，包括基础百分比、确信度等级、乘数、分析师评分贡献等。)
    *   `stop_loss_config`: Includes `base_n_atr_for_stop`, `n_atr_adjustments` (for market state, sentiment, conviction), `min_n_atr`, `max_n_atr`. Also includes parameters for **crisis exits** (e.g., `use_crisis_exit_for_long`, `crisis_exit_strategy_for_long`, `news_sentiment_crisis_categories_for_long_exit`, `crisis_tightened_n_atr_stop`) and **counter-signal exits** (e.g., `use_counter_signal_exit`, `threshold_exit_on_counter_signal_confidence`, relevant sentiment categories for counter-signals). (包含基础ATR止损、N倍ATR动态调整因子，以及新增的危机退出和反向信号退出参数。)
    *   `profit_taking_config`: Includes R/R targets, partial profit-taking, trailing stop parameters. Also includes parameters for **greed-based exits** (e.g., `use_greed_based_profit_taking`, `news_sentiment_greed_categories_for_buy_exit`, `min_profit_factor_for_greed_exit_buy`, `greedy_exit_strategy_buy`) and **fear-based short covering** (e.g., `use_fear_based_short_covering`, `news_sentiment_fear_categories_for_short_cover`, `min_profit_factor_for_fear_cover_short`). (包含风险回报比止盈、部分止盈、追踪止损，以及新增的基于贪婪信号和恐惧信号的退出参数。)
    *   `market_data_agent_config`: Includes `market_state_config` (for overall market state calculation) and potentially default ATR period for assets if not specified by CEO. (包含市场状态计算配置和资产ATR周期。)
    *   `contrarian_analyzer_config`: Its thresholds (e.g., `base_threshold_sentiment_contrarian_buy`) are now considered *base* thresholds before dynamic adjustments by the CEO using market state. (其阈值现在作为CEO进行动态调整前的基础阈值。)
*   Other configurations for news sentiment, specialist agents, and risk management remain.

## 7. 项目结构 (Project Structure)
(No changes to the project structure itself from the previous version of this document, but the responsibilities of `ceo_agent.py`, `market_data_agent.py`, and `contrarian_strategy_config.yaml` are expanded.)

以下是本项目的主要文件和目录结构图示，旨在提供一个清晰的概览。部分目录（如 `configs/`, `logs/`, `tests/` 内的详细结构）为推荐或基于通用实践的推断。

```
.
├── .gitignore                    # Git忽略配置文件
├── ARCHITECTURE.md               # 系统架构文档 (本文档)
├── README.md                     # 项目主说明文档 (中/英文)
├── CONTRARIAN_BOT_DEVELOPMENT_LOG.md # (新增) 逆向交易机器人开发日志
├── config_api_keys.json          # (或 config_api_keys_sample) API密钥配置文件
├── requirements.txt              # Python项目依赖列表
├── install.sh                    # (可能存在) 安装脚本
├── setup.py                      # (可能存在) Python包安装配置文件
│
├── configs/                      # (建议) 存放所有策略和应用配置
│   └── contrarian_strategy_config.yaml # (示例) 核心策略配置文件
│
├── logs/                         # (建议) 存放程序运行日志和交易记录
│   └── contrarian_trades.log     # (示例) 交易日志文件
│
├── deepseek_finrobot/            # 项目核心Python包 (包含新增的专家分析智能体和风险管理智能体)
│   ├── __init__.py               # 包初始化文件
│   │
│   ├── agents/                   # 智能体模块
│   │   ├── __init__.py           # 包初始化文件
│   │   ├── chief_executive_officer_agent.py  # (位于 contrarian_trader/src/agents/) CEO主控智能体 
│   │   ├── short_term_value_filter_agent.py  # 短期价值筛选智能体
│   │   ├── catalyst_scout_agent.py           # 催化剂侦察智能体
│   │   ├── market_theme_sentiment_agent.py   # 市场主题情绪智能体
│   │   ├── macro_policy_impact_agent.py    # 宏观政策影响智能体
│   │   ├── risk_management_agent.py        # 风险管理智能体
│   │   ├── news_sentiment_agent.py         # (位于 contrarian_trader/src/agents/) 新闻情绪分析智能体
│   │   ├── market_data_agent.py            # (位于 contrarian_trader/src/agents/) 市场数据智能体
│   │   └── ...
│   │
│   ├── core_logic/               # (位于 contrarian_trader/src/core_logic/) 核心策略逻辑模块
│   │   ├── __init__.py
│   │   └── contrarian_signal_generator.py # (即 contrarian_analyzer.py) 逆向信号生成器
│   │
│   ├── data_source/              # 数据源接口模块 (部分可能在 contrarian_trader/src/data_sources/)
│   │   ├── __init__.py
│   │   ├── akshare_utils.py      
│   │   └── ...
│   │
│   ├── cli.py                    # 命令行接口主程序
│   ├── openai_adapter.py         # 与LLM API交互的适配器
│   └── utils.py                  # 通用工具函数
│
├── contrarian_trader/            # "Contrarian Bot" 模块的特定代码
│   └── src/
│       ├── agents/               # CEO, News, Market Data agents reside here
│       │   └── ceo_agent.py
│       │   └── news_agent.py
│       │   └── market_data_agent.py
│       │   └── ... 
│       └── core_logic/           # Contrarian Analyzer resides here
│           └── contrarian_analyzer.py
│       └── ...
│
└── tests/                        # 测试代码目录
    ├── __init__.py
    └── ...
```

This architecture is designed to be modular and extensible, allowing for refinement of individual agents or addition of new analytical capabilities in the future, always centered around the core contrarian philosophy. (此架构设计旨在实现模块化和可扩展性，允许未来对单个智能体进行改进或添加新的分析能力，同时始终围绕核心的逆向投资理念。)

[end of ARCHITECTURE.md]
