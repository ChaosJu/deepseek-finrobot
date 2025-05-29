# 潜在的导入
import autogen # type: ignore
from typing import List, Dict, Any, Tuple

class RiskManagementAgent:
    """
    RiskManagementAgent (风险管理代理) 为 CEO 代理提出的交易提供风险监督。
    它根据预定义的风险参数检查建议的交易。
    此代理主要基于规则，例如资产黑名单、最大分配比例、现金余额要求等。

    RiskManagementAgent provides risk oversight for trades proposed by the CEO Agent.
    It checks proposed trades against pre-defined risk parameters.
    This agent is primarily rule-based, e.g. asset blacklist, max allocation, cash balance requirements, etc.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 RiskManagementAgent。
        加载风险管理相关的配置参数，例如单一资产最大分配比例、单一行业最大分配比例、
        最低现金余额百分比、每日最大回撤百分比、资产黑名单、单一资产最大持股数以及是否允许卖空。

        Initializes the RiskManagementAgent.
        Loads risk management related configuration parameters, such as max allocation per asset,
        max allocation per sector, min cash balance percentage, max daily drawdown percentage,
        asset blacklist, max shares per asset, and whether short selling is allowed.

        Args:
            config (dict, optional): 代理逻辑的配置参数。默认为 None。
                                     Configuration parameters for the agent's logic. Defaults to None.
        """
        self.config = config if config else {}
        # Default risk parameters if not provided in request_data during assessment
        self.default_risk_params = {
            "MAX_ALLOCATION_PER_ASSET": self.config.get("MAX_ALLOCATION_PER_ASSET", 0.20), # Max 20% of portfolio in one asset
            "MAX_ALLOCATION_PER_SECTOR": self.config.get("MAX_ALLOCATION_PER_SECTOR", 0.50), # Max 50% of portfolio in one sector
            "MIN_CASH_BALANCE_PERCENTAGE": self.config.get("MIN_CASH_BALANCE_PERCENTAGE", 0.02), # Min 2% cash
            "MAX_DAILY_DRAWDOWN_PERCENTAGE": self.config.get("MAX_DAILY_DRAWDOWN_PERCENTAGE", 0.05), # Max 5% daily loss
            "ASSET_BLACKLIST": self.config.get("ASSET_BLACKLIST", []),
            "ASSET_MAX_SHARES": self.config.get("ASSET_MAX_SHARES", {}), # e.g. {"AAPL": 1000}
            "ALLOW_SHORT_SELLING": self.config.get("ALLOW_SHORT_SELLING", False)
        }

    def _get_asset_details_from_portfolio(self, asset_id: str, holdings: List[Dict[str, Any]]) -> Tuple[float, float, str | None]:
        """
        (私有方法) 从投资组合中获取特定资产的当前市值、数量和所属行业。

        (Private method) Helper to get current market value, quantity, and sector of an asset in the portfolio.

        Args:
            asset_id (str): 要查询的资产ID。Asset ID to query.
            holdings (List[Dict[str, Any]]): 当前投资组合的持仓列表。List of current portfolio holdings.

        Returns:
            Tuple[float, float, str | None]: 包含资产当前市值、数量和行业名称的元组。如果资产未找到，则返回 (0.0, 0.0, None)。
                                             A tuple containing the asset's current market value, quantity, and sector name. 
                                             Returns (0.0, 0.0, None) if the asset is not found.
        """
        for holding in holdings:
            if holding.get('asset_id') == asset_id:
                return holding.get('current_market_value', 0.0), holding.get('quantity', 0.0), holding.get('sector')
        return 0.0, 0.0, None

    def _get_sector_value_from_portfolio(self, sector: str, holdings: List[Dict[str, Any]]) -> float:
        """
        (私有方法) 计算投资组合中特定行业的总市值。

        (Private method) Helper to get total market value of a specific sector in the portfolio.

        Args:
            sector (str): 要计算总市值的行业名称。Name of the sector to calculate total value for.
            holdings (List[Dict[str, Any]]): 当前投资组合的持仓列表。List of current portfolio holdings.

        Returns:
            float: 指定行业的总市值。
                   Total market value of the specified sector.
        """
        if not sector:
            return 0.0
        return sum(h.get('current_market_value', 0.0) for h in holdings if h.get('sector') == sector)

    def assess_trade_risk(self, request_data: dict) -> dict:
        """
        根据配置的风险参数评估建议交易的风险。
        此方法检查多项规则，包括资产黑名单、现金充足性、最大分配限制（按资产和行业）、
        每日最大回撤、以及卖空限制。

        Assesses the risk of a proposed trade against configured risk parameters.
        This method checks multiple rules, including asset blacklist, cash adequacy, 
        maximum allocation limits (per asset and per sector), daily maximum drawdown, 
        and short selling restrictions.

        Args:
            request_data (dict): 包含风险评估所需数据的字典：
                                 A dictionary containing data for risk assessment:
                - 'proposed_trade' (dict): 建议的交易信息
                                           {
                                               'asset_id': str,  # 资产ID
                                               'direction': str ("BUY" or "SELL"), # 交易方向 ("买入" 或 "卖出")
                                               'quantity': float, # 数量
                                               'estimated_price': float, # 估计价格
                                               'estimated_trade_value': float, # 估计交易价值
                                               'sector': str (optional) # 资产所属行业 (对于新买入资产尤其重要)
                                           }
                - 'current_portfolio_snapshot' (dict): 当前投资组合快照
                                                      {
                                                          'total_portfolio_value': float, # 投资组合总价值
                                                          'cash_balance': float, # 现金余额
                                                          'holdings': List[dict] (each holding with 'asset_id', 'quantity', 'sector', 'current_market_value'), # 持仓列表
                                                          'daily_pnl': float (optional), # 当日盈亏 (可选)
                                                          'total_portfolio_value_at_start_of_day': float (optional) # 当日初始总投资组合价值 (可选, 用于回撤计算)
                                                      }
                - 'risk_parameters' (dict, optional): 用于此检查的特定风险参数。如果未提供，则使用代理初始化时加载的默认参数。
                                                      Specific risk parameters for this check. If not provided, uses default parameters loaded during agent initialization.

        Returns:
            dict: 包含风险评估结果的字典，例如：
                  A dictionary containing the risk assessment, for example:
                  {
                      'asset_id': str, # 资产ID
                      'overall_assessment': str ("APPROVED", "WARNING", "REJECTED"), # 总体评估 ("批准", "警告", "拒绝")
                      'warnings_or_rejections': List[dict] (empty if "APPROVED"), # 警告或拒绝列表 (如果 "APPROVED" 则为空)，每个条目包含：
                                                                                # (empty if "APPROVED"), each with:
                          - 'rule_violated': str (e.g., "MAX_ALLOCATION_PER_ASSET"), # 违反的规则 (例如："单一资产最大分配比例")
                          - 'message': str (Detailed explanation), # 详细说明
                          - 'suggested_modification': str (optional, e.g., "Reduce quantity to X") # 建议修改 (可选，例如："将数量减少到 X")
                  }
        """
        proposed_trade = request_data.get('proposed_trade', {})
        portfolio_snapshot = request_data.get('current_portfolio_snapshot', {})
        
        # Use risk_parameters from request if provided, else use agent's default config
        risk_params = request_data.get('risk_parameters', self.default_risk_params)

        asset_id = proposed_trade.get('asset_id')
        direction = proposed_trade.get('direction')
        quantity = proposed_trade.get('quantity', 0.0)
        estimated_price = proposed_trade.get('estimated_price', 0.0)
        estimated_trade_value = proposed_trade.get('estimated_trade_value', quantity * estimated_price)

        total_portfolio_value = portfolio_snapshot.get('total_portfolio_value', 0.0)
        # Ensure total_portfolio_value is not zero to prevent division by zero errors later
        if total_portfolio_value == 0 and direction == "BUY": # Critical for BUY as many checks depend on it
             return {
                'asset_id': asset_id,
                'overall_assessment': "REJECTED",
                'warnings_or_rejections': [{
                    'rule_violated': "PORTFOLIO_DATA_MISSING",
                    'message': "Total portfolio value is zero or missing, cannot assess buy trade risk.",
                }]
            }
        
        cash_balance = portfolio_snapshot.get('cash_balance', 0.0)
        holdings = portfolio_snapshot.get('holdings', [])
        daily_pnl = portfolio_snapshot.get('daily_pnl') # Can be None
        # For daily drawdown, we need portfolio value at start of day. If not provided, this check might be skipped or adapted.
        # Assuming it's passed if the rule is to be applied strictly.
        portfolio_value_sod = portfolio_snapshot.get('total_portfolio_value_at_start_of_day', total_portfolio_value)


        warnings_rejections: List[Dict[str, str]] = []
        is_critical_rejection = False # Flag to mark if a hard rejection occurred

        # 1. Blacklist Check
        asset_blacklist = risk_params.get("ASSET_BLACKLIST", [])
        if asset_id in asset_blacklist:
            warnings_rejections.append({
                'rule_violated': "ASSET_BLACKLIST",
                'message': f"Asset {asset_id} is on the blacklist."
            })
            is_critical_rejection = True

        # --- BUY Trade Specific Checks ---
        if direction == "BUY" and not is_critical_rejection:
            # 2a. Cash Adequacy for Trade
            if cash_balance < estimated_trade_value:
                warnings_rejections.append({
                    'rule_violated': "INSUFFICIENT_CASH_FOR_TRADE",
                    'message': f"Insufficient cash for trade. Need {estimated_trade_value:.2f}, have {cash_balance:.2f}.",
                })
                is_critical_rejection = True # Cannot execute if no cash
            
            # 2b. Min Cash Balance Post-Trade
            min_cash_pct = risk_params.get("MIN_CASH_BALANCE_PERCENTAGE", 0.0)
            if total_portfolio_value > 0 and (cash_balance - estimated_trade_value) / total_portfolio_value < min_cash_pct:
                warnings_rejections.append({
                    'rule_violated': "MIN_CASH_BALANCE_POST_TRADE",
                    'message': f"Trade would reduce cash balance below minimum threshold of {min_cash_pct*100:.2f}%. Projected cash: {(cash_balance - estimated_trade_value)/total_portfolio_value*100:.2f}%."
                })
                # This might be a warning or rejection based on policy. For now, a warning.

            # 3. Max Shares per Asset
            asset_max_shares_config = risk_params.get("ASSET_MAX_SHARES", {})
            if asset_id in asset_max_shares_config:
                _, current_quantity, _ = self._get_asset_details_from_portfolio(asset_id, holdings)
                max_shares_for_asset = asset_max_shares_config[asset_id]
                if (current_quantity + quantity) > max_shares_for_asset:
                    suggested_qty = max(0, max_shares_for_asset - current_quantity)
                    warnings_rejections.append({
                        'rule_violated': "ASSET_MAX_SHARES",
                        'message': f"Proposed quantity for {asset_id} ({current_quantity + quantity}) exceeds max shares limit ({max_shares_for_asset}).",
                        'suggested_modification': f"Reduce quantity by at least {quantity - suggested_qty:.0f} to {suggested_qty:.0f} shares."
                    })
                    # This could be a hard rejection or a modifiable warning.
            
            # 4. Max Allocation per Asset
            max_alloc_asset_pct = risk_params.get("MAX_ALLOCATION_PER_ASSET", 1.0) # Default to 100% if not set (no limit)
            current_asset_value, _, asset_sector = self._get_asset_details_from_portfolio(asset_id, holdings)
            projected_asset_value = current_asset_value + estimated_trade_value
            if total_portfolio_value > 0 and (projected_asset_value / total_portfolio_value) > max_alloc_asset_pct:
                # Calculate max allowable additional value: (max_alloc_asset_pct * total_portfolio_value) - current_asset_value
                max_additional_value = (max_alloc_asset_pct * total_portfolio_value) - current_asset_value
                suggested_qty = int(max_additional_value / estimated_price) if estimated_price > 0 else 0
                warnings_rejections.append({
                    'rule_violated': "MAX_ALLOCATION_PER_ASSET",
                    'message': f"Proposed allocation for {asset_id} ({projected_asset_value/total_portfolio_value*100:.2f}%) exceeds limit ({max_alloc_asset_pct*100:.2f}%).",
                    'suggested_modification': f"Reduce quantity to approx. {max(0, suggested_qty):.0f} shares (max additional value: {max_additional_value:.2f})."
                })

            # 5. Max Allocation per Sector
            # Asset's sector needs to be known. Assume it's passed in proposed_trade or available in holdings for existing assets.
            # For a new asset, its sector must be provided in proposed_trade.
            proposed_asset_sector = proposed_trade.get('sector', asset_sector) # Get sector from proposal, fallback to existing if any
            if proposed_asset_sector:
                max_alloc_sector_pct = risk_params.get("MAX_ALLOCATION_PER_SECTOR", 1.0)
                current_sector_value = self._get_sector_value_from_portfolio(proposed_asset_sector, holdings)
                
                # If the asset being bought is already in the sector, its current value is part of current_sector_value.
                # If it's a new asset in this sector, or more of an existing one, add the trade value.
                projected_sector_value = current_sector_value
                if asset_sector != proposed_asset_sector: # If asset is changing sector (unlikely) or is new to this sector calculation
                    projected_sector_value += estimated_trade_value
                elif asset_sector == proposed_asset_sector: # Adding more to an existing asset in the sector
                     projected_sector_value = current_sector_value - current_asset_value + projected_asset_value # More accurate: total sector value if this specific asset's value changes

                if total_portfolio_value > 0 and (projected_sector_value / total_portfolio_value) > max_alloc_sector_pct:
                    warnings_rejections.append({
                        'rule_violated': "MAX_ALLOCATION_PER_SECTOR",
                        'message': f"Proposed allocation for sector {proposed_asset_sector} ({projected_sector_value/total_portfolio_value*100:.2f}%) exceeds limit ({max_alloc_sector_pct*100:.2f}%)."
                    })
            # else:
                # warnings_rejections.append({'rule_violated': "MISSING_SECTOR_INFO", 'message': f"Sector for asset {asset_id} not provided for sector allocation check."})


        # --- SELL Trade Specific Checks ---
        if direction == "SELL":
            current_asset_value, current_quantity, _ = self._get_asset_details_from_portfolio(asset_id, holdings)
            if quantity > current_quantity: # This is a short sell attempt
                allow_short_selling = risk_params.get("ALLOW_SHORT_SELLING", False)
                if not allow_short_selling:
                    warnings_rejections.append({
                        'rule_violated': "SHORT_SELLING_NOT_ALLOWED",
                        'message': f"Attempt to sell {quantity} shares of {asset_id}, but only {current_quantity} held and short selling is disallowed.",
                    })
                    is_critical_rejection = True
                # else: Add rules for short selling if allowed (e.g. margin, specific stocks only)

        # --- General Checks (apply to all trades if not already hard rejected) ---
        if not is_critical_rejection:
            # 6. Daily Drawdown Limit
            max_drawdown_pct = risk_params.get("MAX_DAILY_DRAWDOWN_PERCENTAGE")
            if daily_pnl is not None and max_drawdown_pct is not None and portfolio_value_sod > 0:
                current_drawdown = daily_pnl / portfolio_value_sod # pnl is usually negative for drawdown
                if current_drawdown < -max_drawdown_pct: # if pnl is -6% and limit is 5% (-0.06 < -0.05)
                    # For BUY trades, this is a strong warning. For SELL, it might be acceptable or even encouraged.
                    # Decision to make it WARNING for BUY, neutral for SELL for now.
                    if direction == "BUY":
                         warnings_rejections.append({
                            'rule_violated': "DAILY_DRAWDOWN_LIMIT_REACHED",
                            'message': f"Daily drawdown ({current_drawdown*100:.2f}%) has exceeded limit (-{max_drawdown_pct*100:.2f}%). Increasing risk exposure is not advised."
                        })
        
        # Determine overall assessment
        if any(r['rule_violated'] in ["ASSET_BLACKLIST", "INSUFFICIENT_CASH_FOR_TRADE", "SHORT_SELLING_NOT_ALLOWED"] for r in warnings_rejections):
            overall_assessment = "REJECTED"
        elif warnings_rejections:
            overall_assessment = "WARNING"
        else:
            overall_assessment = "APPROVED"
            
        return {
            'asset_id': asset_id,
            'overall_assessment': overall_assessment,
            'warnings_or_rejections': warnings_rejections
        }

# 示例用法 (用于测试)
# Example usage (for testing)
if __name__ == '__main__':
    # 此部分用于直接测试代理的方法，
    # This section would be for direct testing of the agent's methods,
    # 而非其在 autogen 框架内的典型操作。
    # not for its typical operation within the autogen framework.
    agent_config_main = {
        "MAX_ALLOCATION_PER_ASSET": 0.15,
        "MAX_ALLOCATION_PER_SECTOR": 0.40,
        "MIN_CASH_BALANCE_PERCENTAGE": 0.05,
        "MAX_DAILY_DRAWDOWN_PERCENTAGE": 0.02,
        "ASSET_BLACKLIST": ["STK_BAD", "SEC_FORBIDDEN"],
        "ASSET_MAX_SHARES": {"STK_GOOD": 1000, "000001.SZ": 500},
        "ALLOW_SHORT_SELLING": False
    }
    risk_agent_main = RiskManagementAgent(config=agent_config_main)

    print("--- Test Case 1: Simple Approved Buy ---")
    test_portfolio1 = {
        'total_portfolio_value': 100000.0, 'cash_balance': 50000.0,
        'holdings': [{'asset_id': 'STK_OTHER', 'quantity': 100, 'sector': 'Tech', 'current_market_value': 10000.0}],
        'total_portfolio_value_at_start_of_day': 100000.0, 'daily_pnl': -100.0 
    }
    test_trade1 = {'asset_id': 'STK_NEW', 'direction': "BUY", 'quantity': 100, 'estimated_price': 10.0, 'sector':'Finance'}
    assessment1 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade1, 'current_portfolio_snapshot': test_portfolio1})
    print(json.dumps(assessment1, indent=2, ensure_ascii=False))

    print("\n--- Test Case 2: Blacklisted Asset ---")
    test_trade2 = {'asset_id': 'STK_BAD', 'direction': "BUY", 'quantity': 100, 'estimated_price': 10.0}
    assessment2 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade2, 'current_portfolio_snapshot': test_portfolio1})
    print(json.dumps(assessment2, indent=2, ensure_ascii=False))

    print("\n--- Test Case 3: Insufficient Cash for Trade ---")
    test_trade3 = {'asset_id': 'STK_NEW', 'direction': "BUY", 'quantity': 100, 'estimated_price': 600.0} # Value 60000
    assessment3 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade3, 'current_portfolio_snapshot': test_portfolio1})
    print(json.dumps(assessment3, indent=2, ensure_ascii=False))
    
    print("\n--- Test Case 4: Exceeds Max Shares per Asset ---")
    test_portfolio4 = {
        'total_portfolio_value': 100000.0, 'cash_balance': 50000.0,
        'holdings': [{'asset_id': '000001.SZ', 'quantity': 450, 'sector': 'Finance', 'current_market_value': 4500.0}],
    }
    test_trade4 = {'asset_id': '000001.SZ', 'direction': "BUY", 'quantity': 100, 'estimated_price': 10.0, 'sector':'Finance'} # Total 550, limit 500
    assessment4 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade4, 'current_portfolio_snapshot': test_portfolio4})
    print(json.dumps(assessment4, indent=2, ensure_ascii=False))

    print("\n--- Test Case 5: Exceeds Max Allocation per Asset ---")
    test_trade5 = {'asset_id': 'STK_BIG', 'direction': "BUY", 'quantity': 200, 'estimated_price': 100.0, 'sector':'Energy'} # Value 20000 (20% of 100k)
    assessment5 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade5, 'current_portfolio_snapshot': test_portfolio1})
    print(json.dumps(assessment5, indent=2, ensure_ascii=False)) # Limit is 15%

    print("\n--- Test Case 6: Exceeds Max Allocation per Sector ---")
    test_portfolio6 = {
        'total_portfolio_value': 100000.0, 'cash_balance': 50000.0,
        'holdings': [
            {'asset_id': 'FIN_A', 'quantity': 100, 'sector': 'Finance', 'current_market_value': 15000.0},
            {'asset_id': 'FIN_B', 'quantity': 100, 'sector': 'Finance', 'current_market_value': 15000.0} # Total Finance = 30k (30%)
        ],
    } # Limit for sector is 40%
    test_trade6 = {'asset_id': 'FIN_C', 'direction': "BUY", 'quantity': 150, 'estimated_price': 100.0, 'sector':'Finance'} # Adds 15k, total 45k (45%)
    assessment6 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade6, 'current_portfolio_snapshot': test_portfolio6})
    print(json.dumps(assessment6, indent=2, ensure_ascii=False))

    print("\n--- Test Case 7: Daily Drawdown Limit Reached ---")
    test_portfolio7 = {
        'total_portfolio_value': 90000.0, 'cash_balance': 50000.0, 'holdings': [],
        'total_portfolio_value_at_start_of_day': 100000.0, 'daily_pnl': -6000.0 # -6% drawdown, limit 2%
    }
    test_trade7 = {'asset_id': 'STK_NEW', 'direction': "BUY", 'quantity': 10, 'estimated_price': 100.0, 'sector':'Tech'}
    assessment7 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade7, 'current_portfolio_snapshot': test_portfolio7})
    print(json.dumps(assessment7, indent=2, ensure_ascii=False))

    print("\n--- Test Case 8: Short Selling Not Allowed ---")
    test_trade8 = {'asset_id': 'STK_SHORT', 'direction': "SELL", 'quantity': 100, 'estimated_price': 10.0} # No STK_SHORT in portfolio
    assessment8 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade8, 'current_portfolio_snapshot': test_portfolio1})
    print(json.dumps(assessment8, indent=2, ensure_ascii=False))
    
    print("\n--- Test Case 9: Min Cash Balance Post-Trade Warning ---")
    test_portfolio9 = {
        'total_portfolio_value': 100000.0, 'cash_balance': 10000.0, # 10% cash
        'holdings': [{'asset_id': 'STK_OTHER', 'quantity': 100, 'sector': 'Tech', 'current_market_value': 90000.0}],
    } # Min cash 5%
    test_trade9 = {'asset_id': 'STK_BUY', 'direction': "BUY", 'quantity': 600, 'estimated_price': 10.0, 'sector':'Energy'} # Cost 6000, cash becomes 4000 (4%)
    assessment9 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade9, 'current_portfolio_snapshot': test_portfolio9})
    print(json.dumps(assessment9, indent=2, ensure_ascii=False))
    
    print("\n--- Test Case 10: Approved Sell (Has Holdings) ---")
    test_portfolio10 = {
        'total_portfolio_value': 100000.0, 'cash_balance': 10000.0,
        'holdings': [{'asset_id': 'STK_SELL', 'quantity': 200, 'sector': 'Retail', 'current_market_value': 20000.0}],
    }
    test_trade10 = {'asset_id': 'STK_SELL', 'direction': "SELL", 'quantity': 50, 'estimated_price': 100.0}
    assessment10 = risk_agent_main.assess_trade_risk({'proposed_trade': test_trade10, 'current_portfolio_snapshot': test_portfolio10})
    print(json.dumps(assessment10, indent=2, ensure_ascii=False))
