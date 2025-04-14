from deepseek_finrobot.data_source.akshare_utils import get_stock_industry_list, get_stock_industry_constituents

# 获取行业列表
industry_list = get_stock_industry_list()

# 查找贵金属行业的代码
precious_metals_code = None
for _, row in industry_list.iterrows():
    if row["板块名称"] == "贵金属":
        precious_metals_code = row["板块代码"]
        break

if precious_metals_code:
    print(f"找到贵金属行业代码: {precious_metals_code}")
    
    # 获取贵金属行业成分股
    stocks = get_stock_industry_constituents(precious_metals_code)
    
    # 打印列名
    print("\n成分股数据列名:")
    print(stocks.columns.tolist())
    
    # 打印前几行数据
    print("\n成分股数据示例:")
    print(stocks.head().to_string())
else:
    print("未找到贵金属行业") 