from deepseek_finrobot.data_source.akshare_utils import get_stock_industry_list

# 获取行业列表
df = get_stock_industry_list()

# 打印行业列表
print("\n行业列表：\n")
print(df[['板块名称', '板块代码']].to_string(index=False)) 