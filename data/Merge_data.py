import pandas as pd

# 加载修改后的温度历史和负载历史文件
temperature_history = pd.read_csv('./o_data/modified_temperature_history.csv')
load_history = pd.read_csv('./o_data/Modified_Load_History_Data.csv')

# 重命名列以便清晰
temperature_history = temperature_history.rename(columns={'h1': 'avg_temperature'})
load_history = load_history.rename(columns={'h1': 'avg_load'})

# 根据 zone_id, year, month 和 day 合并两个 DataFrame
merged_data = pd.merge(load_history, temperature_history, on=['zone_id', 'year', 'month', 'day'], how='inner')

# 保存合并后的 DataFrame 到新文件
merged_data.to_csv('./o_data/merged_data.csv', index=False)

# 显示合并后的数据
print("Merged Data:")
print(merged_data.head())