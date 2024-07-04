import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载数据
data = pd.read_csv('./o_data/cleaned_merged_data_with_season.csv')

# 提取需要归一化的列（负载和温度）
columns_to_normalize = [col for col in data.columns if 'avg_load' in col or 'avg_temperature' in col]

# 初始化归一化器
scaler = MinMaxScaler()

# 归一化负载和温度列
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# 保存归一化后的数据
data.to_csv('./o_data/normalized_data.csv', index=False)

print("Only load and temperature columns have been normalized and saved.")
