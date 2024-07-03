import pandas as pd

# 假设 temperature_history 已经被读取为一个 DataFrame
# 读取数据
temperature_history = pd.read_csv('./o_data/temperature_history_file.csv')

# 计算每行的平均值，并覆盖 h1 列
temperature_history['h1'] = temperature_history.loc[:, 'h1':'h24'].mean(axis=1)

# 删除 h2 到 h24 列
columns_to_drop = [f'h{i}' for i in range(2, 25)]
temperature_history = temperature_history.drop(columns=columns_to_drop)

# 保存修改后的数据到新文件
temperature_history.to_csv('./o_data/modified_temperature_history.csv', index=False)

# 显示修改后的数据
print("Modified Temperature History Data:")
print(temperature_history.head())
