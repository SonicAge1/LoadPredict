import pandas as pd

# 假设 load_history 已经被读取为一个 DataFrame
# 读取数据
load_history = pd.read_csv('./o_data/load_history.csv')

# 计算每行的平均值，并覆盖 h1 列
load_history['h1'] = load_history.loc[:, 'h1':'h24'].mean(axis=1)

# 删除 h2 到 h24 列
columns_to_drop = [f'h{i}' for i in range(2, 25)]
load_history = load_history.drop(columns=columns_to_drop)

# 保存修改后的数据到新文件
load_history.to_csv('./o_data/modified_load_history.csv', index=False)

# 显示修改后的数据
print("Modified load History Data:")
print(load_history.head())
