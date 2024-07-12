import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# 加载数据
data = pd.read_csv('./o_data/cleaned_merged_data_with_season.csv')

# 提取需要归一化的列（负载和温度）
columns_to_normalize = ['avg_load', 'avg_temperature']

# 初始化归一化器
scaler_load = MinMaxScaler()
scaler_temp = MinMaxScaler()

# 归一化负载列
data[['avg_load']] = scaler_load.fit_transform(data[['avg_load']])
# 归一化温度列
data[['avg_temperature']] = scaler_temp.fit_transform(data[['avg_temperature']])

# 保存归一化后的数据
data.to_csv('./o_data/normalized_data.csv', index=False)

# 保存缩放器以便之后使用
joblib.dump(scaler_load, './o_data/scaler_load.pkl')
joblib.dump(scaler_temp, './o_data/scaler_temp.pkl')
