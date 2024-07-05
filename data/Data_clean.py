import pandas as pd

# 加载合并的数据文件
merged_data = pd.read_csv('./o_data/Merged_Data .csv')

# 显示数据的基本信息
print("Initial Data Info:")
print(merged_data.info())
print("\nInitial Data Description:")
print(merged_data.describe())

# 检查缺失值
print("\nMissing Values:")
print(merged_data.isnull().sum())

# 填充或删除缺失值（这里我们选择删除包含缺失值的行）
cleaned_data = merged_data.dropna()

# 检查并处理重复值
print("\nDuplicate Values:")
print(cleaned_data.duplicated().sum())
cleaned_data = cleaned_data.drop_duplicates()

# 保存清洗后的数据到新文件
cleaned_data.to_csv('./o_data/cleaned_merged_data.csv', index=False)

# 显示清洗后的数据
print("\nCleaned Data Info:")
print(cleaned_data.info())
print("\nCleaned Data Description:")
print(cleaned_data.describe())
print("\nCleaned Data Head:")
print(cleaned_data.head())
