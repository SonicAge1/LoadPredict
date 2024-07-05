import pandas as pd

# 加载清洗后的数据文件
cleaned_data = pd.read_csv('./o_data/cleaned_merged_data.csv')

# 定义一个函数来确定季节
def get_season(month, day):
    if (month == 3 and day >= 21) or (month in [4, 5]) or (month == 6 and day < 21):
        return 0  # 春季
    elif (month == 6 and day >= 21) or (month in [7, 8]) or (month == 9 and day < 23):
        return 1  # 夏季
    elif (month == 9 and day >= 23) or (month in [10, 11]) or (month == 12 and day < 21):
        return 2  # 秋季
    else:
        return 3  # 冬季

# 应用函数到每一行，创建新的 season 列
cleaned_data['season'] = cleaned_data.apply(lambda row: get_season(row['month'], row['day']), axis=1)

# 保存带有新的 season 列的数据到新文件
cleaned_data.to_csv('./o_data/cleaned_merged_data_with_season.csv', index=False)

# 显示添加 season 列后的数据
print("\nData with Season Column:")
print(cleaned_data.head())
