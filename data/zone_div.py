import pandas as pd

# 加载预处理后的数据文件
data = pd.read_csv('./o_data/normalized_data.csv')

# 获取所有的地区ID
zone_ids = data['zone_id'].unique()

# 根据地区ID拆分数据并保存到不同的文件中
for zone_id in zone_ids:
    zone_data = data[data['zone_id'] == zone_id]
    output_file = f'./dataset/zone_{zone_id}_data.csv'
    zone_data.to_csv(output_file, index=False)
    print(f"Saved data for Zone ID {zone_id} to {output_file}")

print("Data has been successfully split by zone.")
