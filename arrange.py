import pandas as pd
import torch
import torch.nn as nn
import joblib

# 定义预测函数
def predict_next_day(model, input_data, scaler):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(input_tensor).item()
        # 只对 avg_load 进行逆标准化
        prediction = scaler.inverse_transform([[prediction]])[0, 0]
    return prediction

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


input_size = 6  # 每个时间步的特征数
hidden_size = 256
num_layers = 2
output_size = 1

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型
model_path = './model/epoch-200.pth'
model = torch.load(model_path)
model.eval()

# 加载缩放器
scaler = joblib.load('./data/o_data/scaler_load.pkl')

# 加载预测输入数据
input_data_path = './prediction_input.csv'
input_data = pd.read_csv(input_data_path)

# 提取特征
input_features = input_data[['year', 'month', 'day', 'avg_load', 'avg_temperature', 'season']].values

# 进行预测并逆标准化
predicted_load = predict_next_day(model, input_features, scaler)
print(f"Predicted load for the next day: {predicted_load:.4f}")
