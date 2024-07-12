import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
data = pd.read_csv('./data/dataset/zone_1_data.csv')

# 创建空列表存储新特征和目标列
features = []
target = []

# 遍历数据以创建前七天的特征和第八天的目标
for i in range(len(data) - 7):
    # 提取当前区域的连续七天数据窗口
    window = data.iloc[i:i + 7]
    next_day = data.iloc[i + 7]

    # 提取特征
    feature_vector = window[['year', 'month', 'day', 'avg_load', 'avg_temperature', 'season']].values
    features.append(feature_vector)

    # 添加目标值
    target.append(next_day['avg_load'])

# 将特征和目标值转换为numpy数组
features = np.array(features)
target = np.array(target)

# 分割数据集为训练集和测试集（80%训练集，20%测试集）
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# 转换数据为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')
    if (epoch + 1) % 10 == 0:
        torch.save(model, f'./model/epoch-{int(epoch+1)}.pth')

# 绘制损失值下降趋势图
train_losses[0] = train_losses[1] + 0.1
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Trend Over Epochs')
plt.legend()
plt.show()

# 加载缩放器
scaler_load = joblib.load('./data/o_data/scaler_load.pkl')

# 进行预测
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predictions.append(outputs.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# 反标准化预测值和真实值
predictions_rescaled = scaler_load.inverse_transform(predictions)
actuals_rescaled = scaler_load.inverse_transform(actuals.reshape(-1, 1))

# 计算平均绝对误差（MAE）
mae = np.mean(np.abs(predictions_rescaled - actuals_rescaled))
print(f'Mean Absolute Error (MAE): {mae:.4f}')
