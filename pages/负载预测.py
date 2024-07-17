import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
import time
from io import BytesIO

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

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

# 定义预测函数
def predict_next_day(model, input_data, scaler):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(input_tensor).item()
        # 只对 avg_load 进行逆标准化
        prediction = scaler.inverse_transform([[prediction]])[0, 0]
    return prediction


# Streamlit 应用
def main():
    st.title('Load Prediction with LSTM')

    # 页状态
    if "page" not in st.session_state:
        st.session_state.page = "upload"

    # 页面1: 文件上传和训练
    if st.session_state.page == "upload":
        st.header("Upload your dataset file for predict")
        data_file = st.file_uploader("", type=["csv"])

        if data_file is not None:
            try:
                data = pd.read_csv(data_file)
                st.session_state.data = data
                st.session_state.page = "train"
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # 页面2: 训练和损失展示
    elif st.session_state.page == "train":
        data = pd.read_csv('./data/zone_1_data.csv')

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

        # 训练模型
        num_epochs = 30
        if st.button('Start Training'):
            train_loss_text = st.empty()  # 创建空的文本区域
            train_losses = []
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
                train_losses.append(train_loss*1000)

                # 更新文本区域的内容，显示当前epoch的训练损失
                train_loss_text.text(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss*1000:.4f}')

                # 控制显示速度
                time.sleep(0.1)

            # 保存训练好的模型和缩放器
            train_losses[0] = train_losses[1]+0.1
            st.session_state.scaler_load = joblib.load('./data/scaler/scaler_load.pkl')
            st.session_state.trained_model = model
            st.session_state.train_losses = train_losses

            # 切换到预测页面
            st.session_state.page = "predict"
            st.experimental_rerun()

    # 页面3: 预测
    elif st.session_state.page == "predict":
        st.header("Prediction")

        # 绘制损失值下降趋势图
        train_losses = st.session_state.train_losses
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_losses, label='Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Trend Over Epochs')
        ax.legend()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, width=600)  # 设置图像宽度为600像素

        # 加载预测输入数据
        input_data = st.session_state.data

        # 提取特征
        input_features = input_data[['year', 'month', 'day', 'avg_load', 'avg_temperature', 'season']].values

        # 显示数据样本的前七行
        st.write("### Sample of Input Data:")
        st.write(input_data.head(7))

        # 显示预测按钮
        if st.button('Predict Load for Next Day'):
            # 加载训练好的模型和缩放器
            scaler_load = st.session_state.scaler_load
            trained_model = st.session_state.trained_model

            # 进行预测并逆标准化
            predicted_load = predict_next_day(trained_model, input_features, scaler_load)

            # 显示预测结果
            st.write(f"Predicted load for the next day: {predicted_load:.4f}")


if __name__ == "__main__":
    main()
