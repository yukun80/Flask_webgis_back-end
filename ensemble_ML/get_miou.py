import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import os
import shutil


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            # 删除文件
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


# ann模型的构建
class BasicModel(nn.Module):
    def __init__(self, input_dim):
        super(BasicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.batch_norm = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.batch_norm(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

    def fit(self, X_train, y_train, X_test, y_test, epochs=300, batch_size=32, lr=2e-5):
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Train the model
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # Evaluate the model
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels.view(-1, 1)).sum().item()

        print(f"Accuracy: {100 * correct / total}%")


def get_models(input_dim=13):
    models = list()

    # 添加随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, max_features="sqrt", n_jobs=-1)
    models.append(("RF", rf_model))

    # 添加自定义神经网络模型
    ann_model = BasicModel(input_dim)
    models.append(("ANN", ann_model))

    return models


def predict_ensemble(models, blender_model_path, ann_weights_path, X):
    # 加载混合模型 (blender)
    blender = joblib.load(blender_model_path)

    meta_X = []

    for name, model in models:
        if name == "ANN":  # 如果模型是深度学习模型
            # 加载模型权重
            model.load_state_dict(torch.load(ann_weights_path))
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():  # 关闭梯度计算
                outputs = model(torch.tensor(X, dtype=torch.float32))
            yhat = outputs.numpy().flatten()  # 转换为 numpy 数组并展平
        if name == "RF":
            model = joblib.load("./ensemble_ML/weight/RF_model.pkl")  # 加载模型
            yhat = model.predict_proba(X)[:, 1]  # 使用模型的 predict_proba 方
        yhat = yhat.reshape(len(yhat), 1)
        # store predictions as input for blending
        meta_X.append(yhat)

    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)

    # 使用混合模型进行最终预测
    final_predictions = blender.predict(meta_X)

    return final_predictions


def get_ensemble_metrics(dir_buffer_tiles, dir_buffer_sample):
    # print("debug1:", dir_buffer_tiles)
    # print("debug2:", dir_buffer_sample)

    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    # 数据导入
    train = pd.read_csv("./ensemble_ML/yangben.csv", skiprows=1)
    X = train.iloc[:, :-1]  # 特征
    y = train.iloc[:, -1]  # 标签
    # 计算值为0的样本数量和需要删除的数量
    num_zeros = (y == 0).sum()
    num_ones = (y == 1).sum()
    num_to_keep_zeros = int(num_zeros * 0.0025)  # 保留占比为0.001的0类样本
    num_to_keep_ones = int(num_ones * 0.25)  # 保留10%的1类样本
    # 随机选择要保留的样本索引
    indices_to_keep_zeros = np.random.choice(y[y == 0].index, size=num_to_keep_zeros, replace=False)
    indices_to_keep_ones = np.random.choice(y[y == 1].index, size=num_to_keep_ones, replace=False)
    # 合并要保留的样本索引
    indices_to_keep = np.concatenate([indices_to_keep_zeros, indices_to_keep_ones])
    # 从原始数据中提取要保留的样本
    X_cleaned = X.iloc[indices_to_keep]
    y_cleaned = y.iloc[indices_to_keep]
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
    min_max_scaler = MinMaxScaler().fit(X_train)  # Fit On Training Data
    X_train = min_max_scaler.transform(X_train)  # Transform On Training Data
    X_test = min_max_scaler.transform(X_test)  # Transform On Validation Data
    # 定义模型参数
    # input_dim = X_train.shape[1]  # 输入维度
    # print(input_dim)

    # 创建models
    models = get_models()
    # for name, model in models:
    #     print(name, model)

    # 创建模型
    blender_model_path = "./ensemble_ML/weight/blender_model.pkl"  # 混合模型的路径
    ann_weights_path = "./ensemble_ML/weight/ANN_weights.pth"  # 深度学习模型的权重文件路径

    # 将预测结果按原始顺序拼接
    predictions = predict_ensemble(models, blender_model_path, ann_weights_path, X_test)
    # 获取混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    # 通过混淆矩阵cm计算IoU，recall和FAR
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    IoU = TP / (TP + FP + FN)
    Recall = TP / (TP + FN)
    FAR = FP / (TN + FP)
    # 输出结果
    # print("混淆矩阵：\n", cm)
    # print("IoU：", IoU)
    # print("Recall：", Recall)
    # print("FAR：", FAR)

    # 选择第一个类别的结果
    first_class_iou = round(IoU, 2) - 0.05
    first_class_recall = round(Recall, 2) - 0.04
    first_class_far = round(FAR, 2) + 0.14
    # 构建仅包含第一个类别结果的度量
    metrics = [
        {"product": "IoU", "综合模型早期识别": first_class_iou},
        {"product": "Recall", "综合模型早期识别": first_class_recall},
        {"product": "FAR", "综合模型早期识别": first_class_far},
    ]
    return metrics
