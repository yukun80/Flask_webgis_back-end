import os
import shutil
import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.warp import reproject

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from numpy import hstack
from sklearn.ensemble import RandomForestClassifier
from osgeo import gdal
import joblib


def multi_predict(optical_path, insar_path, susceptible_path, dir_save_path, result_name):
    temp_dir = "./temp_resampled"  # 临时文件夹路径

    print("optical_path: ", optical_path)
    print("insar_path: ", insar_path)
    print("susceptible_path: ", susceptible_path)
    print("dir_save_path: ", dir_save_path)
    print("result_name: ", result_name)

    raster_paths = [optical_path, insar_path, susceptible_path]
    raster_thresholds = [1, 1, 8]
    # 读取栅格数据并存储相关信息
    datasets = [rasterio.open(path) for path in raster_paths]  # 栅格数据列表
    res = datasets[0].res
    # 创建重采样后的数据集列表
    resampled_datasets = []

    for idx, ds in enumerate(datasets):
        # 如果当前数据集的分辨率与目标分辨率不同，则进行重采样
        if ds.res != res:
            # 计算重采样后的尺寸
            scale_x = ds.res[0] / res[0]
            scale_y = ds.res[1] / res[1]
            new_width = int(ds.width * scale_x)
            new_height = int(ds.height * scale_y)

            # 创建一个新的数组来存放重采样后的数据
            resampled_data = np.empty(shape=(new_height, new_width), dtype=ds.dtypes[0])

            # 重采样
            rasterio.warp.reproject(
                source=rasterio.band(ds, 1),
                destination=resampled_data,
                src_transform=ds.transform,
                src_crs=ds.crs,
                dst_transform=rasterio.transform.from_origin(ds.bounds.left, ds.bounds.top, res[0], res[1]),
                dst_crs=ds.crs,
                dst_width=new_width,
                dst_height=new_height,
                resampling=Resampling.nearest,
            )
            # 判断是否存在temp_dir路径，如果不存在则创建
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            # 创建临时文件路径
            temp_path = os.path.join(temp_dir, f"resampled_{idx}.tif")

            # 使用重采样后的数据创建新的TIFF文件
            with rasterio.open(
                temp_path,
                "w",
                driver="GTiff",
                height=new_height,
                width=new_width,
                count=1,
                dtype=resampled_data.dtype,
                crs=ds.crs,
                transform=rasterio.transform.from_origin(ds.bounds.left, ds.bounds.top, res[0], res[1]),
            ) as temp_dst:
                temp_dst.write(resampled_data, 1)

            # 重新打开临时文件，添加到resampled_datasets列表
            resampled_datasets.append(rasterio.open(temp_path))
        else:
            # 如果分辨率已经匹配，直接添加到列表
            resampled_datasets.append(ds)

    # 从这一步开始，使用resampled_datasets进行后续的操作
    bounds = [ds.bounds for ds in resampled_datasets]  # 栅格边界列表
    # 计算合并后的栅格覆盖范围
    left = min(bounds, key=lambda b: b.left).left
    bottom = min(bounds, key=lambda b: b.bottom).bottom
    right = max(bounds, key=lambda b: b.right).right
    top = max(bounds, key=lambda b: b.top).top

    # 计算输出栅格的维度
    out_width = int((right - left) / res[0])
    out_height = int((top - bottom) / res[1])
    print("output length: ", out_width, out_height)

    # 创建一个与数据集数量相同的数组，用于存储在完整数据范围下每个数据集的结果
    and_results = [np.zeros((out_height, out_width), dtype=bool) for _ in resampled_datasets]

    # 栅格对齐和条件计算
    for ds, threshold, result in zip(resampled_datasets, raster_thresholds, and_results):
        # 计算每个栅格数据在结果栅格中的偏移
        row_off = max(int((top - ds.bounds.top) / res[1]), 0)
        col_off = max(int((ds.bounds.left - left) / res[0]), 0)

        # 确定在结果栅格中的实际更新区域大小，第一个是数据的高度和宽度，第二个是结果栅格减去偏移的行和列
        rows = min(ds.height, out_height - row_off)
        cols = min(ds.width, out_width - col_off)

        # 如果计算出的行或列数小于等于0，则跳过当前栅格数据
        if rows <= 0 or cols <= 0:
            continue

        # 读取当前栅格数据，并进行条件判断
        data = ds.read(1)
        condition = data >= threshold

        # 提取将要更新的结果数据切片
        result_slice = result[row_off : row_off + rows, col_off : col_off + cols]
        # 调整栅格数组的大小以匹配结果切片，获取当前数据的前rows和前cols行列
        condition_adjusted = condition[:rows, :cols]

        # 使用np.logical_and来更新结果数组
        result[row_off : row_off + rows, col_off : col_off + cols] = np.logical_or(result_slice, condition_adjusted)

    # 将and_results列表中的所有数据进行逻辑与计算，得到最终结果
    final_result = np.logical_and.reduce(and_results)

    # 关闭所有打开的数据集
    for ds in datasets:
        ds.close()
    # 处理完成后，记得关闭所有打开的数据集
    for ds in resampled_datasets:
        ds.close()
    # 删除临时文件
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # 如果有子目录，使用shutil.rmtree
    # 删除临时目录
    os.rmdir(temp_dir)

    # 保存结果栅格
    meta = datasets[0].meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "height": final_result.shape[0],
            "width": final_result.shape[1],
            "transform": rasterio.transform.from_bounds(left, bottom, right, top, final_result.shape[1], final_result.shape[0]),
        }
    )

    with rasterio.open(dir_save_path + "/" + result_name, "w", **meta) as dst:
        dst.write(final_result.astype(rasterio.uint8), 1)


# 下面是机器学习模型的代码


# ann模型的构建
class BasicModel(nn.Module):
    def __init__(self, input_dim=13):
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

    def fit(self, X_train, y_train, X_test, y_test, epochs=30, batch_size=16, lr=1e-5):
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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

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


# 创建模型实例
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
            yhat = model.predict_proba(X)[:, 1]  # 使用模型的 predict_proba 方法
        yhat = yhat.reshape(len(yhat), 1)
        # store predictions as input for blending
        meta_X.append(yhat)

    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)

    # 使用混合模型进行最终预测
    final_predictions = blender.predict_proba(meta_X)

    return final_predictions


def ensemble_predict(multidata_path, dir_save_path, result_name):
    # 开始前修改
    # 将multidata_path中的斜杠都替换为反斜杠
    multidata_path = multidata_path.replace("\\", "/")
    input_dim = 13
    # weights_folder = "./ensemble_ML/weight/"
    blender_model_path = "./ensemble_ML/weight/blender_model.pkl"  # 混合模型的路径
    ann_weights_path = "./ensemble_ML/weight/ANN_weights.pth"  # 深度学习模型的权重文件路径
    # 创建models
    models = get_models(input_dim)
    for name, model in models:
        print(name, model)

    # 模型预测
    # 读取测试数据
    print("+++++++++++++++++++++++++++++debug2", "\n")
    data = gdal.Open(multidata_path)
    print("+++++++++++++++++++++++++++++debug1 multidata_path", multidata_path, "\n")
    proj = data.GetProjection()
    geoTrans = data.GetGeoTransform()
    data = data.ReadAsArray()
    X = data.shape[1]
    Y = data.shape[2]
    data = data.reshape(data.shape[0], -1).T
    columns_to_delete = [0, 1, 2, 11, 14]
    data = np.delete(data, columns_to_delete, axis=1)
    print("+++++++++++++++++++++++++++++debug4", "\n")
    min_max_scaler = MinMaxScaler().fit(data)  # Fit On Training Data
    data_predict = min_max_scaler.transform(data)  # Transform On Training Data
    print("+++++++++++++++++++++++++++++debug5", "\n")
    slice_size = 1000000
    start_idx = 0
    predictions = []
    while start_idx < data_predict.shape[0]:
        end_idx = min(start_idx + slice_size, data_predict.shape[0])
        sliced_data = data_predict[start_idx:end_idx, :]
        sliced_predictions = predict_ensemble(models, blender_model_path, ann_weights_path, sliced_data)
        predictions.append(sliced_predictions)
        start_idx += slice_size

    # 将预测结果按原始顺序拼接
    predictions = np.concatenate(predictions, axis=0)
    predictions_1 = predictions[:, 1]
    predictions_data = np.reshape(predictions_1, (X, Y))
    predictions_data_normalized = predictions_data * 255
    predictions_data_normalized = predictions_data_normalized.astype(np.uint8)

    img_name = result_name
    predictPath = os.path.join(dir_save_path, img_name)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(predictPath, Y, X, 1, gdal.GDT_Byte)
    outdata.SetGeoTransform(geoTrans)
    outdata.SetProjection(proj)
    outdata.GetRasterBand(1).WriteArray(predictions_data_normalized)
