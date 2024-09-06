import csv
import os
from os.path import join

import numpy as np

# from PIL import Image
from osgeo import gdal


# 设标签宽W，长H
def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def save_confusion_matrix(matrix, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = join(output_path, "confusion_matrix.csv")
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Predicted/Actual"] + [f"Class {i}" for i in range(matrix.shape[0])])
        for i in range(matrix.shape[0]):
            writer.writerow([f"Class {i}"] + list(matrix[i]))
    # print(f"Confusion matrix saved to {file_path}")


def compute_metrics(gt_dir, pred_dir, tif_name_list, num_classes, output_path=None):
    hist = np.zeros((num_classes, num_classes))

    for tif_name in tif_name_list:
        pred_path = join(pred_dir, tif_name)
        gt_path = join(gt_dir, tif_name)

        # 使用gdal打开tif文件并读取数据
        pred_data = gdal.Open(pred_path)
        gt_data = gdal.Open(gt_path)

        # 将gdal数据集转换为numpy数组
        pred = pred_data.ReadAsArray()
        label = gt_data.ReadAsArray()

        if label.shape != pred.shape:
            # print("Skipping due to size mismatch:", pred_path, gt_path)
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    # 计算各项指标
    IoU = np.diag(hist) / np.maximum(hist.sum(1) + hist.sum(0) - np.diag(hist), 1)
    Recall = np.diag(hist) / np.maximum(hist.sum(1), 1)
    FAR = (hist.sum(0) - np.diag(hist)) / np.maximum(hist.sum(0), 1)

    # 如果outpath不为空
    # if output_path is not None:
    #     save_confusion_matrix(hist, output_path)

    return IoU, Recall, FAR
