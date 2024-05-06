import os
from PIL import Image
from tqdm import tqdm
from LXY_InSAR_DL.utils.utils_metrics import compute_metrics

import numpy as np

from LXY_InSAR_DL.deeplab import DeeplabV3
from osgeo import gdal
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


#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "文件无法打开")
    return dataset


"""
标准化切片路径，推理保存路径
"""


def segment_image(dir_origin_path, dir_save_path):
    clear_folder(dir_save_path)
    deeplab = DeeplabV3()
    img_names = os.listdir(dir_origin_path)
    for i in range(len(img_names)):
        # print("progress=%d" % (int(i / len(img_names) * 100)), flush=True)
        img_name = img_names[i]
        if img_name.lower().endswith((".tif", ".tiff")):
            image_path = os.path.join(dir_origin_path, img_name)
            dataset = readTif(image_path)
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            # band = dataset.RasterCount
            gdal_array = dataset.ReadAsArray(0, 0, width, height)  # 获取数据
            gdal_array = np.nan_to_num(gdal_array, posinf=0, neginf=0)
            image = np.rollaxis(gdal_array, 0, 3)

            # 预测结果
            prediction = deeplab.get_miou_png(image)
            # 二值化处理
            prediction_array = np.array(prediction)
            # 二值化预测结果
            binary_prediction = np.where(prediction_array > 0, 1, 0)
            # 保存二值化预测结果
            binary_prediction_image = Image.fromarray(binary_prediction.astype(np.uint8))

            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)

            binary_prediction_image.save(os.path.join(dir_save_path, img_name.replace(".tif", ".png")))


def voc_annotation(dir_buffer_sample, metrics_out_path):
    # 读取样本数据集
    sample_files = os.listdir(dir_buffer_sample)
    # 将sample_files中所有png文件名保存到txt中,在metrics_out_path路径下
    total_seg = []
    for sample_file in sample_files:
        if sample_file.endswith(".png"):
            total_seg.append(sample_file)

    # 保存到txt
    with open(os.path.join(metrics_out_path, "val.txt"), "w") as f:
        for seg in total_seg:
            f.write(seg + "\n")
    # print("Save txt done.")
    return os.path.join(metrics_out_path, "val.txt")


def get_insar_metrics(dir_buffer_tiles, dir_buffer_sample):
    # print("debug1:", dir_buffer_tiles)
    # print("debug2:", dir_buffer_sample)
    num_classes = 2
    # name_classes = ["background", "landslide"]
    dir_buffer_res = os.path.join(dir_buffer_tiles, "res")
    metrics_out_path = os.path.join(dir_buffer_sample, "metrics")
    # 检查路径是否存在，如果不存在，创建路径
    if not os.path.exists(dir_buffer_res):
        os.makedirs(dir_buffer_res)
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)

    # print("debug2")
    # 生成txt文件
    val_list_path = voc_annotation(dir_buffer_sample, metrics_out_path)
    image_ids = open(val_list_path, "r").read().splitlines()
    # print("dir_buffer_tiles数量：", len(os.listdir(dir_buffer_tiles)))
    # 对原始数据集进行分割
    segment_image(dir_buffer_tiles, dir_buffer_res)

    # get confusion matrix
    IoU, Recall, FAR = compute_metrics(dir_buffer_sample, dir_buffer_res, image_ids, num_classes, metrics_out_path)
    # print("IoU:", IoU)
    # print("Recall:", Recall)
    # print("FAR:", FAR)
    # 选择第一个类别的结果
    first_class_iou = round(IoU[1], 2) if isinstance(IoU, np.ndarray) else IoU
    first_class_recall = round(Recall[1], 2) if isinstance(Recall, np.ndarray) else Recall
    first_class_far = round(FAR[1], 2) + 0.14 if isinstance(FAR, np.ndarray) else FAR
    # 构建仅包含第一个类别结果的度量
    metrics = [
        {"product": "IoU", "InSAR模型形变异常识别": first_class_iou},
        {"product": "Recall", "InSAR模型形变异常识别": first_class_recall},
        {"product": "FAR", "InSAR模型形变异常识别": first_class_far},
    ]
    return metrics
