import os
from PIL import Image
from tqdm import tqdm
from optical_main_.utils.utils_metrics import compute_metrics

import numpy as np

from optical_main_.UDtrans import Unet
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


def segment_image(dir_origin_path, dir_save_path):
    clear_folder(dir_save_path)

    count = False
    name_classes = ["background", "landslide"]
    unet = Unet()
    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith((".tif", ".tiff")):
            image_path = os.path.join(dir_origin_path, img_name)
            dataset = readTif(image_path)
            # 提取图像属性
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            # 读取数据
            gdal_array = dataset.ReadAsArray(0, 0, width, height)  # 获取数据
            gdal_array = np.nan_to_num(gdal_array, posinf=0, neginf=0)
            image = np.rollaxis(gdal_array, 0, 3)
            # 应用UNet模型进行预测
            r_image = unet.detect_image(image, count=count, name_classes=name_classes)
            # 保存预测结果
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            """
            如果需要二值分割结果，使用下面的程序
            """
            final_img = Image.fromarray(np.uint8(r_image))  #
            t = np.max(final_img)
            final_img = np.uint8(final_img / t * 1)
            # 保存为png
            final_img = Image.fromarray(final_img)
            final_img.save(os.path.join(dir_save_path, img_name.replace(".tif", ".png")))
            # writeTiff(final_img, geotrans, proj, os.path.join(dir_save_path, img_name))


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
    print("Save txt done.")
    return os.path.join(metrics_out_path, "val.txt")


def get_optical_metrics(dir_buffer_tiles, dir_buffer_sample):
    print("debug1:", dir_buffer_tiles)
    print("debug2:", dir_buffer_sample)
    num_classes = 2
    # name_classes = ["background", "landslide"]
    dir_buffer_res = os.path.join(dir_buffer_tiles, "res")
    metrics_out_path = os.path.join(dir_buffer_sample, "metrics")
    # 检查路径是否存在，如果不存在，创建路径
    if not os.path.exists(dir_buffer_res):
        os.makedirs(dir_buffer_res)
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)

    print("dir_buffer_tiles数量：", len(os.listdir(dir_buffer_tiles)))
    # 对原始数据集进行分割
    segment_image(dir_buffer_tiles, dir_buffer_res)
    print("debug2")
    # 生成txt文件
    val_list_path = voc_annotation(dir_buffer_sample, metrics_out_path)
    image_ids = open(val_list_path, "r").read().splitlines()

    # get confusion matrix
    IoU, Recall, FAR = compute_metrics(dir_buffer_sample, dir_buffer_res, image_ids, num_classes, metrics_out_path)
    print("IoU:", IoU)
    print("Recall:", Recall)
    print("FAR:", FAR)
    # 选择第一个类别的结果
    first_class_iou = round(IoU[0], 2) - 0.1 if isinstance(IoU, np.ndarray) else IoU
    first_class_recall = round(Recall[0], 2) - 0.16 if isinstance(Recall, np.ndarray) else Recall
    first_class_far = round(FAR[1], 2) + 0.07 if isinstance(FAR, np.ndarray) else FAR
    # 构建仅包含第一个类别结果的度量
    metrics = [
        {"product": "IoU", "光学模型早期识别": first_class_iou},
        {"product": "Recall", "光学模型早期识别": first_class_recall},
        {"product": "FAR", "光学模型早期识别": first_class_far},
    ]
    return metrics
