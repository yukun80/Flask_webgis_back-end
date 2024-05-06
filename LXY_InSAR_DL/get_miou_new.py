import numpy as np
import os
from osgeo import gdal
from deeplab import DeeplabV3  # 假设有DeeplabV3类
from utils.utils_metrics import compute_mIoU, show_results
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "文件无法打开")
    return dataset


if __name__ == "__main__":
    miou_mode = 0
    num_classes = 2  # 二分类问题
    name_classes = ["Background", "Object"]  # 背景和物体两类

    VOCdevkit_path = r"D:\lxy\kangding\deeplearning_jiance\dataset\dataset_0425\VOCDevkit"

    image_ids = open(os.path.join(VOCdevkit_path, r"VOC2007\ImageSets\Segmentation\val.txt"), "r").read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, r"VOC2007/SegmentationClass/")
    miou_out_path = r"D:\lxy\kangding\deeplearning_jiance\jiance\Data\VOCDevkit\test_temp"
    pred_dir = os.path.join(miou_out_path, "detection-results")

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        deeplab = DeeplabV3()  # 初始化模型
        print("Load model done.")

        print("Get predict result.")
        for image_id in image_ids:
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".tif")

            dataset = readTif(image_path)
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            band = dataset.RasterCount
            proj = dataset.GetProjection()
            geotrans = dataset.GetGeoTransform()
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

            binary_prediction_image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        # 读取验证样本并进行二值化处理
        binary_gt_dir = os.path.join(miou_out_path, "ground-truth")
        if not os.path.exists(binary_gt_dir):
            os.makedirs(binary_gt_dir)
        for image_id in image_ids:
            image_path = os.path.join(gt_dir, image_id + ".png")
            # 读取 PNG 格式的图像
            image = Image.open(image_path)
            # 转换为 numpy 数组
            image_array = np.array(image)
            # 获取图像尺寸
            height, width = image_array.shape
            # 二值化处理
            binary_gt = np.where(image_array[:, :] > 0, 1, 0)
            # 保存二值化处理后的图像
            binary_gt_image = Image.fromarray(binary_gt.astype(np.uint8))
            binary_gt_image.save(os.path.join(binary_gt_dir, image_id + ".png"))

        hist, IoUs, PA_Recall, Precision = compute_mIoU(binary_gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
