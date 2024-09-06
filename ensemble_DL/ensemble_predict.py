import os
from tqdm import tqdm
import numpy as np
from PIL import Image

# from UDtrans import Unet_ONNX, Unet
from ensemble_DL.UDtrans_prob import Unet_ONNX, Unet
from osgeo import gdal


#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "文件无法打开")
    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if "int8" in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif "int16" in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:  # 单波段
        # 获取图像的高度和宽度
        im_height, im_width = im_data.shape
        # im_data = np.array([im_data])
        # im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    # dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    # 单波段
    dataset = driver.Create(path, int(im_width), int(im_height), 1, datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        # 单波段
        dataset.GetRasterBand(1).WriteArray(im_data)
    # for i in range(im_bands):
    #     dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def predict_main(dir_origin_path, dir_save_path):

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
            band = dataset.RasterCount
            proj = dataset.GetProjection()
            geotrans = dataset.GetGeoTransform()
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
            添加输出概率代码，直接保存浮点数组而不转换为uint8
            注释掉下面三行
            """
            prob_map = r_image[..., 1]
            # # Convert PIL Image to NumPy array before attempting to change its type
            writeTiff(prob_map.astype(np.float32), geotrans, proj, os.path.join(dir_save_path, img_name))

            """
            如果需要二值分割结果，使用下面的程序
            """
            # final_img = Image.fromarray(np.uint8(r_image))  #
            # t = np.max(final_img)
            # final_img = np.uint8(final_img / t * 1)
            # # 保存为单波段 TIFF 格式
            # writeTiff(final_img, geotrans, proj, os.path.join(dir_save_path, img_name))
