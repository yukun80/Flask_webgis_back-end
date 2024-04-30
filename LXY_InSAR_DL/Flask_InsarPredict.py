import numpy as np

from LXY_InSAR_DL.deeplab import DeeplabV3

from osgeo import gdal
import os


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
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def predict_main(dir_origin_path, dir_save_path, result_name="result"):
    print("检测函数成功进入1")
    print(dir_origin_path)
    print(dir_save_path)
    print(result_name)
    # clear_folder(dir_save_path)
    deeplab = DeeplabV3()
    count = False
    name_classes = ["Safe Zone", "Low-Risk Zone", "Moderate-Risk Zone", "High-Risk Zone", "Very High-Risk Zone"]

    img_names = os.listdir(dir_origin_path)
    for i in range(len(img_names)):
        print("progress=%d" % (int(i / len(img_names) * 100)), flush=True)
        img_name = img_names[i]
        if img_name.lower().endswith((".tif", ".tiff")):
            image_path = os.path.join(dir_origin_path, img_name)
            dataset = readTif(image_path)
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            band = dataset.RasterCount
            proj = dataset.GetProjection()
            geotrans = dataset.GetGeoTransform()
            gdal_array = dataset.ReadAsArray(0, 0, width, height)  # 获取数据
            gdal_array = np.nan_to_num(gdal_array, posinf=0, neginf=0)
            image = np.rollaxis(gdal_array, 0, 3)
            # topimage = image[:, :, 0]
            # image = np.expand_dims(topimage, 2).repeat(2, axis=2)
            image[image < -10] = 0
            result_img = np.full_like(image, 0)[:, :, 0]
            temp_img = result_img[:, :]
            temp_list = []
            temp_list.append(temp_img)
            temp_list.append(temp_img)
            temp_list.append(temp_img)
            result_img = np.array(temp_list)
            result_img = np.rollaxis(result_img, 0, 3)

            CropSize = 256
            RepetitionRate = 0.5  # 0.5
            for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                    sub_img = image[
                        int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                        int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize,
                        :,
                    ]
                    # sub_img = Image.fromarray(np.uint8(sub_img))
                    sub_r_image = deeplab.detect_image(sub_img, count=count, name_classes=name_classes)
                    result_img[
                        int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                        int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize,
                        :,
                    ] = sub_r_image

            for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                sub_img = image[
                    int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                    (width - CropSize) : width,
                    :,
                ]
                # sub_img = Image.fromarray(np.uint8(sub_img))
                sub_r_image = deeplab.detect_image(sub_img, count=count, name_classes=name_classes)
                result_img[
                    int(i * CropSize * (1 - RepetitionRate)) : int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                    (width - CropSize) : width,
                    :,
                ] = sub_r_image

            for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                sub_img = image[
                    (height - CropSize) : height,
                    int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize,
                    :,
                ]
                # sub_img = Image.fromarray(np.uint8(sub_img))
                sub_r_image = deeplab.detect_image(sub_img, count=count, name_classes=name_classes)
                result_img[
                    (height - CropSize) : height,
                    int(j * CropSize * (1 - RepetitionRate)) : int(j * CropSize * (1 - RepetitionRate)) + CropSize,
                    :,
                ] = sub_r_image

            # 最后一块
            sub_img = image[(height - CropSize) : height, (width - CropSize) : width, :]
            # sub_img = Image.fromarray(np.uint8(sub_img))
            sub_r_image = deeplab.detect_image(sub_img, count=count, name_classes=name_classes)
            result_img[(height - CropSize) : height, (width - CropSize) : width, :] = sub_r_image

            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            final_img = np.transpose(result_img, [2, 0, 1])
            t = np.max(final_img)
            # final_img = np.uint8(final_img / t * 1)
            final_img = np.uint8(final_img / t * 255)
            writeTiff(final_img[0, :, :], geotrans, proj, os.path.join(dir_save_path, result_name))
