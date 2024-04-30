# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import numpy as np
from PIL import Image

from unet import Unet_ONNX, Unet
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


if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在unet.py_346行左右处的Unet_ONNX
    # ----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    # -------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    count = False
    name_classes = ["background", "landslide"]
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    # ---LanC流域
    # 定性分析测试区域
    # dir_origin_path = "../DataCollection/4_TestData/LanC/_Test_combined_tiles_256"
    # dir_save_path = "../DataCollection/5_Res/LanC/Res_tiles_unetCot_256"
    # dir_save_path = '../DataCollection/5_Res/LanC/Res_tiles_unet3_256/'
    # 定量分析测试区域
    # dir_origin_path = "../DataCollection/2_TrainData/LanC/_Train_combined_tiles_256"
    # dir_save_path = "../DataCollection/6_Eva/LanC/UNet_256/tif"
    # dir_save_path = "../DataCollection/6_Eva/LanC/Cot-UNet_256/tif"
    # dir_save_path = "../DataCollection/6_Eva/LanC/UNet3_256/tif"
    # dir_save_path = "../DataCollection/6_Eva/LanC/ResUNet_256/tif"

    # ---JinS流域
    # 定性分析测试区域
    # dir_origin_path = "../DataCollection/4_TestData/JinS/_Test_combined_tiles_256"
    # dir_save_path = "../DataCollection/5_Res/JinS/Res_tiles_DropUNet_256_231218"
    # dir_save_path = "../DataCollection/5_Res/JinS/Res_tiles_DropUNet_256_231220"
    # 定量分析区域
    # dir_origin_path = "../DataCollection/4_TestData/JinS/_Val_combined_tiles_256"
    # dir_save_path = "../DataCollection/6_Eva/JinS/DropUNet_256_231218/tif"
    # dir_save_path = "../DataCollection/6_Eva/JinS/DropUNet_256_231220/tif"

    # ---CZ全域
    # 定性分析测试区域
    # dir_origin_path = "../DataCollection/4_TestData/CZ/_Test_combined_tiles_256"
    # dir_save_path = "../DataCollection/5_Res/CZ/Res_tiles_DropU_CZ_256_231224"
    # dir_save_path = "../DataCollection/5_Res/CZ/Res_tiles_DropU_ALLCZ_256_231224"
    # dir_save_path = "../DataCollection/5_Res/CZ/Res_tiles_DropU_CZ_256_231226"
    # dir_save_path = "../DataCollection/5_Res/CZ/Res_tiles_LovaszU_CZ_256_231228"
    # 定量分析区域
    # dir_origin_path = "../DataCollection/4_TestData/CZ/Cal_test_combined_tiles_256"
    dir_origin_path = "../DataCollection/4_TestData/CZ/test_combined_tiles_5km"
    # dir_save_path = "../DataCollection/6_Eva/CZ/DropU_ALLCZ_256_231224/tif"
    # dir_save_path = "../DataCollection/6_Eva/CZ/DropU_CZ_256_231224/tif"
    # dir_save_path = "../DataCollection/6_Eva/CZ/DropU_CZ_256_231226/tif"
    # dir_save_path = "../DataCollection/6_Eva/CZ/LovaszU_CZ_256_231228/tif"
    dir_save_path = "../DataCollection/6_Eva/CZ/DropU_CZ5km_256_231226/tif"

    # dir_save_path = "./VOCdevkit/VOC2007/Res_tiles_eva"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode != "predict_onnx":
        unet = Unet()
    else:
        yolo = Unet_ONNX()

    if mode == "predict":
        """
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        """
        while True:
            img = input("Input image filename:")
            try:
                image = Image.open(img)
            except:
                print("Open Error! Try again!")
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

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
                final_img = Image.fromarray(np.uint8(r_image))  #
                # >> 如果结果为三通道RGB，进行转置
                # final_img = np.transpose(r_image, [2, 0, 1])
                t = np.max(final_img)
                final_img = np.uint8(final_img / t * 1)
                # 保存为单波段 TIFF 格式
                # final_img = np.uint8(r_image)
                # print("final_img.shape: ", final_img.shape)
                writeTiff(final_img, geotrans, proj, os.path.join(dir_save_path, img_name))

    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input("Input image filename:")
            try:
                image = Image.open(img)
            except:
                print("Open Error! Try again!")
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
