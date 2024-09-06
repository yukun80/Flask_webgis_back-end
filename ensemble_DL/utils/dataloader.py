import os

import cv2
import numpy as np
import torch
from osgeo import gdal
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        try:
            annotation_line = self.annotation_lines[index]
            name = annotation_line.split()[0]
        # 接下来的数据处理逻辑...
        except IndexError:
            print(f"Error processing line {index}: '{annotation_line}'")
        # 可以选择跳过这个项目或返回一个默认值
        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#
        tiff_file_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".tif")
        dataset = gdal.Open(tiff_file_path)
        width = dataset.RasterXSize  # 栅格矩阵的列数
        height = dataset.RasterYSize  # 栅格矩阵的行数
        gdal_array = dataset.ReadAsArray(0, 0, width, height)  # 获取数据
        # gdal_array = np.nan_to_num(gdal_array, posinf=0, neginf=0)
        new_array = np.rollaxis(gdal_array, 0, 3)  # (C,H,W) --> (H,W,C)
        # new_array = new_array * 255  # 后向散射在正常范围内，极化分解值可能较低，建议制作数据集时即将部分波段预处理
        jpg = new_array  # (C,H,W) --> (H,W,C)
        # jpg = Image.fromarray(np.uint8(new_array))

        png = Image.open(os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png"))
        # -------------------------------#
        #   数据增强
        # -------------------------------#
        # jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        # 将图像转为np.array格式后，进行归一化，随后转为torch.Tensor，将通道维度放到第一维
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        # 将png中的255转为num_classes
        png[png >= self.num_classes] = self.num_classes
        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new("L", [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_label = Image.new("L", (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


class UnetDataset_Potsdam(Dataset):
    def __init__(self, image_files, label_files, input_shape, num_classes, dataset_path, train):
        super(UnetDataset_Potsdam, self).__init__()
        self.image_files = image_files
        self.label_files = label_files
        self.length = len(self.image_files)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        # print(f"Processing image {image_path} and label {label_path}")
        # 使用gdal读取tif图像
        if self.train:
            tiff_file_path = os.path.join(self.dataset_path, "img_dir/train", image_path + ".tif")
        else:
            tiff_file_path = os.path.join(self.dataset_path, "img_dir/val", image_path + ".tif")
        image_dataset = gdal.Open(tiff_file_path)
        if image_dataset is None:
            raise ValueError(f"无法打开图像文件: {image_path}")

        image_array = image_dataset.ReadAsArray()
        image_array = np.transpose(image_array, (1, 2, 0))  # 转换为(H, W, C)

        # 使用gdal读取tif标签
        if self.train:
            tiff_label_path = os.path.join(self.dataset_path, "ann_dir/train", label_path + ".tif")
        else:
            tiff_label_path = os.path.join(self.dataset_path, "ann_dir/val", label_path + ".tif")

        label_dataset = gdal.Open(tiff_label_path)
        if label_dataset is None:
            raise ValueError(f"无法打开标签文件: {label_path}")
        label_array = label_dataset.ReadAsArray().astype(np.int64)

        if image_array.shape[:2] != label_array.shape[:2]:
            raise ValueError(f"图像和标签大小不匹配: {image_path} 和 {label_path}")

        # 确保标签数组的大小与目标形状匹配
        if label_array.shape[0] != self.input_shape[0] or label_array.shape[1] != self.input_shape[1]:
            print(f"Error label path: {label_path}")
            raise ValueError(f"标签数组大小 {label_array.shape} 与目标形状 {self.input_shape} 不匹配")

        # 数据预处理和增强
        image_array = np.transpose(preprocess_input(np.array(image_array, np.float64)), [2, 0, 1])
        label_array[label_array >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[label_array.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return image_array, label_array, seg_labels


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs, dtype=np.int8)).long()  # 转换数据类型为 int32
    # pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
