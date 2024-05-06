import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from osgeo import gdal
import rasterio
from rasterio.windows import Window

# from tqdm import tqdm
import shutil

import Flask_tif2shp
import Flask_res_tiles_merge
import Flask_geoserver
import optical_main_.Flask_predict as Flask_predict
import LXY_InSAR_DL.Flask_InsarPredict as Flask_InsarPredict
import ensemble_ML.Flask_fusionDetection as Flask_fusionDetection
from optical_main_.get_miou import get_optical_metrics
from LXY_InSAR_DL.get_miou import get_insar_metrics
from ensemble_ML.get_miou import get_ensemble_metrics

Flask_app = Flask(__name__)
CORS(Flask_app)
# app = Flask(__name__)
# app.logger.setLevel(logging.INFO)


### step1: 检查数据是否满足要求
def process_sentinel_image_nan(file_path):
    """
    直接处理 Sentinel-2 影像中的每个波段，将所有NaN值设置为0。
    """
    dataset = gdal.Open(file_path, gdal.GA_Update)
    if dataset is None:
        print(f"Unable to open file {file_path}")
        return

    # print(dataset.RasterCount)
    result_info = {"file_path": file_path, "bands_processed": []}
    for band in range(1, dataset.RasterCount + 1):
        raster_band = dataset.GetRasterBand(band)
        nodata_value = raster_band.GetNoDataValue()  # 获取 NoData 值
        # print("nodata_value:", nodata_value)
        data = raster_band.ReadAsArray().astype(float)  # 读取波段数据

        # 如果存在 NoData 值，将其转换为 NaN，然后将所有NaN值设置为0
        if nodata_value is not None:
            data[data == nodata_value] = np.nan

        # print(data.shape)
        # 直接将所有NaN值设置为0
        data[np.isnan(data)] = 0
        print(f"Band {band}: NaN or NoData values processed. Setting NaNs to zero...")

        raster_band.WriteArray(data)
        result_info["bands_processed"].append(band)

    dataset = None  # 关闭文件
    return result_info, True


### step2:遥感影像切片
## 2.1 清空目标文件夹
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


## 2.1 分层归一化
def normalize_data(data):
    """
    Normalizes the data using the formula from the provided image:
    for each band of the image, subtract the mean of the band and divide by the standard deviation.
    """
    # Calculate the mean and std for each band
    means = np.mean(data, axis=(1, 2), keepdims=True)
    stds = np.std(data, axis=(1, 2), keepdims=True)
    # Avoid division by zero
    stds[stds == 0] = 1
    # Normalize data
    normalized_data = (data - means) / stds
    return normalized_data


## 2.2 切片操作

# 分块大小
block_size = 256


# 分块读取和处理函数，添加了补0操作
def process_block(src, x_start, y_start, block_size):
    # 读取图层的块
    block = src.read(window=Window(x_start, y_start, block_size, block_size))

    # 补0操作：确保块的形状是 (num_channels, 256, 256)，先创建一个全0的padded_block
    padded_shape = (block.shape[0], block_size, block_size)
    padded_block = np.zeros(padded_shape, dtype=block.dtype)

    # 填充实际数据
    padded_block[:, : block.shape[1], : block.shape[2]] = block

    return padded_block


# 直接保存分块函数
def save_block(block, profile, x_idx, y_idx, folder):
    output_path = os.path.join(folder, f"sliced_{x_idx}_{y_idx}.tif")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(block)


def process_and_slice_images(file_paths, output_folder):
    # 检查文件夹是否存在，如果存在则创建一个新的文件夹
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    clear_folder(output_folder)
    # 同时打开保存在列表中
    src_files = [rasterio.open(fp) for fp in file_paths]

    # 打开所有文件并检查尺寸一致性
    src_files = [rasterio.open(fp) for fp in file_paths]
    shapes = [src.shape for src in src_files]
    if not all(shape == shapes[0] for shape in shapes[1:]):
        raise ValueError("Global dimensions must match among all files")

    # 准备合并后的profile
    profile_combined = src_files[0].profile.copy()
    # 更新元数据以反映所有波段的总数和新的切片尺寸
    profile_combined["count"] = sum(src.count for src in src_files)
    profile_combined["height"] = profile_combined["width"] = block_size

    # 得到总行数和列数
    overall_height, overall_width = src_files[0].shape

    # 分块遍历并处理
    for i in range(0, overall_height, block_size):
        for j in range(0, overall_width, block_size):
            # 为每个文件生成和堆叠切片
            blocks = [process_block(src, j, i, block_size) for src in src_files]
            combined_block = np.vstack(blocks)
            # combined_block = normalize_data(combined_block)
            save_block(combined_block, profile_combined, i // block_size, j // block_size, output_folder)


# 设置工作目录
PATH_MAP = {"/default": "D:\\_codeProject\\SlideDetect-main\\DataCollection\\7_System"}
# PATH_MAP = {"/default": "D:\\DisasterWebSys\\DataStorage"}


def resolve_path(directory_path):
    """解析请求中的路径参数，将其映射为实际的文件系统路径。"""
    for key, value in PATH_MAP.items():
        if directory_path.startswith(key):
            return directory_path.replace(key, value, 1)
    # 如果路径不在PATH_MAP中，返回None或抛出异常
    return None


@Flask_app.route("/list_paths", methods=["GET"])
def list_paths():
    # 从请求中获取目录路径参数，默认为某个根目录
    directory_path = request.args.get("dir", "/default")
    # 将映射的路径转换为实际的文件系统路径
    directory_path = resolve_path(directory_path)  # 解析路径
    # 确保路径存在
    if not directory_path or not os.path.exists(directory_path):
        return jsonify({"error": "Directory does not exist"}), 404
    try:
        # 列出目录下的所有文件和文件夹
        paths = [{"name": item, "isDir": os.path.isdir(os.path.join(directory_path, item))} for item in os.listdir(directory_path)]
        # paths = list_files_recursive(directory_path)
        return jsonify(paths)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 获取所有图层路径
@Flask_app.route("/list_layers", methods=["GET"])
def list_layers():
    workspace = request.args.get("workspace", "predict_result")
    status, layers_names = Flask_geoserver.get_layers_from_workspace(workspace)
    if status == 200:
        # 成功获取图层名称，返回JSON列表
        return jsonify({"layers": layers_names})
    else:
        # 返回错误信息
        return jsonify({"error": layers_names}), status


# 栅格转矢量
@Flask_app.route("/tif2shp_calculate", methods=["POST"])
def tif2shp_calculate():
    # 从前端获取路径
    data = request.json
    tifImage = data.get("tifImage")
    tifType = data.get("tifType")
    res_folder = data.get("res_folder")
    res_name = data.get("res_name")
    threshold = 0
    if tifType == "optical":
        threshold = 0.7
    elif tifType == "insar":
        threshold = 62
    elif tifType == "ensemble":
        threshold = 7

    if not tifImage or not res_folder:
        return jsonify({"error": "Missing required parameters"}), 400
    # print("+++++++++++++++debug0 tifImage:", tifImage, "\n")
    # print("+++++++++++++++debug1 res_folder:", res_folder, "\n")
    # print("+++++++++++++++debug2 res_name:", res_name, "\n")
    # print("+++++++++++++++debug3 threshold:", threshold, "\n")
    # print("+++++++++++++++debug4 tifType:", tifType, "\n")
    try:
        resolved_tifImage = resolve_path(tifImage)
        resolved_res_folder = resolve_path(res_folder)
        # 结果路径
        result_path = os.path.join(resolved_res_folder, res_name)
        # print("+++++++++++++++debug1 resolved_tifImage:", resolved_tifImage, "\n")
        # print("+++++++++++++++debug2 resolved_res_folder:", resolved_res_folder, "\n")
        # print("+++++++++++++++debug3 result_path:", result_path, "\n")
        # print("+++++++++++++++debug4 threshold:", threshold, "\n")
        # 生成矢量边界
        Flask_tif2shp.extract_and_sort_vector_boundary(resolved_tifImage, threshold, result_path, tifType)

        shp_name, extension = os.path.splitext(res_name)
        # print(shp_name + "_debug1")
        Flask_geoserver.upload_shapefile(shp_name, result_path)
        # 获取图层geojson链接
        geojson_url = Flask_geoserver.get_geojson_url(shp_name)
        # 返回成功响应以及适量geojson链接
        return jsonify({"message": "Success", "geojson_url": geojson_url}), 200
    except Exception as e:
        # 捕获异常并返回错误响应
        # app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500


# 影像标准化处理
@Flask_app.route("/process_images", methods=["GET"])
def process_images():
    file_paths_str = request.args.get("file_paths")  # 遥感影像路径列表
    output_folder_path = request.args.get("output_folder")  # 获取输出文件夹路径

    if not file_paths_str or not output_folder_path:
        return jsonify({"error": "Missing required parameters"}), 400

    # 使用逗号分隔的字符串获取文件路径数组
    file_paths = file_paths_str.split(",")
    # 使用resolve_path函数转换每个文件路径
    resolved_file_paths = [resolve_path(path) for path in file_paths]
    # 转换输出文件夹路径
    resolved_output_folder_path = resolve_path(output_folder_path)

    results = []

    for file_path in resolved_file_paths:
        if not os.path.exists(file_path):
            return jsonify({"error": "File does not exist", "file_path": file_path}), 404

        result_info, success = process_sentinel_image_nan(file_path)
        if success:
            results.append(result_info)
        else:
            return jsonify({"error": result_info, "file_path": file_path}), 500

    # 确保输出文件夹路径存在，若不存在可以选择创建
    if not os.path.exists(resolved_output_folder_path):
        os.makedirs(resolved_output_folder_path, exist_ok=True)

    process_and_slice_images(resolved_file_paths, resolved_output_folder_path)

    return jsonify(results), 200


# 光学检测
@Flask_app.route("/predict_images", methods=["POST"])
def predict_images():
    # 从前端获取路径
    data = request.json
    predict_tiles_path = data.get("tiles_folder")  # 从JSON体中获取
    predict_res_path = data.get("res_folder")
    result_name = data.get("res_name")

    if not predict_tiles_path or not predict_res_path:
        return jsonify({"error": "Missing required parameters"}), 400
    try:
        # 标准切片数据集路径
        resolved_tiles_path = resolve_path(predict_tiles_path)
        # 结果文件夹路径
        resolved_res_path = resolve_path(predict_res_path)
        # 临时文件夹保存推理切片
        predict_tiles_path = os.path.join(resolved_res_path, "temp")
        # 切片推理
        Flask_predict.predict_main(resolved_tiles_path, predict_tiles_path)
        # 结果路径
        result_path = os.path.join(resolved_res_path, result_name)
        # 合并切片
        Flask_res_tiles_merge.merge_tiles(predict_tiles_path, result_path, block_size)
        # 清空临时文件夹
        clear_folder(predict_tiles_path)

        layer_name, extension = os.path.splitext(result_name)
        # print(layer_name + "_debug")
        # print(result_path + "_debug2")

        (
            store_status,
            store_msg,
            layer_status,
            layer_msg,
        ) = Flask_geoserver.upload_to_geoserver(result_path, layer_name, layer_name)
        # print(store_status, store_msg, layer_status, layer_msg)

        # 指定图层样式
        # 上传成功，指定样式1
        # random_num = random.randint(1, 6)
        random_num = 1
        style_name = f"ResultStyle_{random_num}"
        # print("debug" + style_name)
        style_status_code, style_msg = Flask_geoserver.assign_style_to_layer(layer_name, style_name)
        # print("style_status", style_status_code, style_msg)
        # 返回成功响应
        return jsonify({"message": "Success"}), 200
    except Exception as e:
        # 捕获异常并返回错误响应
        return jsonify({"error": str(e)}), 500


# InSAR检测
@Flask_app.route("/predict_insar", methods=["POST"])
def predict_insar():
    data = request.json
    predict_tiles_path = data.get("tiles_folder")
    predict_res_path = data.get("res_folder")
    result_name = data.get("res_name")

    if not predict_tiles_path or not predict_res_path:
        return jsonify({"error": "Missing required parameters"}), 400
    try:
        resolved_tiles_path = resolve_path(predict_tiles_path)
        resolved_res_path = resolve_path(predict_res_path)

        Flask_InsarPredict.predict_main(resolved_tiles_path, resolved_res_path, result_name)
        result_path = os.path.join(resolved_res_path, result_name)

        layer_name, extension = os.path.splitext(result_name)
        # print(layer_name + "_debug")
        # print(result_path + "_debug2")

        (
            store_status,
            store_msg,
            layer_status,
            layer_msg,
        ) = Flask_geoserver.upload_to_geoserver(result_path, layer_name, layer_name)
        # print(store_status, store_msg, layer_status, layer_msg)

        # 指定图层样式
        # 上传成功，随机指定一个样式
        # random_num = random.randint(1, 6)
        random_num = 2
        style_name = f"ResultStyle_{random_num}"
        # print("debug" + style_name)
        style_status_code, style_msg = Flask_geoserver.assign_style_to_layer(layer_name, style_name)
        # print("style_status", style_status_code, style_msg)
        # 返回成功响应
        return jsonify({"message": "Success"}), 200
    except Exception as e:
        # 捕获异常并返回错误响应
        return jsonify({"error": str(e)}), 500


# 集成学习检测
@Flask_app.route("/predict_multiResult", methods=["POST"])
def predict_multiResult():
    data = request.json
    predict_optical_path = data.get("optical_file")
    # predict_insar_path = data.get("insar_file")
    # predict_susceptible_path = data.get("landslide_file")
    predict_res_path = data.get("res_folder")
    result_name = data.get("res_name")
    # print("+++++++++++++++++++++++++++++debug0 result_name", result_name, "\n")
    if not predict_optical_path:
        return jsonify({"error": "Missing required parameters"}), 400
    try:
        resolved_optical_path = resolve_path(predict_optical_path)
        # resolved_insar_path = resolve_path(predict_insar_path)
        # resolved_susceptible_path = resolve_path(predict_susceptible_path)
        resolved_res_path = resolve_path(predict_res_path)
        # print("+++++++++++++++++++++++++++++debug1 resolved_optical_path", resolved_optical_path, "\n")
        # print("+++++++++++++++++++++++++++++debug2 resoloved_res_path", resolved_res_path, "\n")

        # Flask_fusionDetection.multi_predict(resolved_optical_path, resolved_insar_path, resolved_susceptible_path, resolved_res_path, result_name)
        Flask_fusionDetection.ensemble_predict(resolved_optical_path, resolved_res_path, result_name)

        result_path = os.path.join(resolved_res_path, result_name)
        layer_name, extension = os.path.splitext(result_name)
        # print(layer_name + "_debug")
        # print(result_path + "_debug2")
        (
            store_status,
            store_msg,
            layer_status,
            layer_msg,
        ) = Flask_geoserver.upload_to_geoserver(result_path, layer_name, layer_name)
        # print(store_status, store_msg, layer_status, layer_msg)
        # 指定图层样式
        # 上传成功，随机指定一个样式
        # random_num = random.randint(1, 6)
        random_num = 3
        style_name = f"ResultStyle_{random_num}"
        # print("debug" + style_name)
        style_status_code, style_msg = Flask_geoserver.assign_style_to_layer(layer_name, style_name)
        # print("style_status", style_status_code, style_msg)
        # 返回成功响应
        return jsonify({"message": "Success"}), 200
    except Exception as e:
        # 捕获异常并返回错误响应
        return jsonify({"error": str(e)}), 500


# 获取工作空间列表
@Flask_app.route("/get_workspaces", methods=["GET"])
def get_workspaces():
    status, workspaces = Flask_geoserver.get_all_workspaces()
    if status == 200:
        return jsonify({"workspaces": workspaces})
    else:
        return jsonify({"error": workspaces}), status


# 获取目标工作空间中图层的名字和数据类型
@Flask_app.route("/get_layers", methods=["GET"])
def get_layers():
    workspace = request.args.get("workspace")
    if not workspace:
        return jsonify({"error": "Workspace parameter is missing"}), 400
    # 调用先前定义的获取图层信息的函数
    layer_info = Flask_geoserver.get_layers_and_types_in_workspace(workspace)
    if layer_info:
        if isinstance(layer_info, list):
            return jsonify({"layers": layer_info})
        else:
            # 假设如果不是列表，则为错误消息
            return jsonify({"error": layer_info}), 500
    else:
        return jsonify({"error": "Failed to retrieve layer information"}), 500


# 获取图层的GeoJSON数据
@Flask_app.route("/get_geojson", methods=["GET"])
def get_geojson():
    layer_name = request.args.get("layer_name")
    workspace = request.args.get("workspace")
    if not layer_name or not workspace:
        return jsonify({"error": "Missing required parameters"}), 400
    # 获取GeoJSON URL
    geojson_url = Flask_geoserver.get_geojson_url(layer_name, workspace)
    if geojson_url:
        return jsonify({"geojson_url": geojson_url}), 200
    else:
        return jsonify({"error": "Failed to retrieve GeoJSON URL"}), 500


# 删除数据库中的图层和存储仓库
@Flask_app.route("/delete_data", methods=["POST"])
def delete_data():
    layer_name = request.json.get("layer_name")
    workspace = request.json.get("workspace")
    layerClass = request.json.get("layer_class")
    if not layer_name or not workspace:
        return jsonify({"error": "Missing required parameters"}), 400
    # 删除数据
    try:
        # 可以在这里调用删除函数，并处理可能的异常或错误
        logOutput = Flask_geoserver.delete_layer_and_store(layer_name, workspace, layerClass)
        print("log:", logOutput)
        return jsonify({"success": f"Layer and store {layer_name} in workspace {workspace} have been deleted"}), 200
    except Exception as e:
        # 如果发生错误，返回错误信息
        print(e)  # 打印异常信息到控制台
        return jsonify({"error": str(e)}), 500


# 获取定量评估指标
@Flask_app.route("/get_metrics", methods=["POST"])
def get_metrics():
    data = request.json
    radio1 = data["radio1"]
    dir_buffer_tiles = data["dir_buffer_tiles"]
    dir_buffer_sample = data["dir_buffer_sample"]
    resolved_buffer_tiles = resolve_path(dir_buffer_tiles)
    resolved_buffer_sample = resolve_path(dir_buffer_sample)
    if radio1 == 1:
        print("radio1:", radio1)
        metrics = get_optical_metrics(resolved_buffer_tiles, resolved_buffer_sample)
    elif radio1 == 2:
        print("radio1:", radio1)
        metrics = get_insar_metrics(resolved_buffer_tiles, resolved_buffer_sample)
    elif radio1 == 3:
        print("radio1:", radio1)
        metrics = get_ensemble_metrics(resolved_buffer_tiles, resolved_buffer_sample)
    else:
        return jsonify({"success": "Invalid radio button value"}), 400

    return jsonify({"metrics": metrics}), 200


@Flask_app.route("/")
def home():
    return "Welcome to the Data Processing Application!"


if __name__ == "__main__":
    Flask_app.run(debug=True)
