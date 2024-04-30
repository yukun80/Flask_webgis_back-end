import os
import rasterio

from rasterio.windows import Window


def get_tile_indices(tile_name):
    """
    从文件名中提取坐标索引。
    假设文件名格式为 'sliced_x_idx_y_idx.tif'
    """
    _, x_idx, y_idx = tile_name.split(".")[0].split("_")
    return int(y_idx), int(x_idx)


def merge_tiles(tiles_folder, output_path, block_size=256):
    # 如果没有result文件夹，则创建一个
    if not os.path.exists(tiles_folder + "/result"):
        os.makedirs(tiles_folder + "/result")
    # 获取所有切片的文件路径
    tile_paths = [os.path.join(tiles_folder, f) for f in os.listdir(tiles_folder) if f.endswith(".tif")]

    if not tile_paths:
        raise ValueError("No tile files found in the specified folder.")

    # 使用第一个切片的元数据作为输出文件的模板
    with rasterio.open(tile_paths[0]) as src:
        profile = src.profile.copy()

    # 更新输出文件的尺寸
    num_blocks_x = max(get_tile_indices(os.path.basename(path))[0] for path in tile_paths) + 1
    num_blocks_y = max(get_tile_indices(os.path.basename(path))[1] for path in tile_paths) + 1
    profile.update(width=block_size * num_blocks_x, height=block_size * num_blocks_y)

    # 创建并写入输出文件
    with rasterio.open(output_path, "w", **profile) as dst:
        # 遍历每个切片文件
        for tile_path in tile_paths:
            with rasterio.open(tile_path) as tile:
                x_idx, y_idx = get_tile_indices(os.path.basename(tile_path))

                # 读取切片数据
                data = tile.read()
                # 写入到目标文件的正确位置
                dst.write(data, window=Window(x_idx * block_size, y_idx * block_size, block_size, block_size))


# 示例使用
# tiles_folder = "../DataCollection/6_Eva/CZ/LovaszU_CZ5km_256_231228/tif"
# output_path = '../DataCollection/5_Res/CZ/Res_tiles_segfB2_CZ_256_231228/result/segfB2_CZ_256.tif'  #
# output_path = "../DataCollection/6_Eva/CZ/LovaszU_CZ5km_256_231228/tif/result/LovaszU_CZ5km_256.tif"  #

# block_size = 256  # 块的大小# merge_tiles(tiles_folder, output_path, block_size)
