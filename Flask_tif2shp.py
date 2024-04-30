from osgeo import gdal, ogr, osr
import numpy as np
import math


"""栅格转矢量
raster_path:栅格路径
threshold: 提取阈值
sorted_output_shapefile: 排序后的矢量输出路径
"""


def extract_and_sort_vector_boundary(raster_path, threshold, sorted_output_shapefile, shp_class):
    # 打开栅格数据
    raster = gdal.Open(raster_path)
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()

    # 创建二值数组，大于等于阈值为1，小于阈值为0
    binary_array = np.where(array >= threshold, 1, 0)

    # 将数组转换回GDAL栅格格式
    driver = gdal.GetDriverByName("MEM")
    binary_raster = driver.Create("", raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_Byte)
    binary_raster.SetProjection(raster.GetProjection())
    binary_raster.SetGeoTransform(raster.GetGeoTransform())
    binary_raster_band = binary_raster.GetRasterBand(1)
    binary_raster_band.WriteArray(binary_array)

    # 创建shapefile
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")
    if shp_driver is None:
        raise ValueError("ESRI Shapefile driver not available.")
    output_vector = shp_driver.CreateDataSource("temp.shp")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjection())
    layer = output_vector.CreateLayer("temp", srs=srs, geom_type=ogr.wkbPolygon)

    # 添加字段
    field_class = ogr.FieldDefn("Class", ogr.OFTString)
    field_class.SetWidth(20)  # 设定一个适当的长度
    layer.CreateField(field_class)
    # 添加实数类型字段，名字为"Area"
    field_area = ogr.FieldDefn("Area_m^2", ogr.OFTReal)
    layer.CreateField(field_area)
    field_longitude = ogr.FieldDefn("Long_deg", ogr.OFTReal)
    layer.CreateField(field_longitude)
    field_latitude = ogr.FieldDefn("Lat_deg", ogr.OFTReal)
    layer.CreateField(field_latitude)

    # 栅格转矢量，只转化非0值（即二值化中的1，大于等于阈值的部分）
    mask_band = binary_raster_band  # 使用同一个二值化后的栅格带作为mask
    gdal.Polygonize(binary_raster_band, mask_band, layer, -1, ["8CONNECTED=8"], callback=None)
    # 计算面积和坐标，写入属性表
    layer.StartTransaction()
    for feature in layer:
        geom = feature.GetGeometryRef()
        centroid = geom.Centroid()
        centroid_x = centroid.GetX()
        centroid_y = centroid.GetY()
        # 考虑纬度影响的面积计算
        area_corrected = geom.GetArea() * (111194.926644558737**2) * math.cos(math.radians(centroid_y))
        feature.SetField("Class", shp_class)
        feature.SetField("Area_m^2", round(area_corrected, 2))
        feature.SetField("Long_deg", round(centroid_x, 2))
        feature.SetField("Lat_deg", round(centroid_y, 2))
        layer.SetFeature(feature)
    layer.CommitTransaction()

    # 关闭临时矢量文件
    output_vector.Destroy()

    # 读取临时文件并按面积排序
    dataSource = shp_driver.Open("temp.shp", 0)  # 0 means read-only
    layer = dataSource.GetLayer()
    features = [(feature.Clone(), feature.GetField("Area_m^2")) for feature in layer]
    features.sort(key=lambda x: x[1], reverse=True)

    # 创建最终排序后的Shapefile
    if sorted_output_shapefile == "temp.shp":
        shp_driver.DeleteDataSource(sorted_output_shapefile)  # 如果输出路径与临时路径相同，先删除原文件
    final_output = shp_driver.CreateDataSource(sorted_output_shapefile)
    final_layer = final_output.CreateLayer(sorted_output_shapefile, srs=layer.GetSpatialRef(), geom_type=layer.GetGeomType())

    # 添加原有的字段
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        final_layer.CreateField(layer_defn.GetFieldDefn(i))

    # 将排序后的features添加到新的layer
    for feature, area in features:
        final_layer.CreateFeature(feature)

    # 清理资源
    dataSource.Destroy()
    final_output.Destroy()
    binary_raster = None
    raster = None
    shp_driver.DeleteDataSource("temp.shp")  # 删除临时文件


# 使用示例
# input_raster_path = "path_to_your_raster.tif"
# sorted_output_shapefile_path = "sorted_output_boundary.shp"
# extract_and_sort_vector_boundary(input_raster_path, 0.7, sorted_output_shapefile_path)
