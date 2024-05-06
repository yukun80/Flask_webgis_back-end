import requests
from requests.auth import HTTPBasicAuth
import zipfile
import os
import json
import re  # 导入正则表达式模块


"""
    获取指定工作空间下的所有图层列表。

    参数:
    - workspace: 工作空间的名称。
    - geoserver_rest_url: GeoServer REST API的基本URL。
    - username: GeoServer的用户名。
    - password: GeoServer的密码。

    返回:
    - status_code: HTTP响应状态码。
    - response_text: 响应内容，包含图层列表或错误信息。
"""


def create_coveragestore(
    store_name,
    workspace="predict_result",
    geoserver_rest_url="http://localhost:8080/geoserver/rest",
    username="admin",
    password="geoserver",
):
    headers = {
        "Content-Type": "application/xml",
    }

    data = f"""<coverageStore>
    <name>{store_name}</name>
    <workspace>{workspace}</workspace>
    <enabled>true</enabled>
    <type>GeoTIFF</type>
</coverageStore>"""

    # 构建创建数据仓库的URL
    url = f"{geoserver_rest_url}/workspaces/{workspace}/coveragestores"

    # 发送POST请求创建数据仓库
    response = requests.post(url, headers=headers, data=data, auth=HTTPBasicAuth(username, password))

    return response.status_code, response.text


def assign_style_to_layer(
    layer_name,
    style_name,
    workspace="predict_result",
    geoserver_rest_url="http://localhost:8080/geoserver/rest",
    username="admin",
    password="geoserver",
):
    headers = {
        "Content-type": "text/xml",
    }
    # 构建设置SLD样式的REST API URL
    url = f"{geoserver_rest_url}/workspaces/{workspace}/layers/{layer_name}"
    data = f"<layer><defaultStyle><name>{style_name}</name></defaultStyle></layer>"

    # 发送PUT请求来指定图层的默认样式
    response = requests.put(url, headers=headers, data=data, auth=HTTPBasicAuth(username, password))
    return response.status_code, response.text


def upload_to_geoserver(
    tif_path,
    layer_name,
    store_name,
    workspace="predict_result",
    geoserver_rest_url="http://localhost:8080/geoserver/rest",
    username="admin",
    password="geoserver",
):
    store_status, store_msg = create_coveragestore(store_name, workspace, geoserver_rest_url, username, password)
    headers = {
        "Content-type": "image/tif",
    }

    # 构建GeoServer REST API的URL
    url = f"{geoserver_rest_url}/workspaces/{workspace}/coveragestores/{store_name}/file.geotiff?coverageName={layer_name}"

    # 读取tif文件
    with open(tif_path, "rb") as f:
        data = f.read()

    # 发送POST请求上传tif文件
    response = requests.put(url, headers=headers, data=data, auth=HTTPBasicAuth(username, password))
    return store_status, store_msg, response.status_code, response.text


def zip_shapefiles(shp_path):
    shx_path = shp_path.replace(".shp", ".shx")
    dbf_path = shp_path.replace(".shp", ".dbf")
    prj_path = shp_path.replace(".shp", ".prj")
    output_zip_path = shp_path.replace(".shp", ".zip")
    # 创建一个ZipFile对象，指定文件名和模式
    with zipfile.ZipFile(output_zip_path, "w") as zipf:
        # 向zip文件中添加文件
        zipf.write(shp_path, arcname=os.path.basename(shp_path))
        zipf.write(shx_path, arcname=os.path.basename(shx_path))
        zipf.write(dbf_path, arcname=os.path.basename(dbf_path))
        zipf.write(prj_path, arcname=os.path.basename(prj_path))


def upload_shapefile(
    store_name,
    shapefile_path,
    workspace="shapefile_result",
    geoserver_rest_url="http://localhost:8080/geoserver/rest",
    username="admin",
    password="geoserver",
):
    headers = {
        "Content-type": "application/zip",
    }
    # 构建上传的 URL
    upload_url = f"{geoserver_rest_url}/workspaces/{workspace}/datastores/{store_name}/file.shp"
    zip_shapefiles(shapefile_path)
    zip_path = shapefile_path.replace(".shp", ".zip")
    try:
        # 读取 ZIP 文件内容
        with open(zip_path, "rb") as file:
            # 发送请求到 GeoServer
            response = requests.put(upload_url, headers=headers, data=file, auth=HTTPBasicAuth(username, password))

        # 检查响应状态
        if response.status_code == 201:
            print("Shapefile uploaded and published successfully.")
        else:
            print(f"Failed to upload Shapefile: {response.status_code} {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_layers_from_workspace(
    workspace="predict_result",
    geoserver_rest_url="http://localhost:8080/geoserver/rest",
    username="admin",
    password="geoserver",
):
    # 构建获取工作空间下所有图层的URL
    url = f"{geoserver_rest_url}/workspaces/{workspace}/layers"

    try:
        # 发送GET请求获取图层列表
        response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"},
        )

        # 检查响应状态码
        if response.status_code == 200:
            # 请求成功，解析响应内容
            layers_info = response.json()
            # 提取图层名称列表
            layers_names = [layer["name"] for layer in layers_info["layers"]["layer"]]
            return 200, layers_names
        else:
            # 请求失败，返回状态码和错误信息
            return response.status_code, response.text
    except requests.RequestException as e:
        # 网络或请求异常处理
        return 500, str(e)


def get_geojson_url(layer_name, workspace="shapefile_result", geoserver_rest_url="http://localhost:8080/geoserver/rest"):
    # 构建 GeoJSON 请求 URL
    geojson_url = f"{geoserver_rest_url.replace('rest', 'ows')}?service=WFS&version=1.0.0&request=GetFeature&typeName={workspace}:{layer_name}&outputFormat=application/json"
    return geojson_url


def get_all_workspaces(geoserver_rest_url="http://localhost:8080/geoserver/rest", username="admin", password="geoserver"):
    headers = {
        "Accept": "application/json",
    }

    # 构建获取所有工作空间的URL
    url = f"{geoserver_rest_url}/workspaces"

    # 发送GET请求获取所有工作空间信息
    response = requests.get(url, headers=headers, auth=HTTPBasicAuth(username, password))

    # 检查请求是否成功
    if response.status_code == 200:
        # 提取所有工作空间的名称
        workspaces = [ws["name"] for ws in response.json()["workspaces"]["workspace"]]
        # workspace_names = [ws["name"] for ws in workspaces]
        # print("workspace_names:", workspaces)
        return 200, workspaces
    else:
        return response.status_code, response.text


def get_layers_and_types_in_workspace(workspace, geoserver_rest_url="http://localhost:8080/geoserver/rest", username="admin", password="geoserver"):
    headers = {
        "Accept": "application/json",
    }

    # 构建获取特定工作空间下所有图层的URL
    url = f"{geoserver_rest_url}/workspaces/{workspace}/layers"

    # 发送GET请求获取所有图层信息
    response = requests.get(url, headers=headers, auth=HTTPBasicAuth(username, password))
    # 打印响应状态码和内容
    # print("Status Code:", response.status_code)
    # print("Response Content:", response.content)
    layer_details = []

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            layers = response.json().get("layers", {}).get("layer", [])
            # 遍历每个图层获取其详细信息
            for layer in layers:
                layer_name = layer["name"]
                # 获取单个图层的详细信息
                layer_url = f"{geoserver_rest_url}/workspaces/{workspace}/layers/{layer_name}"
                layer_response = requests.get(layer_url, headers=headers, auth=HTTPBasicAuth(username, password))
                if layer_response.status_code == 200:
                    # 解析数据存储类型
                    layer_info = layer_response.json()
                    store_type = layer_info["layer"]["resource"]["@class"].split(".")[-1]
                    layer_details.append((layer_name, store_type))
                else:
                    layer_details.append((layer_name, "Error: Failed to retrieve layer details"))
        except json.JSONDecodeError:
            return f"Error: Received status {response.status_code}, Response is not valid JSON - {response.text}"
    else:
        return f"Error: {response.status_code} - {response.text}"
    # print("layer_details:", layer_details)
    return layer_details


# 获取工作空间所有的存储仓库名称
def get_stores(workspace, geoserver_rest_url="http://localhost:8080/geoserver/rest", username="admin", password="geoserver"):
    """获取指定工作区中所有存储的列表"""
    auth = HTTPBasicAuth(username, password)
    url = f"{geoserver_rest_url}/workspaces/{workspace}/datastores.json"
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        datastores = response.json()
        return [store["name"] for store in datastores["dataStores"]["dataStore"]]
    else:
        raise Exception(f"Failed to retrieve stores: {response.text}")


# 根据图层名称获取存储名称
def get_store_of_layer(
    layer_name, workspace, layer_class, geoserver_rest_url="http://localhost:8080/geoserver/rest", username="admin", password="geoserver"
):
    """获取特定图层的存储名称和类型"""
    auth = HTTPBasicAuth(username, password)
    layer_url = f"{geoserver_rest_url}/layers/{layer_name}.json"
    # print("+++++++++++++++debug3:", layer_url)
    response = requests.get(layer_url, auth=auth)
    # print("+++++++++++++++debug4 response:", response.text)  # 打印完整响应内容
    if response.status_code == 200:
        layer_info = response.json()
        resource_href = layer_info["layer"]["resource"]["href"]
        # 使用正则表达式从 href 中提取存储名称，匹配 datastores 或 coveragestores
        match = re.search(r"/(datastores|coveragestores)/([^/]+)/", resource_href)
        if match:
            store_type = match.group(1)
            store_name = match.group(2)  # 提取存储名称
            store_type = "datastores" if store_type == "datastores" else "coveragestores"
            print("+++++++++++++++debug5:", store_name, store_type)
            return store_name, store_type
        else:
            raise Exception("Failed to extract store name from the resource href")
    else:
        raise Exception(f"Failed to retrieve layer info: {response.text}")


def delete_layer_and_store(
    layer_name, workspace, layer_class, geoserver_rest_url="http://localhost:8080/geoserver/rest", username="admin", password="geoserver"
):
    # print("+++++++++++++++debug2:", layer_name, workspace, layer_class, geoserver_rest_url, username, password)
    # 为请求设置基本认证
    auth = HTTPBasicAuth(username, password)
    store_name, store_type = get_store_of_layer(layer_name, workspace, layer_class, geoserver_rest_url, username, password)
    # print("+++++++++++++++debug6:", store_name, store_type)
    # 删除图层
    if layer_class not in ["featureType", "coverage"]:
        raise ValueError("Invalid layer class provided. Choose from 'layers', 'featuretypes', or 'coverages'.")

    layer_url = f"{geoserver_rest_url}/workspaces/{workspace}/layers/{layer_name}"
    # print("+++++++++++++++debug7:", layer_url)
    delete_layer_response = requests.delete(layer_url, auth=auth)
    if delete_layer_response.status_code not in [200, 202, 204]:
        raise Exception(f"Failed to delete layer: {delete_layer_response.text}")

    # 删除存储
    store_url = f"{geoserver_rest_url}/workspaces/{workspace}/{store_type}/{store_name}"
    # print("+++++++++++++++debug7:", store_url)
    delete_store_response = requests.delete(store_url + "?recurse=true", auth=auth)
    if delete_store_response.status_code not in [200, 202, 204]:
        raise Exception(f"Failed to delete store: {delete_store_response.text}")

    return "Layer and store deleted successfully"
