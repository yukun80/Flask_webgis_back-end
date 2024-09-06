# 基于 Python 3.11 官方镜像
FROM python:3.11-slim

# 设置非交互式安装，防止某些安装脚本等待用户输入
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    pkg-config \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
# 删除不要的镜像缓存，节约空间

# 设置环境变量，帮助 pip 找到 gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
# 设置工作目录
WORKDIR /app
# 将本地代码复制到容器中
COPY . /app

# 安装 numpy
RUN pip install numpy
RUN gdal-config --version
# 安装Python GDAL绑定
RUN pip install GDAL==3.6.2

# 安装 Cython 和其他 Python 依赖
RUN pip install --no-cache-dir --default-timeout=500 -r requirements.txt

# 设置环境变量
ENV FLASK_APP=Flask_app.py

# 对外暴露端口
EXPOSE 5000

# 启动 Flask 应用
CMD ["flask", "run", "--host=0.0.0.0"]
