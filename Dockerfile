# 1. 选用轻量且稳定的 Python 基础镜像
FROM python:3.11-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 安装解压工具 (处理我们传上去的 ZIP)
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# 4. 把你的代码和压缩包全部复制进太空舱
COPY . .

# 5. 安装 Python 依赖库
RUN pip install --no-cache-dir -r requirements.txt

# 6. ⚡ 核心魔法：解压 3 万条法律数据，然后删掉压缩包省空间
# 注意：如果解压出错，程序会报错，所以必须确保 zip 文件名正确
RUN unzip -o chroma_db.zip && unzip -o docstore.zip && rm *.zip

# 7. 暴露 Hugging Face 要求的固定端口
EXPOSE 7860

# 8. 点火！启动 FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]