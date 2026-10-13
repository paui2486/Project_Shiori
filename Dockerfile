FROM python:3.12-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統相依套件
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 複製專案原始碼
COPY . .

# 暴露 Gradio 預設通訊埠
EXPOSE 7860

# 啟動 Gradio Server
CMD ["python", "main_graph.py"]