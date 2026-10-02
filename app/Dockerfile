FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# python3 → python3.11 にシンボリックリンク
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# 不安定な回線でも大きなwheelのダウンロードが途切れにくいよう、タイムアウトと再試行を増やす
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10

COPY requirements.txt constraints.txt ./
# torch(約780MB)は別レイヤーで先に入れる。成功後はビルドキャッシュに残り、再ビルド時に再ダウンロードしない
RUN pip3 install --no-cache-dir torch -c constraints.txt
RUN pip3 install --no-cache-dir -r requirements.txt -c constraints.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
