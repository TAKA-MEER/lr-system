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

# 遅い・途切れる回線への対策(torch関連は合計2GB以上ある):
# - Ubuntu同梱のpip(22.0)は途切れたダウンロードを再開できず、壊れたファイルでハッシュ不一致になる。
#   途中から再開できる(--resume-retries)新しいpipに入れ替える
# - ダウンロード済みのwheelはキャッシュマウントに残し、ビルドをやり直しても再ダウンロードしない
# - それでも失敗した場合に備え、ビルド内で数回やり直す
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PIP_RESUME_RETRIES=20
RUN python3 -m pip install --no-cache-dir "pip==25.2"

COPY requirements.txt constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    for i in 1 2 3 4 5; do \
      python3 -m pip install -r requirements.txt -c constraints.txt && exit 0; \
      echo "pip install 失敗($i回目)。キャッシュ済みのファイルを使って再試行します"; sleep 10; \
    done; exit 1

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
