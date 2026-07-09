#!/bin/bash
# Ollamaサーバーをバックグラウンドで起動
ollama serve &
SERVER_PID=$!

# 起動待ち
sleep 5

# モデルが未取得の場合のみpull
if ! ollama list | grep -q "qwen3.5:9b"; then
    echo "qwen3.5:9b をダウンロード中..."
    ollama pull qwen3.5:9b
fi

echo "Ollama 準備完了"
wait $SERVER_PID
