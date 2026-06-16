#!/bin/bash
# Ollamaサーバーをバックグラウンドで起動
ollama serve &
SERVER_PID=$!

# 起動待ち
sleep 5

# モデルが未取得の場合のみpull
if ! ollama list | grep -q "qwen2.5:7b"; then
    echo "qwen2.5:7b をダウンロード中..."
    ollama pull qwen2.5:7b
fi

echo "Ollama 準備完了"
wait $SERVER_PID
