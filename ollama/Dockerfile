FROM ollama/ollama:latest

# コンテナ起動時にモデルを自動pull（初回のみダウンロード）
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
